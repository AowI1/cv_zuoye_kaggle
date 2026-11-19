"""Backbone wrappers and batched embedding extraction.

Supported backbones:
- ``miewid``: conservationxlabs/miewid-msv3 (timm efficientnetv2_rw_m + GeM + BN,
  2152-dim), loaded from a local snapshot with transformers custom code.
- ``mega``: BVRA/MegaDescriptor-L-384 (timm swin_large_patch4_window12_384,
  1536-dim), loaded from a local state dict.

Both models are kept frozen; embeddings are L2-normalised before returning.
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .config import IMAGENET_MEAN, IMAGENET_STD, MODEL_SPECS
from .data import iter_batches


class Backbone:
    name = "base"

    def __init__(self, device: str = "mps", size: int | None = None,
                 use_tta: bool = False, letterbox: bool = False):
        self.device = device
        self.size = size or MODEL_SPECS[self.name]["size"]
        self.use_tta = use_tta
        self.letterbox = letterbox
        self.model = None

    # subclasses implement _load()
    def _load(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def load(self):
        self.model = self._load()
        self.model.eval()
        return self

    def _to_tensor(self, batch_uint8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(batch_uint8).float().div_(255.0)
        x = x.permute(0, 3, 1, 2).contiguous()
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def _forward(self, x: torch.Tensor) -> np.ndarray:
        x = x.to(self.device)
        dtype = next(self.model.parameters()).dtype
        if dtype == torch.float16:
            x = x.half()
        out = self.model(x)
        if isinstance(out, dict):
            out = out.get("x_norm_clstoken", out.get("pooler_output"))
        return F.normalize(out.float(), dim=1).cpu().numpy()

    @torch.no_grad()
    def embed(self, df, indices: Sequence[int] | None = None,
              batch_size: int = 16, n_threads: int = 4,
              do_preprocess: bool = True, verbose: bool = False) -> np.ndarray:
        """Extract L2-normalised embeddings for the given metadata rows."""
        feats = []
        total = len(df) if indices is None else len(indices)
        done = 0
        for chunk, batch in iter_batches(
            df, indices, batch_size, n_threads, do_preprocess,
            self.size, self.letterbox,
        ):
            x = self._to_tensor(batch)
            f = self._forward(x)
            if self.use_tta:
                f_flip = self._forward(torch.flip(x, dims=[3]))
                f = F.normalize(
                    torch.from_numpy(f + f_flip), dim=1
                ).numpy()
            feats.append(f.astype(np.float32))
            done += len(chunk)
            if verbose and (done % (batch_size * 5) == 0 or done == total):
                print(f"  [{self.name}] {done}/{total}", flush=True)
        out = np.concatenate(feats, axis=0)
        return out


class _GeM(torch.nn.Module):
    """Generalised-mean pooling used by the MiewID backbone."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return torch.nn.functional.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class _MiewIDNet(torch.nn.Module):
    """Re-implementation of the MiewID-msv3 head (timm backbone + GeM + BN).

    The original Hugging Face custom code targets transformers 4.4x; this
    standalone version loads the same weights without the transformers
    dependency (the checkpoint keys are ``backbone.*``, ``bn.*`` and an
    unused ``final.*`` ArcFace head).
    """

    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "efficientnetv2_rw_m", pretrained=False, num_classes=0
        )
        self.backbone.global_pool = _GeM()
        self.bn = torch.nn.BatchNorm1d(2152)

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        return self.bn(x)


class MiewID(Backbone):
    name = "miewid"

    def _load(self):
        from safetensors.torch import load_file

        model = _MiewIDNet()
        state = load_file(str(MODEL_SPECS["miewid"]["dir"] / "model.safetensors"))
        state = {k: v for k, v in state.items() if not k.startswith("final.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"MiewID weight mismatch: missing={missing[:5]} "
                f"unexpected={unexpected[:5]}"
            )
        return model.to(self.device)


class Mega(Backbone):
    name = "mega"

    def _load(self):
        import timm

        cfg_dir = MODEL_SPECS["mega"]["dir"]
        model = timm.create_model(
            "swin_large_patch4_window12_384", pretrained=False,
            num_classes=0, global_pool="avg",
        )
        sd = torch.load(cfg_dir / "pytorch_model.bin", map_location="cpu",
                        weights_only=True)
        # The published checkpoint is a full training state; the weights live
        # under "model" (no classifier head since num_classes=0).
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
        # The checkpoint was saved by an older timm which attached the patch
        # merging module at the *end* of each stage; current timm attaches it
        # at the *start* of the next stage. The operations are identical, only
        # the key names shift by one stage. Old attention buffers are
        # recomputed by the new implementation and can be dropped.
        import re
        remapped = {}
        for k, v in sd.items():
            if "relative_position_index" in k or k.endswith("attn_mask"):
                continue
            m2 = re.match(r"layers\.(\d)\.downsample\.(.*)", k)
            if m2:
                k = f"layers.{int(m2.group(1)) + 1}.downsample.{m2.group(2)}"
            remapped[k] = v
        missing, unexpected = model.load_state_dict(remapped, strict=True)
        return model.to(self.device)


BACKBONES = {"miewid": MiewID, "mega": Mega}


def get_backbone(name: str, **kwargs) -> Backbone:
    return BACKBONES[name](**kwargs).load()
