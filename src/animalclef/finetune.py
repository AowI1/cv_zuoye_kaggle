"""Metric-learning fine-tuning of the MiewID backbone (ArcFace).

The public top solutions fine-tune MiewID with a margin loss on the labelled
database identities before clustering. Frozen features plateau around ARI
0.3 for lynx/salamander; a short fine-tune adapts the embedding to each
species' appearance.

Design notes
- PK sampling (P identities x K images) so every batch carries positive
  pairs; identity sampling is uniform over identities, which matters for
  lynx (few identities) vs turtle (many).
- Backbone learning rate is 100x smaller than the head (as in the winning
  solution) to avoid destroying the pretrained features.
- Augmentation is deliberately mild: horizontal flip, random-resized crop and
  small colour jitter. Species preprocessing is applied first, exactly as at
  extraction time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from .data import load_image


class IdentityDataset(Dataset):
    """Image -> (CHW float tensor, identity index)."""

    def __init__(self, df, dataset: str | None, size: int, train: bool = True,
                 pre_max_side: int = 512):
        self.df = df.reset_index(drop=True)
        self.dataset = dataset
        self.size = size
        self.train = train
        self.pre_max_side = pre_max_side
        ids = sorted(self.df["identity"].astype(str).unique())
        self.classes = ids
        self.class_to_idx = {c: i for i, c in enumerate(ids)}
        self.labels = self.df["identity"].astype(str).map(self.class_to_idx).values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        # slightly larger then crop, so the crop is a true augmentation
        big = int(round(self.size * 1.15))
        ds = self.dataset or str(row["dataset"])
        img = load_image(row["abs_path"], ds, do_preprocess=True,
                         size=big, pre_max_side=self.pre_max_side)
        h, w = img.shape[:2]
        if self.train:
            ch, cw = self.size, self.size
            y = np.random.randint(0, max(1, h - ch + 1))
            x = np.random.randint(0, max(1, w - cw + 1))
            img = img[y:y + ch, x:x + cw]
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
        else:
            y = (h - self.size) // 2
            x = (w - self.size) // 2
            img = img[y:y + self.size, x:x + self.size]
        x_t = torch.from_numpy(np.ascontiguousarray(img)).float().div_(255.0)
        x_t = x_t.permute(2, 0, 1)
        mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
        return (x_t - mean) / std, int(self.labels[i])


class PKBatchSampler(Sampler[list[int]]):
    """P identities x K images per batch."""

    def __init__(self, labels: np.ndarray, p: int, k: int, seed: int = 0):
        self.labels = np.asarray(labels)
        self.p, self.k = p, k
        self.rng = np.random.default_rng(seed)
        self.by_class = {}
        for idx, lab in enumerate(self.labels):
            self.by_class.setdefault(int(lab), []).append(idx)
        self.classes = [c for c, v in self.by_class.items() if len(v) >= 2]
        self.length = max(1, len(self.labels) // (p * k))

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        for _ in range(self.length):
            cls = self.rng.choice(self.classes, size=min(self.p, len(self.classes)),
                                  replace=False)
            batch = []
            for c in cls:
                pool = self.by_class[int(c)]
                take = self.rng.choice(len(pool), size=min(self.k, len(pool)),
                                       replace=len(pool) < self.k)
                batch.extend(pool[i] for i in take)
            yield batch


class ArcFaceHead(torch.nn.Module):
    def __init__(self, in_features: int, num_classes: int,
                 margin: float = 0.3, scale: float = 30.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_classes, in_features))
        torch.nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None):
        cos = F.linear(F.normalize(features), F.normalize(self.weight))
        if labels is None:
            return cos
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        target = torch.cos(theta + self.margin)
        onehot = F.one_hot(labels, num_classes=self.weight.shape[0]).bool()
        logits = torch.where(onehot, target, cos) * self.scale
        return logits


@dataclass
class TrainCfg:
    epochs: int = 12
    p: int = 8
    k: int = 4
    backbone_lr: float = 1e-5
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    workers: int = 4
    size: int = 440
    seed: int = 42


def train(model: torch.nn.Module, head: torch.nn.Module, dataset: IdentityDataset,
          cfg: TrainCfg, device: str, log=print):
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=PKBatchSampler(dataset.labels, cfg.p, cfg.k, cfg.seed),
        num_workers=cfg.workers, prefetch_factor=2, persistent_workers=cfg.workers > 0,
    )
    opt = torch.optim.AdamW([
        {"params": model.parameters(), "lr": cfg.backbone_lr},
        {"params": head.parameters(), "lr": cfg.head_lr},
    ], weight_decay=cfg.weight_decay)
    steps = max(1, len(loader) * cfg.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[cfg.backbone_lr, cfg.head_lr], total_steps=steps,
        pct_start=0.15)
    model.train()
    head.train()
    for epoch in range(cfg.epochs):
        total, correct, loss_sum, n = 0, 0, 0.0, 0
        for step, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            feats = model(x)
            logits = head(feats, y)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 5.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                acc = (logits.argmax(1) == y).float().mean().item()
            total += acc * len(y)
            correct += acc * len(y)
            loss_sum += loss.item() * len(y)
            n += len(y)
            if step % 10 == 0:
                log(f"  epoch {epoch+1}/{cfg.epochs} step {step}/{len(loader)} "
                    f"loss {loss_sum/max(n,1):.3f} acc {total/max(n,1):.3f}")
        log(f"epoch {epoch+1}: loss {loss_sum/max(n,1):.4f} "
            f"train_acc {total/max(n,1):.4f} lr {sched.get_last_lr()}")
