#!/usr/bin/env python
"""Fine-tune the MiewID backbone with ArcFace on the labelled database.

    python scripts/finetune_miewid.py --species LynxID2025 --epochs 12
    python scripts/finetune_miewid.py --species all --epochs 8

Writes models/miewid-ft/<name>/best.pt, loadable by
`extract_embeddings.py --backbone miewid_ft --checkpoint ...`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from animalclef.config import MODELS, SPECIES_WITH_TRAIN, get_device  # noqa: E402
from animalclef.data import load_metadata  # noqa: E402
from animalclef.embed import _MiewIDNet  # noqa: E402
from animalclef.finetune import (ArcFaceHead, IdentityDataset, TrainCfg,  # noqa: E402
                                 train)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default="LynxID2025",
                    help="one species, comma-separated list, or 'all'")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--size", type=int, default=440)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.species == "all":
        species = list(SPECIES_WITH_TRAIN)
        name = "all"
    else:
        species = args.species.split(",")
        name = species[0] if len(species) == 1 else "multi"

    meta = load_metadata()
    train_df = meta[meta["dataset"].isin(species) & (meta["split"] == "train")]
    train_df = train_df[train_df["identity"].notna()]
    print(f"[{name}] {len(train_df)} images, "
          f"{train_df['identity'].nunique()} identities, {len(species)} species")

    ds = IdentityDataset(train_df, None, size=args.size, train=True)
    print(f"[{name}] {len(ds.classes)} classes")

    device = get_device()
    model = _MiewIDNet()
    from safetensors.torch import load_file
    state = load_file(str(MODELS / "miewid-msv3" / "model.safetensors"))
    state = {k: v for k, v in state.items() if not k.startswith("final.")}
    model.load_state_dict(state, strict=False)
    model = model.to(device)

    for p in model.parameters():
        p.requires_grad = True

    head = ArcFaceHead(2152, len(ds.classes)).to(device)
    cfg = TrainCfg(epochs=args.epochs, p=args.p, k=args.k,
                   workers=args.workers, size=args.size, seed=args.seed)
    t0 = time.time()
    train(model, head, ds, cfg, device)
    print(f"[{name}] trained in {(time.time()-t0)/60:.1f} min")

    out = Path(args.out) if args.out else (MODELS / "miewid-ft" / name / "best.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": model.state_dict(), "head": head.state_dict(),
                "species": species, "classes": ds.classes,
                "config": cfg.__dict__}, out)
    (out.parent / "meta.json").write_text(json.dumps(
        {"species": species, "n_images": len(train_df),
         "n_identities": len(ds.classes), "epochs": args.epochs,
         "size": args.size}, indent=2))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
