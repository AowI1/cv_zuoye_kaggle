#!/usr/bin/env python
"""Extract and cache backbone embeddings for all species splits.

Usage:
    python scripts/extract_embeddings.py --backbone miewid --species all
    python scripts/extract_embeddings.py --backbone mega --species all --tta
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from animalclef.config import CACHE, SPECIES, SPECIES_KEY, get_device
from animalclef.data import load_metadata
from animalclef.embed import get_backbone


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["miewid", "mega"])
    ap.add_argument("--species", default="all")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--size", type=int, default=0, help="override input size")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--letterbox", action="store_true")
    ap.add_argument("--no-preprocess", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--suffix", default="", help="cache suffix, e.g. '_tta'")
    args = ap.parse_args()

    device = get_device()
    meta = load_metadata()
    species = SPECIES if args.species == "all" else [args.species]

    t0 = time.time()
    model = get_backbone(
        args.backbone, device=device,
        size=args.size or None, use_tta=args.tta, letterbox=args.letterbox,
    )
    print(f"[{args.backbone}] loaded on {device} in {time.time()-t0:.1f}s", flush=True)

    for ds in species:
        key = SPECIES_KEY[ds]
        for split in ["train", "test"]:
            rows = meta[(meta["dataset"] == ds) & (meta["split"] == split)]
            if len(rows) == 0:
                continue
            out = CACHE / f"emb_{args.backbone}_{key}_{split}{args.suffix}.npy"
            ids_out = CACHE / f"emb_{args.backbone}_{key}_{split}{args.suffix}_ids.npy"
            if out.exists() and not args.force:
                emb = np.load(out)
                if len(emb) == len(rows):
                    print(f"[skip] {out.name} exists ({emb.shape})", flush=True)
                    continue
            t = time.time()
            print(f"[{args.backbone}] {ds}/{split}: {len(rows)} images", flush=True)
            emb = model.embed(
                rows, batch_size=args.batch_size, n_threads=args.threads,
                do_preprocess=not args.no_preprocess, verbose=True,
            )
            np.save(out, emb)
            np.save(ids_out, rows["image_id"].values)
            dt = time.time() - t
            print(f"[done] {out.name} {emb.shape} in {dt:.0f}s "
                  f"({len(rows)/max(dt,1e-9):.1f} img/s)", flush=True)
            (CACHE / f"emb_{args.backbone}_{key}_{split}{args.suffix}.json").write_text(
                json.dumps({
                    "backbone": args.backbone, "dataset": ds, "split": split,
                    "n": int(len(rows)), "size": model.size, "tta": args.tta,
                    "letterbox": args.letterbox,
                    "preprocess": not args.no_preprocess,
                })
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
