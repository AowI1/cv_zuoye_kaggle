#!/usr/bin/env python
"""Open-set benchmark ARI for a set of (cached) backbones.

Cheap, leakage-free comparison used to decide whether a new embedding
(fine-tuned backbone, new fusion) is worth a Kaggle submission:

    python scripts/eval_backbone.py --species LynxID2025 --backbones miewid,mega
    python scripts/eval_backbone.py --species LynxID2025 --backbones miewid_ft

Thresholds are swept per seed for every configuration, so all methods get the
same (optimistic) threshold selection and the comparison is like-for-like.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from animalclef import cluster as cl  # noqa: E402
from animalclef.benchmark import ari, make_identity_split  # noqa: E402
from animalclef.config import CACHE, SPECIES_KEY  # noqa: E402
from animalclef.data import load_metadata  # noqa: E402
from animalclef.similarity import cosine  # noqa: E402


def embs(backbone: str, ds: str, split: str) -> np.ndarray:
    return np.load(CACHE / f"emb_{backbone}_{SPECIES_KEY[ds]}_{split}.npy")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--backbones", default="miewid,mega")
    ap.add_argument("--seeds", default="100,101,102,103")
    ap.add_argument("--weights", default="", help="e.g. 0.7,0.3")
    args = ap.parse_args()

    backbones = args.backbones.split(",")
    weights = ([float(w) for w in args.weights.split(",")]
               if args.weights else [1.0] * len(backbones))
    seeds = [int(s) for s in args.seeds.split(",")]

    meta = load_metadata()
    scores = []
    for seed in seeds:
        db, q = make_identity_split(meta, args.species, seed=seed)
        pos_db = np.asarray(db["_pos"])
        pos_q = np.asarray(q["_pos"])
        sim = None
        for b, w in zip(backbones, weights):
            e = embs(b, args.species, "train")
            s = cosine(e[pos_q], e[pos_q])
            sim = w * s if sim is None else sim + w * s
        true = q["true_identity"].astype(str).values
        best = max(ari(true, cl.agglomerative(sim, t))
                   for t in np.linspace(0.05, 0.95, 91))
        scores.append(best)
        print(f"  seed {seed}: {best:.3f}")

    print(f"{args.species} {backbones} -> mean ARI {np.mean(scores):.3f} "
          f"± {np.std(scores):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
