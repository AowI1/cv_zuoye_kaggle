#!/usr/bin/env python
"""Precompute LightGlue local-feature similarity on candidate pairs.

Mirrors scripts/extract_local.py (same candidate gating, same score
normalisation) but matches with LightGlue instead of brute force + RANSAC.

    python scripts/extract_local_glue.py --species TexasHornedLizards --split test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from animalclef.config import CACHE, SPECIES_KEY, get_device  # noqa: E402
from animalclef.data import load_metadata  # noqa: E402
from animalclef.localfeat_glue import GlueLocalSimilarity  # noqa: E402
from animalclef.similarity import cosine, fuse, rank_normalize  # noqa: E402


def gating_similarity(meta, species: str, split: str, key: str) -> np.ndarray:
    """Calibrated fused query-query similarity (same as the tuning procedure)."""
    from animalclef.pipeline import Config, fit_calibrators

    weights = {"miewid": 0.5, "mega": 0.5}
    best_path = ROOT / "configs" / f"best_{species}.json"
    if best_path.exists():
        weights = json.loads(best_path.read_text()).get("weights", weights)
        weights = {k: v for k, v in weights.items()
                   if k in ("miewid", "mega") and v > 0}
    train = meta[(meta["dataset"] == species) & (meta["split"] == "train")]
    test_embs = {b: np.load(CACHE / f"emb_{b}_{key}_{split}.npy")
                 for b in weights if (CACHE / f"emb_{b}_{key}_{split}.npy").exists()}
    if len(train) > 0:
        train_embs = {b: np.load(CACHE / f"emb_{b}_{key}_train.npy")
                      for b in test_embs
                      if (CACHE / f"emb_{b}_{key}_train.npy").exists()}
        cals = fit_calibrators(train_embs, train, Config(dataset=species))
        qq = {b: cals[b].transform(cosine(e, e)) for b, e in test_embs.items()}
    else:
        qq = {b: rank_normalize(cosine(e, e)) for b, e in test_embs.items()}
    return fuse(qq, {b: weights[b] for b in test_embs})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--gate-percentile", type=float, default=95.0)
    ap.add_argument("--max-side", type=int, default=800)
    ap.add_argument("--nfeatures", type=int, default=2000)
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    meta = load_metadata()
    rows = meta[(meta["dataset"] == args.species) & (meta["split"] == args.split)]
    rows = rows.reset_index(drop=True)
    key = SPECIES_KEY[args.species]
    out = CACHE / f"local_lightglue_{key}_{args.split}.npy"
    if out.exists() and not args.force:
        print(f"[skip] {out.name} {np.load(out).shape}")
        return 0

    sim = gating_similarity(meta, args.species, args.split, key)
    device = get_device()
    t0 = time.time()
    glue = GlueLocalSimilarity(device=device, max_side=args.max_side,
                               nfeatures=args.nfeatures,
                               score_thr=args.score_thr)
    local = glue(rows, sim, top_k=args.top_k,
                 gate_percentile=args.gate_percentile)
    np.save(out, local)
    print(f"[done] {out.name} {local.shape} "
          f"nonzero={int((local > 0).sum())} in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
