#!/usr/bin/env python
"""Precompute local-feature (SIFT/RootSIFT/ORB) similarity matrices on candidate pairs.

The global backbone similarity gates which pairs are worth matching, so run
after embeddings are cached.

    python scripts/extract_local.py --species SalamanderID2025 --split test --method root_sift
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from animalclef.config import CACHE, SPECIES_KEY
from animalclef.data import load_metadata
from animalclef.localfeat import local_similarity
from animalclef.similarity import cosine


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--method", default="root_sift", choices=["sift", "root_sift", "orb"])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--min-sim", type=float, default=-1.0,
                    help="absolute gate; <0 uses a percentile gate")
    ap.add_argument("--gate-percentile", type=float, default=95.0)
    ap.add_argument("--max-side", type=int, default=800)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--gate", default="calibrated",
                    choices=["calibrated", "cosine"],
                    help="candidate gating similarity source")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    meta = load_metadata()
    rows = meta[(meta["dataset"] == args.species) & (meta["split"] == args.split)]
    rows = rows.reset_index(drop=True)
    key = SPECIES_KEY[args.species]
    out = CACHE / f"local_{args.method}_{key}_{args.split}.npy"
    if out.exists() and not args.force:
        print(f"[skip] {out} exists {np.load(out).shape}")
        return 0

    # Candidate gating must match the tuning procedure: benchmark tuning gates
    # on the *calibrated fused* similarity, so the test matrix must too,
    # otherwise the local channel is fused on a different candidate set and
    # the operating point does not transfer.
    gate = args.gate
    if gate == "calibrated":
        import json
        from animalclef.pipeline import Config, fit_calibrators
        from animalclef.similarity import fuse
        from animalclef import cluster as cl  # noqa: F401
        best_path = Path(__file__).resolve().parents[1] / "configs" / f"best_{args.species}.json"
        weights = {"miewid": 0.5, "mega": 0.5}
        if best_path.exists():
            weights = json.loads(best_path.read_text()).get("weights", weights)
            weights = {k: v for k, v in weights.items()
                       if k in ("miewid", "mega") and v > 0}
        train = meta[(meta["dataset"] == args.species) & (meta["split"] == "train")]
        test_embs = {}
        for b in weights:
            p = CACHE / f"emb_{b}_{key}_{args.split}.npy"
            if p.exists():
                test_embs[b] = np.load(p)
        if len(train) > 0:
            train_embs = {b: np.load(CACHE / f"emb_{b}_{key}_train.npy")
                          for b in test_embs
                          if (CACHE / f"emb_{b}_{key}_train.npy").exists()}
            cals = fit_calibrators(train_embs, train, Config(dataset=args.species))
            qq = {b: cals[b].transform(cosine(e, e)) for b, e in test_embs.items()}
            sim = fuse(qq, {b: weights[b] for b in test_embs})
        else:
            from animalclef.similarity import rank_normalize
            sim = fuse({b: rank_normalize(cosine(e, e))
                        for b, e in test_embs.items()}, weights)
    else:
        emb_path = CACHE / f"emb_miewid_{key}_{args.split}.npy"
        if not emb_path.exists():
            raise FileNotFoundError(f"need global embeddings first: {emb_path}")
        emb = np.load(emb_path)
        sim = cosine(emb, emb)

    t = time.time()
    local = local_similarity(
        rows, sim, method=args.method, top_k=args.top_k,
        min_sim=None if args.min_sim < 0 else args.min_sim,
        gate_percentile=args.gate_percentile, max_side=args.max_side, n_threads=args.threads,
    )
    np.save(out, local)
    nonzero = int((local > 0).sum())
    print(f"[done] {out.name} {local.shape}, nonzero pairs={nonzero}, "
          f"elapsed={time.time()-t:.0f}s")
    print(f"  score stats: mean={local[local>0].mean():.4f} "
          f"p90={np.percentile(local[local>0], 90):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
