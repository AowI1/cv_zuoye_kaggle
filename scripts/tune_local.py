#!/usr/bin/env python
"""Tune the local-feature fusion weight on one open-set benchmark split.

Computes RootSIFT/RANSAC similarities for the query subset only (candidates
gated by the calibrated global similarity), then sweeps fusion weights and
clustering thresholds. The chosen weight is merged into the species config.

    python scripts/tune_local.py --species SalamanderID2025 --seeds 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from animalclef import cluster as cl
from animalclef.benchmark import ari
from animalclef.data import load_metadata
from animalclef.localfeat import local_similarity, normalize_local
from animalclef.pipeline import Config, cluster_similarity, refine
from tune import SplitData

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
from benchmark_core import build_splits, load_emb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--method", default="root_sift", choices=["sift", "root_sift", "orb"])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--min-sim", type=float, default=-1.0,
                    help="global-similarity gate for candidate pairs")
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    ds = args.species
    meta = load_metadata()
    embs = {}
    for b in ["miewid", "mega"]:
        e = load_emb(b, ds, "train")
        if e is not None:
            embs[b] = e
    best_cfg_path = CONFIG_DIR / f"best_{ds}.json"
    base = json.loads(best_cfg_path.read_text()) if best_cfg_path.exists() else {}
    weights = base.get("weights", {"miewid": 0.5, "mega": 0.5})
    weights = {k: v for k, v in weights.items() if k in embs}
    threshold0 = base.get("threshold", 0.5)
    use_rr = base.get("use_rerank", False)
    rr = base.get("rr", [20, 6, 0.3])

    splits = build_splits(meta, ds, range(args.seeds), base.get("new_id_frac", 0.4))
    results = []
    for si, (db, qry) in enumerate(splits):
        sd = SplitData(ds, db, qry, embs)
        gsim = sd.fused(weights)
        t = time.time()
        qdf = qry.reset_index(drop=True)
        lsim = local_similarity(
            qdf, gsim, method=args.method, top_k=args.top_k,
            min_sim=None if args.min_sim < 0 else args.min_sim,
            gate_percentile=95.0, n_threads=8,
        )
        print(f"split {si}: local sims done in {time.time()-t:.0f}s "
              f"({(lsim>0).sum()} directed nonzero)", flush=True)
        lsim = normalize_local(lsim)
        results.append((sd, gsim, lsim))

    post = base.get("post", {})
    cfg_base = Config(
        dataset=ds, weights=weights, threshold=threshold0,
        use_rerank=use_rr, rr_k1=rr[0], rr_k2=rr[1], rr_lambda=rr[2],
        merge_small=post.get("merge_small", 2),
        merge_threshold=post.get("merge_threshold", 0.4),
        split_big=post.get("split_big", 10**9),
        split_threshold=post.get("split_threshold", 0.65),
        anchor=post.get("anchor", False),
        anchor_score=post.get("anchor_score", 0.6),
        anchor_margin=post.get("anchor_margin", 0.1),
    )

    best = None
    for wl in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        # Re-ranking depends only on the fused matrix, not on the clustering
        # threshold: compute once per weight instead of once per threshold.
        fused = []
        for sd, gsim, lsim in results:
            sim = (gsim + wl * lsim) / (1.0 + wl)
            if use_rr:
                sim = cl.rerank_similarity(sim, rr[0], rr[1], rr[2])
            fused.append((sd, sim, sd.fused_qd(weights)))
        for t in np.arange(0.02, 0.85, 0.02):
            cfg = replace(cfg_base, threshold=float(t))
            aris = []
            for sd, sim, qd in fused:
                labels = cluster_similarity(sim, cfg)
                labels = refine(labels, sim, cfg, qd, sd.db_ids)
                aris.append(ari(sd.true, labels))
            m = float(np.mean(aris))
            if best is None or m > best["ari"]:
                best = {"ari": m, "w_local": wl, "threshold": float(t),
                        "std": float(np.std(aris))}
    print(f"best local fusion: w={best['w_local']} t={best['threshold']:.3f} "
          f"ARI={best['ari']:.4f} ± {best['std']:.4f}")

    prev_score = base.get("score") or -1.0
    improved = best["ari"] > prev_score + 1e-6
    base_out = {
        **base,
        "dataset": ds,
        "weights": {**weights, "local": best["w_local"] if improved else 0.0},
        "threshold": best["threshold"] if improved else threshold0,
        "local_method": args.method,
        "local_weight": best["w_local"] if improved else 0.0,
        "score_global_only": prev_score,
        "score_with_local": best["ari"],
        "post": {**post,
                 "merge_small": cfg_base.merge_small,
                 "merge_threshold": cfg_base.merge_threshold,
                 "split_big": cfg_base.split_big,
                 "split_threshold": cfg_base.split_threshold,
                 "anchor": cfg_base.anchor},
    }
    if improved:
        base_out["score"] = best["ari"]
        base_out["std"] = best["std"]
        base_out["seeds"] = args.seeds
    best_cfg_path.write_text(json.dumps(base_out, indent=2))
    print(f"{'updated' if improved else 'kept global config'} -> {best_cfg_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
