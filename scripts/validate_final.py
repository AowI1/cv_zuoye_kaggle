#!/usr/bin/env python
"""Evaluate the stored per-species configs on held-out benchmark seeds.

Tuning used seeds 0-3; this script reports unbiased ARI on seeds 100+.
Local-feature similarities are recomputed per split (they depend on the
calibrated global similarity).

    python scripts/validate_final.py --seeds 100 101 102 103
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
from animalclef.config import SPECIES_WITH_TRAIN
from animalclef.data import load_metadata
from animalclef.localfeat import local_similarity, normalize_local
from animalclef.pipeline import Config, cluster_similarity, refine
from tune import SplitData
from benchmark_core import build_splits, load_emb

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102, 103])
    ap.add_argument("--species", default="all")
    args = ap.parse_args()

    meta = load_metadata()
    species = SPECIES_WITH_TRAIN if args.species == "all" else [args.species]
    summary = {}
    for ds in species:
        cfg_path = CONFIG_DIR / f"best_{ds}.json"
        if not cfg_path.exists():
            print(f"[skip] no config for {ds}")
            continue
        best = json.loads(cfg_path.read_text())
        post = best.get("post", {})
        weights = {k: v for k, v in best["weights"].items()
                   if k in ("miewid", "mega") and v > 0}
        cfg = Config(
            dataset=ds, weights=weights, threshold=best["threshold"],
            use_rerank=best.get("use_rerank", False),
            rr_k1=best["rr"][0], rr_k2=best["rr"][1], rr_lambda=best["rr"][2],
            merge_small=post.get("merge_small", 2),
            merge_threshold=post.get("merge_threshold", 0.4),
            split_big=post.get("split_big", 10**9),
            split_threshold=post.get("split_threshold", 0.65),
            anchor=post.get("anchor", False),
            anchor_score=post.get("anchor_score", 0.6),
            anchor_margin=post.get("anchor_margin", 0.1),
            use_orientation_calibration=best.get("use_orientation_calibration", False),
        )
        embs = {b: e for b in ["miewid", "mega"]
                if (e := load_emb(b, ds, "train")) is not None}
        w_local = float(best.get("local_weight", 0.0))
        method = best.get("local_method", "root_sift")

        splits = build_splits(meta, ds, args.seeds, best.get("new_id_frac", 0.4))
        aris, ks = [], []
        for db, qry in splits:
            sd = SplitData(ds, db, qry, embs,
                           use_orientation=cfg.use_orientation_calibration)
            sim = sd.fused(weights)
            if w_local > 0:
                qdf = qry.reset_index(drop=True)
                lsim = normalize_local(local_similarity(
                    qdf, sim, method=method, top_k=10, gate_percentile=95.0,
                    n_threads=8, verbose=False,
                ))
                sim = (sim + w_local * lsim) / (1.0 + w_local)
            if cfg.use_rerank:
                sim = cl.rerank_similarity(sim, cfg.rr_k1, cfg.rr_k2, cfg.rr_lambda)
            labels = cluster_similarity(sim, cfg)
            labels = refine(labels, sim, cfg, sd.fused_qd(weights), sd.db_ids)
            aris.append(ari(sd.true, labels))
            ks.append(len(set(labels)))
        summary[ds] = {"ari_mean": float(np.mean(aris)),
                       "ari_std": float(np.std(aris)),
                       "clusters": ks, "n": len(aris)}
        print(f"{ds:<20s} held-out ARI={np.mean(aris):.4f} ± {np.std(aris):.4f} "
              f"(clusters {ks})", flush=True)

    out = CONFIG_DIR / "validation_heldout.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
