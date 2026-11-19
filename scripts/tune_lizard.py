#!/usr/bin/env python
"""Zero-shot configuration for TexasHornedLizards.

The species ships with no labelled training data, so its threshold cannot be
tuned on ARI. Instead we:
  1. rank-normalise the similarity matrices of the two backbones (no
     calibration data available),
  2. fuse them with a local-feature channel,
  3. sweep the clustering threshold and select it by silhouette score on the
     precomputed distance matrix, reporting cluster-count stability too.

Usage: python scripts/tune_lizard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from animalclef import cluster as cl
from animalclef.config import CACHE
from animalclef.localfeat import normalize_local
from animalclef.similarity import fuse, rank_normalize
from sklearn.metrics import silhouette_score

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"



def main() -> int:
    embs = {}
    for b in ["miewid", "mega"]:
        p = CACHE / f"emb_{b}_lizard_test.npy"
        if p.exists():
            embs[b] = np.load(p)
    if not embs:
        raise FileNotFoundError("lizard test embeddings missing")

    sims = {}
    for b, e in embs.items():
        sims[b] = rank_normalize(e @ e.T)
    w = {"miewid": 0.5, "mega": 0.5}
    gsim = fuse(sims, {k: v for k, v in w.items() if k in sims})

    lsim_path = CACHE / "local_root_sift_lizard_test.npy"
    lsim = normalize_local(np.load(lsim_path)) if lsim_path.exists() else None
    if lsim is not None:
        print(f"local features: {(lsim > 0).sum() // 2} pairs, "
              f"max={lsim.max():.3f}")
    else:
        print("warning: no local feature cache for lizards")

    # Threshold selection without labels.
    #
    # Silhouette and partition-stability both degenerate to the all-singleton
    # solution for sparse fine-grained data, so we transfer a *density prior*
    # measured on the most similar species with labels (SalamanderID2025:
    # ARI is flat for mean cluster size 1.4-2.1 and peaks near 1.5, see
    # RESULTS.md).  We choose the threshold whose partition has mean cluster
    # size closest to 1.5.
    TARGET_MEAN_SIZE = 1.5
    target_k = max(2, int(round(len(gsim) / TARGET_MEAN_SIZE)))
    rows = []
    # Local pattern features are the only identity signal available zero-shot
    # (validated as decisive on SalamanderID2025); skip the pure-global variant.
    for wl in [0.5, 1.0, 1.5, 2.0]:
        base = gsim if lsim is None else (gsim + wl * lsim) / (1.0 + wl)
        for t in np.arange(0.30, 0.99, 0.01):
            labels = cl.agglomerative(base, float(t))
            n_clusters = len(set(labels))
            d = 1.0 - base
            np.fill_diagonal(d, 0.0)
            sil = (float(silhouette_score(d, labels, metric="precomputed"))
                   if 2 <= n_clusters < len(labels) else -1.0)
            rows.append({"w_local": wl, "threshold": float(t),
                         "n_clusters": n_clusters, "silhouette": sil,
                         "dist_to_target": abs(n_clusters - target_k)})

    # Prefer the density prior; among near-misses prefer higher silhouette.
    rows.sort(key=lambda r: (r["dist_to_target"], -r["silhouette"]))
    best = rows[0]
    print(f"target clusters={target_k} (mean size {TARGET_MEAN_SIZE})")
    print("\ntop candidates by density prior:")
    for r in rows[:8]:
        print(f"  w={r['w_local']:.1f} t={r['threshold']:.3f} "
              f"clusters={r['n_clusters']} sil={r['silhouette']:.3f}")

    out = {
        "dataset": "TexasHornedLizards",
        "zero_shot": True,
        "weights": {"miewid": 0.5, "mega": 0.5, "local": best["w_local"]},
        "threshold": best["threshold"],
        "use_rerank": False,
        "rr": [20, 6, 0.3],
        "post": {"merge_small": 1, "merge_threshold": 0.4,
                 "split_big": 10**9, "split_threshold": 0.65, "anchor": False},
        "consensus": False,
        "local_method": "root_sift",
        "local_weight": best["w_local"],
        "score_global_only": None,
        "score_with_local": None,
        "silhouette": best["silhouette"],
        "n_clusters": best["n_clusters"],
    }
    path = CONFIG_DIR / "best_TexasHornedLizards.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
