#!/usr/bin/env python
"""Per-species hyperparameter tuning on the open-set benchmark.

Stages (all evaluated on several identity splits, mean ARI selects the winner):
  1. backbone weights x clustering threshold
  2. k-reciprocal re-ranking on the best combinations
  3. post-processing grid (merge-small / split-big / anchor merge)
  4. consensus clustering across per-backbone partitions
  5. orientation-conditioned calibration (lynx) and local-feature fusion

Usage:
    python scripts/tune.py --species lynx --seeds 5
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from animalclef import cluster as cl
from animalclef.benchmark import ari
from animalclef.calibration import Calibrator
from animalclef.config import CACHE, SPECIES_WITH_TRAIN
from animalclef.data import load_metadata
from animalclef.pipeline import (Config, cluster_similarity,
                                 fit_calibrators, refine)
from animalclef.similarity import cosine, fuse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_core import build_splits, load_emb

OUT_DIR = Path(__file__).resolve().parents[1] / "configs"
OUT_DIR.mkdir(exist_ok=True)


class SplitData:
    """Cached calibrated similarities for one benchmark split."""

    def __init__(self, dataset, db, qry, embs, use_orientation=False):
        self.db, self.qry = db, qry
        self.true = qry["true_identity"].values
        self.use_orientation = use_orientation
        cfg = Config(dataset=dataset, use_orientation_calibration=use_orientation)
        db_embs = {b: e[db["_pos"].values] for b, e in embs.items()}
        q_embs = {b: e[qry["_pos"].values] for b, e in embs.items()}
        q_orient = qry["orientation"].values if use_orientation else None
        cals = fit_calibrators(db_embs, db, cfg)
        self.qq, self.qd = {}, {}
        for b in embs:
            self.qq[b] = cals[b].transform(cosine(q_embs[b], q_embs[b]), q_orient)
            self.qd[b] = cals[b].transform(cosine(q_embs[b], db_embs[b]))
        self.db_ids = np.asarray(db["identity"].astype(str).values, dtype=object)
        self._qq_cache, self._qd_cache = {}, {}

    def fused(self, weights):
        key = tuple(sorted((k, float(v)) for k, v in weights.items() if v > 0 and k in self.qq))
        if key not in self._qq_cache:
            self._qq_cache[key] = fuse(self.qq, dict(key))
        return self._qq_cache[key]

    def fused_qd(self, weights):
        key = tuple(sorted((k, float(v)) for k, v in weights.items() if v > 0 and k in self.qd))
        if key not in self._qd_cache:
            self._qd_cache[key] = fuse(self.qd, dict(key))
        return self._qd_cache[key]


def eval_config(splits, cfg: Config, weights: dict, rerank_sim=None,
                consensus=False) -> list[float]:
    out = []
    for sd in splits:
        sim = rerank_sim if rerank_sim is not None else sd.fused(weights)
        if consensus:
            parts = []
            for b in sd.qq:
                if weights.get(b, 0) > 0:
                    parts.append(cl.agglomerative(sd.qq[b], cfg.threshold))
            labels = cl.consensus_cluster(parts, 1.0 - cfg.threshold)
        else:
            labels = cluster_similarity(sim, cfg)
        labels = refine(labels, sim, cfg, sd.fused_qd(weights), sd.db_ids)
        out.append(ari(sd.true, labels))
    return out


def search(splits, dataset, cfg_base: Config, weights_grid, thresholds,
           label=""):
    best = None
    for weights in weights_grid:
        for t in thresholds:
            cfg = replace(cfg_base, threshold=float(t), weights=dict(weights))
            aris = eval_config(splits, cfg, weights)
            m = float(np.mean(aris))
            if best is None or m > best["ari"]:
                best = {"ari": m, "std": float(np.std(aris)), "weights": dict(weights),
                        "threshold": float(t), "label": label}
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True, choices=SPECIES_WITH_TRAIN)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--new-id-frac", type=float, default=0.4)
    ap.add_argument("--backbones", default="miewid,mega")
    ap.add_argument("--local", default="", help="comma list: root_sift,sift,kaze")
    from_s = time.time()

    args = ap.parse_args()
    ds = args.species
    meta = load_metadata()
    embs = {}
    for b in args.backbones.split(","):
        e = load_emb(b, ds, "train")
        if e is None:
            raise FileNotFoundError(f"missing embeddings for {b}/{ds}; run extract first")
        embs[b] = e

    splits = build_splits(meta, ds, range(args.seeds), args.new_id_frac)
    sds = [SplitData(ds, db, q, embs) for db, q in splits]
    print(f"[{ds}] {len(sds)} splits ready "
          f"(query sizes {[len(s.qry) for s in sds]})", flush=True)

    cfg_base = Config(dataset=ds, merge_small=1, split_big=10**9)
    backbone_names = list(embs)
    weight_grid = []
    for k in range(len(backbone_names) + 1):
        for combo in itertools.combinations(backbone_names, k):
            if not combo:
                continue
            w = {b: 1.0 for b in combo}
            if len(w) == 2:
                weight_grid.append({backbone_names[0]: 0.7, backbone_names[1]: 0.3})
                weight_grid.append({backbone_names[0]: 0.3, backbone_names[1]: 0.7})
            weight_grid.append(w)
    # dedupe
    seen, wg = set(), []
    for w in weight_grid:
        key = tuple(sorted(w.items()))
        if key not in seen:
            seen.add(key)
            wg.append(w)

    thresholds = np.arange(0.02, 0.86, 0.02)
    stage1 = search(sds, ds, cfg_base, wg, thresholds, "stage1")
    print(f"stage1: ARI={stage1['ari']:.4f} ± {stage1['std']:.4f} "
          f"weights={stage1['weights']} t={stage1['threshold']:.3f}", flush=True)
    best = stage1

    # ---- stage 2: re-ranking ----
    cfg2 = replace(cfg_base, threshold=stage1["threshold"], weights=stage1["weights"])
    rr_best = None
    for k1, k2, lam in itertools.product([5, 10, 20], [3, 6], [0.1, 0.3, 0.5]):
        cfg = replace(cfg2, use_rerank=True, rr_k1=k1, rr_k2=k2, rr_lambda=lam)
        aris = []
        for sd in sds:
            sim = cl.rerank_similarity(sd.fused(cfg.weights), k1, k2, lam)
            labels = cluster_similarity(sim, cfg)
            labels = refine(labels, sim, cfg,
                            sd.fused_qd(cfg.weights), sd.db_ids)
            aris.append(ari(sd.true, labels))
        m = float(np.mean(aris))
        if rr_best is None or m > rr_best["ari"]:
            rr_best = {"ari": m, "std": float(np.std(aris)), "rr": (k1, k2, lam)}
    use_rr = rr_best["ari"] > stage1["ari"]
    if use_rr:
        best = {"ari": rr_best["ari"], "std": rr_best["std"],
                "weights": stage1["weights"], "threshold": stage1["threshold"],
                "rr": rr_best["rr"], "label": "stage2"}
        print(f"stage2: re-ranking helps: {rr_best['rr']} ARI={rr_best['ari']:.4f}",
              flush=True)
    else:
        print(f"stage2: re-ranking does not help ({rr_best['ari']:.4f} < "
              f"{stage1['ari']:.4f})", flush=True)

    # Re-tune the threshold with the chosen re-ranking setting.
    def build_sims(weights, cfg):
        out = []
        for sd in sds:
            if cfg.use_rerank:
                out.append(cl.rerank_similarity(
                    sd.fused(weights), cfg.rr_k1, cfg.rr_k2, cfg.rr_lambda))
            else:
                out.append(sd.fused(weights))
        return out

    # ---- stage 3: post-processing ----
    base_cfg = replace(cfg_base, weights=best["weights"],
                       threshold=best["threshold"])
    if use_rr:
        base_cfg = replace(base_cfg, use_rerank=True,
                           rr_k1=best["rr"][0], rr_k2=best["rr"][1],
                           rr_lambda=best["rr"][2])
    sims = build_sims(base_cfg.weights, base_cfg)
    post_best = {"ari": -1}
    for merge_small, merge_thr, split_big, split_thr, anchor in itertools.product(
        [1, 2], [0.4, 0.6], [10**9, 60], [0.65], [False, True]
    ):
        cfg = replace(base_cfg, merge_small=merge_small, merge_threshold=merge_thr,
                      split_big=split_big, split_threshold=split_thr, anchor=anchor)
        aris = []
        for sd, sim in zip(sds, sims):
            labels = cluster_similarity(sim, cfg)
            labels = refine(labels, sim, cfg,
                            sd.fused_qd(cfg.weights), sd.db_ids)
            aris.append(ari(sd.true, labels))
        m = float(np.mean(aris))
        if m > post_best["ari"]:
            post_best = {"ari": m, "std": float(np.std(aris)),
                         "post": dict(merge_small=merge_small,
                                      merge_threshold=merge_thr,
                                      split_big=split_big,
                                      split_threshold=split_thr,
                                      anchor=anchor)}
    if post_best["ari"] > best["ari"]:
        best = {**best, **post_best, "label": "stage3"}
    print(f"stage3: post={post_best['post']} ARI={post_best['ari']:.4f}", flush=True)

    # ---- stage 4: consensus clustering ----
    cons_best = None
    for t in np.arange(0.3, 0.86, 0.05):
        cfg = replace(base_cfg, threshold=float(t))
        parts_all = []
        for sd in sds:
            parts = [cl.agglomerative(sd.qq[b], cfg.threshold)
                     for b in sd.qq if cfg.weights.get(b, 0) > 0]
            parts_all.append(parts)
        aris = []
        for sd, parts in zip(sds, parts_all):
            labels = cl.consensus_cluster(parts, cfg.threshold)
            labels = refine(labels, sd.fused(cfg.weights), cfg,
                            sd.fused_qd(cfg.weights), sd.db_ids)
            aris.append(ari(sd.true, labels))
        m = float(np.mean(aris))
        if cons_best is None or m > cons_best["ari"]:
            cons_best = {"ari": m, "std": float(np.std(aris)), "threshold": float(t)}
    if cons_best["ari"] > best["ari"]:
        best = {**best, "ari": cons_best["ari"], "std": cons_best["std"],
                "threshold": cons_best["threshold"], "consensus": True,
                "label": "stage4"}
    print(f"stage4: consensus ARI={cons_best['ari']:.4f}", flush=True)

    # ---- stage 5: orientation-conditioned calibration ----
    # Only meaningful when the metadata has real orientation information
    # (lynx camera traps); other species are mostly "unknown".
    best_orientation = False
    if (sds[0].qry["orientation"].astype(str) != "unknown").mean() > 0.5:
        sds_ori = [SplitData(ds, db, q, embs, use_orientation=True) for db, q in splits]
        cfg5 = replace(base_cfg, use_orientation_calibration=True)
        aris = []
        for sd in sds_ori:
            sim = sd.fused(cfg5.weights)
            if cfg5.use_rerank:
                sim = cl.rerank_similarity(sim, cfg5.rr_k1, cfg5.rr_k2, cfg5.rr_lambda)
            labels = cluster_similarity(sim, cfg5)
            labels = refine(labels, sim, cfg5, sd.fused_qd(cfg5.weights), sd.db_ids)
            aris.append(ari(sd.true, labels))
        m = float(np.mean(aris))
        print(f"stage5: orientation calibration ARI={m:.4f}", flush=True)
        if m > best["ari"]:
            best = {**best, "ari": m, "std": float(np.std(aris)),
                    "label": "stage5"}
            best_orientation = True

    final = {
        "dataset": ds,
        "score": best["ari"],
        "std": best["std"],
        "label": best["label"],
        "weights": best["weights"],
        "threshold": best["threshold"],
        "use_rerank": bool(use_rr and best["label"] != "stage1"),
        "rr": best.get("rr", [20, 6, 0.3]),
        "post": best.get("post", {}),
        "consensus": best.get("consensus", False),
        "use_orientation_calibration": best_orientation,
        "new_id_frac": args.new_id_frac,
        "seeds": args.seeds,
        "elapsed_s": time.time() - from_s,
    }
    out = OUT_DIR / f"best_{ds}.json"
    out.write_text(json.dumps(final, indent=2))
    print(f"\nBEST [{ds}] ARI={final['score']:.4f} ± {final['std']:.4f} "
          f"({final['label']}) -> {out}", flush=True)
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
