#!/usr/bin/env python
"""Component ablation on the open-set benchmark.

Every configuration gets its clustering threshold tuned on the same splits, so
the reported differences come from the component under test, not from
threshold luck.

    python scripts/ablation.py --species all --seeds 3
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from animalclef import cluster as cl
from animalclef.benchmark import ari
from animalclef.config import SPECIES_WITH_TRAIN
from animalclef.data import load_metadata
from animalclef.pipeline import Config, cluster_similarity, refine
from animalclef.similarity import fuse
from tune import SplitData
from benchmark_core import build_splits, load_emb


def eval_cfg(sds, cfg: Config, weights, use_rr=False, rr=(20, 6, 0.3),
             threshold: float | None = None):
    aris = []
    for sd in sds:
        sim = sd.fused(weights)
        if use_rr:
            sim = cl.rerank_similarity(sim, rr[0], rr[1], rr[2])
        c = replace(cfg, threshold=cfg.threshold if threshold is None else threshold)
        labels = cluster_similarity(sim, c)
        labels = refine(labels, sim, c, sd.fused_qd(weights), sd.db_ids)
        aris.append(ari(sd.true, labels))
    return float(np.mean(aris)), float(np.std(aris))


def best_threshold(sds, cfg, weights, use_rr=False, rr=(20, 6, 0.3),
                   grid=np.arange(0.02, 0.9, 0.02)):
    best = (-1, None)
    for t in grid:
        m, _ = eval_cfg(sds, cfg, weights, use_rr, rr, float(t))
        if m > best[0]:
            best = (m, float(t))
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default="all")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    meta = load_metadata()
    species = SPECIES_WITH_TRAIN if args.species == "all" else [args.species]
    rows = []
    for ds in species:
        embs = {b: e for b in ["miewid", "mega"]
                if (e := load_emb(b, ds, "train")) is not None}
        splits = build_splits(meta, ds, range(args.seeds))
        sds = [SplitData(ds, db, q, embs) for db, q in splits]
        cfg0 = Config(dataset=ds, merge_small=1, split_big=10**9)

        configs = []
        if "mega" in embs:
            configs.append(("mega only", {"mega": 1.0}, False, None))
        if "miewid" in embs:
            configs.append(("miewid only", {"miewid": 1.0}, False, None))
        if len(embs) == 2:
            configs.append(("equal fusion 0.5/0.5",
                            {"miewid": 0.5, "mega": 0.5}, False, None))
            configs.append(("tuned fusion",
                            {"miewid": 0.7, "mega": 0.3}, False, None))
            configs.append(("tuned fusion + rerank",
                            {"miewid": 0.7, "mega": 0.3}, True, (20, 6, 0.3)))
        configs.append(("best fusion + anchor",
                        {"miewid": 0.7, "mega": 0.3} if len(embs) == 2
                        else {"miewid": 1.0}, False, None))

        for name, weights, use_rr, rr in configs:
            weights = {k: v for k, v in weights.items() if k in embs}
            cfg = replace(cfg0)
            if "anchor" in name:
                cfg = replace(cfg, anchor=True, anchor_score=0.6,
                              anchor_margin=0.1)
            m, t = best_threshold(sds, cfg, weights, use_rr, rr or (20, 6, 0.3))
            # re-evaluate the std at the selected threshold
            _, s = eval_cfg(sds, cfg, weights, use_rr, rr or (20, 6, 0.3), t)
            rows.append({"dataset": ds, "config": name, "ari": m, "std": s,
                         "threshold": t})
            print(f"{ds:<20s} {name:<34s} ARI={m:.4f} ± {s:.4f} (t={t:.2f})",
                  flush=True)

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "ablation_results.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
