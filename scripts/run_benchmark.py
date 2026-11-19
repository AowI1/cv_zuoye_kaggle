#!/usr/bin/env python
"""Evaluate pipeline configurations on the identity-level open-set benchmark.

Example:
    python scripts/run_benchmark.py --species all --seeds 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from animalclef.benchmark import ari
from animalclef.config import CACHE, SPECIES_WITH_TRAIN
from animalclef.data import load_metadata
from animalclef.pipeline import Config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_core import build_splits, load_emb, run_config_on_split


def default_configs(dataset: str) -> list[Config]:
    base = [
        Config(dataset=dataset, weights={"miewid": 1.0}, method="agglo", threshold=0.5),
        Config(dataset=dataset, weights={"mega": 1.0}, method="agglo", threshold=0.5),
        Config(dataset=dataset, weights={"miewid": 0.5, "mega": 0.5}, method="agglo", threshold=0.5),
    ]
    for t in (0.3, 0.4, 0.6, 0.7):
        for w in ([1.0, 0.0], [0.0, 1.0], [0.5, 0.5]):
            base.append(Config(
                dataset=dataset,
                weights={"miewid": w[0], "mega": w[1]},
                method="agglo", threshold=t,
            ))
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default="all")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--new-id-frac", type=float, default=0.4)
    ap.add_argument("--backbones", default="miewid,mega")
    args = ap.parse_args()

    meta = load_metadata()
    species = SPECIES_WITH_TRAIN if args.species == "all" else [args.species]
    backbones = args.backbones.split(",")

    rows = []
    for ds in species:
        embs = {}
        for b in backbones:
            e = load_emb(b, ds, "train")
            if e is not None:
                embs[b] = e
        if not embs:
            print(f"[skip] {ds}: no embeddings cached")
            continue
        print(f"== {ds}: backbones={list(embs)} ==")
        splits = build_splits(meta, ds, range(args.seeds), args.new_id_frac)
        for cfg in default_configs(ds):
            cfg.weights = {k: v for k, v in cfg.weights.items() if k in embs and v > 0}
            if not cfg.weights:
                continue
            aris = []
            for db, qry in splits:
                try:
                    pred = run_config_on_split(cfg, db, qry, embs)
                    aris.append(ari(qry["true_identity"], pred))
                except Exception as e:  # noqa: BLE001
                    print(f"  [error] {e}")
            if aris:
                tag = f"{cfg.method}_t{cfg.threshold:.2f}_w" + "-".join(
                    f"{k}{v:.1f}" for k, v in cfg.weights.items())
                rows.append({"dataset": ds, "config": tag,
                             "ari": float(np.mean(aris)), "std": float(np.std(aris)),
                             "n": len(aris), "ari_list": aris})
                print(f"  {tag:<32s} ARI={np.mean(aris):.4f} ± {np.std(aris):.4f}")

    df = pd.DataFrame(rows).sort_values(["dataset", "ari"], ascending=[True, False])
    out = CACHE.parent / "benchmark_results.csv"
    df.drop(columns=["ari_list"]).to_csv(out, index=False)
    print(f"\nsaved {out}")
    if len(df):
        best = df.groupby("dataset").head(3)
        print("\nTop configs:")
        print(best[["dataset", "config", "ari", "std"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
