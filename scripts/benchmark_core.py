"""Shared helpers for benchmark/tuning scripts."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from animalclef.benchmark import make_identity_split
from animalclef.config import CACHE, SPECIES_KEY, SPECIES_WITH_TRAIN
from animalclef.pipeline import Config, fit_calibrators, run_species

# Test-set sizes of the real competition (used to size pseudo-query sets).
REAL_TEST_SIZE = {
    "LynxID2025": 946,
    "SalamanderID2025": 689,
    "SeaTurtleID2022": 500,
    "TexasHornedLizards": 274,
}


def load_emb(backbone: str, dataset: str, split: str, suffix: str = "") -> np.ndarray | None:
    key = SPECIES_KEY[dataset]
    p = CACHE / f"emb_{backbone}_{key}_{split}{suffix}.npy"
    return np.load(p) if p.exists() else None


def available_backbones(dataset: str, splits=("train", "test"), suffix: str = "") -> list[str]:
    out = []
    for b in ["miewid", "mega", "sift", "root_sift", "kaze"]:
        if all((CACHE / f"emb_{b}_{SPECIES_KEY[dataset]}_{s}.npy").exists() or b.startswith(("sift", "root", "kaze")) for s in splits):
            out.append(b)
    return [b for b in out if not b.startswith(("sift", "root", "kaze"))]


def build_splits(meta: pd.DataFrame, dataset: str, seeds: range,
                 new_id_frac: float = 0.4, query_size: int | None = None):
    """Return list of (db_df, query_df) identity splits."""
    qs = query_size or min(REAL_TEST_SIZE[dataset], 600)
    splits = []
    for seed in seeds:
        db, qry = make_identity_split(
            meta, dataset, seed=seed, new_id_frac=new_id_frac, query_size=qs,
        )
        splits.append((db, qry))
    return splits


def run_config_on_split(
    cfg: Config,
    db: pd.DataFrame,
    qry: pd.DataFrame,
    emb_train: dict[str, np.ndarray],
    qry_source: str = "train",
    emb_test: dict[str, np.ndarray] | None = None,
    local_sims: dict[tuple, np.ndarray] | None = None,
):
    """Run a Config on one split; returns predicted labels for the query rows."""
    if qry_source == "train":
        q_embs = {b: e[qry["_pos"].values] for b, e in emb_train.items()}
        d_embs = {b: e[db["_pos"].values] for b, e in emb_train.items()}
    else:
        # Pseudo-query comes from the real test split (no labels) - unused here.
        assert emb_test is not None
        q_embs = {b: e for b, e in emb_test.items()}
        d_embs = {b: e[db["_pos"].values] for b, e in emb_train.items()}

    if local_sims is not None:
        q_embs = dict(q_embs)  # local sims handled by caller via cfg.weights
    return run_species(q_embs, qry, d_embs, db, cfg)


def fmt_results(rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    g = df.groupby(["dataset", "config"])["ari"].agg(["mean", "std", "count"]).reset_index()
    lines = []
    for _, r in g.iterrows():
        lines.append(
            f"{r['dataset']:<20s} {r['config']:<28s} ARI={r['mean']:.4f} "
            f"± {0.0 if pd.isna(r['std']) else r['std']:.4f} (n={int(r['count'])})"
        )
    return "\n".join(lines)
