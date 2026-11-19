#!/usr/bin/env python
"""Unit test of the benchmark -> calibration -> clustering pipeline on synthetic data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from animalclef import cluster as cl
from animalclef import postprocess as pp
from animalclef.benchmark import ari, make_identity_split
from animalclef.calibration import Calibrator
from animalclef.pipeline import Config, run_species
from animalclef.similarity import cosine, fuse


def make_fake(seed=0, n_ids=60, per_id=4, dim=64, noise=0.12):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_ids, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    embs, ids = [], []
    for i in range(n_ids):
        for _ in range(per_id):
            v = centers[i] + noise * rng.normal(size=dim)
            embs.append(v / np.linalg.norm(v))
            ids.append(i)
    meta = pd.DataFrame({
        "image_id": np.arange(len(ids)),
        "identity": [f"id{i}" for i in ids],
        "dataset": "Fake",
        "split": "train",
        "orientation": "unknown",
        "abs_path": ["x"] * len(ids),
    })
    return meta, np.array(embs, dtype=np.float32)


def main() -> int:
    meta, emb = make_fake()
    db, qry = make_identity_split(meta, "Fake", seed=0, new_id_frac=0.4,
                                 query_size=100)
    assert (db["_pos"].isin(qry["_pos"])).sum() == 0, "db/query overlap"
    db_ids = set(db["identity"])
    new_in_query = qry[~qry["identity"].isin(db_ids)]
    assert len(new_in_query) > 0, "expected new identities in query"
    print(f"split ok: db={len(db)} query={len(qry)} "
          f"new-id query rows={len(new_in_query)}")

    db_embs = {"mega": emb[db["_pos"].values]}
    q_embs = {"mega": emb[qry["_pos"].values]}
    cfg = Config(dataset="Fake", weights={"mega": 1.0}, threshold=0.5,
                 merge_small=2, merge_threshold=0.4, split_big=10**9)
    labels = run_species(q_embs, qry, db_embs, db, cfg)
    score = ari(qry["true_identity"], labels)
    print(f"pipeline ARI={score:.3f} clusters={len(set(labels))}")
    assert score > 0.6, f"pipeline broken, ARI={score}"

    # calibration sanity: scores map to [0,1] and separate same/diff
    cal = Calibrator().fit(cosine(emb, emb), np.asarray(meta["identity"]))
    csim = cal.transform(cosine(emb, emb))
    same = np.asarray(meta["identity"].values)[:, None] == np.asarray(meta["identity"].values)[None, :]
    np.fill_diagonal(same, False)
    print(f"calibration: same={csim[same].mean():.3f} diff={csim[~same].mean():.3f}")
    assert csim[same].mean() > csim[~same].mean()

    # postprocess edge cases
    labels2 = pp.merge_small(np.array([0, 0, 1, 2, 2, 3]), np.eye(6, dtype=np.float32),
                             min_size=2, merge_threshold=-1.0)
    assert len(set(labels2)) == 2, labels2
    # A cluster of 6 that really contains two coherent groups must be split.
    sim_split = np.zeros((6, 6), dtype=np.float32)
    sim_split[:3, :3] = 0.95
    sim_split[3:, 3:] = 0.95
    np.fill_diagonal(sim_split, 1.0)
    labels3 = pp.split_big(np.zeros(6, dtype=int), sim_split,
                           max_size=4, split_threshold=0.9)
    assert len(set(labels3)) == 2, labels3
    labels4 = pp.anchor_merge(np.array([0, 0, 1, 1]), np.ones((4, 4)),
                              np.array(["a", "a", "a", "a"]),
                              min_score=0.0, min_margin=-1.0)
    assert len(set(labels4)) == 1, labels4

    # consensus
    p1 = np.array([0, 0, 1, 1, 2, 2])
    p2 = np.array([0, 0, 1, 1, 1, 1])
    cons = cl.consensus_cluster([p1, p2], 0.4)
    print("consensus labels:", cons)
    assert len(set(cons)) <= 2

    print("UNIT TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
