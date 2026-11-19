"""Open-set benchmark harness.

The competition test set has no public labels, so we build a faithful local
evaluation on the *labelled database split*: identities are divided into
"known" (present in the pseudo-database) and "new" (only appear in the
pseudo-query), and query images of known identities are held out as
re-observations.  This mirrors the open-world structure of the real task and
gives us ground truth to measure ARI.

The official score is the mean ARI over the four species; TexasHornedLizards
ships without any training data, so it cannot be simulated this way and is
tuned separately with unsupervised criteria.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def make_identity_split(
    meta: pd.DataFrame,
    dataset: str,
    seed: int = 0,
    new_id_frac: float = 0.4,
    query_size: int = 500,
    min_images_for_new: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the labelled rows of one species into pseudo-db / pseudo-query.

    Returns (db, query). Both frames keep the original metadata plus a
    ``true_identity`` column (for query: the hidden identity, where new
    identities are disjoint from db identities).
    """
    train = meta[(meta["dataset"] == dataset) & (meta["split"] == "train")].copy()
    if len(train) == 0:
        raise ValueError(f"{dataset} has no training rows")
    train["_pos"] = np.arange(len(train))
    rng = np.random.default_rng(seed)
    ids = train["identity"].unique()

    n_new = int(round(len(ids) * new_id_frac))
    new_ids = set(rng.choice(ids, size=n_new, replace=False))
    new_rows = train[train["identity"].isin(new_ids)]
    # Keep the pseudo-query at the size of the real test split: cap the number
    # of images per new identity so identity-rich species (lynx) do not blow
    # the benchmark up beyond the real 946-image query set.
    if len(new_rows) > query_size:
        cap = max(1, int(np.ceil(query_size / max(1, len(new_ids)))))
        new_rows = new_rows.groupby("identity", group_keys=False).head(cap)
        if len(new_rows) > query_size:
            new_rows = new_rows.iloc[rng.permutation(len(new_rows))[:query_size]]
        new_ids = set(new_rows["identity"].unique())

    known = train[~train["identity"].isin(new_ids)]
    # Sample query images per identity, always leaving at least one in db.
    budget = max(0, query_size - len(new_rows))
    counts = known.groupby("identity").size()
    eligible = counts[counts >= 2]
    if len(eligible) == 0:
        raise ValueError("no identity has >=2 images")

    frac = min(0.8, budget / max(1, int(eligible.sum())))
    q_parts = []
    for ident, group in known.groupby("identity"):
        n = len(group)
        if n < 2:
            continue
        k = min(n - 1, max(1, int(round(frac * n))))
        take = rng.choice(group.index.values, size=k, replace=False)
        q_parts.append(group.loc[take])
    q_known = pd.concat(q_parts) if q_parts else known.iloc[0:0]

    # Keep the total pseudo-query at the real test size: trim known-identity
    # query rows (never the new identities, which define the open-set part).
    if len(q_known) + len(new_rows) > query_size and len(q_known) > 0:
        excess = len(q_known) + len(new_rows) - query_size
        drop = rng.permutation(len(q_known))[:min(excess, len(q_known))]
        q_known = q_known.drop(q_known.index[drop])

    query = pd.concat([q_known, new_rows])
    db = train.drop(index=query.index)
    query = query.copy()
    query["true_identity"] = query["identity"].astype(str)
    db = db.copy()
    db["true_identity"] = db["identity"].astype(str)
    return db.reset_index(drop=True), query.reset_index(drop=True)


def ari(true_ids, pred_labels) -> float:
    return float(adjusted_rand_score(true_ids, pred_labels))


def summarize(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    return (
        df.groupby(["dataset", "config"])["ari"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
