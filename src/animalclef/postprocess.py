"""Cluster post-processing.

These refinements operate on a hard partition plus the similarity matrix:

- ``merge_small``: absorb tiny clusters into their most similar neighbour when
  the link is strong enough.  Prevents the classic over-splitting failure where
  single-image clusters destroy ARI in sparse species (salamander).
- ``split_big``: re-cluster unusually large clusters at a stricter threshold,
  undoing greedy over-merges.
- ``anchor_merge``: merge query clusters that both look like the *same* known
  database individual (using calibrated query-database similarities).  This
  injects database structure into the test partition without forcing strict
  joint clustering, which public solutions found too rigid.
"""
from __future__ import annotations

import numpy as np


def relabel(labels: np.ndarray) -> np.ndarray:
    _, out = np.unique(labels, return_inverse=True)
    return out.astype(np.int32)


def cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(labels, return_counts=True)
    return dict(zip(vals.tolist(), counts.tolist()))


def merge_small(labels: np.ndarray, sim: np.ndarray, min_size: int = 2,
                merge_threshold: float = 0.5, max_passes: int = 5) -> np.ndarray:
    """Merge clusters smaller than ``min_size`` into their best neighbour."""
    labels = labels.copy()
    for _ in range(max_passes):
        sizes = cluster_sizes(labels)
        small = [c for c, s in sizes.items() if s < min_size]
        if not small:
            break
        merged_any = False
        for c in small:
            members = np.where(labels == c)[0]
            other = labels != c
            if not other.any():
                continue
            # Mean similarity from this cluster to every other image, then the
            # per-cluster mean via bincount (vectorised: O(n) per cluster).
            sims = sim[np.ix_(members, np.where(other)[0])].mean(axis=0)
            other_labels = labels[other]
            sums = np.bincount(other_labels, weights=sims)
            counts = np.bincount(other_labels)
            present = counts > 0
            means = np.where(present, sums / np.maximum(counts, 1), -np.inf)
            best_idx = int(np.argmax(means))
            best_sim, best_label = float(means[best_idx]), best_idx
            if best_label is not None and best_sim >= merge_threshold:
                labels[members] = best_label
                merged_any = True
        if not merged_any:
            break
    return relabel(labels)


def split_big(labels: np.ndarray, sim: np.ndarray, max_size: int,
              split_threshold: float, linkage: str = "average",
              min_split_size: int = 2) -> np.ndarray:
    """Re-cluster clusters larger than ``max_size`` at a stricter threshold."""
    from .cluster import agglomerative

    labels = labels.copy()
    next_label = labels.max() + 1
    for c, size in list(cluster_sizes(labels).items()):
        if size <= max_size:
            continue
        members = np.where(labels == c)[0]
        sub = agglomerative(sim[np.ix_(members, members)], split_threshold, linkage)
        sub_sizes = cluster_sizes(sub)
        if len(sub_sizes) <= 1:
            continue
        for sub_c, sub_size in sub_sizes.items():
            idx = members[sub == sub_c]
            if sub_size < min_split_size:
                continue
            labels[idx] = c if sub_c == 0 else next_label
            if sub_c != 0:
                next_label += 1
    return relabel(labels)


def anchor_merge(labels: np.ndarray, sim_qdb: np.ndarray, db_ids: np.ndarray,
                 min_score: float = 0.6, min_margin: float = 0.1,
                 min_cluster_size: int = 1) -> np.ndarray:
    """Merge clusters whose strongest database identity is the same."""
    labels = labels.copy()
    sizes = cluster_sizes(labels)
    unique_ids, inv = np.unique(db_ids.astype(str), return_inverse=True)
    # Per-cluster mean similarity to each database identity.
    cluster_ids = sorted(sizes)
    # Vectorised equivalent of
    #   scores[c, u] = mean_{i in c} max_{j in identity u} sim_qdb[i, j]
    # Sort database columns by identity and use reduceat for the per-image max;
    # this is the hot path in tuning (called thousands of times).
    order = np.argsort(inv, kind="stable")
    inv_sorted = inv[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(inv_sorted)) + 1))
    per_image_max = np.maximum.reduceat(
        sim_qdb[:, order].astype(np.float32), starts, axis=1
    )  # (n_query, n_identities)

    cid_to_idx = {c: i for i, c in enumerate(cluster_ids)}
    row_cluster = np.array([cid_to_idx[int(c)] for c in labels])
    sums = np.zeros((len(cluster_ids), len(unique_ids)), dtype=np.float64)
    np.add.at(sums, row_cluster, per_image_max)
    counts = np.bincount(row_cluster, minlength=len(cluster_ids)).astype(np.float64)
    scores = (sums / np.maximum(counts, 1.0)[:, None]).astype(np.float32)

    best = scores.argmax(axis=1)
    best_scores = scores[np.arange(len(cluster_ids)), best]
    if scores.shape[1] > 1:
        part = np.partition(scores, -2, axis=1)[:, -2:]
        second = np.minimum(part[:, 0], part[:, 1])
    else:
        second = np.zeros_like(best_scores)
    margin = best_scores - second

    label_to_anchor: dict[int, str] = {}
    for ci, c in enumerate(cluster_ids):
        if sizes[c] < min_cluster_size:
            continue
        if best_scores[ci] >= min_score and margin[ci] >= min_margin:
            label_to_anchor[c] = unique_ids[best[ci]]

    groups: dict[str, int] = {}
    for c in cluster_ids:
        anchor = label_to_anchor.get(c)
        if anchor is None:
            continue
        if anchor in groups:
            labels[labels == c] = groups[anchor]
        else:
            groups[anchor] = c
    return relabel(labels)
