"""Similarity computation and multi-backbone fusion."""
from __future__ import annotations

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix between L2-normalised embeddings."""
    return (a @ b.T).astype(np.float32)


def topk_neighbors(sim: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and values of the top-k most similar entries per row."""
    if k >= sim.shape[1]:
        order = np.argsort(-sim, axis=1)
    else:
        part = np.argpartition(-sim, k, axis=1)[:, :k]
        order = np.take_along_axis(
            part, np.argsort(-np.take_along_axis(sim, part, axis=1), axis=1), axis=1
        )
    return order, np.take_along_axis(sim, order, axis=1)


def fuse(sims: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted average of (calibrated) similarity matrices."""
    total = None
    wsum = 0.0
    for name, w in weights.items():
        if w == 0.0 or name not in sims:
            continue
        s = sims[name]
        total = w * s if total is None else total + w * s
        wsum += w
    if total is None:
        raise ValueError("all fusion weights are zero")
    return (total / wsum).astype(np.float32)


def rank_normalize(sim: np.ndarray) -> np.ndarray:
    """Map pairwise similarities to [0, 1] by within-matrix rank.

    Used for zero-shot species where no labelled calibration data exists.
    """
    n = sim.shape[0]
    iu = np.triu_indices(n, 1)
    vals = sim[iu]
    ranks = np.argsort(np.argsort(vals)).astype(np.float32) / max(1, len(vals) - 1)
    out = np.zeros_like(sim, dtype=np.float32)
    out[iu] = ranks
    out[(iu[1], iu[0])] = ranks
    np.fill_diagonal(out, 1.0)
    return out
