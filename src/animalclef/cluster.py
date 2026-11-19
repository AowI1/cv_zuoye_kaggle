"""Clustering algorithms operating on similarity matrices."""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN


def to_distance(sim: np.ndarray) -> np.ndarray:
    d = 1.0 - np.asarray(sim, dtype=np.float64)
    d = np.clip(d, 0.0, 2.0)
    np.fill_diagonal(d, 0.0)
    return d


def agglomerative(sim: np.ndarray, threshold: float, linkage: str = "average",
                  n_clusters: int | None = None) -> np.ndarray:
    """Agglomerative clustering on a similarity matrix.

    ``threshold`` is a *similarity* threshold: pairs above it may be merged.
    """
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage=linkage,
        distance_threshold=None if n_clusters else max(1e-6, 1.0 - threshold),
    )
    return model.fit_predict(to_distance(sim))


def dbscan(sim: np.ndarray, eps: float, min_samples: int = 2,
           metric: str = "cosine") -> np.ndarray:
    """DBSCAN on a similarity matrix, thresholded like the reference baseline."""
    d = to_distance(sim)
    return DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(d)


def chinese_whispers(sim: np.ndarray, threshold: float, iterations: int = 20,
                     top_k: int = 20, k_core_degree: int = 1,
                     seed: int = 42) -> np.ndarray:
    """Thresholded-graph Chinese Whispers with k-core backbone filtering."""
    random.seed(seed)
    np.random.seed(seed)
    n = sim.shape[0]
    adj = sim >= threshold
    np.fill_diagonal(adj, False)

    # k-core backbone: iteratively drop nodes with degree < k_core_degree.
    keep = np.ones(n, dtype=bool)
    if k_core_degree > 1:
        deg = adj.sum(axis=1)
        active = deg >= k_core_degree
        for _ in range(10):
            new_active = active & ((adj & active[None, :]).sum(axis=1) >= k_core_degree)
            if new_active.sum() == active.sum():
                break
            active = new_active
        keep = active

    # Mutual-kNN filtering.
    neighbor = np.argsort(-sim, axis=1)[:, :min(top_k, n)]
    knn_mask = np.zeros_like(adj)
    rows = np.repeat(np.arange(n), neighbor.shape[1])
    knn_mask[rows, neighbor.ravel()] = True
    graph = adj & knn_mask & knn_mask.T
    # Isolated nodes keep their threshold edges so they can still attach.
    isolated = graph.sum(axis=1) == 0
    graph[isolated] = adj[isolated]

    labels = list(range(n))
    nodes = [i for i in range(n) if keep[i]] or list(range(n))
    for _ in range(iterations):
        random.shuffle(nodes)
        changes = 0
        for node in nodes:
            nbrs = np.where(graph[node])[0]
            if len(nbrs) == 0:
                continue
            weights = defaultdict(float)
            for nei in nbrs:
                weights[labels[nei]] += float(sim[node, nei])
            best = max(weights.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            if best != labels[node]:
                labels[node] = best
                changes += 1
        if changes == 0:
            break
    _, out = np.unique(labels, return_inverse=True)
    return out.astype(np.int32)


def rerank_similarity(sim: np.ndarray, k1: int = 20, k2: int = 6,
                      lambda_value: float = 0.3) -> np.ndarray:
    """k-reciprocal re-ranking (ported from the winning solution)."""
    sim = np.asarray(sim, dtype=np.float32)
    n = sim.shape[0]
    original_dist = 1.0 - sim
    original_dist = np.clip(original_dist, 0.0, None)
    col_max = np.max(original_dist, axis=0)
    col_max[col_max == 0] = 1.0
    original_dist = (original_dist / col_max).T.astype(np.float32)

    v = np.zeros((n, n), dtype=np.float32)
    initial_rank = np.argsort(original_dist, axis=1).astype(np.int32)
    half_k1 = int(np.around(k1 / 2.0)) + 1
    for i in range(n):
        forward = initial_rank[i, : k1 + 1]
        backward = initial_rank[forward, : k1 + 1]
        fi = np.where(backward == i)[0]
        k_reciprocal = forward[fi]
        expand = k_reciprocal.copy()
        for q in k_reciprocal:
            q_forward = initial_rank[q, :half_k1]
            q_backward = initial_rank[q_forward, :half_k1]
            fi_q = np.where(q_backward == q)[0]
            q_kr = q_forward[fi_q]
            if len(q_kr) > 0 and len(np.intersect1d(q_kr, k_reciprocal)) > (2.0 / 3.0) * len(q_kr):
                expand = np.append(expand, q_kr)
        expand = np.unique(expand)
        weight = np.exp(-original_dist[i, expand])
        wsum = weight.sum()
        if wsum > 0:
            v[i, expand] = weight / wsum

    if k2 > 1:
        v_qe = np.zeros_like(v)
        for i in range(n):
            v_qe[i] = np.mean(v[initial_rank[i, :k2]], axis=0)
        v = v_qe

    jaccard = np.zeros((n, n), dtype=np.float32)
    inv_index = [np.where(v[:, j] != 0)[0] for j in range(n)]
    for i in range(n):
        temp_min = np.zeros(n, dtype=np.float32)
        nz = np.where(v[i] != 0)[0]
        for k in nz:
            temp_min[inv_index[k]] += np.minimum(v[i, k], v[inv_index[k], k])
        jaccard[i] = 1.0 - temp_min / (2.0 - temp_min + 1e-12)

    final = jaccard * (1.0 - lambda_value) + original_dist * lambda_value
    final = np.clip(final, 0.0, 1.0)
    np.fill_diagonal(final, 0.0)
    return (1.0 - final).astype(np.float32)


def coassociation(partitions: list[np.ndarray]) -> np.ndarray:
    """Consensus matrix: fraction of partitions placing two images together."""
    n = len(partitions[0])
    co = np.zeros((n, n), dtype=np.float32)
    for p in partitions:
        co += (p[:, None] == p[None, :]).astype(np.float32)
    co /= float(len(partitions))
    return co


def consensus_cluster(partitions: list[np.ndarray], threshold: float = 0.5,
                      linkage: str = "average") -> np.ndarray:
    """Cluster the co-association matrix (partition-level ensemble)."""
    co = coassociation(partitions)
    model = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage=linkage,
        distance_threshold=max(1e-6, 1.0 - threshold),
    )
    return model.fit_predict(1.0 - co)
