"""Chinese Whispers clustering helpers for similarity matrices.

This module clusters samples from a pairwise similarity matrix. It builds a
graph where nodes are images and edge weights are similarity scores, optionally
filters the graph with k-core and mutual-kNN constraints, then runs the Chinese
Whispers label-propagation algorithm to produce cluster labels. It also includes
a k-reciprocal re-ranking function that can refine similarity scores before
clustering.
"""

from collections import defaultdict
import numpy as np
import random
from src.utilities.graph_construction import (build_backbone_graph, 
                                build_k_reciprocal_graph, 
                                build_nx_graph_from_similarity,
                                extract_kcore_backbone
                                )


def chinese_whispers(sim_matrix, threshold, iterations, top_k, k_core_degree, seed=42, skip_mutual_knn=False):
    """Cluster samples with k-core-filtered Chinese Whispers.

    Args:
        sim_matrix: Square pairwise similarity matrix.
        threshold: Minimum similarity required to create graph edges.
        iterations: Maximum number of Chinese Whispers label-propagation passes.
        top_k: Number of neighbors used for mutual-kNN filtering.
        k_core_degree: Minimum degree used to extract the graph backbone.
        seed: Random seed controlling node update order.
        skip_mutual_knn: If True, use only backbone and threshold filtering. This
            is recommended when ``sim_matrix`` has already been re-ranked.

    Returns:
        Dense integer cluster labels from 0 to ``num_clusters - 1``.
    """
    # Seed both Python and NumPy randomness so shuffled node orders are
    # reproducible across runs.
    random.seed(seed)
    np.random.seed(seed)

    N = sim_matrix.shape[0]

    # Build the initial threshold graph, then extract the k-core backbone used as
    # the reliable subset of nodes for label propagation.
    G = build_nx_graph_from_similarity(sim_matrix, threshold)
    backbone = extract_kcore_backbone(G, k_core_degree)
    backbone_nodes = list(backbone.nodes())

    # If the backbone is empty after filtering, fall back to all nodes so every
    # sample can still receive a cluster label.
    if len(backbone_nodes) == 0:
        backbone_nodes = list(range(N))

    # Choose the graph construction mode. Mutual-kNN is stricter; backbone-only
    # is better after external re-ranking/refinement.
    if skip_mutual_knn:
        graph_list = build_backbone_graph(sim_matrix, backbone_nodes, threshold)
    else:
        graph_list = build_k_reciprocal_graph(sim_matrix, backbone_nodes, threshold, top_k)

    # Chinese Whispers starts with every node in its own cluster. During each
    # iteration, nodes adopt the strongest weighted neighboring label.
    labels = list(range(N))
    nodes = list(backbone_nodes)

    for _ in range(iterations):
        # Random update order is part of the Chinese Whispers algorithm and helps
        # avoid deterministic bias from node indexing.
        random.shuffle(nodes)
        changes = 0

        for node in nodes:
            neighbors = graph_list[node]
            if not neighbors:
                continue

            # Accumulate total edge weight per neighboring cluster label.
            label_weights = defaultdict(float)
            for nei, w in neighbors:
                label_weights[labels[nei]] += w

            if not label_weights:
                continue

            # Pick the label with the largest total neighbor weight. The
            # secondary key prefers the smaller label id on exact ties.
            best_label = max(label_weights.items(), key=lambda kv: (kv[1], -kv[0]))[0]

            if best_label != labels[node]:
                labels[node] = best_label
                changes += 1

        # Stop early once a full pass makes no label changes.
        if changes == 0:
            break

    # Compress arbitrary surviving label ids into a clean 0..K-1 label range.
    _, new_labels = np.unique(labels, return_inverse=True)
    return new_labels
