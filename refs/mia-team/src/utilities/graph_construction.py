import numpy as np
import networkx as nx

def get_topk_neighbors(sim_matrix, top_k):
    """Return the top-k most similar neighbor indices for each sample.

    Args:
        sim_matrix: Square pairwise similarity matrix.
        top_k: Number of neighbors to keep per sample, excluding the sample
            itself.

    Returns:
        A list where each row contains the indices of the top-k neighbors for
        the corresponding sample.
    """
    N = sim_matrix.shape[0]

    # A node cannot use itself as a neighbor, so the maximum valid k is N - 1.
    top_k = min(top_k, N - 1)

    # Copy before modifying the diagonal so the caller's similarity matrix stays
    # unchanged.
    sims = sim_matrix.copy()

    # Remove self-matches from consideration by making diagonal scores invalid.
    np.fill_diagonal(sims, -np.inf)

    # argpartition is faster than full sorting when only the top-k indices are
    # needed. The returned top-k entries are not guaranteed to be internally
    # sorted, which is fine because later stages only need membership.
    topk_idx = np.argpartition(sims, -top_k, axis=1)[:, -top_k:]
    return topk_idx.tolist()


def build_nx_graph_from_similarity(sim_matrix, threshold=0.3):
    """Build a weighted NetworkX graph by thresholding the similarity matrix.

    Args:
        sim_matrix: Square pairwise similarity matrix.
        threshold: Minimum similarity required to create an edge.

    Returns:
        Undirected NetworkX graph whose edge weights come from ``sim_matrix``.
    """
    # Work on a copy so thresholding does not mutate the original similarities.
    adj_matrix = sim_matrix.copy()

    # Self-edges are not useful for clustering and can interfere with graph
    # analysis, so remove the diagonal.
    np.fill_diagonal(adj_matrix, 0)

    # Keep only confident similarities as graph edges.
    adj_matrix[adj_matrix < threshold] = 0

    # NetworkX treats nonzero matrix entries as weighted undirected edges.
    G = nx.from_numpy_array(adj_matrix)
    return G


def extract_kcore_backbone(G, k_core_degree=2):
    """Extract a stable k-core backbone from a graph.

    The k-core keeps only nodes with at least ``k_core_degree`` connections
    inside the retained subgraph. This removes weakly connected nodes before
    label propagation.

    Args:
        G: NetworkX graph built from similarity scores.
        k_core_degree: Minimum internal degree required for the backbone.

    Returns:
        The k-core graph when available; otherwise the original graph as a safe
        fallback.
    """
    # If no edges exist, k-core extraction cannot add useful structure.
    if G.number_of_edges() == 0:
        return G

    try:
        backbone = nx.k_core(G, k=k_core_degree)

        # If the chosen k is too strict and removes every node, fall back to the
        # original graph so clustering still has nodes to process.
        if backbone.number_of_nodes() == 0:
            return G
        return backbone
    except Exception:
        # NetworkX can raise for invalid/degenerate graph cases. Falling back to
        # the original graph keeps the pipeline robust.
        return G


def build_k_reciprocal_graph(sim_matrix, backbone_nodes, threshold, top_k):
    """Build an adjacency list using backbone, threshold, and mutual-kNN filters.

    An edge is kept only when:
    1. both nodes are in the k-core backbone,
    2. each node is in the other's top-k neighbor list,
    3. their similarity is at least ``threshold``.

    Args:
        sim_matrix: Square pairwise similarity matrix.
        backbone_nodes: Node indices retained by k-core filtering.
        threshold: Minimum similarity required for an edge.
        top_k: Number of nearest neighbors used for reciprocal filtering.

    Returns:
        Weighted adjacency list, where ``graph_list[i]`` contains
        ``(neighbor_index, similarity_weight)`` pairs.
    """
    N = sim_matrix.shape[0]

    # Compute top-k neighbors for every node and convert them to a boolean
    # adjacency matrix for fast mutual-neighbor checks.
    topk = np.array(get_topk_neighbors(sim_matrix, top_k))
    k_nearest = np.zeros((N, N), dtype=bool)
    row_indices = np.arange(N).reshape(-1, 1)
    k_nearest[row_indices, topk] = True

    # Mutual-kNN means i selected j and j selected i. This reduces one-way noisy
    # matches that often appear in dense similarity matrices.
    mutual_knn = k_nearest & k_nearest.T

    # Build a mask that allows only edges where both endpoints are part of the
    # k-core backbone.
    in_backbone = np.zeros(N, dtype=bool)
    in_backbone[backbone_nodes] = True
    backbone_mask = in_backbone[:, None] & in_backbone[None, :]

    # Combine all filters into the final edge mask.
    valid_sim = sim_matrix >= threshold
    final_mask = mutual_knn & backbone_mask & valid_sim
    sources, targets = np.nonzero(final_mask)
    weights = sim_matrix[sources, targets]

    # Convert the dense mask into the adjacency-list format used by Chinese
    # Whispers label propagation.
    graph_list = [[] for _ in range(N)]
    for s, t, w in zip(sources, targets, weights):
        graph_list[s].append((t, w))
    return graph_list


def build_backbone_graph(sim_matrix, backbone_nodes, threshold):
    """Build a backbone adjacency list without mutual-kNN filtering.

    This is useful when re-ranking has already refined the similarity matrix.
    In that case, applying mutual-kNN again can be too restrictive, so this
    function keeps all threshold-passing edges between backbone nodes.

    Args:
        sim_matrix: Square pairwise similarity matrix.
        backbone_nodes: Node indices retained by k-core filtering.
        threshold: Minimum similarity required for an edge.

    Returns:
        Weighted adjacency list for Chinese Whispers.
    """
    N = sim_matrix.shape[0]

    # Mark which nodes are part of the backbone.
    in_backbone = np.zeros(N, dtype=bool)
    in_backbone[backbone_nodes] = True
    backbone_mask = in_backbone[:, None] & in_backbone[None, :]

    # Remove self-edges, then keep threshold-passing similarities whose endpoints
    # both belong to the backbone.
    adj = sim_matrix.copy()
    np.fill_diagonal(adj, 0)
    valid = (adj >= threshold) & backbone_mask
    sources, targets = np.nonzero(valid)
    weights = sim_matrix[sources, targets]

    # Convert matrix edges to an adjacency list. Cast values to Python int/float
    # for easier downstream handling and debugging.
    graph_list = [[] for _ in range(N)]
    for s, t, w in zip(sources, targets, weights):
        graph_list[s].append((int(t), float(w)))
    return graph_list
