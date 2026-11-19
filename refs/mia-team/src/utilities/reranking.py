import numpy as np


def re_ranking(sim_matrix, k1=20, k2=6, lambda_value=0.3):
    """Refine a similarity matrix with k-reciprocal re-ranking.

    This implementation follows the common person/animal re-identification
    re-ranking idea: convert similarities to distances, build k-reciprocal
    neighborhoods, compute Jaccard distance between neighborhood distributions,
    and blend that distance with the original distance.

    Args:
        sim_matrix: Square pairwise similarity matrix with larger values meaning
            more similar samples.
        k1: Neighborhood size for k-reciprocal expansion.
        k2: Query-expansion neighborhood size. Values greater than 1 average
            neighborhood distributions over the top-k2 nearest samples.
        lambda_value: Blend weight for the original distance. Lower values rely
            more on Jaccard/k-reciprocal distance.

    Returns:
        Re-ranked similarity matrix in ``float32`` format.
    """
    # Work in float32 for memory efficiency on large pairwise matrices.
    sim_matrix = np.asarray(sim_matrix, dtype=np.float32)
    N = sim_matrix.shape[0]

    # Convert similarity to distance and clamp negative values caused by scores
    # above 1.0 or numerical noise.
    original_dist = 1.0 - sim_matrix
    original_dist = np.clip(original_dist, 0.0, None)

    # Column normalization reduces scale differences across samples before
    # ranking neighbors.
    col_max = np.max(original_dist, axis=0)
    col_max[col_max == 0] = 1.0
    original_dist = (original_dist / col_max).T.astype(np.float32)

    # V stores each sample's soft k-reciprocal neighborhood distribution.
    V = np.zeros((N, N), dtype=np.float32)

    # Lower distance means closer neighbor, so argsort gives nearest-first ranks.
    initial_rank = np.argsort(original_dist, axis=1).astype(np.int32)

    half_k1 = int(np.around(k1 / 2.0)) + 1
    for i in range(N):
        # Forward neighbors are i's nearest k1 samples. Backward ranks identify
        # which of those neighbors also rank i among their nearest k1 samples.
        forward = initial_rank[i, : k1 + 1]
        backward = initial_rank[forward, : k1 + 1]
        fi = np.where(backward == i)[0]
        k_reciprocal = forward[fi]
        k_reciprocal_expand = k_reciprocal.copy()

        # Expand the reciprocal set with neighbors whose own reciprocal set
        # strongly overlaps with i's reciprocal set.
        for q in k_reciprocal:
            q_forward = initial_rank[q, :half_k1]
            q_backward = initial_rank[q_forward, :half_k1]
            fi_q = np.where(q_backward == q)[0]
            q_kr = q_forward[fi_q]
            if len(q_kr) > 0 and len(np.intersect1d(q_kr, k_reciprocal)) > (2.0 / 3.0) * len(q_kr):
                k_reciprocal_expand = np.append(k_reciprocal_expand, q_kr)

        k_reciprocal_expand = np.unique(k_reciprocal_expand)

        # Convert distances inside the expanded reciprocal set into normalized
        # affinity weights.
        weight = np.exp(-original_dist[i, k_reciprocal_expand])
        wsum = np.sum(weight)
        if wsum > 0:
            V[i, k_reciprocal_expand] = weight / wsum

    # Query expansion smooths each row of V by averaging neighborhood
    # distributions from the nearest k2 samples.
    if k2 > 1:
        V_qe = np.zeros_like(V)
        for i in range(N):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe

    # Compute Jaccard distance between sparse reciprocal-neighborhood
    # distributions. inv_index speeds up lookup of rows with nonzero weight in
    # each column.
    jaccard_dist = np.zeros((N, N), dtype=np.float32)
    inv_index = [np.where(V[:, j] != 0)[0] for j in range(N)]

    for i in range(N):
        temp_min = np.zeros(N, dtype=np.float32)
        nz = np.where(V[i, :] != 0)[0]
        ind_imgs = [inv_index[j] for j in nz]
        for k in range(len(nz)):
            temp_min[ind_imgs[k]] += np.minimum(V[i, nz[k]], V[ind_imgs[k], nz[k]])
        jaccard_dist[i] = 1.0 - temp_min / (2.0 - temp_min + 1e-12)

    # Blend neighborhood-based distance with the original distance, then convert
    # back to similarity.
    final_dist = jaccard_dist * (1.0 - lambda_value) + original_dist * lambda_value
    final_dist = np.clip(final_dist, 0.0, 1.0)
    np.fill_diagonal(final_dist, 0.0)

    return (1.0 - final_dist).astype(np.float32)
