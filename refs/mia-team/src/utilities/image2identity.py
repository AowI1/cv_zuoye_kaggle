"""Utilities for assigning test samples to known training identities.

This module converts image-level similarity scores into identity-level scores.
Given a test-vs-train similarity matrix and the identity label of each training
image, it groups train images by identity, averages each test sample's strongest
matches per identity, and summarizes the best known-identity prediction with a
confidence margin.
"""

import numpy as np
from collections import defaultdict


def build_identity_index(train_ids):
    """Group training image indices by identity label.

    Args:
        train_ids: Sequence of identity labels, one label per training image.

    Returns:
        Dictionary mapping each identity label to the list of train-image indices
        that belong to that identity.
    """
    # defaultdict(list) lets each identity collect indices without manually
    # checking whether the identity has been seen before.
    identity_to_indices = defaultdict(list)

    # Store every training-image position under its known identity. These
    # positions are later used as columns in the test-vs-train similarity matrix.
    for idx, identity in enumerate(train_ids):
        identity_to_indices[identity].append(idx)

    # Return a normal dict so callers do not accidentally create new empty keys
    # by reading a missing identity.
    return dict(identity_to_indices)


def compute_identity_scores(sim_test_train, train_ids, top_r=3):
    """Convert test-vs-train image similarities into test-vs-identity scores.

    For each known identity, this function gathers all train columns belonging to
    that identity. A test sample's score for the identity is the mean of its top
    ``top_r`` similarities to that identity's training images. If an identity has
    fewer than ``top_r`` images, all available images are averaged.

    Args:
        sim_test_train: 2D array where rows are test samples and columns are
            training samples.
        train_ids: Sequence of identity labels aligned with the train columns of
            ``sim_test_train``.
        top_r: Number of strongest train matches per identity to average.

    Returns:
        A tuple ``(identity_names, identity_scores)`` where ``identity_names`` is
        the column order of the identity score matrix, and ``identity_scores`` has
        shape ``(num_test, num_identities)``.
    """
    # Build the mapping from each identity name to the train columns where that
    # identity appears.
    identity_to_indices = build_identity_index(train_ids)
    identity_names = list(identity_to_indices.keys())

    # Allocate one score column per known identity for every test sample.
    num_test = sim_test_train.shape[0]
    num_ids = len(identity_names)
    identity_scores = np.zeros((num_test, num_ids), dtype=np.float32)

    # Fill each identity column independently by aggregating the strongest image
    # matches from that identity's training examples.
    for j, identity in enumerate(identity_names):
        cols = identity_to_indices[identity]
        sims = sim_test_train[:, cols]
        r = min(top_r, sims.shape[1])

        # If the identity has r or fewer train images, all of them are used. This
        # avoids unnecessary partitioning and handles single-image identities.
        if r == sims.shape[1]:
            identity_scores[:, j] = sims.mean(axis=1)
        else:
            # np.partition selects the top-r values per row without fully sorting
            # every similarity row, which is faster for large train sets.
            top_vals = np.partition(sims, -r, axis=1)[:, -r:]
            identity_scores[:, j] = top_vals.mean(axis=1)

    return identity_names, identity_scores


def summarize_best_identities(identity_names, identity_scores):
    """Find each test sample's best known identity and confidence margin.

    Args:
        identity_names: Identity labels corresponding to the columns of
            ``identity_scores``.
        identity_scores: 2D array of test-vs-identity scores produced by
            ``compute_identity_scores``.

    Returns:
        A tuple containing:
        - best identity label per test sample.
        - best identity score per test sample.
        - second-best identity score per test sample.
        - margin between best and second-best scores.
    """
    # Locate the highest-scoring identity column for each test sample.
    best_idx = identity_scores.argmax(axis=1)

    # Gather the best score from each row using the selected column indices.
    best_scores = identity_scores[np.arange(identity_scores.shape[0]), best_idx]

    # Convert names to an object array so label strings/objects can be indexed by
    # the NumPy array of best column positions.
    best_identity_names = np.array(identity_names, dtype=object)[best_idx]

    # The margin needs the second-best score. With two or more identities, use
    # partition to get the top two scores per row without sorting all identities.
    if identity_scores.shape[1] >= 2:
        top2 = np.partition(identity_scores, -2, axis=1)[:, -2:]
        second_best_scores = np.minimum(top2[:, 0], top2[:, 1])
    else:
        # If there is only one known identity, there is no competitor. A zero
        # second-best score makes the margin equal to the best score.
        second_best_scores = np.zeros(identity_scores.shape[0], dtype=np.float32)

    # Margin measures how clearly the best identity beats the nearest competitor.
    # Downstream thresholding can use it as a confidence signal.
    margins = best_scores - second_best_scores
    return best_identity_names, best_scores, second_best_scores, margins
