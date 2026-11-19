import numpy as np
import pandas as pd
from pathlib import Path


def load_full_database_matrix(npy_path, dataset_name):
    """
    Load one full train/train similarity matrix bundle saved as:
        {dataset_name: {"sim": ..., "indices": ...}}
    """
    obj = np.load(npy_path, allow_pickle=True).item()
    bundle = obj[dataset_name]
    sim = bundle["sim"].astype(np.float32)
    indices = np.asarray(bundle["indices"])
    return sim, indices


def align_metadata_to_matrix(metadata_csv, dataset_name, matrix_indices):
    """
    Align train metadata rows to the matrix order.

    IMPORTANT:
    Here, matrix_indices are assumed to be row positions inside the
    dataset-specific TRAIN subset, not global image_id values.
    """
    meta = pd.read_csv(metadata_csv)

    meta_ds = meta[
        (meta["dataset"] == dataset_name) &
        (meta["split"] == "train")
    ].copy().reset_index(drop=True)

    matrix_indices = np.asarray(matrix_indices)

    assert matrix_indices.min() >= 0, "Matrix indices contain negative values."
    assert matrix_indices.max() < len(meta_ds), (
        f"Matrix index max {matrix_indices.max()} exceeds train subset size {len(meta_ds)}"
    )

    meta_aligned = meta_ds.iloc[matrix_indices].reset_index(drop=True)

    assert len(meta_aligned) == len(matrix_indices), (
        f"Aligned metadata rows ({len(meta_aligned)}) do not match matrix size ({len(matrix_indices)})"
    )

    return meta_aligned


def split_identity_images(identity_positions, rng, db_frac=0.7):
    """
    Split positions for one KNOWN identity into:
      - database images
      - known-query images

    Guarantees at least 1 db and 1 query image whenever possible.
    """
    positions = np.array(identity_positions, dtype=int)
    rng.shuffle(positions)

    n = len(positions)

    if n == 1:
        # Cannot be both db and query; keep single-image identity out of KNOWN pool ideally
        return positions, np.array([], dtype=int)

    n_db = max(1, int(round(db_frac * n)))
    n_db = min(n_db, n - 1)  # leave at least 1 for query

    db_pos = positions[:n_db]
    query_pos = positions[n_db:]

    return db_pos, query_pos


def build_open_set_split_from_full_database_matrix(
    npy_path,
    metadata_csv,
    dataset_name,
    seed=42,
    known_identity_frac=0.7,
    db_frac_within_known=0.7,
    min_images_per_known_identity=2,
    save_dir=None,
):
    """
    Build an open-set validation split from a FULL database/database similarity matrix.

    Pipeline:
      1. Split identities into KNOWN and UNKNOWN
      2. KNOWN identities:
           some images -> database
           remaining images -> known query
      3. UNKNOWN identities:
           all images -> unknown query
      4. Slice full sim matrix into:
           sim_query_db
           sim_query_query

    Returns a dictionary with everything needed for validation / Optuna.
    """
    rng = np.random.default_rng(seed)

    # Load full matrix and aligned metadata
    sim_full, matrix_indices = load_full_database_matrix(npy_path, dataset_name)
    meta_aligned = align_metadata_to_matrix(metadata_csv, dataset_name, matrix_indices)

    # Group positions by identity
    identity_to_positions = (
        meta_aligned.reset_index()
        .groupby("identity")["index"]
        .apply(list)
        .to_dict()
    )

    all_identities = np.array(list(identity_to_positions.keys()), dtype=object)

    # Eligible known identities must have enough images to split into db/query
    eligible_known = np.array([
        ident for ident, pos_list in identity_to_positions.items()
        if len(pos_list) >= min_images_per_known_identity
    ], dtype=object)

    ineligible_known = np.array([
        ident for ident, pos_list in identity_to_positions.items()
        if len(pos_list) < min_images_per_known_identity
    ], dtype=object)

    # Choose known identities only from eligible pool
    n_known = max(1, int(round(len(eligible_known) * known_identity_frac)))
    known_identities = rng.choice(eligible_known, size=n_known, replace=False)

    # Unknown identities are:
    #   - all remaining eligible identities not chosen as known
    #   - plus the ineligible identities
    known_identity_set = set(known_identities.tolist())
    unknown_identities = np.array([
        ident for ident in all_identities
        if ident not in known_identity_set
    ], dtype=object)

    db_positions = []
    query_known_positions = []
    query_unknown_positions = []

    # Split known identities into db / known-query
    for ident in known_identities:
        pos_list = identity_to_positions[ident]
        db_pos, qk_pos = split_identity_images(
            pos_list,
            rng=rng,
            db_frac=db_frac_within_known
        )
        db_positions.extend(db_pos.tolist())
        query_known_positions.extend(qk_pos.tolist())

    # Unknown identities: all images go to query_unknown
    for ident in unknown_identities:
        pos_list = identity_to_positions[ident]
        query_unknown_positions.extend(pos_list)

    # Final ordered query set = known-query + unknown-query
    db_positions = np.array(sorted(db_positions), dtype=int)
    query_known_positions = np.array(sorted(query_known_positions), dtype=int)
    query_unknown_positions = np.array(sorted(query_unknown_positions), dtype=int)
    query_positions = np.concatenate([query_known_positions, query_unknown_positions])

    # Slice matrices
    sim_query_db = sim_full[np.ix_(query_positions, db_positions)]
    sim_query_query = sim_full[np.ix_(query_positions, query_positions)]

    # Labels and masks
    y_true_query = meta_aligned.iloc[query_positions]["identity"].values
    y_true_db = meta_aligned.iloc[db_positions]["identity"].values

    known_mask_query = np.isin(y_true_query, known_identities)
    unknown_mask_query = ~known_mask_query

    result = {
        "dataset_name": dataset_name,
        "sim_query_db": sim_query_db.astype(np.float32),
        "sim_query_query": sim_query_query.astype(np.float32),
        "y_true_query": y_true_query,
        "y_true_db": y_true_db,
        "query_image_ids": meta_aligned.iloc[query_positions]["image_id"].values,
        "db_image_ids": meta_aligned.iloc[db_positions]["image_id"].values,
        "known_mask_query": known_mask_query,
        "unknown_mask_query": unknown_mask_query,
        "known_identities": np.array(known_identities, dtype=object),
        "unknown_identities": np.array(unknown_identities, dtype=object),
        "meta_query": meta_aligned.iloc[query_positions].reset_index(drop=True),
        "meta_db": meta_aligned.iloc[db_positions].reset_index(drop=True),
        "db_positions": db_positions,
        "query_positions": query_positions,
    }

    # Optional saving
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        prefix = save_dir / dataset_name

        np.save(str(prefix) + "_sim_query_db.npy", result["sim_query_db"])
        np.save(str(prefix) + "_sim_query_query.npy", result["sim_query_query"])
        np.save(str(prefix) + "_y_true_query.npy", result["y_true_query"])
        np.save(str(prefix) + "_y_true_db.npy", result["y_true_db"])
        np.save(str(prefix) + "_known_mask_query.npy", result["known_mask_query"])
        np.save(str(prefix) + "_unknown_mask_query.npy", result["unknown_mask_query"])
        result["meta_query"].to_csv(str(prefix) + "_meta_query.csv", index=False)
        result["meta_db"].to_csv(str(prefix) + "_meta_db.csv", index=False)

    # Summary prints
    print(f"\n=== {dataset_name} open-set split summary ===")
    print("full matrix shape         :", sim_full.shape)
    print("num total train identities:", len(all_identities))
    print("num known identities      :", len(known_identities))
    print("num unknown identities    :", len(unknown_identities))
    print("database images           :", len(db_positions))
    print("query images              :", len(query_positions))
    print("  known-query images      :", len(query_known_positions))
    print("  unknown-query images    :", len(query_unknown_positions))
    print("sim_query_db shape        :", result["sim_query_db"].shape)
    print("sim_query_query shape     :", result["sim_query_query"].shape)

    return result