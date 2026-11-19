
import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# AnimalCLEF / MiewID-style validation split builder
# ------------------------------------------------------------
# - Uses the metadata CSV you uploaded
# - Creates:
#       1) fine_tune_train.csv
#       2) val_known.csv     -> seen identities, held-out images
#       3) val_unknown.csv   -> identities fully removed from train
#       4) val_all.csv       -> union of known + unknown validation
#
# Design goals:
# - species-wise split
# - open-set validation
# - preserve as much training data as possible
# - roughly mimic AnimalCLEF 2025 query composition for the 3 species
# ------------------------------------------------------------

CHALLENGE_LIKE_TARGETS = {
    # Official AnimalCLEF 2025 query identities:
    # Lynx   : known 19, unknown 22
    # Sal    : known 59, unknown 49
    # Turtle : known 50, unknown 50
    #
    # For known validation:
    # - salamanders are sparse, so hold out 1 image / identity
    # - lynx and turtles are denser, so hold out 2 images / identity
    "LynxID2025": {
        "known_ids": 19,
        "unknown_ids": 22,
        "known_val_imgs_per_id": 2,
        "min_train_after_known": 2,
        "min_unknown_imgs": 2,
    },
    "SalamanderID2025": {
        "known_ids": 59,
        "unknown_ids": 49,
        "known_val_imgs_per_id": 1,
        "min_train_after_known": 1,
        "min_unknown_imgs": 2,
    },
    "SeaTurtleID2022": {
        "known_ids": 50,
        "unknown_ids": 50,
        "known_val_imgs_per_id": 2,
        "min_train_after_known": 2,
        "min_unknown_imgs": 2,
    },
}

def load_metadata(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"dataset", "image_id", "path", "proc_path", "split", "identity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def train_only(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["split"].eq("train") & df["identity"].notna()].copy()
    out["identity"] = out["identity"].astype(str)
    return out

def summarize_species(df_train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for species, sdf in df_train.groupby("dataset", sort=False):
        counts = sdf["identity"].value_counts()
        rows.append({
            "species": species,
            "train_images": int(len(sdf)),
            "train_identities": int(counts.size),
            "min_imgs_per_id": int(counts.min()),
            "median_imgs_per_id": float(counts.median()),
            "mean_imgs_per_id": float(counts.mean()),
            "max_imgs_per_id": int(counts.max()),
            "n_ids_with_1_img": int((counts == 1).sum()),
            "n_ids_with_2_imgs": int((counts == 2).sum()),
            "n_ids_with_3_imgs": int((counts == 3).sum()),
            "n_ids_with_ge_4_imgs": int((counts >= 4).sum()),
        })
    return pd.DataFrame(rows)

def heuristic_target_from_distribution(counts: pd.Series):
    """
    Fallback if a species is not in CHALLENGE_LIKE_TARGETS.
    This keeps the split conservative for sparse species.
    """
    n_ids = counts.size
    med = counts.median()

    if med <= 2:
        # very sparse
        known_ids = max(10, round(0.10 * n_ids))
        unknown_ids = max(8, round(0.08 * n_ids))
        known_val_imgs_per_id = 1
        min_train_after_known = 1
        min_unknown_imgs = 2
    else:
        # denser
        known_ids = max(15, round(0.12 * n_ids))
        unknown_ids = max(10, round(0.10 * n_ids))
        known_val_imgs_per_id = 2
        min_train_after_known = 2
        min_unknown_imgs = 2

    return {
        "known_ids": int(known_ids),
        "unknown_ids": int(unknown_ids),
        "known_val_imgs_per_id": int(known_val_imgs_per_id),
        "min_train_after_known": int(min_train_after_known),
        "min_unknown_imgs": int(min_unknown_imgs),
    }

def choose_unknown_ids(counts: pd.Series, unknown_ids: int, min_unknown_imgs: int) -> list[str]:
    """
    Choose unknown IDs from the smallest eligible identities first.
    Reason: preserve more training images while still making unknown
    validation non-trivial (at least min_unknown_imgs per unknown ID).
    """
    eligible = counts[counts >= min_unknown_imgs].sort_values(kind="stable")
    chosen = eligible.iloc[:min(unknown_ids, len(eligible))].index.tolist()
    return chosen

def choose_known_ids(
    counts: pd.Series,
    excluded_ids: list[str],
    known_ids: int,
    known_val_imgs_per_id: int,
    min_train_after_known: int,
) -> list[str]:
    """
    Choose known IDs from the remaining identities.
    We prefer larger identities so we can hold out images without hurting train.
    """
    remaining = counts[~counts.index.isin(excluded_ids)]
    min_total_needed = known_val_imgs_per_id + min_train_after_known
    eligible = remaining[remaining >= min_total_needed].sort_values(ascending=False, kind="stable")
    chosen = eligible.iloc[:min(known_ids, len(eligible))].index.tolist()
    return chosen

def build_species_split(sdf: pd.DataFrame, cfg: dict, seed: int = 42):
    rng = np.random.default_rng(seed)

    counts = sdf["identity"].value_counts()

    # ---- Unknown validation identities: fully removed from train
    unknown_ids = choose_unknown_ids(
        counts=counts,
        unknown_ids=cfg["unknown_ids"],
        min_unknown_imgs=cfg["min_unknown_imgs"],
    )

    val_unknown = sdf[sdf["identity"].isin(unknown_ids)].copy()
    val_unknown["val_type"] = "unknown"

    remaining = sdf[~sdf["identity"].isin(unknown_ids)].copy()

    # ---- Known validation identities: hold out a few images, keep identity in train
    remaining_counts = remaining["identity"].value_counts()
    known_ids = choose_known_ids(
        counts=remaining_counts,
        excluded_ids=[],
        known_ids=cfg["known_ids"],
        known_val_imgs_per_id=cfg["known_val_imgs_per_id"],
        min_train_after_known=cfg["min_train_after_known"],
    )

    val_known_parts = []
    train_parts = []

    for identity, g in remaining.groupby("identity", sort=False):
        g = g.sample(frac=1.0, random_state=seed)  # stable shuffle

        if identity in known_ids:
            k = min(cfg["known_val_imgs_per_id"], len(g))
            g_val = g.iloc[:k].copy()
            g_train = g.iloc[k:].copy()
            g_val["val_type"] = "known"
            val_known_parts.append(g_val)
            train_parts.append(g_train)
        else:
            train_parts.append(g)

    val_known = pd.concat(val_known_parts, ignore_index=True) if val_known_parts else remaining.iloc[0:0].copy()
    train_ft = pd.concat(train_parts, ignore_index=True) if train_parts else remaining.iloc[0:0].copy()

    # Optional cleanup:
    # remove identities that became empty in train (should not happen with our constraints)
    train_ft = train_ft[train_ft["identity"].notna()].copy()

    return train_ft, val_known, val_unknown

def build_all_splits(df: pd.DataFrame, seed: int = 42):
    df_train = train_only(df)

    all_train = []
    all_known = []
    all_unknown = []
    recommended_rows = []

    for species, sdf in df_train.groupby("dataset", sort=False):
        counts = sdf["identity"].value_counts()

        cfg = CHALLENGE_LIKE_TARGETS.get(species, heuristic_target_from_distribution(counts))

        train_ft, val_known, val_unknown = build_species_split(sdf, cfg, seed=seed)

        all_train.append(train_ft)
        all_known.append(val_known)
        all_unknown.append(val_unknown)

        recommended_rows.append({
            "species": species,
            "train_images": int(len(sdf)),
            "train_identities": int(counts.size),
            "recommended_known_val_ids": int(val_known["identity"].nunique()),
            "recommended_known_val_images": int(len(val_known)),
            "recommended_unknown_val_ids": int(val_unknown["identity"].nunique()),
            "recommended_unknown_val_images": int(len(val_unknown)),
            "train_images_left_after_split": int(len(train_ft)),
            "train_ids_left_after_split": int(train_ft["identity"].nunique()),
        })

    train_ft = pd.concat(all_train, ignore_index=True)
    val_known = pd.concat(all_known, ignore_index=True)
    val_unknown = pd.concat(all_unknown, ignore_index=True)
    val_all = pd.concat([val_known, val_unknown], ignore_index=True)

    rec_df = pd.DataFrame(recommended_rows)
    return train_ft, val_known, val_unknown, val_all, rec_df

def main(csv_path: str, out_dir: str = "."):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata(csv_path)
    train_ft, val_known, val_unknown, val_all, rec_df = build_all_splits(df, seed=42)
    summary_df = summarize_species(train_only(df))

    train_ft.to_csv(out_dir / "fine_tune_train.csv", index=False)
    val_known.to_csv(out_dir / "val_known.csv", index=False)
    val_unknown.to_csv(out_dir / "val_unknown.csv", index=False)
    val_all.to_csv(out_dir / "val_all.csv", index=False)
    rec_df.to_csv(out_dir / "recommended_val_sizes.csv", index=False)
    summary_df.to_csv(out_dir / "species_train_summary.csv", index=False)

    print("\n=== Species summary ===")
    print(summary_df.to_string(index=False))

    print("\n=== Recommended validation sizes ===")
    print(rec_df.to_string(index=False))

    print("\nSaved files:")
    for name in [
        "fine_tune_train.csv",
        "val_known.csv",
        "val_unknown.csv",
        "val_all.csv",
        "recommended_val_sizes.csv",
        "species_train_summary.csv",
    ]:
        print(" -", out_dir / name)

if __name__ == "__main__":
    csv_path = r"metadata.csv"
    out_dir = r"miewid_split_outputs"
    main(csv_path, out_dir)
