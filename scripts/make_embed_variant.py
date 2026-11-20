#!/usr/bin/env python
"""Swap one species' clustering into the archived best submission.

Used when the *embeddings* of a species change (e.g. cropped extraction):
the other three species are copied verbatim from the archived best file, so
the Kaggle score difference attributes to that species alone.

    python scripts/make_embed_variant.py --species SalamanderID2025 \
        --name salam_crop085 --target-mean 2.0 \
        --weights '{"miewid": 0.5, "mega": 0.5, "local": 1.5}'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import make_submission as ms  # noqa: E402
import sweep_variants as sv  # noqa: E402
from animalclef.config import DATA, SUBMISSIONS  # noqa: E402

ARCHIVE = SUBMISSIONS / "archive" / "submission_private0.51932.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--target-mean", type=float, default=None)
    ap.add_argument("--weights", default="{}")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--date-boost", type=float, default=None)
    ap.add_argument("--local-method", default=None,
                    help="local-feature matrix to fuse (root_sift | lightglue)")
    ap.add_argument("--backbones", default="miewid,mega")
    args = ap.parse_args()

    override = json.loads(args.weights)
    if args.threshold is not None:
        override["threshold"] = args.threshold
    if args.date_boost is not None:
        override["date_boost"] = args.date_boost
    if args.local_method is not None:
        override["local_method"] = args.local_method

    ds = args.species
    labels = sv.labels_for(ds, ms.load_best(ds), override, args.target_mean,
                           args.backbones.split(","))
    n_clusters = len(set(labels))
    test = sv._STATE[ds]["test"]
    new = pd.DataFrame({"image_id": test["image_id"].values.astype(int),
                        "cluster": [f"cluster_{ds}_{int(c)}" for c in labels]})

    base = pd.read_csv(ARCHIVE)
    keep = ~base["image_id"].isin(new["image_id"])
    out = pd.concat([base[keep], new], ignore_index=True)
    out = base[["image_id"]].merge(out, on="image_id", how="left")
    assert out["cluster"].notna().all() and len(out) == len(base)

    path = SUBMISSIONS / "variants" / f"{args.name}.csv"
    out.to_csv(path, index=False)

    manifest_path = SUBMISSIONS / "variants" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["variants"] = [v for v in manifest["variants"]
                            if v["name"] != args.name]
    manifest["variants"].append({
        "name": args.name, "species": ds, "clusters": n_clusters,
        "override": override, "target_mean": args.target_mean,
        "file": path.name,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[{args.name}] {ds}: {n_clusters} clusters -> {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
