#!/usr/bin/env python
"""Rebuild the best leaderboard submission from the sweep configuration.

The winning combination is stored in configs/best_combo.json; this script
regenerates submissions/submission_best.csv from cache/ and verifies it
against the file that was actually scored on Kaggle.

    python scripts/make_best_submission.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import make_submission as ms  # noqa: E402
import sweep_variants as sv  # noqa: E402
from animalclef.config import DATA, SPECIES, SUBMISSIONS  # noqa: E402


def main() -> int:
    spec = json.loads((ROOT / "configs" / "best_combo.json").read_text())
    backbones = spec.get("backbones", ["miewid", "mega"])

    labels = {}
    for ds in SPECIES:
        entry = spec["species"][ds]
        best = ms.load_best(ds)
        override = entry.get("override", {})
        target = entry.get("target_mean_cluster_size")
        labels[ds] = sv.labels_for(ds, best, override, target, backbones)
        print(f"[{ds}] {len(set(labels[ds]))} clusters")

    rows = []
    for ds in SPECIES:
        for iid, lab in zip(sv._STATE[ds]["test"]["image_id"].values, labels[ds]):
            rows.append({"image_id": int(iid), "cluster": f"cluster_{ds}_{int(lab)}"})
    sample = pd.read_csv(DATA / "sample_submission.csv")
    out = SUBMISSIONS / "submission_best.csv"
    sample[["image_id"]].merge(pd.DataFrame(rows), on="image_id", how="left") \
        .to_csv(out, index=False)
    print(f"wrote {out} ({len(sample)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
