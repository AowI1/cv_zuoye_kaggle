#!/usr/bin/env python
"""Quick end-to-end smoke test of models, preprocessing and similarity."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from animalclef.config import get_device
from animalclef.data import load_metadata
from animalclef.embed import get_backbone
from animalclef.similarity import cosine


def main() -> int:
    device = get_device()
    print("device:", device)
    meta = load_metadata()
    # 12 lynx images from two identities
    train = meta[(meta["dataset"] == "LynxID2025") & (meta["split"] == "train")]
    ids = train["identity"].unique()[:2]
    rows = train[train["identity"].isin(ids)].groupby("identity").head(4)
    print(f"test rows: {len(rows)}, identities: {rows['identity'].nunique()}")

    for name in ["miewid", "mega"]:
        t = time.time()
        model = get_backbone(name, device=device, use_tta=False)
        print(f"[{name}] loaded in {time.time()-t:.1f}s")
        t = time.time()
        emb = model.embed(rows, batch_size=4, n_threads=4, verbose=True)
        dt = time.time() - t
        print(f"[{name}] embeddings {emb.shape} in {dt:.1f}s "
              f"({len(rows)/dt:.1f} img/s), norms={np.linalg.norm(emb, axis=1)[:3]}")
        sim = cosine(emb, emb)
        ids_arr = np.asarray(rows["identity"].values); same = ids_arr[:, None] == ids_arr[None, :]
        np.fill_diagonal(same, False)
        print(f"[{name}] mean same-id sim={sim[same].mean():.4f} "
              f"diff-id sim={sim[~same].mean():.4f}")
        assert emb.shape[1] in (2152, 1536), emb.shape
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
