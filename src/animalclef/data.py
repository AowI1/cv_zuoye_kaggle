"""Dataset loading with threaded prefetch.

The official metadata.csv has one row per image with columns
``image_id, identity, path, date, orientation, species, split, dataset``.
Test rows have a missing ``identity`` (that is what we predict).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from .config import DATA
from .preprocess import preprocess


def load_metadata(root: Path = DATA) -> pd.DataFrame:
    meta = pd.read_csv(root / "metadata.csv")
    meta["abs_path"] = meta["path"].apply(lambda p: str(root / p))
    missing = ~meta["abs_path"].map(lambda p: Path(p).exists())
    if missing.any():
        raise FileNotFoundError(
            f"{missing.sum()} metadata paths do not exist, e.g. "
            f"{meta.loc[missing, 'abs_path'].iloc[0]}"
        )
    meta["orientation"] = meta["orientation"].fillna("unknown").astype(str)
    return meta


def load_image(path: str, dataset: str, do_preprocess: bool = True,
               size: int | None = None, letterbox: bool = False,
               pre_max_side: int | None = 512) -> np.ndarray:
    """Load one RGB image, downscale, apply species preprocessing, resize.

    Images arrive at wildly different resolutions (600-2600 px); downscaling to
    ``pre_max_side`` before the OpenCV recipes keeps CPU preprocessing cheap
    while retaining enough detail for 384-512 px model inputs.
    """
    with Image.open(path) as im:
        img = np.array(im.convert("RGB"), dtype=np.uint8)
    if pre_max_side is not None:
        h, w = img.shape[:2]
        if max(h, w) > pre_max_side:
            import cv2
            scale = pre_max_side / max(h, w)
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)
    img = preprocess(img, dataset, enabled=do_preprocess)
    if size is not None:
        img = resize(img, size, letterbox=letterbox)
    return img


def resize(img: np.ndarray, size: int, letterbox: bool = False) -> np.ndarray:
    """Resize to ``size``x``size`` using cv2 (fast path, uint8 RGB)."""
    import cv2
    if letterbox:
        h, w = img.shape[:2]
        scale = size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
        out = np.full((size, size, 3), 0, dtype=np.uint8)
        top, left = (size - nh) // 2, (size - nw) // 2
        out[top:top + nh, left:left + nw] = img
        return out
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)


def iter_batches(df: pd.DataFrame, indices: Sequence[int] | None,
                 batch_size: int, n_threads: int, do_preprocess: bool,
                 size: int, letterbox: bool = False) -> Iterator[tuple[list[int], np.ndarray]]:
    """Yield (indices, uint8 NHWC batch) with threaded image loading.

    Loading and species preprocessing are CPU-bound and release the GIL inside
    OpenCV/PIL, so a small thread pool keeps the accelerator fed.
    """
    if indices is None:
        indices = np.arange(len(df))
    indices = list(indices)
    paths = df["abs_path"].values
    datasets = df["dataset"].values

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            futs = [
                pool.submit(load_image, paths[i], datasets[i], do_preprocess,
                            size, letterbox)
                for i in chunk
            ]
            arrays = [f.result() for f in futs]
            yield chunk, np.stack(arrays)
