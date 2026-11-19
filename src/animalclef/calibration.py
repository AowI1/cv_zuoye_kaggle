"""Similarity calibration.

Raw cosine scores from different backbones live on different scales, so they
cannot be averaged directly. We fit a monotone isotonic mapping from cosine to
``P(same individual)`` using identity labels of the database split.

Extension over the reference solutions: for lynx we additionally condition the
map on the *orientation pair* of the two images (left/right/top/front/back).
Camera-trap views of different flanks are much harder to match, so a global
calibration systematically over-scales left-right pairs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression


def _pair_labels(ids) -> np.ndarray:
    ids = np.asarray(ids, dtype=object)
    same = ids[:, None] == ids[None, :]
    np.fill_diagonal(same, False)
    return same


def _sample_pairs(sim: np.ndarray, same: np.ndarray, max_per_class: int,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = sim.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    vals = sim[iu, ju]
    labs = same[iu, ju]
    pos = np.where(labs)[0]
    neg = np.where(~labs)[0]
    if len(pos) > max_per_class:
        pos = rng.choice(pos, max_per_class, replace=False)
    if len(neg) > max_per_class:
        neg = rng.choice(neg, max_per_class, replace=False)
    sel = np.concatenate([pos, neg])
    return vals[sel], labs[sel].astype(np.float32)


def _orientation_bucket(o: str) -> str:
    o = (o or "unknown").lower()
    if o in ("left", "right", "front", "back"):
        return o
    if o.startswith("top") or o == "down" or o == "bottom":
        return "top"
    return "other"


@dataclass
class Calibrator:
    """Isotonic calibration, optionally conditioned on orientation pairs."""
    global_map: IsotonicRegression | None = None
    orientation_maps: dict[tuple[str, str], IsotonicRegression] | None = None
    min_pairs_per_bucket: int = 200

    def fit(self, sim: np.ndarray, ids: np.ndarray,
            orientation: np.ndarray | None = None,
            max_pairs: int = 400_000, seed: int = 42) -> "Calibrator":
        rng = np.random.default_rng(seed)
        same = _pair_labels(ids)
        x, y = _sample_pairs(sim, same, max_pairs // 2, rng)
        self.global_map = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        ).fit(x, y)

        if orientation is not None:
            buckets = np.array([_orientation_bucket(o) for o in orientation])
            self.orientation_maps = {}
            for a in sorted(set(buckets)):
                for b in sorted(set(buckets)):
                    if a > b:
                        continue
                    mask = ((buckets[:, None] == a) & (buckets[None, :] == b)) | (
                        (buckets[:, None] == b) & (buckets[None, :] == a)
                    )
                    np.fill_diagonal(mask, False)
                    xb, yb = _sample_pairs(sim, same & mask, max_pairs // 2, rng)
                    if len(xb) >= self.min_pairs_per_bucket and len(np.unique(yb)) > 1:
                        self.orientation_maps[(a, b)] = IsotonicRegression(
                            y_min=0.0, y_max=1.0, out_of_bounds="clip"
                        ).fit(xb, yb)
        return self

    def transform(self, sim: np.ndarray,
                  orientation: np.ndarray | None = None) -> np.ndarray:
        out = self.global_map.predict(sim.ravel()).reshape(sim.shape)
        if orientation is not None and self.orientation_maps:
            buckets = np.array([_orientation_bucket(o) for o in orientation])
            out = out.copy()
            for (a, b), iso in self.orientation_maps.items():
                mask = ((buckets[:, None] == a) & (buckets[None, :] == b)) | (
                    (buckets[:, None] == b) & (buckets[None, :] == a)
                )
                if mask.any():
                    out[mask] = iso.predict(sim[mask])
        return out.astype(np.float32)
