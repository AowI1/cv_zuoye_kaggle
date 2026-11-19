"""Classical local-feature similarity (SIFT / RootSIFT / ORB + RANSAC).

Deep global embeddings are excellent for turtles but weak for fine-grained
patterns (salamander spots, lizard speckles).  Local keypoint matching adds a
complementary signal: two different individuals rarely share enough
geometrically consistent keypoint correspondences.

To stay affordable we only match *candidate pairs* proposed by the fused global
similarity (top-k neighbours per image plus pairs above a low threshold).

Scores are normalised to [0, 1] and can be fed through the same isotonic
calibration used for the neural backbones.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


def _extract(args):
    path, method, max_side, root_sift = args
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    h, w = gray.shape
    scale = max_side / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))),
                          interpolation=cv2.INTER_AREA)
    if method in ("sift", "root_sift"):
        detector = cv2.SIFT_create(nfeatures=1200)
    elif method == "orb":
        detector = cv2.ORB_create(nfeatures=1500)
    else:
        raise ValueError(f"unknown local method {method}")
    kps, desc = detector.detectAndCompute(gray, None)
    if desc is None or len(kps) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    pts = np.array([kp.pt for kp in kps], dtype=np.float32)
    if method != "orb":
        desc = desc.astype(np.float32)
    if method == "root_sift":
        desc = desc / (desc.sum(axis=1, keepdims=True) + 1e-7)
        desc = np.sqrt(desc)
    return pts, desc


def _match(desc_a: np.ndarray, desc_b: np.ndarray, pts_a: np.ndarray,
           pts_b: np.ndarray, ratio: float) -> float:
    if len(desc_a) < 4 or len(desc_b) < 4:
        return 0.0
    norm = cv2.NORM_HAMMING if desc_a.dtype == np.uint8 else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm)
    try:
        matches = matcher.knnMatch(desc_a, desc_b, k=2)
    except cv2.error:
        return 0.0
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < 4:
        return 0.0
    src = np.float32([pts_a[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([pts_b[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if mask is None:
        return 0.0
    inliers = float(mask.sum())
    # Normalise by the smaller keypoint set so dense images are not favoured.
    denom = max(1.0, min(len(pts_a), len(pts_b)))
    return min(1.0, inliers / denom)


def candidate_pairs(sim: np.ndarray, top_k: int = 10,
                    min_sim: float | None = None,
                    gate_percentile: float = 95.0) -> list[tuple[int, int]]:
    """Symmetric candidate pairs from a global similarity matrix.

    Candidates are the union of each image's top-k neighbours and all pairs
    above a similarity gate.  Raw cosine baselines differ wildly across
    species (lizard pairs are all similar), so the gate defaults to a
    within-matrix percentile rather than an absolute value.
    """
    n = sim.shape[0]
    k = min(top_k, n - 1)
    order = np.argpartition(-sim, k, axis=1)[:, :k]
    pairs = set()
    for i in range(n):
        for j in order[i]:
            if i == j:
                continue
            a, b = (i, int(j)) if i < j else (int(j), i)
            pairs.add((a, b))
    if min_sim is None:
        iu = np.triu_indices(n, 1)
        min_sim = float(np.percentile(sim[iu], gate_percentile))
    ii, jj = np.where(sim >= min_sim)
    for a, b in zip(ii, jj):
        if a < b:
            pairs.add((int(a), int(b)))
    return sorted(pairs)


def local_similarity(
    df,
    sim_global: np.ndarray,
    method: str = "root_sift",
    top_k: int = 10,
    min_sim: float | None = None,
    gate_percentile: float = 95.0,
    max_side: int = 800,
    ratio: float = 0.75,
    n_threads: int = 8,
    verbose: bool = True,
) -> np.ndarray:
    """Sparse local-feature similarity matrix aligned with ``df`` row order."""
    n = len(df)
    paths = df["abs_path"].tolist()
    if verbose:
        print(f"  [local:{method}] describing {n} images...", flush=True)
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        descs = list(pool.map(
            _extract, [(p, method, max_side, method == "root_sift") for p in paths]
        ))
    pairs = candidate_pairs(sim_global, top_k=top_k, min_sim=min_sim,
                            gate_percentile=gate_percentile)
    if verbose:
        print(f"  [local:{method}] matching {len(pairs)} candidate pairs...", flush=True)

    out = np.zeros((n, n), dtype=np.float32)

    def work(pair):
        i, j = pair
        pts_a, desc_a = descs[i]
        pts_b, desc_b = descs[j]
        return i, j, _match(desc_a, desc_b, pts_a, pts_b, ratio)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        done = 0
        for i, j, s in pool.map(work, pairs):
            out[i, j] = out[j, i] = s
            done += 1
            if verbose and done % 5000 == 0:
                print(f"    matched {done}/{len(pairs)}", flush=True)
    return out


def normalize_local(lsim: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Scale a sparse local-similarity matrix so strong matches approach 1.

    Raw RANSAC-inlier ratios are very small (0.005-0.08); without rescaling
    they would be negligible in a weighted fusion with calibrated
    probabilities.
    """
    nz = lsim[lsim > 0]
    if nz.size == 0:
        return lsim
    scale = float(np.percentile(nz, percentile))
    return np.clip(lsim / max(scale, 1e-9), 0.0, 1.0).astype(np.float32)
