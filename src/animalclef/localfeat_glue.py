"""LightGlue local-feature similarity (SIFT keypoints, learned matching).

Same interface and scoring convention as ``localfeat.local_similarity`` so the
two channels can be swapped: candidate pairs come from the global similarity,
each pair is matched, geometrically verified, and scored as
``inliers / min(#keypoints)``.

LightGlue replaces the brute-force + ratio-test + RANSAC matcher.  The
descriptors are plain L2-normalised SIFT (the pretrained LightGlue weights
were trained on those, not on RootSIFT).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from .localfeat import candidate_pairs


def _extract_sift(args):
    path, max_side, nfeatures = args
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return (1, 1), np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    h, w = gray.shape
    scale = max_side / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))),
                          interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kps, desc = sift.detectAndCompute(gray, None)
    if kps is None or desc is None or len(kps) < 4:
        return gray.shape, np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    pts = np.array([k.pt for k in kps], dtype=np.float32)
    desc = desc.astype(np.float32)
    desc /= (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-7)
    return gray.shape, pts, desc


def _lafs(pts: np.ndarray, device: str) -> torch.Tensor:
    n = len(pts)
    kp = torch.from_numpy(pts).to(device).float()
    laf = torch.zeros(1, n, 2, 3, device=device)
    laf[0, :, 0, 0] = 1.0
    laf[0, :, 1, 1] = 1.0
    laf[0, :, 0, 2] = kp[:, 0]
    laf[0, :, 1, 2] = kp[:, 1]
    return laf


class GlueLocalSimilarity:
    def __init__(self, device: str = "cpu", max_side: int = 800,
                 nfeatures: int = 2000, score_thr: float = 0.5,
                 n_threads: int = 8, verbose: bool = True):
        from kornia.feature import LightGlueMatcher
        self.device = device
        self.max_side = max_side
        self.nfeatures = nfeatures
        self.score_thr = score_thr
        self.n_threads = n_threads
        self.verbose = verbose
        self.matcher = LightGlueMatcher(
            "sift", {"depth_confidence": -1, "width_confidence": -1}
        ).eval().to(device)

    def _descriptors(self, paths: list[str]):
        if self.verbose:
            print(f"  [glue] describing {len(paths)} images...", flush=True)
        with ThreadPoolExecutor(max_workers=self.n_threads) as pool:
            return list(pool.map(_extract_sift,
                                 [(p, self.max_side, self.nfeatures) for p in paths]))

    def _score_pair(self, a, b) -> float:
        (hw_a, pts_a, desc_a), (hw_b, pts_b, desc_b) = a, b
        if len(pts_a) < 4 or len(pts_b) < 4:
            return 0.0
        d1 = torch.from_numpy(desc_a)[None].to(self.device)
        d2 = torch.from_numpy(desc_b)[None].to(self.device)
        with torch.inference_mode():
            scores, idx = self.matcher(
                d1, d2, _lafs(pts_a, self.device), _lafs(pts_b, self.device),
                (int(hw_a[0]), int(hw_a[1])), (int(hw_b[0]), int(hw_b[1])),
            )
        if len(idx) < 4:
            return 0.0
        keep = scores.reshape(-1) >= self.score_thr
        idx = idx[keep]
        if len(idx) < 4:
            return 0.0
        src = pts_a[idx[:, 0].cpu().numpy()].reshape(-1, 1, 2)
        dst = pts_b[idx[:, 1].cpu().numpy()].reshape(-1, 1, 2)
        _, mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if mask is None:
            return 0.0
        inliers = float(mask.sum())
        return min(1.0, inliers / max(1.0, min(len(pts_a), len(pts_b))))

    def __call__(self, df, sim_global: np.ndarray, top_k: int = 10,
                 min_sim: float | None = None,
                 gate_percentile: float = 95.0) -> np.ndarray:
        n = len(df)
        descs = self._descriptors(df["abs_path"].tolist())
        pairs = candidate_pairs(sim_global, top_k=top_k, min_sim=min_sim,
                                gate_percentile=gate_percentile)
        if self.verbose:
            print(f"  [glue] matching {len(pairs)} candidate pairs on "
                  f"{self.device}...", flush=True)
        out = np.zeros((n, n), dtype=np.float32)
        for done, (i, j) in enumerate(pairs, 1):
            s = self._score_pair(descs[i], descs[j])
            out[i, j] = out[j, i] = s
            if self.verbose and done % 500 == 0:
                print(f"    matched {done}/{len(pairs)}", flush=True)
        return out
