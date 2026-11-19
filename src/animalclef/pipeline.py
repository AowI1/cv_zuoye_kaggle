"""End-to-end species pipeline: calibrate -> fuse -> re-rank -> cluster -> refine."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import cluster as cl
from . import postprocess as pp
from .calibration import Calibrator
from .similarity import cosine, fuse


@dataclass
class Config:
    """Per-species pipeline configuration."""
    dataset: str
    weights: dict[str, float] = field(default_factory=lambda: {"miewid": 0.5, "mega": 0.5, "local": 0.0})
    use_orientation_calibration: bool = False

    use_rerank: bool = False
    rr_k1: int = 20
    rr_k2: int = 6
    rr_lambda: float = 0.3

    method: str = "agglo"          # "agglo" | "cw" | "dbscan"
    threshold: float = 0.5         # similarity threshold
    cw_top_k: int = 20
    cw_iterations: int = 20
    cw_k_core: int = 1

    merge_small: int = 2
    merge_threshold: float = 0.5
    split_big: int = 60
    split_threshold: float = 0.65

    anchor: bool = False
    anchor_score: float = 0.6
    anchor_margin: float = 0.1


def fit_calibrators(
    db_embs: dict[str, np.ndarray],
    db_meta: pd.DataFrame,
    cfg: Config,
    max_fit_images: int = 1024,
    seed: int = 42,
) -> dict[str, Calibrator]:
    """Fit one isotonic calibrator per backbone on database pairs."""
    rng = np.random.default_rng(seed)
    n = len(db_meta)
    idx = np.arange(n) if n <= max_fit_images else rng.choice(n, max_fit_images, replace=False)
    ids = np.asarray(db_meta["identity"].astype(str).values, dtype=object)[idx]
    orient = db_meta["orientation"].values[idx] if cfg.use_orientation_calibration else None

    cals = {}
    for name, emb in db_embs.items():
        sim = cosine(emb[idx], emb[idx])
        cal = Calibrator().fit(sim, ids, orientation=orient)
        cals[name] = cal
    return cals


def calibrated_sims(
    qry_embs: dict[str, np.ndarray],
    qry_meta: pd.DataFrame,
    cals: dict[str, Calibrator],
    db_embs: dict[str, np.ndarray] | None = None,
    use_orientation: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-backbone calibrated query-query and query-database similarities."""
    qq, qd = {}, {}
    q_orient = qry_meta["orientation"].values if use_orientation else None
    for name, emb in qry_embs.items():
        qq[name] = cals[name].transform(cosine(emb, emb), q_orient)
        if db_embs is not None and name in db_embs:
            # Orientation conditioning only applies to query-query pairs; for
            # query-database we use the global map (database orientations are
            # covered by the global fit anyway).
            qd[name] = cals[name].transform(cosine(emb, db_embs[name]))
    return qq, qd


def cluster_similarity(sim: np.ndarray, cfg: Config) -> np.ndarray:
    if cfg.method == "agglo":
        return cl.agglomerative(sim, cfg.threshold)
    if cfg.method == "cw":
        return cl.chinese_whispers(
            sim, cfg.threshold, iterations=cfg.cw_iterations,
            top_k=cfg.cw_top_k, k_core_degree=cfg.cw_k_core,
        )
    if cfg.method == "dbscan":
        return cl.dbscan(sim, eps=1.0 - cfg.threshold, min_samples=2)
    raise ValueError(f"unknown method {cfg.method}")


def refine(labels: np.ndarray, sim: np.ndarray, cfg: Config,
           sim_qdb: np.ndarray | None = None,
           db_ids: np.ndarray | None = None) -> np.ndarray:
    labels = pp.split_big(labels, sim, cfg.split_big, cfg.split_threshold)
    labels = pp.merge_small(labels, sim, cfg.merge_small, cfg.merge_threshold)
    if cfg.anchor and sim_qdb is not None and db_ids is not None:
        labels = pp.anchor_merge(
            labels, sim_qdb, db_ids,
            min_score=cfg.anchor_score, min_margin=cfg.anchor_margin,
        )
    return pp.relabel(labels)


def run_species(
    qry_embs: dict[str, np.ndarray],
    qry_meta: pd.DataFrame,
    db_embs: dict[str, np.ndarray] | None,
    db_meta: pd.DataFrame | None,
    cfg: Config,
    cals: dict[str, Calibrator] | None = None,
) -> np.ndarray:
    """Run the full pipeline for one species and return cluster labels."""
    if cals is None:
        if db_embs is None or db_meta is None or len(db_meta) == 0:
            raise ValueError("calibration requires a labelled database")
        cals = fit_calibrators(db_embs, db_meta, cfg)

    qq, qd = calibrated_sims(
        qry_embs, qry_meta, cals,
        db_embs if db_embs else None,
        use_orientation=cfg.use_orientation_calibration,
    )

    weights = {k: v for k, v in cfg.weights.items() if k in qq and v > 0}
    sim = fuse(qq, weights)

    if cfg.use_rerank:
        sim = cl.rerank_similarity(sim, cfg.rr_k1, cfg.rr_k2, cfg.rr_lambda)

    labels = cluster_similarity(sim, cfg)

    db_ids = np.asarray(db_meta["identity"].astype(str).values, dtype=object) if db_meta is not None else None
    sim_qdb = fuse(qd, weights) if qd else None
    labels = refine(labels, sim, cfg, sim_qdb=sim_qdb, db_ids=db_ids)
    return labels
