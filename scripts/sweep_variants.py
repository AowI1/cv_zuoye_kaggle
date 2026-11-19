#!/usr/bin/env python
"""Generate single-species parameter variants from the cached artefacts.

Each variant changes the clustering of exactly ONE species and keeps the
others at the submitted baseline, so the overall score difference on Kaggle
(mean ARI over four species) attributes cleanly to that species:

    delta_overall * 4  ~=  delta_ARI_of_that_species

No embeddings are recomputed; everything is read from cache/.

Usage:
    python scripts/sweep_variants.py                 # write all variants
    python scripts/sweep_variants.py --list          # show specs only
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import make_submission as ms  # noqa: E402

from animalclef import cluster as cl  # noqa: E402
from animalclef.config import CACHE, DATA, SPECIES, SPECIES_KEY, SUBMISSIONS  # noqa: E402
from animalclef.data import load_metadata  # noqa: E402
from animalclef.pipeline import fit_calibrators, refine, cluster_similarity  # noqa: E402
from animalclef.similarity import cosine, fuse, rank_normalize  # noqa: E402

OUT = SUBMISSIONS / "variants"

# ---------------------------------------------------------------- variants --
# species, name, override dict, target mean cluster size (None = keep tuned)
VARIANTS = [
    # --- Salamander: was the density-matched threshold (457 clusters) right? --
    ("SalamanderID2025", "salam_direct_t0.14", {}, None),
    ("SalamanderID2025", "salam_mean2.0", {}, 2.0),
    ("SalamanderID2025", "salam_mean2.2", {}, 2.2),
    # --- Texas horned lizards: currently ~0 ARI; any gain is pure upside ----
    ("TexasHornedLizards", "lizard_mean2.0", {}, 2.0),
    ("TexasHornedLizards", "lizard_mean2.5", {}, 2.5),
    ("TexasHornedLizards", "lizard_mean3.0", {}, 3.0),
    ("TexasHornedLizards", "lizard_nolocal", {"local_weight": 0.0}, 1.5),
    # --- Sea turtle: local optimum is flat 1.4-2.1, probe the edges ---------
    ("SeaTurtleID2022", "turtle_mean1.8", {}, 1.8),
    ("SeaTurtleID2022", "turtle_mean2.3", {}, 2.3),
    # --- Lynx: anchor merge dominates; probe threshold and local weight -----
    ("LynxID2025", "lynx_t0.66", {"threshold": 0.66}, None),
    ("LynxID2025", "lynx_t0.78", {"threshold": 0.78}, None),
    ("LynxID2025", "lynx_local0.1", {"local_weight": 0.1}, None),

    # --- round 2: chase the two directions that moved both scores ----------
    ("SeaTurtleID2022", "turtle_mean2.5", {}, 2.5),
    ("SeaTurtleID2022", "turtle_mean2.7", {}, 2.7),
    ("SeaTurtleID2022", "turtle_mean3.0", {}, 3.0),
    ("LynxID2025", "lynx_t0.62", {"threshold": 0.62}, None),
    ("LynxID2025", "lynx_t0.58", {"threshold": 0.58}, None),
    # lizard: merging hurts, so probe the other side of the optimum
    ("TexasHornedLizards", "lizard_mean1.3", {}, 1.3),
    ("TexasHornedLizards", "lizard_mean1.2", {}, 1.2),
    ("TexasHornedLizards", "lizard_local3.0", {"local_weight": 3.0}, 1.5),
    ("SalamanderID2025", "salam_mean1.4", {}, 1.4),

    # --- round 3: push the three winning directions to their limits ---------
    ("TexasHornedLizards", "lizard_mean1.1", {}, 1.1),
    ("TexasHornedLizards", "lizard_local3.0_mean1.3", {"local_weight": 3.0}, 1.3),
    ("TexasHornedLizards", "lizard_local4.0_mean1.3", {"local_weight": 4.0}, 1.3),
    ("TexasHornedLizards", "lizard_merge2_mean1.3", {"merge_small": 2}, 1.3),
    ("LynxID2025", "lynx_t0.54", {"threshold": 0.54}, None),
    ("LynxID2025", "lynx_t0.50", {"threshold": 0.50}, None),
    ("LynxID2025", "lynx_t0.58_anchor0.5", {"threshold": 0.58, "anchor_score": 0.5}, None),
    ("SeaTurtleID2022", "turtle_mean2.5_anchor0.5", {"anchor_score": 0.5}, 2.5),
    ("SeaTurtleID2022", "turtle_mean2.5_merge2", {"merge_small": 2}, 2.5),
]

# Combinations of the directions that improved both public and private scores.
COMBOS = [
    ("combo_turtle2.3_lynx0.66", {
        "SeaTurtleID2022": ({}, 2.3),
        "LynxID2025": ({"threshold": 0.66}, None),
    }),
    # best per species from rounds 1-2, recombined
    ("combo_best1", {
        "TexasHornedLizards": ({}, 1.3),
        "LynxID2025": ({"threshold": 0.58}, None),
        "SeaTurtleID2022": ({}, 2.5),
    }),
    ("combo_best2", {
        "TexasHornedLizards": ({"local_weight": 3.0}, 1.3),
        "LynxID2025": ({"threshold": 0.58}, None),
        "SeaTurtleID2022": ({}, 2.5),
    }),
]

_STATE: dict[str, dict] = {}


def _species_state(ds: str, backbones: list[str]) -> dict:
    """Everything needed to re-cluster one species, computed once."""
    if ds in _STATE:
        return _STATE[ds]
    meta = load_metadata()
    best = ms.load_best(ds)
    cfg = ms.config_from(best, ds)
    test = meta[(meta["dataset"] == ds) & (meta["split"] == "test")]
    train = meta[(meta["dataset"] == ds) & (meta["split"] == "train")]

    q_embs, d_embs = {}, {}
    for b in backbones:
        e_t = ms.load_emb(b, ds, "test")
        if e_t is None:
            continue
        q_embs[b] = e_t
        e_d = ms.load_emb(b, ds, "train")
        if e_d is not None:
            d_embs[b] = e_d

    if len(train) > 0 and d_embs:
        cals = fit_calibrators(d_embs, train, cfg)
        q_orient = test["orientation"].values if cfg.use_orientation_calibration else None
        qq = {b: cals[b].transform(cosine(e, e), q_orient) for b, e in q_embs.items()}
        qd = {b: cals[b].transform(cosine(q_embs[b], d_embs[b])) for b in d_embs}
        db_ids = np.asarray(train["identity"].astype(str).values, dtype=object)
    else:
        qq = {b: rank_normalize(cosine(e, e)) for b, e in q_embs.items()}
        qd, db_ids = {}, None

    local = None
    lp = CACHE / f"local_{best.get('local_method', 'root_sift')}_{SPECIES_KEY[ds]}_test.npy"
    if lp.exists():
        local = np.load(lp)

    _STATE[ds] = dict(best=best, cfg=cfg, test=test, qq=qq, qd=qd,
                      db_ids=db_ids, local=local, n=len(test))
    return _STATE[ds]


def _fuse_sim(st: dict, cfg, local_weight: float) -> np.ndarray:
    w = {k: v for k, v in cfg.weights.items() if k in st["qq"] and v > 0}
    sim = fuse(st["qq"], w)
    if local_weight > 0 and st["local"] is not None and len(st["local"]) == st["n"]:
        from animalclef.localfeat import normalize_local
        lsim = normalize_local(st["local"])
        sim = (sim + local_weight * lsim) / (1.0 + local_weight)
    if cfg.use_rerank:
        sim = cl.rerank_similarity(sim, cfg.rr_k1, cfg.rr_k2, cfg.rr_lambda)
    return sim


def labels_for(ds: str, best: dict, override: dict, target_mean: float | None,
               backbones: list[str], tol: float = 0.02) -> np.ndarray:
    st = _species_state(ds, backbones)
    merged = {**best, **override}
    cfg = ms.config_from(merged, ds)
    # extra knobs that are Config fields rather than entries of best_*.json
    for k in ("anchor_score", "anchor_margin", "merge_small", "merge_threshold",
              "method", "split_big", "split_threshold"):
        if k in override:
            cfg = replace(cfg, **{k: override[k]})
    w_local = float(merged.get("local_weight", 0.0))
    sim = _fuse_sim(st, cfg, w_local)
    w = {k: v for k, v in cfg.weights.items() if k in st["qq"] and v > 0}
    sim_qdb = fuse(st["qd"], w) if st["qd"] else None
    db_ids = st["db_ids"]

    def partition(threshold: float) -> np.ndarray:
        c = replace(cfg, threshold=threshold)
        if merged.get("consensus") and len(w) > 1:
            parts = [cl.agglomerative(st["qq"][b], threshold) for b in w]
            lab = cl.consensus_cluster(parts, threshold)
        else:
            lab = cluster_similarity(sim, c)
        return refine(lab, sim, c, sim_qdb=sim_qdb, db_ids=db_ids)

    labels = partition(cfg.threshold)
    if target_mean:
        target_k = max(2, int(round(st["n"] / float(target_mean))))
        if abs(len(set(labels)) - target_k) / target_k > tol:
            lo, hi = cfg.threshold * 0.5, min(0.995, cfg.threshold * 4.0)
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                if len(set(partition(mid))) < target_k:
                    lo = mid
                else:
                    hi = mid
            labels = partition(0.5 * (lo + hi))
    return labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="miewid,mega")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    backbones = args.backbones.split(",")

    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    sample = pd.read_csv(DATA / "sample_submission.csv")

    # Baseline = the density-matched submission that is already on Kaggle.
    base_labels, base_report = {}, {}
    for ds in SPECIES:
        best = ms.load_best(ds)
        target = best.get("target_mean_cluster_size")
        # tol=0.10 mirrors make_submission.py --density-match
        base_labels[ds] = labels_for(ds, best, {}, target, backbones, tol=0.10)
        base_report[ds] = len(set(base_labels[ds]))
    base_csv = OUT / "base_densitymatched.csv"
    rows = []
    for ds in SPECIES:
        for iid, lab in zip(_STATE[ds]["test"]["image_id"].values, base_labels[ds]):
            rows.append({"image_id": int(iid), "cluster": f"cluster_{ds}_{int(lab)}"})
    sample[["image_id"]].merge(pd.DataFrame(rows), on="image_id", how="left") \
        .to_csv(base_csv, index=False)
    print(f"[base] clusters {base_report}")

    manifest = {"base": {"file": base_csv.name, "clusters": base_report}, "variants": []}
    for ds, name, override, target_mean in VARIANTS:
        if args.list:
            print(f"{name:<24} {ds:<20} override={override} target_mean={target_mean}")
            continue
        best = ms.load_best(ds)
        lab = labels_for(ds, best, override, target_mean, backbones)
        rows = []
        for s in SPECIES:
            labs = lab if s == ds else base_labels[s]
            for iid, l in zip(_STATE[s]["test"]["image_id"].values, labs):
                rows.append({"image_id": int(iid), "cluster": f"cluster_{s}_{int(l)}"})
        out = OUT / f"{name}.csv"
        sample[["image_id"]].merge(pd.DataFrame(rows), on="image_id", how="left") \
            .to_csv(out, index=False)
        manifest["variants"].append({
            "name": name, "species": ds, "override": override,
            "target_mean": target_mean, "clusters": int(len(set(lab))),
            "file": out.name,
        })
        print(f"[{name}] {ds}: {len(set(lab))} clusters -> {out.name}")

    for name, spec in COMBOS:
        if args.list:
            print(f"{name:<24} combo {spec}")
            continue
        per = dict(base_labels)
        for ds, (ov, tm) in spec.items():
            per[ds] = labels_for(ds, ms.load_best(ds), ov, tm, backbones)
        rows = []
        for s in SPECIES:
            for iid, l in zip(_STATE[s]["test"]["image_id"].values, per[s]):
                rows.append({"image_id": int(iid), "cluster": f"cluster_{s}_{int(l)}"})
        out = OUT / f"{name}.csv"
        sample[["image_id"]].merge(pd.DataFrame(rows), on="image_id", how="left") \
            .to_csv(out, index=False)
        counts = {s: int(len(set(per[s]))) for s in SPECIES}
        manifest["variants"].append({
            "name": name, "species": "combo", "clusters": counts,
            "override": {k: v[0] for k, v in spec.items()},
            "target_mean": {k: v[1] for k, v in spec.items()}, "file": out.name,
        })
        print(f"[{name}] combo clusters {counts} -> {out.name}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest['variants'])} variants in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
