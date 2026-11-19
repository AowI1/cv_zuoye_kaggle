#!/usr/bin/env python
"""Run the tuned pipeline on the real competition test set and write submission.csv."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from animalclef import cluster as cl
from animalclef.config import (CACHE, DATA, SPECIES, SPECIES_KEY,
                               SPECIES_WITH_TRAIN, SUBMISSIONS)
from animalclef.data import load_metadata
from animalclef.pipeline import Config, cluster_similarity, fit_calibrators
from animalclef.similarity import cosine, fuse, rank_normalize

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

DEFAULT_CFG = {
    "weights": {"miewid": 0.5, "mega": 0.5},
    "threshold": 0.5,
    "use_rerank": False,
    "rr": [20, 6, 0.3],
    "post": {},
    "consensus": False,
}


def load_best(dataset: str) -> dict:
    p = CONFIG_DIR / f"best_{dataset}.json"
    if p.exists():
        return {**DEFAULT_CFG, **json.loads(p.read_text())}
    return dict(DEFAULT_CFG)


def config_from(best: dict, dataset: str) -> Config:
    post = best.get("post", {})
    return Config(
        dataset=dataset,
        weights=best["weights"],
        threshold=best["threshold"],
        use_rerank=best["use_rerank"],
        rr_k1=best["rr"][0], rr_k2=best["rr"][1], rr_lambda=best["rr"][2],
        merge_small=post.get("merge_small", 2),
        merge_threshold=post.get("merge_threshold", 0.4),
        split_big=post.get("split_big", 60),
        split_threshold=post.get("split_threshold", 0.65),
        anchor=post.get("anchor", False),
        use_orientation_calibration=best.get("use_orientation_calibration", False),
    )


def load_emb(backbone: str, dataset: str, split: str, suffix: str = ""):
    p = CACHE / f"emb_{backbone}_{SPECIES_KEY[dataset]}_{split}{suffix}.npy"
    return np.load(p) if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SUBMISSIONS / "submission.csv"))
    ap.add_argument("--backbones", default="miewid,mega")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--density-match", action="store_true",
                    help="adjust thresholds so the partition density matches the "
                         "benchmark-validated target (hedges against test-set "
                         "similarity distribution shift)")
    args = ap.parse_args()

    meta = load_metadata()
    sample = pd.read_csv(DATA / "sample_submission.csv")
    backbones = args.backbones.split(",")
    all_rows = []
    report = {}

    for ds in SPECIES:
        best = load_best(ds)
        cfg = config_from(best, ds)
        key = SPECIES_KEY[ds]
        test = meta[(meta["dataset"] == ds) & (meta["split"] == "test")]
        train = meta[(meta["dataset"] == ds) & (meta["split"] == "train")]
        q_embs, d_embs = {}, {}
        for b in backbones:
            e_t = load_emb(b, ds, "test", args.suffix)
            if e_t is None:
                continue
            q_embs[b] = e_t
            e_d = load_emb(b, ds, "train", args.suffix)
            if e_d is not None:
                d_embs[b] = e_d
        if not q_embs:
            print(f"[skip] {ds}: no test embeddings")
            continue

        t0 = time.time()
        if len(train) > 0 and d_embs:
            cals = fit_calibrators(d_embs, train, cfg)
            q_orient = test["orientation"].values if cfg.use_orientation_calibration else None
            qq = {b: cals[b].transform(cosine(e, e), q_orient) for b, e in q_embs.items()}
            qd = {b: cals[b].transform(cosine(q_embs[b], d_embs[b])) for b in d_embs}
            db_ids = np.asarray(train["identity"].astype(str).values, dtype=object)
        else:
            # Zero-shot species (TexasHornedLizards): no calibration data.
            # Rank-normalise similarities into [0, 1] instead.
            qq = {b: rank_normalize(cosine(e, e)) for b, e in q_embs.items()}
            qd, db_ids = {}, None

        w = {k: v for k, v in cfg.weights.items() if k in qq and v > 0}
        sim = fuse(qq, w)

        # Local-feature channel (RootSIFT/RANSAC), fused before re-ranking so
        # both signals participate in neighbour refinement.
        w_local = float(best.get("local_weight", 0.0))
        if w_local > 0:
            method = best.get("local_method", "root_sift")
            local_path = CACHE / f"local_{method}_{key}_test.npy"
            if local_path.exists():
                lsim = np.load(local_path)
                if lsim.shape[0] == len(test):
                    from animalclef.localfeat import normalize_local
                    lsim = normalize_local(lsim)
                    sim = (sim + w_local * lsim) / (1.0 + w_local)
                    print(f"  [{ds}] fused local features w={w_local}")
                else:
                    print(f"  [{ds}] local cache shape mismatch "
                          f"{lsim.shape} vs {len(test)}, ignored")
            else:
                print(f"  [{ds}] no local cache at {local_path.name}, ignored")

        if cfg.use_rerank:
            sim = cl.rerank_similarity(sim, cfg.rr_k1, cfg.rr_k2, cfg.rr_lambda)

        from animalclef.pipeline import refine
        sim_qdb = fuse(qd, w) if qd else None

        def partition(threshold: float) -> np.ndarray:
            c = replace(cfg, threshold=threshold)
            if best.get("consensus") and len([b for b in w if w[b] > 0]) > 1:
                parts = [cl.agglomerative(qq[b], threshold) for b in w if w[b] > 0]
                lab = cl.consensus_cluster(parts, threshold)
            else:
                lab = cluster_similarity(sim, c)
            return refine(lab, sim, c, sim_qdb=sim_qdb, db_ids=db_ids)

        labels = partition(cfg.threshold)
        target_mean = best.get("target_mean_cluster_size")
        if args.density_match and target_mean:
            target_k = max(2, int(round(len(test) / float(target_mean))))
            cur_k = len(set(labels))
            if abs(cur_k - target_k) / target_k > 0.10:
                lo, hi = cfg.threshold * 0.5, min(0.995, cfg.threshold * 4.0)
                for _ in range(24):
                    mid = 0.5 * (lo + hi)
                    k = len(set(partition(mid)))
                    if k < target_k:
                        lo = mid
                    else:
                        hi = mid
                cfg = replace(cfg, threshold=0.5 * (lo + hi))
                labels = partition(cfg.threshold)
                print(f"  [{ds}] density match: {cur_k} -> {len(set(labels))} "
                      f"clusters (target {target_k}, t={cfg.threshold:.3f})")

        labels_str = [f"cluster_{ds}_{int(c)}" for c in labels]
        for image_id, lab in zip(test["image_id"].values, labels_str):
            all_rows.append({"image_id": int(image_id), "cluster": lab})
        n_clusters = len(set(labels_str))
        report[ds] = {
            "n_test": int(len(test)),
            "n_clusters": int(n_clusters),
            "mean_cluster_size": round(len(test) / max(1, n_clusters), 2),
            "seconds": round(time.time() - t0, 1),
            "config": best,
        }
        print(f"[{ds}] {len(test)} test images -> {n_clusters} clusters "
              f"({time.time()-t0:.1f}s)", flush=True)

    sub = pd.DataFrame(all_rows)
    out = sample[["image_id"]].merge(sub, on="image_id", how="left")
    missing = out["cluster"].isna().sum()
    if missing:
        out["cluster"] = out["cluster"].fillna("cluster_unknown_0")
        print(f"[warn] {missing} sample rows had no prediction")
    assert out["image_id"].duplicated().sum() == 0
    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(out)} rows, "
          f"{out['cluster'].nunique()} unique clusters)")
    (SUBMISSIONS / "submission_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
