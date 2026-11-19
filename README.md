# AnimalCLEF26 — Open-World Wildlife Re-Identification

End-to-end solution for the [AnimalCLEF26 @ CVPR & CLEF competition](https://www.kaggle.com/competitions/animal-clef-2026):
group **2,409 unlabeled test images** across four species into individual
animals (open-world clustering), scored by mean ARI over species.

This repository contains a full pipeline built **from scratch** on the
official competition data. Because the test labels are private, every design
decision is measured on an identity-level **open-set benchmark** built from
the labelled database split, with tuning and held-out seeds kept separate.
Several algorithmic extensions to the public solutions are implemented and
ablated (see `RESULTS.md`); some were adopted, some were measured and rejected.

---

## Task

| Species | Train (DB) | Identities | Test | Key challenge |
|---|---|---|---|---|
| LynxID2025 | 2,957 | 77 | 946 | IR vs visible lighting, flank orientation |
| SalamanderID2025 | 1,388 | 587 | 689 | Extreme sparsity (2.4 img/id), spot patterns |
| SeaTurtleID2022 | 8,729 | 438 | 500 | Long time span, underwater variation |
| TexasHornedLizards | 0 | 0 | 274 | **Zero-shot**: no training data at all |

**Metric**: ARI per species, averaged over the four species.
**Output**: `image_id, cluster` with arbitrary group labels (ARI is
label-invariant — only the *partition* matters).

## Pipeline

```
                          ┌─────────────── labelled database ───────────────┐
                          │                                                 │
 test image   ──► species-aware preprocessing ──► MiewID-MSv3 (2152d) ─┐   │
 (train/test)      (IR norm, CLAHE, Sobel, ...)     MegaDescriptor-L ──┤   │
                          │                         SIFT/RootSIFT ┴──────┐ │
                          ▼                                              │ │
              isotonic calibration on DB pairs  ◄───────────────────────┘ │
              (score → P(same individual), per backbone,                 │
               orientation-conditioned for lynx)                          │
                          │                                              │
                          ▼                                              │
          calibrated weighted fusion  ◄──── k-reciprocal re-ranking      │
                          │                                              │
                          ▼                                              │
       HAC / Chinese Whispers / DBSCAN / consensus  ◄─────────────────────┘
                          │                     (anchor merge uses DB identities)
                          ▼
       split-big → merge-small → anchor-merge → submission
```

### Innovations over the public solutions

1. **Identity-level open-set benchmark.** The competition test labels are
   private, so we synthesize a faithful open-world benchmark from the
   database: identities are split into *known* (seen in the pseudo-database)
   and *new* (query-only), and query images of known identities emulate
   re-sightings. All hyperparameters are tuned against ground-truth ARI on
   this benchmark (multiple seeds / new-identity ratios), not on the public
   leaderboard.
2. **Orientation-conditioned calibration (lynx).** Camera-trap images carry
   `left/right/top/front/back` orientation metadata. Matching opposite flanks
   is intrinsically harder, so a single isotonic map over-scales those pairs;
   we fit a separate score→probability map per orientation pair and fall back
   to the global map when a bucket has too few pairs.
3. **Similarity-level multi-backbone fusion with per-backbone isotonic
   calibration** instead of the reference CSV majority vote — smooth, and it
   makes local-feature scores commensurable with neural embeddings.
4. **Classical local-feature channel (RootSIFT + RANSAC)** computed
   only on candidate pairs proposed by global similarity. Fine-grained
   pattern species (salamander, lizard) benefit where global embeddings fail.
5. **Partition-level consensus clustering** across backbones as an
   alternative to a single fused graph (co-association + average linkage).
6. **Database-anchored cluster merging**: query clusters whose strongest
   database identity agrees are merged, injecting database structure without
   forcing the brittle joint train+test clustering that public solutions
   found harmful.
7. **Zero-shot handling for Texas horned lizards**: no calibration data
   exists, so similarities are rank-normalized and clustered with priors
   transferred from the other species.

## Repository layout

```
src/animalclef/          # library
  config.py              # paths, species, device
  data.py                # metadata + threaded image loading
  preprocess.py          # species-specific image enhancement
  embed.py               # MiewID / MegaDescriptor backbones, TTA
  similarity.py          # cosine, top-k, fusion
  calibration.py         # isotonic (+ orientation-conditioned) calibration
  cluster.py             # HAC, Chinese Whispers, DBSCAN, k-reciprocal re-ranking, consensus
  postprocess.py         # split-big, merge-small, anchor merge
  localfeat.py           # SIFT / RootSIFT + RANSAC matching
  pipeline.py            # per-species orchestration
  benchmark.py           # open-set identity splits + ARI
scripts/                 # entry points (download, extract, benchmark, tune, submit)
configs/                 # tuned per-species configurations
cache/                   # cached embeddings / similarity artifacts
submissions/             # generated submission.csv
```

## Reproduction

```bash
# 1. data (official competition data mirrored on Hugging Face)
python scripts/download_data.py
cd data/animalclef2026 && unzip -q train.zip && unzip -q test.zip

# 2. frozen model weights (MiewID-MSv3, MegaDescriptor-L-384)
python scripts/download_models.py

# 3. embeddings (MPS on Apple Silicon, or CUDA)
python scripts/extract_embeddings.py --backbone miewid
python scripts/extract_embeddings.py --backbone mega

# 4. local-feature candidate similarities (SIFT/RANSAC, CPU)
python scripts/extract_local.py --species SalamanderID2025 --split test
python scripts/extract_local.py --species LynxID2025 --split test
python scripts/extract_local.py --species SeaTurtleID2022 --split test
python scripts/extract_local.py --species TexasHornedLizards --split test

# 5. tune per species on the open-set benchmark
python scripts/tune.py --species LynxID2025 --seeds 4        # backbone fusion, re-rank, post
python scripts/tune.py --species SalamanderID2025 --seeds 4
python scripts/tune.py --species SeaTurtleID2022 --seeds 4
python scripts/tune_local.py --species LynxID2025 --seeds 2  # local-feature weight
python scripts/tune_local.py --species SalamanderID2025 --seeds 2
python scripts/tune_local.py --species SeaTurtleID2022 --seeds 2
python scripts/tune_lizard.py                                # zero-shot density prior

# 6. validate on held-out benchmark seeds, then submit
python scripts/validate_final.py --seeds 100 101 102 103
python scripts/make_submission.py                    # primary submission
python scripts/make_submission.py --density-match \
    --out submissions/submission_densitymatched.csv  # density-matched variant
python scripts/ablation.py --species all --seeds 3
```

## Results (local open-set benchmark, mean ARI)

| Species | Global only | Final pipeline | Held-out validation (seeds 100–103) |
|---|---|---|---|
| LynxID2025 | 0.139 | 0.305 | **0.313 ± 0.026** |
| SalamanderID2025 | 0.152 | 0.316 | **0.325 ± 0.033** |
| SeaTurtleID2022 | 0.744 | 0.744 | **0.749 ± 0.011** |
| TexasHornedLizards | — (zero-shot) | density prior | not evaluable |

Held-out mean over the three evaluable species: **0.462**.

The official leaderboard labels are private, so these are local estimates on
identity-level open-set splits of the labelled database (tuning seeds 0–3,
validation seeds 100–103). See `RESULTS.md` and `docs/METHOD.md`.

## Notes on data

The official competition data is required (Kaggle competition rules). A
Hugging Face mirror (`BVRA/animal-clef-2026`) provides `train.zip`,
`test.zip`, `metadata.csv` and `sample_submission.csv`; weights come from
`conservationxlabs/miewid-msv3` and `BVRA/MegaDescriptor-L-384`.
