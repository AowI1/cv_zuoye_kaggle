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

## Leaderboard result

| | public | private |
|---|---|---|
| Final submission | 0.43818 | **0.52528** |

On the final leaderboard that score sits at **rank 9 of 232 teams
(top 3.9%)** — 8th scores 0.53445 and 10th 0.47075. Our submissions were
made after the competition closed (deadline 2026-04-30), so the score is not
officially ranked; it is what the competition's own scorer returns on the
private split.

The final partition (2,409 images → 867 clusters):

| Species | Clusters | Configuration |
|---|---|---|
| LynxID2025 | 129 | threshold 0.22, MiewID 0.7 / Mega 0.3, database anchor merge |
| SalamanderID2025 | 344 | mean size 2.0, equal backbone weights, RootSIFT w=1.5 |
| SeaTurtleID2022 | 157 | mean size 2.3 + same-day similarity boost 0.70 |
| TexasHornedLizards | 237 | mean size 1.15, centre-crop 0.85, RootSIFT w=2.0 |

Reproduce with `python scripts/make_best_submission.py` (configuration in
`configs/best_combo.json`); every probe and every rejected idea is recorded
in `docs/SWEEP.md`. Starting from the benchmark-tuned pipeline the same
submission files move the score from 0.37244 to 0.52528.

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

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # tested on Python 3.15.2
```

Paths are resolved relative to the repository, so it runs from any clone
location. Set `ANIMALCLEF_ROOT` to keep `data/`, `models/` and `cache/`
somewhere else.

## Fast path (no GPU, ~2 MB of downloads)

`cache/` already contains every embedding, local-feature similarity matrix
and tuned configuration, and `submissions/` the generated outputs. To rebuild
the exact submission from the committed cache:

```bash
pip install numpy pandas scikit-learn opencv-python-headless pillow
python scripts/download_data.py --meta-only   # metadata.csv + sample_submission.csv
python scripts/make_submission.py
python scripts/make_submission.py --density-match \
    --out submissions/submission_densitymatched.csv
```

Nothing else is required: no model weights, no image archive, no GPU.

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

The local benchmark ranks feature quality well but underestimates the true
test density, so the operating points were re-tuned directly on the
leaderboard (see `docs/SWEEP.md`); the leaderboard score above is the one
that matters.

The official leaderboard labels are private, so these are local estimates on
identity-level open-set splits of the labelled database (tuning seeds 0–3,
validation seeds 100–103). See `RESULTS.md` and `docs/METHOD.md`.

## Notes on data

The official competition data is required (Kaggle competition rules). A
Hugging Face mirror (`BVRA/animal-clef-2026`) provides `train.zip`,
`test.zip`, `metadata.csv` and `sample_submission.csv`; weights come from
`conservationxlabs/miewid-msv3` and `BVRA/MegaDescriptor-L-384`.
