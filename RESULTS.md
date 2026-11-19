# AnimalCLEF26 — Results and evidence

All numbers are **ARI** on the local identity-level open-set benchmark
(mean ± std over independent known/new-identity splits of the labelled
database). The official test labels are private; this benchmark is our
evaluation protocol. Tuning used seeds 0–3; held-out numbers use seeds
100–103 (never seen during tuning).

## Benchmark protocol

- 40 % of identities are *new*: they appear only in the pseudo-query set.
- Known identities contribute re-sightings; each keeps ≥1 image in the
  pseudo-database.
- The pseudo-query is sized like the real competition test split
  (lynx/salamander 600, turtle 500).
- Every configuration gets its clustering threshold tuned on the same splits.
- **Mean over the four species is the competition metric**; Texas horned
  lizards have no labels and are excluded from local evaluation.

## Final pipeline

| Species | Global-only (tuned) | Final config (tuning seeds) | **Held-out (seeds 100–103)** |
|---|---|---|---|
| LynxID2025 | 0.139 | 0.305 | **0.313 ± 0.026** |
| SalamanderID2025 | 0.152 | 0.316 | **0.325 ± 0.033** |
| SeaTurtleID2022 | 0.744 | 0.744 | **0.749 ± 0.011** |
| TexasHornedLizards | — | zero-shot density prior | not evaluable |

Held-out mean over the three evaluable species: **0.462**.

### Per-species configuration

| Species | Backbone weights | Re-rank | Local (RootSIFT) w | Anchor merge | Threshold |
|---|---|---|---|---|---|
| Lynx | MiewID 0.7 / Mega 0.3 | no | 0 (did not generalise) | yes | 0.72 |
| Salamander | MiewID 0.7 / Mega 0.3 | no | 1.5 | no | 0.14 |
| Turtle | MiewID 0.7 / Mega 0.3 | no | 0 | yes | 0.19 |
| Lizard | 0.5 / 0.5 (rank-normalised) | no | 2.0 | no | density prior |

## Component ablations

`scripts/ablation.py`, seeds 0–2, threshold re-tuned per configuration:

| Configuration | Lynx | Salamander | Turtle |
|---|---|---|---|
| MegaDescriptor only | 0.090 | 0.105 | 0.686 |
| MiewID only | 0.131 | 0.153 | 0.543 |
| Equal fusion 0.5 / 0.5 | 0.123 | 0.150 | 0.720 |
| Tuned fusion 0.7 / 0.3 | 0.139 | **0.160** | 0.728 |
| Tuned fusion + re-ranking | 0.146 | 0.132 | 0.514 |
| Tuned fusion + anchor merge | **0.305** | 0.160 | **0.743** |

Additional local-feature channel (not part of the ablation grid; measured
with the full post-processing, `tune_local.py`):

| Species | Without local | With local | Weight |
|---|---|---|---|
| Salamander | 0.152 | **0.316** | 1.5 |
| Lynx | 0.305 (anchor, tuned) | 0.311 (2 seeds) → held-out 0.274 | 0.1 / 0.7 |
| Turtle | 0.744 | 0.741 | — (disabled) |

Findings:

- **Turtle**: fusion is the main lever (+0.19 over MiewID alone); re-ranking
  badly hurts (−0.21); MegaDescriptor alone is surprisingly strong (0.686).
- **Lynx**: database-anchor merging is the dominant lever (+0.17); the
  backbone fusion adds little because lynx similarity is weak.
- **Salamander**: the classical local-feature channel is the dominant lever,
  roughly doubling ARI.

## Negative / rejected results

Implemented and measured, then disabled:

| Idea | Result |
|---|---|
| Orientation-conditioned calibration (lynx) | 0.696 vs 0.704 — no gain |
| Consensus clustering across backbones | lynx 0.105, salamander 0.123, turtle 0.693 — worse |
| k-reciprocal re-ranking on turtle / salamander | turtle 0.514 vs 0.728; salamander 0.132 vs 0.160 |
| Local features on turtle | 0.741 vs 0.744 |
| Silhouette / partition-stability threshold selection (lizard) | degenerate to all-singleton partitions |
| hflip TTA | no gain on MPS, 2x inference cost |
| Local-feature weight for lynx (w=0.7, tuning seeds) | held-out 0.274 vs 0.313 without — seed overfit |

## Submissions (real test set)

Both files have 2,409 rows and are validated against `sample_submission.csv`
(identical ids and order, no missing values):

| Species | Test images | `submission.csv` | `submission_densitymatched.csv` |
|---|---|---|---|
| LynxID2025 | 946 | 549 | 549 |
| SalamanderID2025 | 689 | 276 | **457** |
| SeaTurtleID2022 | 500 | 240 | 240 |
| TexasHornedLizards | 274 | 179 | 179 |

**Why two files.** Direct threshold transfer over-merges salamander: the test
similarity distribution has a much heavier tail than the benchmark (fraction
of pairs above the tuned threshold 0.14: **2.86 %** on test vs **0.30 %** on a
benchmark split; p99 0.208 vs 0.093). The resulting test density
(2.5 img/cluster) lies outside the validated 1.4–2.1 band. Lynx (1.72 vs
1.61) and turtle (2.08 vs 1.92) transfer well.

- `submission.csv` — direct transfer of benchmark-tuned thresholds (primary;
  no unvalidated correction).
- `submission_densitymatched.csv` — additionally re-selects the threshold so
  the mean cluster size matches the held-out optimum (1.5 for salamander).
  Only salamander changes (276 → 457 clusters).

## Zero-shot lizard sensitivity

Salamander (most similar sparse pattern species) ARI as a function of
partition density at fixed tuning:

| mean cluster size | 2.06 | 1.54 | 1.44 | 1.38 | 1.31 | 1.21 | 1.13 |
|---|---|---|---|---|---|---|---|
| ARI | 0.283 | 0.312 | 0.308 | 0.299 | 0.291 | 0.234 | 0.165 |

ARI is flat for mean cluster sizes 1.4–2.1 and collapses when over-fragmenting,
so for Texas horned lizards (no labels) we select the threshold whose
partition has mean cluster size 1.5 (→ 179 clusters).
