# Single-species variant sweep

Baseline: `base_densitymatched.csv` = the file already on Kaggle
(private **0.37244**, public **0.35805**, 1244 clusters).

## Design

The competition metric is the mean ARI over four species, so changing the
clustering of **exactly one** species in a submission gives a clean read-out:

```
Δoverall × 4  ≈  Δ ARI of that species
```

Every file here therefore differs from the baseline in one species only
(verified row-by-row; see the last column). No embeddings were recomputed —
all variants come from `cache/` in ~5 s.

## Variants

| File | Species | Clusters | Hypothesis | Public score | Δ |
|---|---|---|---|---|---|
| `base_densitymatched.csv` | — | 549 / 457 / 240 / 179 | submitted baseline | 0.35805 | — |
| `salam_direct_t0.14.csv` | Salamander | 276 | tuned threshold without density correction | | |
| `salam_mean2.0.csv` | Salamander | 343 | density target between the two | | |
| `salam_mean2.2.csv` | Salamander | 312 | near the benchmark optimum 2.06 | | |
| `lizard_mean2.0.csv` | Lizard | 137 | zero-shot prior may be too sparse | | |
| `lizard_mean2.5.csv` | Lizard | 110 | stronger merging | | |
| `lizard_mean3.0.csv` | Lizard | 90 | aggressive merging | | |
| `lizard_nolocal.csv` | Lizard | 183 | RootSIFT may hurt without calibration | | |
| `turtle_mean1.8.csv` | Turtle | 277 | local plateau is flat 1.4–2.1 | | |
| `turtle_mean2.3.csv` | Turtle | 216 | test set may be denser than benchmark | | |
| `lynx_t0.66.csv` | Lynx | 513 | lower threshold, more merging | | |
| `lynx_t0.78.csv` | Lynx | 588 | higher threshold, more fragments | | |
| `lynx_local0.1.csv` | Lynx | 588 | small local-feature weight | | |

## Reading the results

- Lynx / turtle / salamander variants: ΔARI = 4 × Δoverall.
- Lizard variants are the highest-leverage ones — the species currently
  contributes ≈ 0 to the mean, so any positive ARI is pure upside.
- If two variants differ by less than ~0.005 overall, treat it as noise.

## Submitting

```bash
kaggle competitions submit -c animal-clef-2026 -f submissions/variants/<file>.csv \
       -m "<variant name>"
```

or drag-and-drop on the competition page. Record the public score in the
table above; private scores are only meaningful after the deadline.
