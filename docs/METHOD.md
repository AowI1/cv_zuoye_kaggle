# Method notes

Design decisions, evidence, and engineering notes for the AnimalCLEF26
pipeline. Numbers refer to the local open-set benchmark (`RESULTS.md`).

## 1. Task framing

The competition gives 15,483 images (13,074 labelled database images across
three species; 2,409 unlabelled test images across four). The metric is the
mean Adjusted Rand Index (ARI) over the four species (label-invariant; only
the partition matters). Texas horned lizards have **no** training images, so
for them the task is pure zero-shot clustering.

## 2. Similarity backbones

| Model | Input | Dim | Role |
|---|---|---|---|
| `conservationxlabs/miewid-msv3` (timm efficientnetv2_rw_m + GeM + BN) | 440² | 2152 | trained for individual re-ID |
| `BVRA/MegaDescriptor-L-384` (Swin-L) | 384² | 1536 | strong species/individual separation |

Both frozen; embeddings L2-normalised. Test-time augmentation (hflip) is
implemented but disabled by default (no measurable gain on MPS, 2x cost).

Engineering notes:

- The MiewID HuggingFace custom code targets transformers 4.4x and breaks on
  transformers 5.x (`all_tied_weights_keys`). We re-implement the head
  (`timm` backbone + GeM + BN) and load the same safetensors directly, which
  removes the transformers dependency entirely.
- The MegaDescriptor checkpoint stores an old-timm state dict where the patch
  merging modules sit at the *end* of each stage; current timm puts them at
  the *start* of the next stage. We remap keys (`layers.i.downsample.* →
  layers.i+1.downsample.*`, dropping stale attention buffers) and load
  strictly. timm is pinned to 0.9.16.
- OpenCV 5 removed KAZE/AKAZE; we use SIFT and RootSIFT (and ORB optionally).

## 3. Species-aware preprocessing

Ported from the winning solution and refined:

- **Lynx**: unsharp mask → CLAHE on V → bilateral. Additionally, IR frames
  are detected by mean HSV saturation (< 22), their chroma is neutralised and
  luminance aligned (LAB) before the recipe, bridging the IR/visible domain
  gap.
- **Turtle**: red-channel boost (×1.3) + gamma 1.5 for underwater colour cast.
- **Salamander**: Sobel-edge overlay (30 %) + gamma 1.5 to emphasise the spot
  pattern.
- **Lizard**: glare inpainting (L > 245) + luminance scaling.
- Images are downscaled to max side 512 *before* the OpenCV recipes (original
  images are 600–2600 px), which keeps CPU preprocessing ~4x cheaper.

## 4. Calibration and fusion

Raw cosine scores from different backbones are not comparable. We fit an
isotonic map `cosine → P(same individual)` per backbone on sampled pairs from
the labelled database (up to 1024 images, 400k pairs). Transformations are
applied to query-query and query-database similarities before a weighted
fusion.

**Orientation-conditioned calibration** (lynx): separate isotonic maps per
orientation pair (left/right/front/back/top), falling back to the global map
for sparse buckets. Measured effect on the benchmark was slightly negative
(0.6958 vs 0.7042 for a variant), so it is disabled in the final config —
kept in the code as an evaluated alternative.

## 5. Clustering and refinement

- Agglomerative clustering (average linkage on the precomputed
  `1 − similarity` distance) is the workhorse; Chinese Whispers and DBSCAN
  are implemented and were beaten in tuning.
- **k-reciprocal re-ranking** (ported from the winning solution) helps lynx
  at intermediate thresholds but *hurts* those configurations that use the
  database anchor, and hurts turtle/salamander strongly (turtle −0.21). The
  final lynx config therefore runs without re-ranking — validated on held-out
  seeds (0.313 vs 0.274 with the tuning-seed re-ranked variant).
- **split-big / merge-small**: re-cluster oversized clusters at a stricter
  threshold; absorb tiny clusters into their most similar neighbour.
- **Database-anchor merging** (this work): query clusters whose strongest
  database identity agrees are merged. This injects database structure into
  the partition without forced joint clustering (the fully joint variant,
  tried in public solutions, collapsed ARI). It is the single largest lever
  for lynx (+0.17 over tuned fusion) and a small gain for turtle (+0.015).
  Critically, the anchor interacts with the *clustering threshold*: tuning
  the two jointly found a much better operating point (t=0.72) than tuning
  the threshold first and adding the anchor afterwards.
- **Consensus clustering** across per-backbone partitions (co-association
  matrix + average linkage) is implemented; it under-performed the fused
  similarity and is off by default.

## 6. Local-feature channel (RootSIFT / RANSAC)

SIFT keypoints (1200/image) on grayscale images at max side 800; RootSIFT
descriptors; mutual ratio test (0.75); homography RANSAC (5 px); score =
inliers / min(#keypoints), normalised by the 99th percentile of its non-zero
distribution.

Candidate pairs are the union of each image's top-10 neighbours and all pairs
above the 95th percentile of the global similarity (an absolute cosine gate
fails because baseline similarity differs by species: p50 0.22 for turtles vs
0.48 for lizards).

Fusion: `S = (S_global + w·S_local)/(1+w)`.

Measured effect (benchmark): salamander 0.152 → **0.316** (w=1.5) — the spot
patterns are exactly what SIFT/RANSAC excels at, and the gain was fully
reproduced on held-out seeds (0.325). Lynx showed large tuning-seed gains
(0.203 → 0.342 at w=0.7) but they did **not** generalise: the held-out ARI of
that configuration was 0.274 vs **0.313** for the anchor configuration without
local features. This is a good example of why the held-out split is kept
separate — a component that looks decisive on two tuning seeds can be seed
overfitting.

## 7. Zero-shot lizards

No labelled data → no calibration and no ARI tuning. We rank-normalise each
backbone's similarity, fuse, add the local channel (w=1.5, transferred from
salamander), and pick the threshold by a **density prior**: the partition
with mean cluster size closest to 1.5. The prior comes from the salamander
sensitivity curve (ARI is flat for mean sizes 1.4–2.1 and peaks ~1.5), which
is the most similar sparse pattern species. Silhouette and partition
stability both degenerate to all-singleton solutions and were rejected.

## 8. Benchmark protocol

Official test labels are private. We construct an identity-level open-set
split of the labelled database: 40 % of identities are new (query-only);
known identities contribute re-sightings; the query size equals the real
competition test size. Hyperparameters are tuned on seeds 0–3 and evaluated
on held-out seeds 100–103.
