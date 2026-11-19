# AnimalCLEF26 Winning Solution

This repository provides our winning solution for the AnimalCLEF26 CVPR & CLEF
competition.

Competition Page:
https://www.kaggle.com/competitions/animal-clef-2026

Leaderboard:
https://www.kaggle.com/competitions/animal-clef-2026/leaderboard

The solution is a notebook-driven individual animal re-identification
pipeline that loads the AnimalCLEF dataset, computes calibrated WildFusion
similarities, uses fine-tuned MiewID backbones, clusters test images
with Chinese Whispers, attaches confident clusters to known train identities,
and generates competition-style submission CSVs with ensemble support.

---

## Team Members

- Seifeldin Elkerdany (Team Leader)
- Mohamed Elbassat (Team Leader)
- Gamal Abouelhamd
- Mohamed Hisham
- Jana Ghoneim
- Mohamed elboraay
- Assem Elkady

---

## Fine-Tuned Weights

We provide the fine-tuned weights through the following links:

- [SphereFace weights](https://www.kaggle.com/models/seifelkerdany/finetuned-miewid-with-sphereface2)
- [ArcFace weights](https://www.kaggle.com/models/mohamedbassat/dyn-arcface-finetuned-miewid-for-clef26)

The runnable full-pipeline notebooks are:

```text
full_pipeline_miewid.ipynb        # Original Hugging Face MiewID
full_pipeline_sphereface.ipynb    # SphereFace fine-tuned MiewID checkpoint
full_pipeline_arcface.ipynb       # ArcFace fine-tuned MiewID checkpoint
```

---

## Dataset Variants

The full-pipeline notebooks expect pre-segmented input images.
Segmentation is not performed by the code in this repository.

| Notebook | MiewID branch | Salamander segmentation dataset |
| --- | --- | --- |
| `full_pipeline_miewid.ipynb` | Original `conservationxlabs/miewid-msv3` | White background |
| `full_pipeline_sphereface.ipynb` | SphereFace fine-tuned checkpoint | White background |
| `full_pipeline_arcface.ipynb` | ArcFace fine-tuned checkpoint | Black background |

Use the white-background salamander segmentation dataset for the MiewID and
SphereFace notebooks.

Use the black-background salamander segmentation dataset for the
ArcFace notebook.

---

## Project Structure

```text
.
├── full_pipeline_miewid.ipynb
├── full_pipeline_sphereface.ipynb
├── full_pipeline_arcface.ipynb
├── ensemble-csvs-clef2026.ipynb

├── Database_optuna_notebooks/
│   ├── Lynx_DB.ipynb
│   ├── Salamnder_DB.ipynb
│   ├── SeaTurtle_DB.ipynb
│   ├── optuna.ipynb
│   └── split.ipynb

├── sphereface_focal_finetune/
│   ├── finetune-miewid.ipynb
│   ├── miewid_open_set_split.py
│   ├── fine_tune_train.csv
│   ├── val_known.csv
│   ├── val_unknown.csv
│   └── val_all.csv

└── src/
    ├── data/
    │   ├── preprocessing.py
    │   └── data_transformation.py
    │
    ├── feature_extraction/
    │   └── wildfusion.py
    │
    ├── clustering_algorithms/
    │   └── chinese_whispers.py
    │
    └── utilities/
        ├── finetuned_model.py
        ├── graph_construction.py
        ├── image2identity.py
        ├── openset_split.py
        └── reranking.py
```

Each `src` subfolder includes an `__init__.py` file, so modules can be imported
with package-style imports:

```python
from src.feature_extraction.wildfusion import WildFusionFeatureExtractor
from src.clustering_algorithms.chinese_whispers import chinese_whispers
from src.utilities.image2identity import compute_identity_scores
```

---

## Pipeline Overview

### 1. Load AnimalCLEF Data

The notebooks use `wildlife-datasets`:

```python
from wildlife_datasets.datasets import AnimalCLEF2026
```

Metadata is split into:

- train/database samples
- test/query samples

---

### 2. Choose the MiewID Branch

`WildFusionFeatureExtractor` always starts from the Hugging Face MiewID
architecture:

```python
hf_tag = "conservationxlabs/miewid-msv3"
```

The model branch is selected through constructor arguments:

```python
# Original MiewID
wildfusion = WildFusionFeatureExtractor(
    dataset_calibration,
    device=device
)

# SphereFace fine-tuned MiewID
wildfusion = WildFusionFeatureExtractor(
    dataset_calibration,
    device=device,
    sphereface_version="/path/to/sphereface_checkpoint.pt",
)

# ArcFace fine-tuned MiewID
wildfusion = WildFusionFeatureExtractor(
    dataset_calibration,
    device=device,
    arcface_version="/path/to/arcface_model_best.bin",
)
```

`sphereface_version` and `arcface_version` are mutually exclusive.
Do not pass both in the same run.

---

### 3. Calibrate WildFusion

`WildFusionFeatureExtractor` builds three calibrated similarity pipelines:

- **MiewID branch**:
  Original MiewID, SphereFace fine-tuned MiewID, or ArcFace fine-tuned MiewID
  using cosine similarity.

- **ALIKED**:
  Local features matched with LightGlue.

- **DISK**:
  Local features matched with LightGlue.

The extractor is intentionally two-step:

```python
wildfusion = WildFusionFeatureExtractor(
    dataset_calibration,
    device=device
)

wildfusion.calibrate()
```

Object construction stores configuration only.
Model loading and calibration happen inside `calibrate()`.

---

### 4. Apply Species-Specific Preprocessing

`SpeciesTransform` chooses preprocessing based on the dataset name:

| Dataset keyword | Processing |
| --- | --- |
| `lynx` | Sharpening, HSV/CLAHE contrast enhancement, bilateral filtering |
| `turtle` | Red-channel compensation and gamma correction |
| `salamander` | Sobel edge overlay and gamma correction |
| Other | No species-specific preprocessing |

---

### 5. Compute Similarities

For each dataset/species, the notebooks compute:

- test-vs-test similarities for clustering query images
- test-vs-train similarities for attaching clusters to known identities

Similarity values are clipped to `[0.0, 1.0]`
before downstream processing.

---

### 6. Score Known Identities

`src.utilities.image2identity` converts image-level test-vs-train scores
into identity-level scores.

For each known identity, it averages the strongest `top_r`
training-image matches, then extracts:

- best identity name
- best score
- second-best score
- margin between best and second-best

---

### 7. Optional Re-Ranking

`src.utilities.reranking.re_ranking` can refine the test-vs-test similarity
matrix using k-reciprocal neighborhoods.

The notebooks enable this only for species listed in `RR_SPECIES`
and only when the species config sets `use_rerank=True`.

---

### 8. Cluster with Chinese Whispers

`src.clustering_algorithms.chinese_whispers.chinese_whispers`
clusters the test images.

The pipeline:

1. Builds a thresholded similarity graph
2. Extracts a k-core backbone
3. Optionally applies mutual-kNN graph filtering
4. Runs weighted Chinese Whispers label propagation

---

### 9. Resolve Final Labels

If a cluster has confident known-identity votes, it receives a known identity label:

```text
cluster_<dataset>_<known_identity>
```

Otherwise, it receives a new identity label:

```text
cluster_<dataset>_new_<cluster_id>
```

---

### 10. Write Submission

The notebooks merge generated labels with the sample submission and write:

```text
submission.csv
submission_sphereface.csv
submission_arcface.csv
```

The ensemble notebook combines predictions from multiple runs.

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

The requirements include:

- `wildlife-datasets`
- `wildlife-tools`
- `transformers==4.45.2`
- `torch`
- `torchvision`
- `opencv-python`
- `numpy`
- `pandas`
- `scikit-learn`
- `networkx`
- `Pillow`
- `lmdb`

On Kaggle or similar hosted environments, some dependency warnings may appear
from preinstalled packages.
The pipeline itself is PyTorch-based.

---

## Running The Notebooks

From the repository root, install dependencies and open the notebook variant you
want to run:

```bash
pip install -r requirements.txt
jupyter notebook full_pipeline_miewid.ipynb
```

Use one of:

```bash
jupyter notebook full_pipeline_miewid.ipynb
jupyter notebook full_pipeline_sphereface.ipynb
jupyter notebook full_pipeline_arcface.ipynb
```

---

## Path Configuration

In the selected notebook, update these paths for your environment:

```python
root = "/path/to/animalclef26"
sample_sub_path = "/path/to/sample_submission.csv"
```

For SphereFace and ArcFace runs, also update the checkpoint path:

```python
sphereface_version = "/path/to/sphereface_checkpoint.pt"
arcface_version = "/path/to/arcface_model_best.bin"
```

For Kaggle, the notebooks expect paths similar to:

```python
root = "/kaggle/input/datasets/seifelkerdany/animalclef26"

sample_sub_path = (
    "/kaggle/input/competitions/animal-clef-2026/"
    "sample_submission.csv"
)
```

---

## Key Configuration

The notebooks store per-dataset parameters in `best_configs`, including:

| Key | Meaning |
| --- | --- |
| `top_r` | Number of best train-image matches averaged per known identity |
| `tau_attach` | Minimum best identity score needed to attach to a known identity |
| `tau_margin` | Minimum margin over the second-best identity |
| `threshold` | Similarity threshold used to build the clustering graph |
| `top_k` | Neighbor count for mutual-kNN graph construction |
| `iterations` | Maximum Chinese Whispers iterations |
| `k_core_degree` | k-core degree for graph backbone extraction |
| `use_rerank` | Whether to apply k-reciprocal re-ranking |
| `rr_k1`, `rr_k2`, `rr_lambda` | Re-ranking hyperparameters |

These values control the main quality/speed tradeoffs of the pipeline.

---

## Notes

- Run notebooks from the repository root so `src.*` imports resolve cleanly.
- Calibration can be expensive because it loads the MiewID branch,
  ALIKED, and DISK.
- The calibration subset size in the notebooks is capped for speed.
- The current implementation is notebook-first;
  there is no maintained CLI entry point in this `src` layout.
- Segmentation must be prepared before running the notebooks.
- The active in-repo preprocessing stage is the species-specific transform
  in `src/data`.
