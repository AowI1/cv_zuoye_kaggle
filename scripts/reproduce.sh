#!/bin/bash
# End-to-end reproduction of the AnimalCLEF26 pipeline.
# Run from the repository root:  bash scripts/reproduce.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Project venv if present, otherwise the active interpreter (override with PY=...).
if [ -z "${PY:-}" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
fi

echo "== 1/7 data (official competition mirror) =="
$PY scripts/download_data.py
if [ ! -d data/animalclef2026/images ]; then
  ( cd data/animalclef2026 && unzip -q train.zip && unzip -q test.zip )
fi

echo "== 2/7 model weights =="
$PY scripts/download_models.py

echo "== 3/7 embedded features (MPS/CUDA) =="
$PY scripts/extract_embeddings.py --backbone miewid --batch-size 16
$PY scripts/extract_embeddings.py --backbone mega --batch-size 12

echo "== 4/7 local-feature similarities (CPU) =="
for sp in SeaTurtleID2022 TexasHornedLizards LynxID2025 SalamanderID2025; do
  $PY scripts/extract_local.py --species "$sp" --split test --method root_sift
done

echo "== 5/7 per-species tuning =="
for sp in LynxID2025 SalamanderID2025 SeaTurtleID2022; do
  $PY scripts/tune.py --species "$sp" --seeds 4
  $PY scripts/tune_local.py --species "$sp" --seeds 2
done
$PY scripts/tune_lizard.py

echo "== 6/7 held-out validation =="
$PY scripts/validate_final.py --seeds 100 101 102 103

echo "== 7/7 submission =="
$PY scripts/make_submission.py
$PY scripts/ablation.py --species all --seeds 3

echo "done -> submissions/submission.csv"
