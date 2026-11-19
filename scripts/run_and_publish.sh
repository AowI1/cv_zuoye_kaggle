#!/bin/bash
# Regenerate the variant sweep from cache, then commit and push the results.
# Usage: bash scripts/run_and_publish.sh ["commit message"]
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${PY:-}" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
fi

MSG=${1:-"Regenerate variant sweep"}

"$PY" scripts/sweep_variants.py

git add submissions/variants scripts/sweep_variants.py scripts/run_and_publish.sh
if git diff --cached --quiet; then
  echo "no changes to publish"
  exit 0
fi
git commit -q -F - <<EOF
$MSG

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git push origin main
echo "pushed $(git log --oneline -1)"
