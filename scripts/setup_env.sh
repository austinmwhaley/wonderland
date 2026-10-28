#!/usr/bin/env bash
# Step 0 of the foundation rebuild (docs/SPINE.md §8, DEC-389): build the CUDA environment.
#
# Idempotent and safe to re-run. System python is never touched — everything lands in a project
# venv (gitignored). torch is installed FIRST from the CUDA-13 wheel index so requirements.txt can
# never resolve it to a CPU build.
#
#   VENV=.venv CUDA_INDEX=https://download.pytorch.org/whl/cu130 ./scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${VENV:-.venv}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu130}"
PY="$VENV/bin/python"

echo "[0/5] python: $(python3 -V)  arch: $(uname -m)"

if [ ! -x "$PY" ]; then
  echo "[1/5] creating venv at $VENV"
  python3 -m venv "$VENV"
else
  echo "[1/5] reusing venv at $VENV"
fi

echo "[2/5] upgrading pip toolchain"
"$PY" -m pip install --quiet --upgrade pip setuptools wheel

echo "[3/5] installing torch from $CUDA_INDEX"
"$PY" -m pip install --index-url "$CUDA_INDEX" torch

echo "[4/5] installing project requirements (+dev)"
"$PY" -m pip install -r requirements.txt -r requirements-dev.txt

echo "[5/5] installing cupy for the CUDA-13 runtime (optional; CPU fallback retained)"
"$PY" -m pip install cupy-cuda13x || echo "  cupy unavailable — GPU data-path rows render UNKNOWN"

echo
echo "Verifying and writing the env receipt..."
"$PY" scripts/env_receipt.py --print

cat <<'EOF'

Next (implementation order, docs/SPINE.md §8):
  .venv/bin/python -m pytest -m "not slow"          # baseline gate on the inherited pipeline
  .venv/bin/python scripts/report_card.py           # the status-quo card
Then step 1: the judge-first card skeleton (four legs x two numbers, UNKNOWN until receipts exist).
EOF
