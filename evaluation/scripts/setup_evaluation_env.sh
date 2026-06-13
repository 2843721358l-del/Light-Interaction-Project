#!/usr/bin/env bash
# Create a standalone environment for all evaluation scripts.
#
# Usage:
#   bash evaluation/scripts/setup_evaluation_env.sh
#
# Optional environment variables:
#   EVAL_ENV_NAME=light-interaction-eval
#   EVAL_ENV_BACKEND=conda
#   VENV_DIR=/tmp/light-interaction-eval
#   PYTHON_VERSION=3.10
#   PREPARE_EVAL_ASSETS=0
#   EVAL_ASSET_CACHE_DIR=$HOME/.cache/light-interaction/vbench
#   TORCH_HOME=$HOME/.cache/light-interaction/torch
#   HF_ENDPOINT=https://hf-mirror.com
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
#     Use https://download.pytorch.org/whl/cu118 for CUDA 11.8,
#     or https://download.pytorch.org/whl/cpu for CPU-only checks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_NAME="${EVAL_ENV_NAME:-light-interaction-eval}"
ENV_BACKEND="${EVAL_ENV_BACKEND:-conda}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PREPARE_EVAL_ASSETS="${PREPARE_EVAL_ASSETS:-1}"
EVAL_ASSET_CACHE_DIR="${EVAL_ASSET_CACHE_DIR:-$HOME/.cache/light-interaction/vbench}"
TORCH_HOME="${TORCH_HOME:-$HOME/.cache/light-interaction/torch}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
export TORCH_HOME

case "$ENV_BACKEND" in
  conda)
    if ! command -v conda >/dev/null 2>&1; then
      echo "Error: conda was not found. Install Miniconda/Anaconda first, or use EVAL_ENV_BACKEND=venv."
      exit 1
    fi

    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
      echo "Using existing conda environment: $ENV_NAME"
    else
      echo "Creating conda environment: $ENV_NAME"
      conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
    fi

    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    ;;
  venv)
    VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-$ENV_NAME}"
    if [ -d "$VENV_DIR" ]; then
      echo "Using existing venv: $VENV_DIR"
    else
      echo "Creating venv: $VENV_DIR"
      python -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    ;;
  current)
    echo "Installing into current Python environment: $(python -c 'import sys; print(sys.executable)')"
    ;;
  *)
    echo "Error: unknown EVAL_ENV_BACKEND=$ENV_BACKEND. Use conda, venv, or current."
    exit 1
    ;;
esac

python -m pip install --upgrade pip "setuptools<81" wheel

echo "Installing PyTorch from: $TORCH_INDEX_URL"
python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"

echo "Installing evaluation dependencies ..."
python -m pip install -r "$REPO_ROOT/evaluation/requirements.txt"

echo ""
echo "Environment ready: $ENV_NAME ($ENV_BACKEND)"
echo "Running import check ..."
python "$REPO_ROOT/evaluation/scripts/check_evaluation_env.py"

if [ "$PREPARE_EVAL_ASSETS" = "1" ]; then
  echo ""
  echo "Preparing evaluation model assets ..."
  python "$REPO_ROOT/evaluation/scripts/prepare_evaluation_assets.py" \
    --cache-dir "$EVAL_ASSET_CACHE_DIR" \
    --torch-home "$TORCH_HOME"
else
  echo ""
  echo "Skipped evaluation asset preparation. Run this later if needed:"
  echo "  python evaluation/scripts/prepare_evaluation_assets.py"
fi

echo ""
echo "Activate with:"
if [ "$ENV_BACKEND" = "conda" ]; then
  echo "  conda activate $ENV_NAME"
elif [ "$ENV_BACKEND" = "venv" ]; then
  echo "  source \"$VENV_DIR/bin/activate\""
else
  echo "  already using current environment"
fi
