#!/usr/bin/env bash
# Create a fresh runtime environment for a patched HY-WorldPlay checkout.
#
# Usage:
#   bash hy-worldplay/scripts/setup_worldplay_env.sh /path/to/HY-WorldPlay
#
# Optional environment variables:
#   WORLDPLAY_ENV_NAME=light-interaction-worldplay
#   WORLDPLAY_ENV_BACKEND=conda
#   VENV_DIR=/tmp/light-interaction-worldplay
#   PYTHON_VERSION=3.10
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
#   TORCH_VERSION=2.6.0
#   TORCHVISION_VERSION=0.21.0
#   TORCHAUDIO_VERSION=2.6.0
#   HF_ENDPOINT=https://hf-mirror.com
#
# Use WORLDPLAY_ENV_BACKEND=venv when conda channels are unavailable.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORLDPLAY_ROOT="${1:-${HY_WORLDPLAY_ROOT:-${WORLDPLAY_ROOT:-}}}"

ENV_NAME="${WORLDPLAY_ENV_NAME:-light-interaction-worldplay}"
ENV_BACKEND="${WORLDPLAY_ENV_BACKEND:-conda}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.6.0}"

if [ -z "$WORLDPLAY_ROOT" ]; then
  echo "Error: missing HY-WorldPlay checkout path."
  echo "Usage: bash hy-worldplay/scripts/setup_worldplay_env.sh /path/to/HY-WorldPlay"
  exit 1
fi

WORLDPLAY_ROOT="$(cd "$WORLDPLAY_ROOT" && pwd)"
if [ ! -f "$WORLDPLAY_ROOT/requirements.txt" ] || [ ! -d "$WORLDPLAY_ROOT/hyvideo" ]; then
  echo "Error: $WORLDPLAY_ROOT does not look like a HY-WorldPlay checkout."
  exit 1
fi

case "$ENV_BACKEND" in
  conda)
    if ! command -v conda >/dev/null 2>&1; then
      echo "Error: conda was not found. Install Miniconda/Anaconda first, or use WORLDPLAY_ENV_BACKEND=venv."
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
    echo "Error: unknown WORLDPLAY_ENV_BACKEND=$ENV_BACKEND. Use conda, venv, or current."
    exit 1
    ;;
esac

python -m pip install --upgrade pip wheel setuptools

echo "Installing PyTorch from: $TORCH_INDEX_URL"
python -m pip install \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION" \
  --index-url "$TORCH_INDEX_URL"

CONSTRAINTS_FILE="$(mktemp)"
trap 'rm -f "$CONSTRAINTS_FILE"' EXIT
printf "torch==%s\ntorchvision==%s\ntorchaudio==%s\n" \
  "$TORCH_VERSION" "$TORCHVISION_VERSION" "$TORCHAUDIO_VERSION" > "$CONSTRAINTS_FILE"

echo "Installing HY-WorldPlay dependencies from: $WORLDPLAY_ROOT/requirements.txt"
python -m pip install -r "$WORLDPLAY_ROOT/requirements.txt" -c "$CONSTRAINTS_FILE"

echo ""
echo "Running import check ..."
python "$REPO_ROOT/hy-worldplay/scripts/check_worldplay_env.py" \
  --worldplay-root "$WORLDPLAY_ROOT"

echo ""
echo "Environment ready: $ENV_NAME ($ENV_BACKEND)"
echo "Activate with:"
if [ "$ENV_BACKEND" = "conda" ]; then
  echo "  conda activate $ENV_NAME"
elif [ "$ENV_BACKEND" = "venv" ]; then
  echo "  source \"$VENV_DIR/bin/activate\""
else
  echo "  already using current environment"
fi
