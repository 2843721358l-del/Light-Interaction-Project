#!/usr/bin/env bash
# Create a fresh runtime environment for a patched Matrix-Game-3 checkout.
#
# Usage:
#   bash matrix-game/scripts/setup_matrix_env.sh /path/to/Matrix-Game/Matrix-Game-3
#
# Optional environment variables:
#   MATRIX_ENV_NAME=light-interaction-matrix
#   MATRIX_ENV_BACKEND=conda|venv|current
#   PYTHON_VERSION=3.12
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MATRIX_ROOT="${1:-${MATRIX_ROOT:-}}"

ENV_NAME="${MATRIX_ENV_NAME:-light-interaction-matrix}"
ENV_BACKEND="${MATRIX_ENV_BACKEND:-conda}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if [ -z "$MATRIX_ROOT" ]; then
  echo "Error: missing Matrix-Game-3 checkout path."
  echo "Usage: bash matrix-game/scripts/setup_matrix_env.sh /path/to/Matrix-Game/Matrix-Game-3"
  exit 1
fi

MATRIX_ROOT="$(cd "$MATRIX_ROOT" && pwd)"
if [ ! -f "$MATRIX_ROOT/requirements.txt" ] || [ ! -f "$MATRIX_ROOT/generate.py" ]; then
  echo "Error: $MATRIX_ROOT does not look like a Matrix-Game-3 checkout."
  exit 1
fi

case "$ENV_BACKEND" in
  conda)
    if ! command -v conda >/dev/null 2>&1; then
      echo "Error: conda was not found. Install Miniconda/Anaconda first, or use MATRIX_ENV_BACKEND=venv."
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
    echo "Error: unknown MATRIX_ENV_BACKEND=$ENV_BACKEND. Use conda, venv, or current."
    exit 1
    ;;
esac

python -m pip install --upgrade pip wheel setuptools

echo "Installing Matrix-Game-3 dependencies from: $MATRIX_ROOT/requirements.txt"
python -m pip install -r "$MATRIX_ROOT/requirements.txt" --extra-index-url "$TORCH_INDEX_URL"

echo ""
echo "Running import check ..."
python "$REPO_ROOT/matrix-game/scripts/check_matrix_env.py" --matrix-root "$MATRIX_ROOT"

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
