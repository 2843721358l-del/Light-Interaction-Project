#!/usr/bin/env bash
# Prepare a patched Matrix-Game-3 checkout for Light Interaction inference.
#
# Usage:
#   bash matrix-game/scripts/setup_matrix_release.sh /path/to/Matrix-Game/Matrix-Game-3
#
# Optional environment variables:
#   MATRIX_ENV_BACKEND=conda|venv|current
#   MATRIX_ENV_NAME=light-interaction-matrix
#   MATRIX_ASSET_OUTPUT_ROOT=/path/to/light-interaction-models
#   PREPARE_MATRIX_ASSETS=0
#   REQUIRE_CUDA=0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MATRIX_ROOT="${1:-${MATRIX_ROOT:-}}"

ENV_BACKEND="${MATRIX_ENV_BACKEND:-conda}"
ENV_NAME="${MATRIX_ENV_NAME:-light-interaction-matrix}"
PREPARE_MATRIX_ASSETS="${PREPARE_MATRIX_ASSETS:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

if [ -z "$MATRIX_ROOT" ]; then
  echo "Usage: bash matrix-game/scripts/setup_matrix_release.sh /path/to/Matrix-Game/Matrix-Game-3"
  exit 1
fi

MATRIX_ROOT="$(cd "$MATRIX_ROOT" && pwd)"

if [ -f "$MATRIX_ROOT/wan/modules/bi_sparse_operation.py" ] \
  && grep -q -- "--acceleration_preset" "$MATRIX_ROOT/generate.py"; then
  echo "Patch already appears to be applied: $MATRIX_ROOT"
else
  bash "$SCRIPT_DIR/apply_patch.sh" "$MATRIX_ROOT"
fi

bash "$SCRIPT_DIR/setup_matrix_env.sh" "$MATRIX_ROOT"

case "$ENV_BACKEND" in
  conda)
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    ;;
  venv)
    VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-$ENV_NAME}"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    ;;
  current)
    ;;
  *)
    echo "Error: unknown MATRIX_ENV_BACKEND=$ENV_BACKEND. Use conda, venv, or current."
    exit 1
    ;;
esac

CUDA_FLAG=()
if [ "$REQUIRE_CUDA" = "1" ]; then
  CUDA_FLAG=(--require-cuda)
fi

python "$SCRIPT_DIR/check_matrix_env.py" \
  --matrix-root "$MATRIX_ROOT" \
  "${CUDA_FLAG[@]}"

if [ "$PREPARE_MATRIX_ASSETS" = "1" ]; then
  ASSET_ENV_FILE="$MATRIX_ROOT/light_interaction_env.sh"

  DOWNLOAD_ARGS=(--env-file "$ASSET_ENV_FILE")
  if [ -n "${MATRIX_ASSET_OUTPUT_ROOT:-}" ]; then
    DOWNLOAD_ARGS+=(--output-root "$MATRIX_ASSET_OUTPUT_ROOT")
  fi

  python "$SCRIPT_DIR/download_matrix_assets.py" "${DOWNLOAD_ARGS[@]}"
  # shellcheck disable=SC1090
  source "$ASSET_ENV_FILE"

  python "$SCRIPT_DIR/check_matrix_assets.py" --ckpt-dir "$MG_CKPT_DIR"

  RUN_HELPER="$MATRIX_ROOT/run_light_interaction.sh"
  cat > "$RUN_HELPER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/light_interaction_env.sh"
export MG_NUM_ITERATIONS="${MG_NUM_ITERATIONS:-8}"
export MG_PRESET="${MG_PRESET:-all}"
exec bash "$SCRIPT_DIR/scripts/run_accel_preset.sh" "$MG_PRESET"
EOF
  chmod +x "$RUN_HELPER"
else
  echo "Skipped model asset download. Set MG_CKPT_DIR before inference."
fi

echo ""
echo "Light Interaction Matrix-Game setup is ready."
echo "Activate with:"
if [ "$ENV_BACKEND" = "conda" ]; then
  echo "  conda activate $ENV_NAME"
elif [ "$ENV_BACKEND" = "venv" ]; then
  echo "  source \"$VENV_DIR/bin/activate\""
else
  echo "  already using current environment"
fi
if [ "$PREPARE_MATRIX_ASSETS" = "1" ]; then
  echo ""
  echo "Run inference with:"
  echo "  cd \"$MATRIX_ROOT\""
  echo "  bash run_light_interaction.sh"
fi
