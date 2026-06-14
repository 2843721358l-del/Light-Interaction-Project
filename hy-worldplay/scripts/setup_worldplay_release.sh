#!/usr/bin/env bash
# Prepare a patched HY-WorldPlay checkout for Light Interaction inference.
#
# Usage:
#   bash hy-worldplay/scripts/setup_worldplay_release.sh /path/to/HY-WorldPlay
#
# Optional environment variables:
#   WORLDPLAY_ENV_BACKEND=conda|venv|current
#   WORLDPLAY_ENV_NAME=light-interaction-worldplay
#   WORLDPLAY_ASSET_OUTPUT_ROOT=/path/to/light-interaction-models
#   PREPARE_WORLDPLAY_ASSETS=0
#   REQUIRE_CUDA=0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORLDPLAY_ROOT="${1:-${HY_WORLDPLAY_ROOT:-${WORLDPLAY_ROOT:-}}}"

ENV_BACKEND="${WORLDPLAY_ENV_BACKEND:-conda}"
ENV_NAME="${WORLDPLAY_ENV_NAME:-light-interaction-worldplay}"
PREPARE_WORLDPLAY_ASSETS="${PREPARE_WORLDPLAY_ASSETS:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
EXPECTED_COMMIT="1588e1336e842b03b0a7860c654ebd7c46bb065e"

if [ -z "$WORLDPLAY_ROOT" ]; then
  echo "Usage: bash hy-worldplay/scripts/setup_worldplay_release.sh /path/to/HY-WorldPlay"
  exit 1
fi

WORLDPLAY_ROOT="$(cd "$WORLDPLAY_ROOT" && pwd)"

CURRENT_COMMIT="$(git -C "$WORLDPLAY_ROOT" rev-parse HEAD 2>/dev/null || true)"
if [ -n "$CURRENT_COMMIT" ] && [[ "$CURRENT_COMMIT" != "$EXPECTED_COMMIT"* ]]; then
  echo "Warning: this patch was tested on upstream commit $EXPECTED_COMMIT, but the target checkout is $CURRENT_COMMIT." >&2
  echo "The patch may still work, but failures are more likely if upstream changed." >&2
fi

bash "$SCRIPT_DIR/apply_patch.sh" "$WORLDPLAY_ROOT"

bash "$SCRIPT_DIR/setup_worldplay_env.sh" "$WORLDPLAY_ROOT"

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
    echo "Error: unknown WORLDPLAY_ENV_BACKEND=$ENV_BACKEND. Use conda, venv, or current."
    exit 1
    ;;
esac

CUDA_FLAG=()
if [ "$REQUIRE_CUDA" = "1" ]; then
  CUDA_FLAG=(--require-cuda)
fi

python "$SCRIPT_DIR/check_worldplay_env.py" \
  --worldplay-root "$WORLDPLAY_ROOT" \
  "${CUDA_FLAG[@]}"

if [ "$PREPARE_WORLDPLAY_ASSETS" = "1" ]; then
  ASSET_ENV_FILE="$WORLDPLAY_ROOT/light_interaction_env.sh"

  DOWNLOAD_ARGS=(--env-file "$ASSET_ENV_FILE")
  if [ -n "${WORLDPLAY_ASSET_OUTPUT_ROOT:-}" ]; then
    DOWNLOAD_ARGS+=(--output-root "$WORLDPLAY_ASSET_OUTPUT_ROOT")
  fi

  python "$SCRIPT_DIR/download_minimal_worldplay_assets.py" "${DOWNLOAD_ARGS[@]}"
  # shellcheck disable=SC1090
  source "$ASSET_ENV_FILE"

  python "$SCRIPT_DIR/check_worldplay_assets.py" \
    --worldplay-root "$WORLDPLAY_ROOT" \
    --model-path "$HY_MODEL_PATH" \
    --action-ckpt "$HY_AR_DISTILL_ACTION_MODEL_PATH"

  RUN_HELPER="$WORLDPLAY_ROOT/run_light_interaction.sh"
  cat > "$RUN_HELPER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/light_interaction_env.sh"
export HY_N_INFERENCE_GPU="${HY_N_INFERENCE_GPU:-1}"
exec bash "$SCRIPT_DIR/run.sh"
EOF
  chmod +x "$RUN_HELPER"
else
  echo "Skipped model asset download. Run download_minimal_worldplay_assets.py before inference."
fi

echo ""
echo "Light Interaction HY-WorldPlay setup is ready."
echo "Activate with:"
if [ "$ENV_BACKEND" = "conda" ]; then
  echo "  conda activate $ENV_NAME"
elif [ "$ENV_BACKEND" = "venv" ]; then
  echo "  source \"$VENV_DIR/bin/activate\""
else
  echo "  already using current environment"
fi
if [ "$PREPARE_WORLDPLAY_ASSETS" = "1" ]; then
  echo ""
  echo "Run inference with:"
  echo "  cd \"$WORLDPLAY_ROOT\""
  echo "  bash run_light_interaction.sh"
fi
