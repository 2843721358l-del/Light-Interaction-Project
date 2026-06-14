#!/usr/bin/env bash
# Run a Matrix-Game-3 Light Interaction acceleration preset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PRESET="${1:-${MG_PRESET:-off}}"
CKPT_DIR="${MG_CKPT_DIR:-$REPO_ROOT/Matrix-Game-3.0}"
OUTPUT_DIR="${MG_OUTPUT_DIR:-$REPO_ROOT/output}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${MG_NUM_GPUS:-1}"
MASTER_PORT="${MG_MASTER_PORT:-29500}"

SIZE="${MG_SIZE:-704*1280}"
NUM_ITERATIONS="${MG_NUM_ITERATIONS:-4}"
NUM_STEPS="${MG_NUM_INFERENCE_STEPS:-3}"
IMAGE="${MG_IMAGE:-$REPO_ROOT/demo_images/001/image.png}"
PROMPT="${MG_PROMPT:-A colorful, animated cityscape with a gas station and various buildings.}"
SAVE_NAME="${MG_SAVE_NAME:-matrix_game_${PRESET}}"

if [[ ! -d "$CKPT_DIR" ]]; then
  echo "Checkpoint directory not found: $CKPT_DIR" >&2
  echo "Set MG_CKPT_DIR to the local Matrix-Game-3.0 model directory." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

action_pairs=()
for ((i = 0; i < NUM_ITERATIONS; i++)); do
  if (( i < NUM_ITERATIONS / 2 )); then
    action_pairs+=("j:q")
  else
    action_pairs+=("l:q")
  fi
done
ACTION_SEQUENCE="${MG_ACTION_SEQUENCE:-$(IFS=,; echo "${action_pairs[*]}")}"

env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" torchrun \
  --nproc_per_node="$NUM_GPUS" \
  --master_port="$MASTER_PORT" \
  "$REPO_ROOT/generate.py" \
  --size "$SIZE" \
  --ckpt_dir "$CKPT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --fa_version "${MG_FA_VERSION:-0}" \
  --use_int8 \
  --num_iterations "$NUM_ITERATIONS" \
  --num_inference_steps "$NUM_STEPS" \
  --image "$IMAGE" \
  --action_sequence "$ACTION_SEQUENCE" \
  --prompt "$PROMPT" \
  --save_name "$SAVE_NAME" \
  --seed "${MG_SEED:-42}" \
  --interactive \
  --compile_vae \
  --lightvae_pruning_rate "${MG_LIGHTVAE_PRUNING_RATE:-0.5}" \
  --vae_type "${MG_VAE_TYPE:-mg_lightvae}" \
  --ulysses_size "$NUM_GPUS" \
  --acceleration_preset "$PRESET"
