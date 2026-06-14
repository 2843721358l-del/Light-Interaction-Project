#!/usr/bin/env bash
# Copyright (c) 2026 Jiacheng Lu and contributors.
# Licensed under the MIT License.
# See the LICENSE file in the repository root for details.

# Apply the Light Interaction acceleration patch to a Matrix-Game-3 checkout.
#
# Usage:
#   bash matrix-game-3.0/scripts/apply_patch.sh /path/to/Matrix-Game/Matrix-Game-3
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash matrix-game-3.0/scripts/apply_patch.sh /path/to/Matrix-Game/Matrix-Game-3"
  exit 1
fi

TARGET_ROOT="$1"
PATCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -d "$TARGET_ROOT" ]; then
  echo "Error: target directory does not exist: $TARGET_ROOT"
  exit 1
fi

if [ ! -f "$TARGET_ROOT/generate.py" ] || [ ! -d "$TARGET_ROOT/wan/modules" ]; then
  echo "Error: target does not look like a Matrix-Game-3 checkout: $TARGET_ROOT"
  exit 1
fi

if ! command -v patch >/dev/null 2>&1; then
  echo "Error: patch utility is required but was not found."
  exit 1
fi

echo "Checking integration patch ..."
patch --dry-run --forward -d "$TARGET_ROOT" -p1 < "$PATCH_ROOT/patches/0001-integrate-acceleration.patch" >/dev/null

echo "Copying acceleration modules and helper scripts ..."
cp -R "$PATCH_ROOT/files/." "$TARGET_ROOT/"

echo "Applying integration patch ..."
patch --forward -d "$TARGET_ROOT" -p1 < "$PATCH_ROOT/patches/0001-integrate-acceleration.patch"

echo ""
echo "Patch applied successfully to: $TARGET_ROOT"
echo "Next: set MG_CKPT_DIR, then run:"
echo "  cd \"$TARGET_ROOT\""
echo "  bash scripts/run_accel_preset.sh all"
