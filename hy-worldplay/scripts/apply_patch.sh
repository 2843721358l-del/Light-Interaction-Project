#!/usr/bin/env bash
# Copyright (c) 2026 Jiacheng Lu and contributors.
# Licensed under the MIT License.
# See the LICENSE file in the repository root for details.

# Apply the Light Interaction acceleration patch to a HY-WorldPlay checkout.
#
# Usage:
#   bash scripts/apply_patch.sh /path/to/HY-WorldPlay
#
# Prerequisites:
#   - HY-WorldPlay cloned from https://github.com/Tencent-Hunyuan/HY-WorldPlay
#   - ``patch`` utility installed (standard on Linux/macOS)
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/apply_patch.sh /path/to/HY-WorldPlay"
  exit 1
fi

TARGET_ROOT="$1"
PATCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_FILE="$PATCH_ROOT/patches/0001-integrate-acceleration.patch"
EXPECTED_COMMIT="1588e1336e842b03b0a7860c654ebd7c46bb065e"

warn_if_commit_mismatch() {
  local current_commit
  current_commit="$(git -C "$TARGET_ROOT" rev-parse HEAD 2>/dev/null || true)"
  if [ -n "$current_commit" ] && [[ "$current_commit" != "$EXPECTED_COMMIT"* ]]; then
    echo "Warning: this patch was tested on upstream commit $EXPECTED_COMMIT, but the target checkout is $current_commit." >&2
    echo "The patch may still work, but failures are more likely if upstream changed." >&2
  fi
}

if [ ! -d "$TARGET_ROOT" ]; then
  echo "Error: target directory does not exist: $TARGET_ROOT"
  exit 1
fi

if [ ! -f "$TARGET_ROOT/hyvideo/generate.py" ] || [ ! -f "$TARGET_ROOT/run.sh" ]; then
  echo "Error: target does not look like a HY-WorldPlay checkout: $TARGET_ROOT"
  exit 1
fi

if ! command -v patch >/dev/null 2>&1; then
  echo "Error: patch utility is required but was not found."
  exit 1
fi

if [ ! -f "$PATCH_FILE" ]; then
  echo "Error: patch file does not exist: $PATCH_FILE"
  exit 1
fi

warn_if_commit_mismatch

echo "Checking model-specific patch file ..."
if patch --dry-run --forward -d "$TARGET_ROOT" -p1 < "$PATCH_FILE" >/dev/null; then
  echo "Copying acceleration modules ..."
  cp -R "$PATCH_ROOT/files/." "$TARGET_ROOT/"

  echo "Applying Light Interaction patch file ..."
  patch --forward -d "$TARGET_ROOT" -p1 < "$PATCH_FILE"

  echo ""
  echo "Patch applied successfully to: $TARGET_ROOT"
elif patch --dry-run --reverse -d "$TARGET_ROOT" -p1 < "$PATCH_FILE" >/dev/null; then
  echo "Patch already appears to be applied. Syncing Light Interaction files ..."
  cp -R "$PATCH_ROOT/files/." "$TARGET_ROOT/"
  echo "Patch files synchronized successfully to: $TARGET_ROOT"
else
  echo "Error: patch cannot be applied." >&2
  echo "Target path: $TARGET_ROOT" >&2
  echo "Tested upstream commit: $EXPECTED_COMMIT" >&2
  echo "Check that the target checkout matches the documented upstream commit." >&2
  echo "Patch file: $PATCH_FILE" >&2
  exit 1
fi

echo ""
echo "Next: set HY_MODEL_PATH and HY_AR_DISTILL_ACTION_MODEL_PATH, then run:"
echo "  cd \"$TARGET_ROOT\""
echo "  bash run.sh"
