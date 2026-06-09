#!/usr/bin/env bash
# Copyright 2026 Jiacheng Lu and contributors
# SPDX-License-Identifier: Apache-2.0
#
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

if [ ! -d "$TARGET_ROOT" ]; then
  echo "Error: target directory does not exist: $TARGET_ROOT"
  exit 1
fi

echo "Copying acceleration modules ..."
cp -R "$PATCH_ROOT/files/." "$TARGET_ROOT/"

echo "Applying integration patch ..."
patch --forward -d "$TARGET_ROOT" -p1 < "$PATCH_ROOT/patches/0001-integrate-acceleration.patch"

echo ""
echo "Patch applied successfully to: $TARGET_ROOT"
echo "Next: set HY_MODEL_PATH and HY_AR_DISTILL_ACTION_MODEL_PATH, then run:"
