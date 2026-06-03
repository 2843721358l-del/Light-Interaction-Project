#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/apply_patch.sh /path/to/HY-WorldPlay"
  exit 1
fi

TARGET_ROOT="$1"
PATCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -d "$TARGET_ROOT" ]; then
  echo "Target directory does not exist: $TARGET_ROOT"
  exit 1
fi

cp -R "$PATCH_ROOT/files/." "$TARGET_ROOT/"

patch --forward -d "$TARGET_ROOT" -p1 < "$PATCH_ROOT/patches/0001-integrate-acceleration.patch"

echo "Applied HY-WorldPlay acceleration patch to: $TARGET_ROOT"
