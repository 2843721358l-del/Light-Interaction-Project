#!/usr/bin/env python3
# Copyright (c) 2026 Jiacheng Lu and contributors.
# Licensed under the MIT License.
"""Validate local model paths needed by the patched HY-WorldPlay run script."""

import argparse
import os
import sys


REQUIRED_MODEL_DIRS = [
    "vae",
    "scheduler",
    "transformer/480p_i2v",
    "text_encoder/llm",
    "text_encoder/byt5-small",
    "text_encoder/Glyph-SDXL-v2",
    "vision_encoder/siglip",
]

REQUIRED_MODEL_FILES = [
    "text_encoder/llm/config.json",
    "text_encoder/byt5-small/config.json",
]


def check_path(path, kind):
    exists = os.path.isdir(path) if kind == "dir" else os.path.isfile(path)
    label = "ok" if exists else "missing"
    print(f"[{label}] {path}")
    return exists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worldplay-root",
        help="Optional patched HY-WorldPlay checkout; checks run.sh and demo image.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="HunyuanVideo-1.5 model root used as HY_MODEL_PATH.",
    )
    parser.add_argument(
        "--action-ckpt",
        required=True,
        help="HY-WorldPlay action checkpoint used as HY_AR_DISTILL_ACTION_MODEL_PATH.",
    )
    args = parser.parse_args()

    ok = True

    if args.worldplay_root:
        root = os.path.abspath(args.worldplay_root)
        ok = check_path(os.path.join(root, "run.sh"), "file") and ok
        ok = check_path(os.path.join(root, "hyvideo"), "dir") and ok
        ok = check_path(os.path.join(root, "assets/img/test.png"), "file") and ok

    model_path = os.path.abspath(args.model_path)
    ok = check_path(model_path, "dir") and ok
    for rel_path in REQUIRED_MODEL_DIRS:
        ok = check_path(os.path.join(model_path, rel_path), "dir") and ok
    for rel_path in REQUIRED_MODEL_FILES:
        ok = check_path(os.path.join(model_path, rel_path), "file") and ok

    action_ckpt = os.path.abspath(args.action_ckpt)
    ok = check_path(action_ckpt, "file") and ok
    if not action_ckpt.endswith(".safetensors"):
        print("[missing] action checkpoint should be a .safetensors file")
        ok = False

    if not ok:
        print("\nModel asset check failed. Download models with upstream download_models.py, then rerun this check.")
        return 1

    print("\nWorldPlay model assets are ready.")
    print("Use these paths before running patched HY-WorldPlay:")
    print(f"  export HY_MODEL_PATH={model_path}")
    print(f"  export HY_AR_DISTILL_ACTION_MODEL_PATH={action_ckpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
