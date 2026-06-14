#!/usr/bin/env python3
# Copyright (c) 2026 Jiacheng Lu and contributors.
# Licensed under the MIT License.
"""Validate local model files needed by patched Matrix-Game-3.0 inference."""

import argparse
from pathlib import Path


REQUIRED_FILES = [
    "base_distilled_model/config.json",
    "base_distilled_model/diffusion_pytorch_model.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.2_VAE.pth",
    "MG-LightVAE.pth",
    "MG-LightVAE_v2.pth",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True, help="Matrix-Game-3.0 checkpoint directory.")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    ok = True
    print(f"Checking Matrix-Game-3.0 assets: {ckpt_dir}")
    for rel_path in REQUIRED_FILES:
        path = ckpt_dir / rel_path
        if path.exists():
            print(f"[ok] {rel_path}")
        else:
            print(f"[missing] {path}")
            ok = False

    if not ok:
        print("\nMatrix-Game-3.0 assets are incomplete.")
        return 1

    print("\nMatrix-Game-3.0 assets are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
