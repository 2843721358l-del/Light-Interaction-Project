#!/usr/bin/env python3
# Copyright (c) 2026 Jiacheng Lu and contributors.
# Licensed under the MIT License.
"""Check whether a patched HY-WorldPlay runtime environment is usable."""

import argparse
import importlib
import os
import sys


REQUIRED_MODULES = [
    ("accelerate", "accelerate"),
    ("cloudpickle", "cloudpickle"),
    ("diffusers", "diffusers"),
    ("einops", "einops"),
    ("ftfy", "ftfy"),
    ("huggingface_hub", "huggingface-hub"),
    ("imageio", "imageio"),
    ("loguru", "loguru"),
    ("modelscope", "modelscope"),
    ("moviepy.editor", "moviepy"),
    ("numpy", "numpy"),
    ("omegaconf", "omegaconf"),
    ("openai", "openai"),
    ("pandas", "pandas"),
    ("peft", "peft"),
    ("PIL", "pillow"),
    ("qwen_vl_utils", "qwen-vl-utils"),
    ("remote_pdb", "remote-pdb"),
    ("safetensors", "safetensors"),
    ("scipy", "scipy"),
    ("tiktoken", "tiktoken"),
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("torchvision", "torchvision"),
    ("tqdm", "tqdm"),
    ("transformers", "transformers"),
    ("triton", "triton"),
]

PATCH_MODULES = [
    ("worldplay_acceleration_config", "Light Interaction config"),
    (
        "hyvideo.models.transformers.modules.longcat_kernel",
        "Light Interaction Triton kernel",
    ),
    (
        "hyvideo.models.transformers.modules.ar_sparse_operation",
        "AR sparse attention",
    ),
    (
        "hyvideo.models.transformers.modules.bi_sparse_operation_for_KV_cache",
        "prefill sparse attention",
    ),
]


def import_one(module_name, package_name):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        print(f"[missing] {package_name}: {exc}")
        return False

    version = getattr(module, "__version__", "unknown")
    print(f"[ok] {package_name}: {version}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worldplay-root",
        required=True,
        help="Path to a patched HY-WorldPlay checkout.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail when torch cannot see a CUDA device.",
    )
    args = parser.parse_args()

    worldplay_root = os.path.abspath(args.worldplay_root)
    if not os.path.isdir(os.path.join(worldplay_root, "hyvideo")):
        print(f"[missing] HY-WorldPlay source tree: {worldplay_root}")
        return 1

    sys.path.insert(0, worldplay_root)

    ok = True
    print("Checking Python packages ...")
    for module_name, package_name in REQUIRED_MODULES:
        ok = import_one(module_name, package_name) and ok

    print("\nChecking Light Interaction patch modules ...")
    for module_name, package_name in PATCH_MODULES:
        ok = import_one(module_name, package_name) and ok

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"\nCUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.get_device_name(0)}")
        elif args.require_cuda:
            print("[missing] CUDA device is required for HY-WorldPlay inference.")
            ok = False
    except Exception as exc:
        print(f"[missing] torch CUDA check failed: {exc}")
        ok = False

    if not ok:
        print("\nWorldPlay environment check failed. Rerun:")
        print("  bash hy-worldplay/scripts/setup_worldplay_env.sh /path/to/HY-WorldPlay")
        return 1

    print("\nWorldPlay environment is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
