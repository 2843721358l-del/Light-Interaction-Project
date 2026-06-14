#!/usr/bin/env python3
# Copyright (c) 2026 Jiacheng Lu and contributors.
# Licensed under the MIT License.
"""Check whether a patched Matrix-Game-3 runtime environment is usable."""

import argparse
import importlib
import os
import sys


REQUIRED_MODULES = [
    ("cv2", "opencv-python"),
    ("diffusers", "diffusers"),
    ("einops", "einops"),
    ("huggingface_hub", "huggingface-hub"),
    ("imageio", "imageio"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("PIL", "pillow"),
    ("scipy", "scipy"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("tqdm", "tqdm"),
    ("transformers", "transformers"),
    ("triton", "triton"),
]

PATCH_MODULES = [
    ("wan.modules.bi_sparse_operation", "Light Interaction sparse attention"),
    ("wan.modules.longcat_kernel", "Light Interaction Triton kernel"),
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
        "--matrix-root",
        required=True,
        help="Path to a patched Matrix-Game-3 checkout.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail when torch cannot see a CUDA device.",
    )
    args = parser.parse_args()

    matrix_root = os.path.abspath(args.matrix_root)
    if not os.path.isfile(os.path.join(matrix_root, "generate.py")):
        print(f"[missing] Matrix-Game-3 source tree: {matrix_root}")
        return 1

    sys.path.insert(0, matrix_root)

    ok = True
    print("Checking Python packages ...")
    for module_name, package_name in REQUIRED_MODULES:
        ok = import_one(module_name, package_name) and ok

    cuda_available = False
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"\nCUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.get_device_name(0)}")
        elif args.require_cuda:
            print("[missing] CUDA device is required for Matrix-Game-3 inference.")
            ok = False
    except Exception as exc:
        print(f"[missing] torch CUDA check failed: {exc}")
        ok = False

    if cuda_available or args.require_cuda:
        print("\nChecking Light Interaction patch modules ...")
        for module_name, package_name in PATCH_MODULES:
            ok = import_one(module_name, package_name) and ok
    else:
        print("\nSkipping Light Interaction sparse kernel import check because CUDA is unavailable.")

    if not ok:
        print("\nMatrix-Game environment check failed. Rerun:")
        print("  bash matrix-game/scripts/setup_matrix_env.sh /path/to/Matrix-Game/Matrix-Game-3")
        return 1

    print("\nMatrix-Game environment is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
