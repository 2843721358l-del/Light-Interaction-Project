#!/usr/bin/env python3
# Copyright (c) 2026 Jiacheng Lu and contributors.
# Licensed under the MIT License.
"""Check whether the Light Interaction evaluation environment is usable."""

import importlib
import os
import sys
import tempfile


os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "light-interaction-matplotlib")
)


REQUIRED_MODULES = [
    ("cv2", "opencv-python"),
    ("lpips", "lpips"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("PIL", "pillow"),
    ("skimage", "scikit-image"),
    ("torch", "torch"),
    ("torchmetrics", "torchmetrics"),
    ("torchvision", "torchvision"),
    ("tqdm", "tqdm"),
    ("vbench", "vbench"),
]


def main():
    missing = []
    for module_name, package_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            missing.append((package_name, exc))
            print(f"[missing] {package_name}: {exc}")
            continue

        version = getattr(module, "__version__", "unknown")
        print(f"[ok] {package_name}: {version}")

    if missing:
        print("\nEnvironment check failed. Install missing packages or rerun:")
        print("  bash evaluation/scripts/setup_evaluation_env.sh")
        return 1

    print("\nEvaluation environment is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
