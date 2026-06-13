#!/usr/bin/env python3
# Copyright (c) 2026 Jiacheng Lu and contributors.
# Licensed under the MIT License.
"""Download/check model assets used by the evaluation scripts."""

import argparse
import os
from pathlib import Path
import tempfile
from urllib.error import URLError
import warnings


DEFAULT_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
]

HF_AMT_URL = "https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=str(Path.home() / ".cache" / "light-interaction" / "vbench"),
        help="Directory for VBench metric checkpoints.",
    )
    parser.add_argument(
        "--torch-home",
        default=str(Path.home() / ".cache" / "light-interaction" / "torch"),
        help="Directory for Torch Hub / LPIPS checkpoints.",
    )
    parser.add_argument(
        "--dimensions",
        nargs="*",
        default=DEFAULT_DIMENSIONS,
        help="VBench dimensions to prepare.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT"),
        help="Optional Hugging Face endpoint mirror, for example https://hf-mirror.com.",
    )
    parser.add_argument(
        "--skip-vbench",
        action="store_true",
        help="Only prepare LPIPS / Torch assets.",
    )
    parser.add_argument(
        "--skip-lpips",
        action="store_true",
        help="Only prepare VBench assets.",
    )
    return parser.parse_args()


def prepare_lpips():
    import lpips

    print("[prepare] LPIPS alex checkpoint")
    lpips.LPIPS(net="alex", verbose=False)


def prepare_huggingface_assets(cache_dir, dimensions):
    if "motion_smoothness" not in dimensions:
        return

    target = cache_dir / "amt_model" / "amt-s.pth"
    if target.exists():
        return

    print("[prepare] VBench AMT checkpoint from Hugging Face")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="lalala125/AMT",
            filename="amt-s.pth",
            local_dir=str(target.parent),
        )
    except Exception as exc:
        print(f"[warn] huggingface_hub download failed: {exc}")
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        url = HF_AMT_URL.replace("https://huggingface.co", endpoint)
        try:
            download_url(url, target)
        except Exception as fallback_exc:
            raise RuntimeError(
                "Failed to download the VBench motion_smoothness checkpoint. "
                "If huggingface.co is blocked, rerun with "
                "`--hf-endpoint https://hf-mirror.com` or set HF_ENDPOINT."
            ) from fallback_exc

    if not target.exists():
        raise RuntimeError(f"Expected AMT checkpoint was not created: {target}")


def download_url(url, target):
    import requests

    print(f"[download] {url}")
    tmp_path = target.with_suffix(target.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=60, allow_redirects=True) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as out:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out.write(chunk)
        tmp_path.replace(target)
    except (OSError, URLError, requests.RequestException):
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def prepare_vbench(cache_dir, dimensions):
    from vbench import aesthetic_quality
    from vbench.utils import init_submodules

    print("[prepare] VBench checkpoints")
    prepare_huggingface_assets(cache_dir, dimensions)
    submodules = init_submodules(dimensions, local=True)
    if "aesthetic_quality" in dimensions:
        aesthetic_quality.get_aesthetic_model(submodules["aesthetic_quality"][1])


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    torch_home = Path(args.torch_home).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch_home.mkdir(parents=True, exist_ok=True)

    os.environ["VBENCH_CACHE_DIR"] = str(cache_dir)
    os.environ["TORCH_HOME"] = str(torch_home)
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ.setdefault(
        "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "light-interaction-matplotlib")
    )
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
    warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated.*")
    warnings.filterwarnings("ignore", message="Arguments other than a weight enum.*")
    warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`.*")

    print(f"[cache] VBENCH_CACHE_DIR={cache_dir}")
    print(f"[cache] TORCH_HOME={torch_home}")
    if args.hf_endpoint:
        print(f"[cache] HF_ENDPOINT={args.hf_endpoint}")

    if not args.skip_lpips:
        prepare_lpips()
    if not args.skip_vbench:
        prepare_vbench(cache_dir, args.dimensions)

    print("\nEvaluation assets are ready.")


if __name__ == "__main__":
    main()
