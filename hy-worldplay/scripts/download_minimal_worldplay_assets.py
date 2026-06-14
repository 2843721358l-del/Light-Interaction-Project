#!/usr/bin/env python3
# Copyright (c) 2026 Jiacheng Lu and contributors.
# Licensed under the MIT License.
"""Download only the model assets needed by the Light Interaction release.

The upstream HY-WorldPlay downloader fetches several action-model variants.
For this repository, inference uses the few-step distilled autoregressive
action checkpoint only.
"""

import argparse
import os
import shlex
import sys
from pathlib import Path


HUNYUAN_ALLOW_PATTERNS = [
    "vae/*",
    "scheduler/*",
    "transformer/480p_i2v/*",
    "text_encoder/llm/*",
    "text_encoder/byt5-small/*",
    "text_encoder/Glyph-SDXL-v2/*",
    "vision_encoder/siglip/*",
]

WORLDPLAY_ALLOW_PATTERNS = [
    "ar_distilled_action_model/*",
]

REQUIRED_HUNYUAN_DIRS = [
    "vae",
    "scheduler",
    "transformer/480p_i2v",
    "text_encoder/llm",
    "text_encoder/byt5-small",
    "text_encoder/Glyph-SDXL-v2",
    "vision_encoder/siglip",
]


def snapshot_download(repo_id, allow_patterns, token, local_dir):
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download
    except ImportError:
        print("Error: huggingface_hub is not installed. Run setup_worldplay_env.sh first.")
        return None

    kwargs = {
        "repo_id": repo_id,
        "allow_patterns": allow_patterns,
        "token": token,
    }
    if local_dir:
        kwargs["local_dir"] = str(local_dir)
    return Path(hf_snapshot_download(**kwargs)).resolve()


def find_distilled_checkpoint(worldplay_path):
    candidates = [
        worldplay_path / "ar_distilled_action_model" / "diffusion_pytorch_model.safetensors",
        worldplay_path / "ar_distilled_action_model" / "model.safetensors",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return candidates[0]


def validate_hunyuan_path(path):
    missing = [rel_path for rel_path in REQUIRED_HUNYUAN_DIRS if not (path / rel_path).is_dir()]
    if missing:
        print("\nError: HunyuanVideo-1.5 download is incomplete. Missing:")
        for rel_path in missing:
            print(f"  - {path / rel_path}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--hunyuan-repo", default="tencent/HunyuanVideo-1.5")
    parser.add_argument("--worldplay-repo", default="tencent/HY-WorldPlay")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional local directory for downloaded assets. By default, uses the Hugging Face cache.",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Optional shell file to write HY_MODEL_PATH and HY_AR_DISTILL_ACTION_MODEL_PATH exports.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve() if args.output_root else None
    hunyuan_local_dir = output_root / "HunyuanVideo-1.5" if output_root else None
    worldplay_local_dir = output_root / "HY-WorldPlay" if output_root else None

    print("Downloading HunyuanVideo-1.5 runtime assets ...")
    hunyuan_path = snapshot_download(
        args.hunyuan_repo,
        HUNYUAN_ALLOW_PATTERNS,
        args.hf_token,
        hunyuan_local_dir,
    )
    if hunyuan_path is None:
        return 1
    if not validate_hunyuan_path(hunyuan_path):
        return 1

    print("\nDownloading HY-WorldPlay distilled AR action checkpoint ...")
    worldplay_path = snapshot_download(
        args.worldplay_repo,
        WORLDPLAY_ALLOW_PATTERNS,
        args.hf_token,
        worldplay_local_dir,
    )
    if worldplay_path is None:
        return 1

    action_ckpt = find_distilled_checkpoint(worldplay_path)
    if not action_ckpt.exists():
        print("\nError: distilled AR action checkpoint was not found after download.")
        print("Expected one of:")
        print(f"  - {worldplay_path / 'ar_distilled_action_model' / 'diffusion_pytorch_model.safetensors'}")
        print(f"  - {worldplay_path / 'ar_distilled_action_model' / 'model.safetensors'}")
        return 1

    print("\nMinimal WorldPlay assets are ready.")
    print("This script intentionally downloads only:")
    print("  - HunyuanVideo-1.5 480P-I2V runtime assets")
    print("  - HY-WorldPlay ar_distilled_action_model")
    print("\nIt does not download the bidirectional model, the multi-step AR model, or RL variants.")
    print("\nUse these paths:")
    print(f"  export HY_MODEL_PATH={hunyuan_path}")
    print(f"  export HY_AR_DISTILL_ACTION_MODEL_PATH={action_ckpt}")
    if args.env_file:
        env_file = Path(args.env_file)
        with env_file.open("w", encoding="utf-8") as f:
            f.write(f"export HY_MODEL_PATH={shlex.quote(str(hunyuan_path))}\n")
            f.write(f"export HY_AR_DISTILL_ACTION_MODEL_PATH={shlex.quote(str(action_ckpt))}\n")
        print(f"\nWrote environment exports to: {env_file}")
    print("\nThen verify:")
    print("  python hy-worldplay/scripts/check_worldplay_assets.py \\")
    print("    --worldplay-root /path/to/HY-WorldPlay \\")
    print(f"    --model-path {hunyuan_path} \\")
    print(f"    --action-ckpt {action_ckpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
