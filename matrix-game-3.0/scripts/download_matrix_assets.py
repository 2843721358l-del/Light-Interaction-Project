#!/usr/bin/env python3
# Copyright (c) 2026 Jiacheng Lu and contributors.
# Licensed under the MIT License.
"""Download Matrix-Game-3.0 model assets for Light Interaction inference."""

import argparse
import os
import shlex
import sys
from pathlib import Path


def snapshot_download(repo_id, token, local_dir):
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download
    except ImportError:
        print("Error: huggingface_hub is not installed. Run setup_matrix_env.sh first.")
        return None

    kwargs = {
        "repo_id": repo_id,
        "token": token,
    }
    if local_dir:
        kwargs["local_dir"] = str(local_dir)
    return Path(hf_snapshot_download(**kwargs)).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument(
        "--model-repo",
        default=os.environ.get("MATRIX_MODEL_REPO", "Skywork/Matrix-Game-3.0"),
        help="Hugging Face repo id for Matrix-Game-3.0 weights.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional local directory for downloaded assets. Defaults to Hugging Face cache.",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Optional shell file to write MG_CKPT_DIR export.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve() if args.output_root else None
    local_dir = output_root / "Matrix-Game-3.0" if output_root else None

    print(f"Downloading Matrix-Game-3.0 assets from: {args.model_repo}")
    ckpt_dir = snapshot_download(args.model_repo, args.hf_token, local_dir)
    if ckpt_dir is None:
        return 1

    print("\nMatrix-Game-3.0 assets are ready.")
    print(f"  export MG_CKPT_DIR={ckpt_dir}")
    if args.env_file:
        env_file = Path(args.env_file)
        with env_file.open("w", encoding="utf-8") as f:
            f.write(f"export MG_CKPT_DIR={shlex.quote(str(ckpt_dir))}\n")
        print(f"\nWrote environment export to: {env_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
