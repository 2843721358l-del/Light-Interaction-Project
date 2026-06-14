#!/usr/bin/env python3
# Copyright (c) 2026 Jiacheng Lu and contributors.
# Licensed under the MIT License.
# See the LICENSE file in the repository root for details.
"""Download and extract the selected 200 VBench-I2V 16:9 evaluation images.

The selected initial images are derived from the official VBench-I2V Image Suite.
This repository does not redistribute the image assets directly.

Usage:
    python evaluation/scripts/download_eval_assets.py
    python evaluation/scripts/download_eval_assets.py --vbench-root /path/to/VBench --skip-download
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


CROP_DIR_CANDIDATES = [
    "16-9",
    "16_9",
    "16:9",
    "16x9",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vbench-root",
        default=None,
        help="Path to an existing VBench repository with vbench2_beta_i2v/.",
    )
    parser.add_argument(
        "--vbench-repo",
        default="https://github.com/Vchitect/VBench.git",
        help="Official VBench repository URL.",
    )
    parser.add_argument(
        "--work-dir",
        default=".cache/light-interaction-vbench",
        help="Working directory for cloning VBench.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/data/sampled_200",
        help="Output directory for the 200 selected images.",
    )
    parser.add_argument(
        "--prompt-json",
        default="evaluation/data/refined_prompts_llava16.json",
        help="Path to the refined prompts JSON (for reference).",
    )
    parser.add_argument(
        "--selected-list",
        default="evaluation/data/selected_vbench_i2v_16_9_200.txt",
        help="Path to the selected filename list.",
    )
    parser.add_argument(
        "--manifest",
        default="evaluation/data/eval_assets_manifest.json",
        help="Path to the evaluation assets manifest for verification.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip running the official VBench-I2V download_data.sh.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip manifest-based file verification.",
    )
    parser.add_argument(
        "--keep-vbench",
        action="store_true",
        help="Keep the VBench clone after extraction (default behavior preserves it).",
    )
    parser.add_argument(
        "--target-ratio",
        default="16-9",
        help="Target crop ratio for selecting image source directory.",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_text_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").rstrip() for line in f]


def load_manifest(manifest_path):
    if not manifest_path.exists():
        return None
    return load_json(manifest_path)


def iter_manifest_files(manifest):
    files = manifest.get("files", [])
    if isinstance(files, dict):
        for rel_path, info in files.items():
            item = dict(info)
            item.setdefault("path", rel_path)
            yield item
    else:
        yield from files


def check_gdown():
    """Check if gdown is available. Returns True if found."""
    try:
        subprocess.run(
            [sys.executable, "-m", "gdown", "--help"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return shutil.which("gdown") is not None


def check_gdown_version_compat():
    """Check if this gdown version supports the --id flag used by VBench's script.

    Newer gdown (≥5.x) uses positional arguments instead of --id. Returns True
    if the version is compatible with the VBench download script's syntax.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "gdown", "--id", "--help"],
            capture_output=True,
            text=True,
        )
        # If --id is recognized, the help message mentions it; otherwise it errors.
        return "unrecognized" not in result.stderr
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def patch_vbench_download_script(script_path):
    """Create a patched version of download_data.sh for newer gdown syntax.

    Newer gdown versions (≥5.x) do not support the --id flag. This function
    reads the original script and rewrites gdown calls to use positional syntax.
    Returns the path to the patched script.
    """
    patched_path = script_path.with_suffix(".sh.patched")
    with open(script_path, encoding="utf-8") as f:
        content = f.read()

    # Rewrite: gdown --id <ID> --output <OUT>  ->  gdown <ID> -O <OUT>
    content = re.sub(
        r"gdown\s+--id\s+(\S+)\s+--output\s+(\S+)",
        r"gdown \1 -O \2",
        content,
    )

    with open(patched_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(patched_path, 0o755)
    return patched_path


def run_vbench_download(vbench_root):
    """Run the official VBench-I2V download_data.sh script."""
    script_path = vbench_root / "vbench2_beta_i2v" / "download_data.sh"
    if not script_path.exists():
        print(f"Error: VBench download script not found at {script_path}", file=sys.stderr)
        return False

    if not check_gdown():
        print(
            "Error: gdown is required to download official VBench-I2V assets.\n"
            "Install it with:\n  pip install gdown",
            file=sys.stderr,
        )
        return False

    # Check gdown version compatibility
    if not check_gdown_version_compat():
        print("Detected incompatible gdown version (--id flag not supported).")
        print("Patching download script for newer gdown syntax...")
        script_path = patch_vbench_download_script(script_path)
        print(f"Using patched script: {script_path}")

    print(f"Running: sh {script_path}")
    result = subprocess.run(
        ["sh", str(script_path)],
        cwd=vbench_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error: VBench download script failed with exit code {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return False
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            print(f"  [download] {line}")

    # Verify the data directory was populated
    data_dir = vbench_root / "vbench2_beta_i2v" / "data"
    crop_dir = data_dir / "crop"
    if not crop_dir.is_dir() or not any(crop_dir.iterdir()):
        print(
            f"Warning: crop directory not found or empty at {crop_dir}.\n"
            "The VBench-I2V download may have failed. Check your network connection and try again.",
            file=sys.stderr,
        )
        # Don't return False here - let the caller handle it with a clearer error
        print("VBench-I2V download completed (but data may be incomplete).")
        return True

    print("VBench-I2V download completed successfully.")
    return True


def locate_crop_dir(vbench_root, selected_names=None, target_ratio=None):
    """Locate the 16:9 crop directory within the VBench I2V assets.

    Prefers explicit candidate names, then falls back to scanning for the
    directory containing the most matched filenames from the selected list.
    """
    crop_base = vbench_root / "vbench2_beta_i2v" / "data" / "crop"

    if not crop_base.exists():
        print(f"Error: crop directory not found at {crop_base}", file=sys.stderr)
        return None

    # Step 1: try exact candidate names
    candidates = []
    if target_ratio:
        ratio_variants = [
            target_ratio,
            target_ratio.replace("-", "_"),
            target_ratio.replace("-", ":"),
            target_ratio.replace("-", "x"),
        ]
        candidates = list(dict.fromkeys(ratio_variants))  # dedup preserving order

    candidates = [crop_base / c for c in candidates if c]
    for c in CROP_DIR_CANDIDATES:
        candidates.append(crop_base / c)

    for full in candidates:
        if full.is_dir():
            print(f"Found crop directory: {full}")
            return full

    # Step 2: scan subdirectories and pick the one with most selected files
    subdirs = [d for d in crop_base.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"Error: no subdirectories found in {crop_base}", file=sys.stderr)
        return None

    selected_set = set(selected_names) if selected_names else set()
    best_dir = None
    best_count = 0
    for subdir in subdirs:
        available = {f.name for f in subdir.iterdir() if f.is_file()}
        count = len(available & selected_set)
        if count > best_count:
            best_count = count
            best_dir = subdir

    if best_dir is not None:
        print(f"Auto-selected crop directory (matched {best_count} / {len(selected_set)} files): {best_dir}")
        return best_dir

    print(f"Error: could not locate a suitable crop directory under {crop_base}", file=sys.stderr)
    return None


def copy_selected(crop_dir, output_dir, selected_names):
    """Copy selected files from crop_dir to output_dir.

    Handles filename normalization for apostrophes: some filenames in the
    selected list use '_s' as a substitute for "'s" due to JSON escaping.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = []

    for filename in selected_names:
        src = crop_dir / filename
        if not src.exists():
            # Try replacing _s with 's (common JSON apostrophe escaping)
            alt_name = filename.replace("_s", "'s")
            src = crop_dir / alt_name
        if not src.exists():
            missing.append(filename)
            continue
        dst = output_dir / filename
        shutil.copy2(src, dst)
        copied += 1

    return copied, missing


def verify_manifest(output_base, manifest_path):
    """Verify files against the manifest for size and sha256."""
    manifest = load_manifest(manifest_path)
    if manifest is None:
        print("Warning: manifest not found, skipping verification.", file=sys.stderr)
        return True

    errors = []
    for item in iter_manifest_files(manifest):
        rel_path = item.get("path")
        if not rel_path:
            continue
        # Resolve path relative to output_base or manifest parent
        candidate = output_base / rel_path
        if not candidate.is_file():
            # Try relative to manifest's parent
            candidate = manifest_path.parent / rel_path
        if not candidate.is_file():
            errors.append(f"missing manifest file: {rel_path}")
            continue

        expected_size = item.get("size")
        if expected_size is not None and candidate.stat().st_size != int(expected_size):
            errors.append(f"size mismatch for {rel_path}")

        expected_sha = item.get("sha256")
        if expected_sha and sha256_file(candidate) != expected_sha:
            errors.append(f"sha256 mismatch for {rel_path}")

    if errors:
        print("Error: asset verification failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return False

    print("Asset verification passed (size and sha256).")
    return True


def main():
    args = parse_args()
    selected_list_path = Path(args.selected_list)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    prompt_json_path = Path(args.prompt_json)

    # Validate selected list
    if not selected_list_path.exists():
        print(f"Error: selected list not found: {selected_list_path}", file=sys.stderr)
        return 1

    selected_names = load_text_lines(selected_list_path)
    if len(selected_names) != 200:
        print(
            f"Error: selected list has {len(selected_names)} lines, expected 200.",
            file=sys.stderr,
        )
        return 1

    # Validate prompt JSON as a sanity check
    if not prompt_json_path.exists():
        print(f"Warning: prompt JSON not found at {prompt_json_path}", file=sys.stderr)

    # Determine VBench root
    if args.vbench_root:
        vbench_root = Path(args.vbench_root).resolve()
        if not vbench_root.is_dir():
            print(f"Error: specified --vbench-root does not exist: {vbench_root}", file=sys.stderr)
            return 1
        print(f"Using existing VBench root: {vbench_root}")
    else:
        work_dir = Path(args.work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        vbench_root = work_dir / "VBench"

        if not vbench_root.is_dir():
            print(f"Cloning VBench from {args.vbench_repo} ...")
            result = subprocess.run(
                ["git", "clone", args.vbench_repo, str(vbench_root)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"Error: failed to clone VBench:\n{result.stderr}", file=sys.stderr)
                return 1
            print("VBench clone completed.")
        else:
            print(f"Using existing VBench clone: {vbench_root}")

    # Check vbench2_beta_i2v structure
    i2v_dir = vbench_root / "vbench2_beta_i2v"
    if not i2v_dir.is_dir():
        print(
            f"Error: vbench2_beta_i2v not found at {i2v_dir}. "
            "VBench repository structure may have changed.",
            file=sys.stderr,
        )
        return 1

    # Step 1: Download official VBench-I2V assets
    if not args.skip_download:
        print("Downloading official VBench-I2V assets...")
        if not run_vbench_download(vbench_root):
            return 1
    else:
        print("Skipping VBench-I2V download (--skip-download).")

    # Step 2: Locate 16:9 crop directory
    crop_dir = locate_crop_dir(vbench_root, selected_names, args.target_ratio)
    if crop_dir is None:
        return 1

    # Step 3: Copy selected files
    copied, missing = copy_selected(crop_dir, output_dir, selected_names)
    if missing:
        print(
            f"Error: {len(missing)} selected files not found in crop directory:",
            file=sys.stderr,
        )
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 1

    print(f"Copied {copied} selected VBench-I2V 16:9 images to {output_dir}")

    # Step 4: Optional manifest verification
    if not args.skip_verify:
        output_base = output_dir.parent  # e.g., evaluation/data/
        if not verify_manifest(output_base, manifest_path):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
