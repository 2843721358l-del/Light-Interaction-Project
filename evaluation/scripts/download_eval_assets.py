#!/usr/bin/env python3
# Copyright (c) 2026 Jiacheng Lu and contributors.
# Licensed under the MIT License.
# See the LICENSE file in the repository root for details.
"""Download and verify externally hosted evaluation assets."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "2843721358l/Light-Interaction-Eval-Assets"
DEFAULT_LOCAL_DIR = "evaluation/data"
DEFAULT_MANIFEST = "evaluation/data/eval_assets_manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--skip-verify", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path):
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def iter_manifest_files(manifest):
    files = manifest.get("files", [])
    if isinstance(files, dict):
        for rel_path, info in files.items():
            item = dict(info)
            item.setdefault("path", rel_path)
            yield item
    else:
        yield from files


def verify_required_layout(local_dir):
    errors = []
    prompt_json = local_dir / "refined_prompts_llava16.json"
    sampled_dir = local_dir / "sampled_200"
    if not prompt_json.is_file():
        errors.append(f"missing prompt JSON: {prompt_json}")
    if not sampled_dir.is_dir():
        errors.append(f"missing sampled image directory: {sampled_dir}")
    return errors


def verify_manifest(local_dir, manifest):
    errors = []
    for item in iter_manifest_files(manifest):
        rel_path = item.get("path")
        if not rel_path:
            continue
        path = local_dir / rel_path
        if not path.is_file():
            errors.append(f"missing manifest file: {rel_path}")
            continue

        expected_size = item.get("size")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            errors.append(f"size mismatch for {rel_path}")

        expected_sha = item.get("sha256")
        if expected_sha and sha256_file(path) != expected_sha:
            errors.append(f"sha256 mismatch for {rel_path}")
    return errors


def main():
    args = parse_args()
    local_dir = Path(args.local_dir)
    manifest_path = Path(args.manifest)
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            local_dir=str(local_dir),
            token=args.hf_token,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        print(f"Error: failed to download evaluation assets from {args.repo_id}: {exc}", file=sys.stderr)
        return 1

    errors = verify_required_layout(local_dir)
    if not args.skip_verify:
        manifest = load_manifest(manifest_path)
        if manifest is None:
            errors.append(f"missing manifest: {manifest_path}")
        elif manifest.get("external_assets", False) and not list(iter_manifest_files(manifest)):
            print("Warning: manifest has no file checksums; only layout verification was performed.")
        else:
            errors.extend(verify_manifest(local_dir, manifest))

    if errors:
        print("Error: evaluation asset verification failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"Evaluation assets are ready in: {local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
