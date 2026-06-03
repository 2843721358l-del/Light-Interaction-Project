#!/usr/bin/env python3
"""Run selected VBench dimensions on a batch of generated videos."""

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from vbench import VBench


DEFAULT_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-json", default="evaluation/data/refined_prompts_llava16.json")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--sandbox-root", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dimensions", nargs="*", default=DEFAULT_DIMENSIONS)
    return parser.parse_args()


def load_prompt_map(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {Path(item["filename"]).stem: item["refined_prompt"] for item in data}


def find_video(root, base_name):
    root = Path(root)
    candidates = [
        root / base_name / "gen.mp4",
        root / base_name.replace(" ", "_") / "gen.mp4",
        root / f"{base_name}.mp4",
        root / f"{base_name.replace(' ', '_')}.mp4",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def read_score(result_path, dimensions):
    if not result_path.exists():
        return {dim: None for dim in dimensions}
    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)
    row = {}
    for dim in dimensions:
        value = data.get(dim) if isinstance(data, dict) else None
        if isinstance(value, list):
            value = value[0]
        row[dim] = value
    return row


def evaluate_one(video_path, prompt, name, sandbox_root, dimensions, device):
    sandbox = sandbox_root / name.replace(" ", "_")
    sandbox.mkdir(parents=True, exist_ok=True)
    input_video = sandbox / "input.mp4"
    shutil.copy(video_path, input_video)

    info_path = sandbox / "info.json"
    metadata = [{
        "video_list": [input_video.name],
        "prompt_en": prompt,
        "dimension": dimensions,
    }]
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    bench = VBench(device=device, full_info_dir=str(info_path), output_path=str(sandbox))
    bench.evaluate(
        videos_path=str(sandbox),
        name="vbench",
        dimension_list=dimensions,
        mode="custom_input",
    )
    scores = read_score(sandbox / "vbench_eval_results.json", dimensions)
    del bench
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(sandbox, ignore_errors=True)
    return scores


def main():
    args = parse_args()
    prompt_map = load_prompt_map(args.prompt_json)
    sandbox_root = Path(args.sandbox_root or f"vbench_sandbox_{int(time.time())}")
    sandbox_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for base_name, prompt in tqdm(sorted(prompt_map.items()), desc="VBench"):
        video_path = find_video(args.video_dir, base_name)
        if not video_path:
            continue
        try:
            scores = evaluate_one(video_path, prompt, base_name, sandbox_root, args.dimensions, args.device)
        except Exception as exc:
            print(f"[warn] VBench failed for {base_name}: {exc}")
            scores = {dim: None for dim in args.dimensions}
        row = {"Video Name": base_name, **scores}
        valid = [v for v in scores.values() if v is not None]
        row["Video_Avg"] = float(np.mean(valid)) if valid else 0.0
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        avg = {"Video Name": "--- TOTAL AVERAGE ---"}
        for col in [c for c in df.columns if c != "Video Name"]:
            avg[col] = df[col].mean()
        pd.concat([df, pd.DataFrame([avg])], ignore_index=True).to_csv(args.output_csv, index=False)

    shutil.rmtree(sandbox_root, ignore_errors=True)


if __name__ == "__main__":
    main()
