#!/usr/bin/env python3
# Copyright (c) 2026 Jiacheng Lu and contributors.
# Licensed under the MIT License.
# See the LICENSE file in the repository root for details.
"""Batch video generation for the fixed 200-prompt evaluation set."""

import argparse
import json
import logging
import multiprocessing as mp
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


DEFAULT_ACTIONS = {
    "left_right": "left-5, right-5.5",
    "forward_backward": "w-5, s-5.5",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-json", default="evaluation/data/refined_prompts_llava16.json")
    parser.add_argument("--hy-worldplay-root", required=True, help="Patched HY-WorldPlay checkout.")
    parser.add_argument("--model-path", required=True, help="HunyuanVideo-1.5 checkpoint directory.")
    parser.add_argument("--action-ckpt", required=True, help="HY-WorldPlay AR distilled action checkpoint.")
    parser.add_argument("--output-root", required=True, help="Directory where generated videos are written.")
    parser.add_argument("--torchrun", default="torchrun")
    parser.add_argument("--pythonpath", default=None, help="Optional PYTHONPATH override.")
    parser.add_argument("--actions", nargs="*", default=[f"{k}={v}" for k, v in DEFAULT_ACTIONS.items()])
    parser.add_argument("--allowed-gpus", default="", help="Comma-separated GPU ids. Empty means all visible GPUs.")
    parser.add_argument("--vram-threshold-mb", type=int, default=20)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=253)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--acceleration-preset", default="all", choices=["off", "context", "sparse", "cache", "all"])
    parser.add_argument("--skip-existing", action="store_true", default=True)
    return parser.parse_args()


def parse_actions(items):
    actions = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Action must use name=pose format: {item}")
        name, pose = item.split("=", 1)
        actions[name.strip()] = pose.strip()
    return actions


def free_gpus(threshold_mb, allowed):
    cmd = "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
    output = subprocess.check_output(cmd, shell=True, text=True).strip()
    available = []
    for line in output.splitlines():
        if not line.strip():
            continue
        gpu_id, used = [int(x.strip()) for x in line.split(",")]
        if allowed and gpu_id not in allowed:
            continue
        if used < threshold_mb:
            available.append(gpu_id)
    return available


def safe_stem(filename):
    return Path(filename).stem.replace(" ", "_")


def run_one(gpu_id, task, args, action_name, pose):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if args.pythonpath:
        env["PYTHONPATH"] = args.pythonpath

    prompt_path = Path(args.prompt_json).resolve()
    image_path = Path(prompt_path.parent.parent, task["image_path"]).resolve()
    out_dir = Path(args.output_root, action_name, safe_stem(task["filename"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.torchrun,
        "--nproc_per_node=1",
        f"--master_port={29540 + gpu_id}",
        "hyvideo/generate.py",
        "--prompt", task["refined_prompt"],
        "--image_path", str(image_path),
        "--resolution", "480p",
        "--aspect_ratio", "16:9",
        "--video_length", str(args.num_frames),
        "--seed", str(args.seed),
        "--rewrite", "false",
        "--sr", "false",
        "--save_pre_sr_video",
        "--output_path", str(out_dir),
        "--model_path", args.model_path,
        "--action_ckpt", args.action_ckpt,
        "--few_step", "true",
        "--pose", pose,
        "--num_inference_steps", str(args.num_steps),
        "--width", str(args.width),
        "--height", str(args.height),
        "--acceleration_preset", args.acceleration_preset,
        "--model_type", "ar",
        "--offloading", "false",
        "--group_offloading", "false",
    ]

    start = time.time()
    result = subprocess.run(cmd, cwd=args.hy_worldplay_root, env=env, text=True, capture_output=True)
    duration = time.time() - start
    if result.returncode == 0:
        logging.info("[GPU %s] done %s/%s in %.1fs", gpu_id, action_name, task["filename"], duration)
    else:
        logging.error("[%s] failed %s/%s\n%s", datetime.now(), action_name, task["filename"], result.stderr[-4000:])


def run_action(action_name, pose, tasks, args, allowed):
    pending = []
    for task in tasks:
        out_dir = Path(args.output_root, action_name, safe_stem(task["filename"]))
        if args.skip_existing and out_dir.exists() and any(out_dir.iterdir()):
            continue
        pending.append(task)

    print(f"[{action_name}] pending {len(pending)} / {len(tasks)}")
    active = {}
    while pending or active:
        finished = [gpu for gpu, proc in active.items() if not proc.is_alive()]
        for gpu in finished:
            active[gpu].join()
            del active[gpu]

        idle = [gpu for gpu in free_gpus(args.vram_threshold_mb, allowed) if gpu not in active]
        while idle and pending and len(active) < args.max_concurrent:
            gpu_id = idle.pop(0)
            task = pending.pop(0)
            proc = mp.Process(target=run_one, args=(gpu_id, task, args, action_name, pose))
            proc.start()
            active[gpu_id] = proc
            print(f"[dispatch] gpu={gpu_id} action={action_name} file={task['filename']}")
            if idle and pending:
                time.sleep(4)

        if pending:
            time.sleep(10)


def main():
    args = parse_args()
    logging.basicConfig(filename="batch_generation.log", level=logging.INFO, format="%(asctime)s %(message)s")
    allowed = {int(x) for x in args.allowed_gpus.split(",") if x.strip()} if args.allowed_gpus else set()
    actions = parse_actions(args.actions)

    with open(args.prompt_json, encoding="utf-8") as f:
        tasks = json.load(f)

    mp.set_start_method("spawn", force=True)
    for action_name, pose in actions.items():
        run_action(action_name, pose, tasks, args, allowed)


if __name__ == "__main__":
    main()
