#!/usr/bin/env python3
# Copyright (c) 2026 Jiacheng Lu and contributors.
# Licensed under the MIT License.
# See the LICENSE file in the repository root for details.
"""Batch video generation for the fixed 200-prompt evaluation set.

The script supports the two released Light Interaction backends:
HY-WorldPlay and Matrix-Game-3.0.
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


DEFAULT_HY_ACTIONS = {
    "left_right": "left-5, right-5.5",
    "forward_backward": "w-5, s-5.5",
}

DEFAULT_MATRIX_ACTIONS = {
    "left_right": "j:q*4,l:q*4",
    "forward_backward": "u:w*4,u:s*4",
}


def drain_results(result_queue):
    succeeded = 0
    failed = 0
    while True:
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            break
        if result.get("ok"):
            succeeded += 1
        else:
            failed += 1
    return succeeded, failed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="hy-worldplay", choices=["hy-worldplay", "matrix-game"])
    parser.add_argument("--prompt-json", default="evaluation/data/refined_prompts_llava16.json")
    parser.add_argument("--hy-worldplay-root", help="Patched HY-WorldPlay checkout.")
    parser.add_argument("--matrix-game-root", help="Patched Matrix-Game-3.0 checkout.")
    parser.add_argument("--model-path", help="HunyuanVideo-1.5 checkpoint directory for HY-WorldPlay.")
    parser.add_argument("--action-ckpt", help="HY-WorldPlay AR distilled action checkpoint.")
    parser.add_argument("--ckpt-dir", help="Matrix-Game-3.0 checkpoint directory.")
    parser.add_argument("--output-root", required=True, help="Directory where generated videos are written.")
    parser.add_argument("--torchrun", default="torchrun")
    parser.add_argument("--pythonpath", default=None, help="Optional PYTHONPATH override.")
    parser.add_argument("--actions", nargs="*", default=None)
    parser.add_argument("--allowed-gpus", default="", help="Comma-separated GPU ids. Empty means all visible GPUs.")
    parser.add_argument("--vram-threshold-mb", type=int, default=20)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=253)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--acceleration-preset", default="all", choices=["off", "context", "sparse", "cache", "all"])
    parser.add_argument("--matrix-size", default="704*1280")
    parser.add_argument("--matrix-num-iterations", type=int, default=8)
    parser.add_argument("--matrix-num-steps", type=int, default=3)
    parser.add_argument("--matrix-fa-version", default="0")
    parser.add_argument("--matrix-lightvae-pruning-rate", default="0.5")
    parser.add_argument("--matrix-vae-type", default="mg_lightvae")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return parser.parse_args()


def parse_actions(items):
    actions = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Action must use name=pose format: {item}")
        name, pose = item.split("=", 1)
        actions[name.strip()] = pose.strip()
    return actions


def default_action_items(backend):
    defaults = DEFAULT_MATRIX_ACTIONS if backend == "matrix-game" else DEFAULT_HY_ACTIONS
    return [f"{key}={value}" for key, value in defaults.items()]


def expand_matrix_action_sequence(spec):
    """Expand j:q*4,l:q*4 into j:q,j:q,j:q,j:q,l:q,l:q,l:q,l:q."""
    expanded = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "*" in item:
            pair, repeat = item.rsplit("*", 1)
            try:
                repeat_count = int(repeat)
            except ValueError as exc:
                raise ValueError(f"Invalid Matrix action repeat count in: {item}") from exc
        else:
            pair = item
            repeat_count = 1
        if ":" not in pair:
            raise ValueError(f"Matrix action must use mouse:key format, got: {pair}")
        expanded.extend([pair.strip()] * repeat_count)
    if not expanded:
        raise ValueError("Matrix action sequence is empty.")
    return ",".join(expanded)


def validate_args(args):
    if args.backend == "hy-worldplay":
        missing = [
            name for name, value in [
                ("--hy-worldplay-root", args.hy_worldplay_root),
                ("--model-path", args.model_path),
                ("--action-ckpt", args.action_ckpt),
            ] if not value
        ]
        if missing:
            raise ValueError(f"{args.backend} backend requires: {', '.join(missing)}")
    elif args.backend == "matrix-game":
        missing = [
            name for name, value in [
                ("--matrix-game-root", args.matrix_game_root),
                ("--ckpt-dir", args.ckpt_dir),
            ] if not value
        ]
        if missing:
            raise ValueError(f"{args.backend} backend requires: {', '.join(missing)}")
        run_helper = Path(args.matrix_game_root) / "scripts" / "run_accel_preset.sh"
        if not run_helper.is_file():
            raise ValueError(f"Matrix helper not found. Apply the patch first: {run_helper}")


def free_gpus(threshold_mb, allowed):
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi was not found; GPU scheduling requires NVIDIA tools.")
    cmd = ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]
    output = subprocess.check_output(cmd, text=True).strip()
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


def resolve_image_path(prompt_json, image_path):
    path = Path(image_path)
    if path.is_absolute():
        return path
    prompt_path = Path(prompt_json).resolve()
    return Path(prompt_path.parent.parent, image_path).resolve()


def build_hy_command(gpu_id, task, args, action_name, pose, out_dir, image_path):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if args.pythonpath:
        env["PYTHONPATH"] = args.pythonpath

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
    return cmd, env, args.hy_worldplay_root


def build_matrix_command(gpu_id, task, args, action_name, pose, out_dir, image_path):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if args.pythonpath:
        env["PYTHONPATH"] = args.pythonpath

    env.update({
        "MG_CKPT_DIR": str(Path(args.ckpt_dir).resolve()),
        "MG_OUTPUT_DIR": str(out_dir),
        "MG_SAVE_NAME": "gen",
        "MG_PRESET": args.acceleration_preset,
        "MG_NUM_GPUS": "1",
        "MG_MASTER_PORT": str(29540 + gpu_id),
        "MG_SIZE": args.matrix_size,
        "MG_NUM_ITERATIONS": str(args.matrix_num_iterations),
        "MG_NUM_INFERENCE_STEPS": str(args.matrix_num_steps),
        "MG_IMAGE": str(image_path),
        "MG_PROMPT": task["refined_prompt"],
        "MG_ACTION_SEQUENCE": expand_matrix_action_sequence(pose),
        "MG_FA_VERSION": args.matrix_fa_version,
        "MG_LIGHTVAE_PRUNING_RATE": args.matrix_lightvae_pruning_rate,
        "MG_VAE_TYPE": args.matrix_vae_type,
        "MG_SEED": str(args.seed),
    })
    cmd = ["bash", "scripts/run_accel_preset.sh", args.acceleration_preset]
    return cmd, env, args.matrix_game_root


def run_one(gpu_id, task, args, action_name, pose, result_queue):
    image_path = resolve_image_path(args.prompt_json, task["image_path"])
    out_dir = Path(args.output_root, action_name, safe_stem(task["filename"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "matrix-game":
        cmd, env, cwd = build_matrix_command(gpu_id, task, args, action_name, pose, out_dir, image_path)
    else:
        cmd, env, cwd = build_hy_command(gpu_id, task, args, action_name, pose, out_dir, image_path)

    start = time.time()
    result = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    duration = time.time() - start
    if result.returncode == 0:
        logging.info("[GPU %s] done %s/%s in %.1fs", gpu_id, action_name, task["filename"], duration)
        result_queue.put({
            "ok": True,
            "action": action_name,
            "filename": task["filename"],
            "duration": duration,
        })
    else:
        logging.error(
            "[%s] failed %s/%s\nSTDOUT:\n%s\nSTDERR:\n%s",
            datetime.now(),
            action_name,
            task["filename"],
            result.stdout[-4000:],
            result.stderr[-4000:],
        )
        result_queue.put({
            "ok": False,
            "action": action_name,
            "filename": task["filename"],
            "duration": duration,
            "returncode": result.returncode,
        })


def run_action(action_name, pose, tasks, args, allowed):
    pending = []
    skipped = 0
    for task in tasks:
        out_dir = Path(args.output_root, action_name, safe_stem(task["filename"]))
        if args.skip_existing and out_dir.exists() and any(out_dir.iterdir()):
            skipped += 1
            continue
        pending.append(task)

    print(f"[{action_name}] pending {len(pending)} / {len(tasks)} (skipped existing: {skipped})")
    active = {}
    result_queue = mp.Queue()
    succeeded = 0
    failed = 0
    while pending or active:
        finished = [gpu for gpu, proc in active.items() if not proc.is_alive()]
        for gpu in finished:
            active[gpu].join()
            if active[gpu].exitcode != 0:
                failed += 1
            del active[gpu]

        new_succeeded, new_failed = drain_results(result_queue)
        succeeded += new_succeeded
        failed += new_failed

        idle = [gpu for gpu in free_gpus(args.vram_threshold_mb, allowed) if gpu not in active]
        while idle and pending and len(active) < args.max_concurrent:
            gpu_id = idle.pop(0)
            task = pending.pop(0)
            proc = mp.Process(target=run_one, args=(gpu_id, task, args, action_name, pose, result_queue))
            proc.start()
            active[gpu_id] = proc
            print(f"[dispatch] gpu={gpu_id} action={action_name} file={task['filename']}")
            if idle and pending:
                time.sleep(4)

        if pending:
            time.sleep(10)

    new_succeeded, new_failed = drain_results(result_queue)
    succeeded += new_succeeded
    failed += new_failed

    summary = {
        "action": action_name,
        "total": len(tasks),
        "skipped": skipped,
        "succeeded": succeeded,
        "failed": failed,
    }
    print(
        f"[{action_name}] summary: total={summary['total']} skipped={summary['skipped']} "
        f"succeeded={summary['succeeded']} failed={summary['failed']}"
    )
    return summary


def main():
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    logging.basicConfig(filename="batch_generation.log", level=logging.INFO, format="%(asctime)s %(message)s")
    allowed = {int(x) for x in args.allowed_gpus.split(",") if x.strip()} if args.allowed_gpus else set()
    actions = parse_actions(args.actions or default_action_items(args.backend))

    with open(args.prompt_json, encoding="utf-8") as f:
        tasks = json.load(f)

    mp.set_start_method("spawn", force=True)
    summaries = []
    for action_name, pose in actions.items():
        summaries.append(run_action(action_name, pose, tasks, args, allowed))

    total = sum(item["total"] for item in summaries)
    skipped = sum(item["skipped"] for item in summaries)
    succeeded = sum(item["succeeded"] for item in summaries)
    failed = sum(item["failed"] for item in summaries)
    print(f"[batch] summary: total={total} skipped={skipped} succeeded={succeeded} failed={failed}")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
