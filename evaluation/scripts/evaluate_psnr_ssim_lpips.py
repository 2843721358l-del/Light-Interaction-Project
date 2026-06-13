#!/usr/bin/env python3
# Copyright (c) 2026 Jiacheng Lu and contributors.
# Licensed under the MIT License.
# See the LICENSE file in the repository root for details.
"""PSNR / SSIM / LPIPS evaluation for Light Interaction.

Supports two modes:
  - Mutual evaluation: test method vs. reference method with window search.
  - Self consistency: return-trajectory temporal consistency within a video.
"""

import argparse
import os
from pathlib import Path
import warnings

import cv2
import lpips
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_pt
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_pt
from tqdm import tqdm


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-dir", help="Reference video directory for mutual evaluation.")
    parser.add_argument("--test-dir", required=True, help="Test video directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", default="eval")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--one-way-sec", type=float, default=5.0)
    parser.add_argument("--mutual-window", type=int, default=30)
    parser.add_argument("--self-window", type=int, default=50)
    parser.add_argument("--run-mutual", action="store_true")
    parser.add_argument("--run-self", action="store_true")
    return parser.parse_args()


def read_video(video_path, device, target_hw=None):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if target_hw and (frame.shape[0] != target_hw[0] or frame.shape[1] != target_hw[1]):
            frame = cv2.resize(frame, (target_hw[1], target_hw[0]))
        frames.append(frame)
    cap.release()
    if not frames:
        return None
    tensor = torch.from_numpy(np.asarray(frames)).permute(0, 3, 1, 2).float() / 255.0
    return tensor.to(device)


def find_video(root, name):
    root = Path(root)
    candidates = [
        root / name / "gen.mp4",
        root / name.replace(" ", "_") / "gen.mp4",
        root / f"{name}.mp4",
        root / f"{name.replace(' ', '_')}.mp4",
    ]
    for path in candidates:
        if path.exists():
            return path
    folder = root / name
    if folder.is_dir():
        videos = sorted(folder.glob("*.mp4"))
        if videos:
            return videos[0]
    folder = root / name.replace(" ", "_")
    if folder.is_dir():
        videos = sorted(folder.glob("*.mp4"))
        if videos:
            return videos[0]
    return None


def cap_psnr(value):
    return 100.0 if value == float("inf") else value


def mutual_metrics(ref_path, test_path, loss_fn, device, search_window):
    ref = read_video(ref_path, device)
    if ref is None:
        return None
    test = read_video(test_path, device, target_hw=(ref.shape[2], ref.shape[3]))
    if test is None:
        return None

    psnr_scores, ssim_scores, lpips_scores = [], [], []
    for i in range(ref.shape[0]):
        ref_frame = ref[i:i + 1]
        start = max(0, i - search_window)
        end = min(test.shape[0], i + search_window + 1)
        if start >= end:
            continue
        window = test[start:end]
        mse = torch.mean((window - ref_frame) ** 2, dim=(1, 2, 3))
        best = torch.argmin(mse).item()
        test_frame = window[best:best + 1]

        psnr_scores.append(cap_psnr(psnr_pt(test_frame, ref_frame, data_range=1.0).item()))
        ssim_scores.append(ssim_pt(test_frame, ref_frame, kernel_size=7, data_range=1.0).item())
        with torch.no_grad():
            lpips_scores.append(loss_fn(ref_frame * 2.0 - 1.0, test_frame * 2.0 - 1.0).item())

    del ref, test
    if not psnr_scores:
        return None
    return np.mean(psnr_scores), np.mean(ssim_scores), np.mean(lpips_scores)


def self_metrics(video_path, loss_fn, device, fps, one_way_sec, search_window):
    video = read_video(video_path, device)
    if video is None or video.shape[0] < 10:
        return None

    total = video.shape[0]
    pivot = int(one_way_sec * fps)
    theoretical_end = pivot * 2
    analyze_limit = min(int(pivot * 0.9), total)
    small = F.interpolate(video, size=(256, 256), mode="bilinear", align_corners=False)

    psnr_scores, ssim_scores, lpips_scores = [], [], []
    for i in range(analyze_limit):
        ideal = min(theoretical_end - i, total - 1)
        start = max(pivot, ideal - search_window)
        end = min(total - 1, ideal + search_window)
        if start >= end:
            continue

        best_score, best_idx = -1.0, -1
        ref_small = small[i:i + 1]
        for j in range(start, end + 1):
            score = ssim_pt(ref_small, small[j:j + 1], kernel_size=3, data_range=1.0).item()
            if score > best_score:
                best_score, best_idx = score, j

        if best_idx >= 0:
            ref_frame = video[i:i + 1]
            match_frame = video[best_idx:best_idx + 1]
            psnr_scores.append(cap_psnr(psnr_pt(ref_frame, match_frame, data_range=1.0).item()))
            ssim_scores.append(ssim_pt(ref_frame, match_frame, kernel_size=7, data_range=1.0).item())
            with torch.no_grad():
                lpips_scores.append(loss_fn(ref_frame * 2.0 - 1.0, match_frame * 2.0 - 1.0).item())

    del video, small
    if not psnr_scores:
        return None
    return np.mean(psnr_scores), np.mean(ssim_scores), np.mean(lpips_scores)


def list_video_ids(video_dir):
    root = Path(video_dir)
    if not root.exists():
        return []
    ids = set()
    for path in root.rglob("*.mp4"):
        ids.add(path.parent.name if path.name == "gen.mp4" else path.stem)
    return sorted(ids)


def write_results(rows, csv_path, id_key):
    if not rows:
        print(f"[warn] no valid rows for {csv_path.name}; CSV was not written.")
        return
    df = pd.DataFrame(rows)
    avg = {id_key: "--- AVERAGE ---"}
    for key in ["PSNR", "SSIM", "LPIPS"]:
        avg[key] = df[key].mean()
    pd.concat([df, pd.DataFrame([avg])], ignore_index=True).to_csv(csv_path, index=False)


def main():
    args = parse_args()
    if not args.run_mutual and not args.run_self:
        args.run_mutual = args.ref_dir is not None
        args.run_self = True

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net="alex", verbose=False).to(device)

    if args.run_mutual:
        if not args.ref_dir:
            raise ValueError("--ref-dir is required for mutual evaluation.")
        rows = []
        missing = 0
        failed = 0
        for video_id in tqdm(list_video_ids(args.ref_dir), desc="Mutual evaluation"):
            ref_path = find_video(args.ref_dir, video_id)
            test_path = find_video(args.test_dir, video_id)
            if not ref_path or not test_path:
                missing += 1
                continue
            metrics = mutual_metrics(ref_path, test_path, loss_fn, device, args.mutual_window)
            if metrics:
                rows.append({"Video_ID": video_id, "PSNR": metrics[0], "SSIM": metrics[1], "LPIPS": metrics[2]})
            else:
                failed += 1
            torch.cuda.empty_cache()
        write_results(rows, out_dir / f"{args.tag}_mutual_metrics.csv", "Video_ID")
        print(f"[mutual] wrote {len(rows)} rows; missing videos: {missing}; decode/metric failures: {failed}")

    if args.run_self:
        rows = []
        missing = 0
        failed = 0
        for video_id in tqdm(list_video_ids(args.test_dir), desc="Self evaluation"):
            video_path = find_video(args.test_dir, video_id)
            if not video_path:
                missing += 1
                continue
            metrics = self_metrics(video_path, loss_fn, device, args.fps, args.one_way_sec, args.self_window)
            if metrics:
                rows.append({"Video_ID": video_id, "PSNR": metrics[0], "SSIM": metrics[1], "LPIPS": metrics[2]})
            else:
                failed += 1
            torch.cuda.empty_cache()
        write_results(rows, out_dir / f"{args.tag}_self_metrics.csv", "Video_ID")
        print(f"[self] wrote {len(rows)} rows; missing videos: {missing}; decode/metric failures: {failed}")


if __name__ == "__main__":
    main()
