# 📊 Evaluation Utilities

<p align="center">
  <a href="../README.md">← Back to Light Interaction</a>
</p>

This directory contains the **fixed-prompt sample set** and **clean evaluation scripts** used for Light Interaction quantitative experiments.

---

## 📂 What Is Included

```text
data/
  refined_prompts_llava16.json   # 200 image-text pairs with refined prompts
  sampled_200/                   # 200 initial images

scripts/
  batch_video_generation.py      # Batch generation across prompts & action groups
  evaluate_psnr_ssim_lpips.py    # PSNR / SSIM / LPIPS evaluation
  evaluate_vbench_batch.py       # VBench multi-dimension evaluation
```

> [!NOTE]
> Generated videos, benchmark outputs, logs, and intermediate debug files are **not** included in this release.

---

## 🖼️ Sample Set

`data/sampled_200/` contains 200 initial images from VBench. Each item corresponds to one entry in `data/refined_prompts_llava16.json`.

Each JSON entry contains:

| Field | Description |
|:---|:---|
| `filename` | Image filename |
| `original_prompt` | Original VBench prompt |
| `refined_prompt` | LLaVA-1.6 refined prompt for richer I2V generation |
| `image_path` | Repository-relative path (portable) |

---

## 🎬 Batch Video Generation

Use `scripts/batch_video_generation.py` to generate videos for all fixed prompts. The script supports multiple camera-action groups and dynamic GPU scheduling.

Required packages include `opencv-python`, `lpips`, `torchmetrics`, `pandas`, and `tqdm`. VBench evaluation additionally requires `vbench` and its model dependencies.

```bash
python evaluation/scripts/batch_video_generation.py \
  --prompt-json evaluation/data/refined_prompts_llava16.json \
  --hy-worldplay-root /path/to/patched/HY-WorldPlay \
  --model-path /path/to/HunyuanVideo-1.5 \
  --action-ckpt /path/to/ar_distilled_action_model/diffusion_pytorch_model.safetensors \
  --output-root outputs/fixed_prompt \
  --torchrun /path/to/torchrun \
  --allowed-gpus 0,1,2,3 \
  --acceleration-preset all
```

Default action groups:

```text
left_right=left-5, right-5.5
forward_backward=w-5, s-5.5
```

Custom action groups can be passed with `--actions name=pose`.

By default, existing non-empty output folders are skipped. Pass `--no-skip-existing` to regenerate them.

---

## 📐 PSNR / SSIM / LPIPS

Use `scripts/evaluate_psnr_ssim_lpips.py` for pixel and perceptual quality metrics. The script supports two evaluation modes.

### Mutual Evaluation (vs. Original)

Compares a test method against a reference method. For each reference frame, the script searches a local temporal window in the test video and chooses the closest frame before computing PSNR, SSIM, and LPIPS.

```bash
python evaluation/scripts/evaluate_psnr_ssim_lpips.py \
  --run-mutual \
  --ref-dir outputs/baseline/left_right \
  --test-dir outputs/light_interaction/left_right \
  --output-dir evaluation_results \
  --tag left_right \
  --mutual-window 30
```

### Self Consistency

Measures return-trajectory consistency. For a forward-then-back or left-then-right camera trajectory, frames from the first half should match frames from the second half after the camera returns.

```bash
python evaluation/scripts/evaluate_psnr_ssim_lpips.py \
  --run-self \
  --test-dir outputs/light_interaction/forward_backward \
  --output-dir evaluation_results \
  --tag forward_backward \
  --one-way-sec 5 \
  --self-window 50
```

### Why Window Search?

Interactive video models may produce small temporal phase shifts even when the visual content is consistent. A strict frame-index comparison can unfairly penalize a method for minor timing offsets. Window search aligns each query frame to the best nearby candidate frame before computing image quality metrics, giving a more stable estimate of visual consistency under camera motion.

---

## 🏷️ VBench

Use `scripts/evaluate_vbench_batch.py` to run selected VBench dimensions on the generated videos.

> [!IMPORTANT]
> VBench requires a dedicated conda environment (e.g. ``vbench_env``) with its own dependencies installed.
> If you encounter ``libnvshmem_host.so.3: cannot open shared object file``, add the nvshmem library path:
> ```bash
> export LD_LIBRARY_PATH=/path/to/nvidia/nvshmem/lib:$LD_LIBRARY_PATH
> ```
> The nvshmem library ships with the ``nvidia-nvshmem`` PyPI package inside some PyTorch environments.

```bash
python evaluation/scripts/evaluate_vbench_batch.py \
  --prompt-json evaluation/data/refined_prompts_llava16.json \
  --video-dir outputs/light_interaction/left_right \
  --output-csv evaluation_results/vbench_left_right.csv
```

Default evaluation dimensions:

| Dimension | What it measures |
|:---|:---|
| `subject_consistency` | Identity preservation across frames |
| `background_consistency` | Scene preservation across frames |
| `motion_smoothness` | Temporal stability |
| `aesthetic_quality` | Visual appeal |
| `imaging_quality` | Low-level visual fidelity |

> [!TIP]
> Run VBench in an environment where VBench and its model dependencies are installed.

---

## 📋 Output Files

The scripts write CSV summaries with per-video scores and an average row:

```text
left_right_mutual_metrics.csv
left_right_self_metrics.csv
vbench_left_right.csv
```

If input videos are missing or cannot be decoded, the scripts report skipped items on stdout so incomplete runs are easier to diagnose.
