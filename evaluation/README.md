# Evaluation Utilities

This directory contains the fixed-prompt sample set and clean evaluation scripts
used for Light Interaction.

The release includes:

```text
data/
  refined_prompts_llava16.json
  sampled_200/

scripts/
  generate_fixed_prompts.py
  evaluate_psnr_ssim_lpips.py
  evaluate_vbench_batch.py
```

Generated videos, benchmark outputs, logs, and intermediate debug files are not
included.

## Sample Set

`data/sampled_200/` contains 200 initial images. Each item corresponds to one
entry in `data/refined_prompts_llava16.json`.

Each JSON entry contains:

```text
filename
original_prompt
refined_prompt
image_path
```

`image_path` uses a repository-relative path so the sample set can be moved with
the repository.

## Batch Video Generation

Use `scripts/generate_fixed_prompts.py` to generate videos for all fixed prompts.
The script supports multiple camera-action groups and dynamic GPU scheduling.

Example:

```bash
python evaluation/scripts/generate_fixed_prompts.py \
  --prompt-json evaluation/data/refined_prompts_llava16.json \
  --hy-worldplay-root /path/to/patched/HY-WorldPlay \
  --model-path /path/to/HunyuanVideo-1.5 \
  --action-ckpt /path/to/ar_distilled_action_model/diffusion_pytorch_model.safetensors \
  --output-root outputs/fixed_prompt \
  --torchrun /path/to/torchrun \
  --allowed-gpus 0,1,2,3 \
  --acceleration-preset all
```

The default action groups are:

```text
left_right=left-5, right-5.5
forward_backward=w-5, s-5.5
```

Custom action groups can be passed with `--actions name=pose`.

## PSNR / SSIM / LPIPS

Use `scripts/evaluate_psnr_ssim_lpips.py` for pixel and perceptual quality
metrics. The script supports two evaluation modes.

### Mutual Evaluation

Mutual evaluation compares a test method against a reference method. For each
reference frame, the script searches a local temporal window in the test video
and chooses the closest frame before computing:

```text
PSNR
SSIM
LPIPS
```

Example:

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

Self evaluation measures return-trajectory consistency. For a forward-then-back
or left-then-right camera trajectory, frames from the first half should match
frames from the second half after the camera returns. The script searches around
the theoretically aligned return frame and computes PSNR, SSIM, and LPIPS.

Example:

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

Interactive video models may produce small temporal phase shifts even when the
visual content is consistent. A strict frame-index comparison can unfairly
penalize a method for minor timing offsets. Window search aligns each query
frame to the best nearby candidate frame before computing image quality metrics.
This gives a more stable estimate of visual consistency under camera motion.

## VBench

Use `scripts/evaluate_vbench_batch.py` to run selected VBench dimensions on the
generated videos.

Default dimensions:

```text
subject_consistency
background_consistency
motion_smoothness
aesthetic_quality
imaging_quality
```

These dimensions were selected because they cover the main quality concerns for
interactive world-model videos:

- subject and background consistency measure identity and scene preservation;
- motion smoothness measures temporal stability;
- aesthetic quality measures visual appeal;
- imaging quality measures low-level visual fidelity.

Example:

```bash
python evaluation/scripts/evaluate_vbench_batch.py \
  --prompt-json evaluation/data/refined_prompts_llava16.json \
  --video-dir outputs/light_interaction/left_right \
  --output-csv evaluation_results/vbench_left_right.csv
```

Run VBench in an environment where VBench and its model dependencies are
installed.

## Output Files

The scripts write CSV summaries such as:

```text
left_right_mutual_metrics.csv
left_right_self_metrics.csv
vbench_left_right.csv
```

Each CSV contains per-video scores and an average row.
