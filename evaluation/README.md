# 📊 Evaluation Utilities

This folder contains the fixed sample set and scripts used for Light Interaction quantitative evaluation.

```text
evaluation/
├── data/
│   ├── refined_prompts_llava16.json
│   └── sampled_200/
├── requirements.txt
└── scripts/
    ├── batch_video_generation.py
    ├── check_evaluation_env.py
    ├── evaluate_psnr_ssim_lpips.py
    ├── evaluate_vbench_batch.py
    └── setup_evaluation_env.sh
```

Generated videos, CSV files, logs, and debug files are not included.

## 🛠️ Evaluation Environment

Recommended setup:

- Use the patched HY-WorldPlay environment only for video generation.
- Use a standalone `light-interaction-eval` environment for PSNR / SSIM / LPIPS and VBench.

This keeps benchmark dependencies out of the model runtime while still giving one unified environment for all evaluation metrics.

```bash
bash evaluation/scripts/setup_evaluation_env.sh
conda activate light-interaction-eval
python evaluation/scripts/check_evaluation_env.py
```

By default the setup script installs PyTorch from the CUDA 12.1 wheel index:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
  bash evaluation/scripts/setup_evaluation_env.sh
```

For CUDA 11.8:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 \
  bash evaluation/scripts/setup_evaluation_env.sh
```

If conda channels are unavailable, use the venv backend:

```bash
EVAL_ENV_BACKEND=venv bash evaluation/scripts/setup_evaluation_env.sh
source .venv-light-interaction-eval/bin/activate
```

To install into the currently activated environment:

```bash
EVAL_ENV_BACKEND=current bash evaluation/scripts/setup_evaluation_env.sh
```

If VBench dependencies conflict with a local machine-specific setup, create a separate VBench environment as a fallback. The preferred documented path remains the unified evaluation environment above.

Model checkpoints are not installed by this script. HY-WorldPlay and HunyuanVideo-1.5 weights should be downloaded following upstream instructions. VBench may download or cache metric weights on first use; keep those caches outside this repository.

## 🖼️ Dataset

`data/refined_prompts_llava16.json` contains 200 entries. Each entry has:

| Field | Description |
|:---|:---|
| `filename` | Initial image filename |
| `original_prompt` | Original VBench prompt |
| `refined_prompt` | Prompt used for I2V generation |
| `image_path` | Repository-relative image path |

`data/sampled_200/` contains the corresponding 200 initial images.

## 🎬 Batch Generation

Run batch generation in the patched HY-WorldPlay environment, not in `light-interaction-eval`, because it calls upstream `hyvideo/generate.py` and uses model checkpoints.

```bash
python evaluation/scripts/batch_video_generation.py \
  --prompt-json evaluation/data/refined_prompts_llava16.json \
  --hy-worldplay-root /path/to/patched/HY-WorldPlay \
  --model-path /path/to/HunyuanVideo-1.5 \
  --action-ckpt /path/to/ar_distilled_action_model/diffusion_pytorch_model.safetensors \
  --output-root outputs/fixed_prompt \
  --allowed-gpus 0,1,2,3 \
  --acceleration-preset all
```

Default action groups:

```text
left_right=left-5, right-5.5
forward_backward=w-5, s-5.5
```

Use `--actions name=pose` to provide custom action groups. Existing non-empty output folders are skipped by default; pass `--no-skip-existing` to regenerate them.

## 📐 PSNR / SSIM / LPIPS

Run these metric scripts in `light-interaction-eval`.

Mutual evaluation compares generated videos with a reference run:

```bash
python evaluation/scripts/evaluate_psnr_ssim_lpips.py \
  --run-mutual \
  --ref-dir outputs/baseline/left_right \
  --test-dir outputs/light_interaction/left_right \
  --output-dir evaluation_results \
  --tag left_right \
  --mutual-window 30
```

Self-consistency evaluates return-trajectory consistency inside one generated video:

```bash
python evaluation/scripts/evaluate_psnr_ssim_lpips.py \
  --run-self \
  --test-dir outputs/light_interaction/forward_backward \
  --output-dir evaluation_results \
  --tag forward_backward \
  --one-way-sec 5 \
  --self-window 50
```

The scripts print missing-video and decode-failure counts so incomplete runs are visible.

## 🏷️ VBench

Run VBench in `light-interaction-eval`.

```bash
python evaluation/scripts/evaluate_vbench_batch.py \
  --prompt-json evaluation/data/refined_prompts_llava16.json \
  --video-dir outputs/light_interaction/left_right \
  --output-csv evaluation_results/vbench_left_right.csv
```

Default dimensions:

- `subject_consistency`
- `background_consistency`
- `motion_smoothness`
- `aesthetic_quality`
- `imaging_quality`

If VBench reports `libnvshmem_host.so.3: cannot open shared object file`, add the nvshmem library path from your VBench environment:

```bash
export LD_LIBRARY_PATH=/path/to/nvidia/nvshmem/lib:$LD_LIBRARY_PATH
```

## 📋 Outputs

Typical CSV outputs:

```text
left_right_mutual_metrics.csv
forward_backward_self_metrics.csv
vbench_left_right.csv
```
