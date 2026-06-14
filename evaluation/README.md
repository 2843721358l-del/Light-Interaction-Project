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
    ├── prepare_evaluation_assets.py
    └── setup_evaluation_env.sh
```

Generated videos, CSV files, logs, and debug files are not included.

## 🛠️ Evaluation Environment

Use the patched HY-WorldPlay environment for video generation, and use this standalone environment for PSNR / SSIM / LPIPS / VBench:

```bash
bash evaluation/scripts/setup_evaluation_env.sh
conda activate light-interaction-eval
```

The setup script installs all metric dependencies, prepares LPIPS/VBench assets, and runs an import check. Assets are cached outside the repository:

- VBench checkpoints: `~/.cache/light-interaction/vbench`
- Torch Hub / LPIPS checkpoints: `~/.cache/light-interaction/torch`

HY-WorldPlay and HunyuanVideo-1.5 generation weights are not installed by this script; prepare them with the minimal downloader below. Evaluation metric assets are prepared by the setup script and kept in cache directories outside this repository.

For the released Light Interaction HY-WorldPlay path, only the HunyuanVideo-1.5 runtime assets and the few-step distilled autoregressive action checkpoint are required. You can prepare just those assets with:

```bash
python hy-worldplay/scripts/download_minimal_worldplay_assets.py
```

The bidirectional action model, multi-step autoregressive action model, and RL variants are not required for the documented evaluation pipeline.

Advanced options:

- `HF_ENDPOINT=https://hf-mirror.com bash evaluation/scripts/setup_evaluation_env.sh` for a Hugging Face mirror.
- `EVAL_ENV_BACKEND=venv bash evaluation/scripts/setup_evaluation_env.sh` if conda is unavailable.
- `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 bash evaluation/scripts/setup_evaluation_env.sh` for CUDA 11.8 wheels.
- `PREPARE_EVAL_ASSETS=0 bash evaluation/scripts/setup_evaluation_env.sh` to skip metric asset preparation.

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
  --action-ckpt /path/to/ar_distilled_action_model/model.safetensors \
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

Run VBench in `light-interaction-eval`. The default device is `auto`: CUDA is used when available, otherwise CPU is used as a slower fallback.

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
