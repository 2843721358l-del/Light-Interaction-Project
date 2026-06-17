# 📊 Evaluation Utilities

This folder contains the fixed sample set and scripts used for Light Interaction quantitative evaluation.

```text
evaluation/
├── data/
│   ├── eval_assets_manifest.json
│   ├── refined_prompts_llava16.json
│   ├── selected_vbench_i2v_16_9_200.txt
│   └── sampled_200/.gitkeep
├── requirements.txt
└── scripts/
    ├── batch_video_generation.py
    ├── check_evaluation_env.py
    ├── download_eval_assets.py
    ├── evaluate_psnr_ssim_lpips.py
    ├── evaluate_vbench_batch.py
    ├── prepare_evaluation_assets.py
    └── setup_evaluation_env.sh
```

Generated videos, CSV files, logs, and debug files are not included.

## 🖼️ Evaluation Assets

The GitHub repository keeps the prompt JSON, asset manifest, selected filename list, and downloader only. The 200 evaluation initial images are selected from the official VBench-I2V Image Suite, specifically the 16:9 cropped image subset. To avoid redistributing third-party image assets, this repository does not host the images directly. The downloader fetches the official VBench-I2V assets and extracts the selected 200 images used in our evaluation.

Download the evaluation images before batch generation:

```bash
python evaluation/scripts/download_eval_assets.py
```

If you already have VBench data, provide the path and skip the download step:

```bash
python evaluation/scripts/download_eval_assets.py \
  --vbench-root /path/to/VBench \
  --skip-download
```

The downloader requires `gdown`. If it is not installed:

```bash
pip install gdown
```

When `evaluation/data/eval_assets_manifest.json` is present, the downloader also checks file sizes and SHA256 hashes against the expected values.

## 🛠️ Evaluation Environment

Use the Light Interaction-enabled HY-WorldPlay environment for video generation, and use this standalone environment for PSNR / SSIM / LPIPS / VBench:

```bash
bash evaluation/scripts/setup_evaluation_env.sh
conda activate light-interaction-eval
```

The setup script installs all metric dependencies, prepares LPIPS/VBench assets, and runs an import check. Assets are cached outside the repository:

- VBench checkpoints: `~/.cache/light-interaction/vbench`
- Torch Hub / LPIPS checkpoints: `~/.cache/light-interaction/torch`

HY-WorldPlay, HunyuanVideo-1.5, and Matrix-Game-3.0 generation weights are not installed by this script. Evaluation metric assets are prepared by the setup script and kept in cache directories outside this repository.

For the released Light Interaction HY-WorldPlay path, only the HunyuanVideo-1.5 runtime assets and the few-step distilled autoregressive action checkpoint are required. You can prepare just those assets with:

```bash
python hy-worldplay/scripts/download_minimal_worldplay_assets.py
```

The bidirectional action model, multi-step autoregressive action model, and RL variants are not required for the documented evaluation pipeline.

For Matrix-Game-3.0 generation, prepare the Light Interaction-enabled Matrix runtime and model assets with:

```bash
bash matrix-game-3.0/scripts/setup_matrix_release.sh /path/to/Matrix-Game/Matrix-Game-3
```

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

`data/sampled_200/` is populated by `download_eval_assets.py` and contains the corresponding 200 initial images after download.

## 🎬 Batch Generation

Run batch generation in the adapted model runtime environment, not in `light-interaction-eval`, because generation calls upstream model code and uses model checkpoints. The same script supports both released backends.

HY-WorldPlay:

```bash
python evaluation/scripts/batch_video_generation.py \
  --backend hy-worldplay \
  --prompt-json evaluation/data/refined_prompts_llava16.json \
  --hy-worldplay-root /path/to/adapted/HY-WorldPlay \
  --model-path /path/to/HunyuanVideo-1.5 \
  --action-ckpt /path/to/ar_distilled_action_model/model.safetensors \
  --output-root outputs/fixed_prompt \
  --allowed-gpus 0,1,2,3 \
  --acceleration-preset all
```

Matrix-Game-3.0:

```bash
python evaluation/scripts/batch_video_generation.py \
  --backend matrix-game \
  --prompt-json evaluation/data/refined_prompts_llava16.json \
  --matrix-game-root /path/to/adapted/Matrix-Game/Matrix-Game-3 \
  --ckpt-dir /path/to/Matrix-Game-3.0 \
  --output-root outputs/fixed_prompt_matrix \
  --allowed-gpus 0,1,2,3 \
  --acceleration-preset all \
  --matrix-num-iterations 8
```

The Matrix-Game-3.0 release helper defaults to `MG_NUM_ITERATIONS=4` for quick demos. The batch evaluation example uses `--matrix-num-iterations 8` to reproduce the 8-interaction left/right and forward/backward evaluation protocol.

Default HY-WorldPlay action groups:

```text
left_right=left-5, right-5.5
forward_backward=w-5, s-5.5
```

Default Matrix-Game-3.0 action groups:

```text
left_right=j:q*4,l:q*4
forward_backward=u:w*4,u:s*4
```

The repeat counts above are generated from `--matrix-num-iterations`; for example, 4 iterations become `j:q*2,l:q*2`.

Use `--actions name=pose` to provide custom HY-WorldPlay action groups. For Matrix-Game-3.0, use `name=mouse:key*repeat,...`, for example `left_right=j:q*4,l:q*4`. Existing non-empty output folders are skipped by default; pass `--no-skip-existing` to regenerate them.

Relative `--output-root` values are resolved from the directory where you launch `batch_video_generation.py`, then passed to the backend as absolute paths. This keeps resume/skip checks and generated outputs in the same location.

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

For Matrix-Game-3.0 return-trajectory videos, use the Matrix self-alignment profile:

```bash
python evaluation/scripts/evaluate_psnr_ssim_lpips.py \
  --run-self \
  --self-profile matrix-game \
  --test-dir outputs/fixed_prompt_matrix/forward_backward \
  --output-dir evaluation_results \
  --tag matrix_forward_backward \
  --self-window 40
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

The same VBench command works for Matrix-Game-3.0 outputs by changing `--video-dir` and `--output-csv`.

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
