# 🔌 HY-WorldPlay Acceleration Patch

This folder provides the Light Interaction patch for upstream [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay).

It does not include the original HY-WorldPlay source tree or any model checkpoints. Clone upstream HY-WorldPlay and download checkpoints from the official sources before applying this patch.

## 📂 Contents

```text
hy-worldplay/
├── files/
│   ├── worldplay_acceleration_config.py
│   └── hyvideo/models/transformers/modules/
│       ├── ar_sparse_operation.py
│       ├── bi_sparse_operation_for_KV_cache.py
│       └── longcat_kernel.py
├── patches/
│   └── 0001-integrate-acceleration.patch
└── scripts/
    ├── apply_patch.sh
    ├── check_worldplay_assets.py
    ├── check_worldplay_env.py
    └── setup_worldplay_env.sh
```

`files/` contains new Light Interaction modules. `patches/0001-integrate-acceleration.patch` modifies upstream HY-WorldPlay entry points, pipeline code, transformer code, attention code, and context-selection logic.

## 📦 Upstream Resources

| Resource | Link |
|:---|:---|
| HY-WorldPlay code | <https://github.com/Tencent-Hunyuan/HY-WorldPlay> |
| HY-WorldPlay checkpoints | <https://huggingface.co/tencent/HY-WorldPlay> |
| HunyuanVideo-1.5 checkpoints | <https://huggingface.co/tencent/HunyuanVideo-1.5> |

## 🚀 Reproducible Setup

```bash
git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay.git
bash hy-worldplay/scripts/apply_patch.sh /path/to/HY-WorldPlay
bash hy-worldplay/scripts/setup_worldplay_env.sh /path/to/HY-WorldPlay
conda activate light-interaction-worldplay
```

The patch script checks that the target looks like a HY-WorldPlay checkout and runs `patch --dry-run` before copying files. The environment script installs PyTorch 2.6.0, HY-WorldPlay requirements, Triton, and the Python packages needed by the Light Interaction modules.

This release was tested on upstream HY-WorldPlay commit:

```text
1588e1336e842b03b0a7860c654ebd7c46bb065e
```

If patching fails, check whether the upstream HY-WorldPlay files have changed. For reproducibility, record the upstream commit hash used in your experiments.

If conda channels are unavailable, use a local venv:

```bash
WORLDPLAY_ENV_BACKEND=venv bash hy-worldplay/scripts/setup_worldplay_env.sh /path/to/HY-WorldPlay
source .venv-light-interaction-worldplay/bin/activate
```

For CUDA 11.8 or CPU-only package checks, override the PyTorch index:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 bash hy-worldplay/scripts/setup_worldplay_env.sh /path/to/HY-WorldPlay
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu bash hy-worldplay/scripts/setup_worldplay_env.sh /path/to/HY-WorldPlay
```

CPU-only environments are useful for import checks, but HY-WorldPlay inference requires NVIDIA GPUs.

## ✅ Environment And Asset Checks

After activating the runtime environment:

```bash
python hy-worldplay/scripts/check_worldplay_env.py \
  --worldplay-root /path/to/HY-WorldPlay \
  --require-cuda
```

After downloading models with upstream `download_models.py` or Hugging Face CLI:

```bash
python hy-worldplay/scripts/check_worldplay_assets.py \
  --worldplay-root /path/to/HY-WorldPlay \
  --model-path /path/to/HunyuanVideo-1.5 \
  --action-ckpt /path/to/HY-WorldPlay/ar_distilled_action_model/diffusion_pytorch_model.safetensors
```

The HunyuanVideo-1.5 model root should contain:

```text
vae/
scheduler/
transformer/480p_i2v/
text_encoder/llm/
text_encoder/byt5-small/
text_encoder/Glyph-SDXL-v2/
vision_encoder/siglip/
```

## ▶️ Run

```bash
cd /path/to/HY-WorldPlay

export HY_MODEL_PATH=/path/to/HunyuanVideo-1.5
export HY_AR_DISTILL_ACTION_MODEL_PATH=/path/to/HY-WorldPlay/ar_distilled_action_model/diffusion_pytorch_model.safetensors
export HY_N_INFERENCE_GPU=1

bash run.sh
```

For a quick smoke test before full 253-frame generation:

```bash
HY_N_INFERENCE_GPU=1 \
HY_NUM_FRAMES=13 \
HY_POSE='left-3' \
HY_OUTPUT_PATH=/tmp/light-interaction-worldplay-smoke-output \
bash run.sh
```

Useful `run.sh` overrides:

| Variable | Default |
|:---|:---|
| `HY_PROMPT` | Built-in demo prompt |
| `HY_IMAGE_PATH` | `./assets/img/test.png` |
| `HY_MODEL_PATH` | empty, must be set |
| `HY_AR_DISTILL_ACTION_MODEL_PATH` | empty, must be set |
| `HY_N_INFERENCE_GPU` | `8` |
| `HY_NUM_FRAMES` | `253` |
| `HY_POSE` | `left-31, right-32` |
| `HY_OUTPUT_PATH` | `./outputs/` |

## 🎛️ Presets

The patched `run.sh` uses `--acceleration_preset all` by default.

| Preset | Behavior |
|:---|:---|
| `off` | Original dense attention and upstream context behavior |
| `context` | Adaptive temporal and spatial context management |
| `sparse` | 3D block sparse attention for KV-cache recomputation and AR denoising |
| `cache` | Denoising cache acceleration |
| `all` | Enables all Light Interaction components |

To switch presets, edit the `--acceleration_preset` argument in the patched `run.sh`.

## ⏱️ Inference Log

The patched pipeline prints an inference summary:

```text
[Inference Summary]
 Prompt & Vision Encode    :    15.45 s
 History Selection         :     1.29 s
 KV Cache Recompute        :   206.89 s
 Denoise Loop              :   217.26 s
 DiT Core (KV + Denoise)   :   424.15 s
 Transformer AR Rollout    :   443.99 s
 VAE Pixel Decoding        :    34.89 s
 Total End-to-End Time     :   494.39 s
 Peak VRAM                 :    48.70 GB
```

For paper-style timing, use:

```text
DiT Core (KV + Denoise) = KV Cache Recompute + Denoise Loop
```

## 🔧 Diagnostics

Relative-L1 denoising diagnostics are disabled by default. Enable them only for development:

```bash
--enable_l1_diagnostics true
```

For Triton autotuning experiments:

```bash
TRITON_REEVALUATE_KEY=1 bash run.sh
```

The release keeps one tested Triton config by default to avoid first-run autotuning noise.

## 📄 License Notes

This folder stores patch-style modifications instead of full upstream files. Upstream HY-WorldPlay files and model weights remain governed by their own licenses. `longcat_kernel.py` includes attribution for LongCat-Video-derived code; see [../NOTICE.md](../NOTICE.md).
