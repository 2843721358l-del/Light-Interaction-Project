# 🔌 HY-WorldPlay Acceleration Patch

This folder provides the Light Interaction patch for upstream [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay).

It does not include the original HY-WorldPlay source tree or any model checkpoints. Clone upstream HY-WorldPlay and download checkpoints from the official sources before applying this patch.

## 📂 Contents

```text
hy-worldplay/
├── LICENSE_LONGCAT
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
    ├── download_minimal_worldplay_assets.py
    ├── setup_worldplay_release.sh
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

The recommended path is one command after cloning upstream HY-WorldPlay:

```bash
git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay.git ../HY-WorldPlay
bash hy-worldplay/scripts/setup_worldplay_release.sh ../HY-WorldPlay
conda activate light-interaction-worldplay
```

This applies the patch, creates the WorldPlay runtime environment, downloads the minimal model assets, and runs environment/asset checks.

This release was tested on upstream HY-WorldPlay commit:

```text
1588e1336e842b03b0a7860c654ebd7c46bb065e
```

Minimal model assets:

- HunyuanVideo-1.5 480P-I2V runtime files
- HY-WorldPlay `ar_distilled_action_model/*.safetensors`

You do not need the bidirectional action model, the multi-step autoregressive action model, or RL variants.

If you already manage environments or checkpoints manually, use `apply_patch.sh`, `setup_worldplay_env.sh`, `download_minimal_worldplay_assets.py`, and the two check scripts separately.

## ▶️ Run

```bash
cd ../HY-WorldPlay
conda activate light-interaction-worldplay
bash run_light_interaction.sh
```

For a quick smoke test before full 253-frame generation:

```bash
HY_NUM_FRAMES=13 HY_POSE='left-3' bash run_light_interaction.sh
```

Useful `run.sh` overrides:

| Variable | Default |
|:---|:---|
| `HY_PROMPT` | Built-in demo prompt |
| `HY_IMAGE_PATH` | `./assets/img/test.png` |
| `HY_MODEL_PATH` | empty, must be set |
| `HY_AR_DISTILL_ACTION_MODEL_PATH` | empty, must be set |
| `HY_N_INFERENCE_GPU` | `1` |
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

## 🔧 Advanced Notes

- Use `WORLDPLAY_ENV_BACKEND=venv` with `setup_worldplay_release.sh` if conda is unavailable.
- Use `WORLDPLAY_ASSET_OUTPUT_ROOT=/path/to/models` to store downloaded weights outside the Hugging Face cache.
- Use `HF_TOKEN=<token>` if gated Hugging Face assets require authentication.
- CPU-only environments can run package checks, but HY-WorldPlay inference requires NVIDIA GPUs.

## ⏱️ Inference Log

The patched pipeline prints an inference summary:

```text
[Inference Summary]
 Prompt & Vision Encode    :     0.63 s
 History Selection         :     0.83 s
 KV Cache Recompute        :   102.35 s
 Denoise Loop              :   121.67 s
 DiT Core (KV + Denoise)   :   224.02 s
 Transformer AR Rollout    :   232.15 s
 VAE Pixel Decoding        :    31.24 s
 Total End-to-End Time     :   264.06 s
 Peak VRAM                 :    75.78 GB
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
