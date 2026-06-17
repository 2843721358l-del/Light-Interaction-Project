# 🔌 Matrix-Game-3.0 Adapter for Light Interaction

This folder provides the Light Interaction adapter for upstream [Matrix-Game-3.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-3).

It does not include the original Matrix-Game-3.0 source tree or any model checkpoints. Clone the upstream repository and download checkpoints from the official sources before installing this adapter.

## 📂 Contents

```text
matrix-game-3.0/
├── LICENSE_LONGCAT
├── files/
│   ├── scripts/
│   │   ├── parse_benchmark_logs.py
│   │   └── run_accel_preset.sh
│   └── wan/modules/
│       ├── bi_sparse_operation.py
│       └── longcat_kernel.py
├── patches/
│   └── 0001-integrate-acceleration.patch
└── scripts/
    ├── apply_patch.sh
    ├── check_matrix_assets.py
    ├── check_matrix_env.py
    ├── download_matrix_assets.py
    ├── setup_matrix_release.sh
    └── setup_matrix_env.sh
```

`files/` contains new Light Interaction modules and benchmark helpers. The lightweight integration diff at `patches/0001-integrate-acceleration.patch` updates upstream Matrix-Game-3.0 entry points, interactive pipeline code, DiT code, and camera context-selection logic.

## 📦 Upstream Resources

| Resource | Link |
|:---|:---|
| Matrix-Game-3.0 code | <https://github.com/SkyworkAI/Matrix-Game> |
| Matrix-Game-3.0 checkpoints | <https://huggingface.co/Skywork/Matrix-Game-3.0> |
| Matrix-Game-3.0 project page | <https://matrix-game-v3.github.io/> |

## 🚀 Install the Adapter

The recommended path is one command after cloning the upstream repository:

```bash
git clone https://github.com/SkyworkAI/Matrix-Game.git ../Matrix-Game
git -C ../Matrix-Game checkout 71c3cd7f741311f8100f6cf9cde942b6c1378d11
bash matrix-game-3.0/scripts/setup_matrix_release.sh ../Matrix-Game/Matrix-Game-3
conda activate light-interaction-matrix
```

This installs the adapter, creates the Matrix-Game-3.0 runtime environment, downloads the model assets, and runs environment/asset checks.

For reproducibility, use the tested upstream commit above. Newer upstream commits may work, but integration compatibility is not guaranteed.

This release was tested on upstream Matrix-Game-3.0 commit:

```text
71c3cd7f741311f8100f6cf9cde942b6c1378d11
```

Required model assets:

- `base_distilled_model/diffusion_pytorch_model.safetensors`
- `models_t5_umt5-xxl-enc-bf16.pth`
- `Wan2.2_VAE.pth`
- `MG-LightVAE.pth` and `MG-LightVAE_v2.pth`

If you already manage environments or checkpoints manually, use `apply_patch.sh`, `setup_matrix_env.sh`, `download_matrix_assets.py`, and the two check scripts separately.

To avoid redistributing upstream-derived files, we provide lightweight integration diffs that are applied to an official Matrix-Game-3.0 checkout.

## ▶️ Run

```bash
cd ../Matrix-Game/Matrix-Game-3
conda activate light-interaction-matrix
bash run_light_interaction.sh
```

For a quick smoke test before full 8-interaction generation:

```bash
MG_NUM_ITERATIONS=2 bash run_light_interaction.sh
```

Useful `run_accel_preset.sh` overrides:

| Variable | Default |
|:---|:---|
| `MG_PROMPT` | Built-in official Matrix-Game-3.0 demo prompt |
| `MG_IMAGE` | `demo_images/001/image.png` |
| `MG_CKPT_DIR` | `./Matrix-Game-3.0` |
| `MG_NUM_GPUS` | `1` |
| `MG_NUM_ITERATIONS` | `4` |
| `MG_NUM_INFERENCE_STEPS` | `3` |
| `MG_ACTION_SEQUENCE` | left actions for the first half, right actions for the second half |
| `MG_OUTPUT_DIR` | `./output` |

The release helper keeps `MG_NUM_ITERATIONS=4` for quick demos. The evaluation batch script uses 8 iterations for the documented left/right and forward/backward protocol.

## 🎛️ Presets

The adapted helper uses `MG_PRESET=all` through `run_light_interaction.sh`.

| Preset | Behavior |
|:---|:---|
| `off` | Original dense attention and upstream context behavior |
| `context` | Adaptive FOV-based memory pruning and compact context sequence |
| `sparse` | 3D block sparse attention |
| `cache` | FOV-conditioned denoising reuse |
| `all` | Enables all Light Interaction components |

To switch presets:

```bash
MG_PRESET=context bash run_light_interaction.sh
```

## 🔧 Advanced Notes

- Use `MATRIX_ENV_BACKEND=venv` with `setup_matrix_release.sh` if conda is unavailable.
- Use `MATRIX_ASSET_OUTPUT_ROOT=/path/to/models` to store downloaded weights outside the Hugging Face cache.
- Use `MATRIX_MODEL_REPO=<repo-id>` if the official Hugging Face repo id changes or a mirror is required.
- Use `HF_TOKEN=<token>` if gated Hugging Face assets require authentication.
- CPU-only environments can run package checks, but Matrix-Game-3.0 inference requires NVIDIA GPUs.

## ⏱️ Inference Log

The Light Interaction-enabled entry point prints an inference summary:

```text
Matrix-Game-3.0 Light Interaction Benchmark
==================================================
Preset                : all
Model Load Time       :   64.98 s
Video Generation Time :   78.96 s
End-to-End Time       :  144.00 s
--------------------------------------------------
Peak VRAM Allocated   :   35.03 GB
Peak VRAM Reserved    :   49.55 GB
==================================================
```

The interactive pipeline also prints the DiT core time:

```text
DiT Core Time: 37.56 s
```

## 🔧 Diagnostics

Relative-L1 denoising diagnostics are disabled by default. Enable them only for development:

```bash
--enable_step_diff_debug
```

For Triton autotuning experiments:

```bash
TRITON_REEVALUATE_KEY=1 bash scripts/run_accel_preset.sh all
```

The release keeps one tested Triton config by default to avoid first-run autotuning noise.

## 📄 License Notes

This folder stores adapter files and lightweight integration diffs instead of full upstream files. Upstream Matrix-Game-3.0 files and model weights remain governed by their own licenses. `longcat_kernel.py` includes attribution for LongCat-Video-derived code; see [../NOTICE.md](../NOTICE.md).
