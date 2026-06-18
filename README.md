<h1 align="center">Light Interaction: Training-Free Inference Acceleration for Interactive Video World Models</h1>

<h3 align="center">
Training-free acceleration for autoregressive interactive video generation
</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2605.31158"><b>📄 Paper</b></a> |
  <a href="https://2843721358l-del.github.io/Light-Interaction-Project/"><b>🌐 Project Page</b></a> |
  <a href="#-demo"><b>🎥 Demo</b></a> |
  <a href="hy-worldplay/README.md"><b>🔌 HY-WorldPlay Support</b></a> |
  <a href="matrix-game-3.0/README.md"><b>🎮 Matrix-Game-3.0 Support</b></a> |
  <a href="evaluation/README.md"><b>📊 Evaluation</b></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.31158"><img src="https://img.shields.io/static/v1?label=arXiv&message=2605.31158&color=b31b1b&logo=arxiv"></a> &ensp;
  <a href="LICENSE"><img src="https://img.shields.io/static/v1?label=License&message=MIT&color=green"></a>
</p>

## 🎥 Demo

<p align="center">
  <video src="https://github.com/user-attachments/assets/b140505b-3608-41cd-b996-cf0490045eca" width="95%"> </video>
</p>

## 💡 Introduction

**Light Interaction** is a training-free inference acceleration framework for autoregressive interactive video world models. Interactive systems such as HY-WorldPlay and Matrix-Game-3.0 generate videos chunk by chunk under user-controlled camera trajectories, but long rollouts are expensive because context memory grows over time, attention scales quadratically, and the transformer is repeatedly executed during denoising.

The key idea is to make computation adapt to interaction dynamics. Light Interaction prunes unreliable spatial memory according to camera-pose-aware retrieval similarity, adjusts temporal context windows based on local latent dynamics, reuses early denoising outputs when revisiting familiar regions, and uses an AR-aware 3D block sparse attention backend to turn algorithmic sparsity into practical speedup.

This repository provides the Light Interaction framework with tested model-specific support for **HY-WorldPlay** and **Matrix-Game-3.0**, along with reproducible setup scripts, fixed-prompt generation helpers, and evaluation utilities.

For license compatibility and faithful upstream reproduction, model-specific changes are applied to official upstream checkouts through patch files during setup.

## 🔥 News

- 📄 [2026/05] Paper released on arXiv.
- 🔥 [2026/06] HY-WorldPlay support released.
- 🔥 [2026/06] Matrix-Game-3.0 support released.
- 📊 [2026/06] Evaluation utilities released.

## 💡 Highlights

- **Training-free**: no retraining or fine-tuning is required.
- **Framework-level acceleration**: adaptive context management, denoising cache acceleration, and AR-aware sparse attention are designed as reusable acceleration components for interactive video world models.
- **Model-specific support**: the release provides tested support for HY-WorldPlay and Matrix-Game-3.0 through setup scripts and patch files applied to official upstream checkouts.
- **Preset-based ablations**: `off`, `context`, `sparse`, `cache`, and `all`.
- **Reproducible evaluation**: fixed 200-prompt sample set with PSNR / SSIM / LPIPS and VBench helpers.

## 📦 Released Components

```text
Light-Interaction-Project/
├── hy-worldplay/      # HY-WorldPlay support files
├── matrix-game-3.0/  # Matrix-Game-3.0 support files
├── evaluation/        # Fixed prompt metadata, external asset manifest, and evaluation scripts
├── asset/             # Project figures
├── NOTICE.md          # Third-party and upstream license notes
└── LICENSE
```

> [!IMPORTANT]
> This repository does not redistribute upstream source trees, checkpoints, generated videos, or benchmark outputs. Users must obtain HY-WorldPlay, Matrix-Game-3.0, and model weights from the official upstream sources.

## 📋 Requirements

- Linux
- NVIDIA GPU with CUDA support
- Python 3.10 for HY-WorldPlay and Python 3.12 for Matrix-Game-3.0
- `patch` command-line utility

HY-WorldPlay inference, Matrix-Game-3.0 inference, and evaluation use separate environments. The setup scripts below install the required packages and run import checks.

## 🚀 Quick Start

### 1. Clone Light Interaction

```bash
git clone https://github.com/2843721358l-del/Light-Interaction-Project.git
cd Light-Interaction-Project
```

### 2. Prepare HY-WorldPlay

Clone upstream HY-WorldPlay, then prepare the Light Interaction-enabled runtime and minimal model assets with one command:

```bash
git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay.git ../HY-WorldPlay
git -C ../HY-WorldPlay checkout 1588e1336e842b03b0a7860c654ebd7c46bb065e
bash hy-worldplay/scripts/setup_worldplay_release.sh ../HY-WorldPlay
```

This downloads only the assets used by this release: HunyuanVideo-1.5 runtime files and the few-step distilled autoregressive HY-WorldPlay action checkpoint. Bidirectional, multi-step AR, and RL action models are not required.

### 3. Prepare Matrix-Game-3.0

Clone the upstream repository, then prepare the Light Interaction-enabled Matrix-Game-3.0 runtime with one command:

```bash
git clone https://github.com/SkyworkAI/Matrix-Game.git ../Matrix-Game
git -C ../Matrix-Game checkout 71c3cd7f741311f8100f6cf9cde942b6c1378d11
bash matrix-game-3.0/scripts/setup_matrix_release.sh ../Matrix-Game/Matrix-Game-3
```

For reproducibility, use the tested upstream commits above. Newer upstream commits may work, but patch compatibility is not guaranteed.

This downloads the required Matrix-Game-3.0 assets from the official Hugging Face repository, including the distilled base model, T5 text encoder, Wan VAE, and Matrix-Game LightVAE checkpoints.

## ▶️ Run Inference

```bash
cd ../HY-WorldPlay
conda activate light-interaction-worldplay
bash run_light_interaction.sh
```

The default preset is `all`, which enables all acceleration components. To reproduce ablations, edit the `--acceleration_preset` argument in `run.sh`.

| Preset | Enabled components |
|:---|:---|
| `off` | No Light Interaction acceleration |
| `context` | Adaptive context management |
| `sparse` | 3D block sparse attention |
| `cache` | Denoising cache acceleration |
| `all` | Context management + sparse attention + denoising cache |

More details are in [hy-worldplay/README.md](hy-worldplay/README.md).

For Matrix-Game-3.0:

```bash
cd ../Matrix-Game/Matrix-Game-3
conda activate light-interaction-matrix
bash run_light_interaction.sh
```

More details are in [matrix-game-3.0/README.md](matrix-game-3.0/README.md).

## 📊 Evaluation

Create the standalone evaluation environment once:

```bash
bash evaluation/scripts/setup_evaluation_env.sh
conda activate light-interaction-eval
```

Evaluation initial images are selected from the official VBench-I2V 16:9 cropped image suite. This repository does not redistribute the images; run the downloader below to fetch the official VBench-I2V assets and extract the selected 200 images.

```bash
python evaluation/scripts/download_eval_assets.py
```

Then use the Light Interaction-enabled HY-WorldPlay or Matrix-Game-3.0 environment to generate videos, and switch to `light-interaction-eval` for PSNR / SSIM / LPIPS and VBench.

### Batch Generation

```bash
python evaluation/scripts/batch_video_generation.py \
  --backend hy-worldplay \
  --prompt-json evaluation/data/refined_prompts_llava16.json \
  --hy-worldplay-root /path/to/HY-WorldPlay \
  --model-path /path/to/HunyuanVideo-1.5 \
  --action-ckpt /path/to/ar_distilled_action_model/model.safetensors \
  --output-root outputs/fixed_prompt \
  --allowed-gpus 0,1,2,3 \
  --acceleration-preset all
```

For Matrix-Game-3.0, use the same script with `--backend matrix-game`, `--matrix-game-root`, and `--ckpt-dir`. Full examples are in [evaluation/README.md](evaluation/README.md).

### PSNR / SSIM / LPIPS

```bash
python evaluation/scripts/evaluate_psnr_ssim_lpips.py \
  --run-mutual \
  --ref-dir outputs/baseline/left_right \
  --test-dir outputs/light_interaction/left_right \
  --output-dir evaluation_results \
  --tag left_right
```

### VBench

```bash
python evaluation/scripts/evaluate_vbench_batch.py \
  --prompt-json evaluation/data/refined_prompts_llava16.json \
  --video-dir outputs/light_interaction/left_right \
  --output-csv evaluation_results/vbench_left_right.csv
```

More evaluation details are in [evaluation/README.md](evaluation/README.md).

## 🏆 Main Results

HY-WorldPlay, 480P image-to-video, single A100:

| Method | PSNR vs. Original | VBench | Latency (s) | Speedup | Peak VRAM (GB) |
|:---|---:|---:|---:|---:|---:|
| Original | - | 0.8190 | 228.60 | 1.00x | 76.57 |
| SVG | 19.48 | 0.8082 | 247.65 | 0.92x | 77.86 |
| BSA | 15.94 | 0.7943 | 474.57 | 0.48x | 75.03 |
| TeaCache | 20.90 | 0.8150 | 203.25 | 1.12x | 76.64 |
| Light Interaction | 24.81 | 0.8220 | 88.24 | 2.59x | 54.66 |

Matrix-Game-3.0, 720P image-to-video, single A100:

| Method | PSNR vs. Original | VBench | Latency (s) | Speedup | Peak VRAM (GB) |
|:---|---:|---:|---:|---:|---:|
| Original | - | 0.7432 | 59.70 | 1.00x | 35.04 |
| SVG | 12.98 | 0.7511 | 96.16 | 0.62x | 35.02 |
| BSA | 13.34 | 0.7336 | 63.26 | 0.94x | 35.03 |
| TeaCache | 19.03 | 0.7146 | 41.49 | 1.44x | 35.32 |
| Light Interaction | 17.76 | 0.7350 | 37.07 | 1.61x | 35.04 |

See the paper for full ablations and metric definitions.

Numbers are measured with the documented tested upstream commits and single-A100 settings. They may vary with GPU type, CUDA/PyTorch/Triton versions, and upstream changes.

## 📚 Documentation Map

| Document | Contents |
|:---|:---|
| [hy-worldplay/README.md](hy-worldplay/README.md) | HY-WorldPlay support files, installation script, presets, timing log, diagnostics |
| [matrix-game-3.0/README.md](matrix-game-3.0/README.md) | Matrix-Game-3.0 support files, setup script, presets, timing log, diagnostics |
| [evaluation/README.md](evaluation/README.md) | Sample set, batch generation, metrics, VBench |
| [NOTICE.md](NOTICE.md) | Upstream scope, third-party attribution, license notes |

## ✅ Release Status

- [x] HY-WorldPlay support release
- [x] Matrix-Game-3.0 support release
- [x] 3D block sparse attention backend with Triton kernels
- [x] Acceleration presets for reproduction and ablation
- [x] Latency and peak-memory reporting
- [x] Evaluation scripts, fixed 200-prompt JSON, and external image-asset manifest
- [ ] Additional upstream model support

## 🧪 Reproducibility Notes

For each experiment, record:

- Upstream HY-WorldPlay commit hash
- Upstream Matrix-Game-3.0 commit hash, if using Matrix-Game-3.0
- Light Interaction commit hash
- GPU type and count
- CUDA, PyTorch, and Triton versions
- Acceleration preset
- Prompt JSON path and action group

Generated videos, logs, CSV files, checkpoints, and debug dumps are intentionally ignored by git.

## 📖 Citation

```bibtex
@misc{lu2026lightinteractiontrainingfreeinference,
  title={Light Interaction: Training-Free Inference Acceleration for Interactive Video World Models},
  author={Jiacheng Lu and Haoyi Zhu and Sipei Yi and Enze Xie and Yu Li and Cheng Zhuo},
  year={2026},
  eprint={2605.31158},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.31158}
}
```

## 🤗 Acknowledgements

This project builds on HY-WorldPlay, HunyuanVideo-1.5, LongCat-Video, Matrix-Game-3.0, VBench, Sparse VideoGen, and TeaCache. See [NOTICE.md](NOTICE.md) for license and attribution details.

## 📄 License

The original Light Interaction code is released under the [MIT License](LICENSE). Model-specific patch files may modify or refer to upstream projects, and upstream projects, model weights, datasets, and third-party-derived files remain governed by their own licenses and usage terms. See [NOTICE.md](NOTICE.md) for details.
