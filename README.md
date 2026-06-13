<h1 align="center">Light Interaction: Training-Free Inference Acceleration for Interactive Video World Models</h1>

<h3 align="center">
Training-free acceleration for autoregressive interactive video generation
</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2605.31158"><b>📄 Paper</b></a> |
  <a href="https://2843721358l-del.github.io/Light-Interaction-Project/"><b>🌐 Project Page</b></a> |
  <a href="hy-worldplay/README.md"><b>🔌 HY-WorldPlay Patch</b></a> |
  <a href="evaluation/README.md"><b>📊 Evaluation</b></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.31158"><img src="https://img.shields.io/static/v1?label=arXiv&message=2605.31158&color=b31b1b&logo=arxiv"></a> &ensp;
  <a href="LICENSE"><img src="https://img.shields.io/static/v1?label=License&message=MIT&color=green"></a>
</p>

**Light Interaction** accelerates autoregressive interactive video generation without model retraining. It combines:

- Adaptive context management
- Denoising cache acceleration
- Hardware-software co-designed 3D block sparse attention

This repository currently releases the **HY-WorldPlay integration patch** and **evaluation scripts**. Matrix-Game-3.0 results are reported in the paper, but its integration patch is not included in this release.

<p align="center">
  <img src="asset/teaser.png" width="90%" alt="Light Interaction overview"/>
</p>

## 🔥 News

- 🔥 [2026/06] HY-WorldPlay acceleration patch and evaluation utilities released.
- 📄 [2026/05] Paper released on arXiv.

## 💡 Highlights

- **Training-free**: no retraining or fine-tuning is required.
- **Plug-in patch release**: applies to an upstream HY-WorldPlay checkout instead of redistributing upstream code.
- **Preset-based ablations**: `off`, `context`, `sparse`, `cache`, and `all`.
- **Reproducible evaluation**: fixed 200-prompt sample set with PSNR / SSIM / LPIPS and VBench helpers.

## 📦 Released Components

```text
Light-Interaction-Project/
├── hy-worldplay/      # Patch package for upstream HY-WorldPlay
├── evaluation/        # Fixed prompts, initial images, and evaluation scripts
├── asset/             # Project figures
├── NOTICE.md          # Third-party and upstream license notes
└── LICENSE
```

> [!IMPORTANT]
> This repository does not redistribute upstream source trees, checkpoints, generated videos, or benchmark outputs. Users must obtain HY-WorldPlay and model weights from the official upstream sources.

## 📋 Requirements

Recommended runtime:

- Linux
- NVIDIA GPU with CUDA support
- Python 3.10
- PyTorch 2.6.0 / torchvision 0.21.0 / torchaudio 2.6.0
- `patch` command-line utility

HY-WorldPlay inference and evaluation use separate environments. Create the patched HY-WorldPlay runtime from a clean upstream checkout:

```bash
git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay.git

bash hy-worldplay/scripts/apply_patch.sh /path/to/HY-WorldPlay
bash hy-worldplay/scripts/setup_worldplay_env.sh /path/to/HY-WorldPlay
conda activate light-interaction-worldplay
python hy-worldplay/scripts/check_worldplay_env.py --worldplay-root /path/to/HY-WorldPlay --require-cuda
```

If conda channels are unavailable, use:

```bash
WORLDPLAY_ENV_BACKEND=venv bash hy-worldplay/scripts/setup_worldplay_env.sh /path/to/HY-WorldPlay
source .venv-light-interaction-worldplay/bin/activate
```

For evaluation metrics, use the standalone evaluation environment rather than mixing metric packages into the HY-WorldPlay runtime:

```bash
bash evaluation/scripts/setup_evaluation_env.sh
conda activate light-interaction-eval
python evaluation/scripts/check_evaluation_env.py
```

This installs PSNR / SSIM / LPIPS / VBench dependencies and prepares the default evaluation metric assets in `~/.cache/light-interaction/`.

If Hugging Face access is slow, use a mirror endpoint:

```bash
HF_ENDPOINT=https://hf-mirror.com bash evaluation/scripts/setup_evaluation_env.sh
```

If conda channels are unavailable on your machine, use the venv backend:

```bash
EVAL_ENV_BACKEND=venv bash evaluation/scripts/setup_evaluation_env.sh
source .venv-light-interaction-eval/bin/activate
```

See [evaluation/README.md](evaluation/README.md) for CUDA wheel options and VBench notes.

## 🚀 Quick Start

### 1. Clone Light Interaction

```bash
git clone https://github.com/2843721358l-del/Light-Interaction-Project.git
cd Light-Interaction-Project
```

### 2. Prepare HY-WorldPlay

Clone upstream HY-WorldPlay, apply the Light Interaction patch, and create the runtime environment:

```bash
git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay.git
bash hy-worldplay/scripts/apply_patch.sh /path/to/HY-WorldPlay
bash hy-worldplay/scripts/setup_worldplay_env.sh /path/to/HY-WorldPlay
conda activate light-interaction-worldplay
```

Download the required checkpoints from:

- HY-WorldPlay: <https://huggingface.co/tencent/HY-WorldPlay>
- HunyuanVideo-1.5: <https://huggingface.co/tencent/HunyuanVideo-1.5>

The patch script performs a dry run before copying files. This release was tested on upstream HY-WorldPlay commit `1588e1336e842b03b0a7860c654ebd7c46bb065e`.

Check that the environment and model paths are complete:

```bash
python hy-worldplay/scripts/check_worldplay_env.py \
  --worldplay-root /path/to/HY-WorldPlay \
  --require-cuda

python hy-worldplay/scripts/check_worldplay_assets.py \
  --worldplay-root /path/to/HY-WorldPlay \
  --model-path /path/to/HunyuanVideo-1.5 \
  --action-ckpt /path/to/HY-WorldPlay/ar_distilled_action_model/diffusion_pytorch_model.safetensors
```

## ▶️ Run Inference

```bash
cd /path/to/HY-WorldPlay

export HY_MODEL_PATH=/path/to/HunyuanVideo-1.5
export HY_AR_DISTILL_ACTION_MODEL_PATH=/path/to/HY-WorldPlay/ar_distilled_action_model/diffusion_pytorch_model.safetensors

# Optional: use one GPU. Upstream run.sh may default to multi-GPU inference.
export HY_N_INFERENCE_GPU=1

bash run.sh
```

For a fast smoke test before full generation:

```bash
HY_N_INFERENCE_GPU=1 \
HY_NUM_FRAMES=13 \
HY_POSE='left-3' \
HY_OUTPUT_PATH=/tmp/light-interaction-worldplay-smoke-output \
bash run.sh
```

The default preset is `all`, which enables all acceleration components. To reproduce ablations, edit the `--acceleration_preset` argument in `run.sh`.

| Preset | Enabled components |
|:---|:---|
| `off` | No Light Interaction acceleration |
| `context` | Adaptive context management |
| `sparse` | 3D block sparse attention |
| `cache` | Denoising cache acceleration |
| `all` | Context management + sparse attention + denoising cache |

More patch details are in [hy-worldplay/README.md](hy-worldplay/README.md).

## 📊 Evaluation

The evaluation folder contains 200 fixed prompts and 200 initial images.

Create the standalone evaluation environment once:

```bash
bash evaluation/scripts/setup_evaluation_env.sh
conda activate light-interaction-eval
python evaluation/scripts/check_evaluation_env.py
```

Use the patched HY-WorldPlay environment to generate videos, then switch to `light-interaction-eval` to compute PSNR / SSIM / LPIPS and VBench scores. The setup step prepares LPIPS and VBench metric assets; rerun `python evaluation/scripts/prepare_evaluation_assets.py` if you move or clear the cache.

### Batch Generation

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

See the paper for full Matrix-Game-3.0 results, ablations, and metric definitions.

## 📚 Documentation Map

| Document | Contents |
|:---|:---|
| [hy-worldplay/README.md](hy-worldplay/README.md) | Patch structure, apply script, presets, timing log, diagnostics |
| [evaluation/README.md](evaluation/README.md) | Sample set, batch generation, metrics, VBench |
| [NOTICE.md](NOTICE.md) | Upstream scope, third-party attribution, license notes |

## ✅ Release Status

- [x] HY-WorldPlay acceleration patch
- [x] 3D block sparse attention backend with Triton kernels
- [x] Acceleration presets for reproduction and ablation
- [x] Latency and peak-memory reporting
- [x] Evaluation scripts and fixed 200-prompt sample set
- [ ] Matrix-Game-3.0 acceleration patch
- [ ] Additional upstream model support

## 🧪 Reproducibility Notes

For each experiment, record:

- Upstream HY-WorldPlay commit hash
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

The Light Interaction patch code is released under the repository [MIT License](LICENSE). Upstream projects and model weights are governed by their own licenses and usage terms.
