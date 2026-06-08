<h1 align="center">Light Interaction</h1>

<h4 align="center">Training-Free Inference Acceleration for Interactive Video World Models</h4>

<h3 align="center">
<a href="https://arxiv.org/abs/2605.31158"><b>📄 Paper</b></a> | <a href="https://github.com/lujiacheng/Light-Interaction-Project"><b>💻 GitHub</b></a> | <a href="https://arxiv.org/abs/2605.31158"><b>🌐 Project Page</b></a>
</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2605.31158"><img src="https://img.shields.io/static/v1?label=arXiv&message=2605.31158&color=b31b1b&logo=arxiv"></a> &ensp;
  <a href="https://github.com/lujiacheng/Light-Interaction-Project"><img src="https://img.shields.io/static/v1?label=Code&message=GitHub&color=black&logo=github"></a> &ensp;
  <a href="LICENSE"><img src="https://img.shields.io/static/v1?label=License&message=Apache%202.0&color=green"></a> &ensp;
  <img src="https://img.shields.io/static/v1?label=Platform&message=Linux&color=blue&logo=linux">
</p>

**Light Interaction** is a training-free inference acceleration framework for interactive video world models. It accelerates autoregressive interactive video generation without model retraining by combining **Adaptive Context Management**, **Denoising Cache Acceleration**, and **Hardware-Software Co-designed 3D Sparse Attention**. On a single A100 GPU, Light Interaction achieves up to **2.59× speedup** on HY-WorldPlay and **1.61× speedup** on Matrix-Game-3.0, while maintaining competitive visual quality with **24.81 PSNR** against the original model.

<p align="center" border-radius="10px">
  <img src="asset/teaser.png" width="90%" alt="teaser_page"/>
</p>

## 🔥 News

- 🔥 [2026/06] 🚀 **Light Interaction code & paper are released!** Training-free inference acceleration for HY-WorldPlay and Matrix-Game-3.0. [[Paper]](https://arxiv.org/abs/2605.31158) | [[GitHub]](https://github.com/lujiacheng/Light-Interaction-Project)
- 🔥 [2026/06] 📄 **Paper** is on ArXiv! Check out the details at [arXiv:2605.31158](https://arxiv.org/abs/2605.31158).

<details>
  <summary>Click to show all updates</summary>

- ✅ [2026/06] HY-WorldPlay acceleration patch released. Includes standalone acceleration modules, integration patch, preset-based ablation switches, and latency / memory reporting.
- ✅ [2026/06] Evaluation utilities released. Includes fixed 200-prompt sample set, batch generation helpers, PSNR / SSIM / LPIPS evaluation, and selected VBench evaluation.
- ✅ [2026/06] Sparse attention backend released with Triton fused kernels for autoregressive interactive video generation.

</details>

## 💡 Introduction

We introduce **Light Interaction**, a training-free inference acceleration framework for interactive video world models. Interactive video world models generate video chunk by chunk in response to user-controlled camera movements, paving the way toward real-time game simulation, virtual scene navigation, and embodied AI training. However, scaling to long interactive trajectories is prohibitively expensive due to growing context memory, quadratic attention complexity, and repeated denoising steps.

**Key Insight:** Interaction naturally enables adaptive computation — the usefulness of different computation evolves with interaction dynamics. Pose-aware retrieval similarity can gate spatial memory, local latent dynamics can adapt temporal context, and early-step outputs can be reused during revisiting.

**Key Techniques:**

- **Adaptive Context Management**: Prunes spatial memory by camera-pose-aware similarity and adjusts temporal windows according to local latent dynamics. Distinguishes novel exploration (discard irrelevant retrieved memory) from trajectory revisiting (retain useful historical views).
- **Denoising Cache Acceleration**: Reuses early-step model outputs for intermediate denoising steps when camera-pose-aware similarity indicates reliable revisiting, while preserving the final step for quality correction.
- **Hardware-Software Co-designed 3D Block Sparse Attention**: Preserves text and current-chunk tokens, sparsifies only historical visual KV blocks, and uses fused Triton kernels to eliminate layout-conversion and gather/scatter overhead under autoregressive causal constraints.

**In summary**, Light Interaction is a **training-free, plug-and-play** acceleration framework that can be applied on top of existing interactive video world models. It requires **no model retraining** and achieves significant speedup with competitive visual quality.

## 🚀 Quick Start

```bash
# Clone the upstream HY-WorldPlay repository
git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay.git

# Clone Light Interaction and apply the acceleration patch
git clone https://github.com/lujiacheng/Light-Interaction-Project.git
cd Light-Interaction-Project/hy-worldplay
bash scripts/apply_patch.sh /path/to/HY-WorldPlay

# Configure model paths and run
cd /path/to/HY-WorldPlay
export HY_MODEL_PATH=/path/to/HunyuanVideo-1.5
export HY_AR_DISTILL_ACTION_MODEL_PATH=/path/to/HY-WorldPlay/ar_distilled_action_model/diffusion_pytorch_model.safetensors
bash run.sh
```

> [!TIP]
> The patched `run.sh` supports five acceleration presets: `off`, `context`, `sparse`, `cache`, `all`. The default preset `all` enables all three acceleration components.

## 📚 Getting Started

- [Installation & Patch Application](hy-worldplay/README.md)
- [Acceleration Presets Guide](hy-worldplay/README.md#presets)
- [Latency & Memory Reporting](hy-worldplay/README.md#latency-reporting)
- [Sparse Attention Backend](hy-worldplay/README.md#triton-autotuning)
- [Evaluation Scripts](evaluation/README.md)
- [Upstream Resources (HY-WorldPlay, HunyuanVideo-1.5)](hy-worldplay/README.md#upstream-resources)

## 📊 Performance

### HY-WorldPlay (480P, Image-to-Video)

| Method | vs. Original | | | Self-Comparison | | | VBench↑ | Latency↓ (s) | Speedup↑ | Mem.↓ (GB) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | **PSNR↑** | **SSIM↑** | **LPIPS↓** | **PSNR↑** | **SSIM↑** | **LPIPS↓** | | | | |
| Original | – | – | – | 18.60 | 0.5678 | 0.2051 | 0.8190 | 228.60 | 1.00× | 76.57 |
| SVG | 19.48 | 0.6028 | 0.2209 | 17.75 | 0.5299 | 0.2187 | 0.8082 | 247.65 | 0.92× | 77.86 |
| BSA | 15.94 | 0.4639 | 0.3755 | 15.44 | 0.4205 | 0.3720 | 0.7943 | 474.57 | 0.48× | 75.03 |
| TeaCache | 20.90 | 0.6588 | 0.1892 | 18.86 | 0.5743 | 0.2054 | 0.8150 | 203.25 | 1.12× | 76.64 |
| **Ours** | **24.81** | **0.6500** | **0.1788** | **18.85** | **0.5854** | **0.1963** | **0.8220** | **88.24** | **2.59×** | **54.66** |

### Matrix-Game-3.0 (720P, Image-to-Video)

| Method | vs. Original | | | Self-Comparison | | | VBench↑ | Latency↓ (s) | Speedup↑ | Mem.↓ (GB) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | **PSNR↑** | **SSIM↑** | **LPIPS↓** | **PSNR↑** | **SSIM↑** | **LPIPS↓** | | | | |
| Original | – | – | – | 15.49 | 0.4685 | 0.4048 | 0.7432 | 59.70 | 1.00× | 35.04 |
| SVG | 12.98 | 0.4170 | 0.5587 | 14.48 | 0.4949 | 0.4406 | 0.7511 | 96.16 | 0.62× | 35.02 |
| BSA | 13.34 | 0.4228 | 0.5795 | 16.66 | 0.5326 | 0.4094 | 0.7336 | 63.26 | 0.94× | 35.03 |
| TeaCache | 19.03 | 0.5619 | 0.3818 | 18.84 | 0.5765 | 0.3602 | 0.7146 | 41.49 | 1.44× | 35.32 |
| **Ours** | **17.76** | **0.5306** | **0.3692** | **14.63** | **0.4570** | **0.4424** | **0.7350** | **37.07** | **1.61×** | **35.04** |

> **vs. Original** compares each method with the original full-computation model. **Self-Comparison** compares frame pairs with similar camera poses within the same revisiting trajectory to evaluate consistency.

### Ablation Study — Individual Components (HY-WorldPlay)

| Variant | Latency↓ (s) | Speedup↑ | vs. Orig. PSNR↑ | Self-Comp. PSNR↑ | VBench↑ | Mem.↓ (GB) |
|:---|---:|---:|---:|---:|---:|---:|
| Original Model | 228.60 | 1.00× | – | 18.60 | 0.8190 | 76.57 |
| Only Context Mgmt. (Temporal) | 152.88 | 1.50× | 31.98 | 20.11 | 0.8208 | 54.66 |
| Only Context Mgmt. (Spatial) | 213.71 | 1.07× | 38.73 | 18.85 | 0.8191 | 76.57 |
| Only Context Mgmt. (Full) | 144.49 | 1.58× | 29.24 | 20.20 | 0.8210 | 54.66 |
| Only Denoising Cache | 198.10 | 1.15× | 55.16 | 19.02 | 0.8199 | 76.57 |
| Only 3D Sparse Attn. | 153.69 | 1.49× | 25.53 | 18.27 | 0.8208 | 76.57 |
| **Full Light Interaction** | **88.24** | **2.59×** | **24.81** | **18.85** | **0.8220** | **54.66** |

## 🎛️ Acceleration Presets

The HY-WorldPlay patch exposes five presets for reproduction and ablation studies:

| Preset | Description |
|:---|:---|
| `off` | All acceleration components disabled. Uses dense attention, upstream temporal context, no FOV filtering, no denoising cache. |
| `context` | Only Adaptive Context Management enabled. Temporal context reduction + spatial/FOV context filtering, dense attention, no denoising cache. |
| `sparse` | Only Soft-Hard Cooperative 3D Sparse Attention enabled. Sparse attention in both AR denoising and KV-cache recomputation, upstream context setting, no denoising cache. |
| `cache` | Only Denoising Cache Acceleration enabled. Dense attention, upstream context setting, FOV-based denoising-step reuse. |
| `all` | **Default.** All three components enabled: context management + 3D sparse attention + denoising cache acceleration. |

## 📂 Repository Structure

```text
Light-Interaction-Project/
├── README.md                    # Project overview (this file)
├── NOTICE.md                    # Third-party notices
├── LICENSE                      # Apache 2.0
├── asset/                       # Teaser and figures
├── hy-worldplay/                # HY-WorldPlay acceleration patch
│   ├── README.md                # Detailed patch documentation
│   ├── files/                   # Standalone acceleration modules
│   ├── patches/                 # Integration patch for upstream files
│   └── scripts/                 # Patch application helper
└── evaluation/                  # Evaluation scripts and benchmarks
    ├── README.md
    ├── data/                    # Fixed 200-prompt evaluation set
    └── scripts/                 # PSNR / SSIM / LPIPS / VBench scripts
```

## ✅ To-Do List

- [✅] HY-WorldPlay acceleration patch (adaptive context mgmt. + 3D sparse attn. + denoising cache)
- [✅] Hardware-software co-designed 3D block sparse attention with Triton fused kernels
- [✅] Five acceleration presets for ablation studies
- [✅] Latency and peak memory reporting infrastructure
- [✅] Evaluation scripts (PSNR, SSIM, LPIPS, VBench)
- [✅] Fixed 200-prompt evaluation set
- [ ] Matrix-Game-3.0 acceleration patch (integration patch under cleaning)
- [ ] Additional upstream model support
- [🚀] See you in the future

## 🤗 Acknowledgements

This project builds upon and thanks the following open-source projects:

- [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) — Interactive video world model
- [HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5) — Video generation foundation model
- [LongCat-Video](https://github.com/Meituan-Dianping/LongCat-Video) — Sparse attention backbone
- [Matrix-Game-3.0](https://github.com/Matrix-Game/Matrix-Game-3.0) — Interactive world model
- [VBench](https://github.com/Vchitect/VBench) — Video generation benchmark
- [Sparse VideoGen (SVG)](https://github.com/mit-han-lab/SparseVideoGen) — Sparse attention baseline
- [TeaCache](https://github.com/LiewFeng/TeaCache) — Denoising cache baseline

We thank the authors and contributors of these projects for releasing their code, models, and tools to the community.

## 📖 BibTeX

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

## 📄 License

This repository contains our patch code and integration diffs. It does **not** redistribute upstream repositories, checkpoints, or generated assets. Upstream projects and model weights are governed by their own licenses.

- HY-WorldPlay has its own upstream license terms. Users must obtain HY-WorldPlay from the official source and comply with its license and acceptable-use policy.
- Parts of the sparse-attention backend are adapted from LongCat-Video and are subject to the original LongCat-Video MIT license notice.

See [NOTICE.md](NOTICE.md) for details.
