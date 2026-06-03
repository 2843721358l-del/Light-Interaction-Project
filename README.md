# Light Interaction

Training-free inference acceleration for interactive video world models.

This repository provides the official code release for **Light Interaction**. The project accelerates autoregressive interactive video generation without model retraining by combining:

* Adaptive context management
* Soft-hard cooperative 3D sparse attention
* Denoising cache acceleration

The current release focuses on the HY-WorldPlay integration. We provide a patch-style package that can be applied on top of the official HY-WorldPlay repository. This repository does **not** redistribute upstream model repositories, model weights, generated videos, logs, or benchmark outputs.

Users should first obtain the original upstream projects and checkpoints, then apply the corresponding patches provided here.

## Release Status

Current release status:

* **HY-WorldPlay**: released. The patch includes standalone acceleration modules, an integration patch, preset-based ablation switches, and latency / memory reporting support.
* **Matrix-Game-3.0**: planned. The integration patch is still being cleaned and will be added in a future update.
* **Evaluation utilities**: planned. Scripts for video quality evaluation, latency benchmarking, and peak-memory reporting will be added after the inference patches are finalized.

Planned updates:

```text
v0.1:
  HY-WorldPlay acceleration patch

v0.2:
  Matrix-Game-3.0 acceleration patch

v0.3:
  evaluation scripts for video quality
```

This repository focuses on reproducing the main quantitative results of the paper, including inference latency, peak memory, and video quality metrics. Intermediate debugging utilities, private development logs, generated videos, and model weights are not included.

## Repository Layout

```text
Light-Interaction-Project/
├── README.md
├── NOTICE.md
├── .gitignore
└── hy-worldplay/
    ├── README.md
    ├── MANIFEST.md
    ├── files/
    ├── patches/
    └── scripts/
```

Current modules:

```text
hy-worldplay/
  HY-WorldPlay integration patch and standalone acceleration modules.
  Status: released.

matrix-game/
  Matrix-Game-3.0 integration patch.
  Status: planned.

evaluation/
  Video quality.
  Status: planned.
```

## HY-WorldPlay Integration

The HY-WorldPlay acceleration patch is available in:

```text
hy-worldplay/
```

It contains:

* Standalone acceleration modules
* An integration patch for upstream HY-WorldPlay files
* A helper script to apply the patch
* Preset-based ablation switches
* Documentation for latency and peak-memory reporting

The patch is released in a conservative patch-style format. It does not include the full HY-WorldPlay codebase or checkpoints. Users should clone the upstream HY-WorldPlay repository first, then apply this patch.

### Upstream Resources

* HY-WorldPlay GitHub: https://github.com/Tencent-Hunyuan/HY-WorldPlay
* HY-WorldPlay checkpoints: https://huggingface.co/tencent/HY-WorldPlay
* HunyuanVideo-1.5 checkpoints: https://huggingface.co/tencent/HunyuanVideo-1.5

Please check and follow the upstream repositories' license terms before using or redistributing any upstream code or weights.

### Quick Start

Clone the upstream HY-WorldPlay repository:

```bash
git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay.git
```

Apply the Light Interaction patch:

```bash
cd Light-Interaction-Project/hy-worldplay
bash scripts/apply_patch.sh /path/to/HY-WorldPlay
```

Then configure model paths in the patched HY-WorldPlay checkout and run:

```bash
cd /path/to/HY-WorldPlay
bash run.sh
```

The patched `run.sh` also supports environment variables for model paths:

```bash
export HY_MODEL_PATH=/path/to/HunyuanVideo-1.5
export HY_AR_DISTILL_ACTION_MODEL_PATH=/path/to/HY-WorldPlay/ar_distilled_action_model/diffusion_pytorch_model.safetensors
bash run.sh
```

See [hy-worldplay/README.md](hy-worldplay/README.md) for detailed instructions.

## Acceleration Presets

The HY-WorldPlay patch exposes five presets for reproduction and ablation studies:

```text
off:
  all acceleration components disabled

context:
  only adaptive context management enabled

sparse:
  only soft-hard cooperative 3D sparse attention enabled

cache:
  only denoising cache acceleration enabled

all:
  all three components enabled
```

These presets are designed to support the main ablation settings of the paper. The default fast configuration enables all three components.

## Latency and Memory Reporting

The patched HY-WorldPlay runtime prints a compact `[Inference Summary]` table at the end of inference.

For paper-style latency reporting, use:

```text
DiT Core (KV + Denoise) = KV Cache Recompute + Denoise Loop
```

The summary reports:

```text
KV Cache Recompute
Denoise Loop
DiT Core (KV + Denoise)
Transformer AR Rollout
VAE Pixel Decoding
Total End-to-End Time
Peak VRAM
```

Runtime may vary slightly across GPU instances, driver versions, CUDA / PyTorch versions, kernel warm-up behavior, and system load. Small fluctuations in latency are expected.

## Sparse Attention Backend

The HY-WorldPlay patch adds sparse-attention integration modules under the original HY-WorldPlay module path:

```text
hyvideo/models/transformers/modules/
```

The main sparse-related files are:

```text
ar_sparse_operation.py
bi_sparse_operation_for_KV_cache.py
longcat_kernel.py
```

Among them, `longcat_kernel.py` is adapted from the LongCat-Video sparse-attention implementation and should preserve the original LongCat-Video MIT license notice. The other sparse integration files are written for Light Interaction to support autoregressive sparse attention scheduling and KV-cache recomputation.

## Diagnostics

Development diagnostics are disabled by default. For example, Relative-L1 denoising diagnostics can be enabled only when needed:

```bash
--enable_l1_diagnostics true
```

Diagnostics and debug dumps are not part of the standard reproduction path.

## What Is Not Included

This repository intentionally omits:

* Upstream model weights
* Full upstream model repositories
* Generated videos
* Logs and debug outputs
* Benchmark result files
* Private development scripts
* Intermediate debugging utilities

The release is intended to provide the clean patch code needed to reproduce the main acceleration path, not the full private experimental workspace.

## License and Upstream Notice

This repository contains our patch code and integration diffs. It does not redistribute upstream repositories, checkpoints, or generated assets. Upstream projects and model weights are governed by their own licenses.

HY-WorldPlay has its own upstream license terms. Users must obtain HY-WorldPlay from the official source and comply with its license and acceptable-use policy.

Parts of the sparse-attention backend are adapted from LongCat-Video and are subject to the original LongCat-Video MIT license notice.

See [NOTICE.md](NOTICE.md) for details.

## Citation

If you use this repository, please cite the Light Interaction paper.

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

## Acknowledgments

This project builds upon and thanks the following open-source projects:

* HY-WorldPlay
* HunyuanVideo-1.5
* LongCat-Video
* Matrix-Game-3.0
* VBench

We thank the authors and contributors of these projects for releasing their code, models, and tools to the community.
