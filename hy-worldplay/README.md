# HY-WorldPlay Acceleration Patch

This repository contains only our acceleration patch for HY-WorldPlay. It does
not redistribute the original HY-WorldPlay repository, model codebase, or model
weights. Users should first obtain the upstream project and checkpoints, then
apply this patch on top.

## Upstream Resources

- HY-WorldPlay GitHub: <https://github.com/Tencent-Hunyuan/HY-WorldPlay>
- HY-WorldPlay checkpoints: <https://huggingface.co/tencent/HY-WorldPlay>
- HunyuanVideo-1.5 checkpoints: <https://huggingface.co/tencent/HunyuanVideo-1.5>

Check and follow the upstream repositories' license terms before using or
redistributing any upstream code or weights.

## What Is Included

New files copied into the upstream checkout:

```text
files/
  worldplay_acceleration_config.py
  hyvideo/models/transformers/modules/ar_sparse_operation.py
  hyvideo/models/transformers/modules/bi_sparse_operation_for_KV_cache.py
  hyvideo/models/transformers/modules/longcat_kernel.py
```

Upstream files modified by the integration patch:

```text
patches/
  0001-integrate-acceleration.patch
    run.sh
    hyvideo/commons/__init__.py
    hyvideo/generate.py
    hyvideo/utils/retrieval_context.py
    hyvideo/pipelines/worldplay_video_pipeline.py
    hyvideo/models/transformers/worldplay_1_5_transformer.py
    hyvideo/models/transformers/modules/attention.py
```

Helper script:

```text
scripts/
  apply_patch.sh
```

The `files/` directory contains our standalone acceleration modules. The patch
modifies the upstream pipeline, transformer, attention module, `generate.py`,
`run.sh`, `hyvideo/commons/__init__.py`, and the context-selection helper so
those modules are used by the original HY-WorldPlay runtime.

We keep large upstream files as a patch instead of redistributing full
replacement copies, because those files are derived from the original
HY-WorldPlay repository and may be subject to the upstream license.

## Apply The Patch

Clone or download the upstream repository first:

```bash
git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay.git
```

Then apply this patch repository to that checkout:

```bash
cd HY-WorldPlay-Acceleration-Patch
bash scripts/apply_patch.sh /path/to/HY-WorldPlay
```

If the target tree has local edits, inspect them first. The script copies files
from `files/` and then applies `patches/0001-integrate-acceleration.patch`.

## Run

After applying the patch and setting model paths in `run.sh`, run:

```bash
cd /path/to/HY-WorldPlay
bash run.sh
```

The patched `run.sh` also accepts environment variables for model paths:

```bash
export HY_MODEL_PATH=/path/to/HunyuanVideo-1.5
export HY_AR_DISTILL_ACTION_MODEL_PATH=/path/to/HY-WorldPlay/ar_distilled_action_model/diffusion_pytorch_model.safetensors
bash run.sh
```

The default preset enables all acceleration components:

```bash
--acceleration_preset all
```

## Presets

```text
off:
  disables all acceleration components.
  Uses dense attention, the upstream temporal context setting, no FOV filtering,
  and no denoising cache.

context:
  enables only context management.
  Uses temporal context reduction plus spatial/FOV context filtering, while
  keeping dense attention and disabling the denoising cache.

sparse:
  enables only soft-hard cooperative 3D sparse attention.
  Applies sparse attention in both autoregressive denoising and KV-cache
  recomputation, while keeping the upstream context setting and disabling the
  denoising cache.

cache:
  enables only denoising cache acceleration.
  Keeps dense attention and the upstream context setting, while enabling
  FOV-based denoising-step reuse.

all:
  enables all three components.
  This is the default setting: context management, 3D sparse attention, and
  denoising cache acceleration are all enabled.
```

## Latency Reporting

The final log prints one compact `[Inference Summary]` table. For paper-style
latency reporting, use:

```text
DiT Core (KV + Denoise) = KV Cache Recompute + Denoise Loop
```

The table reports:

```text
KV Cache Recompute
Denoise Loop
DiT Core (KV + Denoise)
Transformer AR Rollout
VAE Pixel Decoding
Total End-to-End Time
Peak VRAM
```

## Triton Autotuning

The LongCat sparse-attention adapter is in:

```text
hyvideo/models/transformers/modules/longcat_kernel.py
```

The release keeps a single tested Triton config by default:

```python
triton.Config({}, num_stages=3, num_warps=8)
```

If you add multiple entries to `configs_fwd_bsa_align`, Triton will benchmark
the candidates and cache the fastest one. The first run can include autotune
warmup overhead. For stable latency numbers, keep the single tested config or
discard the first autotuned warmup run.

To re-evaluate the autotune choice per shape:

```bash
TRITON_REEVALUATE_KEY=1 bash run.sh
```

## Diagnostics

Relative-L1 denoising diagnostics are disabled by default. Enable them only for
development:

```bash
--enable_l1_diagnostics true
```

## Release Notes

This patch repository is intended to contain only our acceleration code and
integration patch. It intentionally omits upstream assets, generated videos,
logs, debug data, checkpoints, and benchmark outputs.
