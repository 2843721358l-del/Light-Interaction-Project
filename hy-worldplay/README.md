# 🔌 HY-WorldPlay Acceleration Patch

<p align="center">
  <a href="../README.md">← Back to Light Interaction</a>
</p>

This directory contains the **Light Interaction acceleration patch** for [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay). It does **not** redistribute the original HY-WorldPlay repository, model codebase, or model weights. Users must first obtain the upstream project and checkpoints, then apply this patch on top.

---

## 📦 Upstream Resources

| Resource | Link |
|:---|:---|
| HY-WorldPlay GitHub | <https://github.com/Tencent-Hunyuan/HY-WorldPlay> |
| HY-WorldPlay checkpoints | <https://huggingface.co/tencent/HY-WorldPlay> |
| HunyuanVideo-1.5 checkpoints | <https://huggingface.co/tencent/HunyuanVideo-1.5> |

> [!IMPORTANT]
> Check and follow the upstream repositories' license terms before using or redistributing any upstream code or weights.

---

## 📂 What Is Included

### New files copied into the upstream checkout

```text
files/
  worldplay_acceleration_config.py
  hyvideo/models/transformers/modules/ar_sparse_operation.py
  hyvideo/models/transformers/modules/bi_sparse_operation_for_KV_cache.py
  hyvideo/models/transformers/modules/longcat_kernel.py
```

### Upstream files modified by the integration patch

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

### Helper script

```text
scripts/
  apply_patch.sh
```

The `files/` directory contains standalone acceleration modules. The patch modifies the upstream pipeline, transformer, attention module, `generate.py`, `run.sh`, `hyvideo/commons/__init__.py`, and the context-selection helper so those modules are integrated into the original HY-WorldPlay runtime.

We keep large upstream files as a patch instead of redistributing full replacement copies, because those files are derived from the original HY-WorldPlay repository and may be subject to the upstream license.

---

## 🚀 Apply The Patch

**Step 1:** Clone the upstream repository:

```bash
git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay.git
```

**Step 2:** Apply this patch to the upstream checkout:

```bash
cd Light-Interaction-Project/hy-worldplay
bash scripts/apply_patch.sh /path/to/HY-WorldPlay
```

> [!NOTE]
> If the target tree has local edits, inspect them first. The script copies files from `files/` and then applies `patches/0001-integrate-acceleration.patch`.

---

## ▶️ Run

**Step 3:** After applying the patch and setting model paths, run:

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

---

## 🎛️ Presets

| Preset | Components | Behavior |
|:---|:---|:---|
| `off` | None | Dense attention, upstream temporal context, no FOV filtering, no denoising cache |
| `context` | Context Mgmt. | Temporal context reduction + spatial/FOV context filtering, dense attention, no denoising cache |
| `sparse` | Sparse Attn. | Sparse attention in both AR denoising and KV-cache recomputation, upstream context, no denoising cache |
| `cache` | Denoising Cache | Dense attention, upstream context, FOV-based denoising-step reuse |
| `all` | All three | **Default.** Context management + 3D sparse attention + denoising cache acceleration |

---

## ⏱️ Latency Reporting

The final inference log prints a compact `[Inference Summary]` table:

```text
============================================================
[Inference Summary]
============================================================
 Prompt & Vision Encode    :    15.45 s
 History Selection         :     1.29 s
 KV Cache Recompute        :   206.89 s
 Denoise Loop              :   217.26 s
 DiT Core (KV + Denoise)   :   424.15 s
 Transformer AR Rollout    :   443.99 s
 VAE Pixel Decoding        :    34.89 s
------------------------------------------------------------
 Total End-to-End Time     :   494.39 s
 Peak VRAM                 :    48.70 GB
============================================================
```

For paper-style latency reporting, use:

```text
DiT Core (KV + Denoise) = KV Cache Recompute + Denoise Loop
```

---

## ⚙️ Triton Autotuning

The LongCat sparse-attention adapter is in:

```text
hyvideo/models/transformers/modules/longcat_kernel.py
```

The release keeps a single tested Triton config by default:

```python
triton.Config({}, num_stages=3, num_warps=8)
```

If you add multiple entries to `configs_fwd_bsa_align`, Triton will benchmark the candidates and cache the fastest one. The first run may include autotune warmup overhead. For stable latency numbers, keep the single tested config or discard the first autotuned warmup run.

To re-evaluate the autotune choice per shape:

```bash
TRITON_REEVALUATE_KEY=1 bash run.sh
```

---

## 🔧 Diagnostics

Relative-L1 denoising diagnostics are disabled by default. Enable them only for development:

```bash
--enable_l1_diagnostics true
```

---

## 📄 Release Notes

This patch repository is intended to contain only our acceleration code and integration patch. It intentionally omits upstream assets, generated videos, logs, debug data, checkpoints, and benchmark outputs.
