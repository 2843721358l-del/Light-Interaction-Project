# 📊 Evaluation Utilities

This folder contains the fixed sample set and scripts used for Light Interaction quantitative evaluation.

```text
evaluation/
├── data/
│   ├── refined_prompts_llava16.json
│   └── sampled_200/
├── requirements.txt
└── scripts/
    ├── batch_video_generation.py
    ├── evaluate_psnr_ssim_lpips.py
    └── evaluate_vbench_batch.py
```

Generated videos, CSV files, logs, and debug files are not included.

## 🛠️ Install Dependencies

Use the same Python environment as the patched HY-WorldPlay runtime when generating videos.

```bash
pip install -r evaluation/requirements.txt
```

VBench should usually be installed in its own environment because it pulls additional model and metric dependencies.

## 🖼️ Dataset

`data/refined_prompts_llava16.json` contains 200 entries. Each entry has:

| Field | Description |
|:---|:---|
| `filename` | Initial image filename |
| `original_prompt` | Original VBench prompt |
| `refined_prompt` | Prompt used for I2V generation |
| `image_path` | Repository-relative image path |

`data/sampled_200/` contains the corresponding 200 initial images.

## 🎬 Batch Generation

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

Default action groups:

```text
left_right=left-5, right-5.5
forward_backward=w-5, s-5.5
```

Use `--actions name=pose` to provide custom action groups. Existing non-empty output folders are skipped by default; pass `--no-skip-existing` to regenerate them.

## 📐 PSNR / SSIM / LPIPS

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

The scripts print missing-video and decode-failure counts so incomplete runs are visible.

## 🏷️ VBench

```bash
python evaluation/scripts/evaluate_vbench_batch.py \
  --prompt-json evaluation/data/refined_prompts_llava16.json \
  --video-dir outputs/light_interaction/left_right \
  --output-csv evaluation_results/vbench_left_right.csv
```

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
