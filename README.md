# Light Interaction

Training-free inference acceleration for interactive video world models.

This repository collects the code patches and evaluation utilities for Light
Interaction. The project accelerates autoregressive interactive video generation
without model retraining by combining:

- Adaptive context management
- Soft-hard cooperative 3D sparse attention
- Denoising cache acceleration

The repository does not redistribute upstream model repositories or model
weights. Users should first obtain the original projects and checkpoints, then
apply the corresponding patches in this repository.

## Repository Layout

```text
hy-worldplay/
  HY-WorldPlay integration patch and standalone acceleration modules.

matrix-game/
  Matrix-Game integration patch. Coming soon.

evaluation/
  Quality and benchmark evaluation scripts. Coming soon.
```

## HY-WorldPlay

The HY-WorldPlay patch is available in:

```text
hy-worldplay/
```

It contains:

- Standalone acceleration modules
- An integration patch for upstream HY-WorldPlay files
- A helper script to apply the patch
- Documentation for presets and latency reporting

Quick start:

```bash
git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay.git
cd Light-Interaction-Project/hy-worldplay
bash scripts/apply_patch.sh /path/to/HY-WorldPlay
```

Then configure model paths in the patched HY-WorldPlay checkout and run:

```bash
cd /path/to/HY-WorldPlay
bash run.sh
```

See [hy-worldplay/README.md](hy-worldplay/README.md) for details.

## Acceleration Presets

The HY-WorldPlay patch exposes five presets for ablation studies:

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

## Upstream Resources

- HY-WorldPlay: <https://github.com/Tencent-Hunyuan/HY-WorldPlay>
- HY-WorldPlay checkpoints: <https://huggingface.co/tencent/HY-WorldPlay>
- HunyuanVideo-1.5 checkpoints: <https://huggingface.co/tencent/HunyuanVideo-1.5>

## License And Upstream Notice

This repository contains our patch code and integration diffs. It does not
redistribute upstream repositories, checkpoints, or generated assets. Upstream
projects and model weights are governed by their own licenses.

See [NOTICE.md](NOTICE.md) for details.

## Citation

If you use this repository, please cite the Light Interaction paper. Citation
information will be added here when the final BibTeX entry is available.
