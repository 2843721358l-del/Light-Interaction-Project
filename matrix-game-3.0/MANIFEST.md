# Patch Manifest

This repository is intended to be applied on top of an upstream Matrix-Game-3.0
checkout.

## New Or Replaced Files

These files are copied directly from `files/` into the target checkout:

```text
wan/modules/bi_sparse_operation.py
wan/modules/longcat_kernel.py
scripts/run_accel_preset.sh
scripts/parse_benchmark_logs.py
```

## Upstream Files Modified By Patch

These upstream files are modified by
`patches/0001-integrate-acceleration.patch`:

```text
generate.py
pipeline/inference_interactive_pipeline.py
utils/cam_utils.py
wan/modules/model.py
```

## Helper Scripts

These scripts are not copied into the upstream checkout. Run them from the
Light Interaction repository and pass the Matrix-Game-3.0 checkout path:

```text
matrix-game-3.0/scripts/apply_patch.sh
matrix-game-3.0/scripts/download_matrix_assets.py
matrix-game-3.0/scripts/setup_matrix_release.sh
matrix-game-3.0/scripts/setup_matrix_env.sh
matrix-game-3.0/scripts/check_matrix_env.py
matrix-game-3.0/scripts/check_matrix_assets.py
```

## Why Use A Patch

The entry point, interactive pipeline, DiT model, and camera-selection helper
are upstream-derived files. To avoid redistributing the full upstream codebase,
this repository stores only the integration diff for those files. Users should
obtain the upstream repository first, then apply this patch.

## License Attribution

`LICENSE_LONGCAT` preserves the MIT license notice for the LongCat-Video-derived
`longcat_kernel.py`.

`bi_sparse_operation.py` is authored by the Light Interaction authors for
Matrix-Game-3.0 autoregressive sparse attention.
