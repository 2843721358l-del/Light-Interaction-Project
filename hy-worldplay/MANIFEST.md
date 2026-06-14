# Patch Manifest

This repository is intended to be applied on top of an upstream HY-WorldPlay
checkout.

## New Or Replaced Files

These files are copied directly from `files/` into the target checkout:

```text
worldplay_acceleration_config.py
hyvideo/models/transformers/modules/ar_sparse_operation.py
hyvideo/models/transformers/modules/bi_sparse_operation_for_KV_cache.py
hyvideo/models/transformers/modules/longcat_kernel.py
```

## Upstream Files Modified By Patch

These upstream files are modified by
`patches/0001-integrate-acceleration.patch`:

```text
run.sh
hyvideo/commons/__init__.py
hyvideo/generate.py
hyvideo/utils/retrieval_context.py
hyvideo/pipelines/worldplay_video_pipeline.py
hyvideo/models/transformers/worldplay_1_5_transformer.py
hyvideo/models/transformers/modules/attention.py
```

## Helper Scripts

These scripts are not copied into the upstream checkout. Run them from the
Light Interaction repository and pass the HY-WorldPlay checkout path:

```text
hy-worldplay/scripts/apply_patch.sh
hy-worldplay/scripts/download_minimal_worldplay_assets.py
hy-worldplay/scripts/setup_worldplay_env.sh
hy-worldplay/scripts/check_worldplay_env.py
hy-worldplay/scripts/check_worldplay_assets.py
```

## Why Use A Patch

The pipeline, transformer, attention module, context-selection helper,
`hyvideo/generate.py`, and `run.sh` are upstream-derived files. To avoid
redistributing the full upstream codebase, this repository stores only the
integration diff for those files. Users should obtain the upstream repository
first, then apply this patch.

## License Attribution

`LICENSE_LONGCAT` preserves the MIT license notice for the LongCat-Video-derived
`longcat_kernel.py`.

`ar_sparse_operation.py` and `bi_sparse_operation_for_KV_cache.py` are authored
by the Light Interaction authors.
