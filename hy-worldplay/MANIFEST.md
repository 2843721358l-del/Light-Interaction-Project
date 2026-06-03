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

## Why Use A Patch

The pipeline, transformer, attention module, context-selection helper,
`hyvideo/generate.py`, and `run.sh` are upstream-derived files. To avoid
redistributing the full upstream codebase, this repository stores only the
integration diff for those files. Users should obtain the upstream repository
first, then apply this patch.
