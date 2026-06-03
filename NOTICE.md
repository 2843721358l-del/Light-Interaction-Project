# Notice

This repository is the official code release hub for Light Interaction. It is
intended to host our acceleration patches, standalone acceleration modules, and
evaluation utilities.

## Repository Scope

This repository does not redistribute original HY-WorldPlay or Matrix-Game
source trees, model checkpoints, datasets, generated videos, benchmark outputs,
or private development logs. Users must obtain upstream projects and checkpoints
separately and comply with the corresponding upstream licenses, acceptable-use
policies, and model terms.

Some files in this repository are integration patches against upstream projects.
Those patches are provided to make our changes reproducible, but they do not
grant any additional rights to the upstream projects, model weights, datasets,
or checkpoints.

## Light Interaction Code

Standalone files authored by the Light Interaction authors are released under
the repository LICENSE unless otherwise stated.

Integration patches may modify or refer to upstream projects. The upstream files
remain governed by their original licenses. Our modifications are provided for
research reproduction under the repository LICENSE where permitted by the
upstream license terms.

## HY-WorldPlay

The HY-WorldPlay integration is released as a patch-style package. This
repository does not redistribute the full HY-WorldPlay source tree or model
weights. Users must obtain HY-WorldPlay from the official source and comply with
the Tencent HY-WorldPlay Community License Agreement and its acceptable-use
policy.

## Matrix-Game-3.0

Matrix-Game-3.0 integration is planned for a future release. This repository
does not currently redistribute Matrix-Game-3.0 source trees or model weights.
Users must obtain Matrix-Game-3.0 from the official source and comply with its
upstream license terms.

## LongCat-Video

The file:

hy-worldplay/files/hyvideo/models/transformers/modules/longcat_kernel.py

is adapted from the LongCat-Video sparse-attention implementation and should
preserve the original LongCat-Video MIT license notice.

Other sparse-attention integration files, including:

hy-worldplay/files/hyvideo/models/transformers/modules/ar_sparse_operation.py
hy-worldplay/files/hyvideo/models/transformers/modules/bi_sparse_operation_for_KV_cache.py

are authored by the Light Interaction authors for autoregressive sparse
attention scheduling and KV-cache recomputation.
