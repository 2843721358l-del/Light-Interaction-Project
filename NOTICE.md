# Notice

This repository is the official code release hub for Light Interaction. It is
intended to host model-specific support files, standalone acceleration modules, and
evaluation utilities.

## Repository Scope

This repository does not redistribute original HY-WorldPlay or Matrix-Game-3.0
source trees, model checkpoints, datasets, generated videos, benchmark outputs,
or private development logs. Users must obtain upstream projects and checkpoints
separately and comply with the corresponding upstream licenses, acceptable-use
policies, and model terms.

Some files in this repository are model-specific patch files against upstream projects.
Those patch files are provided to make our changes reproducible, but they do not
grant any additional rights to the upstream projects, model weights, datasets,
or checkpoints.

## Light Interaction Code

Standalone files authored by the Light Interaction authors are released under
the repository LICENSE unless otherwise stated.

Model-specific patch files may modify or refer to upstream projects. The upstream files
remain governed by their original licenses. Our modifications are provided for
research reproduction under the repository LICENSE where permitted by the
upstream license terms.

## HY-WorldPlay

The HY-WorldPlay support files provide the model-specific changes required to apply
Light Interaction to an official upstream HY-WorldPlay checkout. This repository
does not redistribute the full HY-WorldPlay source tree or model weights. Users must
obtain HY-WorldPlay from the official source and comply with the Tencent HY-WorldPlay
Community License Agreement and its acceptable-use policy.

## Matrix-Game-3.0

The Matrix-Game-3.0 support files provide the model-specific changes required to apply
Light Interaction to an official upstream Matrix-Game-3.0 checkout. This repository
does not redistribute the full Matrix-Game-3.0 source tree or model weights. Users must
obtain Matrix-Game-3.0 from the official source and comply with its upstream license terms.

## LongCat-Video

The file:

hy-worldplay/files/hyvideo/models/transformers/modules/longcat_kernel.py
matrix-game-3.0/files/wan/modules/longcat_kernel.py

is adapted from the LongCat-Video sparse-attention implementation and should
preserve the original LongCat-Video MIT license notice.

Other sparse-attention implementation files, including:

hy-worldplay/files/hyvideo/models/transformers/modules/ar_sparse_operation.py
hy-worldplay/files/hyvideo/models/transformers/modules/bi_sparse_operation_for_KV_cache.py
matrix-game-3.0/files/wan/modules/bi_sparse_operation.py

are authored by the Light Interaction authors for autoregressive sparse
attention scheduling and KV-cache recomputation.
