# Copyright 2026 Jiacheng Lu and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""User-facing inference acceleration configuration for HY-WorldPlay.

Provides five presets (off, context, sparse, cache, all) for ablation studies
and a runtime override API for fine-grained control.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Preset overrides: each key maps to a dict of non-default field values.
# Used by ``WorldPlayAccelerationConfig.from_preset`` to avoid repetitive code.
# ---------------------------------------------------------------------------
_PRESET_OVERRIDES: Dict[str, dict] = {
    "off": {
        "enable_sparse_attention": False,
        "enable_prefill_sparse": False,
        "enable_decode_sparse": False,
        "context_mode": "temporal",
        "context_memory_frames": 20,
        "context_temporal_size": 12,
        "enable_spatial_context_filter": False,
        "enable_denoise_cache": False,
        "enable_timing": False,
    },
    "context": {
        "enable_sparse_attention": False,
        "enable_prefill_sparse": False,
        "enable_decode_sparse": False,
        "enable_denoise_cache": False,
    },
    "sparse": {
        "context_mode": "temporal",
        "context_memory_frames": 20,
        "context_temporal_size": 12,
        "enable_spatial_context_filter": False,
        "enable_denoise_cache": False,
    },
    "cache": {
        "enable_sparse_attention": False,
        "enable_prefill_sparse": False,
        "enable_decode_sparse": False,
        "context_mode": "temporal",
        "context_memory_frames": 20,
        "context_temporal_size": 12,
        "enable_spatial_context_filter": False,
    },
    "all": {},
}


@dataclass
class WorldPlayAccelerationConfig:
    """Inference acceleration switches for HY-WorldPlay.

    The default values correspond to the ``all`` preset (every component
    enabled).  Use :meth:`from_preset` to build an alternative configuration
    for ablation studies, and :meth:`apply_overrides` for per-run tweaks via
    the CLI.
    """

    # ---- preset tag (informational) --------------------------------------
    preset: str = "all"

    # ---- sparse attention -------------------------------------------------
    #: Enable the hardware-software co-designed 3D block sparse attention.
    enable_sparse_attention: bool = True
    #: Apply sparse attention during the bidirectional prefill phase.
    enable_prefill_sparse: bool = True
    #: Apply sparse attention during the autoregressive decode phase.
    enable_decode_sparse: bool = True
    #: First chunk (0-indexed) where decode sparse attention is activated.
    sparse_decode_start_chunk: int = 3
    #: Top-k retention ratio for the prefill sparse index (higher = denser).
    sparse_prefill_topk_ratio: float = 0.3
    #: Top-k retention ratio for the decode sparse index (higher = denser).
    sparse_decode_topk_ratio: float = 0.175
    #: 3D block shape as (T, H, W).
    sparse_tile_size: Tuple[int, int, int] = (4, 8, 4)
    #: Number of most-recent chunks always kept in the sparse mask.
    sparse_recent_protect: int = 4

    # ---- adaptive context management --------------------------------------
    #: Context selection mode: "spatial_temporal", "temporal", or "none".
    context_mode: str = "spatial_temporal"
    #: Maximum number of historical frames retained for spatial retrieval.
    context_memory_frames: int = 12
    #: Recent temporal window size (in latent units) for local conditioning.
    context_temporal_size: int = 4
    #: Gate retrieved spatial memory by camera-pose-aware similarity.
    enable_spatial_context_filter: bool = True
    #: Cosine-similarity threshold for spatial memory gating.
    spatial_similarity_threshold: float = 0.7
    #: Number of predicted future latents used for context construction.
    context_pred_latent_size: int = 4

    # ---- denoising cache acceleration -------------------------------------
    #: Reuse early-step model outputs for intermediate denoising steps.
    enable_denoise_cache: bool = True
    #: Denoising step indices whose outputs are reused (0-indexed).
    denoise_cache_steps: Tuple[int, ...] = (1, 2)
    #: Pose-similarity threshold that gates denoising-cache activation.
    denoise_cache_similarity_threshold: float = 0.7

    # ---- diagnostics ------------------------------------------------------
    #: Print per-phase timing information after inference.
    enable_timing: bool = True
    #: Compute and log relative-L1 denoising diagnostics (dev-only).
    enable_l1_diagnostics: bool = False

    # ------------------------------------------------------------------
    # Factory & helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(cls, preset: str) -> "WorldPlayAccelerationConfig":
        """Build a configuration from a named preset.

        Args:
            preset: One of ``off``, ``context``, ``sparse``, ``cache``, ``all``.

        Returns:
            A fully-initialised ``WorldPlayAccelerationConfig``.

        Raises:
            ValueError: If *preset* is not recognised.
        """
        preset = (preset or "all").lower()
        overrides = _PRESET_OVERRIDES.get(preset)
        if overrides is None:
            raise ValueError(
                f"Unknown acceleration preset '{preset}'. "
                "Choose from: off, context, sparse, cache, all."
            )
        cfg = cls(preset=preset)
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    def apply_overrides(
        self,
        *,
        enable_sparse_attention=None,
        enable_prefill_sparse=None,
        enable_decode_sparse=None,
        context_mode=None,
        context_memory_frames=None,
        context_temporal_size=None,
        enable_spatial_context_filter=None,
        spatial_similarity_threshold=None,
        enable_denoise_cache=None,
        enable_l1_diagnostics=None,
    ) -> "WorldPlayAccelerationConfig":
        """Apply per-field keyword overrides in-place.

        Only non-``None`` values are applied; all other fields are left
        unchanged.  Consistency guards ensure derived flags stay in sync
        (e.g., disabling sparse attention also disables prefill & decode
        sparse).

        Returns:
            ``self`` for chaining.
        """
        for name, value in {
            "enable_sparse_attention": enable_sparse_attention,
            "enable_prefill_sparse": enable_prefill_sparse,
            "enable_decode_sparse": enable_decode_sparse,
            "context_mode": context_mode,
            "context_memory_frames": context_memory_frames,
            "context_temporal_size": context_temporal_size,
            "enable_spatial_context_filter": enable_spatial_context_filter,
            "spatial_similarity_threshold": spatial_similarity_threshold,
            "enable_denoise_cache": enable_denoise_cache,
            "enable_l1_diagnostics": enable_l1_diagnostics,
        }.items():
            if value is not None:
                setattr(self, name, value)

        # ---- consistency guards -------------------------------------------
        if not self.enable_sparse_attention:
            self.enable_prefill_sparse = False
            self.enable_decode_sparse = False
        if self.context_mode in ("none", "temporal"):
            self.enable_spatial_context_filter = False
        return self

    def to_attn_param(self) -> Dict[str, object]:
        """Export the subset of fields consumed by the sparse-attention backend.

        Returns:
            A flat dict suitable for passing as ``**kwargs`` to the attention
            module configuration.
        """
        return {
            "enable_sparse": self.enable_sparse_attention,
            "enable_prefill_sparse": self.enable_sparse_attention and self.enable_prefill_sparse,
            "enable_decode_sparse": self.enable_sparse_attention and self.enable_decode_sparse,
            "sparse_decode_start_chunk": self.sparse_decode_start_chunk,
            "sparse_prefill_topk_ratio": self.sparse_prefill_topk_ratio,
            "sparse_decode_topk_ratio": self.sparse_decode_topk_ratio,
            "sparse_tile_size": list(self.sparse_tile_size),
            "sparse_recent_protect": self.sparse_recent_protect,
            "enable_denoise_cache": self.enable_denoise_cache,
            "denoise_cache_steps": tuple(self.denoise_cache_steps),
            "enable_l1_diagnostics": self.enable_l1_diagnostics,
        }
