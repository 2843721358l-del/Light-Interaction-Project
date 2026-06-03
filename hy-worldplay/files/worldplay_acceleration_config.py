from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class WorldPlayAccelerationConfig:
    """User-facing inference acceleration switches for HY-WorldPlay."""

    preset: str = "all"

    # Sparse attention.
    enable_sparse_attention: bool = True
    enable_prefill_sparse: bool = True
    enable_decode_sparse: bool = True
    sparse_decode_start_chunk: int = 3
    sparse_prefill_topk_ratio: float = 0.3
    sparse_decode_topk_ratio: float = 0.175
    sparse_tile_size: Tuple[int, int, int] = (4, 8, 4)
    sparse_recent_protect: int = 4

    # History/context selection.
    context_mode: str = "spatial_temporal"
    context_memory_frames: int = 12
    context_temporal_size: int = 4
    enable_spatial_context_filter: bool = True
    spatial_similarity_threshold: float = 0.7
    context_pred_latent_size: int = 4

    # Denoising cache / FOV reuse.
    enable_denoise_cache: bool = True
    denoise_cache_steps: Tuple[int, ...] = (1, 2)
    denoise_cache_similarity_threshold: float = 0.7

    # Optional diagnostics.
    enable_timing: bool = True
    enable_l1_diagnostics: bool = False

    @classmethod
    def from_preset(cls, preset: str) -> "WorldPlayAccelerationConfig":
        preset = (preset or "all").lower()
        if preset == "off":
            return cls(
                preset="off",
                enable_sparse_attention=False,
                enable_prefill_sparse=False,
                enable_decode_sparse=False,
                context_mode="temporal",
                context_memory_frames=20,
                context_temporal_size=12,
                enable_spatial_context_filter=False,
                enable_denoise_cache=False,
                enable_timing=False,
            )
        if preset == "context":
            return cls(
                preset="context",
                enable_sparse_attention=False,
                enable_prefill_sparse=False,
                enable_decode_sparse=False,
                context_mode="spatial_temporal",
                context_memory_frames=12,
                context_temporal_size=4,
                enable_spatial_context_filter=True,
                enable_denoise_cache=False,
            )
        if preset == "sparse":
            return cls(
                preset="sparse",
                context_mode="temporal",
                context_memory_frames=20,
                context_temporal_size=12,
                enable_spatial_context_filter=False,
                enable_denoise_cache=False,
            )
        if preset == "cache":
            return cls(
                preset="cache",
                enable_sparse_attention=False,
                enable_prefill_sparse=False,
                enable_decode_sparse=False,
                context_mode="temporal",
                context_memory_frames=20,
                context_temporal_size=12,
                enable_spatial_context_filter=False,
                enable_denoise_cache=True,
            )
        if preset == "all":
            return cls(preset="all")
        raise ValueError(
            f"Unknown acceleration preset '{preset}'. "
            "Choose from: off, context, sparse, cache, all."
        )

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

        if not self.enable_sparse_attention:
            self.enable_prefill_sparse = False
            self.enable_decode_sparse = False
        if self.context_mode == "none":
            self.enable_spatial_context_filter = False
        if self.context_mode == "temporal":
            self.enable_spatial_context_filter = False
        return self

    def to_attn_param(self) -> Dict[str, object]:
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
