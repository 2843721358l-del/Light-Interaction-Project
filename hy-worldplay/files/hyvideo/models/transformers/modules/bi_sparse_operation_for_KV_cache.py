# Copyright (c) 2026 Jiacheng Lu and contributors.
# Licensed under the MIT License.
# See the LICENSE file in the repository root for details.

import torch
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from hyvideo.commons.infer_state import get_infer_state
from hyvideo.commons.parallel_states import get_parallel_state
from hyvideo.utils.communications import all_to_all_4D

try:
    import triton
    import triton.language as tl
    from .longcat_kernel import longcat_flash_attn
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("[SparseAttn] Triton not found. Falling back to SDPA.")


# ==============================================================================
# Triton kernels.
# ==============================================================================
if HAS_TRITON:
    # --------------------------------------------------------------------------
    # Kernel 1: Q Preparation (Fusion: Padding + Tiling + Mean Pooling)
    # --------------------------------------------------------------------------
    @triton.jit
    def _fused_q_prep_kernel(
        Src_ptr, Dst_ptr, Mean_ptr,
        s_b, s_h, s_l, s_d, d_b, d_h, d_n, d_blk, d_d, m_b, m_h, m_n, m_d,
        NUM_HEADS, T, H_img, W_img, tt, th, tw, nh, nw,
        BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_SIZE_FLOAT: tl.constexpr
    ):
        blk_idx, pid_bh = tl.program_id(0), tl.program_id(1)
        cur_head, cur_batch = pid_bh % NUM_HEADS, pid_bh // NUM_HEADS
        
        # Pointers
        src_base = Src_ptr + cur_batch * s_b + cur_head * s_h
        dst_base = Dst_ptr + cur_batch * d_b + cur_head * d_h
        mean_base = Mean_ptr + cur_batch * m_b + cur_head * m_h
        
        # Grid Logic (Mapping 1D block index back to 3D coordinates)
        cur_nw = blk_idx % nw
        rem = blk_idx // nw
        cur_nh, cur_nt = rem % nh, rem // nh
        
        base_t, base_h, base_w = cur_nt * tt, cur_nh * th, cur_nw * tw
        
        # Inner Loop
        offs_flat = tl.arange(0, BLOCK_SIZE)
        offs_tw = offs_flat % tw
        rem_t = offs_flat // tw
        offs_th, offs_tt = rem_t % th, rem_t // th
        
        coords_t = base_t + offs_tt
        coords_h = base_h + offs_th
        coords_w = base_w + offs_tw
        
        # Boundary Check (Padding Logic Fused Here)
        mask = (coords_t < T) & (coords_h < H_img) & (coords_w < W_img)
        idx_l = coords_t * (H_img * W_img) + coords_h * W_img + coords_w
        offs_d = tl.arange(0, HEAD_DIM)
        
        # Load (Implicit Padding with other=0.0)
        val = tl.load(src_base + idx_l[:, None] * s_l + offs_d[None, :] * s_d, mask=mask[:, None], other=0.0)
        
        # Store Tiled
        dst_off = blk_idx * d_n + offs_flat[:, None] * d_blk + offs_d[None, :] * d_d
        tl.store(dst_base + dst_off, val)
        
        # Mean Pooling (Fused)
        mean_val = tl.sum(val.to(tl.float32), axis=0) / BLOCK_SIZE_FLOAT
        tl.store(mean_base + blk_idx * m_n + offs_d * m_d, mean_val.to(val.dtype))

    # --------------------------------------------------------------------------
    # Kernel 2: KV Preparation (Fusion: Padding + Tiling + Mean Pooling + Concat Offset)
    # --------------------------------------------------------------------------
    @triton.jit
    def _fused_kv_prep_kernel(
        SK_ptr, SV_ptr, DK_ptr, DV_ptr, MK_ptr,
        sk_b, sk_h, sk_l, sk_d, sv_b, sv_h, sv_l, sv_d,
        dk_b, dk_h, dk_n, dk_blk, dk_d, dv_b, dv_h, dv_n, dv_blk, dv_d,
        m_b, m_h, m_n, m_d,
        NUM_HEADS, T, H_img, W_img, tt, th, tw, nh, nw, 
        dst_offset_blk, pool_offset_blk,
        BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_SIZE_FLOAT: tl.constexpr
    ):
        blk_idx, pid_bh = tl.program_id(0), tl.program_id(1)
        cur_head, cur_batch = pid_bh % NUM_HEADS, pid_bh // NUM_HEADS
        
        # Base Pointers
        sk_base, sv_base = SK_ptr + cur_batch * sk_b + cur_head * sk_h, SV_ptr + cur_batch * sv_b + cur_head * sv_h
        dk_base, dv_base = DK_ptr + cur_batch * dk_b + cur_head * dk_h, DV_ptr + cur_batch * dv_b + cur_head * dv_h
        mk_base = MK_ptr + cur_batch * m_b + cur_head * m_h
        
        # Grid Logic
        cur_nw = blk_idx % nw
        rem = blk_idx // nw
        cur_nh, cur_nt = rem % nh, rem // nh
        
        offs_flat = tl.arange(0, BLOCK_SIZE)
        offs_tw = offs_flat % tw
        rem_t = offs_flat // tw
        offs_th, offs_tt = rem_t % th, rem_t // th
        
        coords_t = cur_nt * tt + offs_tt
        coords_h = cur_nh * th + offs_th
        coords_w = cur_nw * tw + offs_tw
        
        mask = (coords_t < T) & (coords_h < H_img) & (coords_w < W_img)
        idx_l = coords_t * (H_img * W_img) + coords_h * W_img + coords_w
        offs_d = tl.arange(0, HEAD_DIM)
        
        # Load
        val_k = tl.load(sk_base + idx_l[:, None] * sk_l + offs_d[None, :] * sk_d, mask=mask[:, None], other=0.0)
        val_v = tl.load(sv_base + idx_l[:, None] * sv_l + offs_d[None, :] * sv_d, mask=mask[:, None], other=0.0)
        
        # Store Tiled (Notice dst_offset_blk for Text skipping)
        dst_off = (blk_idx + dst_offset_blk) * dk_n + offs_flat[:, None] * dk_blk + offs_d[None, :] * dk_d
        tl.store(dk_base + dst_off, val_k)
        tl.store(dv_base + dst_off, val_v)
        
        # Store K Mean (Separate buffer, usually no offset needed relative to itself)
        # But if mean buffer includes text, we need offset. Here we assume Mean buffer is VISUAL ONLY for Prefill logic usually.
        # However, to be safe, we use pool_offset_blk if needed.
        tl.store(mk_base + (blk_idx + pool_offset_blk) * m_n + offs_d * m_d, (tl.sum(val_k.to(tl.float32), axis=0) / BLOCK_SIZE_FLOAT).to(val_k.dtype))

    # --------------------------------------------------------------------------
    # Kernel 3: Untile (Fusion: Scatter + Padding Removal)
    # --------------------------------------------------------------------------
    @triton.jit
    def _fused_untile_kernel(
        Src_ptr, Dst_ptr,
        s_b, s_h, s_n, s_blk, s_d, d_b, d_h, d_l, d_d,
        NUM_HEADS, T, H_img, W_img, tt, th, tw, nh, nw,
        BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr
    ):
        blk_idx, pid_bh = tl.program_id(0), tl.program_id(1)
        cur_head, cur_batch = pid_bh % NUM_HEADS, pid_bh // NUM_HEADS
        
        src_base = Src_ptr + cur_batch * s_b + cur_head * s_h
        dst_base = Dst_ptr + cur_batch * d_b + cur_head * d_h
        
        cur_nw = blk_idx % nw
        rem = blk_idx // nw
        cur_nh, cur_nt = rem % nh, rem // nh
        
        offs_flat = tl.arange(0, BLOCK_SIZE)
        offs_tw = offs_flat % tw
        rem_t = offs_flat // tw
        offs_th, offs_tt = rem_t % th, rem_t // th
        
        coords_t = cur_nt * tt + offs_tt
        coords_h = cur_nh * th + offs_th
        coords_w = cur_nw * tw + offs_tw
        
        mask = (coords_t < T) & (coords_h < H_img) & (coords_w < W_img)
        offs_d = tl.arange(0, HEAD_DIM)
        
        val = tl.load(src_base + blk_idx * s_n + offs_flat[:, None] * s_blk + offs_d[None, :] * s_d)
        
        idx_l = coords_t * (H_img * W_img) + coords_h * W_img + coords_w
        tl.store(dst_base + idx_l[:, None] * d_l + offs_d[None, :] * d_d, val, mask=mask[:, None])

    # --------------------------------------------------------------------------
    # Launchers (Helper functions)
    # --------------------------------------------------------------------------
    def launch_q_prep(src, dst, mean, T, H, W, tt, th, tw, nh, nw):
        B, NH, _, D = src.shape
        sb, sh, sl, sd = src.stride()
        db, dh, dn, dblk, dd = dst.stride()
        mb, mh, mn, md = mean.stride()
        num_blocks = ((T + tt - 1) // tt) * nh * nw
        BLOCK_SIZE = tt * th * tw
        _fused_q_prep_kernel[(num_blocks, B * NH)](
            src, dst, mean,
            sb, sh, sl, sd, db, dh, dn, dblk, dd, mb, mh, mn, md,
            NH, T, H, W, tt, th, tw, nh, nw,
            BLOCK_SIZE=BLOCK_SIZE, HEAD_DIM=D, BLOCK_SIZE_FLOAT=float(BLOCK_SIZE),
            num_warps=4, num_stages=2
        )

    def launch_kv_prep(k_src, v_src, k_dst, v_dst, k_mean, dst_off, pool_off, T, H, W, tt, th, tw, nh, nw):
        B, NH, _, D = k_src.shape
        skb, skh, skl, skd = k_src.stride()
        svb, svh, svl, svd = v_src.stride()
        dkb, dkh, dkn, dkblk, dkd = k_dst.stride()
        dvb, dvh, dvn, dvblk, dvd = v_dst.stride()
        mb, mh, mn, md = k_mean.stride()
        num_blocks = ((T + tt - 1) // tt) * nh * nw
        BLOCK_SIZE = tt * th * tw
        _fused_kv_prep_kernel[(num_blocks, B * NH)](
            k_src, v_src, k_dst, v_dst, k_mean,
            skb, skh, skl, skd, svb, svh, svl, svd,
            dkb, dkh, dkn, dkblk, dkd, dvb, dvh, dvn, dvblk, dvd,
            mb, mh, mn, md,
            NH, T, H, W, tt, th, tw, nh, nw, dst_off, pool_off,
            BLOCK_SIZE=BLOCK_SIZE, HEAD_DIM=D, BLOCK_SIZE_FLOAT=float(BLOCK_SIZE),
            num_warps=4, num_stages=2
        )

    def launch_untile(src, dst, T, H, W, tt, th, tw, nh, nw):
        B, NH, _, D = dst.shape
        sb, sh, sn, sblk, sd = src.stride()
        db, dh, dl, dd = dst.stride()
        num_blocks = ((T + tt - 1) // tt) * nh * nw
        BLOCK_SIZE = tt * th * tw
        _fused_untile_kernel[(num_blocks, B * NH)](
            src, dst,
            sb, sh, sn, sblk, sd, db, dh, dl, dd,
            NH, T, H, W, tt, th, tw, nh, nw,
            BLOCK_SIZE=BLOCK_SIZE, HEAD_DIM=D,
            num_warps=4
        )

# ==============================================================================
# ==============================================================================
# Sparse index generation
# ==============================================================================
def generate_prefill_indices(q_in, k_in, text_len, grid_thw, params, device):
    """Build block-sparse indices for the bidirectional prefill phase.

    Similar to the AR decode path, but adds a spatio-temporal neighbour bias
    (Manhattan distance ≤ 1) so that blocks adjacent to the query block are
    always retained.  This preserves local structure during KV-cache
    construction.

    Args:
        q_in: Query pool  ``(B, H, Nq, D)`` or tiled ``(B, H, Nq_blk, Bs, D)``.
        k_in: Key pool    ``(B, H, Nk, D)`` or tiled.
        text_len: Text token count (offsets visual indices).
        grid_thw: Grid ``(T_chunks, H_blks, W_blks)``.
        params: Dict with ``tile_size``, ``topk_ratio``, ``sim_metric``.
        device: Output device.

    Returns:
        Tuple ``(final_indices, final_counts)``.
    """
    if q_in.dim() == 5:
        q_pool = q_in.mean(dim=-2)
        BLOCK_SIZE = q_in.shape[-2]
    else:
        q_pool = q_in
        tt, th, tw = params['tile_size']
        BLOCK_SIZE = tt * th * tw

    if k_in.dim() == 5:
        k_pool = k_in.mean(dim=-2)
    else:
        k_pool = k_in
    
    HEAD_DIM = q_pool.shape[-1]
    
    metric = params.get('sim_metric', 'cosine')
    if metric == 'cosine':
        sim = torch.matmul(F.normalize(q_pool, dim=-1), F.normalize(k_pool, dim=-1).transpose(-2, -1))
    else: 
        sim = torch.matmul(q_pool, k_pool.transpose(-2, -1))
        if metric == 'scaled_dot_product':
            sim *= (1.0 / math.sqrt(HEAD_DIM))

    B, H, N_q, N_k = sim.shape
    Nt, Nh, Nw = grid_thw 
    
    score_for_ranking = sim.clone()
    LARGE_BIAS = 1e6
    
    idx_arr = torch.arange(N_k, device=device)
    t_idx = idx_arr // (Nh * Nw)
    rem = idx_arr % (Nh * Nw)
    h_idx = rem // Nw
    w_idx = rem % Nw
    
    t_q, t_k = t_idx.unsqueeze(1), t_idx.unsqueeze(0)
    h_q, h_k = h_idx.unsqueeze(1), h_idx.unsqueeze(0)
    w_q, w_k = w_idx.unsqueeze(1), w_idx.unsqueeze(0)
    
    dist = (t_q - t_k).abs() + (h_q - h_k).abs() + (w_q - w_k).abs()
    is_neighbor = dist <= 1
    score_for_ranking = score_for_ranking + is_neighbor.to(score_for_ranking.dtype) * LARGE_BIAS

    target_k = int(N_k * params['topk_ratio'])
    min_budget = 7 
    k_budget = max(target_k, min_budget)
    k_budget = min(k_budget, N_k)
    
    _, vis_indices = torch.topk(score_for_ranking, k=k_budget, dim=-1)
    
    vis_indices, _ = torch.sort(vis_indices, dim=-1) 
    num_txt_blk = (text_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    vis_indices = vis_indices + num_txt_blk 
    txt_indices = torch.arange(num_txt_blk, device=device).view(1, 1, 1, -1).expand(B, H, N_q, -1)
    final_indices = torch.cat([txt_indices, vis_indices], dim=-1).int().contiguous()
    final_counts = torch.full((B, H, N_q), final_indices.shape[-1], dtype=torch.int32, device=device)
    
    return final_indices, final_counts

# ==============================================================================
# Prefill sparse attention entry point
# ==============================================================================
def run_sparse_prefill_attention(q, k_txt, v_txt, k_vis, v_vis, block_idx, latent_hw, sparse_params):
    """Execute a single bidirectional (prefill) sparse-attention forward pass.

    The first three chunks always run dense (warm-up heuristic).  After that,
    Triton-fused tiling and sparse-index generation are applied; if Triton is
    unavailable the call falls back to full SDPA.

    Args:
        q: Query ``(B, H, Nq, D)``.
        k_txt, v_txt: Text KV  ``(B, H, N_txt, D)``.
        k_vis, v_vis: Visual KV ``(B, H, N_vis, D)``.
        block_idx: Current chunk index.
        latent_hw: ``(H_lat, W_lat)`` of the latent grid.
        sparse_params: Dict with ``tile_size``, ``topk_ratio``,
            ``recent_protect``, ``sim_metric``.

    Returns:
        Attention output ``(B, H, Nq, D)``.
    """
    infer_state = get_infer_state()
    current_global_chunk = getattr(infer_state, "current_chunk_index", 0)

    LATENT_H, LATENT_W = latent_hw
    vis_len_q = q.shape[2]
    
    text_len = k_txt.shape[2]
    vis_len_k = k_vis.shape[2]
    
    # Dense fallback: early warm-up chunks, missing Triton, or non-uniform
    # latent shape (cannot tile cleanly).
    need_dense = (
        current_global_chunk < 3
        or not HAS_TRITON
        or (vis_len_q % (LATENT_H * LATENT_W) != 0)
    )
    if need_dense:
        return F.scaled_dot_product_attention(
            q, 
            torch.cat([k_txt, k_vis], dim=2), 
            torch.cat([v_txt, v_vis], dim=2), 
            dropout_p=0.0, 
            is_causal=False
        )

    from .longcat_kernel import longcat_flash_attn 

    tt, th, tw = sparse_params.get('tile_size', (4, 8, 4))
    BLOCK_SIZE = tt * th * tw
    
    nh = (LATENT_H + (th - LATENT_H % th) % th) // th
    nw = (LATENT_W + (tw - LATENT_W % tw) % tw) // tw
    
    T_q = vis_len_q // (LATENT_H * LATENT_W)
    T_k = vis_len_k // (LATENT_H * LATENT_W)
    
    nt_q = (T_q + (tt - T_q % tt) % tt) // tt
    nt_k = (T_k + (tt - T_k % tt) % tt) // tt
    
    num_txt_blk = (text_len + (BLOCK_SIZE - text_len % BLOCK_SIZE) % BLOCK_SIZE) // BLOCK_SIZE
    num_vis_blk_q = nt_q * nh * nw
    num_vis_blk_k = nt_k * nh * nw
    B, H, _, D = q.shape
    q_tiled = torch.empty((B, H, num_vis_blk_q, BLOCK_SIZE, D), device=q.device, dtype=q.dtype)
    q_pool_vis = torch.empty((B, H, num_vis_blk_q, D), device=q.device, dtype=q.dtype)
    
    k_all = torch.empty((B, H, num_txt_blk + num_vis_blk_k, BLOCK_SIZE, D), device=q.device, dtype=q.dtype)
    v_all = torch.empty((B, H, num_txt_blk + num_vis_blk_k, BLOCK_SIZE, D), device=q.device, dtype=q.dtype)
    k_pool_vis = torch.empty((B, H, num_vis_blk_k, D), device=q.device, dtype=q.dtype) 
    pad_txt = (BLOCK_SIZE - text_len % BLOCK_SIZE) % BLOCK_SIZE
    k_txt_padded, v_txt_padded = k_txt, v_txt
    if pad_txt > 0:
        k_txt_padded = F.pad(k_txt, (0, 0, 0, pad_txt))
        v_txt_padded = F.pad(v_txt, (0, 0, 0, pad_txt))
    k_all[:, :, :num_txt_blk].copy_(k_txt_padded.view(B, H, -1, BLOCK_SIZE, D))
    v_all[:, :, :num_txt_blk].copy_(v_txt_padded.view(B, H, -1, BLOCK_SIZE, D))

    launch_q_prep(q, q_tiled, q_pool_vis, T_q, LATENT_H, LATENT_W, tt, th, tw, nh, nw)
    
    launch_kv_prep(k_vis, v_vis, k_all, v_all, k_pool_vis, num_txt_blk, 0,
                   T_k, LATENT_H, LATENT_W, tt, th, tw, nh, nw)

    indices, counts = generate_prefill_indices(
        q_pool_vis, k_pool_vis, text_len, (nt_k, nh, nw), 
        sparse_params, q.device
    )

    o_flat = longcat_flash_attn(
        q_tiled.flatten(2, 3), k_all.flatten(2, 3), v_all.flatten(2, 3), 
        indices, counts, sm_scale=1.0/(D**0.5), logical_block_size=BLOCK_SIZE
    )
    
    o_out = torch.empty_like(q)
    launch_untile(o_flat.view(B, H, num_vis_blk_q, BLOCK_SIZE, D), o_out, T_q, LATENT_H, LATENT_W, tt, th, tw, nh, nw)

    return o_out
