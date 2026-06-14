"""Block-sparse attention utilities for Light Interaction Matrix-Game-3.0.

Authored for the Light Interaction adaptation. The Triton kernel entry point is
implemented in ``longcat_kernel.py``.
"""

import torch
import torch.nn.functional as F
import math

# Lightweight profiling switch for sparse-attention development.
# Keep disabled by default to avoid affecting release performance.
ENABLE_SPARSE_PROFILING = False

try:
    import triton
    import triton.language as tl
    from .longcat_kernel import longcat_flash_attn
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("[SparseAttn] Triton not found. Falling back to SDPA.")

if HAS_TRITON:
    @triton.jit
    def _fused_q_prep_kernel(
        Src_ptr, Dst_ptr, Mean_ptr,
        s_b, s_h, s_l, s_d, d_b, d_h, d_n, d_blk, d_d, m_b, m_h, m_n, m_d,
        NUM_HEADS, T, H_img, W_img, tt, th, tw, nh, nw,
        BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_SIZE_FLOAT: tl.constexpr
    ):
        blk_idx, pid_bh = tl.program_id(0), tl.program_id(1)
        cur_head, cur_batch = pid_bh % NUM_HEADS, pid_bh // NUM_HEADS
        
        src_base = Src_ptr + cur_batch * s_b + cur_head * s_h
        dst_base = Dst_ptr + cur_batch * d_b + cur_head * d_h
        mean_base = Mean_ptr + cur_batch * m_b + cur_head * m_h
        
        cur_nw = blk_idx % nw
        rem = blk_idx // nw
        cur_nh, cur_nt = rem % nh, rem // nh
        
        base_t, base_h, base_w = cur_nt * tt, cur_nh * th, cur_nw * tw
        
        offs_flat = tl.arange(0, BLOCK_SIZE)
        offs_tw = offs_flat % tw
        rem_t = offs_flat // tw
        offs_th, offs_tt = rem_t % th, rem_t // th
        
        coords_t = base_t + offs_tt
        coords_h = base_h + offs_th
        coords_w = base_w + offs_tw
        
        mask = (coords_t < T) & (coords_h < H_img) & (coords_w < W_img)
        idx_l = coords_t * (H_img * W_img) + coords_h * W_img + coords_w
        offs_d = tl.arange(0, HEAD_DIM)
        
        val = tl.load(src_base + idx_l[:, None] * s_l + offs_d[None, :] * s_d, mask=mask[:, None], other=0.0)
        
        dst_off = blk_idx * d_n + offs_flat[:, None] * d_blk + offs_d[None, :] * d_d
        tl.store(dst_base + dst_off, val)
        
        mean_val = tl.sum(val.to(tl.float32), axis=0) / BLOCK_SIZE_FLOAT
        tl.store(mean_base + blk_idx * m_n + offs_d * m_d, mean_val.to(val.dtype))

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
        
        sk_base, sv_base = SK_ptr + cur_batch * sk_b + cur_head * sk_h, SV_ptr + cur_batch * sv_b + cur_head * sv_h
        dk_base, dv_base = DK_ptr + cur_batch * dk_b + cur_head * dk_h, DV_ptr + cur_batch * dv_b + cur_head * dv_h
        mk_base = MK_ptr + cur_batch * m_b + cur_head * m_h
        
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
        
        val_k = tl.load(sk_base + idx_l[:, None] * sk_l + offs_d[None, :] * sk_d, mask=mask[:, None], other=0.0)
        val_v = tl.load(sv_base + idx_l[:, None] * sv_l + offs_d[None, :] * sv_d, mask=mask[:, None], other=0.0)
        
        dst_off = (blk_idx + dst_offset_blk) * dk_n + offs_flat[:, None] * dk_blk + offs_d[None, :] * dk_d
        tl.store(dk_base + dst_off, val_k)
        tl.store(dv_base + dst_off, val_v)
        
        tl.store(mk_base + (blk_idx + pool_offset_blk) * m_n + offs_d * m_d, (tl.sum(val_k.to(tl.float32), axis=0) / BLOCK_SIZE_FLOAT).to(val_k.dtype))

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

    def launch_q_prep(src, dst, mean, T, H, W, tt, th, tw, nh, nw):
        B, NH, _, D = src.shape
        num_blocks = ((T + tt - 1) // tt) * nh * nw
        BLOCK_SIZE = tt * th * tw
        _fused_q_prep_kernel[(num_blocks, B * NH)](
            src, dst, mean,
            *src.stride(), *dst.stride(), *mean.stride(),
            NH, T, H, W, tt, th, tw, nh, nw,
            BLOCK_SIZE=BLOCK_SIZE, HEAD_DIM=D, BLOCK_SIZE_FLOAT=float(BLOCK_SIZE),
            num_warps=8, num_stages=3
        )

    def launch_kv_prep(k_src, v_src, k_dst, v_dst, k_mean, dst_off, pool_off, T, H, W, tt, th, tw, nh, nw):
        B, NH, _, D = k_src.shape
        num_blocks = ((T + tt - 1) // tt) * nh * nw
        BLOCK_SIZE = tt * th * tw
        _fused_kv_prep_kernel[(num_blocks, B * NH)](
            k_src, v_src, k_dst, v_dst, k_mean,
            *k_src.stride(), *v_src.stride(),
            *k_dst.stride(), *v_dst.stride(),
            *k_mean.stride(),
            NH, T, H, W, tt, th, tw, nh, nw, dst_off, pool_off,
            BLOCK_SIZE=BLOCK_SIZE, HEAD_DIM=D, BLOCK_SIZE_FLOAT=float(BLOCK_SIZE),
            num_warps=8, num_stages=3
        )

    def launch_untile(src, dst, T, H, W, tt, th, tw, nh, nw):
        B, NH, _, D = dst.shape
        num_blocks = ((T + tt - 1) // tt) * nh * nw
        BLOCK_SIZE = tt * th * tw
        _fused_untile_kernel[(num_blocks, B * NH)](
            src, dst,
            *src.stride(), *dst.stride(),
            NH, T, H, W, tt, th, tw, nh, nw,
            BLOCK_SIZE=BLOCK_SIZE, HEAD_DIM=D, num_warps=4
        )

def generate_prefill_indices(q_in, k_in, text_len, grid_thw, params, device):
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
    t_idx, rem = idx_arr // (Nh * Nw), idx_arr % (Nh * Nw)
    h_idx, w_idx = rem // Nw, rem % Nw
    
    t_q, t_k = t_idx.unsqueeze(1), t_idx.unsqueeze(0)
    h_q, h_k = h_idx.unsqueeze(1), h_idx.unsqueeze(0)
    w_q, w_k = w_idx.unsqueeze(1), w_idx.unsqueeze(0)
    
    dist = (t_q - t_k).abs() + (h_q - h_k).abs() + (w_q - w_k).abs()
    is_neighbor = dist <= 1
    score_for_ranking = score_for_ranking + is_neighbor.to(score_for_ranking.dtype) * LARGE_BIAS

    target_k = int(N_k * params.get('topk_ratio', 0.25))
    k_budget = min(max(target_k, 7), N_k)
    
    _, vis_indices = torch.topk(score_for_ranking, k=k_budget, dim=-1)
    vis_indices, _ = torch.sort(vis_indices, dim=-1) 
    
    num_txt_blk = (text_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    vis_indices = vis_indices + num_txt_blk 
    txt_indices = torch.arange(num_txt_blk, device=device).view(1, 1, 1, -1).expand(B, H, N_q, -1)
    
    final_indices = torch.cat([txt_indices, vis_indices], dim=-1).int().contiguous()
    final_counts = torch.full((B, H, N_q), final_indices.shape[-1], dtype=torch.int32, device=device)
    
    return final_indices, final_counts

def run_sparse_attention(q, k, v, text_len, block_idx, latent_hw, sparse_params):
    # Initialize optional profiling probes.
    if ENABLE_SPARSE_PROFILING:
        events = {}
        def mark(name):
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            events[name] = evt
        mark("start")

    LATENT_H, LATENT_W = latent_hw
    vis_len_q = q.shape[2]
    
    if not HAS_TRITON or (vis_len_q % (LATENT_H * LATENT_W) != 0):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    vis_len_k = k.shape[2] - text_len 
    tt, th, tw = sparse_params.get('tile_size', (4, 4, 8))
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

    # Stage 1: allocate tiled sparse-attention buffers.
    if ENABLE_SPARSE_PROFILING: mark("alloc_start")
    
    q_tiled = torch.empty((B, H, num_vis_blk_q, BLOCK_SIZE, D), device=q.device, dtype=q.dtype)
    q_pool_vis = torch.empty((B, H, num_vis_blk_q, D), device=q.device, dtype=q.dtype)
    
    k_all = torch.empty((B, H, num_txt_blk + num_vis_blk_k, BLOCK_SIZE, D), device=q.device, dtype=q.dtype)
    v_all = torch.empty((B, H, num_txt_blk + num_vis_blk_k, BLOCK_SIZE, D), device=q.device, dtype=q.dtype)
    k_pool_vis = torch.empty((B, H, num_vis_blk_k, D), device=q.device, dtype=q.dtype) 

    pad_txt = (BLOCK_SIZE - text_len % BLOCK_SIZE) % BLOCK_SIZE
    k_txt, v_txt = k[:, :, :text_len], v[:, :, :text_len]
    if pad_txt > 0 and text_len > 0:
        k_txt = F.pad(k_txt, (0, 0, 0, pad_txt))
        v_txt = F.pad(v_txt, (0, 0, 0, pad_txt))
    if text_len > 0:
        k_all[:, :, :num_txt_blk].copy_(k_txt.view(B, H, -1, BLOCK_SIZE, D))
        v_all[:, :, :num_txt_blk].copy_(v_txt.view(B, H, -1, BLOCK_SIZE, D))

    if ENABLE_SPARSE_PROFILING: mark("alloc_end")

    # Stage 2: tile Q/K/V tensors and pool block features.
    if ENABLE_SPARSE_PROFILING: mark("prep_start")
    
    launch_q_prep(q, q_tiled, q_pool_vis, T_q, LATENT_H, LATENT_W, tt, th, tw, nh, nw)
    launch_kv_prep(k[:, :, text_len:], v[:, :, text_len:], k_all, v_all, k_pool_vis, num_txt_blk, 0,
                   T_k, LATENT_H, LATENT_W, tt, th, tw, nh, nw)
                   
    if ENABLE_SPARSE_PROFILING: mark("prep_end")

    # Stage 3: build sparse block indices.
    if ENABLE_SPARSE_PROFILING: mark("index_start")
    
    indices, counts = generate_prefill_indices(
        q_pool_vis, k_pool_vis, text_len, (nt_k, nh, nw), 
        sparse_params, q.device
    )
    
    if ENABLE_SPARSE_PROFILING: mark("index_end")

    # Stage 4: run long-sequence sparse attention.
    if ENABLE_SPARSE_PROFILING: mark("attn_start")
    
    # Long-sequence block sparse attention.
    o_flat = longcat_flash_attn(
        q_tiled.flatten(2, 3), k_all.flatten(2, 3), v_all.flatten(2, 3), 
        indices, counts, sm_scale=1.0/(D**0.5), logical_block_size=BLOCK_SIZE
    )
    
    if ENABLE_SPARSE_PROFILING: mark("attn_end")
    
    # Stage 5: restore tiled output to the original token layout.
    if ENABLE_SPARSE_PROFILING: mark("untile_start")
    
    o_out = torch.empty_like(q)
    launch_untile(o_flat.view(B, H, num_vis_blk_q, BLOCK_SIZE, D), o_out, T_q, LATENT_H, LATENT_W, tt, th, tw, nh, nw)
    
    if ENABLE_SPARSE_PROFILING: mark("untile_end")

    # Print profiling report when explicitly enabled.
    if ENABLE_SPARSE_PROFILING:
        # Synchronize once to report absolute CUDA timings.
        torch.cuda.synchronize()
        
        t_alloc = events["alloc_start"].elapsed_time(events["alloc_end"])
        t_prep  = events["prep_start"].elapsed_time(events["prep_end"])
        t_idx   = events["index_start"].elapsed_time(events["index_end"])
        t_attn  = events["attn_start"].elapsed_time(events["attn_end"])
        t_untl  = events["untile_start"].elapsed_time(events["untile_end"])
        t_total = events["start"].elapsed_time(events["untile_end"])
        
        print(f"[Block {block_idx:2d}] Sparse Workflow: "
              f"Total {t_total:>5.2f}ms | "
              f"Alloc {t_alloc:>4.2f}ms | "
              f"Tiling {t_prep:>4.2f}ms | "
              f"Index {t_idx:>4.2f}ms | "
              f"SparseAttn {t_attn:>4.2f}ms | "
              f"Untile {t_untl:>4.2f}ms")

    return o_out
