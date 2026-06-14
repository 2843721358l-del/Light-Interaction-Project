# Adapted from LongCat-Video.
# Original copyright (c) 2025 Meituan.
# Licensed under the MIT License.
#
# Modifications for Light Interaction:
# - adapted the sparse-attention kernel to Matrix-Game-3.0
# - integrated it with autoregressive inference and camera-aware sparse attention
#
# Modifications copyright (c) 2026 Jiacheng Lu and contributors.
"""Triton block-sparse attention kernel used by Light Interaction."""

import torch
import triton
import triton.language as tl
import os

# Triton autotune candidates. Keep one tested config enabled by default to
# avoid first-run autotuning noise in release runs.
configs_fwd_bsa_align = [
    #triton.Config({}, num_stages=2, num_warps=4),
    #triton.Config({}, num_stages=3, num_warps=4),
    triton.Config({}, num_stages=2, num_warps=8),
    #triton.Config({}, num_stages=3, num_warps=8),
]

fwd_bsa_reevaluate_keys = ['Q_LEN', 'K_LEN'] if os.environ.get('TRITON_REEVALUATE_KEY', '0') == '1' else []

@triton.autotune(list(configs_fwd_bsa_align), key=fwd_bsa_reevaluate_keys)
@triton.jit
def _attn_fwd_bsa_align(
    Q, K, V, sm_scale, M, Out, 
    block_indices, block_indices_lens, 
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_bz, stride_bh, stride_bm, stride_bs,
    stride_lz, stride_lh, stride_lm,
    H, Q_LEN, K_LEN,
    HEAD_DIM: tl.constexpr, 
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
):
    # Grid ID
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Base Offsets
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    b_offset = off_z.to(tl.int64) * stride_bz + off_h.to(tl.int64) * stride_bh
    l_offset = off_z.to(tl.int64) * stride_lz + off_h.to(tl.int64) * stride_lh

    # Block Pointers (Base)
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset, shape=(Q_LEN, HEAD_DIM), strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset, shape=(K_LEN, HEAD_DIM), strides=(stride_vn, stride_vk),
        offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0)
    )
    KT_block_ptr = tl.make_block_ptr(
        base=K + k_offset, shape=(HEAD_DIM, K_LEN), strides=(stride_kk, stride_kn),
        offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1)
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset, shape=(Q_LEN, HEAD_DIM), strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0)
    )

    # Initialize the index pointer once to avoid repeated stride arithmetic.
    index_ptr = block_indices + b_offset + start_m * stride_bm
    S = tl.load(block_indices_lens + l_offset + start_m * stride_lm)

    # Accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504 

    # Load Q 
    q = tl.load(Q_block_ptr)

    # Software-pipeline the sparse block-id loads.
    next_block_id = tl.load(index_ptr).to(tl.int32)
    index_ptr += stride_bs

    for i in range(S):
        # Use the block id prefetched by the previous iteration.
        block_id = next_block_id
        
        # Prefetch the next block id.
        is_not_last = i < S - 1
        next_block_id = tl.load(index_ptr, mask=is_not_last, other=0).to(tl.int32)
        index_ptr += stride_bs 
        
        # Compute aligned K/V offsets.
        lo = block_id * BLOCK_N
        KT_curr = tl.advance(KT_block_ptr, (0, lo))
        V_curr = tl.advance(V_block_ptr, (lo, 0))

        # Load K/V blocks and run the FlashAttention update.
        kT = tl.load(KT_curr)
        qkT = tl.dot(q, kT)

        # Softmax & FlashAttention Math
        m_ij = tl.maximum(m_i, tl.max(qkT, 1) * qk_scale)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        qkT = qkT * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qkT)
        
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        
        v = tl.load(V_curr)
        acc = tl.dot(p.to(v.dtype), v, acc)

    # Epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_ptrs = M + off_hz * Q_LEN + offs_m
    tl.store(m_ptrs, m_i)


def longcat_flash_attn(q, k, v, block_indices, block_counts, sm_scale=None, logical_block_size=128):

    half_dtypes = (torch.float16, torch.bfloat16)
    
    # Convert only non-half tensors to avoid extra casts in normal inference.
    if q.dtype not in half_dtypes:
        q = q.to(torch.bfloat16)
    if k.dtype not in half_dtypes:
        k = k.to(torch.bfloat16)
    if v.dtype not in half_dtypes:
        v = v.to(torch.bfloat16)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    block_indices = block_indices.contiguous()
    block_counts = block_counts.contiguous()

    B, H, Q_len, HEAD_DIM = q.shape
    K_len = k.shape[2]

    if sm_scale is None:
        sm_scale = 1.0 / (HEAD_DIM ** 0.5)

    BLOCK_N = logical_block_size 
    assert K_len % BLOCK_N == 0, f"K Length ({K_len}) must be divisible by logical_block_size ({BLOCK_N})"

    o = torch.empty_like(q)
    m_buffer = torch.empty((B, H, Q_len), device=q.device, dtype=torch.float32)

    BLOCK_M = 128
    grid = lambda META: (
        triton.cdiv(Q_len, BLOCK_M),
        B * H
    )

    _attn_fwd_bsa_align[grid](
        Q=q, K=k, V=v, sm_scale=sm_scale, M=m_buffer, Out=o,
        block_indices=block_indices,
        block_indices_lens=block_counts,
        
        stride_qz=q.stride(0), stride_qh=q.stride(1), stride_qm=q.stride(2), stride_qk=q.stride(3),
        stride_kz=k.stride(0), stride_kh=k.stride(1), stride_kn=k.stride(2), stride_kk=k.stride(3),
        stride_vz=v.stride(0), stride_vh=v.stride(1), stride_vn=v.stride(2), stride_vk=v.stride(3),
        stride_oz=o.stride(0), stride_oh=o.stride(1), stride_om=o.stride(2), stride_ok=o.stride(3),
        
        stride_bz=block_indices.stride(0), stride_bh=block_indices.stride(1), 
        stride_bm=block_indices.stride(2), stride_bs=block_indices.stride(3),
        
        stride_lz=block_counts.stride(0), stride_lh=block_counts.stride(1), stride_lm=block_counts.stride(2),
        
        H=H, Q_LEN=Q_len, K_LEN=K_len, HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    return o
