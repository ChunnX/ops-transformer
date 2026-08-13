# Python API modifications for __init__.py
#
# Add to deep_gemm/__init__.py the new TopK interfaces.
# Below shows the additions needed (not a full file rewrite).

# =============================================================================
# Add to the existing import section
# =============================================================================
from . import attention
# ... existing imports ...

# =============================================================================
# New: TopK-enabled MQA logits interfaces
# =============================================================================

# Non-paged (prefill)
def fp8_mqa_logits_with_topk(
    q, kv, weights,
    cu_seq_len_k_start, cu_seq_len_k_end,
    sparse_count,
    max_seqlen_k=0
):
    """
    MQA logits + TopK selection (non-paged).

    Computes: relu(Q @ K^T * scale) * weight, sum over heads, then TopK.

    Args:
        q: torch.Tensor [seq_len, num_heads, head_dim], FP8_E4M3
        kv: tuple of (kv_data [seq_len_kv, head_dim], kv_sf [seq_len_kv])
        weights: torch.Tensor [seq_len, num_heads], float32 or bf16
        cu_seq_len_k_start: torch.Tensor [seq_len], int32
        cu_seq_len_k_end: torch.Tensor [seq_len], int32
        sparse_count: int, number of top-K indices to select (1~2048)
        max_seqlen_k: int, max KV sequence length (0 = use seq_len_kv)

    Returns:
        sparse_indices: torch.Tensor [seq_len, sparse_count], int32
            Global KV token indices, sorted by logit descending.
            -1 for padding.
    """
    return attention.fp8_fp4_mqa_logits_with_topk(
        (q, None), kv, weights,
        cu_seq_len_k_start, cu_seq_len_k_end,
        sparse_count, max_seqlen_k
    )

# FP8/FP4 with MX scale factor (non-paged)
def fp8_fp4_mqa_logits_with_topk(
    q, kv, weights,
    cu_seq_len_k_start, cu_seq_len_k_end,
    sparse_count,
    max_seqlen_k=0
):
    """
    FP8/FP4 MQA logits + TopK with MX scale factors (non-paged).

    Args:
        q: tuple of (q_fp [seq_len, num_heads, head_dim], q_sf [seq_len, num_heads])
           q_fp: FP8_E4M3 or PackedFP4
           q_sf: int32 (packed UE8M0), or None for non-MX
        kv: tuple of (kv_fp [seq_len_kv, head_dim], kv_sf [seq_len_kv])
        weights: [seq_len, num_heads], float32 or bf16
        sparse_count: int (1~2048)
        max_seqlen_k: int

    Returns:
        sparse_indices: [seq_len, sparse_count], int32
    """
    return attention.fp8_fp4_mqa_logits_with_topk(
        q, kv, weights,
        cu_seq_len_k_start, cu_seq_len_k_end,
        sparse_count, max_seqlen_k
    )

# Paged (decode)
def fp8_paged_mqa_logits_with_topk(
    q, fused_kv_cache, weights,
    context_lens, block_table, schedule_meta,
    max_context_len, sparse_count
):
    """
    Paged MQA logits + TopK (decode).

    Args:
        q: [batch, next_n, num_heads, head_dim], FP8_E4M3
        fused_kv_cache: [num_blocks, block_kv, 1, head_dim+sf_bytes], uint8
        weights: [batch*next_n, num_heads], float32 or bf16
        context_lens: [batch, next_n], int32
        block_table: [batch, max_blocks], int32
        schedule_meta: [num_sms+1, 2], int32 (from get_paged_mqa_logits_metadata)
        max_context_len: int
        sparse_count: int (1~2048)

    Returns:
        sparse_indices: [batch*next_n, sparse_count], int32
    """
    return attention.fp8_fp4_paged_mqa_logits_with_topk(
        (q, None), fused_kv_cache, weights,
        context_lens, block_table, schedule_meta,
        max_context_len, sparse_count
    )

# FP8/FP4 paged with MX scale factor
def fp8_fp4_paged_mqa_logits_with_topk(
    q, fused_kv_cache, weights,
    context_lens, block_table, schedule_meta,
    max_context_len, sparse_count
):
    """
    Paged FP8/FP4 MQA logits + TopK with MX scale factors.

    Args:
        q: tuple of (q_fp, q_sf)
        fused_kv_cache, weights, context_lens, block_table, schedule_meta: same as above
        max_context_len: int
        sparse_count: int (1~2048)

    Returns:
        sparse_indices: [batch*next_n, sparse_count], int32
    """
    return attention.fp8_fp4_paged_mqa_logits_with_topk(
        q, fused_kv_cache, weights,
        context_lens, block_table, schedule_meta,
        max_context_len, sparse_count
    )


# =============================================================================
# Usage example (B200 / SM100)
# =============================================================================
"""
import torch
import deep_gemm

# --- Non-paged (prefill) example ---
seq_len = 4096
seq_len_kv = 4096
num_heads = 64
head_dim = 128
sparse_count = 2048

q = torch.randn(seq_len, num_heads, head_dim, device='cuda').to(torch.float8_e4m3fn)
kv_data = torch.randn(seq_len_kv, head_dim, device='cuda').to(torch.float8_e4m3fn)
kv_sf = torch.ones(seq_len_kv, device='cuda', dtype=torch.float32)
weights = torch.randn(seq_len, num_heads, device='cuda', dtype=torch.float32)

# Causal mask: token i attends to tokens [0, i]
cu_start = torch.zeros(seq_len, device='cuda', dtype=torch.int32)
cu_end = torch.arange(1, seq_len + 1, device='cuda', dtype=torch.int32)

sparse_indices = deep_gemm.fp8_mqa_logits_with_topk(
    q, (kv_data, kv_sf), weights,
    cu_start, cu_end,
    sparse_count=sparse_count
)
# sparse_indices: [seq_len, 2048] int32

# --- Paged (decode) example ---
batch_size = 32
next_n = 1
block_kv = 64
num_kv_blocks = 1024
max_context_len = 4096

q_decode = torch.randn(batch_size, next_n, num_heads, head_dim, device='cuda').to(torch.float8_e4m3fn)
fused_kv = torch.zeros(num_kv_blocks, block_kv, 1, head_dim + 4, device='cuda', dtype=torch.uint8)
weights_decode = torch.randn(batch_size * next_n, num_heads, device='cuda', dtype=torch.float32)
context_lens = torch.full((batch_size, next_n), max_context_len, device='cuda', dtype=torch.int32)
block_table = torch.arange(num_kv_blocks, device='cuda', dtype=torch.int32).reshape(batch_size, -1)

schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
    context_lens, block_kv, num_sms=148)

sparse_indices_decode = deep_gemm.fp8_paged_mqa_logits_with_topk(
    q_decode, fused_kv, weights_decode,
    context_lens, block_table, schedule_meta,
    max_context_len, sparse_count
)
# sparse_indices_decode: [batch_size * next_n, 2048] int32
"""
