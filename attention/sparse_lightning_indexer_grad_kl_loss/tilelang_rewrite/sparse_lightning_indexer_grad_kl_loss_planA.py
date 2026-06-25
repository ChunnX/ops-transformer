"""Plan A: K-parallel kernel + Python postprocessing.

Architecture:
  Kernel (block_num = B * s1_tile_num * n_num, K-parallel):
    C1(MM1+MM2) -> V1(P_reduced chunk, ReduceSumY chunk) per K-block
  Python (torch ops):
    SoftmaxI, diff, dWeight, reluGrad, Loss, MM6(dQueryIndex), Cast, dKeyIndex
"""

import math
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import tilelang
import tilelang.language as T

tilelang.disable_cache()

# ==============================================================================
# Constants
# ==============================================================================
BLOCK_M = 16
BLOCK_N = 128
BLOCK_K = 128
BLOCK_N1 = 32
BLOCK_NIDX1 = 16

DQUERY = 512
DQUERY_INDEX = 128
DROPE = 64

_kernel_cache = {}
_workspace_cache = {}


def _clear_caches():
    global _kernel_cache, _workspace_cache
    _kernel_cache.clear()
    _workspace_cache.clear()


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# ==============================================================================
# Tiling Data
# ==============================================================================
@dataclass
class SparseLightningIndexerGradKLLossTilingData:
    B: int; S1: int; S2: int; N1: int; N2: int; G: int
    Nidx1: int; Nidx2: int; DQuery: int; DQueryIndex: int; K: int
    sparse_mode: int = 3

    @property
    def s1_tile_num(self) -> int:
        return _ceil_div(self.S1, BLOCK_M)

    @property
    def n_num(self) -> int:
        return self.K // BLOCK_N

    @property
    def block_num(self) -> int:
        """K-parallel: B * s1_tile_num * n_num cores."""
        return self.B * self.s1_tile_num * self.n_num


# ==============================================================================
# Parameter Validation
# ==============================================================================
def check_param_valid(
    query: torch.Tensor, key: torch.Tensor,
    query_index: torch.Tensor, key_index: torch.Tensor,
    weights: torch.Tensor, sparse_indices: torch.Tensor,
    softmax_max: torch.Tensor, softmax_sum: torch.Tensor,
    *,
    sparse_mode: int = 3,
    layout_query: str = "BSND",
) -> SparseLightningIndexerGradKLLossTilingData:
    if layout_query == "TND":
        assert query.dim() == 3, f"query dim must be 3 for TND, got {query.dim()}"
        S1, N1, DQuery = query.shape
        _, N2, _ = key.shape
        _, Nidx1, DQueryIndex = query_index.shape
        _, Nidx2, _ = key_index.shape
        B = 1
    else:
        assert query.dim() == 4, f"query dim must be 4 for BSND, got {query.dim()}"
        B, S1, N1, DQuery = query.shape
        _, S2, N2, _ = key.shape
        _, _, Nidx1, DQueryIndex = query_index.shape
        _, _, Nidx2, _ = key_index.shape

    assert query.dtype in (torch.float16, torch.bfloat16)
    assert N2 == 1, f"N2 must be 1 (MQA), got {N2}"
    G = N1 // N2
    assert DQuery == DQUERY
    assert DQueryIndex == DQUERY_INDEX
    assert N1 in (32, 64, 128)
    assert Nidx1 in (8, 16, 32, 64)
    K = sparse_indices.shape[-1]
    assert K % 1024 == 0 and 1024 <= K <= 8192
    assert sparse_mode == 3

    return SparseLightningIndexerGradKLLossTilingData(
        B=B, S1=S1, S2=S2, N1=N1, N2=N2, G=G,
        Nidx1=Nidx1, Nidx2=Nidx2,
        DQuery=DQuery, DQueryIndex=DQueryIndex,
        K=K, sparse_mode=sparse_mode,
    )


# ==============================================================================
# Plan A Kernel: C1(MM1+MM2) + V1(P_reduced, ReduceSumY) per K-block
# ==============================================================================
@tilelang.jit(
    out_idx=[-3, -2, -1],
    pass_configs=pass_configs,
)
def planA_kernel(
    B, S1, S2, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
    d_rope, input_dtype, calc_dtype, scale_value,
):
    s1_block = BLOCK_M
    n_block = BLOCK_N
    k_block = BLOCK_K
    s1_tile_num = _ceil_div(S1, s1_block)
    n_num = K // n_block
    d_num_query = DQuery // k_block
    d_num_rope = _ceil_div(d_rope, k_block) if d_rope > 0 else 0
    d_num_total = d_num_query + d_num_rope
    d_num_query_index = DQueryIndex // k_block
    n1_num = N1 // BLOCK_N1
    nidx1_num = Nidx1 // BLOCK_NIDX1

    @T.prim_func
    def main(
        Query: T.Tensor((B, S1, N1, DQuery), input_dtype),
        QueryRope: T.Tensor((B, S1, N1, d_rope), input_dtype),
        GatherKey: T.Tensor((B, S1, K, DQuery), input_dtype),
        GatherKeyRope: T.Tensor((B, S1, K, d_rope), input_dtype),
        QueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), input_dtype),
        GatherKeyIndex: T.Tensor((B, S1, K, DQueryIndex), input_dtype),
        Weights: T.Tensor((B, S1, Nidx1), input_dtype),
        SoftmaxMax: T.Tensor((B, N2, S1, G), calc_dtype),
        SoftmaxSum: T.Tensor((B, N2, S1, G), calc_dtype),
        bmm1Res: T.Tensor((B, S1, N1, K), calc_dtype),
        bmm2Res: T.Tensor((B, S1, Nidx1, K), calc_dtype),
        PReduced: T.Tensor((B, S1, K), calc_dtype),
        ReduceSumY: T.Tensor((B, S1, K), calc_dtype),
    ):
        # block_num = B * s1_tile_num * n_num (K-parallel)
        with T.Kernel(B * s1_tile_num * n_num, is_npu=True) as (cid, vid):
            bz = cid // (s1_tile_num * n_num)
            r = cid % (s1_tile_num * n_num)
            m = r // n_num
            n_idx = r % n_num

            s1_start = m * s1_block
            s1_end = T.min(s1_start + s1_block, S1)
            k_start = n_idx * n_block
            k_end = k_start + n_block

            # ============================================================
            # C Scope 1: MM1 + MM2 for this K-block
            # ============================================================
            with T.Scope("C"):
                Q_L1 = T.alloc_L1((N1, k_block), input_dtype)
                GK_L1 = T.alloc_L1((n_block, k_block), input_dtype)
                C_L0C = T.alloc_L0C((N1, n_block), calc_dtype)

                QI_L1 = T.alloc_L1((Nidx1, k_block), input_dtype)
                GKI_L1 = T.alloc_L1((n_block, k_block), input_dtype)
                C2_L0C = T.alloc_L0C((Nidx1, n_block), calc_dtype)

                for s1_row in T.serial(s1_start, s1_end):
                    # --- MM1: Q||QRope @ GK||GKRope^T ---
                    for d_idx in T.serial(d_num_total):
                        if d_idx < d_num_query:
                            T.copy(
                                Query[bz, s1_row, 0:N1, d_idx * k_block:(d_idx + 1) * k_block],
                                Q_L1,
                            )
                        else:
                            rope_d = d_idx - d_num_query
                            T.copy(
                                QueryRope[bz, s1_row, 0:N1, rope_d * k_block:(rope_d + 1) * k_block],
                                Q_L1,
                            )
                        if d_idx < d_num_query:
                            T.copy(
                                GatherKey[bz, s1_row, k_start:k_end, d_idx * k_block:(d_idx + 1) * k_block],
                                GK_L1,
                            )
                        else:
                            rope_d = d_idx - d_num_query
                            T.copy(
                                GatherKeyRope[bz, s1_row, k_start:k_end, rope_d * k_block:(rope_d + 1) * k_block],
                                GK_L1,
                            )
                        T.gemm_v0(Q_L1, GK_L1, C_L0C, transpose_B=True, init=(d_idx == 0))
                    T.copy(C_L0C, bmm1Res[bz, s1_row, 0:N1, k_start:k_end])

                    # --- MM2: QI @ GKI^T + ReLU ---
                    for k_idx in T.serial(d_num_query_index):
                        T.copy(
                            QueryIndex[bz, s1_row, 0:Nidx1, k_idx * k_block:(k_idx + 1) * k_block],
                            QI_L1,
                        )
                        T.copy(
                            GatherKeyIndex[bz, s1_row, k_start:k_end, k_idx * k_block:(k_idx + 1) * k_block],
                            GKI_L1,
                        )
                        T.gemm_v0(QI_L1, GKI_L1, C2_L0C, transpose_B=True, init=(k_idx == 0))
                    T.copy(
                        C2_L0C,
                        bmm2Res[bz, s1_row, 0:Nidx1, k_start:k_end],
                        enable_relu=True,
                    )

                T.set_cross_flag("FIX", 1)

            # ============================================================
            # V Scope 1: P_reduced + ReduceSumY for this K-block
            # ============================================================
            with T.Scope("V"):
                T.wait_cross_flag(1)

                bmm1_ub = T.alloc_shared((BLOCK_N1, n_block), calc_dtype)
                bmm2_ub = T.alloc_shared((BLOCK_NIDX1, n_block), calc_dtype)
                weight_ub = T.alloc_shared(Nidx1, calc_dtype)
                weight_fp16_ub = T.alloc_shared(Nidx1, input_dtype)

                p_chunk_ub = T.alloc_shared(n_block, calc_dtype)
                rsy_chunk_ub = T.alloc_shared(n_block, calc_dtype)

                sm_max_ub = T.alloc_shared(G, calc_dtype)
                sm_sum_ub = T.alloc_shared(G, calc_dtype)
                eps_ub = T.alloc_shared(n_block, calc_dtype)

                T.tile.fill(eps_ub, 1e-8)

                for s1_id in T.serial(s1_start, s1_end):
                    # Load softmaxMax and softmaxSum
                    T.copy(SoftmaxMax[bz, 0, s1_id, 0:G], sm_max_ub)
                    T.copy(SoftmaxSum[bz, 0, s1_id, 0:G], sm_sum_ub)
                    T.tile.abs(sm_sum_ub, sm_sum_ub)
                    T.tile.max(sm_sum_ub, sm_sum_ub, eps_ub)
                    T.barrier_all()

                    # Load weights
                    T.copy(Weights[bz, s1_id, 0:Nidx1], weight_fp16_ub)
                    T.tile.cast(weight_ub, weight_fp16_ub, "CAST_NONE", Nidx1)
                    T.barrier_all()

                    # ---- Pass 1: P_reduced[k_start:k_end] ----
                    T.tile.fill(p_chunk_ub, 0.0)
                    for n1_c in T.serial(n1_num):
                        n1_s = n1_c * BLOCK_N1
                        T.copy(
                            bmm1Res[bz, s1_id, n1_s:n1_s + BLOCK_N1, k_start:k_end],
                            bmm1_ub,
                        )
                        T.tile.mul(bmm1_ub, bmm1_ub, scale_value)
                        for g_i in T.serial(BLOCK_N1):
                            T.tile.sub(bmm1_ub[g_i, :], bmm1_ub[g_i, :], sm_max_ub[n1_s + g_i])
                        T.tile.exp(bmm1_ub, bmm1_ub)
                        T.barrier_all()
                        for g_i in T.serial(BLOCK_N1):
                            T.tile.div(bmm1_ub[g_i, :], bmm1_ub[g_i, :], sm_sum_ub[n1_s + g_i])
                        T.barrier_all()
                        for g_i in T.serial(BLOCK_N1):
                            T.tile.add(p_chunk_ub, p_chunk_ub, bmm1_ub[g_i, :])
                        T.barrier_all()
                    T.tile.mul(p_chunk_ub, p_chunk_ub, 1.0 / N1)
                    T.copy(p_chunk_ub, PReduced[bz, s1_id, k_start:k_end])

                    # ---- Pass 2: ReduceSumY[k_start:k_end] ----
                    T.tile.fill(rsy_chunk_ub, 0.0)
                    for nidx_c in T.serial(nidx1_num):
                        nidx_s = nidx_c * BLOCK_NIDX1
                        T.copy(
                            bmm2Res[bz, s1_id, nidx_s:nidx_s + BLOCK_NIDX1, k_start:k_end],
                            bmm2_ub,
                        )
                        for g_i in T.serial(BLOCK_NIDX1):
                            T.tile.mul(bmm2_ub[g_i, :], bmm2_ub[g_i, :], weight_ub[nidx_s + g_i])
                        T.barrier_all()
                        for g_i in T.serial(BLOCK_NIDX1):
                            T.tile.add(rsy_chunk_ub, rsy_chunk_ub, bmm2_ub[g_i, :])
                        T.barrier_all()
                    T.copy(rsy_chunk_ub, ReduceSumY[bz, s1_id, k_start:k_end])

    return main


# ==============================================================================
# Python Preprocessing: Gather Key/KeyIndex/KeyRope
# ==============================================================================
def _preprocess_gather(
    key: torch.Tensor,
    key_index: torch.Tensor,
    key_rope: Optional[torch.Tensor],
    sparse_indices: torch.Tensor,
    B: int, S1: int, K: int,
    DQuery: int, DQueryIndex: int,
    d_rope_orig: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = key.device
    sparse_idx_3d = sparse_indices[:, :, 0, :]
    valid_mask = (sparse_idx_3d >= 0).unsqueeze(-1).to(dtype)
    sparse_idx_safe = sparse_idx_3d.clamp(min=0)

    key_2d = key[:, :, 0, :]
    idx_k = sparse_idx_safe.reshape(B, S1 * K).unsqueeze(-1).expand(B, S1 * K, DQuery)
    gather_key = torch.gather(key_2d, 1, idx_k).reshape(B, S1, K, DQuery).contiguous()
    gather_key.mul_(valid_mask)

    key_index_2d = key_index[:, :, 0, :]
    idx_ki = sparse_idx_safe.reshape(B, S1 * K).unsqueeze(-1).expand(B, S1 * K, DQueryIndex)
    gather_key_index = torch.gather(key_index_2d, 1, idx_ki).reshape(B, S1, K, DQueryIndex).contiguous()
    gather_key_index.mul_(valid_mask)

    if key_rope is not None and d_rope_orig > 0:
        key_rope_2d = key_rope[:, :, 0, :]
        idx_rope = sparse_idx_safe.reshape(B, S1 * K).unsqueeze(-1).expand(B, S1 * K, d_rope_orig)
        gather_key_rope = torch.gather(key_rope_2d, 1, idx_rope).reshape(B, S1, K, d_rope_orig).contiguous()
        gather_key_rope.mul_(valid_mask)
    else:
        gather_key_rope = torch.zeros(B, S1, K, d_rope_orig, dtype=dtype, device=device)

    return gather_key, gather_key_index, gather_key_rope


# ==============================================================================
# Python Postprocessing: dKeyIndex via scatter_add
# ==============================================================================
def _compute_d_key_index(
    relu_grad: torch.Tensor,
    query_index: torch.Tensor,
    sparse_indices: torch.Tensor,
    B: int, S1: int, S2: int,
    Nidx1: int, Nidx2: int, K: int,
    DQueryIndex: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    relu_grad_fp32 = relu_grad.float()
    query_index_fp32 = query_index.float()

    relu_grad_t = relu_grad_fp32.permute(0, 1, 3, 2)  # (B, S1, K, Nidx1)
    scatter_add = torch.bmm(
        relu_grad_t.reshape(B * S1, K, Nidx1),
        query_index_fp32.reshape(B * S1, Nidx1, DQueryIndex),
    ).reshape(B, S1, K, DQueryIndex)

    d_key_index = torch.zeros(B, S2, Nidx2, DQueryIndex, dtype=torch.float32, device=device)
    sparse_idx = sparse_indices[:, :, 0, :].clamp(min=0)
    valid_mask = (sparse_indices[:, :, 0, :] >= 0).unsqueeze(-1).to(torch.float32)

    scatter_add_masked = scatter_add * valid_mask
    idx_flat = sparse_idx.reshape(B, S1 * K).unsqueeze(-1).expand(B, S1 * K, DQueryIndex)
    vals_flat = scatter_add_masked.reshape(B, S1 * K, DQueryIndex)

    d_key_index_2d = d_key_index[:, :, 0, :]
    d_key_index_2d.scatter_add_(1, idx_flat, vals_flat)

    return d_key_index.to(dtype)


# ==============================================================================
# Python: V1 rest + C2 + V2
# ==============================================================================
def _python_postprocess(
    bmm2_res: torch.Tensor,        # (B, S1, Nidx1, K) FP32, ReLU'd
    p_reduced: torch.Tensor,       # (B, S1, K) FP32
    reduce_sum_y: torch.Tensor,    # (B, S1, K) FP32
    weights: torch.Tensor,         # (B, S1, Nidx1) FP16/BF16
    gather_key_index: torch.Tensor,# (B, S1, K, DQueryIndex) FP16/BF16
    input_dtype: torch.dtype,
    weights_orig_dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Python implementation of V1 Pass3-5 + C2(MM6) + V2(Cast)."""
    B, S1, Nidx1, K = bmm2_res.shape
    DQueryIndex = gather_key_index.shape[-1]
    n_num = K // BLOCK_N

    # --- Pass 3: Global SoftmaxI ---
    softmax_i = torch.softmax(reduce_sum_y.float(), dim=-1)  # (B, S1, K)

    # --- Pass 4: diff, dWeight, reluGrad ---
    diff = softmax_i - p_reduced.float()  # (B, S1, K)

    # dWeight[g] = sum_k(diff[k] * bmm2[g,k])
    d_weight = (diff.unsqueeze(2) * bmm2_res.float()).sum(dim=-1)  # (B, S1, Nidx1)

    # reluGrad[g,k] = weight[g] * diff[k] * bool(bmm2[g,k] > 0)
    weights_fp32 = weights.float()
    relu_grad = weights_fp32.unsqueeze(-1) * diff.unsqueeze(2) * (bmm2_res.float() > 0).float()
    # reluGrad: (B, S1, Nidx1, K)

    # --- Pass 5: KL Loss ---
    eps = 1e-8
    p_safe = p_reduced.float().clamp(min=eps)
    si_safe = softmax_i.clamp(min=eps)
    loss_per_k = p_safe * (torch.log(p_safe) - torch.log(si_safe))
    kl_loss = loss_per_k.sum(dim=-1).sum()  # scalar

    # --- C2 + V2: MM6 (dQueryIndex = sum_K reluGrad @ GKI) ---
    rg_flat = relu_grad.to(input_dtype).reshape(B * S1, Nidx1, K)
    gki_flat = gather_key_index.reshape(B * S1, K, DQueryIndex)

    d_query_index_ws = torch.zeros(B * S1, Nidx1, DQueryIndex, dtype=torch.float32, device=device)
    for n_idx in range(n_num):
        k_s = n_idx * BLOCK_N
        k_e = k_s + BLOCK_N
        d_query_index_ws += torch.bmm(
            rg_flat[:, :, k_s:k_e].float(),
            gki_flat[:, k_s:k_e, :].float(),
        )

    d_query_index = d_query_index_ws.reshape(B, S1, Nidx1, DQueryIndex).to(input_dtype)

    d_weight_out = d_weight.to(weights_orig_dtype if weights_orig_dtype == torch.float32 else input_dtype)

    return d_query_index, d_weight_out, relu_grad, kl_loss


# ==============================================================================
# Main Entry: BSND
# ==============================================================================
def sparse_lightning_indexer_grad_kl_loss(
    query: torch.Tensor,
    key: torch.Tensor,
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    weights: torch.Tensor,
    sparse_indices: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    scale_value: float,
    *,
    query_rope: Optional[torch.Tensor] = None,
    key_rope: Optional[torch.Tensor] = None,
    actual_seq_lengths_query: Optional[torch.Tensor] = None,
    actual_seq_lengths_key: Optional[torch.Tensor] = None,
    layout_query: str = "BSND",
    sparse_mode: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tiling = check_param_valid(
        query, key, query_index, key_index, weights, sparse_indices,
        softmax_max, softmax_sum,
        sparse_mode=sparse_mode, layout_query=layout_query,
    )

    if layout_query == "TND":
        return _sparse_lightning_indexer_grad_kl_loss_tnd(
            query, key, query_index, key_index, weights, sparse_indices,
            softmax_max, softmax_sum, scale_value,
            query_rope=query_rope, key_rope=key_rope,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            sparse_mode=sparse_mode,
        )

    B, S1, S2 = tiling.B, tiling.S1, tiling.S2
    N1, N2, G = tiling.N1, tiling.N2, tiling.G
    Nidx1, Nidx2 = tiling.Nidx1, tiling.Nidx2
    DQuery, DQueryIndex, K = tiling.DQuery, tiling.DQueryIndex, tiling.K
    n_num = tiling.n_num
    input_dtype = query.dtype
    weights_orig_dtype = weights.dtype
    device = query.device

    input_dtype_str = "float16" if input_dtype == torch.float16 else "bfloat16"
    calc_dtype_str = "float"

    # Pad S2 to VECTOR_BASEN(=256) alignment
    S2_pad = _ceil_div(S2, 256) * 256
    if S2_pad > S2:
        key = F.pad(key, (0, 0, 0, 0, 0, S2_pad - S2, 0, 0))
        key_index = F.pad(key_index, (0, 0, 0, 0, 0, S2_pad - S2, 0, 0))
        if key_rope is not None:
            key_rope = F.pad(key_rope, (0, 0, 0, 0, 0, S2_pad - S2, 0, 0))

    if weights.dtype == torch.float32:
        weights = weights.to(input_dtype)

    # ---- Preprocessing: Gather ----
    d_rope_orig = query_rope.shape[-1] if query_rope is not None else 0
    gather_key, gather_key_index, gather_key_rope = _preprocess_gather(
        key, key_index, key_rope, sparse_indices,
        B, S1, K, DQuery, DQueryIndex, d_rope_orig, input_dtype,
    )

    # Pad RoPE dim to BLOCK_K alignment
    if d_rope_orig > 0:
        d_rope = _ceil_div(d_rope_orig, BLOCK_K) * BLOCK_K
        if d_rope > d_rope_orig:
            pad_len = d_rope - d_rope_orig
            query_rope = F.pad(query_rope, (0, pad_len))
            gather_key_rope = F.pad(gather_key_rope, (0, pad_len))
    else:
        d_rope = BLOCK_K
    if query_rope is None:
        query_rope = torch.zeros(B, S1, N1, d_rope, dtype=input_dtype, device=device)
    if d_rope_orig == 0:
        gather_key_rope = torch.zeros(B, S1, K, d_rope, dtype=input_dtype, device=device)

    # ---- Kernel compilation cache ----
    cache_key = (B, S1, S2_pad, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
                 d_rope, input_dtype_str, calc_dtype_str, scale_value)
    if cache_key in _kernel_cache:
        func = _kernel_cache[cache_key]
    else:
        func = planA_kernel(
            B, S1, S2_pad, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
            d_rope, input_dtype_str, calc_dtype_str, scale_value,
        )
        _kernel_cache[cache_key] = func

    # ---- Workspace allocation and caching ----
    ws_key = (B, S1, N1, Nidx1, K, DQueryIndex, input_dtype, device)
    if ws_key in _workspace_cache:
        ws = _workspace_cache[ws_key]
    else:
        bmm1_res_ws = torch.zeros(B, S1, N1, K, dtype=torch.float32, device=device)
        bmm2_res_ws = torch.zeros(B, S1, Nidx1, K, dtype=torch.float32, device=device)
        p_reduced_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=device)
        reduce_sum_y_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=device)
        ws = (bmm1_res_ws, bmm2_res_ws, p_reduced_ws, reduce_sum_y_ws)
        _workspace_cache[ws_key] = ws

    bmm1_res_ws, bmm2_res_ws, p_reduced_ws, reduce_sum_y_ws = ws

    # ---- Launch kernel ----
    bmm2_res_ws, p_reduced_ws, reduce_sum_y_ws = func(
        query, query_rope, gather_key, gather_key_rope,
        query_index, gather_key_index,
        weights, softmax_max, softmax_sum,
        bmm1_res_ws, bmm2_res_ws,
        p_reduced_ws, reduce_sum_y_ws,
    )

    # ---- Python: V1 rest + C2 + V2 ----
    d_query_index, d_weight, relu_grad, kl_loss = _python_postprocess(
        bmm2_res_ws, p_reduced_ws, reduce_sum_y_ws,
        weights, gather_key_index,
        input_dtype, weights_orig_dtype, device,
    )

    # ---- Postprocessing: dKeyIndex via scatter_add ----
    d_key_index = _compute_d_key_index(
        relu_grad, query_index, sparse_indices,
        B, S1, S2, Nidx1, Nidx2, K, DQueryIndex,
        device, input_dtype,
    )

    if S2_pad > S2:
        d_key_index = d_key_index[:, :S2, :, :]

    return d_query_index, d_key_index, d_weight, kl_loss.npu() if hasattr(kl_loss, 'npu') else kl_loss.cuda()


# ==============================================================================
# TND layout entry
# ==============================================================================
def _sparse_lightning_indexer_grad_kl_loss_tnd(
    query, key, query_index, key_index, weights, sparse_indices,
    softmax_max, softmax_sum, scale_value, *,
    query_rope=None, key_rope=None,
    actual_seq_lengths_query=None, actual_seq_lengths_key=None,
    sparse_mode=3,
):
    def _unsq(t):
        return t.unsqueeze(0) if t is not None else None

    d_qi, d_ki, dw, loss = sparse_lightning_indexer_grad_kl_loss(
        _unsq(query), _unsq(key), _unsq(query_index), _unsq(key_index),
        _unsq(weights), _unsq(sparse_indices),
        _unsq(softmax_max), _unsq(softmax_sum),
        scale_value,
        query_rope=_unsq(query_rope), key_rope=_unsq(key_rope),
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        layout_query="BSND", sparse_mode=sparse_mode,
    )
    return d_qi.squeeze(0), d_ki.squeeze(0), dw.squeeze(0), loss
