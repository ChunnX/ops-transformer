import torch
import torch_npu
import math
import numpy as np
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

import tilelang
import tilelang.language as T

tilelang.disable_cache()

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128

VECTOR_BASEG = 16
VECTOR_BASEN = 256

DQUERY = 512
DQUERY_INDEX = 128

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass
class SparseLightningIndexerGradKLLossTilingData:
    B: int
    S1: int
    S2: int
    N1: int
    N2: int
    G: int
    Nidx1: int
    Nidx2: int
    DQuery: int
    DQueryIndex: int
    K: int
    sparse_mode: int = 3

    @property
    def g_size(self) -> int:
        return self.G

    @property
    def n2_size(self) -> int:
        return self.N2

    @property
    def block_num(self) -> int:
        return self.B * self.N2 * ceil_div(self.S1, BLOCK_M)

    @property
    def n_num(self) -> int:
        return self.K // BLOCK_N

    @property
    def d_num_query(self) -> int:
        return self.DQuery // BLOCK_K

    @property
    def d_num_query_index(self) -> int:
        return self.DQueryIndex // BLOCK_K


def check_param_valid(
    query: torch.Tensor,
    key: torch.Tensor,
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    weights: torch.Tensor,
    sparse_indices: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    *,
    sparse_mode: int = 3,
    layout_query: str = "BSND",
) -> SparseLightningIndexerGradKLLossTilingData:
    assert query.dim() == 4, f"query dim must be 4 (BSND), got {query.dim()}"
    assert key.dim() == 4, f"key dim must be 4 (BSND), got {key.dim()}"
    assert query_index.dim() == 4, f"queryIndex dim must be 4, got {query_index.dim()}"
    assert key_index.dim() == 4, f"keyIndex dim must be 4, got {key_index.dim()}"
    assert query.dtype in (torch.float16, torch.bfloat16), f"query dtype must be float16/bfloat16, got {query.dtype}"
    assert key.dtype == query.dtype, f"key dtype must match query, got {key.dtype}"
    assert query_index.dtype == query.dtype, f"queryIndex dtype must match query, got {query_index.dtype}"
    assert key_index.dtype == query.dtype, f"keyIndex dtype must match query, got {key_index.dtype}"
    assert weights.dtype in (torch.float16, torch.bfloat16, torch.float32), f"weights dtype must be float16/bfloat16/float32, got {weights.dtype}"
    if weights.dtype != torch.float32:
        assert weights.dtype == query.dtype, f"weights dtype must match query when not float32, got {weights.dtype}"
    assert sparse_mode == 3, f"sparse_mode must be 3, got {sparse_mode}"

    B, S1, N1, DQuery = query.shape
    _, S2, N2, _ = key.shape
    _, _, Nidx1, DQueryIndex = query_index.shape
    _, _, Nidx2, _ = key_index.shape
    K = sparse_indices.shape[-1]

    assert N2 == 1, f"N2 must be 1 (MQA), got {N2}"
    G = N1 // N2
    assert N1 == N2 * G, f"N1({N1}) must be divisible by N2({N2})"
    assert DQuery == DQUERY, f"DQuery must be {DQUERY}, got {DQuery}"
    assert DQueryIndex == DQUERY_INDEX, f"DQueryIndex must be {DQUERY_INDEX}, got {DQueryIndex}"
    assert N1 in (32, 64, 128), f"N1 must be 32/64/128, got {N1}"
    assert Nidx1 in (8, 16, 32, 64), f"Nidx1 must be 8/16/32/64, got {Nidx1}"
    assert K % 1024 == 0 and 1024 <= K <= 8192, f"K must be in [1024,8192] and multiple of 1024, got {K}"

    return SparseLightningIndexerGradKLLossTilingData(
        B=B, S1=S1, S2=S2, N1=N1, N2=N2, G=G,
        Nidx1=Nidx1, Nidx2=Nidx2, DQuery=DQuery, DQueryIndex=DQueryIndex,
        K=K, sparse_mode=sparse_mode,
    )


# ==============================================================================
# Kernel 1: Cube1 - Matmul1 (query @ gatherKey^T) + Matmul2 (queryIndex @ gatherKeyIndex^T)
# GatherKey: (B, S1, K, DQuery) - no G dimension (MQA, all groups share same key)
# GatherKeyIndex: (B, S1, K, DQueryIndex) - no Nidx1 dimension (all gidx share same keyIndex)
# Output: bmm1Res (B, S1, N1, K), bmm2Res (B, S1, Nidx1, K) with ReLU
# ==============================================================================
@tilelang.jit(out_idx=[-2, -1], workspace_idx=[], pass_configs=pass_configs)
def tilelang_cube1_bsnd(
    B, S1, S2, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
    input_dtype, calc_dtype
):
    block_m = BLOCK_M
    block_n = BLOCK_N
    block_k = BLOCK_K
    s1_tile_num = ceil_div(S1, block_m)
    block_num = B * s1_tile_num
    n_num = K // block_n
    d_num_query = DQuery // block_k
    d_num_query_index = DQueryIndex // block_k

    @T.prim_func
    def main(
        Query: T.Tensor((B, S1, N1, DQuery), input_dtype),
        GatherKey: T.Tensor((B, S1, K, DQuery), input_dtype),
        QueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), input_dtype),
        GatherKeyIndex: T.Tensor((B, S1, K, DQueryIndex), input_dtype),
        bmm1Res: T.Tensor((B, S1, N1, K), calc_dtype),
        bmm2Res: T.Tensor((B, S1, Nidx1, K), calc_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bz = cid // s1_tile_num
            m = cid % s1_tile_num
            s1_start = m * block_m
            s1_end = T.min(s1_start + block_m, S1)

            with T.Scope("C"):
                T.barrier_all()

                # --- Matmul1: query @ gatherKey^T -> bmm1Res ---
                # query[b, s1, n1, :]: (1, DQuery)
                # gatherKey[b, s1, k_start:k_end, :]: (block_n, DQuery)
                # result: (1, block_n) -> bmm1Res[b, s1, n1, k_start:k_end]
                Q_L1 = T.alloc_L1((1, block_k), input_dtype)
                GK_L1 = T.alloc_L1((block_n, block_k), input_dtype)
                C_L0C = T.alloc_L0C((1, block_n), calc_dtype)

                for s1_row in T.serial(block_m):
                    for n1_idx in T.serial(N1):
                        for n_idx in T.serial(n_num):
                            T.barrier_all()
                            for k_idx in T.serial(d_num_query):
                                T.copy(
                                    Query[bz, s1_start + s1_row, n1_idx, k_idx * block_k:(k_idx + 1) * block_k],
                                    Q_L1,
                                )
                                T.barrier_all()
                                T.copy(
                                    GatherKey[bz, s1_start + s1_row, n_idx * block_n:(n_idx + 1) * block_n, k_idx * block_k:(k_idx + 1) * block_k],
                                    GK_L1,
                                )
                                T.barrier_all()
                                T.gemm_v0(Q_L1, GK_L1, C_L0C, transpose_B=True, init=(k_idx == 0))
                                T.barrier_all()
                            T.copy(
                                C_L0C,
                                bmm1Res[bz, s1_start + s1_row, n1_idx, n_idx * block_n:(n_idx + 1) * block_n],
                            )
                            T.barrier_all()

                # --- Matmul2: queryIndex @ gatherKeyIndex^T -> bmm2Res (with ReLU) ---
                # queryIndex[b, s1, gidx, :]: (1, DQueryIndex)
                # gatherKeyIndex[b, s1, k_start:k_end, :]: (block_n, DQueryIndex)
                # result: (1, block_n) -> bmm2Res[b, s1, gidx, k_start:k_end]
                QI_L1 = T.alloc_L1((1, block_k), input_dtype)
                GKI_L1 = T.alloc_L1((block_n, block_k), input_dtype)
                C2_L0C = T.alloc_L0C((1, block_n), calc_dtype)

                for s1_row in T.serial(block_m):
                    for gidx_idx in T.serial(Nidx1):
                        for n_idx in T.serial(n_num):
                            T.barrier_all()
                            for k_idx in T.serial(d_num_query_index):
                                T.copy(
                                    QueryIndex[bz, s1_start + s1_row, gidx_idx, k_idx * block_k:(k_idx + 1) * block_k],
                                    QI_L1,
                                )
                                T.barrier_all()
                                T.copy(
                                    GatherKeyIndex[bz, s1_start + s1_row, n_idx * block_n:(n_idx + 1) * block_n, k_idx * block_k:(k_idx + 1) * block_k],
                                    GKI_L1,
                                )
                                T.barrier_all()
                                T.gemm_v0(QI_L1, GKI_L1, C2_L0C, transpose_B=True, init=(k_idx == 0))
                                T.barrier_all()
                            T.copy(
                                C2_L0C,
                                bmm2Res[bz, s1_start + s1_row, gidx_idx, n_idx * block_n:(n_idx + 1) * block_n],
                                enable_relu=True,
                            )
                            T.barrier_all()

    return main


# ==============================================================================
# Kernel 2: Cube2 - Matmul4 (reluGrad @ gatherKeyIndex^T -> dQueryIndex)
# reluGrad: (B, S1, Nidx1, K) - computed in Python layer
# GatherKeyIndex: (B, S1, K, DQueryIndex)
# Output: dQueryIndex (B, S1, Nidx1, DQueryIndex)
# ==============================================================================
@tilelang.jit(out_idx=[-1], workspace_idx=[], pass_configs=pass_configs)
def tilelang_cube2_bsnd(
    B, S1, S2, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
    input_dtype, calc_dtype
):
    block_m = BLOCK_M
    block_n = BLOCK_N
    block_k = BLOCK_K
    s1_tile_num = ceil_div(S1, block_m)
    block_num = B * s1_tile_num
    n_num = K // block_n
    d_num_query_index = DQueryIndex // block_k

    @T.prim_func
    def main(
        ReluGrad: T.Tensor((B, S1, Nidx1, K), input_dtype),
        GatherKeyIndex: T.Tensor((B, S1, K, DQueryIndex), input_dtype),
        dQueryIndex_ws: T.Tensor((B, S1, Nidx1, DQueryIndex), calc_dtype),
        dQueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), input_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bz = cid // s1_tile_num
            m = cid % s1_tile_num
            s1_start = m * block_m
            s1_end = T.min(s1_start + block_m, S1)

            with T.Scope("C"):
                T.barrier_all()

                # Matmul4: dQueryIndex[gidx, d] = sum_k reluGrad[gidx, k] * gatherKeyIndex[k, d]
                # reluGrad: (1, block_n), gatherKeyIndex: (block_n, block_k)
                # gemm: (1, block_n) @ (block_n, block_k) = (1, block_k)
                RG_L1 = T.alloc_L1((1, block_n), input_dtype)
                GKI_L1 = T.alloc_L1((block_n, block_k), input_dtype)
                DQ_L0C = T.alloc_L0C((1, block_k), calc_dtype)

                for s1_row in T.serial(block_m):
                    for gidx_idx in T.serial(Nidx1):
                        for d_idx in T.serial(d_num_query_index):
                            for n_idx in T.serial(n_num):
                                T.barrier_all()
                                T.copy(
                                    ReluGrad[bz, s1_start + s1_row, gidx_idx, n_idx * block_n:(n_idx + 1) * block_n],
                                    RG_L1,
                                )
                                T.barrier_all()
                                T.copy(
                                    GatherKeyIndex[bz, s1_start + s1_row, n_idx * block_n:(n_idx + 1) * block_n, d_idx * block_k:(d_idx + 1) * block_k],
                                    GKI_L1,
                                )
                                T.barrier_all()
                                T.gemm_v0(RG_L1, GKI_L1, DQ_L0C, init=(n_idx == 0))
                                T.barrier_all()
                            T.copy(
                                DQ_L0C,
                                dQueryIndex_ws[bz, s1_start + s1_row, gidx_idx, d_idx * block_k:(d_idx + 1) * block_k],
                            )
                            T.barrier_all()

            # Cast dQueryIndex_ws FP32 -> FP16
            with T.Scope("V"):
                dq_fp32_ub = T.alloc_ub(block_k, calc_dtype)
                dq_fp16_ub = T.alloc_ub(block_k, input_dtype)
                for s1_row in T.serial(block_m):
                    for gidx_idx in T.serial(Nidx1):
                        for d_idx in T.serial(d_num_query_index):
                            T.barrier_all()
                            T.copy(
                                dQueryIndex_ws[bz, s1_start + s1_row, gidx_idx, d_idx * block_k:(d_idx + 1) * block_k],
                                dq_fp32_ub,
                            )
                            T.barrier_all()
                            T.tile.cast(dq_fp16_ub, dq_fp32_ub, "CAST_ROUND", block_k)
                            T.barrier_all()
                            T.copy(
                                dq_fp16_ub,
                                dQueryIndex[bz, s1_start + s1_row, gidx_idx, d_idx * block_k:(d_idx + 1) * block_k],
                            )
                            T.barrier_all()

    return main


# ==============================================================================
# Gather preprocessing: from Key/KeyIndex + sparseIndices -> GatherKey/GatherKeyIndex
# ==============================================================================
def _preprocess_gather(key, key_index, sparse_indices, B, S1, S2, N2, Nidx2, K, DQuery, DQueryIndex):
    device = key.device
    dtype = key.dtype

    # sparse_indices: (B, S1, Nidx2, K), Nidx2=1
    sparse_idx_3d = sparse_indices[:, :, 0, :]  # (B, S1, K)
    sparse_idx_3d = sparse_idx_3d.clamp(min=0)
    batch_idx = torch.arange(B, device=device).view(B, 1, 1)

    # Gather key: (B, S2, N2, DQuery) -> (B, S1, K, DQuery)
    # For each (b, s1, k), gather key[b, sparse_indices[b,s1,k], :, :]
    gather_key = key[batch_idx, sparse_idx_3d, 0, :]  # (B, S1, K, DQuery)

    # Gather keyIndex: (B, S2, Nidx2, DQueryIndex) -> (B, S1, K, DQueryIndex)
    gather_key_index = key_index[batch_idx, sparse_idx_3d, 0, :]  # (B, S1, K, DQueryIndex)

    return gather_key.contiguous(), gather_key_index.contiguous()


# ==============================================================================
# Python-layer vector computation (arch22 reference)
# ==============================================================================
def _compute_vector_outputs(
    bmm1_res, bmm2_res, weights, softmax_max, softmax_sum,
    scale_value, B, S1, N1, N2, G, Nidx1, K, DQuery, DQueryIndex,
    input_dtype,
):
    """
    Compute softmaxI, P_reduced, dWeight, reluGrad in Python (arch22 logic).

    bmm1_res: (B, S1, N1, K) float32 - query @ gatherKey^T
    bmm2_res: (B, S1, Nidx1, K) float32 - ReLU(queryIndex @ gatherKeyIndex^T)
    weights: (B, S1, Nidx1) float16
    softmax_max: (B, N2, S1, G) float32
    softmax_sum: (B, N2, S1, G) float32
    scale_value: float

    Returns:
        d_weights: (B, S1, Nidx1) float32
        relu_grad: (B, S1, Nidx1, K) float16
        kl_loss: float32 scalar
    """
    # P = softmax(bmm1Res * scale) per (b, s1, g)
    # bmm1_res: (B, S1, N1, K) where N1 = N2 * G = G (since N2=1)
    # Reshape to (B, S1, G, K) for per-group softmax
    bmm1 = bmm1_res * scale_value  # (B, S1, N1, K)

    # softmax_max/sum: (B, N2, S1, G) -> (B, S1, G) since N2=1
    sm_max = softmax_max.squeeze(1)  # (B, S1, G)
    sm_sum = softmax_sum.squeeze(1)  # (B, S1, G)

    # P[b, s1, g, k] = exp(bmm1[b,s1,g,k] - sm_max[b,s1,g]) / sm_sum[b,s1,g]
    # bmm1 shape: (B, S1, G, K), sm_max shape: (B, S1, G) -> unsqueeze to (B, S1, G, 1)
    P = torch.exp(bmm1 - sm_max.unsqueeze(-1)) / sm_sum.unsqueeze(-1)  # (B, S1, G, K)

    # P_reduced[k] = (1/G) * sum_g P[g, k]
    P_reduced = P.mean(dim=2)  # (B, S1, K)

    # softmaxI = softmax(sum_g(ReLU(bmm2Res[g,k]) * weight[g]))
    # bmm2_res already has ReLU applied (enable_relu=True in kernel)
    # bmm2_res: (B, S1, Nidx1, K) - this is reluRes
    relu_res = bmm2_res  # (B, S1, Nidx1, K)

    # weight: (B, S1, Nidx1) -> unsqueeze to (B, S1, Nidx1, 1)
    w = weights.float().unsqueeze(-1)  # (B, S1, Nidx1, 1)

    # weighted_relu = reluRes * weight: (B, S1, Nidx1, K)
    weighted_relu = relu_res * w  # (B, S1, Nidx1, K)

    # reduceSumY[k] = sum_g(weighted_relu[g, k]): sum over Nidx1 (g) dimension
    reduce_sum_y = weighted_relu.sum(dim=2)  # (B, S1, K)

    # softmaxI = softmax(reduceSumY) per (b, s1)
    # Standard softmax: exp(x - max) / sum(exp(x - max))
    rsy_max = reduce_sum_y.max(dim=-1, keepdim=True).values  # (B, S1, 1)
    rsy_exp = torch.exp(reduce_sum_y - rsy_max)  # (B, S1, K)
    rsy_sum = rsy_exp.sum(dim=-1, keepdim=True)  # (B, S1, 1)
    softmax_i = rsy_exp / rsy_sum  # (B, S1, K)

    # dWeight[g] = sum_k((softmaxI[k] - P_reduced[k]) * reluRes[g, k])
    diff = softmax_i - P_reduced  # (B, S1, K)
    # relu_res: (B, S1, Nidx1, K), diff: (B, S1, K) -> unsqueeze diff
    d_weights = (diff.unsqueeze(2) * relu_res).sum(dim=-1)  # (B, S1, Nidx1)

    # reluGrad[g, k] = weight[g] * (softmaxI[k] - P_reduced[k]) * I(reluRes[g,k] > 0)
    # Since reluRes = ReLU(bmm2Res), reluRes >= 0 always. I(reluRes > 0) = I(reluRes > 0)
    relu_mask = (relu_res > 0).float()  # (B, S1, Nidx1, K)
    relu_grad = w * diff.unsqueeze(2) * relu_mask  # (B, S1, Nidx1, K)
    relu_grad_half = relu_grad.to(input_dtype)

    # KL Loss = sum over all elements of (softmaxI * log(softmaxI / P_reduced))
    # This is computed per (b, s1) as sum_k(softmaxI[k] * (log(softmaxI[k]) - log(P_reduced[k])))
    # Use log_softmax for numerical stability
    # Note: P_reduced is already a probability distribution, softmaxI is too
    # KL = sum_k(softmaxI[k] * log(softmaxI[k] / P_reduced[k]))
    eps = 1e-10
    kl_loss = (softmax_i * (torch.log(softmax_i + eps) - torch.log(P_reduced + eps))).sum()

    return d_weights, relu_grad_half, kl_loss


# ==============================================================================
# ScatterAdd: compute dKeyIndex from reluGrad and queryIndex
# ==============================================================================
def _compute_d_key_index(relu_grad, query_index, sparse_indices, B, S1, S2, Nidx2, K, DQueryIndex, device, dtype):
    """
    scatterAdd = reluGrad^T @ queryIndex per (b, s1)
    Then scatter into dKeyIndex using sparseIndices.
    """
    relu_grad_fp32 = relu_grad.float()  # (B, S1, Nidx1, K)
    query_index_fp32 = query_index.float()  # (B, S1, Nidx1, DQueryIndex)

    # scatterAdd[b, s1] = reluGrad[b,s1]^T @ queryIndex[b,s1]
    # relu_grad: (B, S1, Nidx1, K) -> transpose last two -> (B, S1, K, Nidx1)
    relu_grad_t = relu_grad_fp32.permute(0, 1, 3, 2)  # (B, S1, K, Nidx1)
    scatter_add = torch.bmm(
        relu_grad_t.reshape(B * S1, K, Nidx1),
        query_index_fp32.reshape(B * S1, Nidx1, DQueryIndex),
    ).reshape(B, S1, K, DQueryIndex)

    # Scatter into dKeyIndex using sparseIndices
    d_key_index = torch.zeros(B, S2, Nidx2, DQueryIndex, dtype=torch.float32, device=device)
    sparse_idx = sparse_indices[:, :, 0, :].clamp(min=0)  # (B, S1, K)

    for b in range(B):
        for s in range(S1):
            valid_mask = sparse_indices[b, s, 0, :] >= 0
            k_indices = sparse_idx[b, s, valid_mask]
            sa_values = scatter_add[b, s, valid_mask]
            if k_indices.numel() > 0:
                d_key_index[b, k_indices, 0, :] += sa_values

    return d_key_index.to(dtype)


# ==============================================================================
# Main entry: BSND layout
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

    B = tiling.B
    S1 = tiling.S1
    S2 = tiling.S2
    N1 = tiling.N1
    N2 = tiling.N2
    G = tiling.G
    Nidx1 = tiling.Nidx1
    Nidx2 = tiling.Nidx2
    DQuery = tiling.DQuery
    DQueryIndex = tiling.DQueryIndex
    K = tiling.K

    input_dtype = query.dtype
    calc_dtype_str = "float"
    input_dtype_str = "float16" if input_dtype == torch.float16 else "bfloat16"
    weights_orig_dtype = weights.dtype

    if layout_query == "BSND" and sparse_mode in (3,):
        if actual_seq_lengths_query is not None:
            actual_q_len = torch.tensor(actual_seq_lengths_query, dtype=torch.int32) if isinstance(actual_seq_lengths_query, list) else actual_seq_lengths_query.to(torch.int32)
        else:
            actual_q_len = torch.full((B,), S1, dtype=torch.int32)

        if actual_seq_lengths_key is not None:
            actual_k_len = torch.tensor(actual_seq_lengths_key, dtype=torch.int32) if isinstance(actual_seq_lengths_key, list) else actual_seq_lengths_key.to(torch.int32)
        else:
            actual_k_len = torch.full((B,), S2, dtype=torch.int32)

        S2_pad = (S2 + VECTOR_BASEN - 1) // VECTOR_BASEN * VECTOR_BASEN
        if S2_pad > S2:
            key = torch.nn.functional.pad(key, (0, 0, 0, 0, 0, S2_pad - S2, 0, 0))
            key_index = torch.nn.functional.pad(key_index, (0, 0, 0, 0, 0, S2_pad - S2, 0, 0))

        if weights.dtype == torch.float32:
            weights = weights.to(input_dtype)

        # Step 1: Gather preprocessing
        gather_key, gather_key_index = _preprocess_gather(
            key, key_index, sparse_indices,
            B, S1, S2_pad, N2, Nidx2, K, DQuery, DQueryIndex,
        )

        # Step 2: Cube1 kernel - Matmul1 + Matmul2
        func_cube1 = tilelang_cube1_bsnd(
            B, S1, S2_pad, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
            input_dtype_str, calc_dtype_str,
        )

        bmm1_res_out = torch.zeros(B, S1, N1, K, dtype=torch.float32, device=query.device)
        bmm2_res_out = torch.zeros(B, S1, Nidx1, K, dtype=torch.float32, device=query.device)

        torch.npu.synchronize()
        bmm1_res, bmm2_res = func_cube1(
            query, gather_key, query_index, gather_key_index,
            bmm1_res_out, bmm2_res_out,
        )
        torch.npu.synchronize()

        # Step 3: Python-layer vector computation
        d_weights, relu_grad, kl_loss = _compute_vector_outputs(
            bmm1_res, bmm2_res, weights, softmax_max, softmax_sum,
            scale_value, B, S1, N1, N2, G, Nidx1, K, DQuery, DQueryIndex,
            input_dtype,
        )

        # Step 4: Cube2 kernel - Matmul4 (dQueryIndex)
        func_cube2 = tilelang_cube2_bsnd(
            B, S1, S2_pad, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
            input_dtype_str, calc_dtype_str,
        )

        d_query_index_ws = torch.zeros(B, S1, Nidx1, DQueryIndex, dtype=torch.float32, device=query.device)
        d_query_index_out = torch.zeros(B, S1, Nidx1, DQueryIndex, dtype=input_dtype, device=query.device)

        torch.npu.synchronize()
        d_query_index = func_cube2(
            relu_grad, gather_key_index,
            d_query_index_ws, d_query_index_out,
        )
        torch.npu.synchronize()

        # Step 5: ScatterAdd (dKeyIndex)
        d_key_index = _compute_d_key_index(
            relu_grad, query_index, sparse_indices,
            B, S1, S2_pad, Nidx2, K, DQueryIndex,
            query.device, input_dtype,
        )

        # Cast dWeights to original dtype
        if weights_orig_dtype == torch.float32:
            d_weights = d_weights.float()
        else:
            d_weights = d_weights.to(input_dtype)

        loss_tensor = kl_loss.reshape(1).to(torch.float32).npu()

        return d_query_index, d_key_index, d_weights, loss_tensor

    raise ValueError(f"Unsupported layout_query={layout_query} with sparse_mode={sparse_mode}")


# ==============================================================================
# TND layout entry
# ==============================================================================
def sparse_lightning_indexer_grad_kl_loss_tnd(
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
    actual_seq_lengths_query: Optional[torch.Tensor] = None,
    actual_seq_lengths_key: Optional[torch.Tensor] = None,
    sparse_mode: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query_b = query.unsqueeze(0)
    key_b = key.unsqueeze(0)
    query_index_b = query_index.unsqueeze(0)
    key_index_b = key_index.unsqueeze(0)
    weights_b = weights.unsqueeze(0)
    sparse_indices_b = sparse_indices.unsqueeze(0)
    softmax_max_b = softmax_max.unsqueeze(0)
    softmax_sum_b = softmax_sum.unsqueeze(0)

    if actual_seq_lengths_query is not None:
        actual_q_len = torch.tensor(actual_seq_lengths_query, dtype=torch.int32) if isinstance(actual_seq_lengths_query, list) else actual_seq_lengths_query.to(torch.int32)
    else:
        actual_q_len = torch.tensor([query.shape[0]], dtype=torch.int32)

    if actual_seq_lengths_key is not None:
        actual_k_len = torch.tensor(actual_seq_lengths_key, dtype=torch.int32) if isinstance(actual_seq_lengths_key, list) else actual_seq_lengths_key.to(torch.int32)
    else:
        actual_k_len = torch.tensor([key.shape[0]], dtype=torch.int32)

    d_qi, d_ki, dw, loss = sparse_lightning_indexer_grad_kl_loss(
        query_b, key_b, query_index_b, key_index_b, weights_b, sparse_indices_b,
        softmax_max_b, softmax_sum_b, scale_value,
        actual_seq_lengths_query=actual_q_len,
        actual_seq_lengths_key=actual_k_len,
        layout_query="BSND",
        sparse_mode=sparse_mode,
    )

    return d_qi.squeeze(0), d_ki.squeeze(0), dw.squeeze(0), loss
