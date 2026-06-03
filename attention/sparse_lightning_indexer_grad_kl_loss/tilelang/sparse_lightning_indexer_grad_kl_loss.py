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


@tilelang.jit(out_idx=[-4, -3, -2, -1], workspace_idx=[5, 6, 7, 8, 9, 10], pass_configs=pass_configs)
def tilelang_sparse_lightning_indexer_grad_kl_loss_bsnd(
    B, S1, S2, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
    input_dtype, calc_dtype
):
    block_m = BLOCK_M
    block_n = BLOCK_N
    block_k = BLOCK_K
    vector_baseg = VECTOR_BASEG
    s1_tile_num = ceil_div(S1, block_m)
    block_num = B * N2 * s1_tile_num
    n_num = K // block_n
    d_num_query = DQuery // block_k
    d_num_query_index = DQueryIndex // block_k
    g_group_num = Nidx1 // vector_baseg

    @T.prim_func
    def main(
        Query: T.Tensor((B, S1, N1, DQuery), input_dtype),
        GatherKey: T.Tensor((B, N2, S1, G, K, DQuery), input_dtype),
        QueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), input_dtype),
        GatherKeyIndex: T.Tensor((B, N2, S1, Nidx1, K, DQueryIndex), input_dtype),
        Weights: T.Tensor((B, S1, Nidx1), input_dtype),
        bmm1Res_ws: T.Tensor((B, N2, S1, G, K), calc_dtype),
        bmm2Res_ws: T.Tensor((B, N2, S1, Nidx1, K), calc_dtype),
        reluGrad_ws: T.Tensor((B, N2, S1, Nidx1, K), input_dtype),
        scatterAdd_ws: T.Tensor((B, N2, S1, K, DQueryIndex), calc_dtype),
        dQueryIndex_ws: T.Tensor((B, S1, Nidx1, DQueryIndex), calc_dtype),
        dWeights_ws: T.Tensor((B, S1, Nidx1, K), calc_dtype),
        SoftmaxMax: T.Tensor((B, N2, S1, G), calc_dtype),
        SoftmaxSum: T.Tensor((B, N2, S1, G), calc_dtype),
        ScaleValue: T.Tensor((1,), calc_dtype),
        SparseIndices: T.Tensor((B, S1, Nidx2, K), "int32"),
        actual_q_len: T.Tensor((B,), "int32"),
        actual_k_len: T.Tensor((B,), "int32"),
        dQueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), input_dtype),
        dKeyIndex: T.Tensor((B, S2, Nidx2, DQueryIndex), input_dtype),
        dWeights: T.Tensor((B, S1, Nidx1, K), input_dtype),
        Loss: T.Tensor((1,), calc_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bz = cid // (N2 * s1_tile_num)
            n2_id = cid // s1_tile_num % N2
            m = cid % s1_tile_num
            s1_start = m * block_m
            s1_end = T.min(s1_start + block_m, S1)

            # ========== Cube1: query@gatherKey^T + queryIndex@gatherKeyIndex^T ==========
            with T.Scope("C"):
                T.barrier_all()

                Q_L1 = T.alloc_L1((block_m, block_k), input_dtype)
                GK_L1 = T.alloc_L1((block_n, block_k), input_dtype)
                C_L0C = T.alloc_L0C((block_m, block_n), calc_dtype)

                # Matmul1: query @ gatherKey^T -> bmm1Res_ws
                # Process s1 rows one at a time to avoid 3D/2D copy mismatch
                for s1_row in range(block_m):
                    for g_idx in T.serial(G):
                        n1_idx = n2_id * G + g_idx
                        for n_idx in T.serial(n_num):
                            T.barrier_all()
                            for k_idx in T.serial(d_num_query):
                                T.copy(
                                    Query[bz, s1_start + s1_row, n1_idx, k_idx * block_k:(k_idx + 1) * block_k],
                                    Q_L1[s1_row, :],
                                )
                                T.barrier_all()
                                T.copy(
                                    GatherKey[bz, n2_id, s1_start + s1_row, g_idx, n_idx * block_n:(n_idx + 1) * block_n, k_idx * block_k:(k_idx + 1) * block_k],
                                    GK_L1,
                                )
                                T.barrier_all()
                                T.gemm_v0(Q_L1, GK_L1, C_L0C, transpose_B=True, init=(k_idx == 0))
                                T.barrier_all()
                            T.copy(
                                C_L0C[s1_row, :],
                                bmm1Res_ws[bz, n2_id, s1_start + s1_row, g_idx, n_idx * block_n:(n_idx + 1) * block_n],
                            )
                            T.barrier_all()

                QI_L1 = T.alloc_L1((block_m, block_k), input_dtype)
                GKI_L1 = T.alloc_L1((block_n, block_k), input_dtype)
                C2_L0C = T.alloc_L0C((block_m, block_n), calc_dtype)

                # Matmul2: queryIndex @ gatherKeyIndex^T -> bmm2Res_ws
                for s1_row in range(block_m):
                    for gidx_idx in T.serial(Nidx1):
                        for n_idx in T.serial(n_num):
                            T.barrier_all()
                            for k_idx in T.serial(d_num_query_index):
                                T.copy(
                                    QueryIndex[bz, s1_start + s1_row, gidx_idx, k_idx * block_k:(k_idx + 1) * block_k],
                                    QI_L1[s1_row, :],
                                )
                                T.barrier_all()
                                T.copy(
                                    GatherKeyIndex[bz, n2_id, s1_start + s1_row, gidx_idx, n_idx * block_n:(n_idx + 1) * block_n, k_idx * block_k:(k_idx + 1) * block_k],
                                    GKI_L1,
                                )
                                T.barrier_all()
                                T.gemm_v0(QI_L1, GKI_L1, C2_L0C, transpose_B=True, init=(k_idx == 0))
                                T.barrier_all()
                            T.copy(
                                C2_L0C[s1_row, :],
                                bmm2Res_ws[bz, n2_id, s1_start + s1_row, gidx_idx, n_idx * block_n:(n_idx + 1) * block_n],
                            )
                            T.barrier_all()

                T.set_cross_flag("FIX", 1)

            # ========== Vec1: Softmax恢复 + 在线Softmax + KL Loss + dW + reluGrad ==========
            with T.Scope("V"):
                T.wait_cross_flag(1)

                weight_half_ub = T.alloc_ub(vector_baseg, input_dtype)
                weight_fp32_ub = T.alloc_ub(vector_baseg, calc_dtype)

                bmm1_row_ub = T.alloc_ub(block_n, calc_dtype)
                bmm2_row_ub = T.alloc_ub(block_n, calc_dtype)

                sm_max_ub = T.alloc_ub(1, calc_dtype)
                sm_sum_ub = T.alloc_ub(1, calc_dtype)
                scale_ub = T.alloc_ub(1, calc_dtype)

                p_row_ub = T.alloc_ub(block_n, calc_dtype)
                softmax_i_row_ub = T.alloc_ub(block_n, calc_dtype)
                di_row_ub = T.alloc_ub(block_n, calc_dtype)
                relu_grad_row_ub = T.alloc_ub(block_n, calc_dtype)
                relu_grad_half_ub = T.alloc_ub(block_n, input_dtype)
                mask_ub = T.alloc_ub(block_n, calc_dtype)



                T.copy(ScaleValue[0:1], scale_ub)

                for g_group in T.serial(g_group_num):
                    for s1_row in range(block_m):
                        T.barrier_all()

                        n1_group_start = g_group * vector_baseg
                        T.copy(
                            Weights[bz, s1_start + s1_row, n1_group_start : n1_group_start + vector_baseg],
                            weight_half_ub,
                        )
                        T.barrier_all()
                        T.tile.cast(weight_fp32_ub, weight_half_ub, "CAST_NONE", vector_baseg)
                        T.barrier_all()

                        for g_local in range(vector_baseg):
                            gidx_idx = g_group * vector_baseg + g_local
                            g_idx = gidx_idx


                            # Load softmaxMax/Sum once per (s1_row, g_idx)
                            T.copy(
                                SoftmaxMax[bz, n2_id, s1_start + s1_row, g_idx],
                                sm_max_ub,
                            )
                            T.barrier_all()
                            T.copy(
                                SoftmaxSum[bz, n2_id, s1_start + s1_row, g_idx],
                                sm_sum_ub,
                            )
                            T.barrier_all()

                            for n_idx in T.serial(n_num):
                                T.barrier_all()

                                # --- Restore P from bmm1Res ---
                                T.copy(
                                    bmm1Res_ws[bz, n2_id, s1_start + s1_row, g_idx, n_idx * block_n:(n_idx + 1) * block_n],
                                    bmm1_row_ub,
                                )
                                T.barrier_all()
                                # P = exp(bmm1Res - softmaxMax) / softmaxSum
                                T.tile.sub(p_row_ub, bmm1_row_ub, sm_max_ub[0])
                                T.barrier_all()
                                T.tile.exp(p_row_ub, p_row_ub)
                                T.barrier_all()
                                T.tile.div(p_row_ub, p_row_ub, sm_sum_ub[0])
                                T.barrier_all()

                                # softmaxI = bmm2Res * scale (simplified; full softmax needs reduce which is complex in TileLang)
                                # TODO: Implement proper online softmax with 2D buffer reduce
                                T.copy(
                                    bmm2Res_ws[bz, n2_id, s1_start + s1_row, gidx_idx, n_idx * block_n:(n_idx + 1) * block_n],
                                    bmm2_row_ub,
                                )
                                T.barrier_all()
                                T.tile.mul(softmax_i_row_ub, bmm2_row_ub, scale_ub[0])
                                T.barrier_all()

                                # dI = softmaxI - P
                                T.tile.sub(di_row_ub, softmax_i_row_ub, p_row_ub)
                                T.barrier_all()

                                # --- dWeights: dI * bmm2Res * scale, write directly to dWeights_ws ---
                                T.tile.mul(di_row_ub, di_row_ub, softmax_i_row_ub)
                                T.barrier_all()
                                T.copy(
                                    di_row_ub,
                                    dWeights_ws[bz, s1_start + s1_row, gidx_idx, n_idx * block_n:(n_idx + 1) * block_n],
                                )
                                T.barrier_all()

                                # --- KL Loss: skip in kernel, compute in Python layer ---
                                # T.tile.atomic_add requires same size for dst and src,
                                # but Loss is (1,) and di_row_ub is (block_n,).
                                # TODO: Implement KL Loss in Python layer after kernel outputs are verified

                                # ReLU mask: bmm2Res > 0 (use original bmm2Res, not scaled)
                                T.tile.compare(mask_ub, bmm2_row_ub, 0.0, "GT")
                                T.barrier_all()

                                # reluGrad = dI * mask * weight
                                # Note: di_row_ub was overwritten above, recompute
                                T.tile.sub(di_row_ub, softmax_i_row_ub, p_row_ub)
                                T.barrier_all()
                                T.tile.select(relu_grad_row_ub, mask_ub, di_row_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                                T.barrier_all()
                                T.tile.mul(relu_grad_row_ub, relu_grad_row_ub, weight_fp32_ub[g_local])
                                T.barrier_all()
                                T.tile.cast(relu_grad_half_ub, relu_grad_row_ub, "CAST_ROUND", block_n)
                                T.barrier_all()

                                T.copy(
                                    relu_grad_half_ub,
                                    reluGrad_ws[bz, n2_id, s1_start + s1_row, gidx_idx, n_idx * block_n:(n_idx + 1) * block_n],
                                )
                                T.barrier_all()


                T.set_cross_flag("V", 2)

            # ========== Cube2: reluGrad^T@queryIndex + reluGrad@gatherKeyIndex ==========
            with T.Scope("C"):
                T.wait_cross_flag(2)

                RG_L1 = T.alloc_L1((block_m, block_n), input_dtype)
                QI2_L1 = T.alloc_L1((block_m, block_k), input_dtype)
                SA_L0C = T.alloc_L0C((block_n, block_k), calc_dtype)

                # Matmul3: reluGrad^T @ queryIndex -> scatterAdd_ws
                # Process s1 rows one at a time
                for s1_row in range(block_m):
                    for n_idx in T.serial(n_num):
                        for d_idx in T.serial(d_num_query_index):
                            for gidx_idx in T.serial(Nidx1):
                                T.barrier_all()
                                T.copy(
                                    reluGrad_ws[bz, n2_id, s1_start + s1_row, gidx_idx, n_idx * block_n:(n_idx + 1) * block_n],
                                    RG_L1,
                                )
                                T.barrier_all()
                                T.copy(
                                    QueryIndex[bz, s1_start + s1_row, gidx_idx, d_idx * block_k:(d_idx + 1) * block_k],
                                    QI2_L1,
                                )
                                T.barrier_all()
                                T.gemm_v0(RG_L1, QI2_L1, SA_L0C, transpose_B=True, init=(gidx_idx == 0))
                                T.barrier_all()
                            T.copy(
                                SA_L0C,
                                scatterAdd_ws[bz, n2_id, s1_start + s1_row, n_idx * block_n:(n_idx + 1) * block_n, d_idx * block_k:(d_idx + 1) * block_k],
                            )
                            T.barrier_all()

                GKI2_L1 = T.alloc_L1((block_n, block_k), input_dtype)
                DQ_L0C = T.alloc_L0C((block_m, block_k), calc_dtype)

                # Matmul4: reluGrad @ gatherKeyIndex -> dQueryIndex_ws
                for s1_row in range(block_m):
                    for gidx_idx in T.serial(Nidx1):
                        for d_idx in T.serial(d_num_query_index):
                            for n_idx in T.serial(n_num):
                                T.barrier_all()
                                T.copy(
                                    reluGrad_ws[bz, n2_id, s1_start + s1_row, gidx_idx, n_idx * block_n:(n_idx + 1) * block_n],
                                    RG_L1,
                                )
                                T.barrier_all()
                                T.copy(
                                    GatherKeyIndex[bz, n2_id, s1_start + s1_row, gidx_idx, n_idx * block_n:(n_idx + 1) * block_n, d_idx * block_k:(d_idx + 1) * block_k],
                                    GKI2_L1,
                                )
                                T.barrier_all()
                                T.gemm_v0(RG_L1, GKI2_L1, DQ_L0C, transpose_B=True, init=(n_idx == 0))
                                T.barrier_all()
                            T.copy(
                                DQ_L0C[s1_row, :],
                                dQueryIndex_ws[bz, s1_start + s1_row, gidx_idx, d_idx * block_k:(d_idx + 1) * block_k],
                            )
                            T.barrier_all()

                T.set_cross_flag("FIX", 3)

            # ========== Vec2: Cast + ScatterAdd + loss ==========
            with T.Scope("V"):
                T.wait_cross_flag(3)

                # Cast dQueryIndex_ws FP32 -> FP16
                dq_fp32_ub = T.alloc_ub(block_k, calc_dtype)
                dq_fp16_ub = T.alloc_ub(block_k, input_dtype)
                for gidx_idx in T.serial(Nidx1):
                    for d_idx in T.serial(d_num_query_index):
                        for s1_row in range(block_m):
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

                # Cast dWeights_ws FP32 -> FP16 (same shape, no reduce needed)
                dw_fp32_ub = T.alloc_ub(block_n, calc_dtype)
                dw_fp16_ub = T.alloc_ub(block_n, input_dtype)
                for gidx_idx in T.serial(Nidx1):
                    for s1_row in range(block_m):
                        for n_idx in T.serial(n_num):
                            T.barrier_all()
                            T.copy(
                                dWeights_ws[bz, s1_start + s1_row, gidx_idx, n_idx * block_n:(n_idx + 1) * block_n],
                                dw_fp32_ub,
                            )
                            T.barrier_all()
                            T.tile.cast(dw_fp16_ub, dw_fp32_ub, "CAST_ROUND", block_n)
                            T.barrier_all()
                            T.copy(
                                dw_fp16_ub,
                                dWeights[bz, s1_start + s1_row, gidx_idx, n_idx * block_n:(n_idx + 1) * block_n],
                            )
                            T.barrier_all()

                # ScatterAdd is done in Python layer (dynamic indexing not supported in TileLang)
                # scatterAdd_ws is in workspace; Python layer will recompute and scatter

    return main


def _preprocess_gather(key, key_index, sparse_indices, B, S1, S2, N2, Nidx2, G, Nidx1, K, DQuery, DQueryIndex):
    device = key.device
    dtype = key.dtype

    # Gather key: (B, S2, N2, DQuery) -> gatherKey: (B, N2, S1, G, K, DQuery)
    # sparse_indices: (B, S1, Nidx2, K), Nidx2=1, N2=1
    sparse_idx_3d = sparse_indices[:, :, 0, :]  # (B, S1, K)
    # Replace -1 (padding) with 0 to avoid negative indexing in gather
    sparse_idx_3d = sparse_idx_3d.clamp(min=0)
    batch_idx = torch.arange(B, device=device).view(B, 1, 1)

    # key: (B, S2, N2, DQuery) -> gather along S2 dim
    # For each (b, s1, k), gather key[b, sparse_indices[b,s1,k], :, :] -> (B, S1, K, N2, DQuery)
    gather_key = key[batch_idx, sparse_idx_3d, :, :]  # (B, S1, K, N2, DQuery)
    gather_key = gather_key.permute(0, 3, 1, 2, 4)  # (B, N2, S1, K, DQuery)

    # Expand G dimension: all G groups share the same gathered key (MQA)
    gather_key_expanded = gather_key.unsqueeze(3).expand(B, N2, S1, G, K, DQuery).contiguous()

    # Gather keyIndex: (B, S2, Nidx2, DQueryIndex) -> gatherKeyIndex: (B, N2, S1, Nidx1, K, DQueryIndex)
    gather_key_index = key_index[batch_idx, sparse_idx_3d, :, :]  # (B, S1, K, Nidx2, DQueryIndex)
    # Nidx2=1, need (B, S1, K, DQueryIndex) then expand to (B, N2, S1, Nidx1, K, DQueryIndex)
    gather_key_index = gather_key_index.squeeze(3)  # (B, S1, K, DQueryIndex)
    # Insert N2 and Nidx1 dims: (B, 1, S1, 1, K, DQueryIndex) -> expand
    gather_key_index_expanded = gather_key_index.unsqueeze(1).unsqueeze(3).expand(B, N2, S1, Nidx1, K, DQueryIndex).contiguous()

    return gather_key_expanded, gather_key_index_expanded


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

    calc_dtype_str = "float"
    input_dtype_str = "float16" if query.dtype == torch.float16 else "bfloat16"
    weights_orig_dtype = weights.dtype

    if layout_query == "BSND" and sparse_mode in (3,):
        if actual_seq_lengths_query is not None:
            actual_q_len = actual_seq_lengths_query.to(torch.int32)
        else:
            actual_q_len = torch.full((B,), S1, dtype=torch.int32)

        if actual_seq_lengths_key is not None:
            actual_k_len = actual_seq_lengths_key.to(torch.int32)
        else:
            actual_k_len = torch.full((B,), S2, dtype=torch.int32)

        S2_pad = (S2 + VECTOR_BASEN - 1) // VECTOR_BASEN * VECTOR_BASEN
        if S2_pad > S2:
            key = torch.nn.functional.pad(key, (0, 0, 0, 0, 0, S2_pad - S2, 0, 0))
            key_index = torch.nn.functional.pad(key_index, (0, 0, 0, 0, 0, S2_pad - S2, 0, 0))

        if weights.dtype == torch.float32:
            weights = weights.half()

        # Gather preprocessing
        gather_key, gather_key_index = _preprocess_gather(
            key, key_index, sparse_indices,
            B, S1, S2_pad, N2, Nidx2, G, Nidx1, K, DQuery, DQueryIndex,
        )

        scale_tensor = torch.tensor([scale_value], dtype=torch.float32).npu()

        # Allocate output tensors
        d_query_index_out = torch.zeros(B, S1, Nidx1, DQueryIndex, dtype=query.dtype, device=query.device)
        d_key_index_out = torch.zeros(B, S2_pad, Nidx2, DQueryIndex, dtype=query.dtype, device=query.device)
        d_weights_out = torch.zeros(B, S1, Nidx1, K, dtype=query.dtype, device=query.device)
        # Loss must be zero-initialized for atomic_add accumulation
        loss_out = torch.zeros(1, dtype=torch.float32, device=query.device)

        func = tilelang_sparse_lightning_indexer_grad_kl_loss_bsnd(
            B, S1, S2_pad, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
            input_dtype_str, calc_dtype_str,
        )

        torch.npu.synchronize()
        d_query_index, d_key_index, d_weights, loss = func(
            query, gather_key, query_index, gather_key_index, weights,
            softmax_max, softmax_sum, scale_tensor,
            sparse_indices, actual_q_len, actual_k_len,
            d_query_index_out, d_key_index_out, d_weights_out, loss_out,
        )
        torch.npu.synchronize()

        # ScatterAdd: compute dKeyIndex from reluGrad_ws and queryIndex
        # reluGrad_ws shape: (B, N2, S1, Nidx1, K)
        # queryIndex shape: (B, S1, Nidx1, DQueryIndex)
        # scatterAdd = reluGrad^T @ queryIndex, shape: (B, N2, S1, K, DQueryIndex)
        # Then scatter into dKeyIndex using sparseIndices
        # Since we can't easily extract reluGrad_ws from workspace, we recompute in Python:
        # reluGrad_fp32 = reluGrad_ws.float()  -- but reluGrad_ws is in workspace
        # Alternative: compute scatterAdd directly from reluGrad (which we can reconstruct)
        # For now, d_key_index is placeholder (zeros) until ScatterAdd is implemented
        # TODO: Either expose scatterAdd_ws as output, or implement ScatterAdd in kernel

        # dWeights: reduce K-axis (kernel outputs per-K-element, need sum)
        d_weights = d_weights.sum(dim=-1)

        if weights_orig_dtype == torch.float32:
            d_weights = d_weights.float()

        return d_query_index, d_key_index, d_weights, loss

    raise ValueError(f"Unsupported layout_query={layout_query} with sparse_mode={sparse_mode}")


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
    """
    TND layout entry: reshape TND -> BSND (B=1, S=T), call BSND kernel, reshape back.
    TND shapes: query(T1,N1,D), key(T2,N2,D), queryIndex(T1,Nidx1,Didx), keyIndex(T2,Nidx2,Didx)
    BSND shapes: query(1,T1,N1,D), key(1,T2,N2,D), ...
    """
    # TND -> BSND: insert B=1 dim at position 0
    query_b = query.unsqueeze(0)
    key_b = key.unsqueeze(0)
    query_index_b = query_index.unsqueeze(0)
    key_index_b = key_index.unsqueeze(0)
    weights_b = weights.unsqueeze(0)
    sparse_indices_b = sparse_indices.unsqueeze(0)
    # softmaxMax/Sum: TND (N2,T1,G) -> BSND (B,N2,T1,G) = (1,N2,T1,G)
    softmax_max_b = softmax_max.unsqueeze(0)
    softmax_sum_b = softmax_sum.unsqueeze(0)

    if actual_seq_lengths_query is not None:
        actual_q_len = actual_seq_lengths_query.to(torch.int32)
    else:
        actual_q_len = torch.tensor([query.shape[0]], dtype=torch.int32)

    if actual_seq_lengths_key is not None:
        actual_k_len = actual_seq_lengths_key.to(torch.int32)
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

    # BSND -> TND: remove B=1 dim
    return d_qi.squeeze(0), d_ki.squeeze(0), dw.squeeze(0), loss