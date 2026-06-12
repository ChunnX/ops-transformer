import torch
import math
import numpy as np
import os
from typing import Optional, Tuple
from dataclasses import dataclass

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

MAX_CORE_NUM = 32

_kernel_cache = {}
_workspace_cache = {}


def _clear_caches():
    global _kernel_cache, _workspace_cache
    _kernel_cache.clear()
    _workspace_cache.clear()


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
    if layout_query == "TND":
        assert query.dim() == 3
        assert key.dim() == 3
        assert query_index.dim() == 3
        assert key_index.dim() == 3
    else:
        assert query.dim() == 4
        assert key.dim() == 4
        assert query_index.dim() == 4
        assert key_index.dim() == 4
    assert query.dtype in (torch.float16, torch.bfloat16)
    assert key.dtype == query.dtype
    assert query_index.dtype == query.dtype
    assert key_index.dtype == query.dtype
    assert weights.dtype in (torch.float16, torch.bfloat16, torch.float32)
    if weights.dtype != torch.float32:
        assert weights.dtype == query.dtype
    assert sparse_mode == 3

    if layout_query == "TND":
        S1, N1, DQuery = query.shape
        S2, N2, _ = key.shape
        _, Nidx1, DQueryIndex = query_index.shape
        _, Nidx2, _ = key_index.shape
        B = 1
    else:
        B, S1, N1, DQuery = query.shape
        _, S2, N2, _ = key.shape
        _, _, Nidx1, DQueryIndex = query_index.shape
        _, _, Nidx2, _ = key_index.shape
    K = sparse_indices.shape[-1]

    assert N2 == 1
    G = N1 // N2
    assert N1 == N2 * G
    assert DQuery == DQUERY
    assert DQueryIndex == DQUERY_INDEX
    assert N1 in (32, 64, 128)
    assert Nidx1 in (8, 16, 32, 64)
    assert K % 1024 == 0 and 1024 <= K <= 8192

    return SparseLightningIndexerGradKLLossTilingData(
        B=B, S1=S1, S2=S2, N1=N1, N2=N2, G=G,
        Nidx1=Nidx1, Nidx2=Nidx2, DQuery=DQuery, DQueryIndex=DQueryIndex,
        K=K, sparse_mode=sparse_mode,
    )


@tilelang.jit(
    out_idx=[-8, -7, -6, -5],
    workspace_idx=[12, 13, 14, 15, 16, 17, 18, 19, 20],
    pass_configs=pass_configs,
)
def tilelang_grad_kl_loss_pipeline_kernel(
    B, S1, S2, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
    d_rope, input_dtype, calc_dtype, scale_value, core_num
):
    s1_block = BLOCK_M
    n_block = BLOCK_N
    k_block = BLOCK_K
    s1_tile_num = ceil_div(S1, s1_block)
    n_num = K // n_block
    total_tasks = B * s1_tile_num
    block_num = core_num
    d_num_query = DQuery // k_block
    d_num_rope = ceil_div(d_rope, k_block)
    d_num_total = d_num_query + d_num_rope
    d_num_query_index = DQueryIndex // k_block
    vector_baseg = Nidx1 if Nidx1 <= VECTOR_BASEG else VECTOR_BASEG
    g_num = Nidx1 // vector_baseg

    @T.prim_func
    def main(
        Query: T.Tensor((B, S1, N1, DQuery), input_dtype),
        QueryRope: T.Tensor((B, S1, N1, d_rope), input_dtype),
        GatherKey: T.Tensor((B, S1, K, DQuery), input_dtype),
        GatherKeyRope: T.Tensor((B, S1, K, d_rope), input_dtype),
        QueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), input_dtype),
        GatherKeyIndex: T.Tensor((B, S1, K, DQueryIndex), input_dtype),
        GatherKeyIndexT: T.Tensor((B, S1, DQueryIndex, K), input_dtype),
        Weights: T.Tensor((B, S1, Nidx1), input_dtype),
        SoftmaxMax: T.Tensor((B, N2, S1, G), calc_dtype),
        SoftmaxSum: T.Tensor((B, N2, S1, G), calc_dtype),
        bmm1Res: T.Tensor((B, S1, N1, K), calc_dtype),
        bmm2Res: T.Tensor((B, S1, Nidx1, K), calc_dtype),
        PReduced: T.Tensor((B, S1, K), calc_dtype),
        ReduceSumY: T.Tensor((B, S1, K), calc_dtype),
        SoftmaxI: T.Tensor((B, S1, K), calc_dtype),
        dQueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), input_dtype),
        dWeight_partial: T.Tensor((B, S1, n_num, Nidx1), input_dtype),
        reluGradRes: T.Tensor((B, S1, Nidx1, K), input_dtype),
        LossOut: T.Tensor((B, S1, n_num), calc_dtype),
        dQueryIndex_ws: T.Tensor((B, S1, Nidx1, DQueryIndex), calc_dtype),
        SoftmaxIMax: T.Tensor((B, S1), calc_dtype),
        SoftmaxISum: T.Tensor((B, S1), calc_dtype),
        SoftmaxIMax_partial: T.Tensor((B, S1, n_num), calc_dtype),
        SoftmaxISum_partial: T.Tensor((B, S1, n_num), calc_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid):
            # Pipeline stage control: use taskId to determine which row/stage to process
            # taskId = cid + pipeline_offset
            
            # Phase 1: Process all tasks (C1 -> V1 -> C1b -> V2 -> C2 -> V3)
            for pipeline_iter in T.serial(0, total_tasks + 2):
                # Determine which task to process at current pipeline stage
                # Core cid processes task (pipeline_iter - cid) if valid
                task_id = pipeline_iter - cid
                
                if task_id >= 0 and task_id < total_tasks:
                    bz = task_id // s1_tile_num
                    m = task_id % s1_tile_num
                    s1_start = m * s1_block
                    s1_end = T.min(s1_start + s1_block, S1)
                    
                    # ============================================================
                    # Stage C1: Matmul1 + Matmul2 (parallel across cores)
                    # ============================================================
                    if pipeline_iter % 6 == 0 or (pipeline_iter >= 6 and (pipeline_iter - cid * 6) % 6 == 0):
                        with T.Scope("C"):
                            Q_L1 = T.alloc_L1((N1, k_block), input_dtype)
                            GK_L1 = T.alloc_L1((n_block, k_block), input_dtype)
                            C_L0C = T.alloc_L0C((N1, n_block), calc_dtype)
                            
                            for s1_row in T.serial(s1_start, s1_end):
                                for n_idx_k in T.serial(n_num):
                                    for d_idx in T.serial(d_num_total):
                                        if d_idx < d_num_query:
                                            T.copy(Query[bz, s1_row, 0:N1, d_idx * k_block:(d_idx + 1) * k_block], Q_L1)
                                        else:
                                            rope_d = d_idx - d_num_query
                                            T.copy(QueryRope[bz, s1_row, 0:N1, rope_d * k_block:(rope_d + 1) * k_block], Q_L1)
                                        if d_idx < d_num_query:
                                            T.copy(GatherKey[bz, s1_row, n_idx_k * n_block:(n_idx_k + 1) * n_block, d_idx * k_block:(d_idx + 1) * k_block], GK_L1)
                                        else:
                                            rope_d = d_idx - d_num_query
                                            T.copy(GatherKeyRope[bz, s1_row, n_idx_k * n_block:(n_idx_k + 1) * n_block, rope_d * k_block:(rope_d + 1) * k_block], GK_L1)
                                        T.gemm_v0(Q_L1, GK_L1, C_L0C, transpose_B=True, init=(d_idx == 0))
                                    T.copy(C_L0C, bmm1Res[bz, s1_row, 0:N1, n_idx_k * n_block:(n_idx_k + 1) * n_block])
                            
                            QI_L1 = T.alloc_L1((Nidx1, k_block), input_dtype)
                            GKI_L1 = T.alloc_L1((n_block, k_block), input_dtype)
                            C2_L0C = T.alloc_L0C((Nidx1, n_block), calc_dtype)
                            
                            for s1_row in T.serial(s1_start, s1_end):
                                for n_idx_k in T.serial(n_num):
                                    for k_idx in T.serial(d_num_query_index):
                                        T.copy(QueryIndex[bz, s1_row, 0:Nidx1, k_idx * k_block:(k_idx + 1) * k_block], QI_L1)
                                        T.copy(GatherKeyIndex[bz, s1_row, n_idx_k * n_block:(n_idx_k + 1) * n_block, k_idx * k_block:(k_idx + 1) * k_block], GKI_L1)
                                        T.gemm_v0(QI_L1, GKI_L1, C2_L0C, transpose_B=True, init=(k_idx == 0))
                                    T.copy(C2_L0C, bmm2Res[bz, s1_row, 0:Nidx1, n_idx_k * n_block:(n_idx_k + 1) * n_block], enable_relu=True)
                    
                    # ============================================================
                    # Stage V1: Compute P_reduced, ReduceSumY partial
                    # ============================================================
                    if pipeline_iter % 6 == 1 or (pipeline_iter >= 6 and (pipeline_iter - cid * 6) % 6 == 1):
                        with T.Scope("V"):
                            bmm1_ub = T.alloc_ub((N1, n_block), calc_dtype)
                            bmm2_ub = T.alloc_ub((Nidx1, n_block), calc_dtype)
                            weight_ub = T.alloc_ub(vector_baseg, calc_dtype)
                            weight_half_ub = T.alloc_ub(vector_baseg, input_dtype)
                            p_reduced_tmp_ub = T.alloc_ub(n_block, calc_dtype)
                            rsy_tmp_ub = T.alloc_ub((vector_baseg, n_block), calc_dtype)
                            rsy_reduce_ub = T.alloc_ub(n_block, calc_dtype)
                            sm_max_ub = T.alloc_ub(G, calc_dtype)
                            sm_sum_ub = T.alloc_ub(G, calc_dtype)
                            eps_sm_ub = T.alloc_ub(G, calc_dtype)
                            
                            for s1_id in T.serial(s1_start, s1_end):
                                T.copy(SoftmaxMax[bz, 0, s1_id, 0:G], sm_max_ub)
                                T.copy(SoftmaxSum[bz, 0, s1_id, 0:G], sm_sum_ub)
                                T.tile.abs(sm_sum_ub, sm_sum_ub)
                                T.tile.fill(eps_sm_ub, 1e-8)
                                T.tile.max(sm_sum_ub, sm_sum_ub, eps_sm_ub)
                                
                                for n_idx_k in T.serial(n_num):
                                    T.copy(bmm1Res[bz, s1_id, 0:N1, n_idx_k * n_block:(n_idx_k + 1) * n_block], bmm1_ub)
                                    T.tile.mul(bmm1_ub, bmm1_ub, scale_value)
                                    for g_id in range(G):
                                        T.tile.sub(bmm1_ub[g_id, :], bmm1_ub[g_id, :], sm_max_ub[g_id])
                                    T.tile.exp(bmm1_ub, bmm1_ub)
                                    for g_id in range(G):
                                        T.tile.div(bmm1_ub[g_id, :], bmm1_ub[g_id, :], sm_sum_ub[g_id])
                                    T.tile.fill(p_reduced_tmp_ub, 0)
                                    for g_id in range(G):
                                        T.tile.add(p_reduced_tmp_ub, p_reduced_tmp_ub, bmm1_ub[g_id, :])
                                    T.tile.mul(p_reduced_tmp_ub, p_reduced_tmp_ub, 1.0 / G)
                                    T.copy(p_reduced_tmp_ub, PReduced[bz, s1_id, n_idx_k * n_block:(n_idx_k + 1) * n_block])
                                    
                                    T.copy(bmm2Res[bz, s1_id, 0:Nidx1, n_idx_k * n_block:(n_idx_k + 1) * n_block], bmm2_ub)
                                    T.tile.fill(rsy_reduce_ub, 0)
                                    for g_id in range(g_num):
                                        T.copy(Weights[bz, s1_id, g_id * vector_baseg:(g_id + 1) * vector_baseg], weight_half_ub)
                                        T.tile.cast(weight_ub, weight_half_ub, "CAST_NONE", vector_baseg)
                                        for i in range(vector_baseg):
                                            T.tile.mul(rsy_tmp_ub[i, :], bmm2_ub[g_id * vector_baseg + i, :], weight_ub[i])
                                        T.reduce_sum(rsy_tmp_ub, rsy_reduce_ub, 0, clear=False)
                                    T.copy(rsy_reduce_ub, ReduceSumY[bz, s1_id, n_idx_k * n_block:(n_idx_k + 1) * n_block])
                    
                    # ============================================================
                    # Stage C1b: Global SoftmaxI max/sum reduce (all cores cooperate)
                    # ============================================================
                    if pipeline_iter % 6 == 2 or (pipeline_iter >= 6 and (pipeline_iter - cid * 6) % 6 == 2):
                        with T.Scope("C"):
                            rsy_chunk_ub = T.alloc_ub(n_block, calc_dtype)
                            rsy_max_ub = T.alloc_ub(1, calc_dtype)
                            rsy_sum_ub = T.alloc_ub(1, calc_dtype)
                            rsy_chunk_max_ub = T.alloc_ub(1, calc_dtype)
                            rsy_chunk_sum_ub = T.alloc_ub(1, calc_dtype)
                            
                            for s1_id in T.serial(s1_start, s1_end):
                                T.tile.fill(rsy_max_ub, -1e30)
                                T.tile.fill(rsy_sum_ub, 0)
                                
                                # Each core processes assigned n_idx_k blocks
                                for n_idx_k in T.serial(n_num):
                                    T.copy(ReduceSumY[bz, s1_id, n_idx_k * n_block:(n_idx_k + 1) * n_block], rsy_chunk_ub)
                                    
                                    # Local max/sum reduce
                                    rsy_2d_ub = T.alloc_ub((16, n_block), calc_dtype)
                                    for i in range(16):
                                        T.tile.mul(rsy_2d_ub[i, :], rsy_chunk_ub, 1.0)
                                    T.reduce_max(rsy_2d_ub, rsy_chunk_max_ub, 1)
                                    T.reduce_sum(rsy_2d_ub, rsy_chunk_sum_ub, 1)
                                    
                                    # Write partial results to GM (will be reduced by V scope)
                                    T.copy(rsy_chunk_max_ub, SoftmaxIMax_partial[bz, s1_id, n_idx_k])
                                    T.copy(rsy_chunk_sum_ub, SoftmaxISum_partial[bz, s1_id, n_idx_k])
                    
                    # ============================================================
                    # Stage V2: Global SoftmaxI finalize + dWeight/reluGrad/Loss
                    # ============================================================
                    if pipeline_iter % 6 == 3 or (pipeline_iter >= 6 and (pipeline_iter - cid * 6) % 6 == 3):
                        with T.Scope("V"):
                            # Global reduce for SoftmaxI
                            max_partial_ub = T.alloc_ub(n_num, calc_dtype)
                            sum_partial_ub = T.alloc_ub(n_num, calc_dtype)
                            global_max_ub = T.alloc_ub(1, calc_dtype)
                            global_sum_ub = T.alloc_ub(1, calc_dtype)
                            
                            for s1_id in T.serial(s1_start, s1_end):
                                T.copy(SoftmaxIMax_partial[bz, s1_id, 0:n_num], max_partial_ub)
                                T.copy(SoftmaxISum_partial[bz, s1_id, 0:n_num], sum_partial_ub)
                                
                                # Final reduce across all K blocks
                                max_2d_ub = T.alloc_ub((2, n_num), calc_dtype)
                                for i in range(n_num):
                                    T.tile.mul(max_2d_ub[0, i], max_partial_ub[i], 1.0)
                                    T.tile.mul(max_2d_ub[1, i], max_partial_ub[i], 1.0)
                                T.reduce_max(max_2d_ub, global_max_ub, 0)
                                
                                sum_2d_ub = T.alloc_ub((2, n_num), calc_dtype)
                                for i in range(n_num):
                                    T.tile.mul(sum_2d_ub[0, i], sum_partial_ub[i], 1.0)
                                    T.tile.mul(sum_2d_ub[1, i], sum_partial_ub[i], 1.0)
                                T.reduce_sum(sum_2d_ub, global_sum_ub, 0)
                                
                                T.copy(global_max_ub, SoftmaxIMax[bz, s1_id])
                                T.copy(global_sum_ub, SoftmaxISum[bz, s1_id])
                                
                                # Compute normalized SoftmaxI
                                rsy_chunk_ub = T.alloc_ub(n_block, calc_dtype)
                                sm_i_chunk_ub = T.alloc_ub(n_block, calc_dtype)
                                
                                for n_idx_k in T.serial(n_num):
                                    T.copy(ReduceSumY[bz, s1_id, n_idx_k * n_block:(n_idx_k + 1) * n_block], rsy_chunk_ub)
                                    T.tile.sub(rsy_chunk_ub, rsy_chunk_ub, global_max_ub[0])
                                    T.tile.exp(rsy_chunk_ub, rsy_chunk_ub)
                                    T.tile.div(rsy_chunk_ub, rsy_chunk_ub, global_sum_ub[0])
                                    T.copy(rsy_chunk_ub, SoftmaxI[bz, s1_id, n_idx_k * n_block:(n_idx_k + 1) * n_block])
                            
                            # Compute dWeight, reluGrad, Loss (per K block)
                            bmm2_ub2 = T.alloc_ub((Nidx1, n_block), calc_dtype)
                            diff_ub = T.alloc_ub(n_block, calc_dtype)
                            softmax_i_chunk_ub = T.alloc_ub(n_block, calc_dtype)
                            p_reduced_chunk_ub = T.alloc_ub(n_block, calc_dtype)
                            dw_accum_ub = T.alloc_ub(vector_baseg, calc_dtype)
                            dw_chunk_ub = T.alloc_ub((vector_baseg, n_block), calc_dtype)
                            dw_reduce_ub = T.alloc_ub(vector_baseg, calc_dtype)
                            rg_ub = T.alloc_ub((vector_baseg, n_block), calc_dtype)
                            rg_half_ub = T.alloc_ub((vector_baseg, n_block), input_dtype)
                            mask_ub = T.alloc_ub(n_block, calc_dtype)
                            ones_ub = T.alloc_ub(n_block, calc_dtype)
                            weight_ub2 = T.alloc_ub(vector_baseg, calc_dtype)
                            weight_half_ub2 = T.alloc_ub(vector_baseg, input_dtype)
                            eps_ub = T.alloc_ub(n_block, calc_dtype)
                            loss_chunk_ub = T.alloc_ub(n_block, calc_dtype)
                            loss_log_p_ub = T.alloc_ub(n_block, calc_dtype)
                            loss_log_si_ub = T.alloc_ub(n_block, calc_dtype)
                            loss_val_ub = T.alloc_ub(n_block, calc_dtype)
                            
                            for s1_id in T.serial(s1_start, s1_end):
                                T.tile.fill(ones_ub, 1.0)
                                T.tile.fill(eps_ub, 1e-8)
                                
                                for n_idx_k in T.serial(n_num):
                                    T.copy(bmm2Res[bz, s1_id, 0:Nidx1, n_idx_k * n_block:(n_idx_k + 1) * n_block], bmm2_ub2)
                                    T.copy(SoftmaxI[bz, s1_id, n_idx_k * n_block:(n_idx_k + 1) * n_block], softmax_i_chunk_ub)
                                    T.copy(PReduced[bz, s1_id, n_idx_k * n_block:(n_idx_k + 1) * n_block], p_reduced_chunk_ub)
                                    T.tile.sub(diff_ub, softmax_i_chunk_ub, p_reduced_chunk_ub)
                                    
                                    for g_id in range(g_num):
                                        T.copy(Weights[bz, s1_id, g_id * vector_baseg:(g_id + 1) * vector_baseg], weight_half_ub2)
                                        T.tile.cast(weight_ub2, weight_half_ub2, "CAST_NONE", vector_baseg)
                                        T.tile.fill(dw_accum_ub, 0)
                                        
                                        for i in range(vector_baseg):
                                            g_row = g_id * vector_baseg + i
                                            T.tile.mul(dw_chunk_ub[i, :], bmm2_ub2[g_row, :], diff_ub)
                                            T.tile.mul(rg_ub[i, :], diff_ub, weight_ub2[i])
                                            T.tile.mul(mask_ub, bmm2_ub2[g_row, :], 10000.0)
                                            T.tile.mul(mask_ub, mask_ub, -1.0)
                                            T.tile.exp(mask_ub, mask_ub)
                                            T.tile.sub(mask_ub, ones_ub, mask_ub)
                                            T.tile.mul(rg_ub[i, :], rg_ub[i, :], mask_ub)
                                        
                                        T.reduce_sum(dw_chunk_ub, dw_reduce_ub, 1)
                                        T.tile.add(dw_accum_ub, dw_accum_ub, dw_reduce_ub)
                                        
                                        T.copy(dw_accum_ub, dWeight_partial[bz, s1_id, n_idx_k, g_id * vector_baseg:(g_id + 1) * vector_baseg])
                                        
                                        T.tile.cast(rg_half_ub, rg_ub, "CAST_ROUND", vector_baseg * n_block)
                                        for i in range(vector_baseg):
                                            g_row = g_id * vector_baseg + i
                                            T.copy(rg_half_ub[i, :], reluGradRes[bz, s1_id, g_row, n_idx_k * n_block:(n_idx_k + 1) * n_block])
                                    
                                    # Partial Loss
                                    T.copy(PReduced[bz, s1_id, n_idx_k * n_block:(n_idx_k + 1) * n_block], loss_chunk_ub)
                                    T.tile.max(loss_chunk_ub, loss_chunk_ub, eps_ub)
                                    T.tile.ln(loss_log_p_ub, loss_chunk_ub)
                                    
                                    T.copy(SoftmaxI[bz, s1_id, n_idx_k * n_block:(n_idx_k + 1) * n_block], loss_chunk_ub)
                                    T.tile.max(loss_chunk_ub, loss_chunk_ub, eps_ub)
                                    T.tile.ln(loss_log_si_ub, loss_chunk_ub)
                                    
                                    T.copy(PReduced[bz, s1_id, n_idx_k * n_block:(n_idx_k + 1) * n_block], loss_chunk_ub)
                                    T.tile.sub(loss_val_ub, loss_log_p_ub, loss_log_si_ub)
                                    T.tile.mul(loss_val_ub, loss_chunk_ub, loss_val_ub)
                                    
                                    loss_2d_ub = T.alloc_ub((16, n_block), calc_dtype)
                                    for i in range(16):
                                        T.tile.mul(loss_2d_ub[i, :], loss_val_ub, 1.0)
                                    loss_sum_ub = T.alloc_ub(16, calc_dtype)
                                    T.reduce_sum(loss_2d_ub, loss_sum_ub, 1)
                                    
                                    T.copy(loss_sum_ub, LossOut[bz, s1_id, n_idx_k])
                    
                    # ============================================================
                    # Stage C2: Matmul4 → dQueryIndex
                    # ============================================================
                    if pipeline_iter % 6 == 4 or (pipeline_iter >= 6 and (pipeline_iter - cid * 6) % 6 == 4):
                        with T.Scope("C"):
                            RG_L1 = T.alloc_L1((Nidx1, n_block), input_dtype)
                            GKIT_L1 = T.alloc_L1((k_block, n_block), input_dtype)
                            DQ_L0C = T.alloc_L0C((Nidx1, k_block), calc_dtype)
                            
                            for s1_row in T.serial(s1_start, s1_end):
                                for d_idx in T.serial(d_num_query_index):
                                    for n_idx_iter in T.serial(n_num):
                                        T.copy(reluGradRes[bz, s1_row, 0:Nidx1, n_idx_iter * n_block:(n_idx_iter + 1) * n_block], RG_L1)
                                        T.copy(GatherKeyIndexT[bz, s1_row, d_idx * k_block:(d_idx + 1) * k_block, n_idx_iter * n_block:(n_idx_iter + 1) * n_block], GKIT_L1)
                                        T.gemm_v0(RG_L1, GKIT_L1, DQ_L0C, transpose_B=True, init=(n_idx_iter == 0))
                                    T.copy(DQ_L0C, dQueryIndex_ws[bz, s1_row, 0:Nidx1, d_idx * k_block:(d_idx + 1) * k_block])
                    
                    # ============================================================
                    # Stage V3: Cast dQueryIndex FP32 → FP16
                    # ============================================================
                    if pipeline_iter % 6 == 5 or (pipeline_iter >= 6 and (pipeline_iter - cid * 6) % 6 == 5):
                        with T.Scope("V"):
                            dq_fp32_ub = T.alloc_ub((Nidx1, k_block), calc_dtype)
                            dq_fp16_ub = T.alloc_ub((Nidx1, k_block), input_dtype)
                            
                            for s1_row in T.serial(s1_start, s1_end):
                                for d_idx in T.serial(d_num_query_index):
                                    T.copy(dQueryIndex_ws[bz, s1_row, 0:Nidx1, d_idx * k_block:(d_idx + 1) * k_block], dq_fp32_ub)
                                    T.tile.cast(dq_fp16_ub, dq_fp32_ub, "CAST_ROUND", Nidx1 * k_block)
                                    T.copy(dq_fp16_ub, dQueryIndex[bz, s1_row, 0:Nidx1, d_idx * k_block:(d_idx + 1) * k_block])

    return main


def _preprocess_gather(key, key_index, sparse_indices, B, S1, S2, N2, Nidx2, K, DQuery, DQueryIndex):
    device = key.device
    dtype = key.dtype

    sparse_idx_3d = sparse_indices[:, :, 0, :]
    valid_mask = (sparse_idx_3d >= 0).unsqueeze(-1).to(dtype)
    sparse_idx_safe = sparse_idx_3d.clamp(min=0)

    key_2d = key[:, :, 0, :]
    key_index_2d = key_index[:, :, 0, :]

    idx_k = sparse_idx_safe.reshape(B, S1 * K).unsqueeze(-1).expand(B, S1 * K, DQuery)
    gather_key = torch.gather(key_2d, 1, idx_k).reshape(B, S1, K, DQuery).contiguous()

    idx_ki = sparse_idx_safe.reshape(B, S1 * K).unsqueeze(-1).expand(B, S1 * K, DQueryIndex)
    gather_key_index = torch.gather(key_index_2d, 1, idx_ki).reshape(B, S1, K, DQueryIndex).contiguous()

    gather_key.mul_(valid_mask)
    gather_key_index.mul_(valid_mask)

    return gather_key, gather_key_index


def _compute_d_key_index(relu_grad, query_index, sparse_indices, B, S1, S2, Nidx1, Nidx2, K, DQueryIndex, device, dtype):
    relu_grad_fp32 = relu_grad.float()
    query_index_fp32 = query_index.float()

    relu_grad_t = relu_grad_fp32.permute(0, 1, 3, 2)
    scatter_add = torch.bmm(
        relu_grad_t.reshape(B * S1, K, Nidx1),
        query_index_fp32.reshape(B * S1, Nidx1, DQueryIndex),
    ).reshape(B, S1, K, DQueryIndex)

    d_key_index = torch.zeros(B, S2, Nidx2, DQueryIndex, dtype=torch.float32, device=device)
    sparse_idx = sparse_indices[:, :, 0, :].clamp(min=0)
    valid_mask = (sparse_indices[:, :, 0, :] >= 0).unsqueeze(-1).to(torch.float32)

    scatter_add_masked = scatter_add * valid_mask
    idx_expanded = sparse_idx.unsqueeze(-1).expand(B, S1, K, DQueryIndex)
    d_key_index_2d = d_key_index[:, :, 0, :]
    d_key_index_2d.scatter_add_(1, idx_expanded.reshape(B, S1 * K, DQueryIndex)[:, :S2, :],
                                 scatter_add_masked.reshape(B, S1 * K, DQueryIndex)[:, :S2, :])

    return d_key_index.to(dtype)


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

    B = tiling.B
    S1 = tiling.S1
    S2 = tiling.S2
    N1 = tiling.N1
    N2 = tiling.N2
    G = tiling.G
    Nidx1 = tiling.Nidx1
    K = tiling.K
    DQuery = tiling.DQuery
    DQueryIndex = tiling.DQueryIndex

    input_dtype = query.dtype
    calc_dtype_str = "float"
    input_dtype_str = "float16" if input_dtype == torch.float16 else "bfloat16"
    weights_orig_dtype = weights.dtype

    n_num = K // BLOCK_N
    core_num = min(B * ceil_div(S1, BLOCK_M), MAX_CORE_NUM)

    S2_pad = (S2 + VECTOR_BASEN - 1) // VECTOR_BASEN * VECTOR_BASEN
    if S2_pad > S2:
        key = torch.nn.functional.pad(key, (0, 0, 0, 0, 0, S2_pad - S2, 0, 0))
        key_index = torch.nn.functional.pad(key_index, (0, 0, 0, 0, 0, S2_pad - S2, 0, 0))

    if weights.dtype == torch.float32:
        weights = weights.to(input_dtype)

    gather_key, gather_key_index = _preprocess_gather(
        key, key_index, sparse_indices,
        B, S1, S2_pad, N2, tiling.Nidx2, K, DQuery, DQueryIndex,
    )

    gather_key_index_t = gather_key_index.permute(0, 1, 3, 2).contiguous()

    d_rope_orig = query_rope.shape[-1] if query_rope is not None else 0
    if query_rope is not None and key_rope is not None:
        _sparse_idx_rope = sparse_indices[:, :, 0, :].clamp(min=0)
        key_rope_2d = key_rope[:, :, 0, :]
        idx_rope = _sparse_idx_rope.reshape(B, S1 * K).unsqueeze(-1).expand(B, S1 * K, d_rope_orig)
        gather_key_rope = torch.gather(key_rope_2d, 1, idx_rope).reshape(B, S1, K, d_rope_orig).contiguous()
        _valid_mask_rope = (sparse_indices[:, :, 0, :] >= 0).unsqueeze(-1).to(query.dtype)
        gather_key_rope.mul_(_valid_mask_rope)
    else:
        gather_key_rope = torch.zeros(B, S1, K, d_rope_orig, dtype=query.dtype, device=query.device)

    d_rope = ((d_rope_orig + BLOCK_K - 1) // BLOCK_K) * BLOCK_K if d_rope_orig > 0 else 0
    if d_rope > d_rope_orig and d_rope_orig > 0:
        pad_len = d_rope - d_rope_orig
        query_rope = torch.nn.functional.pad(query_rope, (0, pad_len))
        gather_key_rope = torch.nn.functional.pad(gather_key_rope, (0, pad_len))

    cache_key = (B, S1, S2_pad, N1, N2, G, tiling.Nidx1, tiling.Nidx2, DQuery, DQueryIndex, K,
                 d_rope, input_dtype_str, calc_dtype_str, scale_value, core_num)

    if cache_key in _kernel_cache:
        func = _kernel_cache[cache_key]
    else:
        func = tilelang_grad_kl_loss_pipeline_kernel(
            B, S1, S2_pad, N1, N2, G, tiling.Nidx1, tiling.Nidx2, DQuery, DQueryIndex, K,
            d_rope, input_dtype_str, calc_dtype_str, scale_value, core_num,
        )
        _kernel_cache[cache_key] = func

    bmm1_res_ws = torch.zeros(B, S1, N1, K, dtype=torch.float32, device=query.device)
    bmm2_res_ws = torch.zeros(B, S1, Nidx1, K, dtype=torch.float32, device=query.device)
    p_reduced_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=query.device)
    reduce_sum_y_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=query.device)
    softmax_i_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=query.device)
    d_query_index_ws = torch.zeros(B, S1, Nidx1, DQueryIndex, dtype=torch.float32, device=query.device)
    d_query_index_out = torch.zeros(B, S1, Nidx1, DQueryIndex, dtype=input_dtype, device=query.device)
    relu_grad_out = torch.zeros(B, S1, Nidx1, K, dtype=input_dtype, device=query.device)
    loss_out = torch.zeros(B, S1, n_num, dtype=torch.float32, device=query.device)
    d_weight_partial = torch.zeros(B, S1, n_num, Nidx1, dtype=torch.float32, device=query.device)
    softmax_i_max = torch.zeros(B, S1, dtype=torch.float32, device=query.device)
    softmax_i_sum = torch.zeros(B, S1, dtype=torch.float32, device=query.device)
    softmax_i_max_partial = torch.zeros(B, S1, n_num, dtype=torch.float32, device=query.device)
    softmax_i_sum_partial = torch.zeros(B, S1, n_num, dtype=torch.float32, device=query.device)

    results = func(
        query, query_rope, gather_key, gather_key_rope,
        query_index, gather_key_index, gather_key_index_t,
        weights, softmax_max, softmax_sum,
        bmm1_res_ws, bmm2_res_ws,
        p_reduced_ws, reduce_sum_y_ws, softmax_i_ws,
        d_query_index_out, d_weight_partial, relu_grad_out, loss_out,
        d_query_index_ws,
        softmax_i_max, softmax_i_sum, softmax_i_max_partial, softmax_i_sum_partial,
    )

    d_query_index = results[0]
    relu_grad = results[2]
    loss_out = results[3]

    d_weight = d_weight_partial.sum(dim=2)
    if weights_orig_dtype == torch.float32:
        d_weight = d_weight.float()
    else:
        d_weight = d_weight.to(input_dtype)

    kl_loss = loss_out.sum()

    d_key_index = _compute_d_key_index(
        relu_grad, query_index, sparse_indices,
        B, S1, S2, Nidx1, tiling.Nidx2, K, DQueryIndex,
        query.device, input_dtype,
    )

    loss_tensor = kl_loss.reshape(1).to(torch.float32).npu()

    return d_query_index, d_key_index, d_weight, loss_tensor