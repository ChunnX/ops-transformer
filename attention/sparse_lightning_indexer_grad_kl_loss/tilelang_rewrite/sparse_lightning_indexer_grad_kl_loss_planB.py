"""Plan B: Triple-kernel with Python bridge.

Architecture:
  Kernel 1 (block_num = B * s1_tile_num * n_num, K-parallel):
    C1(MM1+MM2, T.pipelined) -> V1(P_reduced chunk, ReduceSumY chunk, vectorized)
  Kernel 2 (block_num = B * s1_tile_num * n_num, K-parallel):
    V1(SoftmaxI + diff + dWeight + reluGrad + Loss, T.pipelined + vectorized)
  Kernel 3 (block_num = B * s1_tile_num * n_num, K-parallel):
    C(MM6: dQueryIndex = reluGrad @ GKIT, gemm + atomic_add) -> V(barrier)
  Python:
    dWeight sum reduction, Loss sum, dKeyIndex scatter_add

Optimizations applied:
  - T.pipelined on C1 d_idx loop (MM1 Double Buffer)
  - T.pipelined on V1 nidx_c loop (overlap copy with compute)
  - MM6 moved into Kernel3 (C→V, eliminates Python bmm)
  - Atomic_add for cross-K-block accumulation (L0C→Global, proven pattern)
"""

import math
import time
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import tilelang
import tilelang.language as T

tilelang.disable_cache()

# ==============================================================================
# Constants (can be overridden before calling the entry function)
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
_kernel2_cache = {}
_kernel3_cache = {}
_workspace_cache = {}


def set_block_config(m=None, n=None, k=None, n1=None, nidx1=None):
    """Override BLOCK constants. Call before sparse_lightning_indexer_grad_kl_loss()."""
    global BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_N1, BLOCK_NIDX1
    if m is not None:
        BLOCK_M = m
    if n is not None:
        BLOCK_N = n
    if k is not None:
        BLOCK_K = k
    if n1 is not None:
        BLOCK_N1 = n1
    if nidx1 is not None:
        BLOCK_NIDX1 = nidx1


def reset_block_config():
    """Reset BLOCK constants to defaults."""
    global BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_N1, BLOCK_NIDX1
    BLOCK_M = 16
    BLOCK_N = 128
    BLOCK_K = 128
    BLOCK_N1 = 32
    BLOCK_NIDX1 = 16


def _clear_caches():
    global _kernel_cache, _kernel2_cache, _kernel3_cache, _workspace_cache
    _kernel_cache.clear()
    _kernel2_cache.clear()
    _kernel3_cache.clear()
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
        """Kernel 1: K-parallel."""
        return self.B * self.s1_tile_num * self.n_num

    @property
    def block_num_k2(self) -> int:
        """Kernel 2: S1-parallel."""
        return self.B * self.s1_tile_num


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
        assert query.dim() == 3
        S1, N1, DQuery = query.shape
        _, N2, _ = key.shape
        _, Nidx1, DQueryIndex = query_index.shape
        _, Nidx2, _ = key_index.shape
        B = 1
    else:
        assert query.dim() == 4
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
# Kernel 1: K-parallel MM1+MM2+P_reduced+ReduceSumY
# ==============================================================================
@tilelang.jit(
    out_idx=[-3, -2, -1],
    pass_configs=pass_configs,
)
def planB_kernel1(
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
        with T.Kernel(B * s1_tile_num * n_num, is_npu=True) as (cid, vid):
            bz = cid // (s1_tile_num * n_num)
            r = cid % (s1_tile_num * n_num)
            m = r // n_num
            n_idx = r % n_num

            s1_start = m * s1_block
            s1_end = T.min(s1_start + s1_block, S1)
            k_start = n_idx * n_block
            k_end = k_start + n_block

            # C1: MM1 + MM2
            with T.Scope("C"):
                Q_L1 = T.alloc_L1((N1, k_block), input_dtype)
                GK_L1 = T.alloc_L1((n_block, k_block), input_dtype)
                C_L0C = T.alloc_L0C((N1, n_block), calc_dtype)

                QI_L1 = T.alloc_L1((Nidx1, k_block), input_dtype)
                GKI_L1 = T.alloc_L1((n_block, k_block), input_dtype)
                C2_L0C = T.alloc_L0C((Nidx1, n_block), calc_dtype)

                for s1_row in T.serial(s1_start, s1_end):
                    # MM1: T.pipelined enables Double Buffer (overlap copy with gemm)
                    for d_idx in T.Pipelined(d_num_total, num_stages=3):
                        if d_idx < d_num_query:
                            T.copy(Query[bz, s1_row, 0:N1, d_idx * k_block:(d_idx + 1) * k_block], Q_L1)
                        else:
                            rope_d = d_idx - d_num_query
                            T.copy(QueryRope[bz, s1_row, 0:N1, rope_d * k_block:(rope_d + 1) * k_block], Q_L1)
                        if d_idx < d_num_query:
                            T.copy(GatherKey[bz, s1_row, k_start:k_end, d_idx * k_block:(d_idx + 1) * k_block], GK_L1)
                        else:
                            rope_d = d_idx - d_num_query
                            T.copy(GatherKeyRope[bz, s1_row, k_start:k_end, rope_d * k_block:(rope_d + 1) * k_block], GK_L1)
                        T.gemm_v0(Q_L1, GK_L1, C_L0C, transpose_B=True, init=(d_idx == 0))
                    T.copy(C_L0C, bmm1Res[bz, s1_row, 0:N1, k_start:k_end])

                    # MM2: single k_idx iteration, no pipelining needed
                    for k_idx in T.serial(d_num_query_index):
                        T.copy(QueryIndex[bz, s1_row, 0:Nidx1, k_idx * k_block:(k_idx + 1) * k_block], QI_L1)
                        T.copy(GatherKeyIndex[bz, s1_row, k_start:k_end, k_idx * k_block:(k_idx + 1) * k_block], GKI_L1)
                        T.gemm_v0(QI_L1, GKI_L1, C2_L0C, transpose_B=True, init=(k_idx == 0))
                    T.copy(C2_L0C, bmm2Res[bz, s1_row, 0:Nidx1, k_start:k_end], enable_relu=True)

                T.set_cross_flag("FIX", 1)

            # V1: P_reduced + ReduceSumY
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
                    T.copy(SoftmaxMax[bz, 0, s1_id, 0:G], sm_max_ub)
                    T.copy(SoftmaxSum[bz, 0, s1_id, 0:G], sm_sum_ub)
                    T.barrier_all()
                    T.tile.abs(sm_sum_ub, sm_sum_ub)
                    T.tile.max(sm_sum_ub, sm_sum_ub, eps_ub)

                    T.copy(Weights[bz, s1_id, 0:Nidx1], weight_fp16_ub)
                    T.tile.cast(weight_ub, weight_fp16_ub, "CAST_NONE", Nidx1)
                    T.barrier_all()

                    # Pass 1: P_reduced (reduce_sum for row accumulation)
                    T.tile.fill(p_chunk_ub, 0.0)
                    for n1_c in T.serial(n1_num):
                        n1_s = n1_c * BLOCK_N1
                        T.copy(bmm1Res[bz, s1_id, n1_s:n1_s + BLOCK_N1, k_start:k_end], bmm1_ub)
                        T.barrier_all()
                        T.tile.mul(bmm1_ub, bmm1_ub, scale_value)
                        for g_i in T.serial(BLOCK_N1):
                            T.tile.sub(bmm1_ub[g_i, :], bmm1_ub[g_i, :], sm_max_ub[n1_s + g_i])
                        T.tile.exp(bmm1_ub, bmm1_ub)
                        for g_i in T.serial(BLOCK_N1):
                            T.tile.div(bmm1_ub[g_i, :], bmm1_ub[g_i, :], sm_sum_ub[n1_s + g_i])
                        T.reduce_sum(bmm1_ub, p_chunk_ub, dim=0, clear=False, real_shape=[BLOCK_N1, n_block])
                    T.tile.mul(p_chunk_ub, p_chunk_ub, 1.0 / N1)
                    T.copy(p_chunk_ub, PReduced[bz, s1_id, k_start:k_end])

                    # Pass 2: ReduceSumY (reduce_sum for row accumulation)
                    T.tile.fill(rsy_chunk_ub, 0.0)
                    for nidx_c in T.serial(nidx1_num):
                        nidx_s = nidx_c * BLOCK_NIDX1
                        T.copy(bmm2Res[bz, s1_id, nidx_s:nidx_s + BLOCK_NIDX1, k_start:k_end], bmm2_ub)
                        T.barrier_all()
                        for g_i in T.serial(BLOCK_NIDX1):
                            T.tile.mul(bmm2_ub[g_i, :], bmm2_ub[g_i, :], weight_ub[nidx_s + g_i])
                        T.reduce_sum(bmm2_ub, rsy_chunk_ub, dim=0, clear=False, real_shape=[BLOCK_NIDX1, n_block])
                    T.copy(rsy_chunk_ub, ReduceSumY[bz, s1_id, k_start:k_end])

    return main


# ==============================================================================
# Kernel 2: K-parallel dWeight/reluGrad/Loss (V1 only, partial outputs)
# ==============================================================================
@tilelang.jit(
    out_idx=[-3, -2, -1],
    pass_configs=pass_configs,
)
def planB_kernel2(
    B, S1, S2, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
    d_rope, input_dtype, calc_dtype, scale_value,
):
    s1_block = BLOCK_M
    n_block = BLOCK_N
    s1_tile_num = _ceil_div(S1, s1_block)
    n_num = K // n_block
    nidx1_num = Nidx1 // BLOCK_NIDX1

    @T.prim_func
    def main(
        bmm2Res: T.Tensor((B, S1, Nidx1, K), calc_dtype),
        Weights: T.Tensor((B, S1, Nidx1), input_dtype),
        PReduced: T.Tensor((B, S1, K), calc_dtype),
        ReduceSumY: T.Tensor((B, S1, K), calc_dtype),
        dWeightPartial: T.Tensor((B, S1, n_num, Nidx1), calc_dtype),
        reluGradRes: T.Tensor((B, S1, Nidx1, K), input_dtype),
        LossPartial: T.Tensor((B, S1, n_num), calc_dtype),
    ):
        # K-parallel: block_num = B * s1_tile_num * n_num
        with T.Kernel(B * s1_tile_num * n_num, is_npu=True) as (cid, vid):
            bz = cid // (s1_tile_num * n_num)
            r = cid % (s1_tile_num * n_num)
            m = r // n_num
            n_idx = r % n_num

            s1_start = m * s1_block
            s1_end = T.min(s1_start + s1_block, S1)
            k_start = n_idx * n_block
            k_end = k_start + n_block

            # V1 only: SoftmaxI + diff + dWeight + reluGrad + Loss
            with T.Scope("V"):
                bmm2_ub = T.alloc_shared((BLOCK_NIDX1, n_block), calc_dtype)
                weight_ub = T.alloc_shared(Nidx1, calc_dtype)
                weight_fp16_ub = T.alloc_shared(Nidx1, input_dtype)

                p_chunk_ub = T.alloc_shared(n_block, calc_dtype)
                si_chunk_ub = T.alloc_shared(n_block, calc_dtype)
                diff_ub = T.alloc_shared(n_block, calc_dtype)

                rg_chunk_ub = T.alloc_shared((BLOCK_NIDX1, n_block), calc_dtype)
                rg_fp16_ub = T.alloc_shared((BLOCK_NIDX1, n_block), input_dtype)
                dw_total_ub = T.alloc_shared(Nidx1, calc_dtype)
                dw_chunk_ub = T.alloc_shared(BLOCK_NIDX1, calc_dtype)
                dw_temp_ub = T.alloc_shared(BLOCK_NIDX1, calc_dtype)

                eps_ub = T.alloc_shared(n_block, calc_dtype)
                mask_2d_ub = T.alloc_shared((BLOCK_NIDX1, n_block), calc_dtype)

                loss_ub = T.alloc_shared(n_block, calc_dtype)
                loss_log_p_ub = T.alloc_shared(n_block, calc_dtype)
                loss_log_si_ub = T.alloc_shared(n_block, calc_dtype)
                rsy_tmp_ub = T.alloc_shared(1, calc_dtype)

                # Softmax workspace (each core redundantly computes global SoftmaxI)
                si_all_ub = T.alloc_shared(K, calc_dtype)
                si_max_ub = T.alloc_shared(1, calc_dtype)
                si_sum_ub = T.alloc_shared(1, calc_dtype)

                T.tile.fill(eps_ub, 1e-8)

                for s1_id in T.serial(s1_start, s1_end):
                    # Load weights
                    T.copy(Weights[bz, s1_id, 0:Nidx1], weight_fp16_ub)
                    T.tile.cast(weight_ub, weight_fp16_ub, "CAST_NONE", Nidx1)
                    T.barrier_all()

                    # ---- Compute global SoftmaxI from ReduceSumY (redundant per core) ----
                    T.copy(ReduceSumY[bz, s1_id, 0:K], si_all_ub)
                    T.barrier_all()
                    T.reduce_max(si_all_ub, si_max_ub, dim=-1, clear=True, real_shape=[1, K])
                    T.tile.sub(si_all_ub, si_all_ub, si_max_ub[0])
                    T.tile.exp(si_all_ub, si_all_ub)
                    T.reduce_sum(si_all_ub, si_sum_ub, dim=-1, clear=True, real_shape=[1, K])
                    T.tile.max(si_sum_ub, si_sum_ub, 1e-8)
                    T.tile.div(si_all_ub, si_all_ub, si_sum_ub[0])
                    T.barrier_all()

                    # Load P_reduced for this K-block and compute diff
                    T.copy(PReduced[bz, s1_id, k_start:k_end], p_chunk_ub)
                    T.copy(si_all_ub[k_start:k_end], si_chunk_ub)
                    T.barrier_all()
                    T.tile.sub(diff_ub, si_chunk_ub, p_chunk_ub)

                    # ---- Pass 4: dWeight partial + reluGrad for this K-block ----
                    T.tile.fill(dw_total_ub, 0.0)

                    # T.Pipelined to overlap bmm2Res copy with dWeight/reluGrad compute
                    for nidx_c in T.Pipelined(nidx1_num, num_stages=3):
                        nidx_s = nidx_c * BLOCK_NIDX1
                        T.copy(
                            bmm2Res[bz, s1_id, nidx_s:nidx_s + BLOCK_NIDX1, k_start:k_end],
                            bmm2_ub,
                        )

                        # dWeight: per-row mul, then reduce_sum
                        for g_i in T.serial(BLOCK_NIDX1):
                            T.tile.mul(rg_chunk_ub[g_i, :], diff_ub, bmm2_ub[g_i, :])
                        T.reduce_sum(rg_chunk_ub, dw_chunk_ub, dim=-1, clear=True, real_shape=[BLOCK_NIDX1, n_block])
                        T.copy(dw_total_ub[nidx_s:nidx_s + BLOCK_NIDX1], dw_temp_ub)
                        T.tile.add(dw_temp_ub, dw_temp_ub, dw_chunk_ub)
                        T.copy(dw_temp_ub, dw_total_ub[nidx_s:nidx_s + BLOCK_NIDX1])

                        # reluGrad: per-row mul, then 2D compare/select
                        for g_i in T.serial(BLOCK_NIDX1):
                            g_abs = nidx_s + g_i
                            T.tile.mul(rg_chunk_ub[g_i, :], diff_ub, weight_ub[g_abs])
                        T.tile.compare(mask_2d_ub, bmm2_ub, 0.0, "GT")
                        T.tile.select(rg_chunk_ub, mask_2d_ub, rg_chunk_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                        # Cast and store reluGrad (2D block copy)
                        T.tile.cast(rg_fp16_ub, rg_chunk_ub, "CAST_ROUND", BLOCK_NIDX1 * n_block)
                        T.copy(
                            rg_fp16_ub,
                            reluGradRes[bz, s1_id, nidx_s:nidx_s + BLOCK_NIDX1, k_start:k_end],
                        )

                    # Store dWeight partial (FP32, summed in Python)
                    T.copy(dw_total_ub, dWeightPartial[bz, s1_id, n_idx, 0:Nidx1])

                    # ---- Loss partial for this K-block ----
                    T.tile.max(p_chunk_ub, p_chunk_ub, eps_ub)
                    T.tile.ln(loss_log_p_ub, p_chunk_ub)
                    T.tile.max(si_chunk_ub, si_chunk_ub, eps_ub)
                    T.tile.ln(loss_log_si_ub, si_chunk_ub)
                    T.tile.sub(loss_ub, loss_log_p_ub, loss_log_si_ub)
                    T.tile.mul(loss_ub, p_chunk_ub, loss_ub)
                    T.reduce_sum(loss_ub, rsy_tmp_ub, dim=-1, clear=True, real_shape=[1, n_block])
                    T.barrier_all()
                    T.copy(rsy_tmp_ub, LossPartial[bz, s1_id, n_idx:n_idx + 1])

    return main


# ==============================================================================
# Kernel 3: MM6 — dQueryIndex = reluGrad @ GKI (C→V, K-parallel, atomic_add)
# ==============================================================================
@tilelang.jit(
    out_idx=[-1],
    pass_configs=pass_configs,
)
def planB_kernel3(
    B, S1, Nidx1, DQueryIndex, K,
    input_dtype, calc_dtype,
):
    s1_block = BLOCK_M
    n_block = BLOCK_N
    s1_tile_num = _ceil_div(S1, s1_block)
    n_num = K // n_block
    nidx1_num = Nidx1 // BLOCK_NIDX1
    dq_num = DQueryIndex // BLOCK_K

    @T.prim_func
    def main(
        reluGradRes: T.Tensor((B, S1, Nidx1, K), input_dtype),
        GatherKeyIndexT: T.Tensor((B, S1, DQueryIndex, K), input_dtype),
        dQueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), calc_dtype),
    ):
        with T.Kernel(B * s1_tile_num * n_num, is_npu=True) as (cid, vid):
            bz = cid // (s1_tile_num * n_num)
            r = cid % (s1_tile_num * n_num)
            m = r // n_num
            n_idx = r % n_num

            s1_start = m * s1_block
            s1_end = T.min(s1_start + s1_block, S1)
            k_start = n_idx * n_block
            k_end = k_start + n_block

            # C: MM6 gemm + atomic_add (matching 0622.py C2 scope pattern)
            with T.Scope("C"):
                RG_L1 = T.alloc_L1((BLOCK_NIDX1, n_block), input_dtype)
                GKIT_L1 = T.alloc_L1((BLOCK_K, n_block), input_dtype)
                DQ_L0C = T.alloc_L0C((BLOCK_NIDX1, BLOCK_K), calc_dtype)

                for s1_row in T.serial(s1_start, s1_end):
                    for dq_c in T.serial(dq_num):
                        dq_s = dq_c * BLOCK_K
                        T.copy(GatherKeyIndexT[bz, s1_row, dq_s:dq_s + BLOCK_K, k_start:k_end], GKIT_L1)
                        for nidx_c in T.serial(nidx1_num):
                            nidx_s = nidx_c * BLOCK_NIDX1
                            T.copy(reluGradRes[bz, s1_row, nidx_s:nidx_s + BLOCK_NIDX1, k_start:k_end], RG_L1)
                            T.gemm_v0(RG_L1, GKIT_L1, DQ_L0C, transpose_B=True, init=True)
                            T.tile.atomic_add(
                                dQueryIndex[bz, s1_row, nidx_s:nidx_s + BLOCK_NIDX1, dq_s:dq_s + BLOCK_K],
                                DQ_L0C,
                            )

                T.set_cross_flag("FIX", 1)

            # V: minimal — wait for C, then barrier (required for CV-kernel validity)
            with T.Scope("V"):
                T.wait_cross_flag(1)
                T.barrier_all()

    return main


# ==============================================================================
# Python Preprocessing: Gather
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

    # Fuse key, key_index, key_rope into a single gather
    key_2d = key[:, :, 0, :]
    ki_2d = key_index[:, :, 0, :]
    if key_rope is not None and d_rope_orig > 0:
        kr_2d = key_rope[:, :, 0, :]
    else:
        kr_2d = torch.zeros(B, key.shape[1], 0, dtype=dtype, device=device)
    key_all = torch.cat([key_2d, ki_2d, kr_2d], dim=-1)
    total_dim = DQuery + DQueryIndex + d_rope_orig

    idx_all = sparse_idx_safe.reshape(B, S1 * K).unsqueeze(-1).expand(B, S1 * K, total_dim)
    gather_all = torch.gather(key_all, 1, idx_all).reshape(B, S1, K, total_dim).contiguous()
    gather_all.mul_(valid_mask)

    gather_key = gather_all[..., :DQuery].contiguous()
    gather_key_index = gather_all[..., DQuery:DQuery + DQueryIndex].contiguous()
    if d_rope_orig > 0:
        gather_key_rope = gather_all[..., DQuery + DQueryIndex:].contiguous()
    else:
        gather_key_rope = torch.zeros(B, S1, K, d_rope_orig, dtype=dtype, device=device)

    return gather_key, gather_key_index, gather_key_rope


# ==============================================================================
# Python Bridge: SoftmaxI + diff
# ==============================================================================
def _python_bridge(
    reduce_sum_y: torch.Tensor,  # (B, S1, K) FP32
    p_reduced: torch.Tensor,     # (B, S1, K) FP32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute global SoftmaxI and diff from ReduceSumY and PReduced."""
    softmax_i = torch.softmax(reduce_sum_y.float(), dim=-1)  # (B, S1, K)
    diff = softmax_i - p_reduced.float()  # (B, S1, K)
    return softmax_i, diff


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

    relu_grad_t = relu_grad_fp32.permute(0, 1, 3, 2)
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
# Main Entry
# ==============================================================================
# Profiling storage (module-level, read by test script after call)
_profile_result: Dict[str, float] = {}


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
    profile: bool = False,
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
    input_dtype = query.dtype
    weights_orig_dtype = weights.dtype
    device = query.device

    input_dtype_str = "float16" if input_dtype == torch.float16 else "bfloat16"
    calc_dtype_str = "float"

    # Pad S2
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

    # Pad RoPE
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

    if profile:
        torch.npu.synchronize()
        _pt = {"gather": time.perf_counter()}
    else:
        _pt = {}

    # ---- Kernel 1 compilation cache ----
    cache_key = (B, S1, S2_pad, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
                 d_rope, input_dtype_str, calc_dtype_str, scale_value,
                 BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_N1, BLOCK_NIDX1)
    if cache_key in _kernel_cache:
        func1 = _kernel_cache[cache_key]
    else:
        func1 = planB_kernel1(
            B, S1, S2_pad, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
            d_rope, input_dtype_str, calc_dtype_str, scale_value,
        )
        _kernel_cache[cache_key] = func1

    # ---- Kernel 2 compilation cache ----
    k2_key = cache_key + ("k2",)
    if k2_key in _kernel2_cache:
        func2 = _kernel2_cache[k2_key]
    else:
        func2 = planB_kernel2(
            B, S1, S2_pad, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
            d_rope, input_dtype_str, calc_dtype_str, scale_value,
        )
        _kernel2_cache[k2_key] = func2

    # ---- Kernel 3 compilation cache ----
    k3_key = (B, S1, Nidx1, DQueryIndex, K, input_dtype_str, calc_dtype_str)
    if k3_key in _kernel3_cache:
        func3 = _kernel3_cache[k3_key]
    else:
        func3 = planB_kernel3(
            B, S1, Nidx1, DQueryIndex, K,
            input_dtype_str, calc_dtype_str,
        )
        _kernel3_cache[k3_key] = func3

    # ---- Workspace ----
    ws_key = (B, S1, N1, Nidx1, K, DQueryIndex, input_dtype, device, BLOCK_N)
    n_num = K // BLOCK_N
    if ws_key in _workspace_cache:
        ws = _workspace_cache[ws_key]
    else:
        bmm1_res_ws = torch.zeros(B, S1, N1, K, dtype=torch.float32, device=device)
        bmm2_res_ws = torch.zeros(B, S1, Nidx1, K, dtype=torch.float32, device=device)
        p_reduced_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=device)
        reduce_sum_y_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=device)
        d_weight_partial = torch.zeros(B, S1, n_num, Nidx1, dtype=torch.float32, device=device)
        loss_partial = torch.zeros(B, S1, n_num, dtype=torch.float32, device=device)
        relu_grad_ws = torch.zeros(B, S1, Nidx1, K, dtype=input_dtype, device=device)
        d_query_index_ws = torch.zeros(B, S1, Nidx1, DQueryIndex, dtype=torch.float32, device=device)
        ws = (bmm1_res_ws, bmm2_res_ws, p_reduced_ws, reduce_sum_y_ws,
              d_weight_partial, loss_partial, relu_grad_ws, d_query_index_ws)
        _workspace_cache[ws_key] = ws

    (bmm1_res_ws, bmm2_res_ws, p_reduced_ws, reduce_sum_y_ws,
     d_weight_partial, loss_partial, relu_grad_ws, d_query_index_ws) = ws

    # ---- Mark compilation done (for profile) ----
    if profile:
        torch.npu.synchronize()
        _pt["compile_done"] = time.perf_counter()

    # ---- Launch Kernel 1 ----
    bmm2_res_ws, p_reduced_ws, reduce_sum_y_ws = func1(
        query, query_rope, gather_key, gather_key_rope,
        query_index, gather_key_index,
        weights, softmax_max, softmax_sum,
        bmm1_res_ws, bmm2_res_ws,
        p_reduced_ws, reduce_sum_y_ws,
    )

    if profile:
        torch.npu.synchronize()
        _pt["kernel1"] = time.perf_counter()

    # ---- Launch Kernel 2 (K-parallel, V1-only, SoftmaxI fused in kernel) ----
    d_weight_partial, relu_grad, loss_partial = func2(
        bmm2_res_ws, weights, p_reduced_ws, reduce_sum_y_ws,
        d_weight_partial,
        relu_grad_ws,
        loss_partial,
    )

    if profile:
        torch.npu.synchronize()
        _pt["kernel2"] = time.perf_counter()

    # ---- Launch Kernel 3 (MM6: dQueryIndex = reluGrad @ GKI, C→V, atomic_add) ----
    gather_key_index_t = gather_key_index.transpose(-1, -2).contiguous()
    d_query_index_ws.zero_()
    ret = func3(relu_grad, gather_key_index_t, d_query_index_ws)
    if isinstance(ret, tuple):
        ret = ret[0]
    if ret is not None and ret is not d_query_index_ws:
        # out_idx allocated a new tensor; zero and re-run for correct init
        d_query_index_ws = ret
        d_query_index_ws.zero_()
        ret = func3(relu_grad, gather_key_index_t, d_query_index_ws)
        if isinstance(ret, tuple):
            ret = ret[0]
        if ret is not None:
            d_query_index_ws = ret
        ws = list(ws)
        ws[7] = d_query_index_ws
        _workspace_cache[ws_key] = tuple(ws)
    elif ret is not None:
        d_query_index_ws = ret

    if profile:
        torch.npu.synchronize()
        _pt["kernel3"] = time.perf_counter()

    # ---- Python Postprocessing (dWeight sum, Loss sum, cast, dKeyIndex scatter_add) ----
    d_weight = d_weight_partial.sum(dim=2).to(weights_orig_dtype if weights_orig_dtype == torch.float32 else input_dtype)
    d_query_index = d_query_index_ws.to(input_dtype)
    kl_loss = loss_partial.sum().reshape(1).to(torch.float32)

    # ---- Postprocessing: dKeyIndex via scatter_add ----
    d_key_index = _compute_d_key_index(
        relu_grad, query_index, sparse_indices,
        B, S1, S2, Nidx1, Nidx2, K, DQueryIndex,
        device, input_dtype,
    )

    if S2_pad > S2:
        d_key_index = d_key_index[:, :S2, :, :]

    if profile:
        torch.npu.synchronize()
        _pt["python_post"] = time.perf_counter()
        _profile_result.clear()
        _profile_result.update({
            "gather_compile": (_pt["compile_done"] - _pt["gather"]) * 1000,
            "kernel1": (_pt["kernel1"] - _pt["compile_done"]) * 1000,
            "kernel2": (_pt["kernel2"] - _pt["kernel1"]) * 1000,
            "kernel3": (_pt["kernel3"] - _pt["kernel2"]) * 1000,
            "python": (_pt["python_post"] - _pt["kernel3"]) * 1000,
            "total": (_pt["python_post"] - _pt["gather"]) * 1000,
        })

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
