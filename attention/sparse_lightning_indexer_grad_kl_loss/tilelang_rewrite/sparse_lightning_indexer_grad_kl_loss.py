"""TileLang-Ascend rewrite of sparse_lightning_indexer_grad_kl_loss.

Derived from the original AscendC kernel's MainProcess pipeline:
  V0(gather) -> C1(MM1+MM2) -> V1(P/SY/SoftmaxI/dW/reluGrad/Loss)
  -> C2(MM6:dQueryIndex) -> V2(Cast) -> post-loop(dKeyIndex ScatterAdd, Loss reduce)

Since TileLang lacks native gather/scatter-by-indices:
  - V0 (gather) -> Python torch.gather preprocessing
  - MM5 (scatterAdd dKeyIndex) -> Python scatter_add_ postprocessing
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
# Constants (matching original AscendC kernel)
# ==============================================================================
BLOCK_M = 16       # S1 rows per core
BLOCK_N = 128      # K-axis matmul tile (matches N_SPLIT_SIZE in original)
BLOCK_K = 128      # D-axis matmul tile (matches K_SPLIT_SIZE in original)
BLOCK_N1 = 32      # N1 chunk size for UB streaming
BLOCK_NIDX1 = 16   # Nidx1 chunk size for UB streaming

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
    def block_num(self) -> int:
        """Each core handles BLOCK_M S1 rows with ALL K."""
        return self.B * self.s1_tile_num

    @property
    def n_num(self) -> int:
        """Number of K-chunks (BLOCK_N=128 each)."""
        return self.K // BLOCK_N


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
        assert key.dim() == 3; assert query_index.dim() == 3; assert key_index.dim() == 3
    else:
        assert query.dim() == 4, f"query dim must be 4 for BSND, got {query.dim()}"
        assert key.dim() == 4; assert query_index.dim() == 4; assert key_index.dim() == 4

    assert query.dtype in (torch.float16, torch.bfloat16)
    assert key.dtype == query.dtype
    assert query_index.dtype == query.dtype
    assert key_index.dtype == query.dtype
    assert weights.dtype in (torch.float16, torch.bfloat16, torch.float32)
    if weights.dtype != torch.float32:
        assert weights.dtype == query.dtype
    assert sparse_mode == 3, f"Only sparse_mode=3 supported, got {sparse_mode}"

    if layout_query == "TND":
        S1, N1, DQuery = query.shape
        _, N2, _ = key.shape
        _, Nidx1, DQueryIndex = query_index.shape
        _, Nidx2, _ = key_index.shape
        B = 1
    else:
        B, S1, N1, DQuery = query.shape
        _, S2, N2, _ = key.shape
        _, _, Nidx1, DQueryIndex = query_index.shape
        _, _, Nidx2, _ = key_index.shape
    K = sparse_indices.shape[-1]

    assert N2 == 1, f"N2 must be 1 (MQA), got {N2}"
    G = N1 // N2
    assert DQuery == DQUERY, f"DQuery must be {DQUERY}, got {DQuery}"
    assert DQueryIndex == DQUERY_INDEX, f"DQueryIndex must be {DQUERY_INDEX}, got {DQueryIndex}"
    assert N1 in (32, 64, 128), f"N1 must be 32/64/128, got {N1}"
    assert Nidx1 in (8, 16, 32, 64), f"Nidx1 must be 8/16/32/64, got {Nidx1}"
    assert K % 1024 == 0 and 1024 <= K <= 8192, f"K must be 1024..8192 multiple of 1024, got {K}"

    return SparseLightningIndexerGradKLLossTilingData(
        B=B, S1=S1, S2=S2, N1=N1, N2=N2, G=G,
        Nidx1=Nidx1, Nidx2=Nidx2,
        DQuery=DQuery, DQueryIndex=DQueryIndex,
        K=K, sparse_mode=sparse_mode,
    )


# ==============================================================================
# TileLang Kernel Definition
# ==============================================================================
@tilelang.jit(
    out_idx=[-4, -3, -2, -1],
    workspace_idx=[10, 11, 12, 13, 14, 15],
    pass_configs=pass_configs,
)
def tilelang_grad_kl_loss_kernel(
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
        GatherKeyIndexT: T.Tensor((B, S1, DQueryIndex, K), input_dtype),
        Weights: T.Tensor((B, S1, Nidx1), input_dtype),
        SoftmaxMax: T.Tensor((B, N2, S1, G), calc_dtype),
        SoftmaxSum: T.Tensor((B, N2, S1, G), calc_dtype),
        bmm1Res: T.Tensor((B, S1, N1, K), calc_dtype),
        bmm2Res: T.Tensor((B, S1, Nidx1, K), calc_dtype),
        PReduced: T.Tensor((B, S1, K), calc_dtype),
        ReduceSumY: T.Tensor((B, S1, K), calc_dtype),
        SoftmaxI: T.Tensor((B, S1, K), calc_dtype),
        dQueryIndex_ws: T.Tensor((B, S1, Nidx1, DQueryIndex), calc_dtype),
        dQueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), input_dtype),
        dWeight: T.Tensor((B, S1, Nidx1), input_dtype),
        reluGradRes: T.Tensor((B, S1, Nidx1, K), input_dtype),
        LossOut: T.Tensor((B, S1), calc_dtype),
    ):
        # block_num = B * s1_tile_num
        with T.Kernel(B * s1_tile_num, is_npu=True) as (cid, vid):
            bz = cid // s1_tile_num
            m = cid % s1_tile_num
            s1_start = m * s1_block
            s1_end = T.min(s1_start + s1_block, S1)

            # ============================================================
            # C Scope 1: MM1 (Q@GK^T) + MM2 (QI@GKI^T with ReLU)
            # Matches original ComputeMm1 + ComputeMm2
            # ============================================================
            with T.Scope("C"):
                Q_L1 = T.alloc_L1((N1, k_block), input_dtype)
                GK_L1 = T.alloc_L1((n_block, k_block), input_dtype)
                C_L0C = T.alloc_L0C((N1, n_block), calc_dtype)

                QI_L1 = T.alloc_L1((Nidx1, k_block), input_dtype)
                GKI_L1 = T.alloc_L1((n_block, k_block), input_dtype)
                C2_L0C = T.alloc_L0C((Nidx1, n_block), calc_dtype)

                for s1_row in T.serial(s1_start, s1_end):
                    # --- MM1 ---
                    for n_idx in T.serial(n_num):
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
                                    GatherKey[bz, s1_row, n_idx * n_block:(n_idx + 1) * n_block, d_idx * k_block:(d_idx + 1) * k_block],
                                    GK_L1,
                                )
                            else:
                                rope_d = d_idx - d_num_query
                                T.copy(
                                    GatherKeyRope[bz, s1_row, n_idx * n_block:(n_idx + 1) * n_block, rope_d * k_block:(rope_d + 1) * k_block],
                                    GK_L1,
                                )
                            T.gemm_v0(Q_L1, GK_L1, C_L0C, transpose_B=True, init=(d_idx == 0))
                        T.copy(
                            C_L0C,
                            bmm1Res[bz, s1_row, 0:N1, n_idx * n_block:(n_idx + 1) * n_block],
                        )

                    # --- MM2 ---
                    for n_idx in T.serial(n_num):
                        for k_idx in T.serial(d_num_query_index):
                            T.copy(
                                QueryIndex[bz, s1_row, 0:Nidx1, k_idx * k_block:(k_idx + 1) * k_block],
                                QI_L1,
                            )
                            T.copy(
                                GatherKeyIndex[bz, s1_row, n_idx * n_block:(n_idx + 1) * n_block, k_idx * k_block:(k_idx + 1) * k_block],
                                GKI_L1,
                            )
                            T.gemm_v0(QI_L1, GKI_L1, C2_L0C, transpose_B=True, init=(k_idx == 0))
                        T.copy(
                            C2_L0C,
                            bmm2Res[bz, s1_row, 0:Nidx1, n_idx * n_block:(n_idx + 1) * n_block],
                            enable_relu=True,
                        )

                T.set_cross_flag("FIX", 1)

            # ============================================================
            # V Scope 1: P, ReduceSumY, GlobalSoftmaxI, diff, dW, reluGrad, Loss
            # Matches original ProcessVector1: VectorP + VectorSy + VectorDwDqDk + VectorLoss
            # ============================================================
            with T.Scope("V"):
                T.wait_cross_flag(1)

                # --- UB allocation (streaming-friendly sizes) ---
                bmm1_ub = T.alloc_shared((BLOCK_N1, n_block), calc_dtype)
                bmm2_ub = T.alloc_shared((BLOCK_NIDX1, n_block), calc_dtype)
                weight_ub = T.alloc_shared(Nidx1, calc_dtype)
                weight_fp16_ub = T.alloc_shared(Nidx1, input_dtype)

                p_chunk_ub = T.alloc_shared(n_block, calc_dtype)
                rsy_chunk_ub = T.alloc_shared(n_block, calc_dtype)
                si_chunk_ub = T.alloc_shared(n_block, calc_dtype)
                diff_ub = T.alloc_shared(n_block, calc_dtype)

                rg_chunk_ub = T.alloc_shared((BLOCK_NIDX1, n_block), calc_dtype)
                rg_fp16_ub = T.alloc_shared((BLOCK_NIDX1, n_block), input_dtype)
                dw_total_ub = T.alloc_shared(Nidx1, calc_dtype)
                dw_chunk_ub = T.alloc_shared(BLOCK_NIDX1, calc_dtype)
                dw_temp_ub = T.alloc_shared(BLOCK_NIDX1, calc_dtype)
                dw_fp16_ub = T.alloc_shared(Nidx1, input_dtype)

                sm_max_ub = T.alloc_shared(G, calc_dtype)
                sm_sum_ub = T.alloc_shared(G, calc_dtype)

                eps_ub = T.alloc_shared(n_block, calc_dtype)
                ones_ub = T.alloc_shared(n_block, calc_dtype)
                mask_ub = T.alloc_shared(n_block, calc_dtype)

                rsy_global_max_ub = T.alloc_shared(1, calc_dtype)
                rsy_global_sum_ub = T.alloc_shared(1, calc_dtype)
                rsy_tmp_ub = T.alloc_shared(1, calc_dtype)

                loss_ub = T.alloc_shared(n_block, calc_dtype)
                loss_log_p_ub = T.alloc_shared(n_block, calc_dtype)
                loss_log_si_ub = T.alloc_shared(n_block, calc_dtype)
                loss_accum_ub = T.alloc_shared(1, calc_dtype)

                T.tile.fill(ones_ub, 1.0)
                T.tile.fill(eps_ub, 1e-8)

                for s1_id in T.serial(s1_start, s1_end):
                    # Load softmaxMax and softmaxSum (shape: B, N2, S1, G)
                    T.copy(SoftmaxMax[bz, 0, s1_id, 0:G], sm_max_ub)
                    T.copy(SoftmaxSum[bz, 0, s1_id, 0:G], sm_sum_ub)
                    T.tile.abs(sm_sum_ub, sm_sum_ub)
                    T.tile.max(sm_sum_ub, sm_sum_ub, eps_ub)
                    T.barrier_all()

                    # Load weights (once, reused)
                    T.copy(Weights[bz, s1_id, 0:Nidx1], weight_fp16_ub)
                    T.tile.cast(weight_ub, weight_fp16_ub, "CAST_NONE", Nidx1)
                    T.barrier_all()

                    # ---- Pass 1: Compute P_reduced = mean_g(softmax(bmm1 * scale)) ----
                    for n_idx in T.serial(n_num):
                        T.tile.fill(p_chunk_ub, 0.0)
                        for n1_c in T.serial(n1_num):
                            n1_s = n1_c * BLOCK_N1
                            T.copy(
                                bmm1Res[bz, s1_id, n1_s:n1_s + BLOCK_N1, n_idx * n_block:(n_idx + 1) * n_block],
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
                        T.copy(
                            p_chunk_ub,
                            PReduced[bz, s1_id, n_idx * n_block:(n_idx + 1) * n_block],
                        )

                    # ---- Pass 2: ReduceSumY = sum_g(weight[g] * ReLU(bmm2[g,:])) ----
                    for n_idx in T.serial(n_num):
                        T.tile.fill(rsy_chunk_ub, 0.0)
                        for nidx_c in T.serial(nidx1_num):
                            nidx_s = nidx_c * BLOCK_NIDX1
                            T.copy(
                                bmm2Res[bz, s1_id, nidx_s:nidx_s + BLOCK_NIDX1, n_idx * n_block:(n_idx + 1) * n_block],
                                bmm2_ub,
                            )
                            for g_i in T.serial(BLOCK_NIDX1):
                                T.tile.mul(bmm2_ub[g_i, :], bmm2_ub[g_i, :], weight_ub[nidx_s + g_i])
                            T.barrier_all()
                            for g_i in T.serial(BLOCK_NIDX1):
                                T.tile.add(rsy_chunk_ub, rsy_chunk_ub, bmm2_ub[g_i, :])
                            T.barrier_all()
                        T.copy(
                            rsy_chunk_ub,
                            ReduceSumY[bz, s1_id, n_idx * n_block:(n_idx + 1) * n_block],
                        )

                    # ---- Pass 3: Global SoftmaxI over all K ----
                    # Step 3a: global max
                    T.tile.fill(rsy_global_max_ub, -1e30)
                    for n_idx in T.serial(n_num):
                        T.copy(
                            ReduceSumY[bz, s1_id, n_idx * n_block:(n_idx + 1) * n_block],
                            rsy_chunk_ub,
                        )
                        T.reduce_max(rsy_chunk_ub, rsy_tmp_ub, dim=-1, real_shape=[1, n_block])
                        T.barrier_all()
                        T.tile.max(rsy_global_max_ub, rsy_global_max_ub, rsy_tmp_ub[0])

                    # Step 3b: exp(rsy - max), write to SoftmaxI workspace, accumulate sum
                    T.tile.fill(rsy_global_sum_ub, 0.0)
                    for n_idx in T.serial(n_num):
                        T.copy(
                            ReduceSumY[bz, s1_id, n_idx * n_block:(n_idx + 1) * n_block],
                            rsy_chunk_ub,
                        )
                        T.tile.sub(rsy_chunk_ub, rsy_chunk_ub, rsy_global_max_ub[0])
                        T.tile.exp(rsy_chunk_ub, rsy_chunk_ub)
                        T.barrier_all()
                        T.copy(
                            rsy_chunk_ub,
                            SoftmaxI[bz, s1_id, n_idx * n_block:(n_idx + 1) * n_block],
                        )
                        T.reduce_sum(rsy_chunk_ub, rsy_tmp_ub, dim=-1, real_shape=[1, n_block])
                        T.barrier_all()
                        T.tile.add(rsy_global_sum_ub, rsy_global_sum_ub, rsy_tmp_ub[0])

                    # Step 3c: normalize SoftmaxI
                    T.tile.max(rsy_global_sum_ub, rsy_global_sum_ub, 1e-8)
                    T.barrier_all()
                    for n_idx in T.serial(n_num):
                        T.copy(
                            SoftmaxI[bz, s1_id, n_idx * n_block:(n_idx + 1) * n_block],
                            si_chunk_ub,
                        )
                        T.tile.div(si_chunk_ub, si_chunk_ub, rsy_global_sum_ub[0])
                        T.copy(
                            si_chunk_ub,
                            SoftmaxI[bz, s1_id, n_idx * n_block:(n_idx + 1) * n_block],
                        )

                    # ---- Pass 4: dWeight and reluGrad (matching VectorDwDqDk) ----
                    # dWeight = sum over K of diff * ReLU(bmm2[g,:])
                    # reluGrad[g,k] = weight[g] * diff[k] * (bmm2[g,k] > 0 ? 1 : 0)
                    T.tile.fill(dw_total_ub, 0.0)

                    for n_idx in T.serial(n_num):
                        # Load SoftmaxI and P_reduced for this K-chunk
                        T.copy(
                            SoftmaxI[bz, s1_id, n_idx * n_block:(n_idx + 1) * n_block],
                            si_chunk_ub,
                        )
                        T.copy(
                            PReduced[bz, s1_id, n_idx * n_block:(n_idx + 1) * n_block],
                            p_chunk_ub,
                        )
                        # diff = SoftmaxI - P_reduced
                        T.tile.sub(diff_ub, si_chunk_ub, p_chunk_ub)
                        T.barrier_all()

                        # Process Nidx1 in chunks
                        for nidx_c in T.serial(nidx1_num):
                            nidx_s = nidx_c * BLOCK_NIDX1
                            T.copy(
                                bmm2Res[bz, s1_id, nidx_s:nidx_s + BLOCK_NIDX1, n_idx * n_block:(n_idx + 1) * n_block],
                                bmm2_ub,
                            )

                            # -- dWeight: dw_chunk[g] = sum_k(diff[k] * bmm2[g,k]) --
                            for g_i in T.serial(BLOCK_NIDX1):
                                T.tile.mul(rg_chunk_ub[g_i, :], diff_ub, bmm2_ub[g_i, :])
                            T.barrier_all()
                            T.reduce_sum(rg_chunk_ub, dw_chunk_ub, dim=-1, clear=True, real_shape=[BLOCK_NIDX1, n_block])
                            T.barrier_all()
                            T.copy(
                                dw_total_ub[nidx_s:nidx_s + BLOCK_NIDX1],
                                dw_temp_ub,
                            )
                            T.tile.add(dw_temp_ub, dw_temp_ub, dw_chunk_ub)
                            T.copy(
                                dw_temp_ub,
                                dw_total_ub[nidx_s:nidx_s + BLOCK_NIDX1],
                            )

                            # -- reluGrad: weight[g] * diff[k] * bool(bmm2[g,k] > 0) --
                            for g_i in T.serial(BLOCK_NIDX1):
                                g_abs = nidx_s + g_i
                                T.tile.mul(rg_chunk_ub[g_i, :], diff_ub, weight_ub[g_abs])
                                T.tile.compare(mask_ub, bmm2_ub[g_i, :], 0.0, "GT")
                                T.tile.select(rg_chunk_ub[g_i, :], mask_ub, rg_chunk_ub[g_i, :], 0.0, "VSEL_TENSOR_SCALAR_MODE")
                            T.barrier_all()

                            # Cast reluGrad to FP16 and store
                            T.tile.cast(rg_fp16_ub, rg_chunk_ub, "CAST_ROUND", BLOCK_NIDX1 * n_block)
                            T.barrier_all()
                            for g_i in T.serial(BLOCK_NIDX1):
                                g_abs = nidx_s + g_i
                                T.copy(
                                    rg_fp16_ub[g_i, :],
                                    reluGradRes[bz, s1_id, g_abs, n_idx * n_block:(n_idx + 1) * n_block],
                                )

                    # Cast and store dWeight
                    T.tile.cast(dw_fp16_ub, dw_total_ub, "CAST_ROUND", Nidx1)
                    T.barrier_all()
                    T.copy(dw_fp16_ub, dWeight[bz, s1_id, 0:Nidx1])

                    # ---- Pass 5: KL Loss (matching VectorLoss) ----
                    # KL(P||I) = sum_k(P[k] * (log(P[k]) - log(I[k])))
                    T.tile.fill(loss_accum_ub, 0.0)
                    for n_idx in T.serial(n_num):
                        T.copy(
                            PReduced[bz, s1_id, n_idx * n_block:(n_idx + 1) * n_block],
                            p_chunk_ub,
                        )
                        T.tile.max(p_chunk_ub, p_chunk_ub, eps_ub)
                        T.tile.ln(loss_log_p_ub, p_chunk_ub)
                        T.barrier_all()

                        T.copy(
                            SoftmaxI[bz, s1_id, n_idx * n_block:(n_idx + 1) * n_block],
                            si_chunk_ub,
                        )
                        T.tile.max(si_chunk_ub, si_chunk_ub, eps_ub)
                        T.tile.ln(loss_log_si_ub, si_chunk_ub)
                        T.barrier_all()

                        T.copy(
                            PReduced[bz, s1_id, n_idx * n_block:(n_idx + 1) * n_block],
                            p_chunk_ub,
                        )
                        T.tile.sub(loss_ub, loss_log_p_ub, loss_log_si_ub)
                        T.tile.mul(loss_ub, p_chunk_ub, loss_ub)
                        T.barrier_all()

                        T.reduce_sum(loss_ub, rsy_tmp_ub, dim=-1, clear=True, real_shape=[1, n_block])
                        T.barrier_all()
                        T.tile.add(loss_accum_ub, loss_accum_ub, rsy_tmp_ub[0])

                    T.copy(loss_accum_ub, LossOut[bz, s1_id:s1_id + 1])

                T.set_cross_flag("V", 12)

            # ============================================================
            # C Scope 2: dQueryIndex = sum_K(reluGrad[K_chunk] @ GKI[K_chunk]^T)
            # Matches original ComputeMm6
            # ============================================================
            with T.Scope("C"):
                T.wait_cross_flag(12)

                RG_L1 = T.alloc_L1((Nidx1, n_block), input_dtype)
                GKIT_L1 = T.alloc_L1((k_block, n_block), input_dtype)
                DQ_L0C = T.alloc_L0C((Nidx1, k_block), calc_dtype)

                for s1_row in T.serial(s1_start, s1_end):
                    for d_idx in T.serial(d_num_query_index):
                        for n_idx in T.serial(n_num):
                            T.copy(
                                reluGradRes[bz, s1_row, 0:Nidx1, n_idx * n_block:(n_idx + 1) * n_block],
                                RG_L1,
                            )
                            T.copy(
                                GatherKeyIndexT[bz, s1_row, d_idx * k_block:(d_idx + 1) * k_block, n_idx * n_block:(n_idx + 1) * n_block],
                                GKIT_L1,
                            )
                            T.gemm_v0(RG_L1, GKIT_L1, DQ_L0C, transpose_B=True, init=(n_idx == 0))
                        T.copy(
                            DQ_L0C,
                            dQueryIndex_ws[bz, s1_row, 0:Nidx1, d_idx * k_block:(d_idx + 1) * k_block],
                        )

                T.set_cross_flag("FIX", 3)

            # ============================================================
            # V Scope 2: Cast dQueryIndex FP32 -> FP16/BF16
            # Matches the Fixpipe cast in original ComputeMm6
            # ============================================================
            with T.Scope("V"):
                T.wait_cross_flag(3)

                dq_fp32_ub = T.alloc_shared((BLOCK_NIDX1, k_block), calc_dtype)
                dq_fp16_ub = T.alloc_shared((BLOCK_NIDX1, k_block), input_dtype)

                for s1_row in T.serial(s1_start, s1_end):
                    for d_idx in T.serial(d_num_query_index):
                        for nidx_c in T.serial(nidx1_num):
                            nidx_s = nidx_c * BLOCK_NIDX1
                            T.copy(
                                dQueryIndex_ws[bz, s1_row, nidx_s:nidx_s + BLOCK_NIDX1, d_idx * k_block:(d_idx + 1) * k_block],
                                dq_fp32_ub,
                            )
                            T.tile.cast(dq_fp16_ub, dq_fp32_ub, "CAST_ROUND", BLOCK_NIDX1 * k_block)
                            T.barrier_all()
                            T.copy(
                                dq_fp16_ub,
                                dQueryIndex[bz, s1_row, nidx_s:nidx_s + BLOCK_NIDX1, d_idx * k_block:(d_idx + 1) * k_block],
                            )

    return main


# ==============================================================================
# Python Preprocessing: Gather Key/KeyIndex/KeyRope by sparseIndices
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

    # sparse_indices: (B, S1, Nidx2=1, K) -> (B, S1, K)
    sparse_idx_3d = sparse_indices[:, :, 0, :]
    valid_mask = (sparse_idx_3d >= 0).unsqueeze(-1).to(dtype)
    sparse_idx_safe = sparse_idx_3d.clamp(min=0)

    # Gather Key: (B, S2, N2=1, DQuery) -> (B, S2, DQuery) -> gather -> (B, S1, K, DQuery)
    key_2d = key[:, :, 0, :]
    idx_k = sparse_idx_safe.reshape(B, S1 * K).unsqueeze(-1).expand(B, S1 * K, DQuery)
    gather_key = torch.gather(key_2d, 1, idx_k).reshape(B, S1, K, DQuery).contiguous()
    gather_key.mul_(valid_mask)

    # Gather KeyIndex: (B, S2, DQueryIndex) -> gather -> (B, S1, K, DQueryIndex)
    key_index_2d = key_index[:, :, 0, :]
    idx_ki = sparse_idx_safe.reshape(B, S1 * K).unsqueeze(-1).expand(B, S1 * K, DQueryIndex)
    gather_key_index = torch.gather(key_index_2d, 1, idx_ki).reshape(B, S1, K, DQueryIndex).contiguous()
    gather_key_index.mul_(valid_mask)

    # Gather KeyRope
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
    # reluGrad: (B, S1, Nidx1, K)  -> dKeyIndex_scatter = reluGrad^T @ queryIndex
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
# Main Entry: BSND layout
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

    # Pad S2 to VECTOR_BASEN(=256) alignment (matches original S2 alignment constraint)
    S2_pad = _ceil_div(S2, 256) * 256
    if S2_pad > S2:
        key = F.pad(key, (0, 0, 0, 0, 0, S2_pad - S2, 0, 0))
        key_index = F.pad(key_index, (0, 0, 0, 0, 0, S2_pad - S2, 0, 0))
        if key_rope is not None:
            key_rope = F.pad(key_rope, (0, 0, 0, 0, 0, S2_pad - S2, 0, 0))

    if weights.dtype == torch.float32:
        weights = weights.to(input_dtype)

    # ---- Preprocessing: Gather Key/KeyIndex/KeyRope by sparseIndices ----
    d_rope_orig = query_rope.shape[-1] if query_rope is not None else 0
    gather_key, gather_key_index, gather_key_rope = _preprocess_gather(
        key, key_index, key_rope, sparse_indices,
        B, S1, K, DQuery, DQueryIndex, d_rope_orig, input_dtype,
    )

    # Pre-transpose GatherKeyIndex -> GatherKeyIndexT (for MM6: reluGrad @ GKI^T)
    gather_key_index_t = gather_key_index.permute(0, 1, 3, 2).contiguous()

    # Pad RoPE dim to BLOCK_K alignment (minimum BLOCK_K to avoid 0-dim tensors in TileLang)
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
        func = tilelang_grad_kl_loss_kernel(
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
        softmax_i_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=device)
        d_query_index_ws = torch.zeros(B, S1, Nidx1, DQueryIndex, dtype=torch.float32, device=device)
        ws = (bmm1_res_ws, bmm2_res_ws, p_reduced_ws,
              reduce_sum_y_ws, softmax_i_ws, d_query_index_ws)
        _workspace_cache[ws_key] = ws

    (bmm1_res_ws, bmm2_res_ws, p_reduced_ws,
     reduce_sum_y_ws, softmax_i_ws, d_query_index_ws) = ws

    d_query_index_out = torch.zeros(B, S1, Nidx1, DQueryIndex, dtype=input_dtype, device=device)
    d_weight_out = torch.zeros(B, S1, Nidx1, dtype=input_dtype, device=device)
    relu_grad_out = torch.zeros(B, S1, Nidx1, K, dtype=input_dtype, device=device)
    loss_out = torch.zeros(B, S1, dtype=torch.float32, device=device)

    Zero = torch.zeros(B, S1, Nidx1, DQueryIndex, dtype=torch.float32, device=device)
    d_query_index_ws.copy_(Zero)

    # ---- Launch kernel ----
    results = func(
        # Inputs (0-9)
        query, query_rope, gather_key, gather_key_rope,
        query_index, gather_key_index, gather_key_index_t,
        weights, softmax_max, softmax_sum,
        # Workspace (10-15)
        bmm1_res_ws, bmm2_res_ws,
        p_reduced_ws, reduce_sum_y_ws, softmax_i_ws,
        d_query_index_ws,
        # Outputs (16-19)
        d_query_index_out, d_weight_out, relu_grad_out, loss_out,
    )

    d_query_index = results[0]
    d_weight = results[1].to(weights_orig_dtype if weights_orig_dtype == torch.float32 else input_dtype)
    relu_grad = results[2]
    kl_loss = results[3].sum().reshape(1).to(torch.float32)

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
