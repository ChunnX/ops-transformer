"""Plan B: Dual-kernel with Python bridge.

Architecture:
  Kernel 1 (block_num = B * s1_tile_num * n_num, K-parallel):
    C1(MM1+MM2) -> V1(P_reduced chunk, ReduceSumY chunk) per K-block
  Python bridge:
    Global SoftmaxI, diff (torch ops)
  Kernel 2 (block_num = B * s1_tile_num, S1-parallel, serial over K):
    V1(dWeight, reluGrad, Loss) -> C2(MM6: dQueryIndex) -> V2(Cast)
  Python:
    dKeyIndex scatter_add
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
_kernel2_cache = {}
_workspace_cache = {}


def _clear_caches():
    global _kernel_cache, _kernel2_cache, _workspace_cache
    _kernel_cache.clear()
    _kernel2_cache.clear()
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
                    # MM1
                    for d_idx in T.serial(d_num_total):
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

                    # MM2
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
                    T.tile.abs(sm_sum_ub, sm_sum_ub)
                    T.tile.max(sm_sum_ub, sm_sum_ub, eps_ub)
                    T.barrier_all()

                    T.copy(Weights[bz, s1_id, 0:Nidx1], weight_fp16_ub)
                    T.tile.cast(weight_ub, weight_fp16_ub, "CAST_NONE", Nidx1)
                    T.barrier_all()

                    # Pass 1: P_reduced
                    T.tile.fill(p_chunk_ub, 0.0)
                    for n1_c in T.serial(n1_num):
                        n1_s = n1_c * BLOCK_N1
                        T.copy(bmm1Res[bz, s1_id, n1_s:n1_s + BLOCK_N1, k_start:k_end], bmm1_ub)
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

                    # Pass 2: ReduceSumY
                    T.tile.fill(rsy_chunk_ub, 0.0)
                    for nidx_c in T.serial(nidx1_num):
                        nidx_s = nidx_c * BLOCK_NIDX1
                        T.copy(bmm2Res[bz, s1_id, nidx_s:nidx_s + BLOCK_NIDX1, k_start:k_end], bmm2_ub)
                        for g_i in T.serial(BLOCK_NIDX1):
                            T.tile.mul(bmm2_ub[g_i, :], bmm2_ub[g_i, :], weight_ub[nidx_s + g_i])
                        T.barrier_all()
                        for g_i in T.serial(BLOCK_NIDX1):
                            T.tile.add(rsy_chunk_ub, rsy_chunk_ub, bmm2_ub[g_i, :])
                        T.barrier_all()
                    T.copy(rsy_chunk_ub, ReduceSumY[bz, s1_id, k_start:k_end])

    return main


# ==============================================================================
# Kernel 2: K-parallel dWeight/reluGrad/Loss + MM6 (partial outputs)
# ==============================================================================
@tilelang.jit(
    out_idx=[-4, -3, -2, -1],
    pass_configs=pass_configs,
)
def planB_kernel2(
    B, S1, S2, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
    d_rope, input_dtype, calc_dtype, scale_value,
):
    s1_block = BLOCK_M
    n_block = BLOCK_N
    k_block = BLOCK_K
    s1_tile_num = _ceil_div(S1, s1_block)
    n_num = K // n_block
    d_num_query_index = DQueryIndex // k_block
    nidx1_num = Nidx1 // BLOCK_NIDX1

    @T.prim_func
    def main(
        bmm2Res: T.Tensor((B, S1, Nidx1, K), calc_dtype),
        Weights: T.Tensor((B, S1, Nidx1), input_dtype),
        Diff: T.Tensor((B, S1, K), calc_dtype),
        PReduced: T.Tensor((B, S1, K), calc_dtype),
        SoftmaxI: T.Tensor((B, S1, K), calc_dtype),
        GatherKeyIndexT: T.Tensor((B, S1, DQueryIndex, K), input_dtype),
        dQueryIndexPartial: T.Tensor((B, S1, n_num, Nidx1, DQueryIndex), calc_dtype),
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

            # ============================================================
            # V Scope 1: dWeight, reluGrad, Loss for this K-block
            # ============================================================
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
                mask_ub = T.alloc_shared(n_block, calc_dtype)

                loss_ub = T.alloc_shared(n_block, calc_dtype)
                loss_log_p_ub = T.alloc_shared(n_block, calc_dtype)
                loss_log_si_ub = T.alloc_shared(n_block, calc_dtype)
                rsy_tmp_ub = T.alloc_shared(1, calc_dtype)

                T.tile.fill(eps_ub, 1e-8)

                for s1_id in T.serial(s1_start, s1_end):
                    # Load weights (shared across all K-blocks for this s1_id)
                    T.copy(Weights[bz, s1_id, 0:Nidx1], weight_fp16_ub)
                    T.tile.cast(weight_ub, weight_fp16_ub, "CAST_NONE", Nidx1)
                    T.barrier_all()

                    # Load Diff, SoftmaxI, PReduced for this K-block
                    T.copy(Diff[bz, s1_id, k_start:k_end], diff_ub)
                    T.copy(SoftmaxI[bz, s1_id, k_start:k_end], si_chunk_ub)
                    T.copy(PReduced[bz, s1_id, k_start:k_end], p_chunk_ub)
                    T.barrier_all()

                    # ---- Pass 4: dWeight partial + reluGrad for this K-block ----
                    T.tile.fill(dw_total_ub, 0.0)

                    for nidx_c in T.serial(nidx1_num):
                        nidx_s = nidx_c * BLOCK_NIDX1
                        T.copy(
                            bmm2Res[bz, s1_id, nidx_s:nidx_s + BLOCK_NIDX1, k_start:k_end],
                            bmm2_ub,
                        )

                        # dWeight: dw_chunk[g] = sum_k(diff[k] * bmm2[g,k])
                        for g_i in T.serial(BLOCK_NIDX1):
                            T.tile.mul(rg_chunk_ub[g_i, :], diff_ub, bmm2_ub[g_i, :])
                        T.barrier_all()
                        T.reduce_sum(rg_chunk_ub, dw_chunk_ub, dim=-1, clear=True, real_shape=[BLOCK_NIDX1, n_block])
                        T.barrier_all()
                        T.copy(dw_total_ub[nidx_s:nidx_s + BLOCK_NIDX1], dw_temp_ub)
                        T.tile.add(dw_temp_ub, dw_temp_ub, dw_chunk_ub)
                        T.copy(dw_temp_ub, dw_total_ub[nidx_s:nidx_s + BLOCK_NIDX1])

                        # reluGrad: weight[g] * diff[k] * bool(bmm2[g,k] > 0)
                        for g_i in T.serial(BLOCK_NIDX1):
                            g_abs = nidx_s + g_i
                            T.tile.mul(rg_chunk_ub[g_i, :], diff_ub, weight_ub[g_abs])
                            T.tile.compare(mask_ub, bmm2_ub[g_i, :], 0.0, "GT")
                            T.tile.select(rg_chunk_ub[g_i, :], mask_ub, rg_chunk_ub[g_i, :], 0.0, "VSEL_TENSOR_SCALAR_MODE")
                        T.barrier_all()

                        # Cast and store reluGrad
                        T.tile.cast(rg_fp16_ub, rg_chunk_ub, "CAST_ROUND", BLOCK_NIDX1 * n_block)
                        T.barrier_all()
                        for g_i in T.serial(BLOCK_NIDX1):
                            g_abs = nidx_s + g_i
                            T.copy(
                                rg_fp16_ub[g_i, :],
                                reluGradRes[bz, s1_id, g_abs, k_start:k_end],
                            )

                    # Store dWeight partial (FP32, summed in Python)
                    T.copy(dw_total_ub, dWeightPartial[bz, s1_id, n_idx, 0:Nidx1])

                    # ---- Loss partial for this K-block ----
                    T.tile.max(p_chunk_ub, p_chunk_ub, eps_ub)
                    T.tile.ln(loss_log_p_ub, p_chunk_ub)
                    T.barrier_all()
                    T.tile.max(si_chunk_ub, si_chunk_ub, eps_ub)
                    T.tile.ln(loss_log_si_ub, si_chunk_ub)
                    T.barrier_all()
                    T.tile.sub(loss_ub, loss_log_p_ub, loss_log_si_ub)
                    T.tile.mul(loss_ub, p_chunk_ub, loss_ub)
                    T.barrier_all()
                    T.reduce_sum(loss_ub, rsy_tmp_ub, dim=-1, clear=True, real_shape=[1, n_block])
                    T.barrier_all()
                    T.copy(rsy_tmp_ub, LossPartial[bz, s1_id, n_idx:n_idx + 1])

                T.set_cross_flag("V", 12)

            # ============================================================
            # C Scope 2: MM6 partial for this K-block
            # ============================================================
            with T.Scope("C"):
                T.wait_cross_flag(12)

                RG_L1 = T.alloc_L1((Nidx1, n_block), input_dtype)
                GKIT_L1 = T.alloc_L1((k_block, n_block), input_dtype)
                DQ_L0C = T.alloc_L0C((Nidx1, k_block), calc_dtype)

                for s1_row in T.serial(s1_start, s1_end):
                    for d_idx in T.serial(d_num_query_index):
                        T.copy(
                            reluGradRes[bz, s1_row, 0:Nidx1, k_start:k_end],
                            RG_L1,
                        )
                        T.copy(
                            GatherKeyIndexT[bz, s1_row, d_idx * k_block:(d_idx + 1) * k_block, k_start:k_end],
                            GKIT_L1,
                        )
                        T.gemm_v0(RG_L1, GKIT_L1, DQ_L0C, transpose_B=True, init=True)
                        T.copy(
                            DQ_L0C,
                            dQueryIndexPartial[bz, s1_row, n_idx, 0:Nidx1, d_idx * k_block:(d_idx + 1) * k_block],
                        )

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

    # GKI_T for Kernel 2 MM6
    gather_key_index_t = gather_key_index.permute(0, 1, 3, 2).contiguous()

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

    # ---- Kernel 1 compilation cache ----
    cache_key = (B, S1, S2_pad, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
                 d_rope, input_dtype_str, calc_dtype_str, scale_value)
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

    # ---- Workspace ----
    ws_key = (B, S1, N1, Nidx1, K, DQueryIndex, input_dtype, device)
    n_num = tiling.n_num
    if ws_key in _workspace_cache:
        ws = _workspace_cache[ws_key]
    else:
        bmm1_res_ws = torch.zeros(B, S1, N1, K, dtype=torch.float32, device=device)
        bmm2_res_ws = torch.zeros(B, S1, Nidx1, K, dtype=torch.float32, device=device)
        p_reduced_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=device)
        reduce_sum_y_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=device)
        d_query_index_partial = torch.zeros(B, S1, n_num, Nidx1, DQueryIndex, dtype=torch.float32, device=device)
        d_weight_partial = torch.zeros(B, S1, n_num, Nidx1, dtype=torch.float32, device=device)
        loss_partial = torch.zeros(B, S1, n_num, dtype=torch.float32, device=device)
        relu_grad_ws = torch.zeros(B, S1, Nidx1, K, dtype=input_dtype, device=device)
        ws = (bmm1_res_ws, bmm2_res_ws, p_reduced_ws, reduce_sum_y_ws,
              d_query_index_partial, d_weight_partial, loss_partial, relu_grad_ws)
        _workspace_cache[ws_key] = ws

    (bmm1_res_ws, bmm2_res_ws, p_reduced_ws, reduce_sum_y_ws,
     d_query_index_partial, d_weight_partial, loss_partial, relu_grad_ws) = ws

    # ---- Launch Kernel 1 ----
    bmm2_res_ws, p_reduced_ws, reduce_sum_y_ws = func1(
        query, query_rope, gather_key, gather_key_rope,
        query_index, gather_key_index,
        weights, softmax_max, softmax_sum,
        bmm1_res_ws, bmm2_res_ws,
        p_reduced_ws, reduce_sum_y_ws,
    )

    # ---- Python Bridge: Global SoftmaxI + diff ----
    softmax_i_ws, diff_ws = _python_bridge(reduce_sum_y_ws, p_reduced_ws)

    # ---- Launch Kernel 2 (K-parallel, partial outputs) ----
    d_query_index_partial, d_weight_partial, relu_grad, loss_partial = func2(
        bmm2_res_ws, weights, diff_ws, p_reduced_ws, softmax_i_ws,
        gather_key_index_t,
        d_query_index_partial,
        d_weight_partial,
        relu_grad_ws,
        loss_partial,
    )

    # ---- Python Reduction: sum partials across K-blocks ----
    d_query_index = d_query_index_partial.sum(dim=2).to(input_dtype)  # (B, S1, n_num, Nidx1, DQ) → (B, S1, Nidx1, DQ)
    d_weight = d_weight_partial.sum(dim=2)  # (B, S1, n_num, Nidx1) → (B, S1, Nidx1)
    d_weight = d_weight.to(weights_orig_dtype if weights_orig_dtype == torch.float32 else input_dtype)
    kl_loss = loss_partial.sum().reshape(1).to(torch.float32)

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
