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

VECTOR_BASEG = 32
VECTOR_BASEN = 256

GATHER_CHUNK_SIZE = 16

DQUERY = 512
DQUERY_INDEX = 128

_kernel_cache = {}
_workspace_cache = {}


def _clear_caches():
    global _kernel_cache, _workspace_cache
    _kernel_cache.clear()
    _workspace_cache.clear()


pass_configs = {
    # 全手动同步：关闭 auto-sync，消除标量触发的大量 PIPE_ALL（V0 192 + V1 广播 320）
    # 关闭后 AscendSyncInsert early-return，核内所有 SetFlag/WaitFlag/PipeBarrier 必须手写
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,

}


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _get_aicore_count() -> int:
    """Read cube core count from NPU device properties. Fallback: 24."""
    try:
        import torch
        if torch.npu.is_available() and torch.npu.device_count() > 0:
            props = torch.npu.get_device_properties(0)
            if hasattr(props, "cube_core_num"):
                return int(props.cube_core_num)
    except Exception:
        pass
    return 24


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
    if layout_query == "TND":
        assert query.dim() == 3, f"query dim must be 3 (TND), got {query.dim()}"
        assert key.dim() == 3, f"key dim must be 3 (TND), got {key.dim()}"
        assert query_index.dim() == 3, f"queryIndex dim must be 3 (TND), got {query_index.dim()}"
        assert key_index.dim() == 3, f"keyIndex dim must be 3 (TND), got {key_index.dim()}"
    else:
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
# Single Kernel: CV-separated architecture (0630: gather moved into kernel)
# V scope 0: Gather Key/KeyIndex/KeyRope by SparseIndices → workspace
# C scope 1: Matmul1 (query @ gatherKey^T) + Matmul2 (queryIndex @ gatherKeyIndex^T)
# V scope 1: softmaxI, P_reduced, dWeight, reluGrad computation
# C scope 2: Matmul4 (reluGrad @ gatherKeyIndex → dQueryIndex)
# V scope 3: cast dQueryIndex FP32→FP16
# ==============================================================================
@tilelang.jit(
    out_idx=[-4, -3, -2, -1],
    workspace_idx=[11, 12, 13, 14, 15, 16, 17, 18],
    pass_configs=pass_configs,
)
def tilelang_grad_kl_loss_single_kernel(
    B, S1, S2, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
    d_rope, input_dtype, calc_dtype, scale_value, block_n=BLOCK_N
):
    s1_block = BLOCK_M
    n_block = block_n
    k_block = BLOCK_K
    s1_tile_num = ceil_div(S1, s1_block)
    n_num = K // n_block
    block_num = B * s1_tile_num * n_num
    d_num_query = DQuery // k_block
    d_num_rope = ceil_div(d_rope, k_block)
    d_num_total = d_num_query + d_num_rope

    d_num_query_index = DQueryIndex // k_block
    vector_baseg = Nidx1 if Nidx1 <= VECTOR_BASEG else VECTOR_BASEG
    g_num = Nidx1 // vector_baseg

    v3_chunks = Nidx1 // 4
    @T.prim_func
    def main(
        # === SPECIAL: int32 indices must be first to avoid dtype inference issues ===
        SparseIndices: T.Tensor((B, S1, Nidx2, K), "int32"),            # 0
        # === INPUTS (1-10) ===
        Query: T.Tensor((B, S1, N1, DQuery), input_dtype),              # 1
        QueryRope: T.Tensor((B, S1, N1, d_rope), input_dtype),          # 2
        Key: T.Tensor((B, S2, N2, DQuery), input_dtype),                # 3
        KeyRope: T.Tensor((B, S2, N2, d_rope), input_dtype),            # 4
        QueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), input_dtype), # 5
        KeyIndex: T.Tensor((B, S2, Nidx2, DQueryIndex), input_dtype),   # 6
        Weights: T.Tensor((B, S1, Nidx1), input_dtype),                 # 7
        SoftmaxMax: T.Tensor((B, N2, S1, G), calc_dtype),               # 8
        SoftmaxSum: T.Tensor((B, N2, S1, G), calc_dtype),               # 9
        dQueryIndexAccum: T.Tensor((B, S1, Nidx1, DQueryIndex), calc_dtype), # 10 (input, NOT workspace: host zeroes it)
        # === WORKSPACES (11-18) ===
        GatherKey_ws: T.Tensor((B, S1, K, DQuery), input_dtype),        # 11
        GatherKeyRope_ws: T.Tensor((B, S1, K, d_rope), input_dtype),    # 12
        GatherKeyIndex_ws: T.Tensor((B, S1, K, DQueryIndex), input_dtype), # 13
        bmm1Res: T.Tensor((B, S1, N1, K), calc_dtype),                  # 14
        bmm2Res: T.Tensor((B, S1, Nidx1, K), calc_dtype),               # 15
        PReduced: T.Tensor((B, S1, K), calc_dtype),                     # 16
        ReduceSumY: T.Tensor((B, S1, K), calc_dtype),                   # 17
        SoftmaxI: T.Tensor((B, S1, K), calc_dtype),                     # 18
        # === OUTPUTS (19-22) ===
        dQueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), input_dtype), # 19
        dWeight_partial: T.Tensor((B, S1, n_num, Nidx1), input_dtype),  # 20
        reluGradRes: T.Tensor((B, S1, Nidx1, K), input_dtype),          # 21
        LossOut: T.Tensor((B, S1, n_num), calc_dtype),             # 22
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bz = cid // (s1_tile_num * n_num)
            m = (cid // n_num) % s1_tile_num
            n_idx_k = cid % n_num
            s1_start = m * s1_block
            s1_end = T.min(s1_start + s1_block, S1)
            k_start = n_idx_k * n_block
            k_end = T.min(k_start + n_block, K)

            # ============================================================
            # V scope 0: Gather Key/KeyIndex/KeyRope → workspace
            # 手动同步（auto-sync 关）：单缓冲 serial，mte2↔mte3 全串行
            #   RAW: MTE3 drain 等 MTE2 fill；WAR: 下轮 fill 等本轮 drain
            # 标量 idx 从 GM 读，依赖 GetValue 同步语义（watch-point）
            # ============================================================
            num_chunks = n_block // GATHER_CHUNK_SIZE
            with T.Scope("V"):
                key_chunk_ub = T.alloc_ub((GATHER_CHUNK_SIZE, DQuery), input_dtype)
                ki_chunk_ub = T.alloc_ub((GATHER_CHUNK_SIZE, DQueryIndex), input_dtype)
                kr_chunk_ub = T.alloc_ub((GATHER_CHUNK_SIZE, max(d_rope, 1)), input_dtype)

                for s1_id in T.serial(s1_start, s1_end):
                    for chunk_id in T.serial(num_chunks):
                        chunk_start = k_start + chunk_id * GATHER_CHUNK_SIZE

                        # MTE2: fill chunk (16 rows of random reads)
                        for i in T.serial(GATHER_CHUNK_SIZE):
                            k_id = chunk_start + i
                            idx = SparseIndices[bz, s1_id, 0, k_id]
                            if idx >= 0:
                                T.copy(Key[bz, idx, 0, 0:DQuery], key_chunk_ub[i, 0:DQuery])
                                T.copy(KeyIndex[bz, idx, 0, 0:DQueryIndex], ki_chunk_ub[i, 0:DQueryIndex])
                                if d_rope > 0:
                                    T.copy(KeyRope[bz, idx, 0, 0:d_rope], kr_chunk_ub[i, 0:d_rope])

                        # RAW: MTE3 drain 等 MTE2 fill 完成
                        T.set_flag("mte2", "mte3", 0)
                        T.wait_flag("mte2", "mte3", 0)

                        # MTE3: write chunk to workspace (16KB burst)
                        T.copy(key_chunk_ub[0:GATHER_CHUNK_SIZE, 0:DQuery],
                               GatherKey_ws[bz, s1_id, chunk_start:chunk_start + GATHER_CHUNK_SIZE, 0:DQuery])
                        T.copy(ki_chunk_ub[0:GATHER_CHUNK_SIZE, 0:DQueryIndex],
                               GatherKeyIndex_ws[bz, s1_id, chunk_start:chunk_start + GATHER_CHUNK_SIZE, 0:DQueryIndex])
                        if d_rope > 0:
                            T.copy(kr_chunk_ub[0:GATHER_CHUNK_SIZE, 0:d_rope],
                                   GatherKeyRope_ws[bz, s1_id, chunk_start:chunk_start + GATHER_CHUNK_SIZE, 0:d_rope])

                        # WAR: 下轮 MTE2 fill（复用同 buffer）等本轮 MTE3 drain 完成
                        T.set_flag("mte3", "mte2", 0)
                        T.wait_flag("mte3", "mte2", 0)

                T.set_cross_flag("V", 10)

            # ============================================================
            # C scope 1: Matmul1 (with rope) + Matmul2
            # 手动同步（auto-sync 关，T.Pipelined 不再生成同步）：
            #   单缓冲 serial，每次迭代 mte2→m（RAW 载入→gemm）/ m→mte2（WAR 复用 L1）
            #   K 维累加完成后 m→fix 再 copy_l0c
            # ============================================================
            with T.Scope("C"):
                T.wait_cross_flag(10)

                Q_L1 = T.alloc_L1((N1, k_block), input_dtype)
                GK_L1 = T.alloc_L1((n_block, k_block), input_dtype)
                C_L0C = T.alloc_L0C((N1, n_block), calc_dtype)

                for s1_row in T.serial(s1_start, s1_end):
                    for d_idx in T.serial(d_num_total):
                        if d_idx < d_num_query:
                            T.copy(Query[bz, s1_row, 0:N1, d_idx * k_block:(d_idx + 1) * k_block], Q_L1)
                        else:
                            rope_d = d_idx - d_num_query
                            T.copy(QueryRope[bz, s1_row, 0:N1, rope_d * k_block:(rope_d + 1) * k_block], Q_L1)
                        if d_idx < d_num_query:
                            T.copy(GatherKey_ws[bz, s1_row, k_start:k_end, d_idx * k_block:(d_idx + 1) * k_block], GK_L1)
                        else:
                            rope_d = d_idx - d_num_query
                            T.copy(GatherKeyRope_ws[bz, s1_row, k_start:k_end, rope_d * k_block:(rope_d + 1) * k_block], GK_L1)
                        # RAW: gemm 等 L1 载入完成
                        T.set_flag("mte2", "m", 0)
                        T.wait_flag("mte2", "m", 0)
                        T.gemm_v0(Q_L1, GK_L1, C_L0C, transpose_B=True, init=(d_idx == 0))
                        # WAR: 下轮 L1 载入等本次 gemm 读完
                        T.set_flag("m", "mte2", 0)
                        T.wait_flag("m", "mte2", 0)
                    # M→FIX: copy_l0c 等 K 维累加完成
                    T.set_flag("m", "fix", 1)
                    T.wait_flag("m", "fix", 1)
                    T.copy(C_L0C, bmm1Res[bz, s1_row, 0:N1, n_idx_k * n_block:(n_idx_k + 1) * n_block])

                QI_L1 = T.alloc_L1((Nidx1, k_block), input_dtype)
                GKI_L1 = T.alloc_L1((n_block, k_block), input_dtype)
                C2_L0C = T.alloc_L0C((Nidx1, n_block), calc_dtype)

                for s1_row in T.serial(s1_start, s1_end):
                    for k_idx in T.serial(d_num_query_index):
                        T.copy(QueryIndex[bz, s1_row, 0:Nidx1, k_idx * k_block:(k_idx + 1) * k_block], QI_L1)
                        T.copy(GatherKeyIndex_ws[bz, s1_row, k_start:k_end, k_idx * k_block:(k_idx + 1) * k_block], GKI_L1)
                        # RAW: gemm 等 L1 载入
                        T.set_flag("mte2", "m", 0)
                        T.wait_flag("mte2", "m", 0)
                        T.gemm_v0(QI_L1, GKI_L1, C2_L0C, transpose_B=True, init=(k_idx == 0))
                        # WAR: 下轮载入等 gemm 读完
                        T.set_flag("m", "mte2", 0)
                        T.wait_flag("m", "mte2", 0)
                    # M→FIX
                    T.set_flag("m", "fix", 1)
                    T.wait_flag("m", "fix", 1)
                    T.copy(C2_L0C, bmm2Res[bz, s1_row, 0:Nidx1, n_idx_k * n_block:(n_idx_k + 1) * n_block], enable_relu=True)

                T.set_cross_flag("FIX", 1)

            # ============================================================
            # V scope 1: All vector ops merged (reduce memory)
            # - P_reduced, ReduceSumY for MY K block
            # - Global SoftmaxI (all cores execute)
            # - dWeight, reluGrad, Loss for MY K block
            # ============================================================
            with T.Scope("V"):
                T.wait_cross_flag(1)

                bmm1_ub = T.alloc_ub((N1, n_block), calc_dtype)
                bmm2_ub = T.alloc_ub((Nidx1, n_block), calc_dtype)
                weight_ub = T.alloc_ub(vector_baseg, calc_dtype)
                weight_all_ub = T.alloc_ub(Nidx1, calc_dtype)
                weight_all_half_ub = T.alloc_ub(Nidx1, input_dtype)
                weight_2d_ub = T.alloc_ub((Nidx1, n_block), calc_dtype)
                p_reduced_tmp_ub = T.alloc_ub(n_block, calc_dtype)
                rsy_tmp_ub = T.alloc_ub((vector_baseg, n_block), calc_dtype)
                rsy_reduce_ub = T.alloc_ub(n_block, calc_dtype)
                sm_max_ub = T.alloc_ub(G, calc_dtype)
                sm_sum_ub = T.alloc_ub(G, calc_dtype)
                eps_sm_ub = T.alloc_ub(G, calc_dtype)
                sm_2d_ub = T.alloc_ub((G, n_block), calc_dtype)
                rsy_chunk_ub = T.alloc_ub(n_block, calc_dtype)
                rsy_bc_ub = T.alloc_ub((1, n_block), calc_dtype)
                rsy_max_ub = T.alloc_ub((1,), calc_dtype)
                rsy_sum_ub = T.alloc_ub((1,), calc_dtype)
                rsy_chunk_max_ub = T.alloc_ub((1,), calc_dtype)
                rsy_chunk_sum_ub = T.alloc_ub((1,), calc_dtype)
                eps_sum_ub = T.alloc_ub((1,), calc_dtype)
                softmax_i_chunk_ub = T.alloc_ub(n_block, calc_dtype)
                rsy_all_ub = T.alloc_ub((n_num, n_block), calc_dtype)
                # row_expand brcb 中转 buffer：(rows, elems_per_block=8 for fp32)
                re_tmp_ub = T.alloc_ub((N1, 8), calc_dtype)

                for s1_id in T.serial(s1_start, s1_end):
                    T.copy(SoftmaxMax[bz, 0, s1_id, 0:G], sm_max_ub)
                    T.copy(SoftmaxSum[bz, 0, s1_id, 0:G], sm_sum_ub)
                    T.set_flag("mte2", "v", 4)
                    T.wait_flag("mte2", "v", 4)
                    T.tile.abs(sm_sum_ub, sm_sum_ub)
                    T.tile.fill(eps_sm_ub, 1e-8)
                    T.pipe_barrier("v")
                    T.tile.max(sm_sum_ub, sm_sum_ub, eps_sm_ub)

                    # P_reduced: broadcast sm_max/sm_sum to 2D once, then pure vector-vector
                    T.copy(bmm1Res[bz, s1_id, 0:N1, n_idx_k * n_block:(n_idx_k + 1) * n_block], bmm1_ub)
                    T.set_flag("mte2", "v", 5)
                    T.wait_flag("mte2", "v", 5)
                    T.tile.mul(bmm1_ub, bmm1_ub, scale_value)
                    T.tile.fill(p_reduced_tmp_ub, 1.0)  # reuse as broadcast seed
                    # sm_max 逐行减：row_expand_sub 融合替代标量广播+sub（消除 128 PIPE_ALL）
                    # bmm1[g,k] = bmm1[g,k] - sm_max[g]，走 brcb+sub_mask（PIPE_V，非 PIPE_ALL）
                    T.pipe_barrier("v")
                    T.tile.row_expand_sub_experiment(bmm1_ub, bmm1_ub, sm_max_ub, re_tmp_ub)
                    T.pipe_barrier("v")
                    T.tile.exp(bmm1_ub, bmm1_ub)
                    # sm_sum 逐行除：row_expand_div 融合替代标量广播+div（消除 128 PIPE_ALL）
                    T.pipe_barrier("v")
                    T.tile.row_expand_div_experiment(bmm1_ub, bmm1_ub, sm_sum_ub, re_tmp_ub)
                    T.pipe_barrier("v")
                    T.reduce_sum(bmm1_ub, p_reduced_tmp_ub, 0)
                    T.pipe_barrier("v")
                    T.tile.mul(p_reduced_tmp_ub, p_reduced_tmp_ub, 1.0 / G)
                    T.set_flag("v", "mte3", 6)
                    T.wait_flag("v", "mte3", 6)
                    T.copy(p_reduced_tmp_ub, PReduced[bz, s1_id, n_idx_k * n_block:(n_idx_k + 1) * n_block])

                    # ReduceSumY: broadcast Weights to 2D once, then pure vector-vector
                    T.copy(Weights[bz, s1_id, 0:Nidx1], weight_all_half_ub)
                    T.set_flag("mte2", "v", 7)
                    T.wait_flag("mte2", "v", 7)
                    T.tile.cast(weight_all_ub, weight_all_half_ub, "CAST_NONE", Nidx1)
                    # WAR: p_reduced_tmp 复用（前面已 mte3 drain 到 PReduced），fill 前等 mte3 读完
                    T.set_flag("mte3", "v", 0)
                    T.wait_flag("mte3", "v", 0)
                    T.tile.fill(p_reduced_tmp_ub, 1.0)  # reuse as broadcast seed
                    # weight 广播：暂保留标量循环（64 PIPE_ALL）——下轮再用 row_expand 处理
                    # weight_2d 需物化（dWeight 循环 (b) 直接读整行 weight_2d[idx,:] 作向量）
                    for k in range(Nidx1):
                        T.pipe_barrier("v")
                        T.tile.mul(weight_2d_ub[k, :], p_reduced_tmp_ub, weight_all_ub[k])
                    # ReduceSumY + GlobalSoftmaxI max: fused K-block pass
                    # rsy_all_ub holds all ReduceSumY blocks (avoid GM re-reads)
                    T.tile.fill(rsy_max_ub, -1e30)
                    for n_rsy in T.serial(n_num):
                        # WAR: bmm2_ub 复用，MTE2 覆盖前等上轮 V 读完
                        T.set_flag("v", "mte2", 2)
                        T.wait_flag("v", "mte2", 2)
                        T.copy(bmm2Res[bz, s1_id, 0:Nidx1, n_rsy * n_block:(n_rsy + 1) * n_block], bmm2_ub)
                        T.set_flag("mte2", "v", 1)
                        T.wait_flag("mte2", "v", 1)
                        T.pipe_barrier("v")
                        T.tile.mul(bmm2_ub, bmm2_ub, weight_2d_ub)
                        T.pipe_barrier("v")
                        T.reduce_sum(bmm2_ub, rsy_reduce_ub, 0)
                        T.pipe_barrier("v")
                        # Save to UB (for sum) + write to GM (for workspace)
                        T.copy(rsy_reduce_ub, rsy_all_ub[n_rsy, :])
                        T.pipe_barrier("mte3")
                        T.copy(rsy_reduce_ub, ReduceSumY[bz, s1_id, n_rsy * n_block:(n_rsy + 1) * n_block])
                        # Fused: update global max from rsy_reduce_ub (still in UB, no GM round-trip)
                        T.tile.mul(rsy_bc_ub[0, :], rsy_reduce_ub, 1.0)
                        T.pipe_barrier("v")
                        T.reduce_max(rsy_bc_ub, rsy_chunk_max_ub, 1)
                        T.pipe_barrier("v")
                        T.tile.max(rsy_max_ub, rsy_max_ub, rsy_chunk_max_ub)

                    # Broadcast rsy_max to vector once (softmax_i_chunk_ub free, p_reduced_tmp still has 1.0)
                    T.pipe_barrier("v")
                    T.tile.mul(softmax_i_chunk_ub, p_reduced_tmp_ub, rsy_max_ub[0])

                    # GlobalSoftmaxI: compute global sum from UB-held ReduceSumY (no GM reads)
                    T.tile.fill(rsy_sum_ub, 0)
                    for n_idx_iter in T.serial(n_num):
                        T.copy(rsy_all_ub[n_idx_iter, :], rsy_chunk_ub)
                        T.pipe_barrier("v")
                        T.tile.sub(rsy_chunk_ub, rsy_chunk_ub, softmax_i_chunk_ub)
                        T.pipe_barrier("v")
                        T.tile.exp(rsy_chunk_ub, rsy_chunk_ub)
                        T.pipe_barrier("v")
                        T.tile.mul(rsy_bc_ub[0, :], rsy_chunk_ub, 1.0)
                        T.pipe_barrier("v")
                        T.reduce_sum(rsy_bc_ub, rsy_chunk_sum_ub, 1)
                        T.pipe_barrier("v")
                        T.tile.add(rsy_sum_ub, rsy_sum_ub, rsy_chunk_sum_ub)

                    T.tile.fill(eps_sum_ub, 1e-8)
                    T.pipe_barrier("v")
                    T.tile.max(rsy_sum_ub, rsy_sum_ub, eps_sum_ub)

                    # Per-core SoftmaxI: use UB-held ReduceSumY + broadcast rsy_max (no GM reads)
                    T.copy(rsy_all_ub[n_idx_k, :], rsy_chunk_ub)
                    T.pipe_barrier("v")
                    T.tile.sub(rsy_chunk_ub, rsy_chunk_ub, softmax_i_chunk_ub)
                    T.pipe_barrier("v")
                    T.tile.exp(rsy_chunk_ub, rsy_chunk_ub)
                    T.pipe_barrier("v")
                    T.tile.div(rsy_chunk_ub, rsy_chunk_ub, rsy_sum_ub[0])
                    T.set_flag("v", "mte3", 4)
                    T.wait_flag("v", "mte3", 4)
                    T.copy(rsy_chunk_ub, SoftmaxI[bz, s1_id, n_idx_k * n_block:(n_idx_k + 1) * n_block])

                    # dWeight, reluGrad for MY K block (same V scope, no cross flag)
                    # Strategy: per-row 1D ops + per-chunk GM load (avoids UB range slicing bug)
                    #   - diff_ub (128,) used directly in 1D muls (no diff_2d needed)
                    #   - weight_2d_ub[idx,:] read directly (scalar idx, no UB range slice)
                    #   - bmm2 loaded per-chunk from GM → dw_chunk_ub (no UB→UB copy)
                    diff_ub = T.alloc_ub(n_block, calc_dtype)
                    dw_chunk_ub = T.alloc_ub((vector_baseg, n_block), calc_dtype)
                    dw_reduce_ub = T.alloc_ub(vector_baseg, calc_dtype)
                    rg_chunk_ub = T.alloc_ub((vector_baseg, n_block), calc_dtype)
                    ones_2d_ub = T.alloc_ub((vector_baseg, n_block), calc_dtype)
                    rg_half_ub = T.alloc_ub((vector_baseg, n_block), input_dtype)
                    dw_half_ub = T.alloc_ub(vector_baseg, input_dtype)
                    eps_ub = T.alloc_ub(n_block, calc_dtype)
                    loss_chunk_ub = T.alloc_ub(n_block, calc_dtype)
                    loss_log_p_ub = T.alloc_ub(n_block, calc_dtype)
                    loss_log_si_ub = T.alloc_ub(n_block, calc_dtype)
                    loss_val_ub = T.alloc_ub(n_block, calc_dtype)

                    T.tile.fill(ones_2d_ub, 1.0)
                    T.tile.fill(eps_ub, 1e-8)

                    # dWeight, reluGrad for MY K block
                    # RAW: SoftmaxI 刚由 mte3 写，mte2 读回前等 mte3 完成
                    T.set_flag("mte3", "mte2", 5)
                    T.wait_flag("mte3", "mte2", 5)
                    T.copy(SoftmaxI[bz, s1_id, n_idx_k * n_block:(n_idx_k + 1) * n_block], softmax_i_chunk_ub)
                    T.copy(PReduced[bz, s1_id, n_idx_k * n_block:(n_idx_k + 1) * n_block], p_reduced_tmp_ub)
                    T.set_flag("mte2", "v", 6)
                    T.wait_flag("mte2", "v", 6)
                    T.tile.sub(diff_ub, softmax_i_chunk_ub, p_reduced_tmp_ub)

                    for g_id in range(g_num):
                        g_start = g_id * vector_baseg

                        T.pipe_barrier("v")
                        T.tile.fill(weight_ub, 0)

                        # Load bmm2 chunk directly from GM (avoids UB→UB range copy bug)
                        # WAR/ordering: dw_chunk 复用，mte2 覆盖前等前面 mte3 写完
                        T.set_flag("mte3", "mte2", 2)
                        T.wait_flag("mte3", "mte2", 2)
                        T.copy(bmm2Res[bz, s1_id, g_start:g_start + vector_baseg, n_idx_k * n_block:(n_idx_k + 1) * n_block], dw_chunk_ub)

                        # Per-row 1D vector ops: all pure VEC, no UB-scalar, no UB range slicing
                        for i in range(vector_baseg):
                            idx = g_start + i
                            # RAW: dw_chunk 由 mte2 载入，vector 读前等（CPP 每行重发，保守保留）
                            T.set_flag("mte2", "v", 7)
                            T.wait_flag("mte2", "v", 7)
                            T.pipe_barrier("v")
                            # mask_i = 1 - exp(bmm2_row * (-10000))
                            T.tile.mul(rsy_tmp_ub[i, :], dw_chunk_ub[i, :], -10000.0)
                            T.pipe_barrier("v")
                            T.tile.exp(rsy_tmp_ub[i, :], rsy_tmp_ub[i, :])
                            T.pipe_barrier("v")
                            T.tile.sub(rsy_tmp_ub[i, :], ones_2d_ub[i, :], rsy_tmp_ub[i, :])
                            # rg_i = weight_2d_row * diff * mask_i  (read weight_2d_ub directly)
                            T.tile.mul(rg_chunk_ub[i, :], weight_2d_ub[idx, :], diff_ub)
                            T.pipe_barrier("v")
                            T.tile.mul(rg_chunk_ub[i, :], rg_chunk_ub[i, :], rsy_tmp_ub[i, :])
                            # dw_i = bmm2_row * diff (overwrite dw_chunk_ub row)
                            T.tile.mul(dw_chunk_ub[i, :], dw_chunk_ub[i, :], diff_ub)

                        T.pipe_barrier("v")
                        T.reduce_sum(dw_chunk_ub, dw_reduce_ub, 1)
                        T.pipe_barrier("v")
                        T.tile.add(weight_ub, weight_ub, dw_reduce_ub)

                        # Write reluGrad for MY K block only
                        T.tile.cast(rg_half_ub, rg_chunk_ub, "CAST_ROUND", vector_baseg * n_block)
                        T.barrier_all()  # 保留：cast rg_half 与 dw_chunk 别名，兜底
                        T.set_flag("v", "mte3", 0)
                        T.wait_flag("v", "mte3", 0)
                        T.pipe_barrier("mte3")
                        T.copy(rg_half_ub, reluGradRes[bz, s1_id, g_id * vector_baseg:(g_id + 1) * vector_baseg, n_idx_k * n_block:(n_idx_k + 1) * n_block])

                        T.pipe_barrier("v")
                        T.tile.cast(dw_half_ub, weight_ub, "CAST_ROUND", vector_baseg)
                        T.barrier_all()  # 保留：cast dw_half 别名，兜底
                        T.set_flag("v", "mte3", 1)
                        T.wait_flag("v", "mte3", 1)
                        # Write partial dWeight for MY K block only
                        T.copy(dw_half_ub, dWeight_partial[bz, s1_id, n_idx_k, g_id * vector_baseg:(g_id + 1) * vector_baseg])

                    # Compute partial Loss for MY K block only (n_idx_k)
                    T.copy(PReduced[bz, s1_id, n_idx_k * n_block:(n_idx_k + 1) * n_block], loss_chunk_ub)
                    T.set_flag("mte2", "v", 0)
                    T.wait_flag("mte2", "v", 0)
                    T.tile.max(loss_chunk_ub, loss_chunk_ub, eps_ub)
                    T.pipe_barrier("v")
                    T.tile.ln(loss_log_p_ub, loss_chunk_ub)
                    # WAR: loss_chunk 复用，下轮 mte2 覆盖前等本轮 ln 读完
                    T.set_flag("v", "mte2", 1)
                    T.wait_flag("v", "mte2", 1)

                    T.copy(SoftmaxI[bz, s1_id, n_idx_k * n_block:(n_idx_k + 1) * n_block], loss_chunk_ub)
                    T.set_flag("mte2", "v", 2)
                    T.wait_flag("mte2", "v", 2)
                    T.tile.max(loss_chunk_ub, loss_chunk_ub, eps_ub)
                    T.pipe_barrier("v")
                    T.tile.ln(loss_log_si_ub, loss_chunk_ub)
                    T.set_flag("v", "mte2", 3)
                    T.wait_flag("v", "mte2", 3)

                    T.copy(PReduced[bz, s1_id, n_idx_k * n_block:(n_idx_k + 1) * n_block], loss_chunk_ub)
                    T.pipe_barrier("v")
                    T.tile.sub(loss_val_ub, loss_log_p_ub, loss_log_si_ub)
                    # RAW: loss_chunk 刚 mte2 载入，mul 读前等
                    T.set_flag("mte2", "v", 4)
                    T.wait_flag("mte2", "v", 4)
                    T.pipe_barrier("v")
                    T.tile.mul(loss_val_ub, loss_chunk_ub, loss_val_ub)

                    T.pipe_barrier("v")
                    T.tile.mul(rsy_bc_ub[0, :], loss_val_ub, 1.0)
                    T.pipe_barrier("v")
                    loss_sum_ub = T.alloc_ub((1,), calc_dtype)
                    T.reduce_sum(rsy_bc_ub, loss_sum_ub, 1)
                    T.set_flag("v", "mte3", 5)
                    T.wait_flag("v", "mte3", 5)

                    # Write partial Loss for MY K block only (single value)
                    T.copy(loss_sum_ub, LossOut[bz, s1_id, n_idx_k:n_idx_k + 1])

                T.set_cross_flag("V", 12)

            # ============================================================
            # C scope 2: Matmul4 — per-core K-block, atomic_add to accumulate
            # reluGrad @ GatherKeyIndex_ws → dQueryIndexAccum
            # (Reads from GatherKeyIndex_ws directly, not transposed)
            # ============================================================
            with T.Scope("C"):
                T.wait_cross_flag(12)

                RG_L1 = T.alloc_L1((Nidx1, n_block), input_dtype)
                GKI2_L1 = T.alloc_L1((n_block, k_block), input_dtype)
                DQ_L0C = T.alloc_L0C((Nidx1, k_block), calc_dtype)

                for s1_row in T.serial(s1_start, s1_end):
                    for d_idx in T.serial(d_num_query_index):
                        T.copy(reluGradRes[bz, s1_row, 0:Nidx1, n_idx_k * n_block:(n_idx_k + 1) * n_block], RG_L1)
                        T.copy(GatherKeyIndex_ws[bz, s1_row, k_start:k_end, d_idx * k_block:(d_idx + 1) * k_block], GKI2_L1)
                        # RAW: gemm 等 L1 载入
                        T.set_flag("mte2", "m", 6)
                        T.wait_flag("mte2", "m", 6)
                        T.gemm_v0(RG_L1, GKI2_L1, DQ_L0C, transpose_B=False, init=True)
                        # M→FIX: atomic_add 等 gemm 完成
                        T.set_flag("m", "fix", 7)
                        T.wait_flag("m", "fix", 7)
                        T.tile.atomic_add(dQueryIndexAccum[bz, s1_row, 0:Nidx1, d_idx * k_block:(d_idx + 1) * k_block], DQ_L0C)

                T.set_cross_flag("FIX", 3)

            # ============================================================
            # V scope 3: Cast dQueryIndex (minimal buffers)
            # ============================================================
            with T.Scope("V"):
                T.wait_cross_flag(3)

                dq_fp32_ub = T.alloc_ub((4, BLOCK_K), calc_dtype)
                dq_fp16_ub = T.alloc_ub((4, BLOCK_K), input_dtype)

                for s1_row in T.serial(s1_start, s1_end):
                    for g_id in T.serial(v3_chunks):
                        for d_idx in T.serial(d_num_query_index):
                            # WAR: dq_fp32 复用，mte2 覆盖前等上轮 cast 读完
                            T.set_flag("v", "mte2", 2)
                            T.wait_flag("v", "mte2", 2)
                            T.copy(dQueryIndexAccum[bz, s1_row, g_id * 4:(g_id + 1) * 4, d_idx * BLOCK_K:(d_idx + 1) * BLOCK_K], dq_fp32_ub)
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            # WAR: dq_fp16 复用，cast 覆盖前等上轮 mte3 读完
                            T.set_flag("mte3", "v", 4)
                            T.wait_flag("mte3", "v", 4)
                            T.tile.cast(dq_fp16_ub, dq_fp32_ub, "CAST_ROUND", 4 * BLOCK_K)
                            T.set_flag("v", "mte3", 1)
                            T.wait_flag("v", "mte3", 1)
                            T.pipe_barrier("mte3")
                            T.copy(dq_fp16_ub, dQueryIndex[bz, s1_row, g_id * 4:(g_id + 1) * 4, d_idx * BLOCK_K:(d_idx + 1) * BLOCK_K])

    return main

# Gather preprocessing: from Key/KeyIndex + sparseIndices -> GatherKey/GatherKeyIndex
# ==============================================================================
def _preprocess_gather(key, key_index, key_rope, sparse_indices, B, S1, S2, N2, Nidx2, K, DQuery, DQueryIndex, d_rope_orig, dtype):
    device = key.device

    sparse_idx_3d = sparse_indices[:, :, 0, :]
    valid_mask = (sparse_idx_3d >= 0).unsqueeze(-1).to(dtype)
    sparse_idx_safe = sparse_idx_3d.clamp(min=0)

    # Fuse key, key_index, key_rope into a single gather (1x cat + 1x gather vs 3x gather)
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
# Python-layer vector computation (arch22 reference) - kept as fallback
# ==============================================================================
def _compute_vector_outputs(
    bmm1_res, bmm2_res, weights, softmax_max, softmax_sum,
    scale_value, B, S1, N1, N2, G, Nidx1, K, DQuery, DQueryIndex,
    input_dtype,
):
    bmm1 = bmm1_res * scale_value

    sm_max = softmax_max.squeeze(1)
    sm_sum = softmax_sum.squeeze(1)

    P = torch.exp(bmm1 - sm_max.unsqueeze(-1)) / sm_sum.unsqueeze(-1)

    P_reduced = P.mean(dim=2)

    relu_res = bmm2_res

    w = weights.float().unsqueeze(-1)

    weighted_relu = relu_res * w

    reduce_sum_y = weighted_relu.sum(dim=2)

    rsy_max = reduce_sum_y.max(dim=-1, keepdim=True).values
    rsy_exp = torch.exp(reduce_sum_y - rsy_max)
    rsy_sum = rsy_exp.sum(dim=-1, keepdim=True)
    softmax_i = rsy_exp / rsy_sum

    diff = softmax_i - P_reduced
    d_weights = (diff.unsqueeze(2) * relu_res).sum(dim=-1)

    relu_mask = (relu_res > 0).float()
    relu_grad = w * diff.unsqueeze(2) * relu_mask
    relu_grad_half = relu_grad.to(input_dtype)

    eps = 1e-10
    kl_loss = (P_reduced * (torch.log(P_reduced + eps) - torch.log(softmax_i + eps))).sum()

    return d_weights, relu_grad_half, kl_loss


# ==============================================================================
# ScatterAdd: compute dKeyIndex via FP16 torch.bmm + scatter_add
# ==============================================================================
def _compute_d_key_index(relu_grad, query_index, sparse_indices, B, S1, S2, Nidx1, Nidx2, K, DQueryIndex, device, dtype):
    # FP16 bmm on Ascend Cube
    relu_grad_t = relu_grad.permute(0, 1, 3, 2).reshape(B * S1, K, Nidx1)
    query_index_flat = query_index.reshape(B * S1, Nidx1, DQueryIndex)
    scatter_add = torch.bmm(relu_grad_t, query_index_flat).reshape(B, S1, K, DQueryIndex)

    d_key_index = torch.zeros(B, S2, Nidx2, DQueryIndex, dtype=torch.float32, device=device)
    sparse_idx = sparse_indices[:, :, 0, :].clamp(min=0)
    valid_mask = (sparse_indices[:, :, 0, :] >= 0).unsqueeze(-1).to(torch.float32)

    scatter_add_masked = scatter_add * valid_mask
    idx_expanded = sparse_idx.unsqueeze(-1).expand(B, S1, K, DQueryIndex)
    d_key_index_2d = d_key_index[:, :, 0, :]
    d_key_index_2d.scatter_add_(1, idx_expanded.reshape(B, S1 * K, DQueryIndex)[:, :S2, :],
                                 scatter_add_masked.reshape(B, S1 * K, DQueryIndex)[:, :S2, :])

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
        return sparse_lightning_indexer_grad_kl_loss_tnd(
            query, key, query_index, key_index, weights, sparse_indices,
            softmax_max, softmax_sum, scale_value,
            query_rope=query_rope,
            key_rope=key_rope,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            sparse_mode=sparse_mode,
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

        _PROFILE = os.environ.get("SLIG_PROFILE", "0") == "1"
        if _PROFILE:
            _t_prep_start = torch.npu.Event(enable_timing=True)
            _t_prep_end = torch.npu.Event(enable_timing=True)
            _t_kernel_start = torch.npu.Event(enable_timing=True)
            _t_kernel_end = torch.npu.Event(enable_timing=True)
            _t_reduce_start = torch.npu.Event(enable_timing=True)
            _t_reduce_end = torch.npu.Event(enable_timing=True)
            _t_scatter_start = torch.npu.Event(enable_timing=True)
            _t_scatter_end = torch.npu.Event(enable_timing=True)
            _t_prep_start.record()
        d_rope_orig = query_rope.shape[-1] if query_rope is not None else 0
        # Gather moved into kernel V0 scope — no host-side preprocess needed


        # Pad rope dims to k_block alignment, and handle None
        d_rope = ((d_rope_orig + BLOCK_K - 1) // BLOCK_K) * BLOCK_K if d_rope_orig > 0 else 0
        if query_rope is None:
            query_rope = torch.zeros(B, S1, N1, 0, dtype=input_dtype, device=query.device)
        if key_rope is None:
            key_rope = torch.zeros(B, S2_pad, N2, 0, dtype=input_dtype, device=query.device)
        if d_rope > d_rope_orig and d_rope_orig > 0:
            pad_len = d_rope - d_rope_orig
            query_rope = torch.nn.functional.pad(query_rope, (0, pad_len))
            key_rope = torch.nn.functional.pad(key_rope, (0, pad_len))

        # Block size: match NPU core count, ensure divisible by GATHER_CHUNK_SIZE
        core_count = _get_aicore_count()
        n_num_ws = core_count
        while K % n_num_ws != 0 or (K // n_num_ws) % GATHER_CHUNK_SIZE != 0:
            n_num_ws += 1
        block_n = K // n_num_ws

        cache_key = (B, S1, S2_pad, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
                     d_rope, input_dtype_str, calc_dtype_str, scale_value, block_n)

        if cache_key in _kernel_cache:
            func = _kernel_cache[cache_key]
        else:
            func = tilelang_grad_kl_loss_single_kernel(
                B, S1, S2_pad, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
                d_rope, input_dtype_str, calc_dtype_str, scale_value, block_n,
            )

            # dump 编译出的 AscendC 源码到文件，供离线核对手动同步转写
            try:
                import os as _os
                _dump_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "0706_kernel_source.cpp")
                with open(_dump_path, "w", encoding="utf-8") as _f:
                    _f.write(func.get_kernel_source())
                print(f"[dump] kernel source written to {_dump_path}")
            except Exception as _e:
                print(f"[dump] failed to write kernel source: {_e}")

            _kernel_cache[cache_key] = func

        ws_key = (B, S1, S2_pad, N1, Nidx1, K, DQueryIndex, d_rope, input_dtype, query.device, n_num_ws)
        if ws_key in _workspace_cache:
            ws = _workspace_cache[ws_key]
            d_query_index_accum, gather_key_ws, gather_key_rope_ws, gather_key_index_ws, \
                bmm1_res_ws, bmm2_res_ws, p_reduced_ws, reduce_sum_y_ws, softmax_i_ws, \
                d_query_index_out, d_weight_partial_out, relu_grad_out, loss_out = ws
        else:
            gather_key_ws = torch.zeros(B, S1, K, DQuery, dtype=input_dtype, device=query.device)
            gather_key_rope_ws = torch.zeros(B, S1, K, d_rope, dtype=input_dtype, device=query.device)
            gather_key_index_ws = torch.zeros(B, S1, K, DQueryIndex, dtype=input_dtype, device=query.device)
            bmm1_res_ws = torch.zeros(B, S1, N1, K, dtype=torch.float32, device=query.device)
            bmm2_res_ws = torch.zeros(B, S1, Nidx1, K, dtype=torch.float32, device=query.device)
            p_reduced_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=query.device)
            reduce_sum_y_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=query.device)
            softmax_i_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=query.device)
            d_query_index_accum = torch.zeros(B, S1, Nidx1, DQueryIndex, dtype=torch.float32, device=query.device)
            d_query_index_out = torch.zeros(B, S1, Nidx1, DQueryIndex, dtype=input_dtype, device=query.device)
            d_weight_partial_out = torch.zeros(B, S1, n_num_ws, Nidx1, dtype=input_dtype, device=query.device)
            relu_grad_out = torch.zeros(B, S1, Nidx1, K, dtype=input_dtype, device=query.device)
            loss_out = torch.zeros(B, S1, n_num_ws, dtype=torch.float32, device=query.device)
            _workspace_cache[ws_key] = (d_query_index_accum, gather_key_ws, gather_key_rope_ws, gather_key_index_ws,
                                        bmm1_res_ws, bmm2_res_ws, p_reduced_ws, reduce_sum_y_ws, softmax_i_ws,
                                        d_query_index_out, d_weight_partial_out, relu_grad_out, loss_out)

        # Zero workspaces before each call (gather_ws filled by V0 scope)
        d_query_index_accum.zero_()

        if _PROFILE:
            _t_prep_end.record()
            _t_kernel_start.record()
        results = func(
            sparse_indices,                             # 0  (int32 first)
            query, query_rope, key, key_rope,          # 1-4
            query_index, key_index,                    # 5-6
            weights, softmax_max, softmax_sum,         # 7-9
            d_query_index_accum,                           # 10 (input, host-zeroed)
            gather_key_ws, gather_key_rope_ws, gather_key_index_ws,  # 11-13
            bmm1_res_ws, bmm2_res_ws,                  # 14-15
            p_reduced_ws, reduce_sum_y_ws, softmax_i_ws,  # 16-18
            d_query_index_out, d_weight_partial_out, relu_grad_out, loss_out,  # 19-22
        )
        if _PROFILE:
            _t_kernel_end.record()
            _t_reduce_start.record()

        d_query_index = results[0]
        d_weight_partial = results[1]
        relu_grad = results[2]
        loss_out = results[3]

        # DEBUG: compare kernel-gathered KeyIndex vs host-gathered KeyIndex
        _DEBUG = os.environ.get("SLIG_DEBUG", "0") == "1"
        if _DEBUG:
            # Compute expected gather on host for comparison
            _, expected_gki, _ = _preprocess_gather(
                key, key_index, key_rope, sparse_indices,
                B, S1, S2_pad, N2, Nidx2, K, DQuery, DQueryIndex, d_rope_orig, query.dtype,
            )
            gki_diff = (gather_key_index_ws.float() - expected_gki.float()).abs()
            print(f"[DEBUG] GatherKeyIndex_ws vs host_gather: max_diff={gki_diff.max().item():.6e}, mean_diff={gki_diff.mean().item():.6e}")
            # Also check dQueryIndexAccum before cast
            print(f"[DEBUG] dQueryIndexAccum: max={d_query_index_accum.max().item():.6e}, min={d_query_index_accum.min().item():.6e}, has_nan={torch.isnan(d_query_index_accum).any().item()}, has_inf={torch.isinf(d_query_index_accum).any().item()}")

        # Accumulate partial dWeight across K blocks
        d_weight = d_weight_partial.sum(dim=2)
        if weights_orig_dtype == torch.float32:
            d_weight = d_weight.float()
        else:
            d_weight = d_weight.to(input_dtype)

        # Accumulate partial Loss across K blocks
        kl_loss = loss_out.sum()

        if _PROFILE:
            _t_reduce_end.record()
            _t_scatter_start = torch.npu.Event(enable_timing=True)
            _t_scatter_end = torch.npu.Event(enable_timing=True)
            _t_scatter_start.record()
        d_key_index = _compute_d_key_index(
            relu_grad, query_index, sparse_indices,
            B, S1, S2, Nidx1, Nidx2, K, DQueryIndex,
            query.device, input_dtype,
        )
        if _PROFILE:
            _t_scatter_end.record()

        loss_tensor = kl_loss.reshape(1).to(torch.float32).npu()


        if _PROFILE:
            torch.npu.synchronize()
            p_ms = _t_prep_start.elapsed_time(_t_prep_end)
            k_ms = _t_kernel_start.elapsed_time(_t_kernel_end)
            r_ms = _t_reduce_start.elapsed_time(_t_reduce_end)
            s_ms = _t_scatter_start.elapsed_time(_t_scatter_end)
            total_ms = p_ms + k_ms + r_ms + s_ms
            print(f"[PROFILE] prep={p_ms:.3f} kernel(incl gather)={k_ms:.3f} reduce={r_ms:.3f} scatter={s_ms:.3f} total={total_ms:.3f}ms")

        return d_query_index, d_key_index, d_weight, loss_tensor

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
    query_rope: Optional[torch.Tensor] = None,
    key_rope: Optional[torch.Tensor] = None,
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
        query_rope=query_rope.unsqueeze(0) if query_rope is not None else None,
        key_rope=key_rope.unsqueeze(0) if key_rope is not None else None,
        actual_seq_lengths_query=actual_q_len,
        actual_seq_lengths_key=actual_k_len,
        layout_query="BSND",
        sparse_mode=sparse_mode,
    )

    return d_qi.squeeze(0), d_ki.squeeze(0), dw.squeeze(0), loss
