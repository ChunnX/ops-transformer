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
V2_SCATTER_BATCH = 16
V2_TARGET_BATCHES_PER_AIV = 16

# One physical core owns a complete (B, S1 tile) task.  These constants mirror
# the original service split: V0 publishes separate gathered Key and KeyIndex
# streams, then C1/V1/C2/V2 consume taskId % 2 workspace slots.
S2_WORKSPACE = 1024
N_SPLIT = 128
K_SPLIT = 128
TASK_PIPELINE_DEPTH = 2
V0_TO_C1_P_FLAGS = (0, 1)
V0_TO_C1_SY_FLAGS = (2, 3)
C1_TO_V1_P_FLAGS = (4, 5)
C1_TO_V1_SY_FLAGS = (6, 7)
V1_TO_C2_FLAGS = (8, 9)
# TileLang-Ascend cannot emit the mode-1 AIV-pair wait used by the official
# kernel on A3.  Mode-2 aggregates the same flag from both AIVs at their AIC.
V1_PSY_TO_AIC_FLAG = 10
V1_PSY_RELAY_FLAG = 12

DQUERY = 512
DQUERY_INDEX = 128

_kernel_cache = {}
_workspace_cache = {}
_debug_dump_done = False


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
    """Read the physical cube-core count from the active NPU."""
    try:
        if torch.npu.is_available() and torch.npu.device_count() > 0:
            props = torch.npu.get_device_properties(torch.npu.current_device())
            if hasattr(props, "cube_core_num"):
                core_count = int(props.cube_core_num)
                if core_count > 0:
                    return core_count
    except Exception as exc:
        raise RuntimeError("Failed to query the NPU cube-core count") from exc
    raise RuntimeError(
        "NPU device properties do not expose a valid cube_core_num"
    )


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


@dataclass(frozen=True)
class TaskSchedule:
    """Runtime partition for complete (B, S1-tile) task ownership."""

    task_count: int
    launch_core_num: int
    tasks_per_core: int


def _make_task_schedule(
    B: int,
    S1: int,
    S2: int,
    K: int,
    hardware_core_num: int,
) -> TaskSchedule:
    task_count = B * S1
    hardware_core_num = max(hardware_core_num, 1)
    scatter_batches = ceil_div(K, V2_SCATTER_BATCH)
    finalize_batches = ceil_div(B * S2, V2_SCATTER_BATCH)
    v2_batches = max(scatter_batches, finalize_batches)
    v2_core_need = ceil_div(
        v2_batches,
        2 * V2_TARGET_BATCHES_PER_AIV,
    )
    active_cores = min(
        max(task_count, v2_core_need, 1),
        hardware_core_num,
    )
    return TaskSchedule(
        task_count=task_count,
        launch_core_num=active_cores,
        tasks_per_core=ceil_div(task_count, active_cores),
    )


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
# Standalone AscendC translation workbench.
#
# The original implementation owns a complete (B, S1) task and runs
# V0(task t), C1/V1/C2(task t-1), and V2(task t-2).  The implementation below
# is intentionally kept in this independent file while that task scheduler is
# translated stage by stage; 0715.py remains the precision baseline and 0720.py
# remains an isolated explicit-L0 experiment.
#
# Current inherited body is the precision reference for each stage.  Do not
# change block ownership independently: the pipeline requires the corresponding V1
# P/SY exchange and C2/V2 task-slot changes in the same scheduler rewrite.
# Single Kernel: CV-separated architecture (0630: gather moved into kernel)
# V scope 0: Gather Key/KeyIndex/KeyRope by SparseIndices → workspace
# C scope 1: Matmul1 (query @ gatherKey^T) + Matmul2 (queryIndex @ gatherKeyIndex^T)
# V scope 1: softmaxI, P_reduced, dWeight, reluGrad computation
# C scope 2: Matmul4 (reluGrad @ gatherKeyIndex → dQueryIndex)
# V scope 2: Cast dQueryIndex FP32→FP16 + Scatter dk_matmul_partial → dKeyIndex
#   Both operations split across AIV0/AIV1 for load balance
# ==============================================================================
@tilelang.jit(
    out_idx=[-4, -3, -2, -1],
    # These workspaces are cached and passed explicitly by the host wrapper.
    # Marking them as workspace_idx makes the JIT allocate private torch.empty
    # tensors and silently ignore the caller-provided buffers.
    workspace_idx=[],
    pass_configs=pass_configs,
)
def tilelang_grad_kl_loss_single_kernel(
    B, S1, S2, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
    launch_core_num,
    d_rope, input_dtype, calc_dtype, scale_value, block_n=BLOCK_N
):
    s1_block = BLOCK_M
    n_block = block_n
    k_block = BLOCK_K
    p_rows_per_chunk = min(N1, 64)
    p_row_chunk_count = ceil_div(N1, p_rows_per_chunk)
    p_cols_per_chunk = min(n_block, 64)
    p_col_chunk_count = ceil_div(n_block, p_cols_per_chunk)
    # Keep V1 backward within the precision-proven 32x64 physical tile.
    bw_col_block = min(n_block, 64)
    assert n_block % bw_col_block == 0
    bw_col_count = n_block // bw_col_block
    s1_tile_num = ceil_div(S1, s1_block)
    n_num = K // n_block
    block_num = B * s1_tile_num * n_num
    d_num_query = DQuery // k_block
    d_num_rope = ceil_div(d_rope, k_block)
    d_num_total = d_num_query + d_num_rope

    d_num_query_index = DQueryIndex // k_block
    c2_dk_k = max(Nidx1, 16)
    vector_baseg = Nidx1 if Nidx1 <= VECTOR_BASEG else VECTOR_BASEG
    g_num = Nidx1 // vector_baseg
    rows_per_aiv = Nidx1 // 2

    v3_chunks = Nidx1 // 4
    task_count = B * S1
    launch_core_num = max(launch_core_num, 1)
    tasks_per_core = ceil_div(task_count, launch_core_num)
    total_aiv_num = launch_core_num * 2
    # The official 16 KiB scatter UB is split equally between ping and pong:
    # 16 rows * 128 fp32 values * 4 bytes = 8 KiB per slot.
    v2_scatter_batch = V2_SCATTER_BATCH
    v2_total_batches = K // v2_scatter_batch
    v2_base_batches_per_core = v2_total_batches // launch_core_num
    v2_tail_batch_cores = v2_total_batches % launch_core_num
    v2_k_padded = K
    dkey_output_rows = B * S2
    assert dkey_output_rows % 32 == 0, (
        "The padded dKey output must preserve 32-row DMA alignment"
    )
    dkey_clear_batch = 32
    dkey_clear_total_batches = dkey_output_rows // dkey_clear_batch
    dkey_finalize_batch = 16
    dkey_finalize_total_batches = dkey_output_rows // dkey_finalize_batch
    loss_reduce_chunk = 128
    # DataCopyPad is block based.  Give every scalar partial its own aligned
    # 32-byte GM slot and make the finalizer consume only complete 128-value
    # chunks; short region copies lower incorrectly on the current backend.
    loss_partial_stride = 8
    loss_partial_span = task_count * 2 * loss_partial_stride
    loss_partial_size = ceil_div(
        loss_partial_span, loss_reduce_chunk
    ) * loss_reduce_chunk
    loss_full_chunk_count = loss_partial_size // loss_reduce_chunk

    @T.macro
    def vector0_gather_task(
        SparseIndices,
        KRealSize,
        Key,
        KeyRope,
        KeyIndex,
        GatherKey_ws,
        GatherKeyRope_ws,
        GatherKeyIndex_ws,
        bz_task,
        s1_begin_task,
        s1_end_task,
        task_slot,
        vector_id,
    ):
        """Gather one task with the same subblock ownership as MergeKv.

        Both AIVs gather all three streams. Each AIV owns one contiguous half
        of the current 1024-row K tile, then both publish the same mode-2
        readiness flags consumed by C1.
        """
        with T.Scope("V"):
            v0_key_ub = T.alloc_ub((2, GATHER_CHUNK_SIZE, DQuery), input_dtype)
            v0_ki_ub = T.alloc_ub((2, GATHER_CHUNK_SIZE, DQueryIndex), input_dtype)
            v0_rope_ub = T.alloc_ub((2, GATHER_CHUNK_SIZE, max(d_rope, 1)), input_dtype)
            v0_zero_key_fp32 = T.alloc_ub(DQuery, calc_dtype)
            v0_zero_ki_fp32 = T.alloc_ub(DQueryIndex, calc_dtype)
            v0_zero_rope_fp32 = T.alloc_ub(max(d_rope, 1), calc_dtype)
            v0_zero_key = T.alloc_ub(DQuery, input_dtype)
            v0_zero_ki = T.alloc_ub(DQueryIndex, input_dtype)
            v0_zero_rope = T.alloc_ub(max(d_rope, 1), input_dtype)

            T.tile.fill(v0_zero_key_fp32, 0.0)
            T.tile.fill(v0_zero_ki_fp32, 0.0)
            if d_rope > 0:
                T.tile.fill(v0_zero_rope_fp32, 0.0)
            T.pipe_barrier("v")
            T.tile.cast(v0_zero_key, v0_zero_key_fp32, "CAST_RINT", DQuery)
            T.tile.cast(v0_zero_ki, v0_zero_ki_fp32, "CAST_RINT", DQueryIndex)
            if d_rope > 0:
                T.tile.cast(v0_zero_rope, v0_zero_rope_fp32, "CAST_RINT", d_rope)
            T.pipe_barrier("v")

            T.set_flag("mte3", "mte2", 0)
            T.set_flag("mte3", "mte2", 1)
            for s1_id in T.serial(s1_begin_task, s1_end_task):
                for k_outer in T.serial(ceil_div(K, S2_WORKSPACE)):
                    outer_start = k_outer * S2_WORKSPACE
                    rows_per_aiv = S2_WORKSPACE // 2
                    aiv_start = outer_start + vector_id * rows_per_aiv
                    gather_chunks = rows_per_aiv // GATHER_CHUNK_SIZE

                    T.wait_flag("mte3", "mte2", 0)
                    for lane in T.serial(GATHER_CHUNK_SIZE):
                        k_id = aiv_start + lane
                        valid_k_size = KRealSize[bz_task, s1_id]
                        sparse_idx = SparseIndices[bz_task, s1_id, 0, k_id]
                        if k_id < valid_k_size:
                            if sparse_idx >= 0:
                                T.copy(Key[bz_task, sparse_idx, 0, 0:DQuery], v0_key_ub[0, lane, 0:DQuery])
                                T.copy(KeyIndex[bz_task, sparse_idx, 0, 0:DQueryIndex], v0_ki_ub[0, lane, 0:DQueryIndex])
                                if d_rope > 0:
                                    T.copy(KeyRope[bz_task, sparse_idx, 0, 0:d_rope], v0_rope_ub[0, lane, 0:d_rope])
                            if sparse_idx < 0:
                                T.copy(v0_zero_key, v0_key_ub[0, lane, 0:DQuery])
                                T.copy(v0_zero_ki, v0_ki_ub[0, lane, 0:DQueryIndex])
                                if d_rope > 0:
                                    T.copy(v0_zero_rope, v0_rope_ub[0, lane, 0:d_rope])
                        if k_id >= valid_k_size:
                            T.copy(v0_zero_key, v0_key_ub[0, lane, 0:DQuery])
                            T.copy(v0_zero_ki, v0_ki_ub[0, lane, 0:DQueryIndex])
                            if d_rope > 0:
                                T.copy(v0_zero_rope, v0_rope_ub[0, lane, 0:d_rope])
                    T.set_flag("mte2", "mte3", 0)

                    for chunk_id in T.serial(gather_chunks):
                        cur = chunk_id % 2
                        nxt = (chunk_id + 1) % 2
                        if chunk_id + 1 < gather_chunks:
                            next_start = aiv_start + (chunk_id + 1) * GATHER_CHUNK_SIZE
                            T.wait_flag("mte3", "mte2", nxt)
                            for lane in T.serial(GATHER_CHUNK_SIZE):
                                k_id = next_start + lane
                                valid_k_size = KRealSize[bz_task, s1_id]
                                sparse_idx = SparseIndices[bz_task, s1_id, 0, k_id]
                                if k_id < valid_k_size:
                                    if sparse_idx >= 0:
                                        T.copy(Key[bz_task, sparse_idx, 0, 0:DQuery], v0_key_ub[nxt, lane, 0:DQuery])
                                        T.copy(KeyIndex[bz_task, sparse_idx, 0, 0:DQueryIndex], v0_ki_ub[nxt, lane, 0:DQueryIndex])
                                        if d_rope > 0:
                                            T.copy(KeyRope[bz_task, sparse_idx, 0, 0:d_rope], v0_rope_ub[nxt, lane, 0:d_rope])
                                    if sparse_idx < 0:
                                        T.copy(v0_zero_key, v0_key_ub[nxt, lane, 0:DQuery])
                                        T.copy(v0_zero_ki, v0_ki_ub[nxt, lane, 0:DQueryIndex])
                                        if d_rope > 0:
                                            T.copy(v0_zero_rope, v0_rope_ub[nxt, lane, 0:d_rope])
                                if k_id >= valid_k_size:
                                    T.copy(v0_zero_key, v0_key_ub[nxt, lane, 0:DQuery])
                                    T.copy(v0_zero_ki, v0_ki_ub[nxt, lane, 0:DQueryIndex])
                                    if d_rope > 0:
                                        T.copy(v0_zero_rope, v0_rope_ub[nxt, lane, 0:d_rope])
                            T.set_flag("mte2", "mte3", nxt)

                        T.wait_flag("mte2", "mte3", cur)
                        cur_start = aiv_start + chunk_id * GATHER_CHUNK_SIZE
                        T.copy(v0_key_ub[cur, 0:GATHER_CHUNK_SIZE, 0:DQuery],
                               GatherKey_ws[bz_task, s1_id, cur_start:cur_start + GATHER_CHUNK_SIZE, 0:DQuery])
                        T.copy(v0_ki_ub[cur, 0:GATHER_CHUNK_SIZE, 0:DQueryIndex],
                               GatherKeyIndex_ws[bz_task, s1_id, cur_start:cur_start + GATHER_CHUNK_SIZE, 0:DQueryIndex])
                        if d_rope > 0:
                            T.copy(v0_rope_ub[cur, 0:GATHER_CHUNK_SIZE, 0:d_rope],
                                   GatherKeyRope_ws[bz_task, s1_id, cur_start:cur_start + GATHER_CHUNK_SIZE, 0:d_rope])
                        T.set_flag("mte3", "mte2", cur)

                    # A mode-2 AIC wait joins the same flag from both AIVs.
                    # Publish both readiness flags after each AIV finishes its
                    # half, so C1 cannot observe either workspace prematurely.
                    if k_outer % 2 == 0:
                        T.set_cross_flag("MTE3", V0_TO_C1_P_FLAGS[0])
                        T.set_cross_flag("MTE3", V0_TO_C1_SY_FLAGS[0])
                    else:
                        T.set_cross_flag("MTE3", V0_TO_C1_P_FLAGS[1])
                        T.set_cross_flag("MTE3", V0_TO_C1_SY_FLAGS[1])

            T.wait_flag("mte3", "mte2", 0)
            T.wait_flag("mte3", "mte2", 1)

    @T.macro
    def cube1_task_l0c_double_experimental(
        Query,
        QueryRope,
        QueryIndex,
        GatherKey_ws,
        GatherKeyRope_ws,
        GatherKeyIndex_ws,
        bmm1Res,
        bmm2Res,
        bz_task,
        s1_begin_task,
        s1_end_task,
        task_slot,
    ):
        """Compute P and SY with exact physical Cube ping-pong buffers."""
        with T.Scope("C"):
            # P uses two independently allocated, physically exact slots.
            # Keeping SY in the following C scope lets the allocator reuse L0
            # capacity after both P slots are drained.
            c1_query_chunk_0_l1 = T.alloc_L1((N1, K_SPLIT), input_dtype)
            c1_query_chunk_1_l1 = T.alloc_L1((N1, K_SPLIT), input_dtype)
            c1_query_chunk_2_l1 = T.alloc_L1((N1, K_SPLIT), input_dtype)
            c1_query_chunk_3_l1 = T.alloc_L1((N1, K_SPLIT), input_dtype)
            c1_query_rope_cache_l1 = T.alloc_L1(
                (N1, max(d_rope, 1)), input_dtype
            )
            c1_key_l1_ping = T.alloc_L1((N_SPLIT, K_SPLIT), input_dtype)
            c1_key_l1_pong = T.alloc_L1((N_SPLIT, K_SPLIT), input_dtype)
            c1_q_l0a_ping = T.alloc_L0A((N1, K_SPLIT), input_dtype)
            c1_q_l0a_pong = T.alloc_L0A((N1, K_SPLIT), input_dtype)
            c1_key_l0b_ping = T.alloc_L0B((K_SPLIT, N_SPLIT), input_dtype)
            c1_key_l0b_pong = T.alloc_L0B((K_SPLIT, N_SPLIT), input_dtype)
            c1_p_l0c_ping = T.alloc_L0C((N1, N_SPLIT), calc_dtype)
            c1_p_l0c_pong = T.alloc_L0C((N1, N_SPLIT), calc_dtype)

            T.set_flag("mte1", "mte2", 0)
            T.set_flag("m", "mte1", 0)
            T.set_flag("fix", "m", 0)
            T.set_flag("mte1", "mte2", 1)
            T.set_flag("m", "mte1", 1)
            T.set_flag("fix", "m", 1)

            for s1_id in T.serial(s1_begin_task, s1_end_task):
                T.copy(
                    Query[bz_task, s1_id, 0:N1, 0:K_SPLIT],
                    c1_query_chunk_0_l1,
                )
                T.copy(
                    Query[bz_task, s1_id, 0:N1, K_SPLIT:2 * K_SPLIT],
                    c1_query_chunk_1_l1,
                )
                T.copy(
                    Query[bz_task, s1_id, 0:N1, 2 * K_SPLIT:3 * K_SPLIT],
                    c1_query_chunk_2_l1,
                )
                T.copy(
                    Query[bz_task, s1_id, 0:N1, 3 * K_SPLIT:4 * K_SPLIT],
                    c1_query_chunk_3_l1,
                )
                if d_rope > 0:
                    T.copy(
                        QueryRope[bz_task, s1_id, 0:N1, 0:d_rope],
                        c1_query_rope_cache_l1[0:N1, 0:d_rope],
                    )
                # Event 4 is dedicated to the one-time Query/Rope preload.
                # The existing C1 tile event IDs remain owned by Key traffic.
                T.set_flag("mte2", "mte1", 4)
                T.wait_flag("mte2", "mte1", 4)

                for k_outer in T.serial(ceil_div(K, S2_WORKSPACE)):
                    if k_outer % 2 == 0:
                        T.wait_cross_flag(V0_TO_C1_P_FLAGS[0])
                    else:
                        T.wait_cross_flag(V0_TO_C1_P_FLAGS[1])

                    for k_inner in T.serial(S2_WORKSPACE // N_SPLIT):
                        k_start = k_outer * S2_WORKSPACE + k_inner * N_SPLIT
                        # Slot 0 owns the first 128-column Key tile.  L1,
                        # L0A/L0B use d_idx parity; L0C uses K-tile parity.
                        T.wait_flag("mte1", "mte2", 0)
                        T.copy(
                            GatherKey_ws[bz_task, s1_id,
                                         k_start:k_start + N_SPLIT,
                                         0:K_SPLIT],
                            c1_key_l1_ping,
                        )
                        T.set_flag("mte2", "mte1", 0)

                        for d_idx in T.serial(d_num_query):
                            if d_idx % 2 == 0:
                                T.wait_flag("m", "mte1", 0)
                                T.wait_flag("mte2", "mte1", 0)
                                if d_idx == 0:
                                    T.copy(c1_query_chunk_0_l1, c1_q_l0a_ping)
                                if d_idx == 2:
                                    T.copy(c1_query_chunk_2_l1, c1_q_l0a_ping)
                                T.copy(c1_key_l1_ping, c1_key_l0b_ping, transpose=True)
                                T.set_flag("mte1", "mte2", 0)
                                T.set_flag("mte1", "m", 0)

                                if d_idx + 1 < d_num_query:
                                    T.wait_flag("mte1", "mte2", 1)
                                    T.copy(
                                        GatherKey_ws[bz_task, s1_id,
                                                     k_start:k_start + N_SPLIT,
                                                     (d_idx + 1) * K_SPLIT:
                                                     (d_idx + 2) * K_SPLIT],
                                        c1_key_l1_pong,
                                    )
                                    T.set_flag("mte2", "mte1", 1)
                                if d_idx + 1 >= d_num_query:
                                    if d_num_rope > 0:
                                        T.wait_flag("mte1", "mte2", 1)
                                        T.copy(
                                            GatherKeyRope_ws[bz_task, s1_id,
                                                              k_start:k_start + N_SPLIT,
                                                              0:K_SPLIT],
                                            c1_key_l1_pong,
                                        )
                                        T.set_flag("mte2", "mte1", 1)

                                T.wait_flag("mte1", "m", 0)
                                if k_inner % 2 == 0:
                                    T.wait_flag("fix", "m", 0)
                                    T.mma(c1_q_l0a_ping, c1_key_l0b_ping, c1_p_l0c_ping, init=(d_idx == 0))
                                else:
                                    T.wait_flag("fix", "m", 1)
                                    T.mma(c1_q_l0a_ping, c1_key_l0b_ping, c1_p_l0c_pong, init=(d_idx == 0))
                                # The next d tile accumulates into this same
                                # L0C slot.  MTE1 may advance on the other
                                # operand slot, but M must finish this write
                                # before its FIX_M token is returned.
                                if d_idx + 1 < d_num_query or d_num_rope > 0:
                                    T.pipe_barrier("m")
                                T.set_flag("m", "mte1", 0)
                                if d_idx + 1 < d_num_query or d_num_rope > 0:
                                    if k_inner % 2 == 0:
                                        T.set_flag("fix", "m", 0)
                                    else:
                                        T.set_flag("fix", "m", 1)

                            if d_idx % 2 == 1:
                                T.wait_flag("m", "mte1", 1)
                                T.wait_flag("mte2", "mte1", 1)
                                if d_idx == 1:
                                    T.copy(c1_query_chunk_1_l1, c1_q_l0a_pong)
                                if d_idx == 3:
                                    T.copy(c1_query_chunk_3_l1, c1_q_l0a_pong)
                                T.copy(c1_key_l1_pong, c1_key_l0b_pong, transpose=True)
                                T.set_flag("mte1", "mte2", 1)
                                T.set_flag("mte1", "m", 1)

                                if d_idx + 1 < d_num_query:
                                    T.wait_flag("mte1", "mte2", 0)
                                    T.copy(
                                        GatherKey_ws[bz_task, s1_id,
                                                     k_start:k_start + N_SPLIT,
                                                     (d_idx + 1) * K_SPLIT:
                                                     (d_idx + 2) * K_SPLIT],
                                        c1_key_l1_ping,
                                    )
                                    T.set_flag("mte2", "mte1", 0)
                                if d_idx + 1 >= d_num_query:
                                    if d_num_rope > 0:
                                        T.wait_flag("mte1", "mte2", 0)
                                        T.copy(
                                            GatherKeyRope_ws[bz_task, s1_id,
                                                              k_start:k_start + N_SPLIT,
                                                              0:K_SPLIT],
                                            c1_key_l1_ping,
                                        )
                                        T.set_flag("mte2", "mte1", 0)

                                T.wait_flag("mte1", "m", 1)
                                if k_inner % 2 == 0:
                                    T.wait_flag("fix", "m", 0)
                                    T.mma(c1_q_l0a_pong, c1_key_l0b_pong, c1_p_l0c_ping, init=(d_idx == 0))
                                else:
                                    T.wait_flag("fix", "m", 1)
                                    T.mma(c1_q_l0a_pong, c1_key_l0b_pong, c1_p_l0c_pong, init=(d_idx == 0))
                                if d_idx + 1 < d_num_query or d_num_rope > 0:
                                    T.pipe_barrier("m")
                                T.set_flag("m", "mte1", 1)
                                if d_idx + 1 < d_num_query or d_num_rope > 0:
                                    if k_inner % 2 == 0:
                                        T.set_flag("fix", "m", 0)
                                    else:
                                        T.set_flag("fix", "m", 1)

                        for rope_idx in T.serial(d_num_rope):
                            stage_idx = d_num_query + rope_idx
                            if stage_idx % 2 == 0:
                                T.wait_flag("m", "mte1", 0)
                                T.wait_flag("mte2", "mte1", 0)
                                T.copy(c1_query_rope_cache_l1[0:N1, rope_idx * K_SPLIT:(rope_idx + 1) * K_SPLIT], c1_q_l0a_ping)
                                T.copy(c1_key_l1_ping, c1_key_l0b_ping, transpose=True)
                                T.set_flag("mte1", "mte2", 0)
                                T.set_flag("mte1", "m", 0)
                                if rope_idx + 1 < d_num_rope:
                                    T.wait_flag("mte1", "mte2", 1)
                                    T.copy(GatherKeyRope_ws[bz_task, s1_id, k_start:k_start + N_SPLIT,
                                                           (rope_idx + 1) * K_SPLIT:(rope_idx + 2) * K_SPLIT], c1_key_l1_pong)
                                    T.set_flag("mte2", "mte1", 1)
                                T.wait_flag("mte1", "m", 0)
                                if k_inner % 2 == 0:
                                    T.wait_flag("fix", "m", 0)
                                    T.mma(c1_q_l0a_ping, c1_key_l0b_ping, c1_p_l0c_ping, init=False)
                                else:
                                    T.wait_flag("fix", "m", 1)
                                    T.mma(c1_q_l0a_ping, c1_key_l0b_ping, c1_p_l0c_pong, init=False)
                                if rope_idx + 1 < d_num_rope:
                                    T.pipe_barrier("m")
                                T.set_flag("m", "mte1", 0)
                                if rope_idx + 1 < d_num_rope:
                                    if k_inner % 2 == 0:
                                        T.set_flag("fix", "m", 0)
                                    else:
                                        T.set_flag("fix", "m", 1)

                            if stage_idx % 2 == 1:
                                T.wait_flag("m", "mte1", 1)
                                T.wait_flag("mte2", "mte1", 1)
                                T.copy(c1_query_rope_cache_l1[0:N1, rope_idx * K_SPLIT:(rope_idx + 1) * K_SPLIT], c1_q_l0a_pong)
                                T.copy(c1_key_l1_pong, c1_key_l0b_pong, transpose=True)
                                T.set_flag("mte1", "mte2", 1)
                                T.set_flag("mte1", "m", 1)
                                if rope_idx + 1 < d_num_rope:
                                    T.wait_flag("mte1", "mte2", 0)
                                    T.copy(GatherKeyRope_ws[bz_task, s1_id, k_start:k_start + N_SPLIT,
                                                           (rope_idx + 1) * K_SPLIT:(rope_idx + 2) * K_SPLIT], c1_key_l1_ping)
                                    T.set_flag("mte2", "mte1", 0)
                                T.wait_flag("mte1", "m", 1)
                                if k_inner % 2 == 0:
                                    T.wait_flag("fix", "m", 0)
                                    T.mma(c1_q_l0a_pong, c1_key_l0b_pong, c1_p_l0c_ping, init=False)
                                else:
                                    T.wait_flag("fix", "m", 1)
                                    T.mma(c1_q_l0a_pong, c1_key_l0b_pong, c1_p_l0c_pong, init=False)
                                if rope_idx + 1 < d_num_rope:
                                    T.pipe_barrier("m")
                                T.set_flag("m", "mte1", 1)
                                if rope_idx + 1 < d_num_rope:
                                    if k_inner % 2 == 0:
                                        T.set_flag("fix", "m", 0)
                                    else:
                                        T.set_flag("fix", "m", 1)

                        if k_inner % 2 == 0:
                            T.set_flag("m", "fix", 0)
                            T.wait_flag("m", "fix", 0)
                            T.copy(c1_p_l0c_ping, bmm1Res[bz_task, s1_id, 0:N1,
                                                           k_start:k_start + N_SPLIT])
                            T.set_flag("fix", "m", 0)
                        else:
                            T.set_flag("m", "fix", 1)
                            T.wait_flag("m", "fix", 1)
                            T.copy(c1_p_l0c_pong, bmm1Res[bz_task, s1_id, 0:N1,
                                                           k_start:k_start + N_SPLIT])
                            T.set_flag("fix", "m", 1)

                T.wait_flag("mte1", "mte2", 0)
                T.wait_flag("mte1", "mte2", 1)
                T.wait_flag("m", "mte1", 0)
                T.wait_flag("m", "mte1", 1)
                T.wait_flag("fix", "m", 0)
                T.wait_flag("fix", "m", 1)
                if task_slot == 0:
                    T.set_cross_flag("FIX", C1_TO_V1_P_FLAGS[0])
                else:
                    T.set_cross_flag("FIX", C1_TO_V1_P_FLAGS[1])

        with T.Scope("C"):
            # This scope begins after P has drained both slots, so these exact
            # SY shapes can reuse the same physical L0 capacity.
            c1_qi_l1 = T.alloc_L1((Nidx1, K_SPLIT), input_dtype)
            c1_ki_l1 = T.alloc_L1((N_SPLIT, K_SPLIT), input_dtype)
            c1_qi_l0a = T.alloc_L0A((Nidx1, K_SPLIT), input_dtype)
            c1_ki_l0b = T.alloc_L0B((K_SPLIT, N_SPLIT), input_dtype)
            c1_sy_l0c = T.alloc_L0C((Nidx1, N_SPLIT), calc_dtype)

            T.set_flag("mte1", "mte2", 1)
            T.set_flag("m", "mte1", 1)
            T.set_flag("fix", "m", 1)
            for s1_id in T.serial(s1_begin_task, s1_end_task):
                T.copy(QueryIndex[bz_task, s1_id, 0:Nidx1, 0:DQueryIndex], c1_qi_l1)
                T.set_flag("mte2", "mte1", 5)
                T.wait_flag("mte2", "mte1", 5)
                # For the current single 128-column QueryIndex block, the
                # left operand is invariant over every K tile in C1-SY.
                # Multi-block shapes retain the original per-d_idx load.
                if d_num_query_index == 1:
                    T.wait_flag("m", "mte1", 1)
                    T.copy(c1_qi_l1, c1_qi_l0a)
                    T.set_flag("m", "mte1", 1)
                for k_outer in T.serial(ceil_div(K, S2_WORKSPACE)):
                    if k_outer % 2 == 0:
                        T.wait_cross_flag(V0_TO_C1_SY_FLAGS[0])
                    else:
                        T.wait_cross_flag(V0_TO_C1_SY_FLAGS[1])

                    for k_inner in T.serial(S2_WORKSPACE // N_SPLIT):
                        k_start = k_outer * S2_WORKSPACE + k_inner * N_SPLIT
                        for d_idx in T.serial(d_num_query_index):
                            T.wait_flag("mte1", "mte2", 1)
                            T.copy(
                                GatherKeyIndex_ws[bz_task, s1_id,
                                                  k_start:k_start + N_SPLIT,
                                                  d_idx * K_SPLIT:(d_idx + 1) * K_SPLIT],
                                c1_ki_l1,
                            )
                            T.set_flag("mte2", "mte1", 1)
                            T.wait_flag("m", "mte1", 1)
                            T.wait_flag("mte2", "mte1", 1)
                            if d_num_query_index != 1:
                                T.copy(
                                    c1_qi_l1[0:Nidx1,
                                             d_idx * K_SPLIT:(d_idx + 1) * K_SPLIT],
                                    c1_qi_l0a,
                                )
                            T.copy(c1_ki_l1, c1_ki_l0b, transpose=True)
                            T.set_flag("mte1", "mte2", 1)
                            T.set_flag("mte1", "m", 1)

                            T.wait_flag("mte1", "m", 1)
                            T.wait_flag("fix", "m", 1)
                            T.mma(c1_qi_l0a, c1_ki_l0b, c1_sy_l0c, init=(d_idx == 0))
                            T.set_flag("m", "mte1", 1)
                            if d_idx + 1 < d_num_query_index:
                                T.set_flag("fix", "m", 1)
                        T.set_flag("m", "fix", 1)
                        T.wait_flag("m", "fix", 1)
                        T.copy(
                            c1_sy_l0c,
                            bmm2Res[bz_task, s1_id, 0:Nidx1,
                                    k_start:k_start + N_SPLIT],
                            enable_relu=True,
                        )
                        T.set_flag("fix", "m", 1)

                T.wait_flag("mte1", "mte2", 1)
                T.wait_flag("m", "mte1", 1)
                T.wait_flag("fix", "m", 1)
                if task_slot == 0:
                    T.set_cross_flag("FIX", C1_TO_V1_SY_FLAGS[0])
                else:
                    T.set_cross_flag("FIX", C1_TO_V1_SY_FLAGS[1])

    @T.macro
    def cube1_task_physical_db_experiment(
        Query,
        QueryRope,
        QueryIndex,
        GatherKey_ws,
        GatherKeyRope_ws,
        GatherKeyIndex_ws,
        bmm1Res,
        bmm2Res,
        bz_task,
        s1_begin_task,
        s1_end_task,
        task_slot,
    ):
        """Precision-proven single-slot C1 pipeline."""
        with T.Scope("C"):
            c1_query_chunk_0_l1 = T.alloc_L1((N1, K_SPLIT), input_dtype)
            c1_query_chunk_1_l1 = T.alloc_L1((N1, K_SPLIT), input_dtype)
            c1_query_chunk_2_l1 = T.alloc_L1((N1, K_SPLIT), input_dtype)
            c1_query_chunk_3_l1 = T.alloc_L1((N1, K_SPLIT), input_dtype)
            c1_query_rope_cache_l1 = T.alloc_L1((N1, max(d_rope, 1)), input_dtype)
            c1_key_l1 = T.alloc_L1((N_SPLIT, K_SPLIT), input_dtype)
            c1_q_l0a = T.alloc_L0A((N1, K_SPLIT), input_dtype)
            c1_key_l0b = T.alloc_L0B((K_SPLIT, N_SPLIT), input_dtype)
            c1_p_l0c = T.alloc_L0C((N1, N_SPLIT), calc_dtype)
            c1_qi_l1 = T.alloc_L1((Nidx1, K_SPLIT), input_dtype)
            c1_ki_l1 = T.alloc_L1((N_SPLIT, K_SPLIT), input_dtype)
            c1_qi_l0a = T.alloc_L0A((Nidx1, K_SPLIT), input_dtype)
            c1_ki_l0b = T.alloc_L0B((K_SPLIT, N_SPLIT), input_dtype)
            c1_sy_l0c = T.alloc_L0C((Nidx1, N_SPLIT), calc_dtype)

            T.set_flag("mte1", "mte2", 0)
            T.set_flag("m", "mte1", 0)
            T.set_flag("fix", "m", 0)
            T.set_flag("mte1", "mte2", 1)
            T.set_flag("m", "mte1", 1)
            T.set_flag("fix", "m", 1)

            for s1_id in T.serial(s1_begin_task, s1_end_task):
                T.copy(Query[bz_task, s1_id, 0:N1, 0:K_SPLIT], c1_query_chunk_0_l1)
                T.copy(Query[bz_task, s1_id, 0:N1, K_SPLIT:2 * K_SPLIT], c1_query_chunk_1_l1)
                T.copy(Query[bz_task, s1_id, 0:N1, 2 * K_SPLIT:3 * K_SPLIT], c1_query_chunk_2_l1)
                T.copy(Query[bz_task, s1_id, 0:N1, 3 * K_SPLIT:4 * K_SPLIT], c1_query_chunk_3_l1)
                if d_rope > 0:
                    T.copy(QueryRope[bz_task, s1_id, 0:N1, 0:d_rope], c1_query_rope_cache_l1[0:N1, 0:d_rope])
                T.set_flag("mte2", "mte1", 4)
                T.wait_flag("mte2", "mte1", 4)

                T.copy(QueryIndex[bz_task, s1_id, 0:Nidx1, 0:DQueryIndex], c1_qi_l1)
                T.set_flag("mte2", "mte1", 5)

                for k_outer in T.serial(ceil_div(K, S2_WORKSPACE)):
                    if k_outer % 2 == 0:
                        T.wait_cross_flag(V0_TO_C1_P_FLAGS[0])
                    else:
                        T.wait_cross_flag(V0_TO_C1_P_FLAGS[1])

                    for k_inner in T.serial(S2_WORKSPACE // N_SPLIT):
                        k_start = k_outer * S2_WORKSPACE + k_inner * N_SPLIT
                        T.wait_flag("mte1", "mte2", 0)
                        T.copy(GatherKey_ws[bz_task, s1_id, k_start:k_start + N_SPLIT, 0:K_SPLIT], c1_key_l1)
                        T.set_flag("mte2", "mte1", 0)

                        for d_idx in T.serial(d_num_query):
                            T.wait_flag("m", "mte1", 0)
                            T.wait_flag("mte2", "mte1", 0)
                            if d_idx == 0:
                                T.copy(c1_query_chunk_0_l1, c1_q_l0a)
                            if d_idx == 1:
                                T.copy(c1_query_chunk_1_l1, c1_q_l0a)
                            if d_idx == 2:
                                T.copy(c1_query_chunk_2_l1, c1_q_l0a)
                            if d_idx == 3:
                                T.copy(c1_query_chunk_3_l1, c1_q_l0a)
                            T.copy(c1_key_l1, c1_key_l0b, transpose=True)
                            T.set_flag("mte1", "mte2", 0)
                            T.set_flag("mte1", "m", 0)

                            if d_idx + 1 < d_num_query:
                                T.wait_flag("mte1", "mte2", 0)
                                T.copy(GatherKey_ws[bz_task, s1_id, k_start:k_start + N_SPLIT,
                                                    (d_idx + 1) * K_SPLIT:(d_idx + 2) * K_SPLIT], c1_key_l1)
                                T.set_flag("mte2", "mte1", 0)
                            if d_idx + 1 >= d_num_query:
                                if d_num_rope > 0:
                                    T.wait_flag("mte1", "mte2", 0)
                                    T.copy(GatherKeyRope_ws[bz_task, s1_id, k_start:k_start + N_SPLIT, 0:K_SPLIT], c1_key_l1)
                                    T.set_flag("mte2", "mte1", 0)

                            T.wait_flag("mte1", "m", 0)
                            T.wait_flag("fix", "m", 0)
                            T.mma(c1_q_l0a, c1_key_l0b, c1_p_l0c, init=(d_idx == 0))
                            T.set_flag("m", "mte1", 0)
                            if d_idx + 1 < d_num_query or d_num_rope > 0:
                                T.set_flag("fix", "m", 0)

                        for rope_idx in T.serial(d_num_rope):
                            T.wait_flag("m", "mte1", 0)
                            T.wait_flag("mte2", "mte1", 0)
                            T.copy(c1_query_rope_cache_l1[0:N1, rope_idx * K_SPLIT:(rope_idx + 1) * K_SPLIT], c1_q_l0a)
                            T.copy(c1_key_l1, c1_key_l0b, transpose=True)
                            T.set_flag("mte1", "mte2", 0)
                            T.set_flag("mte1", "m", 0)

                            if rope_idx + 1 < d_num_rope:
                                T.wait_flag("mte1", "mte2", 0)
                                T.copy(GatherKeyRope_ws[bz_task, s1_id, k_start:k_start + N_SPLIT,
                                                       (rope_idx + 1) * K_SPLIT:(rope_idx + 2) * K_SPLIT], c1_key_l1)
                                T.set_flag("mte2", "mte1", 0)

                            T.wait_flag("mte1", "m", 0)
                            T.wait_flag("fix", "m", 0)
                            T.mma(c1_q_l0a, c1_key_l0b, c1_p_l0c, init=False)
                            T.set_flag("m", "mte1", 0)
                            if rope_idx + 1 < d_num_rope:
                                T.set_flag("fix", "m", 0)

                        T.set_flag("m", "fix", 0)
                        T.wait_flag("m", "fix", 0)
                        T.copy(c1_p_l0c, bmm1Res[bz_task, s1_id, 0:N1, k_start:k_start + N_SPLIT])
                        T.set_flag("fix", "m", 0)

                T.wait_flag("mte1", "mte2", 0)
                T.wait_flag("m", "mte1", 0)
                T.wait_flag("fix", "m", 0)
                if task_slot == 0:
                    T.set_cross_flag("FIX", C1_TO_V1_P_FLAGS[0])
                else:
                    T.set_cross_flag("FIX", C1_TO_V1_P_FLAGS[1])

                T.wait_flag("mte2", "mte1", 5)
                if d_num_query_index == 1:
                    T.wait_flag("m", "mte1", 1)
                    T.copy(c1_qi_l1, c1_qi_l0a)
                    T.set_flag("m", "mte1", 1)
                for k_outer in T.serial(ceil_div(K, S2_WORKSPACE)):
                    if k_outer % 2 == 0:
                        T.wait_cross_flag(V0_TO_C1_SY_FLAGS[0])
                    else:
                        T.wait_cross_flag(V0_TO_C1_SY_FLAGS[1])

                    for k_inner in T.serial(S2_WORKSPACE // N_SPLIT):
                        k_start = k_outer * S2_WORKSPACE + k_inner * N_SPLIT
                        for d_idx in T.serial(d_num_query_index):
                            T.wait_flag("mte1", "mte2", 1)
                            T.copy(GatherKeyIndex_ws[bz_task, s1_id, k_start:k_start + N_SPLIT,
                                                     d_idx * K_SPLIT:(d_idx + 1) * K_SPLIT], c1_ki_l1)
                            T.set_flag("mte2", "mte1", 1)
                            T.wait_flag("m", "mte1", 1)
                            T.wait_flag("mte2", "mte1", 1)
                            if d_num_query_index != 1:
                                T.copy(c1_qi_l1[0:Nidx1, d_idx * K_SPLIT:(d_idx + 1) * K_SPLIT], c1_qi_l0a)
                            T.copy(c1_ki_l1, c1_ki_l0b, transpose=True)
                            T.set_flag("mte1", "mte2", 1)
                            T.set_flag("mte1", "m", 1)

                            T.wait_flag("mte1", "m", 1)
                            T.wait_flag("fix", "m", 1)
                            T.mma(c1_qi_l0a, c1_ki_l0b, c1_sy_l0c, init=(d_idx == 0))
                            T.set_flag("m", "mte1", 1)
                            if d_idx + 1 < d_num_query_index:
                                T.set_flag("fix", "m", 1)
                        T.set_flag("m", "fix", 1)
                        T.wait_flag("m", "fix", 1)
                        T.copy(c1_sy_l0c, bmm2Res[bz_task, s1_id, 0:Nidx1,
                                                  k_start:k_start + N_SPLIT], enable_relu=True)
                        T.set_flag("fix", "m", 1)

                T.wait_flag("mte1", "mte2", 1)
                T.wait_flag("m", "mte1", 1)
                T.wait_flag("fix", "m", 1)
                if task_slot == 0:
                    T.set_cross_flag("FIX", C1_TO_V1_SY_FLAGS[0])
                else:
                    T.set_cross_flag("FIX", C1_TO_V1_SY_FLAGS[1])

    @T.macro
    def cube1_task(
        Query,
        QueryRope,
        QueryIndex,
        GatherKey_ws,
        GatherKeyRope_ws,
        GatherKeyIndex_ws,
        bmm1Res,
        bmm2Res,
        bz_task,
        s1_begin_task,
        s1_end_task,
        task_slot,
    ):
        """Precision-proven single-slot C1 baseline."""
        with T.Scope("C"):
            c1_query_chunk_0_l1 = T.alloc_L1((N1, K_SPLIT), input_dtype)
            c1_query_chunk_1_l1 = T.alloc_L1((N1, K_SPLIT), input_dtype)
            c1_query_chunk_2_l1 = T.alloc_L1((N1, K_SPLIT), input_dtype)
            c1_query_chunk_3_l1 = T.alloc_L1((N1, K_SPLIT), input_dtype)
            c1_query_rope_cache_l1 = T.alloc_L1((N1, max(d_rope, 1)), input_dtype)
            c1_key_l1 = T.alloc_L1((N_SPLIT, K_SPLIT), input_dtype)
            c1_q_l0a = T.alloc_L0A((N1, K_SPLIT), input_dtype)
            c1_key_l0b = T.alloc_L0B((K_SPLIT, N_SPLIT), input_dtype)
            c1_p_l0c = T.alloc_L0C((N1, N_SPLIT), calc_dtype)
            c1_qi_l1 = T.alloc_L1((Nidx1, K_SPLIT), input_dtype)
            c1_ki_l1 = T.alloc_L1((N_SPLIT, K_SPLIT), input_dtype)
            c1_qi_l0a = T.alloc_L0A((Nidx1, K_SPLIT), input_dtype)
            c1_ki_l0b = T.alloc_L0B((K_SPLIT, N_SPLIT), input_dtype)
            c1_sy_l0c = T.alloc_L0C((Nidx1, N_SPLIT), calc_dtype)

            T.set_flag("mte1", "mte2", 0)
            T.set_flag("m", "mte1", 0)
            T.set_flag("fix", "m", 0)
            T.set_flag("mte1", "mte2", 1)
            T.set_flag("m", "mte1", 1)
            T.set_flag("fix", "m", 1)

            for s1_id in T.serial(s1_begin_task, s1_end_task):
                T.copy(Query[bz_task, s1_id, 0:N1, 0:K_SPLIT], c1_query_chunk_0_l1)
                T.copy(Query[bz_task, s1_id, 0:N1, K_SPLIT:2 * K_SPLIT], c1_query_chunk_1_l1)
                T.copy(Query[bz_task, s1_id, 0:N1, 2 * K_SPLIT:3 * K_SPLIT], c1_query_chunk_2_l1)
                T.copy(Query[bz_task, s1_id, 0:N1, 3 * K_SPLIT:4 * K_SPLIT], c1_query_chunk_3_l1)
                if d_rope > 0:
                    T.copy(QueryRope[bz_task, s1_id, 0:N1, 0:d_rope], c1_query_rope_cache_l1[0:N1, 0:d_rope])
                T.set_flag("mte2", "mte1", 4)
                T.wait_flag("mte2", "mte1", 4)

                T.copy(QueryIndex[bz_task, s1_id, 0:Nidx1, 0:DQueryIndex], c1_qi_l1)
                T.set_flag("mte2", "mte1", 5)

                for k_outer in T.serial(ceil_div(K, S2_WORKSPACE)):
                    if k_outer % 2 == 0:
                        T.wait_cross_flag(V0_TO_C1_P_FLAGS[0])
                    else:
                        T.wait_cross_flag(V0_TO_C1_P_FLAGS[1])
                    for k_inner in T.serial(S2_WORKSPACE // N_SPLIT):
                        k_start = k_outer * S2_WORKSPACE + k_inner * N_SPLIT
                        T.wait_flag("mte1", "mte2", 0)
                        T.copy(GatherKey_ws[bz_task, s1_id, k_start:k_start + N_SPLIT, 0:K_SPLIT], c1_key_l1)
                        T.set_flag("mte2", "mte1", 0)

                        for d_idx in T.serial(d_num_query):
                            T.wait_flag("m", "mte1", 0)
                            T.wait_flag("mte2", "mte1", 0)
                            if d_idx == 0:
                                T.copy(c1_query_chunk_0_l1, c1_q_l0a)
                            if d_idx == 1:
                                T.copy(c1_query_chunk_1_l1, c1_q_l0a)
                            if d_idx == 2:
                                T.copy(c1_query_chunk_2_l1, c1_q_l0a)
                            if d_idx == 3:
                                T.copy(c1_query_chunk_3_l1, c1_q_l0a)
                            T.copy(c1_key_l1, c1_key_l0b, transpose=True)
                            T.set_flag("mte1", "mte2", 0)
                            T.set_flag("mte1", "m", 0)
                            if d_idx + 1 < d_num_query:
                                T.wait_flag("mte1", "mte2", 0)
                                T.copy(GatherKey_ws[bz_task, s1_id, k_start:k_start + N_SPLIT,
                                                    (d_idx + 1) * K_SPLIT:(d_idx + 2) * K_SPLIT], c1_key_l1)
                                T.set_flag("mte2", "mte1", 0)
                            if d_idx + 1 >= d_num_query:
                                if d_num_rope > 0:
                                    T.wait_flag("mte1", "mte2", 0)
                                    T.copy(GatherKeyRope_ws[bz_task, s1_id, k_start:k_start + N_SPLIT, 0:K_SPLIT], c1_key_l1)
                                    T.set_flag("mte2", "mte1", 0)
                            T.wait_flag("mte1", "m", 0)
                            T.wait_flag("fix", "m", 0)
                            T.mma(c1_q_l0a, c1_key_l0b, c1_p_l0c, init=(d_idx == 0))
                            T.set_flag("m", "mte1", 0)
                            if d_idx + 1 < d_num_query or d_num_rope > 0:
                                T.set_flag("fix", "m", 0)

                        for rope_idx in T.serial(d_num_rope):
                            T.wait_flag("m", "mte1", 0)
                            T.wait_flag("mte2", "mte1", 0)
                            T.copy(c1_query_rope_cache_l1[0:N1, rope_idx * K_SPLIT:(rope_idx + 1) * K_SPLIT], c1_q_l0a)
                            T.copy(c1_key_l1, c1_key_l0b, transpose=True)
                            T.set_flag("mte1", "mte2", 0)
                            T.set_flag("mte1", "m", 0)
                            if rope_idx + 1 < d_num_rope:
                                T.wait_flag("mte1", "mte2", 0)
                                T.copy(GatherKeyRope_ws[bz_task, s1_id, k_start:k_start + N_SPLIT,
                                                       (rope_idx + 1) * K_SPLIT:(rope_idx + 2) * K_SPLIT], c1_key_l1)
                                T.set_flag("mte2", "mte1", 0)
                            T.wait_flag("mte1", "m", 0)
                            T.wait_flag("fix", "m", 0)
                            T.mma(c1_q_l0a, c1_key_l0b, c1_p_l0c, init=False)
                            T.set_flag("m", "mte1", 0)
                            if rope_idx + 1 < d_num_rope:
                                T.set_flag("fix", "m", 0)

                        T.set_flag("m", "fix", 0)
                        T.wait_flag("m", "fix", 0)
                        T.copy(c1_p_l0c, bmm1Res[bz_task, s1_id, 0:N1, k_start:k_start + N_SPLIT])
                        T.set_flag("fix", "m", 0)

                T.wait_flag("mte1", "mte2", 0)
                T.wait_flag("m", "mte1", 0)
                T.wait_flag("fix", "m", 0)
                if task_slot == 0:
                    T.set_cross_flag("FIX", C1_TO_V1_P_FLAGS[0])
                else:
                    T.set_cross_flag("FIX", C1_TO_V1_P_FLAGS[1])

                T.wait_flag("mte2", "mte1", 5)
                if d_num_query_index == 1:
                    T.wait_flag("m", "mte1", 1)
                    T.copy(c1_qi_l1, c1_qi_l0a)
                    T.set_flag("m", "mte1", 1)
                for k_outer in T.serial(ceil_div(K, S2_WORKSPACE)):
                    if k_outer % 2 == 0:
                        T.wait_cross_flag(V0_TO_C1_SY_FLAGS[0])
                    else:
                        T.wait_cross_flag(V0_TO_C1_SY_FLAGS[1])
                    for k_inner in T.serial(S2_WORKSPACE // N_SPLIT):
                        k_start = k_outer * S2_WORKSPACE + k_inner * N_SPLIT
                        for d_idx in T.serial(d_num_query_index):
                            T.wait_flag("mte1", "mte2", 1)
                            T.copy(GatherKeyIndex_ws[bz_task, s1_id, k_start:k_start + N_SPLIT,
                                                      d_idx * K_SPLIT:(d_idx + 1) * K_SPLIT], c1_ki_l1)
                            T.set_flag("mte2", "mte1", 1)
                            T.wait_flag("m", "mte1", 1)
                            T.wait_flag("mte2", "mte1", 1)
                            if d_num_query_index != 1:
                                T.copy(c1_qi_l1[0:Nidx1, d_idx * K_SPLIT:(d_idx + 1) * K_SPLIT], c1_qi_l0a)
                            T.copy(c1_ki_l1, c1_ki_l0b, transpose=True)
                            T.set_flag("mte1", "mte2", 1)
                            T.set_flag("mte1", "m", 1)
                            T.wait_flag("mte1", "m", 1)
                            T.wait_flag("fix", "m", 1)
                            T.mma(c1_qi_l0a, c1_ki_l0b, c1_sy_l0c, init=(d_idx == 0))
                            T.set_flag("m", "mte1", 1)
                            if d_idx + 1 < d_num_query_index:
                                T.set_flag("fix", "m", 1)
                        T.set_flag("m", "fix", 1)
                        T.wait_flag("m", "fix", 1)
                        T.copy(c1_sy_l0c, bmm2Res[bz_task, s1_id, 0:Nidx1,
                                                  k_start:k_start + N_SPLIT], enable_relu=True)
                        T.set_flag("fix", "m", 1)

                T.wait_flag("mte1", "mte2", 1)
                T.wait_flag("m", "mte1", 1)
                T.wait_flag("fix", "m", 1)
                if task_slot == 0:
                    T.set_cross_flag("FIX", C1_TO_V1_SY_FLAGS[0])
                else:
                    T.set_cross_flag("FIX", C1_TO_V1_SY_FLAGS[1])

    @T.macro
    def vector1_psy_task(
        KRealSize,
        ValidKMask,
        SoftmaxMax,
        SoftmaxSum,
        Weights,
        bmm1Res,
        bmm2Res,
        PReduced,
        SoftmaxI,
        bz_task,
        s1_begin_task,
        s1_end_task,
        task_slot,
        vector_id,
    ):
        """Translate the first half of ProcessVector1.

        AIV0 owns VectorP and materializes PReduced.  AIV1 owns VectorSy and
        materializes SoftmaxI.  Both operate over all K chunks of the same
        task, rather than repeating a full-K reduction on every K launch.
        """
        with T.Scope("V"):
            if task_slot == 0:
                if vector_id == 0:
                    T.wait_cross_flag(C1_TO_V1_P_FLAGS[0])
                else:
                    T.wait_cross_flag(C1_TO_V1_SY_FLAGS[0])
            else:
                if vector_id == 0:
                    T.wait_cross_flag(C1_TO_V1_P_FLAGS[1])
                else:
                    T.wait_cross_flag(C1_TO_V1_SY_FLAGS[1])

            for s1_id in T.serial(s1_begin_task, s1_end_task):
                if vector_id == 0:
                    # P and SY run on different AIVs.  Keep their allocations
                    # branch-local so the UB planner can reuse the same space.
                    v1_p_bmm1 = T.alloc_ub((N1, n_block), calc_dtype)
                    v1_p_max = T.alloc_ub(N1, calc_dtype)
                    v1_p_sum = T.alloc_ub(N1, calc_dtype)
                    # Reused after sum clamping as the 64-row partial reduce.
                    v1_p_partial = T.alloc_ub(max(N1, n_block), calc_dtype)
                    v1_p_out = T.alloc_ub(n_block, calc_dtype)
                    v1_p_tmp = T.alloc_ub((N1, 8), calc_dtype)
                    v1_p_valid = T.alloc_ub(n_block, calc_dtype)

                    T.copy(SoftmaxMax[bz_task, 0, s1_id, 0:N1], v1_p_max)
                    T.copy(SoftmaxSum[bz_task, 0, s1_id, 0:N1], v1_p_sum)
                    T.set_flag("mte2", "v", 0)
                    T.set_flag("mte3", "v", 0)
                    T.set_flag("v", "mte2", 1)
                    T.wait_flag("mte2", "v", 0)
                    T.tile.abs(v1_p_sum, v1_p_sum)
                    T.tile.fill(v1_p_partial[0:N1], 1e-8)
                    T.pipe_barrier("v")
                    T.tile.max(v1_p_sum, v1_p_sum, v1_p_partial[0:N1])
                    for n_idx in T.serial(n_num):
                        n_start = n_idx * n_block
                        valid_k_size = KRealSize[bz_task, s1_id]
                        T.wait_flag("v", "mte2", 1)
                        T.wait_flag("mte3", "v", 0)
                        if n_start < valid_k_size:
                            T.copy(
                                bmm1Res[bz_task, s1_id, 0:N1,
                                        n_start:n_start + n_block],
                                v1_p_bmm1,
                            )
                            T.copy(
                                ValidKMask[bz_task, s1_id, n_start:n_start + n_block],
                                v1_p_valid,
                            )
                            T.set_flag("mte2", "v", 1)
                            T.wait_flag("mte2", "v", 1)
                            T.tile.mul(v1_p_bmm1, v1_p_bmm1, scale_value)
                            T.pipe_barrier("v")
                            # mask repeat is uint8_t.  A 128x128 FP32 tile would
                            # lower to repeat=256 and wrap to zero, so process at
                            # most 64 rows per row-expand instruction.
                            for p_row_chunk in T.serial(p_row_chunk_count):
                                p_row_start = p_row_chunk * p_rows_per_chunk
                                for p_col_chunk in T.serial(p_col_chunk_count):
                                    p_col_start = p_col_chunk * p_cols_per_chunk
                                    T.tile.row_expand_sub_experiment(
                                        v1_p_bmm1[
                                            p_row_start:p_row_start + p_rows_per_chunk,
                                            p_col_start:p_col_start + p_cols_per_chunk,
                                        ],
                                        v1_p_bmm1[
                                            p_row_start:p_row_start + p_rows_per_chunk,
                                            p_col_start:p_col_start + p_cols_per_chunk,
                                        ],
                                        v1_p_max[
                                            p_row_start:p_row_start + p_rows_per_chunk
                                        ],
                                        v1_p_tmp[
                                            p_row_start:p_row_start + p_rows_per_chunk,
                                            0:8,
                                        ],
                                    )
                            T.pipe_barrier("v")
                            T.tile.exp(v1_p_bmm1, v1_p_bmm1)
                            T.pipe_barrier("v")
                            for p_row_chunk in T.serial(p_row_chunk_count):
                                p_row_start = p_row_chunk * p_rows_per_chunk
                                for p_col_chunk in T.serial(p_col_chunk_count):
                                    p_col_start = p_col_chunk * p_cols_per_chunk
                                    T.tile.row_expand_div_experiment(
                                        v1_p_bmm1[
                                            p_row_start:p_row_start + p_rows_per_chunk,
                                            p_col_start:p_col_start + p_cols_per_chunk,
                                        ],
                                        v1_p_bmm1[
                                            p_row_start:p_row_start + p_rows_per_chunk,
                                            p_col_start:p_col_start + p_cols_per_chunk,
                                        ],
                                        v1_p_sum[
                                            p_row_start:p_row_start + p_rows_per_chunk
                                        ],
                                        v1_p_tmp[
                                            p_row_start:p_row_start + p_rows_per_chunk,
                                            0:8,
                                        ],
                                    )
                            T.pipe_barrier("v")
                            if n_start + n_block > valid_k_size:
                                for p_mask_row in T.serial(N1):
                                    T.tile.mul(
                                        v1_p_bmm1[p_mask_row, 0:n_block],
                                        v1_p_bmm1[p_mask_row, 0:n_block],
                                        v1_p_valid,
                                    )
                                T.pipe_barrier("v")
                            # The Ascend fast path used here is stable for
                            # 64x128 -> 128 (also used by VectorSy), while a single
                            # 128x128 axis-0 reduction corrupts the result.  Match
                            # the official N-loop structure and accumulate two
                            # partial reductions in UB.
                            T.tile.fill(v1_p_out, 0.0)
                            T.pipe_barrier("v")
                            for p_reduce_chunk in T.serial(p_row_chunk_count):
                                p_reduce_start = p_reduce_chunk * p_rows_per_chunk
                                T.reduce_sum(
                                    v1_p_bmm1[
                                        p_reduce_start:p_reduce_start + p_rows_per_chunk,
                                        0:n_block,
                                    ],
                                    v1_p_partial[0:n_block],
                                    0,
                                )
                                T.pipe_barrier("v")
                                T.tile.add(
                                    v1_p_out,
                                    v1_p_out,
                                    v1_p_partial[0:n_block],
                                )
                                T.pipe_barrier("v")
                            T.tile.mul(v1_p_out, v1_p_out, 1.0 / N1)
                        else:
                            T.tile.fill(v1_p_out, 0.0)
                        T.set_flag("v", "mte2", 1)
                        T.set_flag("v", "mte3", 0)
                        T.wait_flag("v", "mte3", 0)
                        T.copy(
                            v1_p_out,
                            PReduced[bz_task, s1_id,
                                     n_start:n_start + n_block],
                        )
                        T.set_flag("mte3", "v", 0)

                    T.wait_flag("v", "mte2", 1)
                    T.wait_flag("mte3", "v", 0)

                if vector_id == 1:
                    # Keep the current BMM2 tile and the loop-carried ReduceSumY
                    # history in one physical allocation.  Separate allocations
                    # were incorrectly coalesced by the UB planner even though
                    # the history remains live across every K-block iteration.
                    v1_sy_storage = T.alloc_ub(
                        (Nidx1 + n_num, n_block), calc_dtype
                    )
                    v1_sy_weight_half = T.alloc_ub(Nidx1, input_dtype)
                    v1_sy_weight = T.alloc_ub(Nidx1, calc_dtype)
                    v1_sy_weight_2d = T.alloc_ub((Nidx1, n_block), calc_dtype)
                    v1_sy_reduce = T.alloc_ub(n_block, calc_dtype)
                    v1_sy_chunk = T.alloc_ub(n_block, calc_dtype)
                    v1_sy_valid_mask = T.alloc_ub(n_block // 16, calc_dtype)
                    v1_sy_broadcast = T.alloc_ub((1, n_block), calc_dtype)
                    v1_sy_max = T.alloc_ub((1,), calc_dtype)
                    v1_sy_sum = T.alloc_ub((1,), calc_dtype)
                    v1_sy_tmp_max = T.alloc_ub((1,), calc_dtype)
                    v1_sy_tmp_sum = T.alloc_ub((1,), calc_dtype)
                    v1_sy_eps = T.alloc_ub((1,), calc_dtype)
                    v1_sy_floor = T.alloc_ub((1,), calc_dtype)
                    v1_sy_tmp = T.alloc_ub((Nidx1, 8), calc_dtype)

                    T.copy(Weights[bz_task, s1_id, 0:Nidx1], v1_sy_weight_half)
                    T.set_flag("mte2", "v", 2)
                    T.set_flag("mte3", "v", 1)
                    T.wait_flag("mte2", "v", 2)
                    T.tile.cast(
                        v1_sy_weight,
                        v1_sy_weight_half,
                        "CAST_NONE",
                        Nidx1,
                    )
                    T.tile.fill(v1_sy_weight_2d, 1.0)
                    T.pipe_barrier("v")
                    # The AscendC row-expand lowering interprets every 64
                    # FP32 elements as one row.  Split a 128-column tile so
                    # BRCB reads exactly Nidx1 scalars and mul_mask stays
                    # within the physical rows.
                    for sy_col_chunk in T.serial(p_col_chunk_count):
                        sy_col_start = sy_col_chunk * p_cols_per_chunk
                        T.tile.row_expand_mul_experiment(
                            v1_sy_weight_2d[
                                0:Nidx1,
                                sy_col_start:sy_col_start + p_cols_per_chunk,
                            ],
                            v1_sy_weight_2d[
                                0:Nidx1,
                                sy_col_start:sy_col_start + p_cols_per_chunk,
                            ],
                            v1_sy_weight,
                            v1_sy_tmp,
                        )
                    T.tile.fill(v1_sy_max, -T.infinity(calc_dtype))
                    T.set_flag("v", "mte2", 3)
                    for n_idx in T.serial(n_num):
                        n_start = n_idx * n_block
                        valid_k_size = KRealSize[bz_task, s1_id]
                        if n_start < valid_k_size:
                            T.wait_flag("v", "mte2", 3)
                            T.copy(
                                bmm2Res[bz_task, s1_id, 0:Nidx1,
                                        n_start:n_start + n_block],
                                v1_sy_storage[0:Nidx1, 0:n_block],
                            )
                            T.copy(
                                ValidKMask[bz_task, s1_id, n_start:n_start + n_block],
                                v1_sy_chunk,
                            )
                            T.set_flag("mte2", "v", 3)
                            T.wait_flag("mte2", "v", 3)
                            T.tile.mul(
                                v1_sy_storage[0:Nidx1, 0:n_block],
                                v1_sy_storage[0:Nidx1, 0:n_block],
                                v1_sy_weight_2d,
                            )
                            T.pipe_barrier("v")
                            T.reduce_sum(
                                v1_sy_storage[0:Nidx1, 0:n_block],
                                v1_sy_reduce,
                                0,
                            )
                            if n_start + n_block > valid_k_size:
                                T.tile.compare(
                                    v1_sy_valid_mask,
                                    v1_sy_chunk,
                                    0.0,
                                    "GT",
                                )
                                T.pipe_barrier("v")
                                T.tile.select(
                                    v1_sy_reduce,
                                    v1_sy_valid_mask,
                                    v1_sy_reduce,
                                    -1e30,
                                    "VSEL_TENSOR_SCALAR_MODE",
                                )
                                T.pipe_barrier("v")
                            T.copy(
                                v1_sy_reduce,
                                v1_sy_storage[Nidx1 + n_idx, 0:n_block],
                            )
                            T.pipe_barrier("v")
                            T.tile.mul(v1_sy_broadcast[0, :], v1_sy_reduce, 1.0)
                            T.pipe_barrier("v")
                            T.reduce_max(v1_sy_broadcast, v1_sy_tmp_max, 1)
                            T.pipe_barrier("v")
                            T.tile.max(v1_sy_max, v1_sy_max, v1_sy_tmp_max)
                            T.set_flag("v", "mte2", 3)
                        else:
                            T.tile.fill(v1_sy_storage[Nidx1 + n_idx, 0:n_block], -1e30)

                    T.tile.fill(v1_sy_floor, -1e20)
                    T.pipe_barrier("v")
                    T.tile.max(v1_sy_max, v1_sy_max, v1_sy_floor)

                    T.wait_flag("v", "mte2", 3)
                    T.tile.fill(v1_sy_sum, 0.0)
                    for n_idx in T.serial(n_num):
                        T.tile.broadcast(v1_sy_broadcast, v1_sy_max)
                        T.pipe_barrier("v")
                        # The ReduceSumY history is no longer needed after the
                        # global max has been found.  Keep exp(x - max) in the
                        # same row so the normalization pass does not repeat it.
                        T.tile.sub(
                            v1_sy_storage[Nidx1 + n_idx, 0:n_block],
                            v1_sy_storage[Nidx1 + n_idx, 0:n_block],
                            v1_sy_broadcast[0, :],
                        )
                        T.pipe_barrier("v")
                        T.tile.exp(
                            v1_sy_storage[Nidx1 + n_idx, 0:n_block],
                            v1_sy_storage[Nidx1 + n_idx, 0:n_block],
                        )
                        T.pipe_barrier("v")
                        T.tile.mul(
                            v1_sy_broadcast[0, :],
                            v1_sy_storage[Nidx1 + n_idx, 0:n_block],
                            1.0,
                        )
                        T.pipe_barrier("v")
                        T.reduce_sum(v1_sy_broadcast, v1_sy_tmp_sum, 1)
                        T.pipe_barrier("v")
                        T.tile.add(v1_sy_sum, v1_sy_sum, v1_sy_tmp_sum)

                    T.tile.fill(v1_sy_eps, 1e-8)
                    T.pipe_barrier("v")
                    T.tile.max(v1_sy_sum, v1_sy_sum, v1_sy_eps)
                    for n_idx in T.serial(n_num):
                        T.wait_flag("mte3", "v", 1)
                        T.tile.broadcast(v1_sy_broadcast, v1_sy_sum)
                        T.pipe_barrier("v")
                        T.tile.div(
                            v1_sy_storage[Nidx1 + n_idx, 0:n_block],
                            v1_sy_storage[Nidx1 + n_idx, 0:n_block],
                            v1_sy_broadcast[0, :],
                        )
                        T.set_flag("v", "mte3", 1)
                        T.wait_flag("v", "mte3", 1)
                        T.copy(
                            v1_sy_storage[Nidx1 + n_idx, 0:n_block],
                            SoftmaxI[bz_task, s1_id,
                                     n_idx * n_block:(n_idx + 1) * n_block],
                        )
                        T.set_flag("mte3", "v", 1)

                    T.wait_flag("mte3", "v", 1)

            T.set_cross_flag("MTE3", V1_PSY_TO_AIC_FLAG)
            T.wait_cross_flag(V1_PSY_RELAY_FLAG)

    @T.macro
    def relay_vector1_psy():
        with T.Scope("C"):
            T.wait_cross_flag(V1_PSY_TO_AIC_FLAG)
            T.set_cross_flag("FIX", V1_PSY_RELAY_FLAG)

    @T.macro
    def vector1_backward_task(
        ValidKMask,
        Weights,
        bmm2Res,
        PReduced,
        SoftmaxI,
        dWeightOut,
        reluGradRes,
        LossPartial,
        bz_task,
        s1_begin_task,
        s1_end_task,
        task_slot,
        vector_id,
    ):
        """Translate VectorDwDqDk + VectorLoss for one complete task.

        The paired AIVs own disjoint Nidx1 halves.  Both consume the P/SY
        workspaces only after the Cube relay observes both producers.  The
        ReLU mask uses an out-of-place Select to avoid source/destination
        aliasing while preserving one vector operation per tile.
        """
        with T.Scope("V"):
            v1_bw_weight_half = T.alloc_ub(rows_per_aiv, input_dtype)
            v1_bw_weight = T.alloc_ub(rows_per_aiv, calc_dtype)
            v1_bw_weight_2d = T.alloc_ub(
                (rows_per_aiv, bw_col_block), calc_dtype
            )
            v1_bw_p = T.alloc_ub(bw_col_block, calc_dtype)
            v1_bw_sy = T.alloc_ub(bw_col_block, calc_dtype)
            v1_bw_diff = T.alloc_ub(bw_col_block, calc_dtype)
            v1_bw_valid = T.alloc_ub(bw_col_block, calc_dtype)
            v1_bw_diff_2d = T.alloc_ub(
                (rows_per_aiv, bw_col_block), calc_dtype
            )
            v1_bw_bmm2 = T.alloc_ub(
                (rows_per_aiv, bw_col_block), calc_dtype
            )
            v1_bw_mask = T.alloc_ub(
                rows_per_aiv * bw_col_block // 16, calc_dtype
            )
            # T.macro symbol collection does not safely propagate allocations
            # through a Python conditional. Keep these four aligned 4x64 mask
            # slots defined for every specialization; only the 16-row path
            # consumes them.
            v1_bw_mask0 = T.alloc_ub(16, calc_dtype)
            v1_bw_mask1 = T.alloc_ub(16, calc_dtype)
            v1_bw_mask2 = T.alloc_ub(16, calc_dtype)
            v1_bw_mask3 = T.alloc_ub(16, calc_dtype)
            v1_bw_rg = T.alloc_ub(
                (rows_per_aiv, bw_col_block), calc_dtype
            )
            v1_bw_rg_half = T.alloc_ub(
                (rows_per_aiv, bw_col_block), input_dtype
            )
            v1_bw_dw = T.alloc_ub(rows_per_aiv, calc_dtype)
            v1_bw_dw_accum = T.alloc_ub(rows_per_aiv, calc_dtype)
            v1_bw_dw_half = T.alloc_ub(rows_per_aiv, input_dtype)
            v1_bw_eps = T.alloc_ub(bw_col_block, calc_dtype)
            v1_bw_log_p = T.alloc_ub(bw_col_block, calc_dtype)
            v1_bw_log_sy = T.alloc_ub(bw_col_block, calc_dtype)
            v1_bw_loss_2d = T.alloc_ub((1, bw_col_block), calc_dtype)
            v1_bw_loss_sum = T.alloc_ub((1,), calc_dtype)
            v1_bw_loss_accum = T.alloc_ub((1,), calc_dtype)
            v1_bw_row_tmp = T.alloc_ub((rows_per_aiv, 8), calc_dtype)

            T.tile.fill(v1_bw_eps, 1e-8)
            # The FP16 reluGrad store shares its physical UB region with the
            # next FP32 BMM2 load, so MTE2 waits for MTE3 directly.
            T.set_flag("mte3", "mte2", 2)
            T.set_flag("mte3", "v", 3)

            for s1_id in T.serial(s1_begin_task, s1_end_task):
                g_start = vector_id * rows_per_aiv
                T.tile.fill(v1_bw_dw_accum, 0.0)
                T.tile.fill(v1_bw_loss_accum, 0.0)
                T.copy(
                    Weights[bz_task, s1_id,
                            g_start:g_start + rows_per_aiv],
                    v1_bw_weight_half,
                )
                T.set_flag("mte2", "v", 2)
                T.wait_flag("mte2", "v", 2)
                T.tile.cast(
                    v1_bw_weight,
                    v1_bw_weight_half,
                    "CAST_NONE",
                    rows_per_aiv,
                )

                if rows_per_aiv == 4:
                    # BRCB-based row expansion requires a multiple of 8 rows.
                    # Broadcast each runtime scalar for the Nidx1=8 path only.
                    for bw_row in T.serial(rows_per_aiv):
                        T.tile.broadcast(
                            v1_bw_weight_2d[
                                bw_row, 0:bw_col_block
                            ],
                            v1_bw_weight[bw_row:bw_row + 1],
                        )
                else:
                    T.tile.fill(v1_bw_weight_2d, 1.0)
                    T.pipe_barrier("v")
                    T.tile.row_expand_mul_experiment(
                        v1_bw_weight_2d,
                        v1_bw_weight_2d,
                        v1_bw_weight,
                        v1_bw_row_tmp,
                    )
                T.pipe_barrier("v")
                T.set_flag("v", "mte2", 3)
                for n_idx in T.serial(n_num):
                    for bw_col_id in T.serial(bw_col_count):
                        bw_col_start = (
                            n_idx * n_block + bw_col_id * bw_col_block
                        )
                        bw_col_end = bw_col_start + bw_col_block

                        T.wait_flag("v", "mte2", 3)
                        T.wait_flag("mte3", "mte2", 2)
                        T.copy(
                            PReduced[bz_task, s1_id,
                                     bw_col_start:bw_col_end],
                            v1_bw_p,
                        )
                        T.copy(
                            SoftmaxI[bz_task, s1_id,
                                     bw_col_start:bw_col_end],
                            v1_bw_sy,
                        )
                        T.copy(
                            bmm2Res[
                                bz_task, s1_id,
                                g_start:g_start + rows_per_aiv,
                                bw_col_start:bw_col_end,
                            ],
                            v1_bw_bmm2,
                        )
                        T.copy(
                            ValidKMask[bz_task, s1_id, bw_col_start:bw_col_end],
                            v1_bw_valid,
                        )
                        T.set_flag("mte2", "v", 3)
                        T.wait_flag("mte2", "v", 3)

                        T.tile.sub(
                            v1_bw_diff,
                            v1_bw_sy,
                            v1_bw_p,
                        )
                        T.pipe_barrier("v")
                        T.tile.broadcast(
                            v1_bw_diff_2d,
                            v1_bw_diff,
                        )

                        if rows_per_aiv == 16:
                            # Compare each statically aligned 4x64 group before
                            # bmm2 is reused for dWeight.  Keep dWeight itself
                            # as its proven single 16x64 reduction.
                            T.tile.compare(
                                v1_bw_mask0,
                                v1_bw_bmm2[0:4, 0:bw_col_block],
                                0.0,
                                "GT",
                            )
                            T.pipe_barrier("v")
                            T.tile.compare(
                                v1_bw_mask1,
                                v1_bw_bmm2[4:8, 0:bw_col_block],
                                0.0,
                                "GT",
                            )
                            T.pipe_barrier("v")
                            T.tile.compare(
                                v1_bw_mask2,
                                v1_bw_bmm2[8:12, 0:bw_col_block],
                                0.0,
                                "GT",
                            )
                            T.pipe_barrier("v")
                            T.tile.compare(
                                v1_bw_mask3,
                                v1_bw_bmm2[12:16, 0:bw_col_block],
                                0.0,
                                "GT",
                            )
                            T.pipe_barrier("v")
                            T.tile.mul(
                                v1_bw_bmm2,
                                v1_bw_bmm2,
                                v1_bw_diff_2d,
                            )
                            T.pipe_barrier("v")
                            T.reduce_sum(
                                v1_bw_bmm2,
                                v1_bw_dw,
                                1,
                            )
                            T.pipe_barrier("v")
                            T.tile.add(v1_bw_dw_accum, v1_bw_dw_accum, v1_bw_dw)
                            T.tile.mul(
                                v1_bw_rg,
                                v1_bw_weight_2d,
                                v1_bw_diff_2d,
                            )
                            T.pipe_barrier("v")
                            T.tile.select(
                                v1_bw_bmm2[0:4, 0:bw_col_block],
                                v1_bw_mask0,
                                v1_bw_rg[0:4, 0:bw_col_block],
                                0.0,
                                "VSEL_TENSOR_SCALAR_MODE",
                            )
                            T.pipe_barrier("v")
                            T.tile.select(
                                v1_bw_bmm2[4:8, 0:bw_col_block],
                                v1_bw_mask1,
                                v1_bw_rg[4:8, 0:bw_col_block],
                                0.0,
                                "VSEL_TENSOR_SCALAR_MODE",
                            )
                            T.pipe_barrier("v")
                            T.tile.select(
                                v1_bw_bmm2[8:12, 0:bw_col_block],
                                v1_bw_mask2,
                                v1_bw_rg[8:12, 0:bw_col_block],
                                0.0,
                                "VSEL_TENSOR_SCALAR_MODE",
                            )
                            T.pipe_barrier("v")
                            T.tile.select(
                                v1_bw_bmm2[12:16, 0:bw_col_block],
                                v1_bw_mask3,
                                v1_bw_rg[12:16, 0:bw_col_block],
                                0.0,
                                "VSEL_TENSOR_SCALAR_MODE",
                            )
                            T.pipe_barrier("v")
                            T.tile.cast(
                                v1_bw_rg_half,
                                v1_bw_bmm2,
                                "CAST_RINT",
                                rows_per_aiv * bw_col_block,
                            )
                            T.pipe_barrier("v")
                        else:
                            T.tile.compare(
                                v1_bw_mask,
                                v1_bw_bmm2,
                                0.0,
                                "GT",
                            )
                            T.pipe_barrier("v")
                            # dWeight consumes the unweighted diff tile and is
                            # accumulated across both 64-column subtiles.
                            T.tile.mul(
                                v1_bw_bmm2,
                                v1_bw_bmm2,
                                v1_bw_diff_2d,
                            )
                            T.pipe_barrier("v")
                            T.reduce_sum(
                                v1_bw_bmm2,
                                v1_bw_dw,
                                1,
                            )
                            T.pipe_barrier("v")
                            T.tile.add(v1_bw_dw_accum, v1_bw_dw_accum, v1_bw_dw)

                            T.tile.mul(
                                v1_bw_rg,
                                v1_bw_weight_2d,
                                v1_bw_diff_2d,
                            )
                            T.pipe_barrier("v")
                            T.tile.select(
                                v1_bw_bmm2,
                                v1_bw_mask,
                                v1_bw_rg,
                                0.0,
                                "VSEL_TENSOR_SCALAR_MODE",
                            )
                            T.pipe_barrier("v")
                            T.tile.cast(
                                v1_bw_rg_half,
                                v1_bw_bmm2,
                                "CAST_RINT",
                                rows_per_aiv * bw_col_block,
                            )
                            T.pipe_barrier("v")
                        T.set_flag("v", "mte3", 2)
                        T.wait_flag("v", "mte3", 2)
                        T.pipe_barrier("mte3")
                        T.copy(
                            v1_bw_rg_half,
                            reluGradRes[bz_task, s1_id,
                                        g_start:g_start + rows_per_aiv,
                                        bw_col_start:bw_col_end],
                        )
                        T.set_flag("mte3", "mte2", 2)

                        # Match VectorLoss: split outer K tiles across the AIV
                        # pair, while the owner consumes both physical subtiles.
                        if n_idx % 2 == vector_id:
                            T.tile.max(
                                v1_bw_p,
                                v1_bw_p,
                                v1_bw_eps,
                            )
                            T.tile.max(
                                v1_bw_sy,
                                v1_bw_sy,
                                v1_bw_eps,
                            )
                            T.pipe_barrier("v")
                            T.tile.ln(
                                v1_bw_log_p,
                                v1_bw_p,
                            )
                            T.tile.ln(
                                v1_bw_log_sy,
                                v1_bw_sy,
                            )
                            T.pipe_barrier("v")
                            T.tile.sub(
                                v1_bw_loss_2d[0, :],
                                v1_bw_log_p,
                                v1_bw_log_sy,
                            )
                            T.pipe_barrier("v")
                            T.tile.mul(
                                v1_bw_loss_2d[0, :],
                                v1_bw_loss_2d[0, :],
                                v1_bw_p,
                            )
                            T.pipe_barrier("v")
                            T.tile.mul(
                                v1_bw_loss_2d[0, :],
                                v1_bw_loss_2d[0, :],
                                v1_bw_valid,
                            )
                            T.pipe_barrier("v")
                            T.reduce_sum(
                                v1_bw_loss_2d, v1_bw_loss_sum, 1
                            )
                            T.pipe_barrier("v")
                            T.tile.add(
                                v1_bw_loss_accum,
                                v1_bw_loss_accum,
                                v1_bw_loss_sum,
                            )
                        T.set_flag("v", "mte2", 3)

                T.wait_flag("v", "mte2", 3)
                T.pipe_barrier("v")
                T.wait_flag("mte3", "v", 3)
                T.tile.cast(
                    v1_bw_dw_half,
                    v1_bw_dw_accum,
                    "CAST_RINT",
                    rows_per_aiv,
                )
                T.set_flag("v", "mte3", 3)
                T.wait_flag("v", "mte3", 3)
                T.copy(
                    v1_bw_dw_half,
                    dWeightOut[bz_task, s1_id,
                               g_start:g_start + rows_per_aiv],
                )
                T.copy(
                    v1_bw_loss_accum,
                    LossPartial[
                        ((bz_task * S1 + s1_id) * 2 + vector_id)
                        * loss_partial_stride:
                        ((bz_task * S1 + s1_id) * 2 + vector_id)
                        * loss_partial_stride + 1
                    ],
                )
                T.set_flag("mte3", "v", 3)

            T.wait_flag("mte3", "mte2", 2)
            T.wait_flag("mte3", "v", 3)
            if task_slot == 0:
                T.set_cross_flag("MTE3", V1_TO_C2_FLAGS[0])
            else:
                T.set_cross_flag("MTE3", V1_TO_C2_FLAGS[1])

    @T.macro
    def cube2_task(
        QueryIndex,
        GatherKeyIndex_ws,
        reluGradRes,
        C2ZeroPad,
        dQueryIndexAccum,
        dk_matmul_partial,
        bz_task,
        s1_begin_task,
        s1_end_task,
        task_slot,
    ):
        """Compute dKey partials and dQuery with explicit MMA pipelines."""
        with T.Scope("C"):
            c2_rg_l1 = T.alloc_L1((c2_dk_k, n_block), input_dtype)
            c2_qi_l1 = T.alloc_L1((c2_dk_k, DQueryIndex), input_dtype)
            c2_gki_l1 = T.alloc_L1((n_block, DQueryIndex), input_dtype)
            c2_dk_rg_l0a = T.alloc_L0A((n_block, c2_dk_k), input_dtype)
            c2_dk_qi_l0b = T.alloc_L0B((c2_dk_k, DQueryIndex), input_dtype)
            c2_dk_l0c = T.alloc_L0C((n_block, DQueryIndex), calc_dtype)
            c2_dq_rg_l0a = T.alloc_L0A((Nidx1, n_block), input_dtype)
            c2_dq_gki_l0b = T.alloc_L0B((n_block, DQueryIndex), input_dtype)
            c2_dq_l0c = T.alloc_L0C((Nidx1, DQueryIndex), calc_dtype)

            T.copy(
                QueryIndex[bz_task, s1_begin_task, 0:Nidx1, 0:DQueryIndex],
                c2_qi_l1[0:Nidx1, 0:DQueryIndex],
            )
            if Nidx1 == 8:
                # BF16 Cube requires a 16-wide reduction dimension.  Keep the
                # logical eight rows intact and zero the physical tail once;
                # later RG loads overwrite only the valid first eight rows.
                T.copy(C2ZeroPad, c2_qi_l1[8:16, 0:DQueryIndex])
                T.copy(C2ZeroPad, c2_rg_l1[8:16, 0:n_block])
            T.set_flag("mte2", "mte1", 6)

            if task_slot == 0:
                T.wait_cross_flag(V1_TO_C2_FLAGS[0])
            else:
                T.wait_cross_flag(V1_TO_C2_FLAGS[1])

            T.set_flag("mte1", "mte2", 2)
            T.set_flag("m", "mte1", 2)
            T.set_flag("fix", "m", 2)
            T.set_flag("m", "mte1", 3)
            T.set_flag("fix", "m", 3)

            T.wait_flag("mte2", "mte1", 6)
            for s1_id in T.serial(s1_begin_task, s1_end_task):
                for n_idx in T.serial(n_num):
                    T.wait_flag("mte1", "mte2", 2)
                    T.copy(
                        reluGradRes[bz_task, s1_id, 0:Nidx1,
                                    n_idx * n_block:(n_idx + 1) * n_block],
                        c2_rg_l1[0:Nidx1, 0:n_block],
                    )
                    T.copy(
                        GatherKeyIndex_ws[bz_task, s1_id,
                                          n_idx * n_block:(n_idx + 1) * n_block,
                                          0:DQueryIndex],
                        c2_gki_l1,
                    )
                    T.set_flag("mte2", "mte1", 2)

                    T.wait_flag("m", "mte1", 2)
                    T.wait_flag("m", "mte1", 3)
                    T.wait_flag("mte2", "mte1", 2)
                    T.copy(c2_rg_l1, c2_dk_rg_l0a, transpose=True)
                    T.copy(c2_qi_l1, c2_dk_qi_l0b)
                    T.copy(
                        c2_rg_l1[0:Nidx1, 0:n_block],
                        c2_dq_rg_l0a,
                    )
                    T.copy(c2_gki_l1, c2_dq_gki_l0b)
                    T.set_flag("mte1", "mte2", 2)
                    T.set_flag("mte1", "m", 2)
                    T.set_flag("mte1", "m", 3)

                    T.wait_flag("mte1", "m", 2)
                    T.wait_flag("fix", "m", 2)
                    T.mma(c2_dk_rg_l0a, c2_dk_qi_l0b, c2_dk_l0c, init=True)
                    T.set_flag("m", "mte1", 2)
                    T.set_flag("m", "fix", 2)
                    T.wait_flag("m", "fix", 2)
                    T.copy(
                        c2_dk_l0c,
                        dk_matmul_partial[bz_task, s1_id,
                                          n_idx * n_block:(n_idx + 1) * n_block,
                                          0:DQueryIndex],
                    )
                    T.set_flag("fix", "m", 2)

                    T.wait_flag("mte1", "m", 3)
                    T.wait_flag("fix", "m", 3)
                    T.mma(c2_dq_rg_l0a, c2_dq_gki_l0b, c2_dq_l0c, init=(n_idx == 0))
                    T.set_flag("m", "mte1", 3)
                    if n_idx + 1 < n_num:
                        T.set_flag("fix", "m", 3)

                T.set_flag("m", "fix", 3)
                T.wait_flag("m", "fix", 3)
                T.copy(
                    c2_dq_l0c,
                    dQueryIndexAccum[bz_task, s1_id, 0:Nidx1, 0:DQueryIndex],
                )
                T.set_flag("fix", "m", 3)

            T.wait_flag("mte1", "mte2", 2)
            T.wait_flag("m", "mte1", 2)
            T.wait_flag("fix", "m", 2)
            T.wait_flag("m", "mte1", 3)
            T.wait_flag("fix", "m", 3)

    @T.macro
    def clear_dkey_accumulator(dKeyIndexAccum, core_id, vector_id):
        """Clear the FP32 scatter target across all AIVs inside the kernel."""
        with T.Scope("V"):
            v2_zero_batch = T.alloc_ub(
                (dkey_clear_batch, DQueryIndex), calc_dtype
            )
            aiv_id = core_id * 2 + vector_id
            clear_base_batches = dkey_clear_total_batches // total_aiv_num
            clear_tail_aivs = dkey_clear_total_batches % total_aiv_num
            clear_batch_count = clear_base_batches + T.if_then_else(
                aiv_id < clear_tail_aivs, 1, 0
            )
            clear_batch_start = (
                aiv_id * clear_base_batches + T.min(aiv_id, clear_tail_aivs)
            )

            T.tile.fill(v2_zero_batch, 0.0)
            T.set_flag("v", "mte3", 5)
            T.wait_flag("v", "mte3", 5)
            for clear_iter in T.serial(clear_batch_count):
                clear_row = (
                    clear_batch_start + clear_iter
                ) * dkey_clear_batch
                T.copy(
                    v2_zero_batch,
                    dKeyIndexAccum[
                        clear_row:clear_row + dkey_clear_batch,
                        0:DQueryIndex,
                    ],
                )
            T.set_flag("mte3", "v", 5)
            T.wait_flag("mte3", "v", 5)

    @T.macro
    def vector2_scatter_all_tasks(
        SparseIndices,
        KRealSize,
        dk_matmul_partial,
        dKeyIndexAccum,
        core_id,
        vector_id,
    ):
        """Batch GM-to-UB and ping-pong scatter over every task using all cores."""
        with T.Scope("V"):
            v2_dk_batch_ping = T.alloc_ub(
                (v2_scatter_batch, DQueryIndex), calc_dtype
            )
            v2_dk_batch_pong = T.alloc_ub(
                (v2_scatter_batch, DQueryIndex), calc_dtype
            )
            core_batch_count = v2_base_batches_per_core + T.if_then_else(
                core_id < v2_tail_batch_cores, 1, 0
            )
            core_batch_start = (
                core_id * v2_base_batches_per_core
                + T.min(core_id, v2_tail_batch_cores)
            )
            first_vector_batches = (core_batch_count + 1) // 2
            vector_batch_count = (
                first_vector_batches
                - vector_id * (2 * first_vector_batches - core_batch_count)
            )
            vector_batch_start = (
                core_batch_start + vector_id * first_vector_batches
            )
            aiv_k_start = vector_batch_start * v2_scatter_batch

            T.set_flag("mte3", "mte2", 6)
            T.set_flag("mte3", "mte2", 7)
            for task_id in T.serial(task_count):
                bz_task = task_id // S1
                s1_task = task_id % S1

                if vector_batch_count > 0:
                    T.wait_flag("mte3", "mte2", 6)
                    T.copy(
                        dk_matmul_partial[
                            bz_task,
                            s1_task,
                            aiv_k_start:aiv_k_start + v2_scatter_batch,
                            0:DQueryIndex,
                        ],
                        v2_dk_batch_ping,
                    )
                    T.set_flag("mte2", "mte3", 6)

                    for batch_id in T.serial(vector_batch_count):
                        current_slot = batch_id % 2
                        next_slot = (batch_id + 1) % 2
                        if batch_id + 1 < vector_batch_count:
                            next_k_start = (
                                aiv_k_start
                                + (batch_id + 1) * v2_scatter_batch
                            )
                            T.wait_flag("mte3", "mte2", next_slot + 6)
                            if next_slot == 0:
                                T.copy(
                                    dk_matmul_partial[
                                        bz_task,
                                        s1_task,
                                        next_k_start:next_k_start + v2_scatter_batch,
                                        0:DQueryIndex,
                                    ],
                                    v2_dk_batch_ping,
                                )
                            else:
                                T.copy(
                                    dk_matmul_partial[
                                        bz_task,
                                        s1_task,
                                        next_k_start:next_k_start + v2_scatter_batch,
                                        0:DQueryIndex,
                                    ],
                                    v2_dk_batch_pong,
                                )
                            T.set_flag("mte2", "mte3", next_slot + 6)

                        T.wait_flag("mte2", "mte3", current_slot + 6)
                        for pair_idx in T.serial(v2_scatter_batch // 2):
                            row_idx = pair_idx * 2
                            k_idx = (
                                aiv_k_start
                                + batch_id * v2_scatter_batch
                                + row_idx
                            )
                            sparse_idx0 = SparseIndices[
                                bz_task, s1_task, 0, k_idx
                            ]
                            sparse_idx1 = SparseIndices[
                                bz_task, s1_task, 0, k_idx + 1
                            ]
                            valid_k_size = KRealSize[bz_task, s1_task]
                            if k_idx < valid_k_size:
                                if sparse_idx0 >= 0:
                                    if current_slot == 0:
                                        T.tile.atomic_add(
                                            dKeyIndexAccum[
                                                bz_task * S2 + sparse_idx0, 0
                                            ],
                                            v2_dk_batch_ping[
                                                row_idx:row_idx + 1, 0:DQueryIndex
                                            ],
                                        )
                                    else:
                                        T.tile.atomic_add(
                                            dKeyIndexAccum[
                                                bz_task * S2 + sparse_idx0, 0
                                            ],
                                            v2_dk_batch_pong[
                                                row_idx:row_idx + 1, 0:DQueryIndex
                                            ],
                                        )
                            if k_idx + 1 < valid_k_size:
                                if sparse_idx1 >= 0:
                                    if current_slot == 0:
                                        T.tile.atomic_add(
                                            dKeyIndexAccum[
                                                bz_task * S2 + sparse_idx1, 0
                                            ],
                                            v2_dk_batch_ping[
                                                row_idx + 1:row_idx + 2,
                                                0:DQueryIndex,
                                            ],
                                        )
                                    else:
                                        T.tile.atomic_add(
                                            dKeyIndexAccum[
                                                bz_task * S2 + sparse_idx1, 0
                                            ],
                                            v2_dk_batch_pong[
                                                row_idx + 1:row_idx + 2,
                                                0:DQueryIndex,
                                            ],
                                        )
                        T.set_flag("mte3", "mte2", current_slot + 6)

            T.wait_flag("mte3", "mte2", 6)
            T.wait_flag("mte3", "mte2", 7)

    @T.macro
    def finalize_vector_outputs(
        dQueryIndexAccum,
        dKeyIndexAccum,
        dQueryIndexOut,
        dKeyIndexOut,
        LossPartial,
        LossOut,
        core_id,
        vector_id,
    ):
        """Distribute output casts and reduce all per-AIV loss partials."""
        with T.Scope("V"):
            v2_finalize_dq_fp32 = T.alloc_ub((1, DQueryIndex), calc_dtype)
            v2_finalize_dq_half = T.alloc_ub((1, DQueryIndex), input_dtype)
            v2_finalize_dk_fp32 = T.alloc_ub(
                (2, dkey_finalize_batch, DQueryIndex), calc_dtype
            )
            v2_finalize_dk_half = T.alloc_ub(
                (2, dkey_finalize_batch, DQueryIndex), input_dtype
            )
            T.set_flag("v", "mte2", 4)
            T.set_flag("mte3", "v", 7)

            # Only the AIV that owns the task casts dQuery.  Distributing this
            # tiny output over every core would introduce unnecessary scalar
            # work and event setup on otherwise scatter-focused cores.
            for dq_task_iter in T.serial(tasks_per_core):
                dq_task = dq_task_iter * launch_core_num + core_id
                if dq_task < task_count:
                    dq_bz = dq_task // S1
                    dq_s1 = dq_task % S1
                    for dq_group_local in T.serial(Nidx1 // 2):
                        dq_group = dq_group_local * 2 + vector_id
                        T.wait_flag("v", "mte2", 4)
                        T.copy(
                            dQueryIndexAccum[
                                dq_bz,
                                dq_s1,
                                dq_group:dq_group + 1,
                                0:DQueryIndex,
                            ],
                            v2_finalize_dq_fp32,
                        )
                        T.set_flag("mte2", "v", 5)
                        T.wait_flag("mte2", "v", 5)
                        T.wait_flag("mte3", "v", 7)
                        T.tile.cast(
                            v2_finalize_dq_half,
                            v2_finalize_dq_fp32,
                            "CAST_RINT",
                            DQueryIndex,
                        )
                        T.set_flag("v", "mte2", 4)
                        T.set_flag("v", "mte3", 5)
                        T.wait_flag("v", "mte3", 5)
                        T.copy(
                            v2_finalize_dq_half,
                            dQueryIndexOut[
                                dq_bz,
                                dq_s1,
                                dq_group:dq_group + 1,
                                0:DQueryIndex,
                            ],
                        )
                        T.set_flag("mte3", "v", 7)

            T.wait_flag("v", "mte2", 4)
            T.wait_flag("mte3", "v", 7)

            aiv_id = core_id * 2 + vector_id
            finalize_base_batches = dkey_finalize_total_batches // total_aiv_num
            finalize_tail_aivs = dkey_finalize_total_batches % total_aiv_num
            finalize_batch_count = finalize_base_batches + T.if_then_else(
                aiv_id < finalize_tail_aivs, 1, 0
            )
            finalize_batch_start = (
                aiv_id * finalize_base_batches
                + T.min(aiv_id, finalize_tail_aivs)
            )

            T.set_flag("v", "mte2", 5)
            T.set_flag("v", "mte2", 6)
            T.set_flag("mte3", "v", 5)
            T.set_flag("mte3", "v", 6)
            for finalize_iter in T.serial(finalize_batch_count):
                finalize_slot = finalize_iter % 2
                finalize_row = (
                    finalize_batch_start + finalize_iter
                ) * dkey_finalize_batch

                T.wait_flag("v", "mte2", finalize_slot + 5)
                T.copy(
                    dKeyIndexAccum[
                        finalize_row:finalize_row + dkey_finalize_batch,
                        0:DQueryIndex,
                    ],
                    v2_finalize_dk_fp32[finalize_slot, :, :],
                )
                T.set_flag("mte2", "v", finalize_slot + 5)
                T.wait_flag("mte2", "v", finalize_slot + 5)
                T.wait_flag("mte3", "v", finalize_slot + 5)
                T.tile.cast(
                    v2_finalize_dk_half[finalize_slot, :, :],
                    v2_finalize_dk_fp32[finalize_slot, :, :],
                    "CAST_RINT",
                    dkey_finalize_batch * DQueryIndex,
                )
                T.set_flag("v", "mte2", finalize_slot + 5)
                T.set_flag("v", "mte3", finalize_slot + 5)
                T.wait_flag("v", "mte3", finalize_slot + 5)
                T.copy(
                    v2_finalize_dk_half[finalize_slot, :, :],
                    dKeyIndexOut[
                        finalize_row:finalize_row + dkey_finalize_batch,
                        0:DQueryIndex,
                    ],
                )
                T.set_flag("mte3", "v", finalize_slot + 5)

            T.wait_flag("v", "mte2", 5)
            T.wait_flag("v", "mte2", 6)
            T.wait_flag("mte3", "v", 5)
            T.wait_flag("mte3", "v", 6)

            if core_id == 0:
                if vector_id == 0:
                    v2_loss_chunk = T.alloc_ub(
                        (1, loss_reduce_chunk), calc_dtype
                    )
                    v2_loss_chunk_sum = T.alloc_ub((1,), calc_dtype)
                    v2_loss_total = T.alloc_ub((1,), calc_dtype)
                    T.tile.fill(v2_loss_total, 0.0)
                    for loss_chunk_id in T.serial(loss_full_chunk_count):
                        loss_chunk_start = loss_chunk_id * loss_reduce_chunk
                        T.set_flag("v", "mte2", 4)
                        T.wait_flag("v", "mte2", 4)
                        T.copy(
                            LossPartial[
                                loss_chunk_start:loss_chunk_start + loss_reduce_chunk
                            ],
                            v2_loss_chunk[0, :],
                        )
                        T.set_flag("mte2", "v", 4)
                        T.wait_flag("mte2", "v", 4)
                        T.reduce_sum(
                            v2_loss_chunk,
                            v2_loss_chunk_sum,
                            1,
                        )
                        T.pipe_barrier("v")
                        T.tile.add(
                            v2_loss_total,
                            v2_loss_total,
                            v2_loss_chunk_sum,
                        )
                    T.set_flag("v", "mte3", 4)
                    T.wait_flag("v", "mte3", 4)
                    T.copy(v2_loss_total, LossOut[0:1])
                    T.set_flag("mte3", "v", 4)
                    T.wait_flag("mte3", "v", 4)

    @T.prim_func
    def main(
        # === SPECIAL: int32 indices must be first to avoid dtype inference issues ===
        SparseIndices: T.Tensor((B, S1, Nidx2, K), "int32"),            # 0
        # === INPUTS (1-11) ===
        Query: T.Tensor((B, S1, N1, DQuery), input_dtype),              # 1
        QueryRope: T.Tensor((B, S1, N1, d_rope), input_dtype),          # 2
        Key: T.Tensor((B, S2, N2, DQuery), input_dtype),                # 3
        KeyRope: T.Tensor((B, S2, N2, d_rope), input_dtype),            # 4
        QueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), input_dtype), # 5
        KeyIndex: T.Tensor((B, S2, Nidx2, DQueryIndex), input_dtype),   # 6
        Weights: T.Tensor((B, S1, Nidx1), input_dtype),                 # 7
        SoftmaxMax: T.Tensor((B, N2, S1, G), calc_dtype),               # 8
        SoftmaxSum: T.Tensor((B, N2, S1, G), calc_dtype),               # 9
        dQueryIndexAccum: T.Tensor((B, S1, Nidx1, DQueryIndex), calc_dtype), # 10 (input, host-zeroed: atomic_add target)
        # === INPUT (11) — dKeyIndex fp32 累加器，host 清零后传入（atomic_add target，照搬 dQueryIndexAccum 模式）===
        dKeyIndex: T.Tensor((B, S2, Nidx2, DQueryIndex), calc_dtype),   # 11 (input, host-zeroed)
        # === WORKSPACES (12-21) ===
        GatherKey_ws: T.Tensor((B, S1, K, DQuery), input_dtype),        # 12
        GatherKeyRope_ws: T.Tensor((B, S1, K, d_rope), input_dtype),    # 13
        GatherKeyIndex_ws: T.Tensor((B, S1, K, DQueryIndex), input_dtype), # 14
        bmm1Res: T.Tensor((B, S1, N1, K), calc_dtype),                  # 15
        bmm2Res: T.Tensor((B, S1, Nidx1, K), calc_dtype),               # 16
        PReduced: T.Tensor((B, S1, K), calc_dtype),                     # 17
        ReduceSumY: T.Tensor((B, S1, K), calc_dtype),                   # 18
        SoftmaxI: T.Tensor((B, S1, K), calc_dtype),                     # 19
        dk_matmul_partial: T.Tensor((B, S1, K, DQueryIndex), calc_dtype), # 20 (workspace: reluGrad^T @ QueryIndex)
        # === OUTPUTS (21-24) ===
        dQueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), input_dtype), # 21
        dWeight_partial: T.Tensor((B, S1, n_num, Nidx1), input_dtype),  # 22
        reluGradRes: T.Tensor((B, S1, Nidx1, K), input_dtype),          # 23
        LossOut: T.Tensor((B, S1, n_num), calc_dtype),             # 24
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
            # 手动同步（auto-sync 关）：chunk 级 double buffer，MTE2 fill ∥ MTE3 drain
            #   预取下一 chunk 的 fill 与本 chunk 的 drain 并发，隐藏 burst 写延迟
            # 标量 idx 从 GM 读，依赖 GetValue 同步语义（watch-point）
            # ============================================================
            num_chunks = n_block // GATHER_CHUNK_SIZE
            with T.Scope("V"):
                # Double-buffered chunked gather：MTE2 fill ∥ MTE3 drain（ping-pong）
                # 照抄官方 examples/elementwise/elementwise_add_pipeline.py 的 WAR-safe 结构：
                #   MTE3_MTE2 = "slot 可写" WAR 事件（预置两个 slot，末尾 clear 排空）
                #   MTE2_MTE3 = "slot 已填" RAW 事件（drain 等 fill）
                # 数据 copy 全部内联（官方范例同样内联，prim_func 内不能用未装饰的嵌套函数）
                # 3D 前导索引 buf[slot, i, :] 由官方范例验证可用。
                key_chunk_ub = T.alloc_ub((2, GATHER_CHUNK_SIZE, DQuery), input_dtype)
                ki_chunk_ub = T.alloc_ub((2, GATHER_CHUNK_SIZE, DQueryIndex), input_dtype)
                kr_chunk_ub = T.alloc_ub((2, GATHER_CHUNK_SIZE, max(d_rope, 1)), input_dtype)

                # 零行 buffer：idx < 0 时写入零值，避免未初始化 UB 数据污染 GatherKey_ws
                zero_key_row = T.alloc_ub(DQuery, input_dtype)
                zero_ki_row = T.alloc_ub(DQueryIndex, input_dtype)
                zero_kr_row = T.alloc_ub(max(d_rope, 1), input_dtype)
                T.tile.fill(zero_key_row, 0)
                T.tile.fill(zero_ki_row, 0)
                if d_rope > 0:
                    T.tile.fill(zero_kr_row, 0)

                # 预置：两个 slot 都标记为"已 drain（可 fill）"
                T.set_flag("mte3", "mte2", 0)
                T.set_flag("mte3", "mte2", 1)

                for s1_id in T.serial(s1_start, s1_end):
                    # prologue：fill chunk 0 → slot 0（MTE2 逐行随机读 16 行）
                    # idx >= 0: 有效 → copy Key/KeyIndex/KeyRope
                    # idx <  0: 无效 → zero-fill（避免未初始化 UB 污染 GatherKey_ws）
                    T.wait_flag("mte3", "mte2", 0)
                    for i in T.serial(GATHER_CHUNK_SIZE):
                        k_id = k_start + i
                        idx = SparseIndices[bz, s1_id, 0, k_id]
                        if idx >= 0:
                            T.copy(Key[bz, idx, 0, 0:DQuery], key_chunk_ub[0, i, 0:DQuery])
                            T.copy(KeyIndex[bz, idx, 0, 0:DQueryIndex], ki_chunk_ub[0, i, 0:DQueryIndex])
                            if d_rope > 0:
                                T.copy(KeyRope[bz, idx, 0, 0:d_rope], kr_chunk_ub[0, i, 0:d_rope])
                        if idx < 0:
                            T.copy(zero_key_row, key_chunk_ub[0, i, 0:DQuery])
                            T.copy(zero_ki_row, ki_chunk_ub[0, i, 0:DQueryIndex])
                            if d_rope > 0:
                                T.copy(zero_kr_row, kr_chunk_ub[0, i, 0:d_rope])
                    T.set_flag("mte2", "mte3", 0)

                    for chunk_id in T.serial(num_chunks):
                        cur = chunk_id % 2
                        nxt = (chunk_id + 1) % 2
                        # 预取下一 chunk → slot nxt（MTE2，与本 chunk 的 MTE3 drain 并发）
                        if chunk_id + 1 < num_chunks:
                            next_start = k_start + (chunk_id + 1) * GATHER_CHUNK_SIZE
                            T.wait_flag("mte3", "mte2", nxt)   # WAR：slot nxt 上轮已 drain
                            for i in T.serial(GATHER_CHUNK_SIZE):
                                k_id = next_start + i
                                idx = SparseIndices[bz, s1_id, 0, k_id]
                                if idx >= 0:
                                    T.copy(Key[bz, idx, 0, 0:DQuery], key_chunk_ub[nxt, i, 0:DQuery])
                                    T.copy(KeyIndex[bz, idx, 0, 0:DQueryIndex], ki_chunk_ub[nxt, i, 0:DQueryIndex])
                                    if d_rope > 0:
                                        T.copy(KeyRope[bz, idx, 0, 0:d_rope], kr_chunk_ub[nxt, i, 0:d_rope])
                                if idx < 0:
                                    T.copy(zero_key_row, key_chunk_ub[nxt, i, 0:DQuery])
                                    T.copy(zero_ki_row, ki_chunk_ub[nxt, i, 0:DQueryIndex])
                                    if d_rope > 0:
                                        T.copy(zero_kr_row, kr_chunk_ub[nxt, i, 0:d_rope])
                            T.set_flag("mte2", "mte3", nxt)    # RAW：slot nxt 已 fill
                        # drain 本 chunk（MTE3 整块 burst 写出）
                        T.wait_flag("mte2", "mte3", cur)       # RAW：slot cur 已 fill
                        cur_start = k_start + chunk_id * GATHER_CHUNK_SIZE
                        T.copy(key_chunk_ub[cur, 0:GATHER_CHUNK_SIZE, 0:DQuery],
                               GatherKey_ws[bz, s1_id, cur_start:cur_start + GATHER_CHUNK_SIZE, 0:DQuery])
                        T.copy(ki_chunk_ub[cur, 0:GATHER_CHUNK_SIZE, 0:DQueryIndex],
                               GatherKeyIndex_ws[bz, s1_id, cur_start:cur_start + GATHER_CHUNK_SIZE, 0:DQueryIndex])
                        if d_rope > 0:
                            T.copy(kr_chunk_ub[cur, 0:GATHER_CHUNK_SIZE, 0:d_rope],
                                   GatherKeyRope_ws[bz, s1_id, cur_start:cur_start + GATHER_CHUNK_SIZE, 0:d_rope])
                        T.set_flag("mte3", "mte2", cur)        # WAR：slot cur 已 drain，可复用

                # clear：排空剩余 2 个 WAR set（保持 event 平衡）
                T.wait_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 1)

                T.set_cross_flag("V", 10)

            # ============================================================
            # C scope 1: Matmul1 (with rope) + Matmul2
            # 手动同步（auto-sync 关，T.Pipelined 不再生成同步）：
            #   单缓冲 serial，每次迭代 mte2→m（RAW 载入→gemm）/ m→mte2（WAR 复用 L1）
            #   K 维累加完成后 m→fix 再 copy_l0c
            # ============================================================
            with T.Scope("C"):
                T.wait_cross_flag(10)

                Q_L1_0 = T.alloc_L1((N1, k_block), input_dtype)
                Q_L1_1 = T.alloc_L1((N1, k_block), input_dtype)
                GK_L1_0 = T.alloc_L1((n_block, k_block), input_dtype)
                GK_L1_1 = T.alloc_L1((n_block, k_block), input_dtype)
                Q_L0A_0 = T.alloc_L0A((N1, k_block), input_dtype)
                Q_L0A_1 = T.alloc_L0A((N1, k_block), input_dtype)
                GK_L0B_0 = T.alloc_L0B((k_block, n_block), input_dtype)
                GK_L0B_1 = T.alloc_L0B((k_block, n_block), input_dtype)
                C_L0C = T.alloc_L0C((N1, n_block), calc_dtype)

                T.set_flag("m", "mte1", 2)
                T.set_flag("m", "mte1", 3)
                T.set_flag("fix", "m", 4)

                for s1_row in T.serial(s1_start, s1_end):
                    # Double-buffer Matmul1 K-loop: overlap copy[d_idx+1] ∥ gemm[d_idx]
                    # WAR-safe ping-pong, slot 0→{Q_L1_0,GK_L1_0}, slot 1→{Q_L1_1,GK_L1_1}
                    # MTE2_M{0,1}: RAW (slot loaded → gemm can read)
                    # M_MTE2{0,1}: WAR (gemm done → slot safe to refill)
                    T.set_flag("m", "mte2", 0)
                    T.set_flag("m", "mte2", 1)

                    # Prologue: load d_idx=0 → slot 0
                    T.wait_flag("m", "mte2", 0)
                    T.copy(Query[bz, s1_row, 0:N1, 0:k_block], Q_L1_0)
                    T.copy(GatherKey_ws[bz, s1_row, k_start:k_end, 0:k_block], GK_L1_0)
                    T.set_flag("mte2", "m", 0)

                    for d_idx in T.serial(d_num_total):
                        cur = d_idx % 2
                        nxt = (d_idx + 1) % 2

                        # Compute current slot
                        T.wait_flag("mte2", "m", cur)
                        if cur == 0:
                            T.wait_flag("m", "mte1", 2)
                            T.copy(Q_L1_0, Q_L0A_0)
                            T.copy(GK_L1_0, GK_L0B_0, transpose=True)
                            T.set_flag("mte1", "m", 2)
                            T.wait_flag("mte1", "m", 2)
                            T.wait_flag("fix", "m", 4)
                            T.mma(Q_L0A_0, GK_L0B_0, C_L0C, init=(d_idx == 0))
                            T.set_flag("m", "mte1", 2)
                            if d_idx + 1 < d_num_total:
                                T.set_flag("fix", "m", 4)
                        if cur == 1:
                            T.wait_flag("m", "mte1", 3)
                            T.copy(Q_L1_1, Q_L0A_1)
                            T.copy(GK_L1_1, GK_L0B_1, transpose=True)
                            T.set_flag("mte1", "m", 3)
                            T.wait_flag("mte1", "m", 3)
                            T.wait_flag("fix", "m", 4)
                            T.mma(Q_L0A_1, GK_L0B_1, C_L0C, init=(d_idx == 0))
                            T.set_flag("m", "mte1", 3)
                            if d_idx + 1 < d_num_total:
                                T.set_flag("fix", "m", 4)
                        T.set_flag("m", "mte2", cur)

                        # Prefetch next iteration (MTE2 load ∥ cube gemm of current)
                        if d_idx + 1 < d_num_total:
                            next_d = d_idx + 1
                            T.wait_flag("m", "mte2", nxt)
                            if next_d < d_num_query:
                                if nxt == 0:
                                    T.copy(Query[bz, s1_row, 0:N1, next_d * k_block:(next_d + 1) * k_block], Q_L1_0)
                                if nxt == 1:
                                    T.copy(Query[bz, s1_row, 0:N1, next_d * k_block:(next_d + 1) * k_block], Q_L1_1)
                            else:
                                rope_d = next_d - d_num_query
                                if nxt == 0:
                                    T.copy(QueryRope[bz, s1_row, 0:N1, rope_d * k_block:(rope_d + 1) * k_block], Q_L1_0)
                                if nxt == 1:
                                    T.copy(QueryRope[bz, s1_row, 0:N1, rope_d * k_block:(rope_d + 1) * k_block], Q_L1_1)
                            if next_d < d_num_query:
                                if nxt == 0:
                                    T.copy(GatherKey_ws[bz, s1_row, k_start:k_end, next_d * k_block:(next_d + 1) * k_block], GK_L1_0)
                                if nxt == 1:
                                    T.copy(GatherKey_ws[bz, s1_row, k_start:k_end, next_d * k_block:(next_d + 1) * k_block], GK_L1_1)
                            else:
                                rope_d = next_d - d_num_query
                                if nxt == 0:
                                    T.copy(GatherKeyRope_ws[bz, s1_row, k_start:k_end, rope_d * k_block:(rope_d + 1) * k_block], GK_L1_0)
                                if nxt == 1:
                                    T.copy(GatherKeyRope_ws[bz, s1_row, k_start:k_end, rope_d * k_block:(rope_d + 1) * k_block], GK_L1_1)
                            T.set_flag("mte2", "m", nxt)

                    # Clear: drain remaining WAR events
                    T.wait_flag("m", "mte2", 0)
                    T.wait_flag("m", "mte2", 1)
                    # M→FIX: copy_l0c waits for all K accumulations done
                    T.set_flag("m", "fix", 4)
                    T.wait_flag("m", "fix", 4)
                    T.copy(C_L0C, bmm1Res[bz, s1_row, 0:N1, n_idx_k * n_block:(n_idx_k + 1) * n_block])
                    T.set_flag("fix", "m", 4)

                T.wait_flag("m", "mte1", 2)
                T.wait_flag("m", "mte1", 3)
                T.wait_flag("fix", "m", 4)

                QI_L1 = T.alloc_L1((Nidx1, k_block), input_dtype)
                GKI_L1 = T.alloc_L1((n_block, k_block), input_dtype)
                QI_L0A = T.alloc_L0A((Nidx1, k_block), input_dtype)
                GKI_L0B = T.alloc_L0B((k_block, n_block), input_dtype)
                C2_L0C = T.alloc_L0C((Nidx1, n_block), calc_dtype)

                T.set_flag("m", "mte1", 5)
                T.set_flag("fix", "m", 5)

                for s1_row in T.serial(s1_start, s1_end):
                    for k_idx in T.serial(d_num_query_index):
                        T.copy(QueryIndex[bz, s1_row, 0:Nidx1, k_idx * k_block:(k_idx + 1) * k_block], QI_L1)
                        T.copy(GatherKeyIndex_ws[bz, s1_row, k_start:k_end, k_idx * k_block:(k_idx + 1) * k_block], GKI_L1)
                        # RAW: gemm 等 L1 载入
                        T.set_flag("mte2", "m", 0)
                        T.wait_flag("mte2", "m", 0)
                        T.wait_flag("m", "mte1", 5)
                        T.copy(QI_L1, QI_L0A)
                        T.copy(GKI_L1, GKI_L0B, transpose=True)
                        T.set_flag("mte1", "m", 5)
                        T.wait_flag("mte1", "m", 5)
                        T.wait_flag("fix", "m", 5)
                        T.mma(QI_L0A, GKI_L0B, C2_L0C, init=(k_idx == 0))
                        T.set_flag("m", "mte1", 5)
                        if k_idx + 1 < d_num_query_index:
                            T.set_flag("fix", "m", 5)
                        # WAR: 下轮载入等 gemm 读完
                        T.set_flag("m", "mte2", 0)
                        T.wait_flag("m", "mte2", 0)
                    # M→FIX
                    T.set_flag("m", "fix", 5)
                    T.wait_flag("m", "fix", 5)
                    T.copy(C2_L0C, bmm2Res[bz, s1_row, 0:Nidx1, n_idx_k * n_block:(n_idx_k + 1) * n_block], enable_relu=True)
                    T.set_flag("fix", "m", 5)

                T.wait_flag("m", "mte1", 5)
                T.wait_flag("fix", "m", 5)

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
                bmm2_0 = T.alloc_ub((Nidx1, n_block), calc_dtype)
                bmm2_1 = T.alloc_ub((Nidx1, n_block), calc_dtype)
                weight_ub = T.alloc_ub(vector_baseg, calc_dtype)
                weight_all_ub = T.alloc_ub(Nidx1, calc_dtype)
                weight_all_half_ub = T.alloc_ub(Nidx1, input_dtype)
                weight_2d_ub = T.alloc_ub((Nidx1, n_block), calc_dtype)
                p_reduced_tmp_ub = T.alloc_ub(n_block, calc_dtype)
                rsy_tmp_ub = T.alloc_ub((vector_baseg, n_block), calc_dtype)
                rsy_reduce_db = T.alloc_ub((2, n_block), calc_dtype)
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
                    # weight 广播：row_expand_mul 替代标量循环（消除 Nidx1×PIPE_ALL）
                    # weight_2d[k,:] = 1.0 * weight_all_ub[k] → 单次 brcb+mask 全并行
                    # weight_2d 需物化（dWeight 循环 (b) 直接读整行 weight_2d[idx,:] 作向量）
                    T.tile.fill(weight_2d_ub, 1.0)
                    T.tile.row_expand_mul_experiment(weight_2d_ub, weight_2d_ub, weight_all_ub, re_tmp_ub)
                    # ReduceSumY + GlobalSoftmaxI max: fused K-block pass
                    # rsy_all_ub holds all ReduceSumY blocks (avoid GM re-reads)
                    T.tile.fill(rsy_max_ub, -T.infinity(calc_dtype))
                    # Double-buffer ReduceSumY loop: overlap copy[n_rsy+1] ∥ compute[n_rsy]
                    # bmm2 DB: V_MTE2{0,1} (WAR) / MTE2_V{0,1} (RAW)
                    T.set_flag("v", "mte2", 0)
                    T.set_flag("v", "mte2", 1)

                    # Prologue: load n_rsy=0 → slot 0
                    T.wait_flag("v", "mte2", 0)
                    T.copy(bmm2Res[bz, s1_id, 0:Nidx1, 0:n_block], bmm2_0)
                    T.set_flag("mte2", "v", 0)

                    for n_rsy in T.serial(n_num):
                        cur = n_rsy % 2
                        nxt = (n_rsy + 1) % 2

                        # Compute current slot
                        T.wait_flag("mte2", "v", cur)
                        if cur == 0:
                            T.tile.mul(bmm2_0, bmm2_0, weight_2d_ub)
                        if cur == 1:
                            T.tile.mul(bmm2_1, bmm2_1, weight_2d_ub)
                        T.pipe_barrier("v")
                        if cur == 0:
                            T.reduce_sum(bmm2_0, rsy_reduce_db[cur, :], 0)
                        if cur == 1:
                            T.reduce_sum(bmm2_1, rsy_reduce_db[cur, :], 0)
                        T.pipe_barrier("v")
                        # Save to UB for global SoftmaxI; ReduceSumY workspace is unused downstream.
                        T.copy(rsy_reduce_db[cur, :], rsy_all_ub[n_rsy, :])
                        # Fused: update global max from rsy_reduce_db[cur, :]
                        T.tile.mul(rsy_bc_ub[0, :], rsy_reduce_db[cur, :], 1.0)
                        T.pipe_barrier("v")
                        T.reduce_max(rsy_bc_ub, rsy_chunk_max_ub, 1)
                        T.pipe_barrier("v")
                        T.tile.max(rsy_max_ub, rsy_max_ub, rsy_chunk_max_ub)
                        # WAR: current slot consumed → safe to refill
                        T.set_flag("v", "mte2", cur)

                        # Prefetch next iteration (MTE2 load ∥ V compute of current)
                        if n_rsy + 1 < n_num:
                            next_n = n_rsy + 1
                            T.wait_flag("v", "mte2", nxt)
                            if nxt == 0:
                                T.copy(bmm2Res[bz, s1_id, 0:Nidx1, next_n * n_block:(next_n + 1) * n_block], bmm2_0)
                            if nxt == 1:
                                T.copy(bmm2Res[bz, s1_id, 0:Nidx1, next_n * n_block:(next_n + 1) * n_block], bmm2_1)
                            T.set_flag("mte2", "v", nxt)

                    # Clear: drain remaining WAR events
                    T.wait_flag("v", "mte2", 0)
                    T.wait_flag("v", "mte2", 1)

                    # Broadcast rsy_max to vector via T.tile.broadcast (BRCB, PIPE_V, 0 PIPE_ALL)
                    # Drain PReduced MTE3 write before refilling p_reduced_tmp_ub.
                    T.set_flag("mte3", "v", 0)
                    T.wait_flag("mte3", "v", 0)
                    T.tile.fill(p_reduced_tmp_ub, 1.0)
                    T.pipe_barrier("v")
                    T.tile.broadcast(rsy_bc_ub, rsy_max_ub)
                    # BRCB and the following V op have a RAW dependency.
                    T.pipe_barrier("v")
                    T.tile.mul(softmax_i_chunk_ub, p_reduced_tmp_ub, rsy_bc_ub[0, :])

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

                    # Per-core SoftmaxI: use UB-held ReduceSumY, then materialize to GM.
                    T.copy(rsy_all_ub[n_idx_k, :], rsy_chunk_ub)
                    T.pipe_barrier("v")
                    T.tile.sub(rsy_chunk_ub, rsy_chunk_ub, softmax_i_chunk_ub)
                    T.pipe_barrier("v")
                    T.tile.exp(rsy_chunk_ub, rsy_chunk_ub)
                    T.pipe_barrier("v")
                    T.tile.broadcast(rsy_bc_ub, rsy_sum_ub)
                    # BRCB and div both use PIPE_V; preserve the broadcast result.
                    T.pipe_barrier("v")
                    T.tile.div(rsy_chunk_ub, rsy_chunk_ub, rsy_bc_ub[0, :])
                    T.set_flag("v", "mte3", 4)
                    T.wait_flag("v", "mte3", 4)
                    T.copy(rsy_chunk_ub, SoftmaxI[bz, s1_id, n_idx_k * n_block:(n_idx_k + 1) * n_block])

                    # dWeight, reluGrad for MY K block (same V scope, no cross flag)
                    # Strategy: 2D-vectorized per-row ops to eliminate scalar tile.mul
                    #   - weight_2d_ub rows → rg_chunk_ub via T.copy (scalar idx, avoids UB range slicing bug)
                    #   - diff_2d_ub broadcast once before g_id loop (32× UB→UB copy, amortized)
                    #   - Op1/Op2/Op3 all use 2D tile.mul (3 ops per g_id vs 96 before)
                    #   - bmm2 loaded per-chunk from GM → dw_chunk_ub (no UB→UB copy)
                    diff_ub = T.alloc_ub(n_block, calc_dtype)
                    diff_2d_ub = T.alloc_ub((vector_baseg, n_block), calc_dtype)
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
                    # RAW: PReduced/SoftmaxI 刚由 mte3 写，mte2 读回前等 mte3 完成.
                    T.set_flag("mte3", "mte2", 5)
                    T.wait_flag("mte3", "mte2", 5)
                    T.copy(SoftmaxI[bz, s1_id, n_idx_k * n_block:(n_idx_k + 1) * n_block], softmax_i_chunk_ub)
                    T.copy(PReduced[bz, s1_id, n_idx_k * n_block:(n_idx_k + 1) * n_block], p_reduced_tmp_ub)
                    T.set_flag("mte2", "v", 6)
                    T.wait_flag("mte2", "v", 6)
                    T.tile.sub(diff_ub, softmax_i_chunk_ub, p_reduced_tmp_ub)

                    # Broadcast diff_ub → diff_2d_ub via BRCB (1 instruction replaces 32 serial copies)
                    T.tile.broadcast(diff_2d_ub, diff_ub)

                    # WAR pre-fill: dw_chunk_ub 初始可安全写入（无前序 V 读者）
                    T.set_flag("v", "mte2", 4)

                    # Keep T.serial bounds static for TIR simplification, then
                    # select the independent Nidx1 half owned by this AIV.
                    for g_id in T.serial(g_num):
                        g_start = g_id * vector_baseg
                        if g_id == vid:
                            T.tile.fill(weight_ub, 0)

                            # WAR: 等 V 读完 dw_chunk_ub → MTE2 安全覆盖（替代 barrier_all 兜底）
                            T.wait_flag("v", "mte2", 4)
                            T.copy(bmm2Res[bz, s1_id, g_start:g_start + vector_baseg, n_idx_k * n_block:(n_idx_k + 1) * n_block], dw_chunk_ub)

                            T.set_flag("mte2", "v", 7)
                            T.wait_flag("mte2", "v", 7)

                            T.tile.mul(rsy_tmp_ub, dw_chunk_ub, -10000.0)
                            T.pipe_barrier("v")
                            T.tile.exp(rsy_tmp_ub, rsy_tmp_ub)
                            T.pipe_barrier("v")
                            T.tile.sub(rsy_tmp_ub, ones_2d_ub, rsy_tmp_ub)

                            # Op1: rg = weight_slice * diff (2D)
                            # weight切片逐行copy（整块copy参数在CANN 9.1下不正确）
                            for i in T.serial(vector_baseg):
                                T.copy(weight_2d_ub[g_start + i, :], rg_chunk_ub[i, :])
                            T.tile.mul(rg_chunk_ub, rg_chunk_ub, diff_2d_ub)
                            # Op2: rg *= rsy_tmp mask (2D, was per-row 1D mul)
                            T.pipe_barrier("v")
                            T.tile.mul(rg_chunk_ub, rg_chunk_ub, rsy_tmp_ub)
                            # Op3: dw *= diff (2D, was per-row 1D mul)
                            T.pipe_barrier("v")
                            T.tile.mul(dw_chunk_ub, dw_chunk_ub, diff_2d_ub)

                            T.pipe_barrier("v")
                            T.reduce_sum(dw_chunk_ub, dw_reduce_ub, 1)
                            T.pipe_barrier("v")
                            T.tile.add(weight_ub, weight_ub, dw_reduce_ub)
                            # WAR: V 已读完 dw_chunk_ub → 通知 MTE2 下轮可覆盖
                            T.set_flag("v", "mte2", 4)

                            T.tile.cast(rg_half_ub, rg_chunk_ub, "CAST_RINT", vector_baseg * n_block)
                            # set_flag("v",...) 自带 V pipe barrier，无需 barrier_all
                            T.set_flag("v", "mte3", 0)
                            T.wait_flag("v", "mte3", 0)
                            T.pipe_barrier("mte3")
                            T.copy(rg_half_ub, reluGradRes[bz, s1_id, g_start:g_start + vector_baseg, n_idx_k * n_block:(n_idx_k + 1) * n_block])

                            T.pipe_barrier("v")
                            T.tile.cast(dw_half_ub, weight_ub, "CAST_RINT", vector_baseg)
                            T.set_flag("v", "mte3", 1)
                            T.wait_flag("v", "mte3", 1)
                            T.copy(dw_half_ub, dWeight_partial[bz, s1_id, n_idx_k, g_start:g_start + vector_baseg])

                    # Clear: 排空 V_MTE2(4) WAR 事件（最后轮 g_id 的 set）
                    T.wait_flag("v", "mte2", 4)

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
                RG_L0A_dq = T.alloc_L0A((Nidx1, n_block), input_dtype)
                GKI_L0B_dq = T.alloc_L0B((n_block, k_block), input_dtype)
                DQ_L0C = T.alloc_L0C((Nidx1, k_block), calc_dtype)
                # dKeyIndex low-level MMA: L1→L0A transpose 替代 V1 预转置
                QI_L1_dk = T.alloc_L1((Nidx1, DQueryIndex), input_dtype)
                RG_L0A_dk = T.alloc_L0A((n_block, Nidx1), input_dtype)
                QI_L0B_dk = T.alloc_L0B((Nidx1, DQueryIndex), input_dtype)
                DK_L0C = T.alloc_L0C((n_block, DQueryIndex), calc_dtype)

                T.set_flag("m", "mte1", 6)
                T.set_flag("fix", "m", 7)

                for s1_row in T.serial(s1_start, s1_end):
                    for d_idx in T.serial(d_num_query_index):
                        T.copy(reluGradRes[bz, s1_row, 0:Nidx1, n_idx_k * n_block:(n_idx_k + 1) * n_block], RG_L1)
                        T.copy(GatherKeyIndex_ws[bz, s1_row, k_start:k_end, d_idx * k_block:(d_idx + 1) * k_block], GKI2_L1)
                        # RAW: gemm 等 L1 载入
                        T.set_flag("mte2", "m", 6)
                        T.wait_flag("mte2", "m", 6)
                        T.wait_flag("m", "mte1", 6)
                        T.copy(RG_L1, RG_L0A_dq)
                        T.copy(GKI2_L1, GKI_L0B_dq)
                        T.set_flag("mte1", "m", 6)
                        T.wait_flag("mte1", "m", 6)
                        T.wait_flag("fix", "m", 7)
                        T.mma(RG_L0A_dq, GKI_L0B_dq, DQ_L0C, init=True)
                        T.set_flag("m", "mte1", 6)
                        # M→FIX: atomic_add 等 gemm 完成
                        T.set_flag("m", "fix", 7)
                        T.wait_flag("m", "fix", 7)
                        T.tile.atomic_add(dQueryIndexAccum[bz, s1_row, 0:Nidx1, d_idx * k_block:(d_idx + 1) * k_block], DQ_L0C)
                        T.set_flag("fix", "m", 7)

                    # dKeyIndex: L1→L0A transpose (硬件转置) + T.mma，替代 V1 预转置
                    # DK = RG^T @ QI  = (n_block,Nidx1) @ (Nidx1,DQI) = (n_block,DQI)
                    # GM→L1 (MTE2): 直接加载 reluGradRes，无需预转置
                    T.copy(reluGradRes[bz, s1_row, 0:Nidx1, n_idx_k * n_block:(n_idx_k + 1) * n_block], RG_L1)
                    T.copy(QueryIndex[bz, s1_row, 0:Nidx1, 0:DQueryIndex], QI_L1_dk)
                    # MTE2→MTE1: L1 就绪，MTE1 可以开始搬运
                    T.set_flag("mte2", "mte1", 10)
                    T.wait_flag("mte2", "mte1", 10)
                    # L1→L0 (MTE1): RG 转置搬运到 L0A，QI 正常搬运到 L0B
                    T.copy(RG_L1, RG_L0A_dk, transpose=True)
                    T.copy(QI_L1_dk, QI_L0B_dk)
                    # MTE1→M: L0 就绪，Cube 可以开始 MMA
                    T.set_flag("mte1", "m", 10)
                    T.wait_flag("mte1", "m", 10)
                    # MMA (Cube)
                    T.mma(RG_L0A_dk, QI_L0B_dk, DK_L0C, init=True)
                    # M→FIX: L0C 就绪，FIX 可以写出
                    T.set_flag("m", "fix", 9)
                    T.wait_flag("m", "fix", 9)
                    # 每 core 独占自己的 K 行，只写一次 → 普通 copy_l0c_to_gm
                    T.copy(DK_L0C, dk_matmul_partial[bz, s1_row, n_idx_k * n_block:(n_idx_k + 1) * n_block, 0:DQueryIndex])

                T.wait_flag("m", "mte1", 6)
                T.wait_flag("fix", "m", 7)

                T.set_cross_flag("FIX", 3)

            # ============================================================
            # V scope 2: Cast dQueryIndex FP32→FP16 + Scatter → dKeyIndex
            #   AIV0/AIV1 split: cast by g_id range, scatter by k_id range
            #   Both ops independent → merged into single V scope
            # ============================================================
            with T.Scope("V"):
                T.wait_cross_flag(3)

                dq_fp32_db = T.alloc_ub((2, 4, BLOCK_K), calc_dtype)
                dq_fp16_db = T.alloc_ub((2, 4, BLOCK_K), input_dtype)
                dk_chunk_0 = T.alloc_ub((1, DQueryIndex), calc_dtype)
                dk_chunk_1 = T.alloc_ub((1, DQueryIndex), calc_dtype)

                g_per_aiv = v3_chunks // 2
                half_block = n_block // 2

                for s1_row in T.serial(s1_start, s1_end):
                    # ---- Part 1: Cast dQueryIndex FP32→FP16 (split g_id, WAR-safe ping-pong DB) ----
                    # Pipeline: MTE2 fill ∥ (V cast → MTE3 write) ∥ MTE2 prefetch
                    # Events: MTE2_V{7,8} (RAW), V_MTE3{7,8} (RAW), MTE3_MTE2{7,8} (WAR)
                    # vid 是 Python int(0/1)，直接算术求 g_start/g_end，避免 if/else 跨分支变量
                    g_start = vid * g_per_aiv           # 0 or 32
                    g_end = g_start + g_per_aiv         # 32 or 64

                    # Pre-fill WAR: both slots initially free
                    T.set_flag("mte3", "mte2", 7)
                    T.set_flag("mte3", "mte2", 8)

                    # Prologue: MTE2 fill + V cast for slot 0 (MTE3 deferred to loop)
                    T.wait_flag("mte3", "mte2", 7)
                    for d_idx in T.serial(d_num_query_index):
                        T.copy(dQueryIndexAccum[bz, s1_row, g_start * 4:(g_start + 1) * 4, d_idx * BLOCK_K:(d_idx + 1) * BLOCK_K], dq_fp32_db[0, :, :])
                    T.set_flag("mte2", "v", 7)
                    T.wait_flag("mte2", "v", 7)
                    T.tile.cast(dq_fp16_db[0, :, :], dq_fp32_db[0, :, :], "CAST_RINT", 4 * BLOCK_K)
                    T.set_flag("v", "mte3", 7)

                    for g_id in T.serial(g_start, g_end):
                        cur = (g_id - g_start) % 2
                        nxt = (g_id - g_start + 1) % 2

                        # MTE3 write slot cur → GM (∥ MTE2+V prefetch of nxt)
                        T.wait_flag("v", "mte3", cur + 7)
                        for d_idx in T.serial(d_num_query_index):
                            T.copy(dq_fp16_db[cur, :, :], dQueryIndex[bz, s1_row, g_id * 4:(g_id + 1) * 4, d_idx * BLOCK_K:(d_idx + 1) * BLOCK_K])
                        # WAR: slot cur consumed → safe to refill
                        T.set_flag("mte3", "mte2", cur + 7)

                        # Prefetch next g_id → slot nxt (MTE2 fill + V cast)
                        next_g = g_id + 1
                        if next_g < g_end:
                            T.wait_flag("mte3", "mte2", nxt + 7)
                            for d_idx in T.serial(d_num_query_index):
                                T.copy(dQueryIndexAccum[bz, s1_row, next_g * 4:(next_g + 1) * 4, d_idx * BLOCK_K:(d_idx + 1) * BLOCK_K], dq_fp32_db[nxt, :, :])
                            T.set_flag("mte2", "v", nxt + 7)
                            T.wait_flag("mte2", "v", nxt + 7)
                            T.tile.cast(dq_fp16_db[nxt, :, :], dq_fp32_db[nxt, :, :], "CAST_RINT", 4 * BLOCK_K)
                            T.set_flag("v", "mte3", nxt + 7)

                    # Clear: drain remaining WAR events
                    T.wait_flag("mte3", "mte2", 7)
                    T.wait_flag("mte3", "mte2", 8)

                    # ---- Part 2: Scatter dk_matmul_partial → dKeyIndex (split k_id across AIVs) ----
                    # WAR-safe ping-pong DB: overlap MTE2 copy[k+1] ∥ MTE3 atomic_add[k]
                    # Events: MTE2_MTE3{5,6} (RAW) / MTE3_MTE2{5,6} (WAR)
                    # TileLang 不支持 if/else 跨分支变量 → DB 逻辑在分支内各写一份
                    k_mid = k_start + half_block  # Python int
                    if vid == 0:
                        T.set_flag("mte3", "mte2", 5)
                        T.set_flag("mte3", "mte2", 6)

                        T.wait_flag("mte3", "mte2", 5)
                        T.copy(dk_matmul_partial[bz, s1_row, k_start:k_start + 1, 0:DQueryIndex], dk_chunk_0)
                        T.set_flag("mte2", "mte3", 5)

                        for k_id in T.serial(k_start, T.min(k_mid, k_end)):
                            cur = (k_id - k_start) % 2
                            nxt = (k_id - k_start + 1) % 2

                            T.wait_flag("mte2", "mte3", cur + 5)
                            idx = SparseIndices[bz, s1_row, 0, k_id]
                            if idx >= 0:
                                if cur == 0:
                                    T.tile.atomic_add(dKeyIndex[bz, idx, 0, 0], dk_chunk_0)
                                if cur == 1:
                                    T.tile.atomic_add(dKeyIndex[bz, idx, 0, 0], dk_chunk_1)
                            T.set_flag("mte3", "mte2", cur + 5)

                            next_k = k_id + 1
                            if next_k < T.min(k_mid, k_end):
                                T.wait_flag("mte3", "mte2", nxt + 5)
                                if nxt == 0:
                                    T.copy(dk_matmul_partial[bz, s1_row, next_k:next_k + 1, 0:DQueryIndex], dk_chunk_0)
                                if nxt == 1:
                                    T.copy(dk_matmul_partial[bz, s1_row, next_k:next_k + 1, 0:DQueryIndex], dk_chunk_1)
                                T.set_flag("mte2", "mte3", nxt + 5)

                        T.wait_flag("mte3", "mte2", 5)
                        T.wait_flag("mte3", "mte2", 6)
                    else:
                        T.set_flag("mte3", "mte2", 5)
                        T.set_flag("mte3", "mte2", 6)

                        T.wait_flag("mte3", "mte2", 5)
                        T.copy(dk_matmul_partial[bz, s1_row, k_mid:k_mid + 1, 0:DQueryIndex], dk_chunk_0)
                        T.set_flag("mte2", "mte3", 5)

                        for k_id in T.serial(T.min(k_mid, k_end), k_end):
                            cur = (k_id - k_mid) % 2
                            nxt = (k_id - k_mid + 1) % 2

                            T.wait_flag("mte2", "mte3", cur + 5)
                            idx = SparseIndices[bz, s1_row, 0, k_id]
                            if idx >= 0:
                                if cur == 0:
                                    T.tile.atomic_add(dKeyIndex[bz, idx, 0, 0], dk_chunk_0)
                                if cur == 1:
                                    T.tile.atomic_add(dKeyIndex[bz, idx, 0, 0], dk_chunk_1)
                            T.set_flag("mte3", "mte2", cur + 5)

                            next_k = k_id + 1
                            if next_k < k_end:
                                T.wait_flag("mte3", "mte2", nxt + 5)
                                if nxt == 0:
                                    T.copy(dk_matmul_partial[bz, s1_row, next_k:next_k + 1, 0:DQueryIndex], dk_chunk_0)
                                if nxt == 1:
                                    T.copy(dk_matmul_partial[bz, s1_row, next_k:next_k + 1, 0:DQueryIndex], dk_chunk_1)
                                T.set_flag("mte2", "mte3", nxt + 5)

                        T.wait_flag("mte3", "mte2", 5)
                        T.wait_flag("mte3", "mte2", 6)

    @T.prim_func
    def main_pipeline(
        # Inputs: 0-11
        SparseIndices: T.Tensor((B, S1, Nidx2, K), "int32"),
        KRealSize: T.Tensor((B, S1), "int32"),
        ValidKMask: T.Tensor((B, S1, K), calc_dtype),
        Query: T.Tensor((B, S1, N1, DQuery), input_dtype),
        QueryRope: T.Tensor((B, S1, N1, d_rope), input_dtype),
        Key: T.Tensor((B, S2, N2, DQuery), input_dtype),
        KeyRope: T.Tensor((B, S2, N2, d_rope), input_dtype),
        QueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), input_dtype),
        KeyIndex: T.Tensor((B, S2, Nidx2, DQueryIndex), input_dtype),
        Weights: T.Tensor((B, S1, Nidx1), input_dtype),
        SoftmaxMax: T.Tensor((B, N2, S1, G), calc_dtype),
        SoftmaxSum: T.Tensor((B, N2, S1, G), calc_dtype),
        # Workspaces: 10-22
        dQueryIndexAccum: T.Tensor((B, S1, Nidx1, DQueryIndex), calc_dtype),
        dKeyIndexAccum: T.Tensor((dkey_output_rows, DQueryIndex), calc_dtype),
        GatherKey_ws: T.Tensor((B, S1, K, DQuery), input_dtype),
        GatherKeyRope_ws: T.Tensor((B, S1, K, d_rope), input_dtype),
        GatherKeyIndex_ws: T.Tensor((B, S1, K, DQueryIndex), input_dtype),
        bmm1Res: T.Tensor((B, S1, N1, K), calc_dtype),
        bmm2Res: T.Tensor((B, S1, Nidx1, K), calc_dtype),
        PReduced: T.Tensor((B, S1, K), calc_dtype),
        SoftmaxI: T.Tensor((B, S1, K), calc_dtype),
        dk_matmul_partial: T.Tensor((B, S1, v2_k_padded, DQueryIndex), calc_dtype),
        reluGradRes: T.Tensor((B, S1, Nidx1, K), input_dtype),
        LossPartial: T.Tensor((loss_partial_size,), calc_dtype),
        C2ZeroPad: T.Tensor((8, DQueryIndex), input_dtype),
        # Outputs: -4..-1
        dQueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), input_dtype),
        dKeyIndexOut: T.Tensor((dkey_output_rows, DQueryIndex), input_dtype),
        dWeightOut: T.Tensor((B, S1, Nidx1), input_dtype),
        LossOut: T.Tensor((1,), calc_dtype),
    ):
        with T.Kernel(launch_core_num, is_npu=True) as (cid, vid):
            for pipe_iter in T.serial(tasks_per_core + TASK_PIPELINE_DEPTH - 1):
                if pipe_iter < tasks_per_core:
                    task_current = pipe_iter * launch_core_num + cid
                    if task_current < task_count:
                        bz_current = task_current // S1
                        s1_current = task_current % S1
                        vector0_gather_task(
                            SparseIndices,
                            KRealSize,
                            Key,
                            KeyRope,
                            KeyIndex,
                            GatherKey_ws,
                            GatherKeyRope_ws,
                            GatherKeyIndex_ws,
                            bz_current,
                            s1_current,
                            s1_current + 1,
                            pipe_iter % 2,
                            vid,
                        )

                if pipe_iter >= 1:
                    task_prev_iter = pipe_iter - 1
                    if task_prev_iter < tasks_per_core:
                        task_prev = task_prev_iter * launch_core_num + cid
                        if task_prev < task_count:
                            bz_prev = task_prev // S1
                            s1_prev = task_prev % S1
                            task_prev_slot = task_prev_iter % 2
                            cube1_task(
                                Query,
                                QueryRope,
                                QueryIndex,
                                GatherKey_ws,
                                GatherKeyRope_ws,
                                GatherKeyIndex_ws,
                                bmm1Res,
                                bmm2Res,
                                bz_prev,
                                s1_prev,
                                s1_prev + 1,
                                task_prev_slot,
                            )
                            vector1_psy_task(
                                KRealSize,
                                ValidKMask,
                                SoftmaxMax,
                                SoftmaxSum,
                                Weights,
                                bmm1Res,
                                bmm2Res,
                                PReduced,
                                SoftmaxI,
                                bz_prev,
                                s1_prev,
                                s1_prev + 1,
                                task_prev_slot,
                                vid,
                            )
                            relay_vector1_psy()
                            vector1_backward_task(
                                ValidKMask,
                                Weights,
                                bmm2Res,
                                PReduced,
                                SoftmaxI,
                                dWeightOut,
                                reluGradRes,
                                LossPartial,
                                bz_prev,
                                s1_prev,
                                s1_prev + 1,
                                task_prev_slot,
                                vid,
                            )
                            cube2_task(
                                QueryIndex,
                                GatherKeyIndex_ws,
                                reluGradRes,
                                C2ZeroPad,
                                dQueryIndexAccum,
                                dk_matmul_partial,
                                bz_prev,
                                s1_prev,
                                s1_prev + 1,
                                task_prev_slot,
                            )

            # All task owners must finish C2 before the all-core deterministic
            # scatter phase starts.  This mirrors the official transition to
            # ProcessDeterVector2 and also lets idle task cores participate.
            T.sync_all()
            clear_dkey_accumulator(dKeyIndexAccum, cid, vid)
            T.sync_all()
            vector2_scatter_all_tasks(
                SparseIndices,
                KRealSize,
                dk_matmul_partial,
                dKeyIndexAccum,
                cid,
                vid,
            )
            T.sync_all()
            finalize_vector_outputs(
                dQueryIndexAccum,
                dKeyIndexAccum,
                dQueryIndex,
                dKeyIndexOut,
                LossPartial,
                LossOut,
                cid,
                vid,
            )

    return main_pipeline

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
# Python-layer vector computation reference - kept as fallback
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
    global _debug_dump_done

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
            actual_q_len = (
                torch.tensor(actual_seq_lengths_query, dtype=torch.int32, device=query.device)
                if isinstance(actual_seq_lengths_query, list)
                else actual_seq_lengths_query.to(device=query.device, dtype=torch.int32)
            )
        else:
            actual_q_len = torch.full((B,), S1, dtype=torch.int32, device=query.device)

        if actual_seq_lengths_key is not None:
            actual_k_len = (
                torch.tensor(actual_seq_lengths_key, dtype=torch.int32, device=query.device)
                if isinstance(actual_seq_lengths_key, list)
                else actual_seq_lengths_key.to(device=query.device, dtype=torch.int32)
            )
        else:
            actual_k_len = torch.full((B,), S2, dtype=torch.int32, device=query.device)

        query_positions = torch.arange(S1, dtype=torch.int32, device=query.device)
        k_real_size = (
            actual_k_len.unsqueeze(1) - actual_q_len.unsqueeze(1) + query_positions + 1
        ).clamp_(min=0, max=K).contiguous()
        valid_k_mask = (
            torch.arange(K, dtype=torch.int32, device=query.device).view(1, 1, K)
            < k_real_size.unsqueeze(-1)
        ).to(torch.float32).contiguous()

        S2_pad = (S2 + VECTOR_BASEN - 1) // VECTOR_BASEN * VECTOR_BASEN
        if S2_pad > S2:
            key = torch.nn.functional.pad(key, (0, 0, 0, 0, 0, S2_pad - S2, 0, 0))
            key_index = torch.nn.functional.pad(key_index, (0, 0, 0, 0, 0, S2_pad - S2, 0, 0))
            if key_rope is not None:
                key_rope = torch.nn.functional.pad(
                    key_rope, (0, 0, 0, 0, 0, S2_pad - S2, 0, 0)
                )

        if weights.dtype == torch.float32:
            weights = weights.to(input_dtype)

        _PROFILE = os.environ.get("SLIG_PROFILE", "0") == "1"
        _DEBUG_REQUESTED = os.environ.get("SLIG_DEBUG", "0") == "1"
        _GATHER_SENTINEL = 123.0

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

        # Matrix and vector stages share one fixed 128-row K tile.  Core
        # parallelism is owned by complete (B, S1) tasks, not K tiles.
        hardware_core_count = _get_aicore_count()
        schedule = _make_task_schedule(
            B,
            S1,
            S2_pad,
            K,
            hardware_core_count,
        )
        core_count = schedule.launch_core_num
        block_n = N_SPLIT
        n_num_ws = K // block_n
        v2_k_padded = K
        loss_partial_stride = 8
        loss_partial_size = ceil_div(
            B * S1 * 2 * loss_partial_stride, 128
        ) * 128

        cache_key = (
            B, S1, S2_pad, N1, N2, G, Nidx1, Nidx2,
            DQuery, DQueryIndex, K, core_count, d_rope,
            input_dtype_str, calc_dtype_str, scale_value, block_n,
        )

        if cache_key in _kernel_cache:
            func = _kernel_cache[cache_key]
        else:
            func = tilelang_grad_kl_loss_single_kernel(
                B, S1, S2_pad, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
                core_count, d_rope, input_dtype_str, calc_dtype_str, scale_value, block_n,
            )

            # # dump 编译出的 AscendC 源码到文件，供离线核对手动同步转写
            try:
                import os as _os
                _dump_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "0721_kernel_source.cpp")
                with open(_dump_path, "w", encoding="utf-8") as _f:
                    _f.write(func.get_kernel_source())
                print(f"[dump] kernel source written to {_dump_path}")
            except Exception as _e:
                print(f"[dump] failed to write kernel source: {_e}")

            _kernel_cache[cache_key] = func

        ws_key = (
            B, S1, S2_pad, N1, Nidx1, K, DQueryIndex, d_rope,
            input_dtype, query.device, n_num_ws, v2_k_padded,
            loss_partial_size,
        )
        if ws_key in _workspace_cache:
            ws = _workspace_cache[ws_key]
            d_query_index_accum, d_key_index_accum, \
                gather_key_ws, gather_key_rope_ws, gather_key_index_ws, \
                bmm1_res_ws, bmm2_res_ws, p_reduced_ws, softmax_i_ws, \
                dk_matmul_partial, relu_grad_out, loss_partial_ws, \
                c2_zero_pad = ws
        else:
            gather_key_ws = torch.zeros(B, S1, K, DQuery, dtype=input_dtype, device=query.device)
            gather_key_rope_ws = torch.zeros(B, S1, K, d_rope, dtype=input_dtype, device=query.device)
            gather_key_index_ws = torch.zeros(B, S1, K, DQueryIndex, dtype=input_dtype, device=query.device)
            bmm1_res_ws = torch.zeros(B, S1, N1, K, dtype=torch.float32, device=query.device)
            bmm2_res_ws = torch.zeros(B, S1, Nidx1, K, dtype=torch.float32, device=query.device)
            p_reduced_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=query.device)
            softmax_i_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=query.device)
            d_query_index_accum = torch.zeros(B, S1, Nidx1, DQueryIndex, dtype=torch.float32, device=query.device)
            # dKeyIndex 融合：matmul 中间结果(workspace) + fp32 累加器输出
            dk_matmul_partial = torch.zeros(
                B, S1, v2_k_padded, DQueryIndex,
                dtype=torch.float32, device=query.device,
            )
            d_key_index_accum = torch.empty(
                B * S2_pad, DQueryIndex,
                dtype=torch.float32, device=query.device,
            )
            relu_grad_out = torch.zeros(B, S1, Nidx1, K, dtype=input_dtype, device=query.device)
            loss_partial_ws = torch.zeros(
                loss_partial_size, dtype=torch.float32, device=query.device
            )
            c2_zero_pad = torch.zeros(
                8, DQueryIndex, dtype=input_dtype, device=query.device
            )
            _workspace_cache[ws_key] = (
                d_query_index_accum, d_key_index_accum,
                gather_key_ws, gather_key_rope_ws, gather_key_index_ws,
                bmm1_res_ws, bmm2_res_ws, p_reduced_ws, softmax_i_ws,
                dk_matmul_partial, relu_grad_out, loss_partial_ws,
                c2_zero_pad,
            )

        if _DEBUG_REQUESTED and not _debug_dump_done:
            gather_key_ws.fill_(_GATHER_SENTINEL)
            gather_key_rope_ws.fill_(_GATHER_SENTINEL)
            gather_key_index_ws.fill_(_GATHER_SENTINEL)

        if _PROFILE:
            _t_prep_end.record()
            _t_kernel_start.record()
        results = func(
            sparse_indices,                             # 0
            k_real_size, valid_k_mask,                  # 1-2
            query, query_rope, key, key_rope,           # 3-6
            query_index, key_index,                     # 7-8
            weights, softmax_max, softmax_sum,          # 9-11
            d_query_index_accum,                        # 12 workspace
            d_key_index_accum,                          # 13 workspace
            gather_key_ws, gather_key_rope_ws, gather_key_index_ws,  # 14-16
            bmm1_res_ws, bmm2_res_ws,                   # 17-18
            p_reduced_ws, softmax_i_ws,                 # 19-20
            dk_matmul_partial, relu_grad_out,           # 21-22
            loss_partial_ws, c2_zero_pad,               # 23-24
        )
        if _PROFILE:
            _t_kernel_end.record()
            _t_reduce_start.record()

        d_query_index = results[0]
        d_key_index_flat = results[1]
        d_weight = results[2]
        loss_out = results[3]

        # DEBUG: identify the first stage that produces a non-finite value.
        _DEBUG = _DEBUG_REQUESTED and not _debug_dump_done
        if _DEBUG:
            def _debug_tensor(name, tensor):
                tensor_fp32 = tensor.float()
                nan_count = torch.isnan(tensor_fp32).sum().item()
                inf_count = torch.isinf(tensor_fp32).sum().item()
                finite_tensor = torch.nan_to_num(
                    tensor_fp32, nan=0.0, posinf=0.0, neginf=0.0
                )
                print(
                    f"[DEBUG] {name}: shape={tuple(tensor.shape)}, "
                    f"nan={nan_count}, inf={inf_count}, "
                    f"finite_abs_max={finite_tensor.abs().max().item():.6e}"
                )

            expected_gk, expected_gki, expected_gkr = _preprocess_gather(
                key, key_index, key_rope, sparse_indices,
                B, S1, S2_pad, N2, Nidx2, K, DQuery, DQueryIndex, d_rope_orig, query.dtype,
            )
            if d_rope_orig > 0 and d_rope > d_rope_orig:
                expected_gkr = torch.nn.functional.pad(
                    expected_gkr, (0, d_rope - d_rope_orig)
                )
            _debug_tensor("Key", key)
            _debug_tensor("Query", query)
            _debug_tensor("QueryIndex", query_index)
            _debug_tensor("KeyIndex", key_index)
            _debug_tensor("Weights", weights)
            _debug_tensor("KRealSize", k_real_size)
            _debug_tensor("ValidKMask", valid_k_mask)
            _debug_tensor("SoftmaxMax", softmax_max)
            _debug_tensor("SoftmaxSum", softmax_sum)
            _debug_tensor("GatherKey_ws", gather_key_ws)
            _debug_tensor("GatherKeyRope_ws", gather_key_rope_ws)
            _debug_tensor("GatherKeyIndex_ws", gather_key_index_ws)
            _debug_tensor("bmm1Res", bmm1_res_ws)
            _debug_tensor("bmm2Res", bmm2_res_ws)
            _debug_tensor("PReduced", p_reduced_ws)
            _debug_tensor("SoftmaxI", softmax_i_ws)
            _debug_tensor("reluGradRes", relu_grad_out)
            _debug_tensor("dk_matmul_partial", dk_matmul_partial)
            _debug_tensor("dQueryIndexAccum", d_query_index_accum)
            _debug_tensor("dKeyIndexAccum", d_key_index_accum)
            _debug_tensor("dWeightOut", d_weight)
            _debug_tensor("LossPartial", loss_partial_ws)
            _debug_tensor("LossOut", loss_out)

            print(
                f"[DEBUG] Gather sentinel remaining: "
                f"Key={(gather_key_ws == _GATHER_SENTINEL).sum().item()}, "
                f"Rope={(gather_key_rope_ws == _GATHER_SENTINEL).sum().item()}, "
                f"KeyIndex={(gather_key_index_ws == _GATHER_SENTINEL).sum().item()}"
            )

            gk_diff = (gather_key_ws.float() - expected_gk.float()).abs()
            gki_diff = (gather_key_index_ws.float() - expected_gki.float()).abs()
            gkr_diff = (gather_key_rope_ws.float() - expected_gkr.float()).abs()
            debug_softmax_max = softmax_max[:, 0, :, :].float()
            debug_softmax_sum = softmax_sum[:, 0, :, :].float().abs().clamp_min(1e-8)
            debug_scaled_bmm1 = bmm1_res_ws.float() * scale_value
            debug_valid_mask = valid_k_mask.float()
            debug_p_reduced = (
                torch.exp(debug_scaled_bmm1 - debug_softmax_max.unsqueeze(-1))
                / debug_softmax_sum.unsqueeze(-1)
            ).mean(dim=2) * debug_valid_mask
            debug_reduce_sum_y = (
                bmm2_res_ws.float() * weights.float().unsqueeze(-1)
            ).sum(dim=2)
            debug_reduce_sum_y = debug_reduce_sum_y.masked_fill(debug_valid_mask == 0, -1e30)
            debug_softmax_i = torch.softmax(debug_reduce_sum_y, dim=-1) * debug_valid_mask
            debug_diff = debug_softmax_i - debug_p_reduced
            debug_relu_grad = (
                weights.float().unsqueeze(-1)
                * debug_diff.unsqueeze(2)
                * (bmm2_res_ws.float() > 0).float()
            )
            debug_d_weight = (
                bmm2_res_ws.float() * debug_diff.unsqueeze(2)
            ).sum(dim=-1)
            debug_p_safe = debug_p_reduced.clamp_min(1e-8)
            debug_softmax_i_safe = debug_softmax_i.clamp_min(1e-8)
            debug_loss = (
                debug_p_reduced
                * (torch.log(debug_p_safe) - torch.log(debug_softmax_i_safe))
            ).sum()
            debug_loss_partials = loss_partial_ws[
                :B * S1 * 2 * loss_partial_stride:loss_partial_stride
            ]
            p_reduced_diff = (p_reduced_ws.float() - debug_p_reduced).abs()
            softmax_i_diff = (softmax_i_ws.float() - debug_softmax_i).abs()
            relu_grad_diff = (relu_grad_out.float() - debug_relu_grad).abs()
            debug_relu_grad_bf16 = debug_relu_grad.to(
                relu_grad_out.dtype
            ).float()
            relu_grad_bf16_diff = (
                relu_grad_out.float() - debug_relu_grad_bf16
            ).abs()
            debug_unmasked_rg = (
                weights.float().unsqueeze(-1) * debug_diff.unsqueeze(2)
            )
            debug_expected_mask = bmm2_res_ws.float() > 0
            debug_actual_rg = relu_grad_out.float()
            debug_strong_candidate = debug_unmasked_rg.abs() > 1e-3
            debug_false_zero = (
                debug_expected_mask
                & debug_strong_candidate
                & (debug_actual_rg == 0)
            )
            debug_false_nonzero = (
                (~debug_expected_mask) & (debug_actual_rg.abs() > 1e-3)
            )

            # Rebuild C2 from the stored BF16 reluGrad to separate V1 from C2.
            debug_dq_from_stored_rg = torch.matmul(
                debug_actual_rg, gather_key_index_ws.float()
            )
            debug_dk_from_stored_rg = torch.matmul(
                debug_actual_rg.transpose(-2, -1), query_index.float()
            )
            debug_c2_dq_diff = (
                d_query_index_accum.float() - debug_dq_from_stored_rg
            ).abs()
            debug_c2_dk_diff = (
                dk_matmul_partial.float() - debug_dk_from_stored_rg
            ).abs()
            d_weight_diff = (d_weight.float() - debug_d_weight).abs()
            loss_diff = (loss_out.float().reshape(-1)[0] - debug_loss).abs()
            p_max_diff = (
                debug_scaled_bmm1.amax(dim=-1) - debug_softmax_max
            ).abs()
            print(
                f"[DEBUG] GatherKey_ws vs host_gather: "
                f"max_diff={gk_diff.max().item():.6e}, "
                f"mean_diff={gk_diff.mean().item():.6e}"
            )
            print(
                f"[DEBUG] GatherKeyIndex_ws vs host_gather: "
                f"max_diff={gki_diff.max().item():.6e}, "
                f"mean_diff={gki_diff.mean().item():.6e}"
            )
            print(
                f"[DEBUG] GatherKeyRope_ws vs host_gather: "
                f"max_diff={gkr_diff.max().item():.6e}, "
                f"mean_diff={gkr_diff.mean().item():.6e}"
            )
            print(
                f"[DEBUG] scaled bmm1 row max vs SoftmaxMax: "
                f"max_diff={p_max_diff.max().item():.6e}, "
                f"mean_diff={p_max_diff.mean().item():.6e}"
            )
            print(
                f"[DEBUG] PReduced vs host formula: "
                f"max_diff={p_reduced_diff.max().item():.6e}, "
                f"mean_diff={p_reduced_diff.mean().item():.6e}"
            )
            print(
                f"[DEBUG] SoftmaxI vs host formula: "
                f"max_diff={softmax_i_diff.max().item():.6e}, "
                f"mean_diff={softmax_i_diff.mean().item():.6e}"
            )
            print(
                f"[DEBUG] bmm2Res ReLU check: "
                f"negative={(bmm2_res_ws < 0).sum().item()}, "
                f"min={bmm2_res_ws.min().item():.6e}"
            )
            print(
                f"[DEBUG] reluGradRes vs host formula: "
                f"max_diff={relu_grad_diff.max().item():.6e}, "
                f"mean_diff={relu_grad_diff.mean().item():.6e}"
            )
            relu_max_flat = int(relu_grad_bf16_diff.reshape(-1).argmax().item())
            relu_max_coords = [0] * relu_grad_bf16_diff.ndim
            relu_remaining = relu_max_flat
            for dim_idx in range(relu_grad_bf16_diff.ndim - 1, -1, -1):
                dim_size = relu_grad_bf16_diff.shape[dim_idx]
                relu_max_coords[dim_idx] = relu_remaining % dim_size
                relu_remaining //= dim_size
            relu_max_coords = tuple(relu_max_coords)
            relu_b, relu_s1, relu_g, relu_k = relu_max_coords
            print(
                f"[DEBUG] reluGradRes vs BF16-rounded host formula: "
                f"max_diff={relu_grad_bf16_diff.max().item():.6e}, "
                f"mean_diff={relu_grad_bf16_diff.mean().item():.6e}, "
                f"diff_gt_1e-3={(relu_grad_bf16_diff > 1e-3).sum().item()}, "
                f"diff_gt_1e-2={(relu_grad_bf16_diff > 1e-2).sum().item()}"
            )
            print(
                f"[DEBUG] reluGrad max-error coord={relu_max_coords}: "
                f"actual={debug_actual_rg[relu_max_coords].item():.6e}, "
                f"expected_bf16={debug_relu_grad_bf16[relu_max_coords].item():.6e}, "
                f"unmasked={debug_unmasked_rg[relu_max_coords].item():.6e}, "
                f"bmm2={bmm2_res_ws[relu_max_coords].float().item():.6e}, "
                f"P={debug_p_reduced[relu_b, relu_s1, relu_k].item():.6e}, "
                f"SY={debug_softmax_i[relu_b, relu_s1, relu_k].item():.6e}, "
                f"weight={weights[relu_b, relu_s1, relu_g].float().item():.6e}"
            )
            print(
                f"[DEBUG] reluGrad inferred mask disagreement: "
                f"false_zero={debug_false_zero.sum().item()}, "
                f"false_nonzero={debug_false_nonzero.sum().item()}"
            )
            print(
                f"[DEBUG] C2 dQueryAccum vs host(stored reluGrad): "
                f"max_diff={debug_c2_dq_diff.max().item():.6e}, "
                f"mean_diff={debug_c2_dq_diff.mean().item():.6e}"
            )
            print(
                f"[DEBUG] C2 dk_partial vs host(stored reluGrad): "
                f"max_diff={debug_c2_dk_diff.max().item():.6e}, "
                f"mean_diff={debug_c2_dk_diff.mean().item():.6e}"
            )
            print(
                f"[DEBUG] dWeightOut vs host formula: "
                f"max_diff={d_weight_diff.max().item():.6e}, "
                f"mean_diff={d_weight_diff.mean().item():.6e}"
            )
            print(
                f"[DEBUG] aligned LossPartial sum vs host formula: "
                f"partial_sum={debug_loss_partials.sum().item():.6e}, "
                f"abs_diff={(debug_loss_partials.sum() - debug_loss).abs().item():.6e}"
            )
            print(
                f"[DEBUG] LossOut vs host formula: "
                f"abs_diff={loss_diff.item():.6e}"
            )
            _debug_dump_done = True

        if weights_orig_dtype == torch.float32:
            d_weight = d_weight.float()

        if _PROFILE:
            _t_reduce_end.record()
            _t_scatter_start = torch.npu.Event(enable_timing=True)
            _t_scatter_end = torch.npu.Event(enable_timing=True)
            _t_scatter_start.record()
        d_key_index = d_key_index_flat.reshape(
            B, S2_pad, Nidx2, DQueryIndex
        )[:, :S2]
        if _PROFILE:
            _t_scatter_end.record()

        loss_tensor = loss_out


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
