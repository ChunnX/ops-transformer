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

# 一个物理核负责一个完整的（B，S1 分块）任务。这些常量对应官方服务
# 的任务划分：V0 分别发布已 gather 的 Key 和 KeyIndex 数据流，随后由
# C1/V1/C2/V2 消费 task ID % 2 对应的工作区槽位。
S2_WORKSPACE = 1024
N_SPLIT = 128
K_SPLIT = 128
TASK_PIPELINE_DEPTH = 2
V0_TO_C1_P_FLAGS = (0, 1)
V0_TO_C1_SY_FLAGS = (2, 3)
C1_TO_V1_P_FLAGS = (4, 5)
C1_TO_V1_SY_FLAGS = (6, 7)
V1_TO_C2_FLAGS = (8, 9)
# TileLang-Ascend 无法生成官方 A3 kernel 使用的模式 1 AIV 对等待。
# 模式 2 在 AIC 处汇聚两个 AIV 的同一组 flag，实现等价等待。
V1_PSY_TO_AIC_FLAGS = (10, 11)
V1_PSY_RELAY_FLAGS = (12, 13)

DQUERY = 512
DQUERY_INDEX = 128

_kernel_cache = {}
_workspace_cache = {}
_full_valid_aux_cache = {}
_tail_aux_cache = {}
_debug_dump_done = False


def _clear_caches():
    global _kernel_cache, _workspace_cache, _full_valid_aux_cache, _tail_aux_cache
    _kernel_cache.clear()
    _workspace_cache.clear()
    _full_valid_aux_cache.clear()
    _tail_aux_cache.clear()


pass_configs = {
    # 全手动同步：关闭自动同步，消除标量触发的大量 PIPE_ALL（V0 192 + V1 广播 320）
    # 关闭后 AscendSyncInsert 提前返回，核内所有 SetFlag/WaitFlag/PipeBarrier 必须手写
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,

}


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _get_aicore_count() -> int:
    """读取当前 NPU 的物理 Cube core 数量。"""
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
    """完整（B，S1-tile）任务所有权的运行时划分。"""

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
# 独立的 AscendC 翻译工作区。
#
# 原始实现负责一个完整的（B，S1）任务，并按流水线执行 V0（任务 t）、
# C1/V1/C2（任务 t-1）和 V2（任务 t-2）。在任务调度器逐阶段翻译期间，
# 下面的实现有意保留在这个独立文件中；0715.py 仍是精度基线，0720.py
# 仍是独立的显式 L0 实验。
#
# 当前继承的主体是各阶段的精度参考实现。不要单独改变 block 所有权：
# 流水线要求在同一次调度器改写中同步调整 V1 的 P/SY 交换以及
# C2/V2 的任务槽位。
# 单内核：CV 分离架构（0630：gather 已移入内核）
# V 阶段 0：根据稀疏索引 gather Key/KeyIndex/KeyRope 到工作区。
# C 阶段 1：Matmul1（query @ gatherKey^T）+ Matmul2（queryIndex @ gatherKeyIndex^T）。
# V 阶段 1：计算 softmaxI、P_reduced、dWeight 和 reluGrad。
# C 阶段 2：Matmul4（reluGrad @ gatherKeyIndex → dQueryIndex）。
# V 阶段 2：dQueryIndex FP32→FP16 转换，以及 dk_matmul_partial → dKeyIndex 的 scatter。
#   两个操作分别由 AIV0/AIV1 分担以实现负载均衡
# ==============================================================================
@tilelang.jit(
    out_idx=[-4, -3, -2, -1],
    # 这些工作区由主机封装层缓存并显式传入。
    # 如果标记为 workspace_idx，JIT 会分配私有 torch.empty 张量，
    # 并静默忽略调用方传入的缓冲区。
    workspace_idx=[],
    pass_configs=pass_configs,
)
def tilelang_grad_kl_loss_single_kernel(
    B, S1, S2, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
    launch_core_num,
    d_rope, input_dtype, calc_dtype, scale_value, block_n=BLOCK_N,
    full_valid=False, k_scatter_exec=None,
    source_s2=None, sparse_k_stride=None, serial_task_pipeline=False,
):
    source_s2 = S2 if source_s2 is None else source_s2
    sparse_k_stride = K if sparse_k_stride is None else sparse_k_stride
    k_scatter_exec = K if k_scatter_exec is None else k_scatter_exec
    assert 0 < source_s2 <= S2
    assert sparse_k_stride >= K
    assert 0 < k_scatter_exec <= K
    assert k_scatter_exec % V2_SCATTER_BATCH == 0
    s1_block = BLOCK_M
    n_block = block_n
    k_block = BLOCK_K
    p_rows_per_chunk = min(N1, 64)
    p_row_chunk_count = ceil_div(N1, p_rows_per_chunk)
    p_cols_per_chunk = min(n_block, 64)
    p_col_chunk_count = ceil_div(n_block, p_cols_per_chunk)
    # V1 反向计算保持在已经验证精度正确的 32x64 物理分块内。
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
    # 诊断版本将跨 task 流水延迟从 1 改为 0，使 V0/C1/V1/C2 在同一轮
    # 处理同一个 task。task 内的 AIC/AIV 依赖仍由原有 cross flag 保证。
    task_pipeline_lag = 0 if serial_task_pipeline else TASK_PIPELINE_DEPTH - 1
    total_aiv_num = launch_core_num * 2
    # 官方 16 KiB scatter UB 在 ping/pong 之间平均划分：
    # 16 行 × 128 个 FP32 值 × 4 字节 = 每个槽位 8 KiB。
    v2_scatter_batch = V2_SCATTER_BATCH
    # V2 不需要搬运 Cube 计算宽度 K 中的无效尾部。该宽度在主机侧静态
    # 确定，因此 T.serial 能直接生成收缩后的固定上界。
    v2_total_batches = k_scatter_exec // v2_scatter_batch
    v2_base_batches_per_core = v2_total_batches // launch_core_num
    v2_tail_batch_cores = v2_total_batches % launch_core_num
    v2_k_padded = K
    # 宏收集阶段会校验条件表达式两侧的 BufferRegion。即使 full-valid
    # 最终不会生成 GetValue，也必须保留可被 [b, s] 合法索引的二维形状。
    k_real_size_shape = (B, S1)
    valid_k_mask_shape = (B, S1, K)
    dkey_output_rows = B * S2
    assert dkey_output_rows % 32 == 0, (
        "The padded dKey output must preserve 32-row DMA alignment"
    )
    dkey_clear_batch = 32
    dkey_clear_total_batches = dkey_output_rows // dkey_clear_batch
    dkey_finalize_batch = 16
    dkey_finalize_total_batches = dkey_output_rows // dkey_finalize_batch
    loss_reduce_chunk = 128
    # DataCopyPad 按 block 工作。为每个标量 partial 分配独立的对齐 32 字节
    # GM 槽位，并让最终处理阶段只消费完整的 128 值分块；当前后端对短区域
    # copy 的降级转换不正确。
    loss_partial_stride = 8
    loss_partial_span = task_count * 2 * loss_partial_stride
    loss_partial_size = ceil_div(
        loss_partial_span, loss_reduce_chunk
    ) * loss_reduce_chunk
    loss_full_chunk_count = loss_partial_size // loss_reduce_chunk

    @T.macro
    def vector0_gather_task(
        SparseIndices,
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
        """按照与 MergeKv 相同的 subblock 所有权 gather 一个任务。

        两个 AIV 都 gather 三条数据流。每个 AIV 负责当前 1024 行 K tile
        的一个连续半区，随后共同发布由 C1 消费的 mode-2 readiness flag。
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
            # 无效索引通过 UB->UB 补零，MTE2 必须等待 Vector 完成 BF16 zero cast。
            T.set_flag("v", "mte2", 0)
            T.wait_flag("v", "mte2", 0)

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
                        sparse_idx = SparseIndices[bz_task, s1_id, 0, k_id]
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
                    T.set_flag("mte2", "mte3", 0)

                    for chunk_id in T.serial(gather_chunks):
                        cur = chunk_id % 2
                        nxt = (chunk_id + 1) % 2
                        if chunk_id + 1 < gather_chunks:
                            next_start = aiv_start + (chunk_id + 1) * GATHER_CHUNK_SIZE
                            T.wait_flag("mte3", "mte2", nxt)
                            for lane in T.serial(GATHER_CHUNK_SIZE):
                                k_id = next_start + lane
                                sparse_idx = SparseIndices[bz_task, s1_id, 0, k_id]
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

                    # 模式 2 的 AIC wait 会汇聚两个 AIV 的同一组 flag。
                    # 每个 AIV 完成自己的一半后再发布就绪 flag，
                    # 避免 C1 过早观察到尚未完成的工作区。
                    flag_slot = (task_slot + k_outer) % 2
                    if flag_slot == 0:
                        T.set_cross_flag("MTE3", V0_TO_C1_P_FLAGS[0])
                        T.set_cross_flag("MTE3", V0_TO_C1_SY_FLAGS[0])
                    else:
                        T.set_cross_flag("MTE3", V0_TO_C1_P_FLAGS[1])
                        T.set_cross_flag("MTE3", V0_TO_C1_SY_FLAGS[1])

            T.wait_flag("mte3", "mte2", 0)
            T.wait_flag("mte3", "mte2", 1)

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
        """已经验证精度正确的单槽 C1 基线。"""
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
                    flag_slot = (task_slot + k_outer) % 2
                    if flag_slot == 0:
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
                    flag_slot = (task_slot + k_outer) % 2
                    if flag_slot == 0:
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
        """翻译 ProcessVector1 的前半部分。

        AIV0 负责 VectorP 并生成 PReduced，AIV1 负责 VectorSy 并生成
        SoftmaxI。两者遍历同一任务的全部 K chunk，而不是在每次 K launch
        中重复完整 K reduction。
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
                valid_k_size = K if full_valid else KRealSize[bz_task, s1_id]
                if vector_id == 0:
                    # P 和 SY 在不同 AIV 上运行。将两者的分配限定在各自分支内，
                    # 使 UB planner 可以重用相同的空间。
                    v1_p_bmm1 = T.alloc_ub((N1, n_block), calc_dtype)
                    v1_p_max = T.alloc_ub(N1, calc_dtype)
                    v1_p_sum = T.alloc_ub(N1, calc_dtype)
                    # sum 截断后重用于 64 行 partial reduce。
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
                        T.wait_flag("v", "mte2", 1)
                        T.wait_flag("mte3", "v", 0)
                        if n_start < valid_k_size:
                            T.copy(
                                bmm1Res[bz_task, s1_id, 0:N1,
                                        n_start:n_start + n_block],
                                v1_p_bmm1,
                            )
                            if not full_valid and n_start + n_block > valid_k_size:
                                T.copy(
                                    ValidKMask[
                                        bz_task,
                                        s1_id,
                                        n_idx * n_block:(n_idx + 1) * n_block,
                                    ],
                                    v1_p_valid,
                                )
                            T.set_flag("mte2", "v", 1)
                            T.wait_flag("mte2", "v", 1)
                            T.tile.mul(v1_p_bmm1, v1_p_bmm1, scale_value)
                            T.pipe_barrier("v")
                            # mask repeat 使用 uint8_t。128x128 FP32 分块会被
                            # lowering 成 repeat=256 并溢出为 0，因此每条
                            # row-expand 指令最多处理 64 行。
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
                            if not full_valid and n_start + n_block > valid_k_size:
                                for p_mask_row in T.serial(N1):
                                    T.tile.mul(
                                        v1_p_bmm1[p_mask_row, 0:n_block],
                                        v1_p_bmm1[p_mask_row, 0:n_block],
                                        v1_p_valid,
                                    )
                                T.pipe_barrier("v")
                            # 这里使用的 Ascend 快速路径对 64x128 -> 128
                            # 稳定（VectorSy 也使用该路径），但单次 128x128
                            # axis-0 reduction 会破坏结果。这里对齐官方 N-loop
                            # 结构，在 UB 中累加两个 partial reduction。
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
                    # 将当前 BMM2 分块和循环携带的 ReduceSumY 历史放在同一个
                    # 物理分配中。即使该历史在每次 K-block 迭代期间都保持存活，
                    # UB planner 仍可能错误合并分离的分配。
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
                    # AscendC 的 row-expand lowering 将每 64 个 FP32 元素解释为
                            # 一行。将 128 列分块拆分，使 BRCB 恰好读取 Nidx1 个标量，
                    # 并让 mul_mask 保持在物理行范围内。
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
                        if n_start < valid_k_size:
                            T.wait_flag("v", "mte2", 3)
                            T.copy(
                                bmm2Res[bz_task, s1_id, 0:Nidx1,
                                        n_start:n_start + n_block],
                                v1_sy_storage[0:Nidx1, 0:n_block],
                            )
                            if not full_valid and n_start + n_block > valid_k_size:
                                T.copy(
                                    ValidKMask[
                                        bz_task,
                                        s1_id,
                                        n_idx * n_block:(n_idx + 1) * n_block,
                                    ],
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
                            if not full_valid and n_start + n_block > valid_k_size:
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
                        # 得到全局 max 后不再需要 ReduceSumY 历史。将 exp(x-max)
                        # 保存在同一行中，使 normalization 阶段不必重复计算。
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

            if task_slot == 0:
                T.set_cross_flag("MTE3", V1_PSY_TO_AIC_FLAGS[0])
                T.wait_cross_flag(V1_PSY_RELAY_FLAGS[0])
            else:
                T.set_cross_flag("MTE3", V1_PSY_TO_AIC_FLAGS[1])
                T.wait_cross_flag(V1_PSY_RELAY_FLAGS[1])

    @T.macro
    def relay_vector1_psy(task_slot):
        with T.Scope("C"):
            if task_slot == 0:
                T.wait_cross_flag(V1_PSY_TO_AIC_FLAGS[0])
                T.set_cross_flag("FIX", V1_PSY_RELAY_FLAGS[0])
            else:
                T.wait_cross_flag(V1_PSY_TO_AIC_FLAGS[1])
                T.set_cross_flag("FIX", V1_PSY_RELAY_FLAGS[1])

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
        """为一个完整任务翻译 VectorDwDqDk + VectorLoss。

        配对的 AIV 分别负责互不重叠的 Nidx1 半区。只有 Cube relay 观察到
        两个 producer 都完成后，双方才消费 P/SY workspace。ReLU mask 使用
        非原地 Select，避免源/目标别名，同时保持每个 tile 一次向量操作。
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
            # T.macro 的符号收集不能安全地将分配穿过 Python 条件传播。
            # 因此在所有 specialization 中都定义这四个对齐的 4x64 mask
            # 槽位；只有 16 行路径会实际使用它们。
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
            # FP16 reluGrad store 与下一轮 FP32 diff_2d 共享物理 UB。
            # 下一轮的覆写者是 Vector，不是 MTE2，故由 MTE3 直接归还给 V。
            T.set_flag("mte3", "v", 2)
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
                    # 基于 BRCB 的 row expansion 要求行数是 8 的倍数。
                    # 仅对 Nidx1=8 路径广播每个运行时标量。
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
                        if not full_valid:
                            T.copy(
                                ValidKMask[bz_task, s1_id, bw_col_start:bw_col_end],
                                v1_bw_valid,
                            )
                        T.set_flag("mte2", "v", 3)
                        T.wait_flag("mte2", "v", 3)
                        # v1_bw_diff_2d 与上一块正在被 MTE3 读取的
                        # v1_bw_rg_half 复用同一物理 UB，覆写前必须等待归还。
                        T.wait_flag("mte3", "v", 2)

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
                            # 在复用 bmm2 计算 dWeight 前，对每个静态对齐的 4x64
                            # 分组执行 Compare。dWeight 本身仍保持已经验证过的
                            # 单次 16x64 reduction。
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
                            # dWeight 消费未加权的 diff 分块，并跨两个 64 列
                            # 子分块累加。
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
                        T.set_flag("mte3", "v", 2)

                        # 对齐 VectorLoss：在 AIV 对之间拆分外层 K 分块，
                        # 由所属 AIV 消费两个物理子 tile。
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
                            if not full_valid:
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

            T.wait_flag("mte3", "v", 2)
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
        """使用显式 MMA 流水线计算 dKey partial 和 dQuery。"""
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
                # BF16 Cube 要求 reduction 维度为 16。保留逻辑上的 8 行，
                # 只将物理尾部清零一次；后续 RG load 只覆盖前 8 个有效行。
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
        """在 kernel 内由所有 AIV 协作清零 FP32 scatter 目标。"""
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
        """使用所有 core，对每个任务执行批量 GM→UB 和 ping-pong scatter。"""
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
                # 每个 task 仅读取一次真实 K 长度，仅用于保护本 task 的
                # atomic scatter。V2 DMA 范围由编译期 k_scatter_exec 决定。
                valid_k_size = K if full_valid else KRealSize[bz_task, s1_task]

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
        """分配输出 cast，并归约所有 AIV 的 loss partial。"""
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

            # 只有拥有该任务的 AIV 负责 dQuery 转换。将这个很小的输出分给
            # 所有核会给主要负责 scatter 的其他核引入额外标量工作
            # 和事件初始化开销。
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
        # 输入：0-11
        SparseIndices: T.Tensor((B, S1, Nidx2, sparse_k_stride), "int32"),
        KRealSize: T.Tensor(k_real_size_shape, "int32"),
        ValidKMask: T.Tensor(valid_k_mask_shape, calc_dtype),
        Query: T.Tensor((B, S1, N1, DQuery), input_dtype),
        QueryRope: T.Tensor((B, S1, N1, d_rope), input_dtype),
        Key: T.Tensor((B, source_s2, N2, DQuery), input_dtype),
        KeyRope: T.Tensor((B, source_s2, N2, d_rope), input_dtype),
        QueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), input_dtype),
        KeyIndex: T.Tensor((B, source_s2, Nidx2, DQueryIndex), input_dtype),
        Weights: T.Tensor((B, S1, Nidx1), input_dtype),
        SoftmaxMax: T.Tensor((B, N2, S1, G), calc_dtype),
        SoftmaxSum: T.Tensor((B, N2, S1, G), calc_dtype),
        # 工作区：10-22
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
        # 输出：-4..-1
        dQueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), input_dtype),
        dKeyIndexOut: T.Tensor((dkey_output_rows, DQueryIndex), input_dtype),
        dWeightOut: T.Tensor((B, S1, Nidx1), input_dtype),
        LossOut: T.Tensor((1,), calc_dtype),
    ):
        with T.Kernel(launch_core_num, is_npu=True) as (cid, vid):
            for pipe_iter in T.serial(tasks_per_core + task_pipeline_lag):
                if pipe_iter < tasks_per_core:
                    task_current = pipe_iter * launch_core_num + cid
                    if task_current < task_count:
                        bz_current = task_current // S1
                        s1_current = task_current % S1
                        vector0_gather_task(
                            SparseIndices,
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

                if pipe_iter >= task_pipeline_lag:
                    task_prev_iter = pipe_iter - task_pipeline_lag
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
                            relay_vector1_psy(task_prev_slot)
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

            # 所有任务负责人必须完成 C2 后，才能开始全核的确定性 scatter。
            # 这对应官方切换到 ProcessDeterVector2 的阶段边界，也允许空闲的
            # 任务核参与后续处理。
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

    return main

# Gather 预处理：根据 Key/KeyIndex 和 sparseIndices 生成 GatherKey/GatherKeyIndex。
# ==============================================================================
def _preprocess_gather(key, key_index, key_rope, sparse_indices, B, S1, S2, N2, Nidx2, K, DQuery, DQueryIndex, d_rope_orig, dtype):
    device = key.device

    sparse_idx_3d = sparse_indices[:, :, 0, :]
    valid_mask = (sparse_idx_3d >= 0).unsqueeze(-1).to(dtype)
    sparse_idx_safe = sparse_idx_3d.clamp(min=0)

    # 将 key、key_index、key_rope 融合为一次 gather（1 次 cat + 1 次 gather，
    # 替代 3 次 gather）。
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
# Python 层向量计算参考实现，仅作为回退路径保留。
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
# ScatterAdd：通过 FP16 torch.bmm + scatter_add 计算 dKeyIndex。
# ==============================================================================
def _compute_d_key_index(relu_grad, query_index, sparse_indices, B, S1, S2, Nidx1, Nidx2, K, DQueryIndex, device, dtype):
    # 在 Ascend Cube 上执行 FP16 bmm。
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


def _host_sequence_lengths(
    sequence_lengths: Optional[torch.Tensor],
    default_length: int,
    batch_size: int,
) -> Optional[list[int]]:
    if sequence_lengths is None:
        return [default_length] * batch_size
    if isinstance(sequence_lengths, (list, tuple)):
        return [int(length) for length in sequence_lengths]
    if isinstance(sequence_lengths, torch.Tensor) and sequence_lengths.device.type == "cpu":
        return [int(length) for length in sequence_lengths.tolist()]
    return None


def _derive_execution_widths(
    actual_query_lengths: Optional[torch.Tensor],
    actual_key_lengths: Optional[torch.Tensor],
    batch_size: int,
    query_length: int,
    key_length: int,
    top_k: int,
) -> Tuple[int, int]:
    """返回 Cube 执行宽度与 V2 scatter 执行宽度。

    两个宽度都只依赖主机可见的长度。若长度仅在 NPU 上可见，不能为了
    取得 maxRealK 引入同步，保守地让二者均退化为原始 K。
    """
    host_query_lengths = _host_sequence_lengths(
        actual_query_lengths, query_length, batch_size
    )
    host_key_lengths = _host_sequence_lengths(
        actual_key_lengths, key_length, batch_size
    )
    if host_query_lengths is None or host_key_lengths is None:
        return top_k, top_k
    if len(host_query_lengths) != batch_size or len(host_key_lengths) != batch_size:
        return top_k, top_k

    max_real_k = max(
        min(top_k, max(0, key_len - query_len + query_length))
        for query_len, key_len in zip(host_query_lengths, host_key_lengths)
    )
    k_exec = max(
        S2_WORKSPACE,
        ceil_div(max_real_k, S2_WORKSPACE) * S2_WORKSPACE,
    )
    k_scatter_exec = max(
        V2_SCATTER_BATCH,
        ceil_div(max_real_k, V2_SCATTER_BATCH) * V2_SCATTER_BATCH,
    )
    return k_exec, min(k_scatter_exec, k_exec)


def _is_static_full_valid(
    actual_query_lengths: Optional[torch.Tensor],
    actual_key_lengths: Optional[torch.Tensor],
    batch_size: int,
    query_length: int,
    key_length: int,
    top_k: int,
) -> bool:
    """仅在主机侧长度可见且每个查询位置均具有完整 K 时启用快路径。"""
    host_query_lengths = _host_sequence_lengths(
        actual_query_lengths, query_length, batch_size
    )
    host_key_lengths = _host_sequence_lengths(
        actual_key_lengths, key_length, batch_size
    )
    if host_query_lengths is None or host_key_lengths is None:
        return False
    if len(host_query_lengths) != batch_size or len(host_key_lengths) != batch_size:
        return False
    return all(
        key_len - query_len + 1 >= top_k
        for query_len, key_len in zip(host_query_lengths, host_key_lengths)
    )


def _get_full_valid_aux_inputs(
    device: torch.device,
    calc_dtype: torch.dtype,
    batch_size: int,
    query_length: int,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 full-valid specialization 的安全占位输入，并跨调用复用。

    TileLang 会校验条件表达式的两个 BufferRegion；即使 full-valid 代码生成
    最终不会读取 KRealSize，也必须给该形状提供合法的二维占位张量。
    """
    cache_key = (device, calc_dtype, batch_size, query_length, top_k)
    cached = _full_valid_aux_cache.get(cache_key)
    if cached is None:
        cached = (
            torch.full(
                (batch_size, query_length),
                top_k,
                dtype=torch.int32,
                device=device,
            ),
            # full-valid specialization 不读取该张量；这里只提供宏收集阶段
            # 所需的合法三维 BufferRegion，因此无需初始化内容。
            torch.empty(
                (batch_size, query_length, top_k),
                dtype=calc_dtype,
                device=device,
            ),
        )
        _full_valid_aux_cache[cache_key] = cached
    return cached


def _get_tail_aux_inputs(
    device: torch.device,
    calc_dtype: torch.dtype,
    host_query_lengths: list[int],
    host_key_lengths: list[int],
    batch_size: int,
    query_length: int,
    top_k: int,
    execution_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """缓存 host 可见长度对应的尾块元数据，避免每次调用启动 NPU 预处理算子。"""
    cache_key = (
        device,
        calc_dtype,
        tuple(host_query_lengths),
        tuple(host_key_lengths),
        batch_size,
        query_length,
        top_k,
        execution_k,
    )
    cached = _tail_aux_cache.get(cache_key)
    if cached is None:
        real_sizes = [
            [
                min(top_k, max(0, key_len - query_len + query_pos + 1))
                for query_pos in range(query_length)
            ]
            for query_len, key_len in zip(host_query_lengths, host_key_lengths)
        ]
        k_real_size = torch.tensor(
            real_sizes,
            dtype=torch.int32,
            device=device,
        )
        valid_k_mask = (
            torch.arange(execution_k, dtype=torch.int32, device=device).view(
                1, 1, execution_k
            )
            < k_real_size.unsqueeze(-1)
        ).to(calc_dtype).contiguous()
        cached = (k_real_size, valid_k_mask)
        _tail_aux_cache[cache_key] = cached
    return cached


# ==============================================================================
# BSND 布局主入口
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
    source_s2 = S2
    sparse_k_stride = K

    input_dtype = query.dtype
    calc_dtype_str = "float"
    input_dtype_str = "float16" if input_dtype == torch.float16 else "bfloat16"
    weights_orig_dtype = weights.dtype

    if layout_query == "BSND" and sparse_mode in (3,):

        full_valid = _is_static_full_valid(
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            B,
            S1,
            S2,
            K,
        )
        k_scatter_exec = K
        if full_valid:
            # 该 specialization 不读取这两个输入，使用缓存的最小占位张量即可。
            k_real_size, valid_k_mask = _get_full_valid_aux_inputs(
                query.device, torch.float32, B, S1, K
            )
        else:
            k_exec, k_scatter_exec = _derive_execution_widths(
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                B,
                S1,
                S2,
                K,
            )
            top_k = K
            K = k_exec
            host_query_lengths = _host_sequence_lengths(
                actual_seq_lengths_query, S1, B
            )
            host_key_lengths = _host_sequence_lengths(
                actual_seq_lengths_key, S2, B
            )
            if host_query_lengths is not None and host_key_lengths is not None:
                k_real_size, valid_k_mask = _get_tail_aux_inputs(
                    query.device,
                    torch.float32,
                    host_query_lengths,
                    host_key_lengths,
                    B,
                    S1,
                    top_k,
                    K,
                )
            else:
                if actual_seq_lengths_query is not None:
                    actual_q_len = actual_seq_lengths_query.to(
                        device=query.device, dtype=torch.int32
                    )
                else:
                    actual_q_len = torch.full(
                        (B,), S1, dtype=torch.int32, device=query.device
                    )

                if actual_seq_lengths_key is not None:
                    actual_k_len = actual_seq_lengths_key.to(
                        device=query.device, dtype=torch.int32
                    )
                else:
                    actual_k_len = torch.full(
                        (B,), S2, dtype=torch.int32, device=query.device
                    )

                query_positions = torch.arange(
                    S1, dtype=torch.int32, device=query.device
                )
                k_real_size = (
                    actual_k_len.unsqueeze(1)
                    - actual_q_len.unsqueeze(1)
                    + query_positions
                    + 1
                ).clamp_(min=0, max=top_k).contiguous()
                valid_k_mask = (
                    torch.arange(K, dtype=torch.int32, device=query.device).view(
                        1, 1, K
                    )
                    < k_real_size.unsqueeze(-1)
                ).to(torch.float32).contiguous()

        S2_pad = (S2 + VECTOR_BASEN - 1) // VECTOR_BASEN * VECTOR_BASEN

        if weights.dtype == torch.float32:
            weights = weights.to(input_dtype)

        _PROFILE = os.environ.get("SLIG_PROFILE", "0") == "1"
        _DEBUG_REQUESTED = os.environ.get("SLIG_DEBUG", "0") == "1"
        _SERIAL_TASK_PIPELINE = (
            os.environ.get("SLIG_SERIAL_TASK_PIPELINE", "0") == "1"
        )
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
        # Gather 已移入内核的 V0 阶段，不再需要主机侧预处理。


        # 将 rope 维度补齐到 k_block 对齐，并处理 None。
        d_rope = ((d_rope_orig + BLOCK_K - 1) // BLOCK_K) * BLOCK_K if d_rope_orig > 0 else 0
        if query_rope is None:
            query_rope = torch.zeros(B, S1, N1, 0, dtype=input_dtype, device=query.device)
        if key_rope is None:
            key_rope = torch.zeros(B, source_s2, N2, 0, dtype=input_dtype, device=query.device)
        if d_rope > d_rope_orig and d_rope_orig > 0:
            pad_len = d_rope - d_rope_orig
            query_rope = torch.nn.functional.pad(query_rope, (0, pad_len))
            key_rope = torch.nn.functional.pad(key_rope, (0, pad_len))

        # Cube 和 Vector 阶段共用固定的 128 列 K 分块。
        # 核并行以完整的（B，S1）任务为单位，而不是以 K 分块为单位。
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
            B, S1, S2_pad, source_s2, N1, N2, G, Nidx1, Nidx2,
            DQuery, DQueryIndex, K, sparse_k_stride, k_scatter_exec, core_count, d_rope,
            input_dtype_str, calc_dtype_str, scale_value, block_n, full_valid,
            _SERIAL_TASK_PIPELINE,
        )

        if cache_key in _kernel_cache:
            func = _kernel_cache[cache_key]
        else:
            func = tilelang_grad_kl_loss_single_kernel(
                B, S1, S2_pad, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
                core_count, d_rope, input_dtype_str, calc_dtype_str, scale_value, block_n,
                full_valid, k_scatter_exec, source_s2, sparse_k_stride,
                _SERIAL_TASK_PIPELINE,
            )

            # # 将编译出的 AscendC 源码转储到文件，供离线核对手动同步转写
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
            # dKeyIndex 融合：matmul 中间结果（工作区）+ fp32 累加器输出
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
            d_query_index_accum,                        # 12，工作区
            d_key_index_accum,                          # 13，工作区
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

        # DEBUG：定位第一个产生非有限值的阶段。
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
                key, key_index, key_rope, sparse_indices[..., :K],
                B, S1, source_s2, N2, Nidx2, K, DQuery, DQueryIndex, d_rope_orig, query.dtype,
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
            if full_valid:
                print("[DEBUG] FullValidStaticPath: enabled")
            else:
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
            debug_valid_mask = (
                torch.ones((B, S1, K), dtype=torch.float32, device=query.device)
                if full_valid
                else valid_k_mask.float()
            )
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

            # 根据保存的 BF16 reluGrad 重建 C2，用于区分 V1 和 C2 的误差来源。
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
# TND 布局入口
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
