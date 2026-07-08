# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""DenseLightningIndexerSoftmaxLseV2 kernel (pypto-pro).

Kernel signature matches ops-transformer's DenseLightningIndexerSoftmaxLseV2
(9 inputs + 1 output + workspace + tiling):
    query_index, key_index, weight,
    cu_seq_lens_q, cu_seq_lens_k,
    seq_used_q, seq_used_k,
    cmp_residual_k, metadata,
    softmax_lse, workspace, tiling

All tensor params are declared as raw byte pointers (pl.Ptr[pl.DT_UINT8]).
The actual shape/stride is supplied at runtime through a tiling dataclass:

    @dataclass
    class OpTiling:
        b: int
        s1: int
        s2: int
        n1: int
        d: int

The kernel reconstructs typed tensor views via ``pl.make_tensor``.
"""

from dataclasses import dataclass

import pypto_pro.language as pl


# ================================================================
#  Tile dimensions and constants
# ================================================================
TS = 1
TKV = 128
TD = 128
TN = 128
TILINGDATA_N = 128
TKV_HALF = TKV // 2

Q_F16 = TN * TD * 2
KT_F16 = TKV * TD * 2
QK_F32 = TKV * TS * 4

MA0 = 0
MA1 = Q_F16 * 2
LA0 = 0
RA0 = 0
CA0 = 0


def _align_up(value, align=1024):
    return ((value + align - 1) // align) * align


VB4_KV = TN * TKV_HALF * 4

VA0 = 0
VA1 = _align_up(VA0 + VB4_KV * 2)
VA2 = _align_up(VA1 + TN * 4 * 2)
VA3 = _align_up(VA2 + 32)
VA4 = _align_up(VA3 + 64 * 4)
assert VA4 + 32 <= 248 * 1024
VAT = _align_up(VA4 + 32)

EVENT_IDS_01 = (0, 1)
QK_READY_IDS = (0, 1)

B = pl.Var("B")
S1 = pl.Var("S1")
S2 = pl.Var("S2")
N1 = pl.Var("N1")
D = pl.Var("D")

# Workspace partition: workspace_max at offset 0 (72 fp32 = 288 bytes, padded to 512),
# workspace buffer at offset WS_MAX_BYTES.
WS_MAX_ELEMS = 72
WS_MAX_BYTES = 512


# ================================================================
#  Tiling data — describes the runtime shape of the (rank-erased) inputs
# ================================================================
@dataclass
class OpTiling:
    b: int
    s1: int
    s2: int
    n1: int
    d: int


# ================================================================
#  VF helper functions
# ================================================================

def process_reduce_max_vf(dst_tile, qk_tile, weight_tile, max_tile, tmp_max, s2_i):
    with pl.section_vf():
        vreg_x = vf.RegTensor(dtype=pl.DT_FP32)
        vreg_w = vf.RegTensor(dtype=pl.DT_FP32)
        vreg_relu = vf.RegTensor(dtype=pl.DT_FP32)
        vreg_mul = vf.RegTensor(dtype=pl.DT_FP32)
        vreg_reduce_sum = vf.RegTensor(dtype=pl.DT_FP32)
        vf.duplicate(vreg_reduce_sum, 0)
        vreg_reduce_max = vf.RegTensor(dtype=pl.DT_FP32)

        preg_all = vf.create_mask(pattern="ALL", dtype=pl.DT_FP32)
        preg_max = vf.update_mask(1, dtype=pl.DT_FP32)

        for m in pl.range(TILINGDATA_N):
            vf.load_align(vreg_x, qk_tile, m * TKV_HALF)
            vf.load_align(vreg_w, weight_tile, m, dist="BRC_B32")
            vf.relu(vreg_relu, vreg_x, preg_all)
            vf.mul(vreg_mul, vreg_relu, vreg_w, preg_all)
            vf.add(vreg_reduce_sum, vreg_reduce_sum, vreg_mul, preg_all)
        vf.reduce(vreg_reduce_max, vreg_reduce_sum, preg_all, reduce_type="MAX", merge_mode="ZEROING")
        vf.store_align(dst_tile[s2_i * TKV_HALF], vreg_reduce_sum, preg_all)
        vf.store_align(max_tile, vreg_reduce_max, preg_all)
    tmp_max = pl.max(tmp_max, pl.getval(max_tile, 0))
    pl.setval(max_tile, 0, tmp_max)


def process_lse_vf(lse_vec_tile, reduce_sum_vec_tile, tmp_max, s2_tiles):
    with pl.section_vf():
        vreg_x = vf.RegTensor(dtype=pl.DT_FP32)
        vreg_max = vf.RegTensor(dtype=pl.DT_FP32)
        vreg_sub = vf.RegTensor(dtype=pl.DT_FP32)
        vreg_exp = vf.RegTensor(dtype=pl.DT_FP32)
        vreg_all_sum = vf.RegTensor(dtype=pl.DT_FP32)
        vreg_reduce_sum = vf.RegTensor(dtype=pl.DT_FP32)
        vreg_log = vf.RegTensor(dtype=pl.DT_FP32)
        vreg_lse = vf.RegTensor(dtype=pl.DT_FP32)
        vf.duplicate(vreg_max, tmp_max)
        vf.duplicate(vreg_all_sum, 0)
        preg_all = vf.create_mask(pattern="ALL", dtype=pl.DT_FP32)
        preg_sum = vf.update_mask(1, dtype=pl.DT_FP32)
        for m in pl.range(s2_tiles):
            vf.load_align(vreg_x, reduce_sum_vec_tile, m * TKV_HALF)
            vf.sub(vreg_sub, vreg_x, vreg_max, preg_all)
            vf.exp(vreg_exp, vreg_sub, preg_all)
            vf.add(vreg_all_sum, vreg_all_sum, vreg_exp, preg_all)
        vf.reduce(vreg_reduce_sum, vreg_all_sum, preg_all, reduce_type="SUM", merge_mode="ZEROING")
        vf.log(vreg_log, vreg_reduce_sum, preg_sum)
        vf.add(vreg_lse, vreg_max, vreg_log, preg_sum)
        vf.store_align(lse_vec_tile, vreg_lse, preg_sum)


# ================================================================
#  Inner kernel — receives typed tensor views + tiling
# ================================================================

def dense_lightning_indexer_softmax_lse_v2_inner(
    query_index, key_index, weight, lse_out, workspace_max, workspace, tiling
):
    s1_dim = tiling.s1
    s2_dim = tiling.s2
    n1_dim = tiling.n1
    s1_tiles = (s1_dim + TS - 1) // TS
    s2_tiles = (s2_dim + TKV - 1) // TKV
    core_id = pl.get_block_idx()
    sub_id = pl.get_subblock_idx()
    num_cores = pl.get_block_num()
    s1_per_core = (s1_tiles + num_cores - 1) // num_cores
    s1_core_start = core_id * s1_per_core
    s1_core_end = s1_core_start + s1_per_core

    qk_vec = pl.make_tile_group(
        type=pl.TileType(shape=[TN, TKV_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA0, mutex_ids=[0, 1]
    )
    weight_vec = pl.make_tile_group(
        type=pl.TileType(shape=[1, TN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA1, mutex_ids=[2, 3]
    )
    lse_vec = pl.make_tile_group(
        type=pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA2, mutex_ids=[4]
    )
    reduce_sum_vec = pl.make_tile_group(
        type=pl.TileType(shape=[1, TKV_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA3, mutex_ids=[5]
    )
    max_vec = pl.make_tile_group(
        type=pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA4, mutex_ids=[6]
    )
    tmp_vec = pl.make_tile_group(
        type=pl.TileType(shape=[TN, TKV_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VAT, mutex_ids=[7, 8]
    )

    with pl.section_cube():
        q_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TN, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=MA0, mutex_ids=[0, 1]
        )
        k_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
            addrs=MA1, mutex_ids=[2, 3]
        )
        left = pl.make_tile_group(
            type=pl.TileType(shape=[TN, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=LA0, mutex_ids=[4, 5]
        )
        right = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=RA0, mutex_ids=[6, 7]
        )
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[TN, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ),
            addrs=CA0, mutex_ids=[8, 9]
        )

        work_start = 0
        work_end = 1
        for work_id in pl.range(work_start, work_end):
            task_id = 0
            b_idx = work_id
            for s1_i in pl.range(s1_core_start, s1_core_end):
                q_tile = q_mat.next()
                pl.load(q_tile, query_index, [b_idx, s1_i * TS, 0, 0])
                for s2_i in pl.range(0, s2_tiles + 1):
                    k_tile_nxt = k_mat.next()
                    if s2_i < s2_tiles:
                        pl.load(k_tile_nxt, key_index, [b_idx, s2_i * TKV, 0, 0], is_transpose=True)
                    if s2_i > 0:
                        k_tile_cur = k_mat.previous()
                        left_tile = left.next()
                        right_tile = right.next()
                        acc_tile = acc.next()
                        pl.move(left_tile, q_tile)
                        pl.move(right_tile, k_tile_cur)
                        pl.matmul(acc_tile, left_tile, right_tile)

                        qk_vec_tile = qk_vec.next()
                        pl.move(qk_vec_tile, acc_tile, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
                        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_IDS[task_id % 2])
                    task_id = task_id + 1

    with pl.section_vector():
        work_start = 0
        work_end = 1
        for work_id in pl.range(work_start, work_end):
            b_idx = work_id
            task_id = 1
            for s1_i in pl.range(s1_core_start, s1_core_end + 1):
                tmp_s2_max = -3.4028234663852886e+38
                weight_vec_nxt = weight_vec.next()
                if s1_i < s1_core_end:
                    pl.load(weight_vec_nxt, weight, [b_idx, s1_i * TS, 0])
                if s1_i > 0:
                    reduce_sum_vec_tile = reduce_sum_vec.next()
                    lse_vec_tile = lse_vec.next()
                    for s2_i in pl.range(0, s2_tiles):
                        qk_vec_tile = qk_vec.next()
                        weight_tile_cur = weight_vec.previous()
                        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_IDS[task_id % 2])

                        tmp_vec_tile = tmp_vec.next()
                        max_tile = max_vec.next()
                        process_reduce_max_vf(reduce_sum_vec_tile, qk_vec_tile, weight_tile_cur, max_tile, tmp_s2_max, s2_i)
                        pl.set_validshape(reduce_sum_vec_tile, [1, TKV_HALF])
                        pl.store(workspace, reduce_sum_vec_tile[s2_i * TKV_HALF], [0, TKV * s2_i + TKV_HALF * sub_id])
                        task_id = task_id + 1
                    pl.set_validshape(max_tile, [1, 1])
                    pl.store(workspace_max, max_tile, [0, core_id * 2 + sub_id])
                    pl.load(max_tile, workspace_max, [0, 1 - sub_id])
                    tmp_s2_max = pl.max(tmp_s2_max, pl.getval(max_tile, 0))
                    process_lse_vf(lse_vec_tile, reduce_sum_vec_tile, tmp_s2_max, s2_tiles)
                    pl.store(lse_out, lse_vec_tile, [0, s1_i - 1])

    return


# ================================================================
#  Kernel — dynamic rank: inputs are raw pointers, shapes come from tiling
# ================================================================
@pl.jit(auto_mutex=True)
def dense_lightning_indexer_softmax_lse_v2(
    query_index: pl.Ptr[pl.DT_UINT8],
    key_index: pl.Ptr[pl.DT_UINT8],
    weight: pl.Ptr[pl.DT_UINT8],
    cu_seq_lens_q: pl.Ptr[pl.DT_UINT8],
    cu_seq_lens_k: pl.Ptr[pl.DT_UINT8],
    seq_used_q: pl.Ptr[pl.DT_UINT8],
    seq_used_k: pl.Ptr[pl.DT_UINT8],
    cmp_residual_k: pl.Ptr[pl.DT_UINT8],
    metadata: pl.Ptr[pl.DT_UINT8],
    softmax_lse: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: OpTiling,
):
    # Reconstruct typed tensor views from rank-erased pointers.
    # BSND layout: [B, S, N, D] contiguous.
    tensor_query_index = pl.make_tensor(
        query_index,
        [tiling.b, tiling.s1, tiling.n1, tiling.d],
        [tiling.s1 * tiling.n1 * tiling.d, tiling.n1 * tiling.d, tiling.d, 1],
        dtype=pl.DT_FP16,
    )
    tensor_key_index = pl.make_tensor(
        key_index,
        [tiling.b, tiling.s2, 1, tiling.d],
        [tiling.s2 * tiling.d, tiling.d, tiling.d, 1],
        dtype=pl.DT_FP16,
    )
    tensor_weight = pl.make_tensor(
        weight,
        [tiling.b, tiling.s1, tiling.n1],
        [tiling.s1 * tiling.n1, tiling.n1, 1],
        dtype=pl.DT_FP32,
    )
    tensor_lse_out = pl.make_tensor(
        softmax_lse,
        [tiling.b, tiling.s1],
        [tiling.s1, 1],
        dtype=pl.DT_FP32,
    )

    # Partition the single workspace buffer:
    #   workspace_max  [1, 72] FP32  at offset 0           (288 bytes, padded to WS_MAX_BYTES)
    #   workspace      [1, S2] FP32  at offset WS_MAX_BYTES
    tensor_workspace_max = pl.make_tensor(
        workspace,
        [1, WS_MAX_ELEMS],
        [WS_MAX_ELEMS, 1],
        dtype=pl.DT_FP32,
    )
    tensor_workspace = pl.make_tensor(
        workspace + WS_MAX_BYTES,
        [1, tiling.s2],
        [tiling.s2, 1],
        dtype=pl.DT_FP32,
    )

    dense_lightning_indexer_softmax_lse_v2_inner(
        tensor_query_index,
        tensor_key_index,
        tensor_weight,
        tensor_lse_out,
        tensor_workspace_max,
        tensor_workspace,
        tiling,
    )
