# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""DenseLightningIndexLseV2 kernel.

Computes softmax LSE for dense lightning indexer.
"""

from dataclasses import dataclass

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField


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
VA1 = _align_up(VA0 + VB4_KV * 2, 32)
VA2 = _align_up(VA1 + TN * 4 * 2, 32)
VA3 = _align_up(VA2 + 1, 32)
VA4 = _align_up(VA3 + 1, 32)
KV_CACHE_NUM = _align_up(248 * 1024 - VA4, 32) // 4
assert VA4 + KV_CACHE_NUM * 4 <= 248 * 1024

MAX_UB_S2_TILES = KV_CACHE_NUM // TKV_HALF

EVENT_IDS_01 = (0, 1)
QK_READY_IDS = (0, 1)
QK_FREE_IDS = (2, 3)

WS_MAX_ELEMS = 72
WS_MAX_BYTES = 288
WS_MAX_CORES = 36


# ================================================================
#  Tiling data
# ================================================================
@dataclass
class OpTiling:
    b: int
    s1: int
    s2: int
    n1: int
    d: int
    isLongS2: int
    mask_mode: int
    cmp_ratio: int


# ================================================================
#  TilingKey
# ================================================================
class DenseLiLseTilingKey:
    isLongS2 = TilingKeyField(bits=1, values=[0, 1])


# ================================================================
#  VF helper functions
# ================================================================

@pl.vector_function
def init_data(max_tile):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vreg_x = vf.full(-3.4028234663852886e+38, dtype=pl.DT_FP32)
    vf.store_align(max_tile, vreg_x, preg_all)


@pl.vector_function
def write_scalar_vf(tile, val):
    preg_one = vf.update_mask(1, dtype=pl.DT_FP32)
    vreg_val = vf.full(val, dtype=pl.DT_FP32)
    vf.store_align(tile, vreg_val, preg_one)


@pl.vector_function
def process_reduce_max_vf(dst_tile, qk_tile, weight_tile, max_tile, n1_dim):
    vreg_reduce_sum = vf.full(0, dtype=pl.DT_FP32)
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_max = vf.update_mask(1, dtype=pl.DT_FP32)
    for m in pl.range(n1_dim):
        vreg_x = vf.load_align(qk_tile, m * TKV_HALF)
        vreg_w = vf.load_align(weight_tile, m, dist=pl.LoadDist.BRC_B32)
        vreg_relu = vf.relu(vreg_x, preg_all)
        vreg_mul = vf.mul(vreg_relu, vreg_w, preg_all)
        vreg_reduce_sum = vf.add(vreg_reduce_sum, vreg_mul, preg_all)
    vreg_reduce_max = vf.reduce_max(vreg_reduce_sum, preg_all, merge_mode=pl.MergeMode.ZEROING)
    vreg_tmp_max = vf.load_align(max_tile, 0)
    vreg_tmp_max = vf.max(vreg_tmp_max, vreg_reduce_max, preg_max)
    vf.store_align(dst_tile, vreg_reduce_sum, preg_all)
    vf.store_align(max_tile, vreg_tmp_max, preg_max)


@pl.vector_function
def process_reduce_sum_vf(dst_tile, reduce_sum_vec_tile, tmp_max, s2_tiles):
    vreg_max = vf.full(tmp_max, dtype=pl.DT_FP32)
    vreg_all_sum = vf.full(0, dtype=pl.DT_FP32)
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_sum = vf.update_mask(1, dtype=pl.DT_FP32)
    for m in pl.range(s2_tiles):
        vreg_x = vf.load_align(reduce_sum_vec_tile, m * TKV_HALF)
        vreg_exp = vf.exp_sub(vreg_x, vreg_max, preg_all)
        vreg_all_sum = vf.add(vreg_all_sum, vreg_exp, preg_all)
    vreg_reduce_sum = vf.reduce_sum(vreg_all_sum, preg_all, merge_mode=pl.MergeMode.ZEROING)
    vf.store_align(dst_tile, vreg_reduce_sum, preg_sum)


@pl.vector_function
def process_lse_vf(lse_vec_tile, reduce_res, tmp_s2_max):
    vreg_max = vf.full(tmp_s2_max, dtype=pl.DT_FP32)
    vreg_sum = vf.full(reduce_res, dtype=pl.DT_FP32)
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_lse = vf.update_mask(1, dtype=pl.DT_FP32)
    vreg_log = vf.log(vreg_sum, preg_all)
    vreg_lse = vf.add(vreg_log, vreg_max, preg_all)
    vf.store_align(lse_vec_tile, vreg_lse, preg_lse)


# ================================================================
#  Inner kernel — receives typed tensor views + workspace
# ================================================================

def dense_lightning_indexer_softmax_lse_v2_inner(
    query_index, key_index, weight, lse_out,
    seqused_q, seqused_k, cmp_residual_k,
    workspace_max, workspace_sum, workspace,
    tiling, metadata
):
    s1_dim = tiling.s1
    s2_dim = tiling.s2
    n1_dim = tiling.n1
    is_long_s2 = tiling.isLongS2
    mask_mode = tiling.mask_mode
    cmp_ratio = tiling.cmp_ratio

    s1_tiles = (s1_dim + TS - 1) // TS
    # s2_tiles = (s2_dim + TKV - 1) // TKV
    core_id = pl.get_block_idx()
    sub_id = pl.get_subblock_idx()

    fore_core_num = pl.getval(metadata, 0)
    tail_core_num = pl.getval(metadata, 1)
    b_s1_per_core = pl.getval(metadata, 2)
    b_s1_per_tail_core = pl.getval(metadata, 3)
    total_cores = fore_core_num + tail_core_num
    work_count = 0

    if core_id < fore_core_num:
        work_count = b_s1_per_core
    elif core_id < total_cores:
        work_count = b_s1_per_tail_core

    #【128，64】基本块用于qk的结果，这里UB开两块做db
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
    max_vec = pl.make_tile_group(
        type=pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA3, mutex_ids=[5]
    )
    reduce_sum_vec = pl.make_tile_group(
        type=pl.TileType(shape=[1, KV_CACHE_NUM], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA4, mutex_ids=[6]
    )
    # tmp_vec = pl.make_tile_group(
    #     type=pl.TileType(shape=[TN, TKV_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
    #     addrs=VAT, mutex_ids=[7, 8]
    # )

    with pl.section_cube():
        q_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TN, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=MA0, mutex_ids=[7, 8]
        )
        k_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
            addrs=MA1, mutex_ids=[9, 10]
        )
        left = pl.make_tile_group(
            type=pl.TileType(shape=[TN, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=LA0, mutex_ids=[11, 12]
        )
        right = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=RA0, mutex_ids=[13, 14]
        )
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[TN, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ),
            addrs=CA0, mutex_ids=[15, 16]
        )

        # core_start_idx = 0#pl.getval(work_ranges, core_id * 2)
        # core_end_idx = 0#pl.getval(work_ranges, core_id * 2 + 1)
        #pl.printf("CCCCCC core_start_idx is %d, core_end_idx + 1 is %d, \n", core_start_idx, core_end_idx + 1)
        task_id = 0
        for j in pl.range(0, work_count):
            work_id = core_id + j * total_cores
            b_idx = work_id // s1_dim
            s1_i = work_id % s1_dim
            q_used_size = pl.getval(seqused_q, b_idx) #if q_used else s1_dim
            k_used_size = pl.getval(seqused_k, b_idx) #if k_used else s2_dim
            valid_s1_start = 0
            valid_k_size = k_used_size
            if mask_mode == 3:
                cur_residual_k = 0
                if cmp_ratio > 1:
                    cur_residual_k = pl.getval(cmp_residual_k, b_idx)
                ori_k_size = k_used_size * cmp_ratio + cur_residual_k
                qk_residual = q_used_size - ori_k_size
                valid_s1_start = pl.max(qk_residual, 0)
                valid_k_size = pl.max((s1_i + 1 - qk_residual) // cmp_ratio, 0)
            if s1_i < q_used_size and s1_i >= valid_s1_start:
                s2_tiles = (valid_k_size + TKV - 1) // TKV

                q_tile = q_mat.next()
                pl.load(q_tile, query_index, [b_idx, s1_i * TS, 0, 0])
                for s2_i in pl.range(0, s2_tiles + 1):
                    k_tile_nxt = k_mat.next() # 0-a 1-b 2-a
                    if s2_i < s2_tiles:
                        #pl.printf("-------p offset is %d, s2_i is %d \n ", s2_i * TKV, s2_i)
                        pl.load(k_tile_nxt, key_index, [b_idx, s2_i * TKV, 0, 0], is_transpose=True) # 0-a 1-b 2-a
                    if s2_i > 0:
                        k_tile_cur = k_mat.previous() # 1-a 2-b
                        left_tile = left.next()
                        right_tile = right.next()
                        acc_tile = acc.next()
                        pl.move(left_tile, q_tile)
                        pl.move(right_tile, k_tile_cur) # 1-a
                        pl.matmul(acc_tile, left_tile, right_tile)
                    
                        # for test
                        # #pl.printf("-------out offset is %d, s2_i is %d \n ", (s2_tiles - 1) * TKV, s2_i)
                        # pl.store(workspace, acc_tile, [b_idx, s1_i, 0, (s2_i - 1) * TKV])

                        qk_vec_tile = qk_vec.next()
                        if task_id > 1:
                            # if core_id == 0:
                            #     pl.printf("CCCCCCCCCCCCCCCCCCCCCCCCCC taskid is %d, AIC WAITID is %d, buff_id is %d, s2_i is %d, s1_i is %d \n", task_id, QK_FREE_IDS[task_id % 2], buff_id % 2, s2_i, s1_i)
                            pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=QK_FREE_IDS[task_id % 2])
                            # pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
                        pl.move(qk_vec_tile, acc_tile, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
                        # if core_id == 0:
                        #     pl.printf("CCCCCCCCCCCCCCCCCCCCCCCCCC taskid is %d, AIC setID is %d, buff_id is %d, s2_i is %d, s1_i is %d\n", task_id, QK_READY_IDS[task_id % 2], buff_id % 2, s2_i, s1_i)
                        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_IDS[task_id % 2])
                        # pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
                        task_id = task_id + 1

    with pl.section_vector():
        task_id = 0
        prev_b_idx = 0
        prev_s1_i = 0
        prev_s2_tiles = 0
        prev_is_s1_valid = False

        for j in pl.range(0, work_count + 1):
            work_id = core_id + j * total_cores
            b_idx = work_id // s1_dim
            s1_i = work_id % s1_dim

            max_tile = max_vec.next()
            init_data(max_tile)

            s2_tiles = 0
            is_s1_valid = False
            if j < work_count:
                q_used_size = pl.getval(seqused_q, b_idx)
                k_used_size = pl.getval(seqused_k, b_idx)
                valid_s1_start = 0
                valid_k_size = k_used_size
                if mask_mode == 3:
                    cur_residual_k = 0
                    if cmp_ratio > 1:
                        cur_residual_k = pl.getval(cmp_residual_k, b_idx)
                    ori_k_size = k_used_size * cmp_ratio + cur_residual_k
                    qk_residual = q_used_size - ori_k_size
                    valid_s1_start = pl.max(qk_residual, 0)
                    valid_k_size = pl.max((s1_i + 1 - qk_residual) // cmp_ratio, 0)
                is_s1_valid = s1_i >= valid_s1_start and s1_i < q_used_size
                if is_s1_valid:
                    s2_tiles = (valid_k_size + TKV - 1) // TKV
                    weight_vec_nxt = weight_vec.next()
                    pl.load(weight_vec_nxt, weight, [b_idx, s1_i * TS, 0])

            if j > 0 and prev_is_s1_valid:
                reduce_sum_vec_tile = reduce_sum_vec.next()
                lse_vec_tile = lse_vec.next()
                if is_long_s2 == 0:
                    for s2_i in pl.range(0, prev_s2_tiles):
                        qk_vec_tile = qk_vec.next()
                        weight_tile_cur = weight_vec.previous()
                        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_IDS[task_id % 2])
                        src_tile = reduce_sum_vec_tile[s2_i * TKV_HALF]
                        pl.set_validshape(src_tile, [1, TKV_HALF])
                        process_reduce_max_vf(src_tile, qk_vec_tile, weight_tile_cur, max_tile, n1_dim)
                        if not (j == work_count and s2_i >= prev_s2_tiles - 2):
                            pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_FREE_IDS[task_id % 2])
                        task_id = task_id + 1
                else:
                    num_chunks = (prev_s2_tiles + MAX_UB_S2_TILES - 1) // MAX_UB_S2_TILES
                    for chunk_idx in pl.range(0, num_chunks):
                        chunk_start = chunk_idx * MAX_UB_S2_TILES
                        chunk_end = chunk_start + MAX_UB_S2_TILES
                        if chunk_end > prev_s2_tiles:
                            chunk_end = prev_s2_tiles
                        chunk_tiles = chunk_end - chunk_start
                        for ci in pl.range(0, chunk_tiles):
                            qk_vec_tile = qk_vec.next()
                            weight_tile_cur = weight_vec.previous()
                            pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_IDS[task_id % 2])
                            src_tile = reduce_sum_vec_tile[ci * TKV_HALF]
                            pl.set_validshape(src_tile, [1, TKV_HALF])
                            process_reduce_max_vf(src_tile, qk_vec_tile, weight_tile_cur, max_tile, n1_dim)
                            if not (j == work_count and (chunk_start + ci) >= prev_s2_tiles - 2):
                                pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_FREE_IDS[task_id % 2])
                            task_id = task_id + 1
                        for ci in pl.range(0, chunk_tiles):
                            ws_tile = reduce_sum_vec_tile[ci * TKV_HALF]
                            pl.set_validshape(ws_tile, [1, TKV_HALF])
                            pl.store(workspace, ws_tile, [core_id, (chunk_start + ci) * TKV + TKV_HALF * sub_id])

                pl.set_validshape(max_tile, [1, 1])
                pl.store(workspace_max, max_tile, [0, core_id * 2 + sub_id])
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=4, sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=4, sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                tmp_s2_max = pl.getval(max_tile, 0)
                pl.set_validshape(max_tile, [1, 2])
                pl.load(max_tile, workspace_max, [0, core_id * 2])
                pl.set_validshape(max_tile, [1, 1])
                v0_tmp_max = pl.getval(max_tile, 0)
                v1_tmp_max = pl.getval(max_tile, 1)
                tmp_s2_max = pl.max(v0_tmp_max, v1_tmp_max)

                if is_long_s2 == 0:
                    process_reduce_sum_vf(lse_vec_tile, reduce_sum_vec_tile, tmp_s2_max, prev_s2_tiles)
                else:
                    ws_chunk_sum = 0.0
                    num_chunks = (prev_s2_tiles + MAX_UB_S2_TILES - 1) // MAX_UB_S2_TILES
                    for chunk_idx in pl.range(0, num_chunks):
                        chunk_start = chunk_idx * MAX_UB_S2_TILES
                        chunk_end = chunk_start + MAX_UB_S2_TILES
                        if chunk_end > prev_s2_tiles:
                            chunk_end = prev_s2_tiles
                        chunk_tiles = chunk_end - chunk_start
                        chunk_tile = reduce_sum_vec.next()
                        for ci in pl.range(0, chunk_tiles):
                            src = chunk_tile[ci * TKV_HALF]
                            pl.set_validshape(src, [1, TKV_HALF])
                            pl.load(src, workspace, [core_id, (chunk_start + ci) * TKV + TKV_HALF * sub_id])
                        process_reduce_sum_vf(lse_vec_tile, chunk_tile, tmp_s2_max, chunk_tiles)
                        ws_chunk_sum = ws_chunk_sum + pl.getval(lse_vec_tile, 0)
                    write_scalar_vf(lse_vec_tile, ws_chunk_sum)

                pl.set_validshape(lse_vec_tile, [1, 1])
                pl.store(workspace_sum, lse_vec_tile, [0, core_id * 2 + sub_id])
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                pl.set_validshape(lse_vec_tile, [1, 2])
                pl.load(lse_vec_tile, workspace_sum, [0, core_id * 2])
                v0_tmp_sum = pl.getval(lse_vec_tile, 0)
                v1_tmp_sum = pl.getval(lse_vec_tile, 1)
                reduce_res = v0_tmp_sum + v1_tmp_sum
                process_lse_vf(lse_vec_tile, reduce_res, tmp_s2_max)
                pl.set_validshape(lse_vec_tile, [1, 1])
                pl.store(lse_out, lse_vec_tile, [prev_b_idx, prev_s1_i])
            elif j > 0:
                lse_vec_tile = lse_vec.next()
                write_scalar_vf(lse_vec_tile, -3.4028234663852886e+38)
                pl.set_validshape(lse_vec_tile, [1, 1])
                pl.store(lse_out, lse_vec_tile, [prev_b_idx, prev_s1_i])

            prev_b_idx = b_idx
            prev_s1_i = s1_i
            prev_s2_tiles = s2_tiles
            prev_is_s1_valid = is_s1_valid
    return


# ================================================================
#  Kernel — dynamic rank: inputs are raw pointers, shapes come from tiling
# ================================================================
@pl.jit(auto_mutex=True,
        tiling_key=DenseLiLseTilingKey,
        datatype={
            "query_index": "input_dtype",
        })
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
    tensor_seqused_q = pl.make_tensor(
        seq_used_q,
        [tiling.b, 1],
        [1, 1],
        dtype=pl.DT_INT32,
    )
    tensor_seqused_k = pl.make_tensor(
        seq_used_k,
        [tiling.b, 1],
        [1, 1],
        dtype=pl.DT_INT32,
    )
    tensor_cmp_residual_k = pl.make_tensor(
        cmp_residual_k,
        [tiling.b, 1],
        [1, 1],
        dtype=pl.DT_INT32,
    )
    tensor_metadata = pl.make_tensor(
        metadata,
        [1, 64],
        [64, 1],
        dtype=pl.DT_INT32,
    )
    tensor_workspace_max = pl.make_tensor(
        workspace,
        [1, WS_MAX_ELEMS],
        [WS_MAX_ELEMS, 1],
        dtype=pl.DT_FP32,
    )
    tensor_workspace_sum = pl.make_tensor(
        workspace + WS_MAX_BYTES,
        [1, WS_MAX_ELEMS],
        [WS_MAX_ELEMS, 1],
        dtype=pl.DT_FP32,
    )
    tensor_workspace = pl.make_tensor(
        workspace + WS_MAX_BYTES * 2,
        [WS_MAX_CORES, tiling.s2],
        [tiling.s2, 1],
        dtype=pl.DT_FP32,
    )

    dense_lightning_indexer_softmax_lse_v2_inner(
        tensor_query_index,
        tensor_key_index,
        tensor_weight,
        tensor_lse_out,
        tensor_seqused_q,
        tensor_seqused_k,
        tensor_cmp_residual_k,
        tensor_workspace_max,
        tensor_workspace_sum,
        tensor_workspace,
        tiling,
        tensor_metadata,
    )
