# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging
import math

import torch
import torch_npu
import pypto_pro.language as pl
from typing import Optional


# ================================================================
#  Tile dimensions and constants
# ================================================================
TS = 1
TG = 128
TG_HALF = TG // 2
TKV = 128
TD = 128
D_TOTAL = 512
SCALE = 1.0 / math.sqrt(D_TOTAL)

# Cube tile byte sizes
Q_F16 = TG * D_TOTAL * 2  # [TG,  TD]  FP16 = 32KB
KT_F16 = TKV * TD * 2  # [TKV, TD]  FP16 = 32KB
QK_F32 = TKV * TS * 4  # [TKV, TS]  FP32 = 64KB

# ---- MAT (512KB) ----
MA0 = 0
MA1 = Q_F16 * 2
print(f"MA0: {MA0}, MA1: {MA1}")

# DN: Left holds K [TKV, TD], Right holds Q^T [TD, TS].
LA0 = 0
LA1 = KT_F16
RA0 = 0
RA1 = Q_F16
CA0 = 0
CA1 = QK_F32

# ---- VEC addresses (248KB on a5) ----
CUBE_BUFFER_NUM = 2
VEC_BUFFER_NUM = 2
GATHER_ROW_NUM = 32
VB4_KV = TG_HALF * TKV * 4 * VEC_BUFFER_NUM # [TG_HALF, TKV] FP32 = 64KB
VB4_GATHER = GATHER_ROW_NUM * D_TOTAL * 2 * VEC_BUFFER_NUM # 64KB
VB4_LSE = TG * 4 * VEC_BUFFER_NUM # 1k
VB4_OUT = TKV * 4 * VEC_BUFFER_NUM # 1k
VB4_GATHER_NZ = GATHER_ROW_NUM * D_TOTAL * 2 * VEC_BUFFER_NUM # 64KB

# Cross-core event IDs
QK_READY_FORWARD_IDS = (0, 1)
QK_READY_BARKWARD_IDS = (2, 3)
GATHER_READY_FORWARD_IDS = (4, 5)
GATHER_READY_BARKWARD_IDS = (6, 7)


def _align_up(value, align=1024):
    return ((value + align - 1) // align) * align

def _ceil_div(value, align):
    return (value + align - 1) // align

test_m = 64


VA0 = 0  # qk_vec  [TKV, TS_HALF] FP32
VA1 = _align_up(VA0 + VB4_KV)  # tmp_vec [TKV, TS_HALF] FP32
VA2 = _align_up(VA1 + VB4_GATHER)  # p_f16   [TKV, TS_HALF] FP16
VA3 = _align_up(VA2 + VB4_LSE)
VA4 = _align_up(VA3 + VB4_OUT)
VA5 = _align_up(VA4 + VB4_GATHER_NZ)

print(f"VA0: {VA0}, VA1: {VA1}, VA2: {VA2}, VA3: {VA3}, VA4: {VA4}, VA5: {VA5}")

EVENT_IDS_01 = (0, 1)
EVENT_IDS_23 = (2, 3)
QK_READY_IDS = (0, 1)
P_READY_IDS = (2, 3)
PV_READY_IDS = (4, 5)
QK_MAX_EID = 2
P_MAX_EID = 4
PV_MAX_EID = 6

# B = pl.Var("B")
# N = pl.Var("N")
# Sq = pl.Var('Sq')
# Sk = pl.Var('Sk')
# G = pl.Var('G')
# K = pl.Var('K')
# D = pl.Var('D')
# T1 = pl.Var("T1")
# T2 = pl.Var("T2")
# MAX_SEQ_K = pl.Var("MAX_SEQ_K")
# B1 = pl.Var("B1")

def get_q_offset(constInfo, runInfo):
    offset = []
    if constInfo.is_tnd:
        offset = [runInfo.t1_offset + runInfo.s1_idx, 0, 0]
    else:
        offset = [runInfo.batch_id, runInfo.s1_idx, 0, 0]
    return offset

# no sparse
def get_k_offset(constInfo, runInfo, d_offset):
    if constInfo.is_tnd:
        return None
    else:
        return [runInfo.batch_id, runInfo.ki, 0, d_offset]

def gather_k_offset(constInfo, runInfo, k_index):
    if constInfo.is_tnd:
        return [runInfo.t2_offset + k_index, 0, 0]
        # return None
    else:
        return [runInfo.batch_id, k_index, 0, 0]

def get_sparse_offset(constInfo, runInfo, first_core_half_s2_base_size):
    offset = 0
    if constInfo.is_tnd:
        offset = (runInfo.t1_offset + runInfo.s1_idx) * constInfo.k_length + constInfo.sub_id * first_core_half_s2_base_size
    else:
        offset = runInfo.batch_id * constInfo.s1_size * constInfo.k_length + runInfo.s1_idx * constInfo.k_length + constInfo.sub_id * first_core_half_s2_base_size
    return offset

def get_out_offset(constInfo, runInfo):
    if constInfo.is_tnd:
        return [runInfo.t1_offset + runInfo.s1_idx, 0, runInfo.ki]
        # return None
    else:
        return [runInfo.batch_id, runInfo.s1_idx, runInfo.ki]

def get_lse_offset(constInfo, runInfo):
    if constInfo.is_tnd:
        # [N2, T1, G]
        return [0, runInfo.t1_offset + runInfo.s1_idx, constInfo.sub_id * constInfo.first_g_size]
        # return None
    else:
        # [B, N2, S1, G]
        return [runInfo.batch_id, 0, runInfo.s1_idx, constInfo.sub_id * constInfo.first_g_size]

def get_seqused(seq_dim, has_seqused, seqused_tensor, batch_id):
    seq_length = 0
    if has_seqused:
        seq_length = pl.getval(seqused_tensor, batch_id)
    else:
        seq_length = seq_dim
    return seq_length
    

@pl.pipeline.stage
def gatherK(k_l1_db, k, sparse_indices, gather_nd_db, gather_nz_db, gather_tmp, constInfo, runInfo):
    half_s2_base_size =  _ceil_div(runInfo.s2_base_size, 2)
    # half_s2_base_size =  32
    first_core_half_s2_base_size = half_s2_base_size
    current_core_half_s2_base_size = first_core_half_s2_base_size
    if constInfo.sub_id == 1:
        current_core_half_s2_base_size = runInfo.s2_base_size - first_core_half_s2_base_size
        # current_core_half_s2_base_size = 0
    
    loop_num = (current_core_half_s2_base_size + GATHER_ROW_NUM - 1) // GATHER_ROW_NUM
    key_l1_slot = k_l1_db.next()
    # pl.set_validshape(key_l1_slot, [runInfo.s2_base_size, D_TOTAL])
    # pl.printf("runInfo.s2_base_size: %d\n", runInfo.s2_base_size)
    pl.set_validshape(key_l1_slot, [D_TOTAL, runInfo.s2_base_size])
    
    # if constInfo.sub_id == 0:
    #     loop_num = 1
    # else:
    #     loop_num = 0
    sparse_indices_s1_offset = get_sparse_offset(constInfo, runInfo, first_core_half_s2_base_size)
    for loop_idx in pl.range(loop_num):
        sparse_indices_k_offset = runInfo.ki + loop_idx * GATHER_ROW_NUM
        gather_k_slot = gather_nd_db.next()
        
        if loop_idx == loop_num - 1:
            actual_row_num = current_core_half_s2_base_size - loop_idx * GATHER_ROW_NUM
        else:
            actual_row_num = GATHER_ROW_NUM
        sparse_indices_s1k_offset = sparse_indices_s1_offset + sparse_indices_k_offset
        for i in pl.range(actual_row_num):
            # offset11 = sparse_indices_s1k_offset + i
            # pl.printf("i: %d, offset11: %d\n", i, offset11)
            k_index = pl.getval(sparse_indices, sparse_indices_s1k_offset + i)
            if k_index < 0:
                break
            # pl.printf("i: %d, offset11: %d, k_index: %d\n", i, offset11, k_index)
            # k_index = 0
            gather_k_slot_single_row = gather_k_slot[i * D_TOTAL]
            pl.set_validshape(gather_k_slot_single_row, [1, D_TOTAL])
            if constInfo.is_tnd:
                pl.load(gather_k_slot_single_row, k, [runInfo.t2_offset + k_index, 0, 0], tile_dims=[0, 2])
            else:
                pl.load(gather_k_slot_single_row, k, [runInfo.batch_id, k_index, 0, 0], tile_dims=[2, 3])

        # gather_k_slot_single_row = gather_k_slot[0*D_TOTAL]
        # pl.set_validshape(gather_k_slot_single_row, [actual_row_num, D_TOTAL])
        # pl.load(gather_k_slot_single_row, k, [runInfo.batch_id, 0, 0, 0], tile_dims=[2, 3])
        
        # pl.load(gather_k_slot, k, [runInfo.batch_id, 0, 0, 0], tile_dims=[2, 3])
        # pl.set_validshape(gather_k_slot, [D_TOTAL, actual_row_num])
        # pl.set_validshape(gather_k_slot, [8, D_TOTAL])

        # return
        
        gather_k_nz_slot = gather_nz_db.next()
        # pl.set_validshape(gather_k_nz_slot, [D_TOTAL, actual_row_num])
        pl.set_validshape(gather_k_nz_slot, [actual_row_num, D_TOTAL])
        pl.set_validshape(gather_k_slot, [actual_row_num, D_TOTAL])
        pl.move(gather_k_nz_slot, gather_k_slot) # ND → NZ
        # pl.system.bar_all()
        # if constInfo.sub_id == 0:
        #     pl.dump_data(gather_k_slot, loc=True)
        # pl.store(gather_tmp, gather_k_slot, [constInfo.sub_id * 26, 0])
        # if runInfo.used_t1_index == 1427:
        #     pl.dump_data(gather_k_nz_slot, loc=True)
        # ii = constInfo.sub_id * first_core_half_s2_base_size + loop_idx * GATHER_ROW_NUM
        # pl.printf("constInfo.sub_id * first_core_half_s2_base_size + loop_idx * GATHER_ROW_NUM: %d, actual_row_num: %d\n", ii, actual_row_num)
        # pl.insert(key_l1_slot, gather_k_nz_slot, [0, 0]) # UB → L1
        pl.insert(key_l1_slot, gather_k_nz_slot, [constInfo.sub_id * first_core_half_s2_base_size + loop_idx * GATHER_ROW_NUM, 0]) # UB → L1
    # pl.system.sync_all(core_type=pl.SyncCoreType)
    pass

@pl.pipeline.stage
def compute_qk(q, k, cur_q_slot, k_l1_db, left_db, right_db, acc_db, qk_vec_db, qkGm, constInfo, runInfo):
    """QK matmul stage (Cube). Writes qk_vec."""
    # batch_id = qi // Sq_dim
    batch_id = runInfo.batch_id
    qk_acc = acc_db.next()
    G_align2 = (constInfo.g_size + 1) // 2 * 2
    pl.set_validshape(qk_acc, [G_align2, TKV])
    
    # query复用
    if runInfo.ki == 0:
        if constInfo.is_tnd:
            pl.load(cur_q_slot, q, [runInfo.t1_offset + runInfo.s1_idx, 0, 0], tile_dims=[1, 2])
        else:
            pl.load(cur_q_slot, q, [runInfo.batch_id, runInfo.s1_idx, 0, 0], tile_dims=[2, 3])
    
    key_l1_slot = k_l1_db.next()
    pl.set_validshape(key_l1_slot, [D_TOTAL, runInfo.s2_base_size])
    # pl.set_validshape(key_l1_slot, [runInfo.s2_base_size, D_TOTAL])
    
    # for d_offset in pl.range(384, 512, TD):
    #     # pl.load(key_l1_slot, k, [batch_id, runInfo.ki, 0, d_offset], is_transpose=True, tile_dims=[1, 3])
    #     qk_left = left_db.next()
    #     pl.set_validshape(qk_left, [128, TD])
    #     qk_right = right_db.next()
    #     # pl.set_validshape(qk_right, [TD, 32])
    #     pl.set_validshape(qk_right, [32, TD])
        
    #     pl.move(qk_left, cur_q_slot, [0, 0])
    #     # pl.move(qk_right, key_l1_slot, [d_offset, 0])
    #     pl.move(qk_right, key_l1_slot, [0, d_offset])
    #     # pl.move(qk_left, cur_q_slot)
    #     # pl.move(qk_right, key_l1_slot)
    #     if d_offset == 0:
    #         pl.matmul(qk_acc, qk_left, qk_right)
    #     else:
    #         pl.matmul(qk_acc, qk_left, qk_right)
    #         # pl.matmul_acc(qk_acc, qk_acc, qk_left, qk_right)

    for d_offset in pl.range(0, D_TOTAL, TD):
        # pl.load(key_l1_slot, k, [batch_id, runInfo.ki, 0, d_offset], is_transpose=True, tile_dims=[1, 3])
        qk_left = left_db.next()
        pl.set_validshape(qk_left, [G_align2, TD])
        qk_right = right_db.next()
        # pl.set_validshape(qk_right, [runInfo.s2_base_size, TD])
        pl.set_validshape(qk_right, [TD, runInfo.s2_base_size])
        
        pl.move(qk_left, cur_q_slot, [0, d_offset])
        # pl.move(qk_left, cur_q_slot, [0, 0])
        # pl.move(qk_right, key_l1_slot, [0, d_offset])
        pl.move(qk_right, key_l1_slot, [d_offset, 0])
        # pl.move(qk_left, cur_q_slot)
        # pl.move(qk_right, key_l1_slot)
        if d_offset == 0:
            pl.matmul(qk_acc, qk_left, qk_right)
        else:
            # pl.matmul(qk_acc, qk_left, qk_right)
            pl.matmul_acc(qk_acc, qk_acc, qk_left, qk_right)

    # for d_offset in pl.range(0, D_TOTAL, TD):
    #     # pl.load(key_l1_slot, k, [batch_id, runInfo.ki, 0, d_offset], is_transpose=True, tile_dims=[1, 3])
    #     qk_left = left_db.next()
    #     pl.set_validshape(qk_left, [G_align2, TD])
    #     qk_right = right_db.next()
    #     pl.set_validshape(qk_right, [TD, runInfo.s2_base_size])
        
    #     pl.move(qk_left, cur_q_slot, [0, d_offset])
    #     pl.move(qk_right, key_l1_slot, [d_offset, 0])
    #     # pl.move(qk_left, cur_q_slot)
    #     # pl.move(qk_right, key_l1_slot)
    #     if d_offset == 0:
    #         pl.matmul(qk_acc, qk_left, qk_right)
    #     else:
    #         # pl.matmul(qk_acc, qk_left, qk_right)
    #         pl.matmul_acc(qk_acc, qk_acc, qk_left, qk_right)
    
    qk_vec_slot = qk_vec_db.next()
    # pl.system.wait_cross_croe(pl.PipeType.FIX, event_id=QK_READY_BARKWARD_IDS[taskId % 2], max_event_id=5, sync_mode=pl.CrossCoreSyncMode.TWO_C_ONE_V)
    
    pl.move(qk_vec_slot, qk_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    # pl.system.set_cross_croe(pl.PipeType.FIX, event_id=QK_READY_FORWARD_IDS[taskId % 2], max_event_id=1, mode_id=0)
    # pl.store(qkGm, qk_acc, [0, 0])

@pl.pipeline.stage
def compute_qk_dense(q, k, cur_q_slot, k_l1_db, left_db, right_db, acc_db, qk_vec_db, qkGm, constInfo, runInfo):
    """QK matmul stage (Cube). Writes qk_vec."""
    # batch_id = qi // Sq_dim
    batch_id = runInfo.batch_id
    qk_acc = acc_db.next()
    G_align2 = (constInfo.g_size + 1) // 2 * 2
    pl.set_validshape(qk_acc, [G_align2, TKV])
    
    # query复用
    if runInfo.ki == 0:
        if constInfo.is_tnd:
            pl.load(cur_q_slot, q, [runInfo.t1_offset + runInfo.s1_idx, 0, 0], tile_dims=[1, 2])
        else:
            pl.load(cur_q_slot, q, [runInfo.batch_id, runInfo.s1_idx, 0, 0], tile_dims=[2, 3])
    
    for d_offset in pl.range(0, D_TOTAL, TD):
        cur_k_slot = k_l1_db.next()
        pl.set_validshape(cur_k_slot, [TD, runInfo.s2_base_size])
        
        if constInfo.is_tnd:
            pl.load(cur_k_slot, k, [runInfo.t2_offset + runInfo.ki, 0, d_offset], is_transpose=True, tile_dims=[0, 2])
        else:
            pl.load(cur_k_slot, k, [runInfo.batch_id, runInfo.ki, 0, d_offset], is_transpose=True, tile_dims=[1, 3])
        qk_left = left_db.next()
        pl.set_validshape(qk_left, [G_align2, TD])
        qk_right = right_db.next()
        pl.set_validshape(qk_right, [TD, runInfo.s2_base_size])
        
        pl.move(qk_left, cur_q_slot, [0, d_offset])
        pl.move(qk_right, cur_k_slot)
        if d_offset == 0:
            pl.matmul(qk_acc, qk_left, qk_right)
        else:
            pl.matmul_acc(qk_acc, qk_acc, qk_left, qk_right)
            
    qk_vec_slot = qk_vec_db.next()
    
    pl.move(qk_vec_slot, qk_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)

# Alternative implementation using multiple pre-configured tiles
@pl.vector_function
def _muls_sel_vf_pse_type1_inner(
    src_tiles,
    lse_tiles,
    softmax_l1_tiles,
    scale,
    src_m,
    repeat_times,
    tail_size,
    p_scalar,
):
    preg_full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(tail_size, dtype=pl.DT_FP32)
    
    vreg_reduce_sum_muls = vf.full(0.0, preg_full)
    vreg_reduce_sum_muls_tail = vf.full(0.0, preg_tail)
    
    for m_idx in pl.range(0, src_m):
        vreg_lse = vf.load_align(lse_tiles, m_idx, dist=pl.LoadDist.BRC_B32)
        
        for block_idx in pl.range(0, repeat_times):
            vreg_src = vf.load_align(src_tiles, m_idx * TKV + block_idx * 64)
            vreg_muls = vf.muls(vreg_src, scale, preg_full)
            vreg_exp = vf.exp_sub(vreg_muls, vreg_lse, preg_full)
            vreg_reduce_sum_muls = vf.add(vreg_reduce_sum_muls, vreg_exp, preg_full)
        
        vreg_src_tail = vf.load_align(src_tiles, m_idx * TKV + repeat_times * 64)
        vreg_muls_tail = vf.muls(vreg_src_tail, scale, preg_tail)
        vreg_exp_tail = vf.exp_sub(vreg_muls_tail, vreg_lse, preg_tail)
        vreg_reduce_sum_muls_tail = vf.add(vreg_reduce_sum_muls_tail, vreg_exp_tail, preg_tail)
    
    for block_idx in pl.range(0, repeat_times):
        vreg_reduce_sum_muls = vf.muls(vreg_reduce_sum_muls, p_scalar, preg_full)
        vf.store_align(softmax_l1_tiles, vreg_reduce_sum_muls, preg_full)
    vreg_reduce_sum_muls_tail = vf.muls(vreg_reduce_sum_muls_tail, p_scalar, preg_tail)
    vf.store_align(softmax_l1_tiles[repeat_times * 64], vreg_reduce_sum_muls_tail, preg_full)


def muls_sel_vf_pse_type1_multi_tile(
    src_tiles,
    lse_tiles,
    softmax_l1_tiles,
    scale,
    src_m,
    s2_real_size,
    p_scalar=1.0,
):
    """
    Alternative VF kernel using pre-configured tiles per block.
    """
    FULL_EXE_SIZE = 64
    repeat_times = 0
    tail_size = s2_real_size
    if s2_real_size > FULL_EXE_SIZE:
        repeat_times = 1
        tail_size = s2_real_size - FULL_EXE_SIZE
    
    _muls_sel_vf_pse_type1_inner(
        src_tiles,
        lse_tiles,
        softmax_l1_tiles,
        scale,
        src_m,
        repeat_times,
        tail_size,
        p_scalar,
    )

@pl.pipeline.stage
def compute_softmax_l1_norm(softmax_l1_norm, qk_vec_db, lse_vec_slot, out_vec_db, qk, constInfo, runInfo):
    qk_vec_slot = qk_vec_db.next()
    out_vec_slot = out_vec_db.next()
    if constInfo.current_g_size > 0:
        # pl.dump_data(lse_vec_slot, loc=True)
        
        # batch_id = runInfo.batch_id
        # s1_idx = runInfo.s1_idx
        
        # if runInfo.used_t1_index == 1427:
        #     pl.store(qk, lse_vec_slot, [constInfo.sub_id, 8], tile_dims=[0, 1])
        
        # pl.system.bar_all()
        # if sub_id == 0:
        #     pl.store(qk, qk_vec_slot, [0, 0])
        # pl.system.wait_cross_croe(pl.PipeType.V, event_id=QK_READY_FORWARD_IDS[taskId % 2], max_event_id=1, sync_mode=pl.CrossCoreSyncMode.TWO_C_ONE_V)
        muls_sel_vf_pse_type1_multi_tile(
            qk_vec_slot,
            lse_vec_slot,
            out_vec_slot,
            SCALE,
            constInfo.current_g_size,
            runInfo.s2_base_size,
            constInfo.softmax_l1_norm_scale,
        )
        
        # pl.system.bar_all()
        # pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        # pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        # if sub_id == 0:
        #     pl.store(qk, out_vec_slot, [sub_id * 1, 0])

        pl.set_validshape(out_vec_slot, [1, runInfo.s2_base_size])
        # if runInfo.used_t1_index == 1427:
        #     pl.store(qk, out_vec_slot, [constInfo.sub_id, 0], tile_dims=[0, 1])
        
        if constInfo.is_tnd:
            pl.store(softmax_l1_norm, out_vec_slot, [runInfo.t1_offset + runInfo.s1_idx, 0, runInfo.ki], tile_dims=[1, 2], atomic=pl.AtomicType.AtomicAdd)
        else:
            pl.store(softmax_l1_norm, out_vec_slot, [runInfo.batch_id, runInfo.s1_idx, 0, runInfo.ki], tile_dims=[2, 3], atomic=pl.AtomicType.AtomicAdd)
        # pl.system.set_cross_croe(pl.PipeType.MTE3, event_id=QK_READY_BARKWARD_IDS[taskId % 2], max_event_id=5, mode_id=0)

def get_seq_length(seqused_q, seq_q, seq_k, constInfo, runInfo):
    if constInfo.has_seqused_q:
        for batch_id in pl.range(runInfo.batch_id, constInfo.b_size):
            if runInfo.next_used_t1_offset > runInfo.used_t1_index:
                break
            
            runInfo.batch_id = batch_id + 1
            next_batch_used_t1 = pl.getval(seqused_q, batch_id + 1)
            runInfo.used_t1_offset = runInfo.next_used_t1_offset
            runInfo.next_used_t1_offset = runInfo.next_used_t1_offset + next_batch_used_t1
        runInfo.s1_idx = runInfo.used_t1_index - runInfo.used_t1_offset
        if constInfo.is_tnd:
            runInfo.t1_offset = pl.getval(seq_q, runInfo.batch_id)
            runInfo.t2_offset = pl.getval(seq_k, runInfo.batch_id)
            runInfo.next_t1_offset = pl.getval(seq_q, runInfo.batch_id + 1)
            runInfo.s1_size = runInfo.next_t1_offset - runInfo.t1_offset
            s2_pre = pl.getval(seq_k, runInfo.batch_id)
            s2_next = pl.getval(seq_k, runInfo.batch_id + 1)
            runInfo.s2_size = s2_next - s2_pre
            # pl.printf("runInfo.batch_id: %d, s2_pre: %d, s2_next: %d, runInfo.s2_size: %d, runInfo.t1_offset: %d, runInfo.t2_offset: %d, runInfo.s1_idx: %d\n", runInfo.batch_id, s2_pre, s2_next, runInfo.s2_size, runInfo.t1_offset, runInfo.t2_offset, runInfo.s1_idx)
    else:
        if constInfo.is_tnd:
            # pl.printf("runInfo.batch_id: %d, runInfo.next_t1_offset: %d, runInfo.used_t1_index: %d\n", runInfo.batch_id, runInfo.next_t1_offset, runInfo.used_t1_index)
            for batch_id in pl.range(runInfo.batch_id, constInfo.b_size):
                # pl.printf("batch_id: %d, runInfo.next_t1_offset: %d, runInfo.used_t1_index: %d\n", batch_id, runInfo.next_t1_offset, runInfo.used_t1_index)
                if runInfo.next_t1_offset > runInfo.used_t1_index:
                    break
                
                runInfo.batch_id = batch_id + 1
                next_batch_t1 = pl.getval(seq_q, runInfo.batch_id + 1)
                runInfo.t1_offset = runInfo.next_t1_offset
                runInfo.next_t1_offset = next_batch_t1
                # pl.printf("batch_id: %d, runInfo.next_t1_offset: %d\n", batch_id, runInfo.next_t1_offset)
            runInfo.t2_offset = pl.getval(seq_k, runInfo.batch_id)
            runInfo.s1_idx = runInfo.used_t1_index - runInfo.t1_offset
            runInfo.s1_size = runInfo.next_t1_offset - runInfo.t1_offset
            s2_pre = pl.getval(seq_k, runInfo.batch_id)
            s2_next = pl.getval(seq_k, runInfo.batch_id + 1)
            
            runInfo.s2_size = s2_next - s2_pre
            # pl.printf("runInfo.batch_id: %d, s2_pre: %d, s2_next: %d, runInfo.s2_size: %d\n", runInfo.batch_id, s2_pre, s2_next, runInfo.s2_size)
        else:
            runInfo.batch_id = runInfo.used_t1_index // constInfo.s1_size
            runInfo.s1_idx = runInfo.used_t1_index % constInfo.s1_size

@pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(sync_only=True))
# @pl.jit(auto_mutex=True)
def smla_softmax_l1_sparse_bsnd_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    softmax_lse: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    sparse_indices: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    seqused_q: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    seqused_k: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    cmp_residual_k: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    meta_data: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    mask_mode: pl.DT_INT64,
    cmp_ratio: pl.DT_INT64,
    softmax_l1_norm: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    tiling_data: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    gather_tmp: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    qk: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    B = q.shape[0]
    Sq = q.shape[1]
    G = q.shape[2]
    D = q.shape[3]
    Sk = k.shape[1]
    K = sparse_indices.shape[3]
    
    # pl.printf("mask_mode: %d, cmp_ratio: %d\n", mask_mode, cmp_ratio)

    total_num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()
    sub_id = pl.get_subblock_idx()
    
    has_seqused_q = pl.getval(tiling_data, 0) == 1
    has_seqused_k = pl.getval(tiling_data, 1) == 1
    has_topk_length = pl.getval(tiling_data, 2) == 1
    has_sparse = pl.getval(tiling_data, 4)
    
    sq_tiles = pl.getval(meta_data, 0)
    former_core_process_num = pl.getval(meta_data, 1)
    remain_core_process_num = pl.getval(meta_data, 2)
    remain_core_num = pl.getval(meta_data, 3)
    num_cores = pl.getval(meta_data, 4)
    if core_id >= num_cores:
        current_core_proccess_num = 0
    elif (core_id < total_num_cores - remain_core_num):
        current_core_proccess_num = former_core_process_num
    else:
        current_core_proccess_num = remain_core_process_num
    
    Sq_dim = Sq
    skv_dim = Sk
    # sq_tiles = B * Sq
    firstHalfG = (G + 2 - 1) // 2
    secondHalfG = G - firstHalfG
    currentHalfG = firstHalfG
    if sub_id == 1:
        currentHalfG = secondHalfG
    
    # constInfo: ConstInfo = ConstInfo()
    constInfo = pl.make_tuple(b_size=B, g_size=G, s1_size=Sq_dim, s2_size=skv_dim, is_sparse=has_sparse, k_length=K, mask_mode=mask_mode, cmp_ratio=cmp_ratio, is_tnd=False, has_seqused_q=has_seqused_q, has_seqused_k=has_seqused_k, first_g_size=firstHalfG, current_g_size=currentHalfG, softmax_l1_norm_scale=1.0/G, sub_id=sub_id, core_id=core_id, num_cores=num_cores)
    runInfos = pl.struct_array(1, "RunInfo", batch_id=0, s1_idx=0, s1_size=Sq_dim, s2_size=skv_dim, used_t1_index=0, used_t1_offset=0, next_used_t1_offset=0, t1_offset=0, next_t1_offset=0, t2_offset=0, ki=0, s2_real_size=0, s2_base_size=TKV, task_id=0)
    runInfo = runInfos[0]
    if has_seqused_q:
        runInfo.next_used_t1_offset = pl.getval(seqused_q, 0)

    # common
    qk_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TG_HALF, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=VA0, mutex_ids=[0,1],
        fwd_ids=QK_READY_FORWARD_IDS, bwd_ids=QK_READY_BARKWARD_IDS
    )
    k_l1_db = pl.make_tile_group(type=pl.TileType(
                shape=[D_TOTAL, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN, valid_shape=[-1, -1], compact=1), addrs=MA1, mutex_ids=[12,13],
                fwd_ids=GATHER_READY_FORWARD_IDS, bwd_ids=GATHER_READY_BARKWARD_IDS
                )
        
    with pl.section_cube():
        q_mat_db = pl.make_tile_group(type=pl.TileType(
                shape=[TG, D_TOTAL], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ, valid_shape=[-1, -1], compact=1), addrs=MA0, mutex_ids=[2,3])
        left_db = pl.make_tile_group(
            type=pl.TileType(shape=[TG, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1), addrs=LA0, mutex_ids=[6,7])
        right_db = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1], compact=1), addrs=RA0, mutex_ids=[8,9]
        )
        acc_db = pl.make_tile_group(
            type=pl.TileType(shape=[TG, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1], compact=1), addrs=CA0, mutex_ids=[10,11]
        )

    with pl.section_vector():
        # vec
        gather_nd_db = pl.make_tile_group(
            type=pl.TileType(shape=[GATHER_ROW_NUM, D_TOTAL], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1), addrs=VA1, mutex_ids=[2,3]
        )
        gather_nz_db = pl.make_tile_group(
            type=pl.TileType(shape=[GATHER_ROW_NUM, D_TOTAL], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1], compact=1), addrs=VA4, mutex_ids=[8,9]
        )
        lse_vec_db = pl.make_tile_group(
            type=pl.TileType(shape=[1, TG], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=VA2, mutex_ids=[4,5]
        )
        out_vec_db = pl.make_tile_group(
            type=pl.TileType(shape=[1, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=VA3, mutex_ids=[6,7]
        )
    tick = 0
    q_count = 0
    
    for used_t1_index in pl.range(core_id, sq_tiles, num_cores):
    # for used_t1_index in pl.range(core_id, 1, num_cores):
        # used_t1_index = 1425
        
        runInfo.used_t1_index = used_t1_index
        # runInfo.batch_id = used_t1_index // Sq_dim
        # runInfo.s1_idx = used_t1_index % Sq_dim
        get_seq_length(seqused_q, seqused_q, seqused_q, constInfo, runInfo)
        
        
        runInfo.s2_base_size = TKV
        
        ### 计算s2_real_size
        current_s1_size = get_seqused(Sq_dim, has_seqused_q, seqused_q, runInfo.batch_id)
        current_s2_size = get_seqused(skv_dim, has_seqused_k, seqused_k, runInfo.batch_id)
        s1_idx = runInfo.s1_idx
        if constInfo.is_sparse == 0:
            if constInfo.mask_mode == 0:
                s2_real_size = current_s2_size
            else:
                if constInfo.cmp_ratio > 1:
                    ori_s2_size = current_s2_size * constInfo.cmp_ratio + pl.getval(cmp_residual_k, runInfo.batch_id)
                else:
                    ori_s2_size = current_s2_size * constInfo.cmp_ratio
                s2_real_size = pl.max((ori_s2_size - current_s1_size + s1_idx + 1), 0)
        else:
            if constInfo.mask_mode == 0:
                s2_real_size = pl.min(constInfo.k_length, current_s2_size)
            else:
                if constInfo.cmp_ratio > 1:
                    ori_s2_size = current_s2_size * constInfo.cmp_ratio + pl.getval(cmp_residual_k, runInfo.batch_id)
                else:
                    ori_s2_size = current_s2_size * constInfo.cmp_ratio
                s2_valid_size = pl.max((ori_s2_size - current_s1_size + s1_idx + 1) // constInfo.cmp_ratio, 0)
                s2_real_size = pl.min(constInfo.k_length, s2_valid_size)

        runInfo.s2_real_size = s2_real_size
        ###
        # pl.system.bar_all()
        
        # pl.printf("runInfo.s2_real_size: %d, constInfo.k_length: %d\n", runInfo.s2_real_size, constInfo.k_length)
        
        # sq_off = [batch_id, qi % Sq_dim, 0, 0]
        # pl.printf("core_id: %d, batch_id: %d, sq_tiles: %d, num_cores: %d\n", core_id, batch_id, sq_tiles, num_cores)
        with pl.section_cube():
            cur_q_slot = q_mat_db.next()
            pl.set_validshape(cur_q_slot, [G, D_TOTAL])
        with pl.section_vector():
            lse_vec_slot = lse_vec_db.next()
            if currentHalfG > 0 and runInfo.s2_real_size > 0:
                pl.set_validshape(lse_vec_slot, [1, currentHalfG])
                pl.load(lse_vec_slot, softmax_lse, [runInfo.batch_id, 0, runInfo.s1_idx, constInfo.sub_id * constInfo.first_g_size], tile_dims=[1, 3])
            # if runInfo.used_t1_index == 1427:
            #     pl.store(qk, lse_vec_slot, [constInfo.sub_id + 2, 8], tile_dims=[0, 1])
            #     # pl.printf("runInfo.batch_id: %d, runInfo.s1_idx: %d, constInfo.sub_id: %d\n", runInfo.batch_id, runInfo.s1_idx, constInfo.sub_id)
            #     lse_1 = pl.getval(softmax_lse, runInfo.batch_id * Sq * G + runInfo.s1_idx * G + constInfo.sub_id * constInfo.first_g_size)
            #     pl.printf("first num: %f\n", lse_1)
            
        
        for ki in pl.range(0, runInfo.s2_real_size, TKV):
            runInfo.ki = ki
            if ki + TKV > runInfo.s2_real_size:
                runInfo.s2_base_size =  runInfo.s2_real_size - ki
            # k_off = [batch_id, ki * TKV, 0, 0]
            # qk_vec = qk_vec_db.next()
            # pl.set_validshape(qk_vec, [firstHalfG, TKV])
            with pl.section_vector():
                gatherK(k_l1_db, k, sparse_indices, gather_nd_db, gather_nz_db, gather_tmp, constInfo, runInfo)
            with pl.section_cube():
                compute_qk(q, k, cur_q_slot, k_l1_db, left_db, right_db, acc_db, qk_vec_db, qk, constInfo, runInfo)
            with pl.section_vector():
                compute_softmax_l1_norm(softmax_l1_norm, qk_vec_db, lse_vec_slot, out_vec_db, qk, constInfo, runInfo)
            tick = tick + 1
        q_count = q_count + 1


@pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(sync_only=True))
# @pl.jit(auto_mutex=True)
def smla_softmax_l1_dense_bsnd_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    softmax_lse: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    sparse_indices: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    seqused_q: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    seqused_k: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    cmp_residual_k: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    meta_data: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    mask_mode: pl.DT_INT64,
    cmp_ratio: pl.DT_INT64,
    softmax_l1_norm: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    tiling_data: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    gather_tmp: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    qk: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    B = q.shape[0]
    Sq = q.shape[1]
    G = q.shape[2]
    D = q.shape[3]
    Sk = k.shape[1]
    K = sparse_indices.shape[3]
    
    # pl.printf("mask_mode: %d, cmp_ratio: %d\n", mask_mode, cmp_ratio)

    total_num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()
    sub_id = pl.get_subblock_idx()
    
    has_seqused_q = pl.getval(tiling_data, 0) == 1
    has_seqused_k = pl.getval(tiling_data, 1) == 1
    has_topk_length = pl.getval(tiling_data, 2) == 1
    has_sparse = pl.getval(tiling_data, 4)
    
    sq_tiles = pl.getval(meta_data, 0)
    former_core_process_num = pl.getval(meta_data, 1)
    remain_core_process_num = pl.getval(meta_data, 2)
    remain_core_num = pl.getval(meta_data, 3)
    num_cores = pl.getval(meta_data, 4)
    if core_id >= num_cores:
        current_core_proccess_num = 0
    elif (core_id < total_num_cores - remain_core_num):
        current_core_proccess_num = former_core_process_num
    else:
        current_core_proccess_num = remain_core_process_num
    
    Sq_dim = Sq
    skv_dim = Sk
    # sq_tiles = B * Sq
    firstHalfG = (G + 2 - 1) // 2
    secondHalfG = G - firstHalfG
    currentHalfG = firstHalfG
    if sub_id == 1:
        currentHalfG = secondHalfG
    
    # constInfo: ConstInfo = ConstInfo()
    constInfo = pl.make_tuple(b_size=B, g_size=G, s1_size=Sq_dim, s2_size=skv_dim, is_sparse=has_sparse, k_length=K, mask_mode=mask_mode, cmp_ratio=cmp_ratio, is_tnd=False, has_seqused_q=has_seqused_q, has_seqused_k=has_seqused_k, first_g_size=firstHalfG, current_g_size=currentHalfG, softmax_l1_norm_scale=1.0/G, sub_id=sub_id, core_id=core_id, num_cores=num_cores)
    runInfos = pl.struct_array(1, "RunInfo", batch_id=0, s1_idx=0, s1_size=Sq_dim, s2_size=skv_dim, used_t1_index=0, used_t1_offset=0, next_used_t1_offset=0, t1_offset=0, next_t1_offset=0, t2_offset=0, ki=0, s2_real_size=0, s2_base_size=TKV, task_id=0)
    runInfo = runInfos[0]
    if has_seqused_q:
        runInfo.next_used_t1_offset = pl.getval(seqused_q, 0)

    # common
    qk_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TG_HALF, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=VA0, mutex_ids=[0,1],
        fwd_ids=QK_READY_FORWARD_IDS, bwd_ids=QK_READY_BARKWARD_IDS
    )
        
    with pl.section_cube():
        k_l1_db = pl.make_tile_group(type=pl.TileType(
                shape=[TD, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN, valid_shape=[-1, -1], compact=1), addrs=MA1, mutex_ids=[12,13]
                )
        q_mat_db = pl.make_tile_group(type=pl.TileType(
                shape=[TG, D_TOTAL], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ, valid_shape=[-1, -1], compact=1), addrs=MA0, mutex_ids=[2,3])
        left_db = pl.make_tile_group(
            type=pl.TileType(shape=[TG, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1), addrs=LA0, mutex_ids=[6,7])
        right_db = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1], compact=1), addrs=RA0, mutex_ids=[8,9]
        )
        acc_db = pl.make_tile_group(
            type=pl.TileType(shape=[TG, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1], compact=1), addrs=CA0, mutex_ids=[10,11]
        )

    with pl.section_vector():
        # vec
        gather_nd_db = pl.make_tile_group(
            type=pl.TileType(shape=[GATHER_ROW_NUM, D_TOTAL], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1), addrs=VA1, mutex_ids=[2,3]
        )
        gather_nz_db = pl.make_tile_group(
            type=pl.TileType(shape=[GATHER_ROW_NUM, D_TOTAL], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1], compact=1), addrs=VA4, mutex_ids=[8,9]
        )
        lse_vec_db = pl.make_tile_group(
            type=pl.TileType(shape=[1, TG], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=VA2, mutex_ids=[4,5]
        )
        out_vec_db = pl.make_tile_group(
            type=pl.TileType(shape=[1, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=VA3, mutex_ids=[6,7]
        )
    tick = 0
    q_count = 0
    
    for used_t1_index in pl.range(core_id, sq_tiles, num_cores):
    # for used_t1_index in pl.range(core_id, 1, num_cores):
        # used_t1_index = 1425
        
        runInfo.used_t1_index = used_t1_index
        # runInfo.batch_id = used_t1_index // Sq_dim
        # runInfo.s1_idx = used_t1_index % Sq_dim
        get_seq_length(seqused_q, seqused_q, seqused_q, constInfo, runInfo)
        
        
        runInfo.s2_base_size = TKV
        
        ### 计算s2_real_size
        current_s1_size = get_seqused(Sq_dim, has_seqused_q, seqused_q, runInfo.batch_id)
        current_s2_size = get_seqused(skv_dim, has_seqused_k, seqused_k, runInfo.batch_id)
        s1_idx = runInfo.s1_idx
        if constInfo.is_sparse == 0:
            if constInfo.mask_mode == 0:
                s2_real_size = current_s2_size
            else:
                if constInfo.cmp_ratio > 1:
                    ori_s2_size = current_s2_size * constInfo.cmp_ratio + pl.getval(cmp_residual_k, runInfo.batch_id)
                else:
                    ori_s2_size = current_s2_size * constInfo.cmp_ratio
                s2_real_size = pl.max((ori_s2_size - current_s1_size + s1_idx + 1) // constInfo.cmp_ratio, 0)
                # pl.printf("ori_s2_size: %d, current_s2_size: %d, constInfo.cmp_ratio: %d, s2_real_size: %d\n", ori_s2_size, current_s2_size, constInfo.cmp_ratio, s2_real_size)
        else:
            if constInfo.mask_mode == 0:
                s2_real_size = pl.min(constInfo.k_length, current_s2_size)
            else:
                if constInfo.cmp_ratio > 1:
                    ori_s2_size = current_s2_size * constInfo.cmp_ratio + pl.getval(cmp_residual_k, runInfo.batch_id)
                else:
                    ori_s2_size = current_s2_size * constInfo.cmp_ratio
                s2_valid_size = pl.max((ori_s2_size - current_s1_size + s1_idx + 1) // constInfo.cmp_ratio, 0)
                s2_real_size = pl.min(constInfo.k_length, s2_valid_size)

        runInfo.s2_real_size = s2_real_size
        ###
        # pl.system.bar_all()
        
        # pl.printf("runInfo.s2_real_size: %d, constInfo.k_length: %d\n", runInfo.s2_real_size, constInfo.k_length)
        
        # sq_off = [batch_id, qi % Sq_dim, 0, 0]
        # pl.printf("core_id: %d, batch_id: %d, sq_tiles: %d, num_cores: %d\n", core_id, batch_id, sq_tiles, num_cores)
        with pl.section_cube():
            cur_q_slot = q_mat_db.next()
            pl.set_validshape(cur_q_slot, [G, D_TOTAL])
        with pl.section_vector():
            lse_vec_slot = lse_vec_db.next()
            if currentHalfG > 0 and runInfo.s2_real_size > 0:
                pl.set_validshape(lse_vec_slot, [1, currentHalfG])
                pl.load(lse_vec_slot, softmax_lse, [runInfo.batch_id, 0, runInfo.s1_idx, constInfo.sub_id * constInfo.first_g_size], tile_dims=[1, 3])
        
        for ki in pl.range(0, runInfo.s2_real_size, TKV):
            runInfo.ki = ki
            if ki + TKV > runInfo.s2_real_size:
                runInfo.s2_base_size =  runInfo.s2_real_size - ki
            with pl.section_cube():
                compute_qk_dense(q, k, cur_q_slot, k_l1_db, left_db, right_db, acc_db, qk_vec_db, qk, constInfo, runInfo)
            with pl.section_vector():
                compute_softmax_l1_norm(softmax_l1_norm, qk_vec_db, lse_vec_slot, out_vec_db, qk, constInfo, runInfo)
            tick = tick + 1
        q_count = q_count + 1

@pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(sync_only=True))
# @pl.jit(auto_mutex=True)
def smla_softmax_l1_sparse_bsnd_kernel_tnd(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    softmax_lse: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    sparse_indices: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    seq_q: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    seq_k: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    seqused_q: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    seqused_k: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    cmp_residual_k: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    meta_data: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    mask_mode: pl.DT_INT64,
    cmp_ratio: pl.DT_INT64,
    softmax_l1_norm: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    tiling_data: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    gather_tmp: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    qk: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    B = seq_q.shape[0] - 1
    G = q.shape[1]
    D = q.shape[2]
    K = sparse_indices.shape[2]
    # pl.printf("mask_mode: %d, cmp_ratio: %d, K: %d\n", mask_mode, cmp_ratio, K)

    total_num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()
    sub_id = pl.get_subblock_idx()
    
    has_seqused_q = pl.getval(tiling_data, 0) == 1
    has_seqused_k = pl.getval(tiling_data, 1) == 1
    has_topk_length = pl.getval(tiling_data, 2) == 1
    
    sq_tiles = pl.getval(meta_data, 0)
    former_core_process_num = pl.getval(meta_data, 1)
    remain_core_process_num = pl.getval(meta_data, 2)
    remain_core_num = pl.getval(meta_data, 3)
    num_cores = pl.getval(meta_data, 4)
    if core_id >= num_cores:
        current_core_proccess_num = 0
    elif (core_id < total_num_cores - remain_core_num):
        current_core_proccess_num = former_core_process_num
    else:
        current_core_proccess_num = remain_core_process_num
    
    # sq_tiles = B * Sq
    firstHalfG = (G + 2 - 1) // 2
    secondHalfG = G - firstHalfG
    currentHalfG = firstHalfG
    if sub_id == 1:
        currentHalfG = secondHalfG
    
    # constInfo: ConstInfo = ConstInfo()
    constInfo = pl.make_tuple(b_size=B, g_size=G, s1_size=0, s2_size=0, is_sparse=1, k_length=K, mask_mode=mask_mode, cmp_ratio=cmp_ratio, is_tnd=True, has_seqused_q=has_seqused_q, has_seqused_k=has_seqused_k, first_g_size=firstHalfG, current_g_size=currentHalfG, softmax_l1_norm_scale=1.0/G, sub_id=sub_id, core_id=core_id, num_cores=num_cores)
    runInfos = pl.struct_array(1, "RunInfo", batch_id=0, s1_idx=0, s1_size=0, s2_size=0, used_t1_index=0, used_t1_offset=0, next_used_t1_offset=0, t1_offset=0, next_t1_offset=0, t2_offset=0, ki=0, s2_real_size=0, s2_base_size=TKV, task_id=0)
    runInfo = runInfos[0]
    if has_seqused_q:
        runInfo.next_used_t1_offset = pl.getval(seqused_q, 0)
    if constInfo.is_tnd:
        runInfo.next_t1_offset = pl.getval(seq_q, 1)

    # common
    qk_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TG_HALF, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=VA0, mutex_ids=[0,1],
        fwd_ids=QK_READY_FORWARD_IDS, bwd_ids=QK_READY_BARKWARD_IDS
    )
    k_l1_db = pl.make_tile_group(type=pl.TileType(
                shape=[D_TOTAL, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN, valid_shape=[-1, -1], compact=1), addrs=MA1, mutex_ids=[12,13],
                fwd_ids=GATHER_READY_FORWARD_IDS, bwd_ids=GATHER_READY_BARKWARD_IDS
                )
        
    with pl.section_cube():
        q_mat_db = pl.make_tile_group(type=pl.TileType(
                shape=[TG, D_TOTAL], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ, valid_shape=[-1, -1], compact=1), addrs=MA0, mutex_ids=[2,3])
        left_db = pl.make_tile_group(
            type=pl.TileType(shape=[TG, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1), addrs=LA0, mutex_ids=[6,7])
        right_db = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1], compact=1), addrs=RA0, mutex_ids=[8,9]
        )
        acc_db = pl.make_tile_group(
            type=pl.TileType(shape=[TG, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1], compact=1), addrs=CA0, mutex_ids=[10,11]
        )

    with pl.section_vector():
        # vec
        gather_nd_db = pl.make_tile_group(
            type=pl.TileType(shape=[GATHER_ROW_NUM, D_TOTAL], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1), addrs=VA1, mutex_ids=[2,3]
        )
        gather_nz_db = pl.make_tile_group(
            type=pl.TileType(shape=[GATHER_ROW_NUM, D_TOTAL], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1], compact=1), addrs=VA4, mutex_ids=[8,9]
        )
        lse_vec_db = pl.make_tile_group(
            type=pl.TileType(shape=[1, TG], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=VA2, mutex_ids=[4,5]
        )
        out_vec_db = pl.make_tile_group(
            type=pl.TileType(shape=[1, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=VA3, mutex_ids=[6,7]
        )
    tick = 0
    q_count = 0
    
    for used_t1_index in pl.range(core_id, sq_tiles, num_cores):
    # for used_t1_index in pl.range(core_id, 1, num_cores):
        # used_t1_index = 1425
        
        runInfo.used_t1_index = used_t1_index
        # runInfo.batch_id = used_t1_index // Sq_dim
        # runInfo.s1_idx = used_t1_index % Sq_dim
        get_seq_length(seqused_q, seq_q, seq_k, constInfo, runInfo)
        
        
        runInfo.s2_base_size = TKV
        
        ### 计算s2_real_size
        current_s1_size = get_seqused(runInfo.s1_size, has_seqused_q, seqused_q, runInfo.batch_id)
        current_s2_size = get_seqused(runInfo.s2_size, has_seqused_k, seqused_k, runInfo.batch_id)
        # pl.printf("runInfo.s1_size: %d, runInfo.s2_size: %d, current_s1_size: %d, current_s2_size: %d\n", runInfo.s1_size, runInfo.s2_size, current_s1_size, current_s2_size)
        s1_idx = runInfo.s1_idx
        if constInfo.is_sparse == 0:
            if constInfo.mask_mode == 0:
                s2_real_size = current_s2_size
            else:
                s2_real_size = pl.max((current_s2_size - current_s1_size + s1_idx + 1) // constInfo.cmp_ratio, 0)
        else:
            if constInfo.mask_mode == 0:
                s2_real_size = pl.min(constInfo.k_length, current_s2_size)
            else:
                if constInfo.cmp_ratio > 1:
                    ori_s2_size = current_s2_size * constInfo.cmp_ratio + pl.getval(cmp_residual_k, runInfo.batch_id)
                else:
                    ori_s2_size = current_s2_size * constInfo.cmp_ratio
                s2_valid_size = pl.max((ori_s2_size - current_s1_size + s1_idx + 1) // constInfo.cmp_ratio, 0)
                s2_real_size = pl.min(constInfo.k_length, s2_valid_size)

        runInfo.s2_real_size = s2_real_size
        ###
        # pl.system.bar_all()
        
        # pl.printf("runInfo.s2_real_size: %d, constInfo.k_length: %d\n", runInfo.s2_real_size, constInfo.k_length)
        
        # sq_off = [batch_id, qi % Sq_dim, 0, 0]
        # pl.printf("core_id: %d, batch_id: %d, sq_tiles: %d, num_cores: %d\n", core_id, batch_id, sq_tiles, num_cores)
        with pl.section_cube():
            cur_q_slot = q_mat_db.next()
            pl.set_validshape(cur_q_slot, [G, D_TOTAL])
        with pl.section_vector():
            lse_vec_slot = lse_vec_db.next()
            if currentHalfG > 0 and runInfo.s2_real_size > 0:
                pl.set_validshape(lse_vec_slot, [1, currentHalfG])
                if constInfo.is_tnd:
                    pl.load(lse_vec_slot, softmax_lse, [0, runInfo.t1_offset + runInfo.s1_idx, constInfo.sub_id * constInfo.first_g_size], tile_dims=[1, 2])
                else:
                    pl.load(lse_vec_slot, softmax_lse, [runInfo.batch_id, 0, runInfo.s1_idx, constInfo.sub_id * constInfo.first_g_size], tile_dims=[1, 3])
        
        for ki in pl.range(0, runInfo.s2_real_size, TKV):
            runInfo.ki = ki
            if ki + TKV > runInfo.s2_real_size:
                runInfo.s2_base_size =  runInfo.s2_real_size - ki
            # k_off = [batch_id, ki * TKV, 0, 0]
            # qk_vec = qk_vec_db.next()
            # pl.set_validshape(qk_vec, [firstHalfG, TKV])
            with pl.section_vector():
                gatherK(k_l1_db, k, sparse_indices, gather_nd_db, gather_nz_db, gather_tmp, constInfo, runInfo)
            with pl.section_cube():
                compute_qk(q, k, cur_q_slot, k_l1_db, left_db, right_db, acc_db, qk_vec_db, qk, constInfo, runInfo)
            with pl.section_vector():
                compute_softmax_l1_norm(softmax_l1_norm, qk_vec_db, lse_vec_slot, out_vec_db, qk, constInfo, runInfo)
            tick = tick + 1
        q_count = q_count + 1

@pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(sync_only=True))
# @pl.jit(auto_mutex=True)
def smla_softmax_l1_dense_bsnd_kernel_tnd(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    softmax_lse: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    sparse_indices: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    seq_q: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    seq_k: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    seqused_q: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    seqused_k: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    cmp_residual_k: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    meta_data: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    mask_mode: pl.DT_INT64,
    cmp_ratio: pl.DT_INT64,
    softmax_l1_norm: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    tiling_data: pl.Tensor[[pl.DYNAMIC,], pl.DT_INT32],
    gather_tmp: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    qk: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    B = seq_q.shape[0] - 1
    G = q.shape[1]
    D = q.shape[2]
    K = sparse_indices.shape[2]
    # pl.printf("mask_mode: %d, cmp_ratio: %d, K: %d\n", mask_mode, cmp_ratio, K)

    total_num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()
    sub_id = pl.get_subblock_idx()
    
    has_seqused_q = pl.getval(tiling_data, 0) == 1
    has_seqused_k = pl.getval(tiling_data, 1) == 1
    has_topk_length = pl.getval(tiling_data, 2) == 1
    has_sparse = pl.getval(tiling_data, 4)
    
    sq_tiles = pl.getval(meta_data, 0)
    former_core_process_num = pl.getval(meta_data, 1)
    remain_core_process_num = pl.getval(meta_data, 2)
    remain_core_num = pl.getval(meta_data, 3)
    num_cores = pl.getval(meta_data, 4)
    if core_id >= num_cores:
        current_core_proccess_num = 0
    elif (core_id < total_num_cores - remain_core_num):
        current_core_proccess_num = former_core_process_num
    else:
        current_core_proccess_num = remain_core_process_num
    
    # sq_tiles = B * Sq
    firstHalfG = (G + 2 - 1) // 2
    secondHalfG = G - firstHalfG
    currentHalfG = firstHalfG
    if sub_id == 1:
        currentHalfG = secondHalfG
    
    # constInfo: ConstInfo = ConstInfo()
    constInfo = pl.make_tuple(b_size=B, g_size=G, s1_size=0, s2_size=0, is_sparse=has_sparse, k_length=K, mask_mode=mask_mode, cmp_ratio=cmp_ratio, is_tnd=True, has_seqused_q=has_seqused_q, has_seqused_k=has_seqused_k, first_g_size=firstHalfG, current_g_size=currentHalfG, softmax_l1_norm_scale=1.0/G, sub_id=sub_id, core_id=core_id, num_cores=num_cores)
    runInfos = pl.struct_array(1, "RunInfo", batch_id=0, s1_idx=0, s1_size=0, s2_size=0, used_t1_index=0, used_t1_offset=0, next_used_t1_offset=0, t1_offset=0, next_t1_offset=0, t2_offset=0, ki=0, s2_real_size=0, s2_base_size=TKV, task_id=0)
    runInfo = runInfos[0]
    if has_seqused_q:
        runInfo.next_used_t1_offset = pl.getval(seqused_q, 0)
    if constInfo.is_tnd:
        runInfo.next_t1_offset = pl.getval(seq_q, 1)

    # common
    qk_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TG_HALF, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=VA0, mutex_ids=[0,1],
        fwd_ids=QK_READY_FORWARD_IDS, bwd_ids=QK_READY_BARKWARD_IDS
    )
        
    with pl.section_cube():
        k_l1_db = pl.make_tile_group(type=pl.TileType(
                shape=[TD, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN, valid_shape=[-1, -1], compact=1), addrs=MA1, mutex_ids=[12,13]
                )
        q_mat_db = pl.make_tile_group(type=pl.TileType(
                shape=[TG, D_TOTAL], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ, valid_shape=[-1, -1], compact=1), addrs=MA0, mutex_ids=[2,3])
        left_db = pl.make_tile_group(
            type=pl.TileType(shape=[TG, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1), addrs=LA0, mutex_ids=[6,7])
        right_db = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1], compact=1), addrs=RA0, mutex_ids=[8,9]
        )
        acc_db = pl.make_tile_group(
            type=pl.TileType(shape=[TG, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1], compact=1), addrs=CA0, mutex_ids=[10,11]
        )

    with pl.section_vector():
        # vec
        gather_nd_db = pl.make_tile_group(
            type=pl.TileType(shape=[GATHER_ROW_NUM, D_TOTAL], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1), addrs=VA1, mutex_ids=[2,3]
        )
        gather_nz_db = pl.make_tile_group(
            type=pl.TileType(shape=[GATHER_ROW_NUM, D_TOTAL], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1], compact=1), addrs=VA4, mutex_ids=[8,9]
        )
        lse_vec_db = pl.make_tile_group(
            type=pl.TileType(shape=[1, TG], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=VA2, mutex_ids=[4,5]
        )
        out_vec_db = pl.make_tile_group(
            type=pl.TileType(shape=[1, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=VA3, mutex_ids=[6,7]
        )
    tick = 0
    q_count = 0
    
    for used_t1_index in pl.range(core_id, sq_tiles, num_cores):
    # for used_t1_index in pl.range(core_id, 1, num_cores):
        # used_t1_index = 1425
        
        runInfo.used_t1_index = used_t1_index
        # runInfo.batch_id = used_t1_index // Sq_dim
        # runInfo.s1_idx = used_t1_index % Sq_dim
        get_seq_length(seqused_q, seq_q, seq_k, constInfo, runInfo)
        
        
        runInfo.s2_base_size = TKV
        
        ### 计算s2_real_size
        current_s1_size = get_seqused(runInfo.s1_size, has_seqused_q, seqused_q, runInfo.batch_id)
        current_s2_size = get_seqused(runInfo.s2_size, has_seqused_k, seqused_k, runInfo.batch_id)
        # pl.printf("runInfo.s1_size: %d, runInfo.s2_size: %d, current_s1_size: %d, current_s2_size: %d\n", runInfo.s1_size, runInfo.s2_size, current_s1_size, current_s2_size)
        s1_idx = runInfo.s1_idx
        if constInfo.is_sparse == 0:
            if constInfo.mask_mode == 0:
                s2_real_size = current_s2_size
            else:
                if constInfo.cmp_ratio > 1:
                    ori_s2_size = current_s2_size * constInfo.cmp_ratio + pl.getval(cmp_residual_k, runInfo.batch_id)
                else:
                    ori_s2_size = current_s2_size * constInfo.cmp_ratio
                s2_real_size = pl.max((ori_s2_size - current_s1_size + s1_idx + 1) // constInfo.cmp_ratio, 0)
                # pl.printf("ori_s2_size: %d, current_s2_size: %d, constInfo.cmp_ratio: %d, s2_real_size: %d\n", ori_s2_size, current_s2_size, constInfo.cmp_ratio, s2_real_size)
        else:
            if constInfo.mask_mode == 0:
                s2_real_size = pl.min(constInfo.k_length, current_s2_size)
            else:
                if constInfo.cmp_ratio > 1:
                    ori_s2_size = current_s2_size * constInfo.cmp_ratio + pl.getval(cmp_residual_k, runInfo.batch_id)
                else:
                    ori_s2_size = current_s2_size * constInfo.cmp_ratio
                s2_valid_size = pl.max((ori_s2_size - current_s1_size + s1_idx + 1) // constInfo.cmp_ratio, 0)
                s2_real_size = pl.min(constInfo.k_length, s2_valid_size)

        runInfo.s2_real_size = s2_real_size
        ###
        # pl.system.bar_all()
        
        # pl.printf("runInfo.s2_real_size: %d, constInfo.k_length: %d\n", runInfo.s2_real_size, constInfo.k_length)
        
        # sq_off = [batch_id, qi % Sq_dim, 0, 0]
        # pl.printf("core_id: %d, batch_id: %d, sq_tiles: %d, num_cores: %d\n", core_id, batch_id, sq_tiles, num_cores)
        with pl.section_cube():
            cur_q_slot = q_mat_db.next()
            pl.set_validshape(cur_q_slot, [G, D_TOTAL])
        with pl.section_vector():
            lse_vec_slot = lse_vec_db.next()
            if currentHalfG > 0 and runInfo.s2_real_size > 0:
                pl.set_validshape(lse_vec_slot, [1, currentHalfG])
                if constInfo.is_tnd:
                    pl.load(lse_vec_slot, softmax_lse, [0, runInfo.t1_offset + runInfo.s1_idx, constInfo.sub_id * constInfo.first_g_size], tile_dims=[1, 2])
                else:
                    pl.load(lse_vec_slot, softmax_lse, [runInfo.batch_id, 0, runInfo.s1_idx, constInfo.sub_id * constInfo.first_g_size], tile_dims=[1, 3])
            # pl.system.bar_all()
        
        for ki in pl.range(0, runInfo.s2_real_size, TKV):
            runInfo.ki = ki
            if ki + TKV > runInfo.s2_real_size:
                runInfo.s2_base_size =  runInfo.s2_real_size - ki
            with pl.section_cube():
                compute_qk_dense(q, k, cur_q_slot, k_l1_db, left_db, right_db, acc_db, qk_vec_db, qk, constInfo, runInfo)
            with pl.section_vector():
                compute_softmax_l1_norm(softmax_l1_norm, qk_vec_db, lse_vec_slot, out_vec_db, qk, constInfo, runInfo)
            tick = tick + 1
        q_count = q_count + 1

def generate_sparse_indices_loop(
    b: int, 
    sq: int, 
    skv: int, 
    k: int, 
    maskmode: int, 
    cmp_ratio: int,
    layout = "BSND",
    seq_q = None,
    seq_k = None
) -> torch.Tensor:
    """
    逐循环版本生成 sparse_indices。
    """
    # 初始化全 0 的 tensor，shape 为 [b, sq, 1, k]
    sparse_indices = torch.zeros(b, sq, 1, k, dtype=torch.int32)
    
    for b_id in range(b):
        cur_sq = sq if layout == "BSND" else seq_q[b_id]
        cur_skv = skv if layout == "BSND" else seq_k[b_id]
        for s1_idx in range(cur_sq):
            if maskmode == 0:
                # 有效长度为 k 跟 cur_skv 的最小值
                limit = min(cur_skv, k)
                # 生成 [0, cur_skv) 范围内的随机整数
                rand_vals = torch.randint(0, cur_skv, (limit,), dtype=torch.int32)
                # rand_vals = torch.arange(limit, dtype=torch.int32)
                sparse_indices[b_id, s1_idx, 0, :limit] = rand_vals
                
            elif maskmode == 3:
                # 根据公式计算 actual_sk
                actual_sk = max((cur_skv - cur_sq + s1_idx + 1) // cmp_ratio, 0)
                limit = min(actual_sk, k)
                
                if limit > 0:
                    # 生成 [0, actual_sk) 范围内的随机整数
                    rand_vals = torch.randint(0, actual_sk, (limit,), dtype=torch.int32)
                    # rand_vals = torch.arange(limit, dtype=torch.int32)
                    sparse_indices[b_id, s1_idx, 0, :limit] = rand_vals
                    
    return sparse_indices

def compute_softmax_l1_norm_loop(
    q: torch.Tensor,                              # [B, Sq, G, D], FP16
    k: torch.Tensor,                              # [B, Sk, 1, D], FP16
    softmax_lse: torch.Tensor,                    # [B, 1, Sq, G], FP32
    seqused_q: Optional[torch.Tensor] = None,     # [B], INT32 or None
    seqused_k: Optional[torch.Tensor] = None,     # [B], INT32 or None
    cmp_residual_k: Optional[torch.Tensor] = None,# [B], INT32 or None
    sparse_indices: Optional[torch.Tensor] = None,# [B, Sq, 1, K], INT32 or None
    mask_mode: int = 0,                           # 掩码模式
    cmp_ratio: int = 1,                           # 压缩比例
    core_num = 32,
    layout = "BSND",
    seq_q = None,
    seq_k = None,
    D: int = 512,
) -> torch.Tensor:
    """
    逐循环版本（严格对应公式，用于验证向量化版本的正确性）。
    """
    B, Sq, G, _ = q.shape
    Sk = k.shape[1]
    device = q.device

    metadata = torch.zeros(1024, dtype=torch.int32, device=device)
    if seqused_q is not None:
        usedQ = seqused_q
    elif layout == "BSND":
        usedQ = [Sq for _ in range(B)]
    else:
        usedQ = seq_q.numpy()
    total_t1 = sum(usedQ)
    if total_t1 <= core_num:
        used_core_num = total_t1
        remain_core_num = 0
        former_core_process_num = 1
        remain_core_process_num = 0
    else:
        former_core_process_num = math.ceil(total_t1 / core_num)
        remain_core_process_num = former_core_process_num - 1
        former_core_num = total_t1 - core_num * remain_core_process_num
        remain_core_num = core_num - former_core_num
        used_core_num = core_num
    metadata[0] = total_t1
    metadata[1] = former_core_process_num
    metadata[2] = remain_core_process_num
    metadata[3] = remain_core_num
    metadata[4] = used_core_num
    
    if sparse_indices is not None:
        k_length = sparse_indices.shape[-1]
    else:
        k_length = Sk
    
    softmax_l1_norm = torch.zeros(B, Sq, 1, k_length, dtype=torch.float32, device=device)
    for b_id in range(B):
        # 处理 seqused 为空的情况
        sq_len = int(seqused_q[b_id].item() if seqused_q is not None else (Sq if layout == "BSND" else seq_q[b_id].item()))
        sk_len = int(seqused_k[b_id].item() if seqused_k is not None else (Sk if layout == "BSND" else seq_k[b_id].item()))

        for s1_id in range(sq_len):
            query = q[b_id, s1_id, :, :].float()       # [G, D]

            # ---------- mask_mode 逻辑 ----------
            if mask_mode == 0:
                actual_sk = sk_len
            else:
                # 根据公式计算 actual_sk，并确保不超过当前 batch 的有效 sk_len
                if cmp_ratio > 1:
                    ori_sk_len = sk_len * cmp_ratio + cmp_residual_k[b_id]
                else:
                    ori_sk_len = sk_len * cmp_ratio
                actual_sk = max((ori_sk_len - sq_len + s1_id + 1) // cmp_ratio, 0)
            
            # ---------- sparse_indices 逻辑 ----------
            if sparse_indices is not None:
                K = sparse_indices.shape[-1]
                # actual_sk = min(actual_sk, K)
                actual_sk = min(actual_sk, K)
                
                if actual_sk > 0:
                    # 取出对应的索引: [actual_sk]
                    indices = sparse_indices[b_id, s1_id, 0, :actual_sk]
                    # 使用高级索引提取 key: [actual_sk, D]
                    key = k[b_id, indices, 0, :].float()
                else:
                    # actual_sk 为 0 时，构造一个空矩阵用于后续计算
                    key = k[b_id, :0, 0, :].float()
            else:
                # 截取有效的 key
                key = k[b_id, :actual_sk, 0, :].float()    # [actual_sk, D]

            # print(f"key: {key}")
            # return key, metadata
            qk = query @ key.T                          # [G, actual_sk]
            # print(f"golden qk: {qk}")
            # return qk, metadata
                
            qk_scale = qk * (1.0 / math.sqrt(D))        # [G, actual_sk]

            lse = softmax_lse[b_id, 0, s1_id, :]        # [G]
            # if b_id == 1 and (s1_id == 935 or s1_id == 934 or s1_id == 936):
            #     print(f"lse0: {lse[:64]}")
            #     print(f"lse1: {lse[64:]}")
            p = torch.exp(qk_scale - lse.unsqueeze(1))  # [G, actual_sk]

            softmax_l1 = p.sum(dim=0) / G               # [actual_sk]
            # if b_id == 1 and s1_id == 935:
            #     print(f"v0: {p[:64].sum(dim=0) / G}, v1: {p[64:].sum(dim=0) / G}")
            
            # 只写入实际计算出的 actual_sk 长度，其余保持初始值 0
            if actual_sk > 0:
                softmax_l1_norm[b_id, s1_id, 0, :actual_sk] = softmax_l1[:actual_sk]

    return softmax_l1_norm, metadata

def find_mismatched_elements(actual, expected, rtol=5e-3, atol=5e-3):
    """
    找出所有不满足 assert_close 容差条件的异常点，打印索引和值。
    判断公式与 torch 内部一致: |actual - expected| > atol + rtol * |expected|
    """
    actual_flat = actual.cpu().flatten().to(torch.float32)
    expected_flat = expected.cpu().flatten().to(torch.float32)

    # 计算绝对误差
    abs_diff = torch.abs(actual_flat - expected_flat)

    # 计算容差阈值: atol + rtol * |expected|
    tolerance = atol + rtol * torch.abs(expected_flat)

    # 布尔掩码：True 表示该点不满足精度要求
    mismatch_mask = abs_diff > tolerance

    # 额外处理 NaN 和 Inf 的情况
    nan_mask = torch.isnan(actual_flat) | torch.isnan(expected_flat)
    inf_mask = torch.isinf(actual_flat) | torch.isinf(expected_flat)
    mismatch_mask = mismatch_mask | nan_mask | inf_mask

    mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False).flatten()

    total = actual_flat.numel()
    mismatch_count = mismatch_indices.numel()

    print(f"总元素数: {total}")
    print(f"异常点数: {mismatch_count} ({mismatch_count / total * 100:.4f}%)")
    print("-" * 80)

    if mismatch_count > 0:
        print(f"{'Index':<15} {'Actual':<20} {'Expected':<20} {'AbsDiff':<20} {'Tolerance':<20}")
        print("-" * 80)
        print_count = 50
        for idx in mismatch_indices:
            i = idx.item()
            a = actual_flat[i].item()
            e = expected_flat[i].item()
            d = abs_diff[i].item()
            t = tolerance[i].item()
            print(f"{i:<15} {a:<20.8f} {e:<20.8f} {d:<20.8f} {t:<20.8f}")
            print_count -= 1
            if print_count <= 0:
                break

        print("-" * 80)
        # 打印前 10 个异常点后提示是否继续（可选）
        if mismatch_count > 50:
            print(f"(仅显示前 50 个异常点，共 {mismatch_count} 个)")

    return mismatch_count == 0

def test_smla_softmax_l1_bs_sparse_a5():
    device = "npu:0"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    for b, sq, g, skv, d, num_cores, maskmode, cmp_ratio, k, has_seqused_q, has_seqused_k, has_topk_length, layout in [
        # (2, 1024, 128, 1024, D_TOTAL, 32, 0, 1, 1024, True, True, False, "BSND"),
        # (2, 1024, 128, 1024, D_TOTAL, 32, 3, 5, 1024, True, True, False, "BSND"),
        (2, 1024, 128, 1024, D_TOTAL, 10, 3, 5, 1024, True, True, False, "BSND"),
        (2, 4096, 64, 1024, D_TOTAL, 10, 0, 4, 512, False, False, False, "BSND"),
    ]:
        logging.info("\nFA-BSND-DN-A5 (b=%s, sq=%s, g=%s, skv=%s, d=%s) cores=%s", b, sq, g, skv, d, num_cores)
        query = torch.rand((b, sq, g, d)
                           , device=device, dtype=torch.float16)
        
        # # 假设 a 是已有的矩阵，这里用随机数模拟
        # query = torch.randn((128, 512), dtype=torch.float16)
        # # 1. 将整个矩阵清零
        # query.zero_()
        # # 2. 将左上角 [128, 128] 的部分设为单位矩阵
        # query[:, :128] = torch.eye(128, dtype=query.dtype, device=query.device)
        # query[:, 128:256] = torch.eye(128, dtype=query.dtype, device=query.device)
        # query[:, 256:384] = torch.eye(128, dtype=query.dtype, device=query.device)
        # query[:, 384:] = torch.eye(128, dtype=query.dtype, device=query.device)
        # query = query.reshape(1,1,128,512).to(device)
        
        key = torch.rand((b, skv, 1, d), device=device, dtype=torch.float16)
        lse = torch.rand((b, 1, sq, g), device=device, dtype=torch.float32)
        
        if k > 0:
            sparse_indices = generate_sparse_indices_loop(b, sq, skv, k, maskmode, cmp_ratio).to(device)
        else:
            sparse_indices = None
        if maskmode == 3 and cmp_ratio > 1:
            cmp_residual_k = torch.randint(0, cmp_ratio, (b,), dtype=torch.int32).to(device)
        else:
            cmp_residual_k = None
            
        print(f"sparse_indices shape: {sparse_indices.shape}\n")
        print(f"cmp_residual_k: {cmp_residual_k}")
        if has_seqused_q:
            seqused_q = torch.randint(0, sq, (b,), dtype=torch.int32).to(device)
            print(f"seqused_q: {seqused_q}")
        else:
            seqused_q = None
        
        if has_seqused_k:
            seqused_k = torch.randint(0, skv, (b,), dtype=torch.int32).to(device)
            print(f"seqused_k: {seqused_k}")
        else:
            seqused_k = None
        
        softmax_l1_norm = torch.zeros((b, sq, 1, k), device=device, dtype=torch.float32)
        tiling_data = torch.zeros((1024,), device=device, dtype=torch.int32)
        tiling_data[0] = 1 if has_seqused_q else 0
        tiling_data[1] = 1 if has_seqused_k else 0
        tiling_data[2] = 1 if has_topk_length else 0
        tiling_data[3] = 1 if layout == "TND" else 0
        tiling_data[4] = 1 if sparse_indices is not None else 0
        
        gatherTmp = torch.zeros((32, 512), device=device, dtype=torch.float16)
        qk = torch.zeros((g, 128), device=device, dtype=torch.float32)
        softmax_scale = 1 / math.sqrt(d)
        layout = "BSND"
        # lse[:,:,:,1] = 1
        # lse[:,:,:,65] = 1
        softmax_l1_norm_golden, metadata = compute_softmax_l1_norm_loop(query.cpu(), key.cpu(), lse.cpu(),
                                                              seqused_q.cpu() if seqused_q is not None else None,
                                                              seqused_k.cpu() if seqused_k is not None else None,
                                                              cmp_residual_k.cpu() if cmp_residual_k is not None else None,
                                                              sparse_indices.cpu() if sparse_indices is not None else None,
                                                              maskmode, cmp_ratio, num_cores)
        print(f"metadata: {metadata}")
        metadata = metadata.to(device)
        
        """
            q: pl.Tensor[[B, Sq, G, D], pl.DT_FP16],
            k: pl.Tensor[[B, Sk, 1, D], pl.DT_FP16],
            softmax_lse: pl.Tensor[[B, 1, Sq, G], pl.DT_FP16],
            seqused_q: pl.Tensor[[B,], pl.DT_INT32],
            seqused_k: pl.Tensor[[B,], pl.DT_INT32],
            cmp_residual_k: pl.Tensor[[B,], pl.DT_INT32],
            softmax_l1_norm: pl.Tensor[[B,Sq,Sk], pl.DT_FP16],
        """
        # print(f"lse: {lse}")
        for i in range(1):
            smla_softmax_l1_sparse_bsnd_kernel[None, num_cores](query, key, lse, sparse_indices, seqused_q, seqused_k, cmp_residual_k, metadata, maskmode, cmp_ratio, softmax_l1_norm, tiling_data, gatherTmp, qk)
        torch.npu.synchronize()
        
        # qk_golden = torch.matmul(q.cpu().to(torch.float32).reshape(g, d)[:,:], k.cpu().to(torch.float32).reshape(skv, d)[:,:].T)
        # print(f"qk: {qk}")
        npu_out = softmax_l1_norm
        print(f"npu_out shape: {npu_out.shape}")
        # torch.set_printoptions(linewidth=2000, threshold=float('inf'))
        # print(f"npu_out: {npu_out}")
        # print(f"key: {key}")
        
        # golden_out = key[:,:,:,:128].reshape(32,128)
        golden_out = softmax_l1_norm_golden
        print(f"golden_out shape: {golden_out.shape}")
        # torch.set_printoptions(profile='default')
        
        diff = (npu_out.cpu().flatten().to(torch.float32) - golden_out.cpu().flatten().to(torch.float32)).abs().max().item()
        # print(f"diff: {diff}, \n npu_out: {npu_out}, \ngolden_out: {golden_out}")
        print(f"diff: {diff}")
        # print(f"qk0: {qk.cpu()[0,:5]}")
        # print(f"qk1: {qk.cpu()[1,:5]}")
        # print(f"lse0=: {qk.cpu()[0,8:]}")
        # print(f"lse1=: {qk.cpu()[1,8:]}")
        # print(f"lse2=: {qk.cpu()[2,8:]}")
        # print(f"lse3=: {qk.cpu()[3,8:]}")
        # print(f"npu_out[2003968]: {npu_out.cpu().flatten().to(torch.float32)[2003968]}")
        # print(f"npu_out[2003969]: {npu_out.cpu().flatten().to(torch.float32)[2003969]}")
        # print(f"npu_out[2006016]: {npu_out.cpu().flatten().to(torch.float32)[2006016]}")
        # 假设 npu_out 和 golden_out 已经准备好
        is_match = find_mismatched_elements(npu_out, golden_out, rtol=5e-3, atol=5e-3)
        if is_match:
            print("✅ 精度校验通过，所有点均在容差范围内。")
        else:
            print("❌ 精度校验失败，存在异常点。")
        torch.testing.assert_close(npu_out.cpu().flatten().to(torch.float32), golden_out.cpu().flatten().to(torch.float32), rtol=5e-3, atol=5e-3)
        logging.info("  PASS")


def generate_seqused(seq_list, b):
    rand_unif = torch.rand(seq_list.shape)
    return (rand_unif * (seq_list.to(torch.float64) + 1)).to(torch.int32)

def compress_bnsd_to_tnd(x: torch.Tensor, seqlist: torch.Tensor) -> torch.Tensor:
    """
    将 [B, S, N, D] 转换为 [T, N, D]
    x: [B, S, N, D]
    seqlist: [B]，表示每个 batch 的有效长度
    """
    if x == None:
        return None
    B, S, N, D = x.shape
    
    # 1. 生成 [B, S] 的位置索引
    idxs = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
    
    # 2. 根据实际长度生成 boolean mask，有效位置为 True
    mask = idxs < seqlist.unsqueeze(1)  # shape: [B, S]
    
    # 3. 使用 mask 提取有效数据
    # PyTorch 会自动将 mask 中为 True 的位置提取出来，前两维合并为 T，后两维保留
    out = x[mask]  # shape: [T, N, D]
    
    return out

def compress_bnsd_to_ntd(x: torch.Tensor, seqlist: torch.Tensor) -> torch.Tensor:
    """
    将 [B, N, S, D] 转换为 [N, T, D]
    x: [B, N, S, D]
    seqlist: [B]，表示每个 batch 的有效长度
    """
    B, N, S, D = x.shape
    
    # 1. 将 [B, N, S, D] 转置为 [B, S, N, D]，使 S 维紧挨在 B 维之后
    x_bsnd = x.transpose(1, 2)
    
    # 2. 生成 [B, S] 的位置索引和 boolean mask
    idxs = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
    mask = idxs < seqlist.unsqueeze(1)  # shape: [B, S]
    
    # 3. 使用 mask 提取有效数据，前两维合并为 T，结果 shape 为 [T, N, D]
    tnd = x_bsnd[mask]
    
    # 4. 将 [T, N, D] 转置为 [N, T, D]
    ntd = tnd.transpose(0, 1)
    
    # 5. （可选）如果后续操作需要连续内存，取消下面这行的注释
    ntd = ntd.contiguous()
    
    return ntd

def build_cumsum_offsets(a: torch.Tensor) -> torch.Tensor:
    """
    a: shape [b], dtype int32
    return: shape [b+1], dtype 与 a 相同
    """
    # 1. 计算前缀和
    cumsum_a = torch.cumsum(a, dim=0)
    
    # 2. 在最前面补 0
    c = torch.cat([
        torch.zeros(1, dtype=a.dtype, device=a.device), 
        cumsum_a
    ])
    
    return c.to(a.dtype)

def test_smla_softmax_l1_tnd_sparse_a5():
    device = "npu:0"
    torch.npu.set_device(device)
    torch.manual_seed(31)
    for b, sq, g, skv, d, num_cores, maskmode, cmp_ratio, k, has_seqused_q, has_seqused_k, has_topk_length, layout in [
        (2, 2048, 128, 1024, D_TOTAL, 32, 0, 1, 512, True, True, False, "TND"),
        (1, 4096, 64, 512, D_TOTAL, 32, 3, 10, 1024, False, False, False, "TND"),
    ]:
        logging.info("\nFA-TND-DN-A5 (b=%s, sq=%s, g=%s, skv=%s, d=%s) cores=%s", b, sq, g, skv, d, num_cores)
        seq_q = torch.randint(1, sq + 1, (b,), dtype=torch.int32)
        seq_k = torch.randint(1, skv + 1, (b,), dtype=torch.int32)
        t1 = sum(seq_q)
        t2 = sum(seq_k)
        seq_q_tensor = build_cumsum_offsets(seq_q).to(device)
        seq_k_tensor = build_cumsum_offsets(seq_k).to(device)
        
        query = torch.rand((b, sq, g, d), device=device, dtype=torch.float16)
        
        # # 假设 a 是已有的矩阵，这里用随机数模拟
        # query = torch.randn((128, 512), dtype=torch.float16)
        # # 1. 将整个矩阵清零
        # query.zero_()
        # # 2. 将左上角 [128, 128] 的部分设为单位矩阵
        # query[:, :128] = torch.eye(128, dtype=query.dtype, device=query.device)
        # query[:, 128:256] = torch.eye(128, dtype=query.dtype, device=query.device)
        # query[:, 256:384] = torch.eye(128, dtype=query.dtype, device=query.device)
        # query[:, 384:] = torch.eye(128, dtype=query.dtype, device=query.device)
        # query = query.reshape(1,1,128,512).to(device)
        
        key = torch.rand((b, skv, 1, d), device=device, dtype=torch.float16)
        lse = torch.rand((b, 1, sq, g), device=device, dtype=torch.float32)
        if k > 0:
            sparse_indices = generate_sparse_indices_loop(b, sq, skv, k, maskmode, cmp_ratio, layout, seq_q, seq_k).to(device)
        else:
            sparse_indices = None
        if maskmode == 3 and cmp_ratio > 1:
            cmp_residual_k = torch.randint(0, cmp_ratio, (b,), dtype=torch.int32).to(device)
        else:
            cmp_residual_k = None
            
        print(f"sparse_indices shape: {sparse_indices.shape}\n")
        print(f"cmp_residual_k: {cmp_residual_k}")
        if has_seqused_q:
            seqused_q = generate_seqused(seq_q, b).to(device)
            # seqused_q[0] = 2
            # seqused_q[1] = 3
            print(f"seqused_q: {seqused_q}")
        else:
            seqused_q = None
        
        if has_seqused_k:
            seqused_k = generate_seqused(seq_k, b).to(device).to(device)
            print(f"seqused_k: {seqused_k}")
        else:
            seqused_k = None
        
        softmax_l1_norm = torch.zeros((t1, 1, k), device=device, dtype=torch.float32)
        tiling_data = torch.zeros((1024,), device=device, dtype=torch.int32)
        tiling_data[0] = 1 if has_seqused_q else 0
        tiling_data[1] = 1 if has_seqused_k else 0
        tiling_data[2] = 1 if has_topk_length else 0
        tiling_data[3] = 1 if layout == "TND" else 0
        tiling_data[4] = 1 if sparse_indices is not None else 0
        
        
        gatherTmp = torch.zeros((52, 512), device=device, dtype=torch.float16)
        qk = torch.zeros((g, 128), device=device, dtype=torch.float32)
        
        # if b == 2:
        #     continue
        
        softmax_l1_norm_golden, metadata = compute_softmax_l1_norm_loop(query.cpu(), key.cpu(), lse.cpu(),
                                                              seqused_q.cpu() if seqused_q is not None else None,
                                                              seqused_k.cpu() if seqused_k is not None else None,
                                                              cmp_residual_k.cpu() if cmp_residual_k is not None else None,
                                                              sparse_indices.cpu() if sparse_indices is not None else None,
                                                              maskmode, cmp_ratio, num_cores, layout, seq_q, seq_k)
        print(f"metadata: {metadata}")
        metadata = metadata.to(device)
        
        query_tnd = compress_bnsd_to_tnd(query.cpu(), seq_q).to(device)
        key_tnd = compress_bnsd_to_tnd(key.cpu(), seq_k).to(device)
        sparse_indices_tnd = compress_bnsd_to_tnd(sparse_indices.cpu(), seq_q).to(device)
        lse_tnd = compress_bnsd_to_ntd(lse.cpu(), seq_q).to(device)
        softmax_l1_norm_golden_tnd = compress_bnsd_to_tnd(softmax_l1_norm_golden, seq_q)
        print(f"softmax_l1_norm_golden shape {softmax_l1_norm_golden.shape}, softmax_l1_norm_golden_tnd shape {softmax_l1_norm_golden_tnd.shape}")
        
        print(f"seq_q_tensor: {seq_q_tensor}")
        print(f"seq_k_tensor: {seq_k_tensor}")
        for i in range(1):
            smla_softmax_l1_sparse_bsnd_kernel_tnd[None, num_cores](query_tnd, key_tnd, lse_tnd, sparse_indices_tnd, seq_q_tensor, seq_k_tensor, seqused_q, seqused_k, cmp_residual_k, metadata, maskmode, cmp_ratio, softmax_l1_norm, tiling_data, gatherTmp, qk)
        torch.npu.synchronize()
        
        # qk_golden = torch.matmul(q.cpu().to(torch.float32).reshape(g, d)[:,:], k.cpu().to(torch.float32).reshape(skv, d)[:,:].T)
        # print(f"qk: {qk}")
        npu_out = softmax_l1_norm
        print(f"npu_out shape: {npu_out.shape}")
        # torch.set_printoptions(linewidth=2000, threshold=float('inf'))
        # print(f"npu_out: {npu_out}")
        # print(f"key: {key}")
        
        # golden_out = key[:,:,:,:128].reshape(32,128)
        golden_out = softmax_l1_norm_golden_tnd
        print(f"golden_out shape: {golden_out.shape}")
        # print(f"golden_out: {golden_out}")
        # torch.set_printoptions(profile='default')
        
        diff = (npu_out.cpu().flatten().to(torch.float32) - golden_out.cpu().flatten().to(torch.float32)).abs().max().item()
        # print(f"diff: {diff}, \n npu_out: {npu_out}, \ngolden_out: {golden_out}")
        print(f"diff: {diff}")
        # 假设 npu_out 和 golden_out 已经准备好
        is_match = find_mismatched_elements(npu_out, golden_out, rtol=5e-3, atol=5e-3)
        if is_match:
            print("✅ 精度校验通过，所有点均在容差范围内。")
        else:
            print("❌ 精度校验失败，存在异常点。")
        torch.testing.assert_close(npu_out.cpu().flatten().to(torch.float32), golden_out.cpu().flatten().to(torch.float32), rtol=5e-3, atol=5e-3)
        logging.info("  PASS")

def test_smla_softmax_l1_bsnd_dense_a5():
    device = "npu:0"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    for b, sq, g, skv, d, num_cores, maskmode, cmp_ratio, k, has_seqused_q, has_seqused_k, has_topk_length, layout in [
        (2, 123, 63, 512, D_TOTAL, 32, 0, 1, 0, False, False, False, "BSND"),
        (2, 1024, 128, 1024, D_TOTAL, 32, 3, 5, 0, True, True, False, "BSND"),
        # (2, 1024, 128, 1024, D_TOTAL, 10, 3, 5, 0, True, True, False, "BSND"),
        # (1, 2, 128, 128, D_TOTAL, 1, 3, 5, 0, False, False, False, "BSND"),
    ]:
        logging.info("\nFA-BSND-DN-A5 (b=%s, sq=%s, g=%s, skv=%s, d=%s) cores=%s", b, sq, g, skv, d, num_cores)
        query = torch.rand((b, sq, g, d)
                           , device=device, dtype=torch.float16)
        
        # # 假设 a 是已有的矩阵，这里用随机数模拟
        # query = torch.randn((128, 512), dtype=torch.float16)
        # # 1. 将整个矩阵清零
        # query.zero_()
        # # 2. 将左上角 [128, 128] 的部分设为单位矩阵
        # query[:, :128] = torch.eye(128, dtype=query.dtype, device=query.device)
        # query[:, 128:256] = torch.eye(128, dtype=query.dtype, device=query.device)
        # query[:, 256:384] = torch.eye(128, dtype=query.dtype, device=query.device)
        # query[:, 384:] = torch.eye(128, dtype=query.dtype, device=query.device)
        # query = query.reshape(1,1,128,512).to(device)
        
        key = torch.rand((b, skv, 1, d), device=device, dtype=torch.float16)
        lse = torch.rand((b, 1, sq, g), device=device, dtype=torch.float32)
        if k > 0:
            sparse_indices = generate_sparse_indices_loop(b, sq, skv, k, maskmode, cmp_ratio).to(device)
        else:
            sparse_indices = None
        if maskmode == 3 and cmp_ratio > 1:
            cmp_residual_k = torch.randint(0, cmp_ratio, (b,), dtype=torch.int32).to(device)
        else:
            cmp_residual_k = None
            
        # print(f"sparse_indices shape: {sparse_indices.shape}\n")
        print(f"cmp_residual_k: {cmp_residual_k}")
        if has_seqused_q:
            seqused_q = torch.randint(0, sq, (b,), dtype=torch.int32).to(device)
            print(f"seqused_q: {seqused_q}")
        else:
            seqused_q = None
        
        if has_seqused_k:
            seqused_k = torch.randint(0, skv, (b,), dtype=torch.int32).to(device)
            print(f"seqused_k: {seqused_k}")
        else:
            seqused_k = None
        
        if k > 0:
            softmax_l1_norm = torch.zeros((b, sq, 1, k), device=device, dtype=torch.float32)
        else:
            softmax_l1_norm = torch.zeros((b, sq, 1, skv), device=device, dtype=torch.float32)
        tiling_data = torch.zeros((1024,), device=device, dtype=torch.int32)
        tiling_data[0] = 1 if has_seqused_q else 0
        tiling_data[1] = 1 if has_seqused_k else 0
        tiling_data[2] = 1 if has_topk_length else 0
        tiling_data[3] = 1 if layout == "TND" else 0
        tiling_data[4] = 1 if sparse_indices is not None else 0
        
        gatherTmp = torch.zeros((32, 512), device=device, dtype=torch.float16)
        qk = torch.zeros((g, 128), device=device, dtype=torch.float32)
        softmax_scale = 1 / math.sqrt(d)
        layout = "BSND"
        
        softmax_l1_norm_golden, metadata = compute_softmax_l1_norm_loop(query.cpu(), key.cpu(), lse.cpu(),
                                                              seqused_q.cpu() if seqused_q is not None else None,
                                                              seqused_k.cpu() if seqused_k is not None else None,
                                                              cmp_residual_k.cpu() if cmp_residual_k is not None else None,
                                                              sparse_indices.cpu() if sparse_indices is not None else None,
                                                              maskmode, cmp_ratio, num_cores)
        print(f"metadata: {metadata}")
        metadata = metadata.to(device)
        
        """
            q: pl.Tensor[[B, Sq, G, D], pl.DT_FP16],
            k: pl.Tensor[[B, Sk, 1, D], pl.DT_FP16],
            softmax_lse: pl.Tensor[[B, 1, Sq, G], pl.DT_FP16],
            seqused_q: pl.Tensor[[B,], pl.DT_INT32],
            seqused_k: pl.Tensor[[B,], pl.DT_INT32],
            cmp_residual_k: pl.Tensor[[B,], pl.DT_INT32],
            softmax_l1_norm: pl.Tensor[[B,Sq,Sk], pl.DT_FP16],
        """
        # print(f"lse: {lse}")
        for i in range(1):
            smla_softmax_l1_dense_bsnd_kernel[None, num_cores](query, key, lse, sparse_indices, seqused_q, seqused_k, cmp_residual_k, metadata, maskmode, cmp_ratio, softmax_l1_norm, tiling_data, gatherTmp, qk)
        torch.npu.synchronize()
        
        # qk_golden = torch.matmul(q.cpu().to(torch.float32).reshape(g, d)[:,:], k.cpu().to(torch.float32).reshape(skv, d)[:,:].T)
        # print(f"qk: {qk}")
        npu_out = softmax_l1_norm
        print(f"npu_out shape: {npu_out.shape}")
        # torch.set_printoptions(linewidth=2000, threshold=float('inf'))
        # print(f"npu_out: {npu_out}")
        # print(f"key: {key}")
        
        # golden_out = key[:,:,:,:128].reshape(32,128)
        golden_out = softmax_l1_norm_golden
        print(f"golden_out shape: {golden_out.shape}")
        # torch.set_printoptions(profile='default')
        
        diff = (npu_out.cpu().flatten().to(torch.float32) - golden_out.cpu().flatten().to(torch.float32)).abs().max().item()
        # print(f"diff: {diff}, \n npu_out: {npu_out}, \ngolden_out: {golden_out}")
        print(f"diff: {diff}")
        # print(f"npu_out[2003968]: {npu_out.cpu().flatten().to(torch.float32)[2003968]}")
        # print(f"npu_out[2003969]: {npu_out.cpu().flatten().to(torch.float32)[2003969]}")
        # print(f"npu_out[2006016]: {npu_out.cpu().flatten().to(torch.float32)[2006016]}")
        # 假设 npu_out 和 golden_out 已经准备好
        is_match = find_mismatched_elements(npu_out, golden_out, rtol=5e-3, atol=5e-3)
        if is_match:
            print("✅ 精度校验通过，所有点均在容差范围内。")
        else:
            print("❌ 精度校验失败，存在异常点。")
        torch.testing.assert_close(npu_out.cpu().flatten().to(torch.float32), golden_out.cpu().flatten().to(torch.float32), rtol=5e-3, atol=5e-3)
        logging.info("  PASS")

def test_smla_softmax_l1_tnd_dense_a5():
    device = "npu:0"
    torch.npu.set_device(device)
    torch.manual_seed(31)
    for b, sq, g, skv, d, num_cores, maskmode, cmp_ratio, k, has_seqused_q, has_seqused_k, has_topk_length, layout in [
        (2, 2048, 128, 1024, D_TOTAL, 32, 0, 1, 0, True, True, False, "TND"),
        (1, 4096, 64, 512, D_TOTAL, 32, 3, 1, 0, False, False, False, "TND"),
        (1, 1, 128, 128, D_TOTAL, 1, 0, 1, 0, False, False, False, "TND"),
    ]:
        logging.info("\nFA-TND-DN-A5 (b=%s, sq=%s, g=%s, skv=%s, d=%s) cores=%s", b, sq, g, skv, d, num_cores)
        seq_q = torch.randint(1, sq + 1, (b,), dtype=torch.int32)
        seq_k = torch.randint(1, skv + 1, (b,), dtype=torch.int32)
        # seq_q[0] = 1277
        # seq_k[0] = 222
        t1 = sum(seq_q)
        t2 = sum(seq_k)
        seq_q_tensor = build_cumsum_offsets(seq_q).to(device)
        seq_k_tensor = build_cumsum_offsets(seq_k).to(device)
        
        query = torch.rand((b, sq, g, d), device=device, dtype=torch.float16)
        
        # # 假设 a 是已有的矩阵，这里用随机数模拟
        # query = torch.randn((128, 512), dtype=torch.float16)
        # # 1. 将整个矩阵清零
        # query.zero_()
        # # 2. 将左上角 [128, 128] 的部分设为单位矩阵
        # query[:, :128] = torch.eye(128, dtype=query.dtype, device=query.device)
        # query[:, 128:256] = torch.eye(128, dtype=query.dtype, device=query.device)
        # query[:, 256:384] = torch.eye(128, dtype=query.dtype, device=query.device)
        # query[:, 384:] = torch.eye(128, dtype=query.dtype, device=query.device)
        # query = query.reshape(1,1,128,512).to(device)
        
        key = torch.rand((b, skv, 1, d), device=device, dtype=torch.float16)
        lse = torch.rand((b, 1, sq, g), device=device, dtype=torch.float32)
        if k > 0:
            sparse_indices = generate_sparse_indices_loop(b, sq, skv, k, maskmode, cmp_ratio, layout, seq_q, seq_k).to(device)
        else:
            sparse_indices = None
        if maskmode == 3 and cmp_ratio > 1:
            cmp_residual_k = torch.randint(0, cmp_ratio, (b,), dtype=torch.int32).to(device)
        else:
            cmp_residual_k = None
            
        print(f"cmp_residual_k: {cmp_residual_k}")
        if has_seqused_q:
            seqused_q = generate_seqused(seq_q, b).to(device)
            # seqused_q[0] = 2
            # seqused_q[1] = 3
            print(f"seqused_q: {seqused_q}")
        else:
            seqused_q = None
        
        if has_seqused_k:
            seqused_k = generate_seqused(seq_k, b).to(device).to(device)
            print(f"seqused_k: {seqused_k}")
        else:
            seqused_k = None
        
        if k > 0:
            softmax_l1_norm = torch.zeros((t1, 1, k), device=device, dtype=torch.float32)
        else:
            softmax_l1_norm = torch.zeros((t1, 1, skv), device=device, dtype=torch.float32)
        tiling_data = torch.zeros((1024,), device=device, dtype=torch.int32)
        tiling_data[0] = 1 if has_seqused_q else 0
        tiling_data[1] = 1 if has_seqused_k else 0
        tiling_data[2] = 1 if has_topk_length else 0
        tiling_data[3] = 1 if layout == "TND" else 0
        tiling_data[4] = 1 if sparse_indices is not None else 0
        
        gatherTmp = torch.zeros((52, 512), device=device, dtype=torch.float16)
        qk = torch.zeros((g, 128), device=device, dtype=torch.float32)
        
        # if b == 2:
        #     continue
        
        softmax_l1_norm_golden, metadata = compute_softmax_l1_norm_loop(query.cpu(), key.cpu(), lse.cpu(),
                                                              seqused_q.cpu() if seqused_q is not None else None,
                                                              seqused_k.cpu() if seqused_k is not None else None,
                                                              cmp_residual_k.cpu() if cmp_residual_k is not None else None,
                                                              sparse_indices.cpu() if sparse_indices is not None else None,
                                                              maskmode, cmp_ratio, num_cores, layout, seq_q, seq_k)
        print(f"metadata: {metadata}")
        metadata = metadata.to(device)
        
        query_tnd = compress_bnsd_to_tnd(query.cpu(), seq_q).to(device)
        key_tnd = compress_bnsd_to_tnd(key.cpu(), seq_k).to(device)
        if sparse_indices is not None:
            sparse_indices_tnd = compress_bnsd_to_tnd(sparse_indices.cpu(), seq_q).to(device)
        else:
            sparse_indices_tnd = None
        lse_tnd = compress_bnsd_to_ntd(lse.cpu(), seq_q).to(device)
        softmax_l1_norm_golden_tnd = compress_bnsd_to_tnd(softmax_l1_norm_golden, seq_q)
        print(f"softmax_l1_norm_golden shape {softmax_l1_norm_golden.shape}, softmax_l1_norm_golden_tnd shape {softmax_l1_norm_golden_tnd.shape}")
        
        print(f"seq_q_tensor: {seq_q_tensor}")
        print(f"seq_k_tensor: {seq_k_tensor}")
        for i in range(1):
            smla_softmax_l1_dense_bsnd_kernel_tnd[None, num_cores](query_tnd, key_tnd, lse_tnd, sparse_indices_tnd, seq_q_tensor, seq_k_tensor, seqused_q, seqused_k, cmp_residual_k, metadata, maskmode, cmp_ratio, softmax_l1_norm, tiling_data, gatherTmp, qk)
        torch.npu.synchronize()
        
        # qk_golden = torch.matmul(q.cpu().to(torch.float32).reshape(g, d)[:,:], k.cpu().to(torch.float32).reshape(skv, d)[:,:].T)
        # print(f"qk: {qk}")
        npu_out = softmax_l1_norm
        print(f"npu_out shape: {npu_out.shape}")
        # torch.set_printoptions(linewidth=2000, threshold=float('inf'))
        # print(f"npu_out: {npu_out}")
        # print(f"key: {key}")
        
        # golden_out = key[:,:,:,:128].reshape(32,128)
        golden_out = softmax_l1_norm_golden_tnd
        print(f"golden_out shape: {golden_out.shape}")
        # print(f"golden_out: {golden_out}")
        # torch.set_printoptions(profile='default')
        
        diff = (npu_out.cpu().flatten().to(torch.float32) - golden_out.cpu().flatten().to(torch.float32)).abs().max().item()
        # print(f"diff: {diff}, \n npu_out: {npu_out}, \ngolden_out: {golden_out}")
        print(f"diff: {diff}")
        # 假设 npu_out 和 golden_out 已经准备好
        is_match = find_mismatched_elements(npu_out, golden_out, rtol=5e-3, atol=5e-3)
        if is_match:
            print("✅ 精度校验通过，所有点均在容差范围内。")
        else:
            print("❌ 精度校验失败，存在异常点。")
        torch.testing.assert_close(npu_out.cpu().flatten().to(torch.float32), golden_out.cpu().flatten().to(torch.float32), rtol=5e-3, atol=5e-3)
        logging.info("  PASS")

if __name__ == "__main__":
    logging.info("SparseMlaSoftmaxL1Norm sparse on A5 CCE")
    logging.info("%s", '=' * 60)
    test_smla_softmax_l1_bs_sparse_a5()
    test_smla_softmax_l1_tnd_sparse_a5()
    test_smla_softmax_l1_bsnd_dense_a5()
    test_smla_softmax_l1_tnd_dense_a5()
    logging.info("\nAll SparseMlaSoftmaxL1Norm sparse tests passed!")
