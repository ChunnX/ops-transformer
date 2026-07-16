# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""SparseFlashMlaSoftmaxL1Norm kernel (pypto-pro).
"""

import math
from dataclasses import dataclass

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField


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

VA0 = 0  # qk_vec  [TKV, TS_HALF] FP32
VA1 = _align_up(VA0 + VB4_KV)  # tmp_vec [TKV, TS_HALF] FP32
VA2 = _align_up(VA1 + VB4_GATHER)  # p_f16   [TKV, TS_HALF] FP16
VA3 = _align_up(VA2 + VB4_LSE)
VA4 = _align_up(VA3 + VB4_OUT)
VA5 = _align_up(VA4 + VB4_GATHER_NZ)

EVENT_IDS_01 = (0, 1)
EVENT_IDS_23 = (2, 3)
QK_READY_IDS = (0, 1)
P_READY_IDS = (2, 3)
PV_READY_IDS = (4, 5)
QK_MAX_EID = 2
P_MAX_EID = 4
PV_MAX_EID = 6

# ================================================================
#  Tiling data
# ================================================================
@dataclass
class OpTiling:
    b: int
    sq: int
    sk: int
    g: int
    d: int
    t1: int
    t2: int
    max_seqlen_k: int
    k_length: int
    cmp_ratio: int
    has_seqused_q: bool
    has_seqused_k: bool
    has_topk_length: bool


# ================================================================
#  TilingKey
# ================================================================
class SmlaTilingKey:
    SPARSE_MODE = TilingKeyField(bits=2, values=[0, 1, 2, 3])
    IS_TND = TilingKeyField(bits=1, values=[0])
    IS_SPARSE = TilingKeyField(bits=1, values=[0])


def get_q_offset(constInfo, runInfo):
    offset = []
    if IS_TND == 1:
        offset = [runInfo.t1_offset + runInfo.s1_idx, 0, 0]
    else:
        offset = [runInfo.batch_id, runInfo.s1_idx, 0, 0]
    return offset

# no sparse
def get_k_offset(constInfo, runInfo, d_offset):
    if IS_TND == 1:
        return None
    else:
        return [runInfo.batch_id, runInfo.ki, 0, d_offset]

def gather_k_offset(constInfo, runInfo, k_index):
    if IS_TND == 1:
        return [runInfo.t2_offset + k_index, 0, 0]
        # return None
    else:
        return [runInfo.batch_id, k_index, 0, 0]

def get_sparse_offset(constInfo, runInfo, first_core_half_s2_base_size):
    offset = 0
    if IS_TND == 1:
        offset = (runInfo.t1_offset + runInfo.s1_idx) * constInfo.k_length + constInfo.sub_id * first_core_half_s2_base_size
    else:
        offset = runInfo.batch_id * constInfo.s1_size * constInfo.k_length + runInfo.s1_idx * constInfo.k_length + constInfo.sub_id * first_core_half_s2_base_size
    return offset

def get_out_offset(constInfo, runInfo):
    if IS_TND == 1:
        return [runInfo.t1_offset + runInfo.s1_idx, 0, runInfo.ki]
        # return None
    else:
        return [runInfo.batch_id, runInfo.s1_idx, runInfo.ki]

def get_lse_offset(constInfo, runInfo):
    if IS_TND == 1:
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
def gatherK(k_l1_db, k, sparse_indices, gather_nd_db, gather_nz_db, constInfo, runInfo):
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
            if IS_TND == 1:
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
def compute_qk(q, k, cur_q_slot, k_l1_db, left_db, right_db, acc_db, qk_vec_db, constInfo, runInfo):
    """QK matmul stage (Cube). Writes qk_vec."""
    # batch_id = qi // Sq_dim
    batch_id = runInfo.batch_id
    qk_acc = acc_db.next()
    G_align2 = (constInfo.g_size + 1) // 2 * 2
    pl.set_validshape(qk_acc, [G_align2, TKV])
    
    # query复用
    if runInfo.ki == 0:
        if IS_TND == 1:
            pl.load(cur_q_slot, q, [runInfo.t1_offset + runInfo.s1_idx, 0, 0], tile_dims=[1, 2])
        else:
            pl.load(cur_q_slot, q, [runInfo.batch_id, runInfo.s1_idx, 0, 0], tile_dims=[2, 3])
    
    key_l1_slot = k_l1_db.next()
    pl.set_validshape(key_l1_slot, [D_TOTAL, runInfo.s2_base_size])

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
    
    qk_vec_slot = qk_vec_db.next()
    
    pl.move(qk_vec_slot, qk_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    # pl.system.set_cross_croe(pl.PipeType.FIX, event_id=QK_READY_FORWARD_IDS[taskId % 2], max_event_id=1, mode_id=0)
    # pl.store(qkGm, qk_acc, [0, 0])

@pl.pipeline.stage
def compute_qk_dense(q, k, cur_q_slot, k_l1_db, left_db, right_db, acc_db, qk_vec_db, constInfo, runInfo):
    """QK matmul stage (Cube). Writes qk_vec."""
    # batch_id = qi // Sq_dim
    batch_id = runInfo.batch_id
    qk_acc = acc_db.next()
    G_align2 = (constInfo.g_size + 1) // 2 * 2
    pl.set_validshape(qk_acc, [G_align2, TKV])
    
    # query复用
    if runInfo.ki == 0:
        if IS_TND == 1:
            pl.load(cur_q_slot, q, [runInfo.t1_offset + runInfo.s1_idx, 0, 0], tile_dims=[1, 2])
        else:
            pl.load(cur_q_slot, q, [runInfo.batch_id, runInfo.s1_idx, 0, 0], tile_dims=[2, 3])
    
    for d_offset in pl.range(0, D_TOTAL, TD):
        cur_k_slot = k_l1_db.next()
        pl.set_validshape(cur_k_slot, [TD, runInfo.s2_base_size])
        
        if IS_TND == 1:
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
def compute_softmax_l1_norm(softmax_l1_norm, qk_vec_db, lse_vec_slot, out_vec_db, constInfo, runInfo):
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
        
        if IS_TND == 1:
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
        if IS_TND == 1:
            runInfo.t1_offset = pl.getval(seq_q, runInfo.batch_id)
            runInfo.t2_offset = pl.getval(seq_k, runInfo.batch_id)
            runInfo.next_t1_offset = pl.getval(seq_q, runInfo.batch_id + 1)
            runInfo.s1_size = runInfo.next_t1_offset - runInfo.t1_offset
            s2_pre = pl.getval(seq_k, runInfo.batch_id)
            s2_next = pl.getval(seq_k, runInfo.batch_id + 1)
            runInfo.s2_size = s2_next - s2_pre
            # pl.printf("runInfo.batch_id: %d, s2_pre: %d, s2_next: %d, runInfo.s2_size: %d, runInfo.t1_offset: %d, runInfo.t2_offset: %d, runInfo.s1_idx: %d\n", runInfo.batch_id, s2_pre, s2_next, runInfo.s2_size, runInfo.t1_offset, runInfo.t2_offset, runInfo.s1_idx)
    else:
        if IS_TND == 1:
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


# ================================================================
#  Kernel — dynamic rank: inputs are raw pointers, shapes come from tiling
# ================================================================
@pl.jit(auto_mutex=True,
    tiling_key=SmlaTilingKey,
    datatype={
        "query": "input_dtype",
    },
    pipeline=pl.pipeline.PipelineConfig(sync_only=True))
def sparse_flash_mla_softmax_l1_norm(
    query: pl.Ptr[pl.DT_UINT8],
    key: pl.Ptr[pl.DT_UINT8],
    lse: pl.Ptr[pl.DT_UINT8],
    sparse_indices_optional: pl.Ptr[pl.DT_UINT8],
    cu_seqlens_q_optional: pl.Ptr[pl.DT_UINT8],
    cu_seqlens_k_optional: pl.Ptr[pl.DT_UINT8],
    seqused_q_optional: pl.Ptr[pl.DT_UINT8],
    seqused_k_optional: pl.Ptr[pl.DT_UINT8],
    cmp_residual_k_optional: pl.Ptr[pl.DT_UINT8],
    topk_length_optional: pl.Ptr[pl.DT_UINT8],
    metadata_optional: pl.Ptr[pl.DT_UINT8],
    softmax_l1_norm: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: OpTiling,
):
    pl.printf("SPARSE_MODE: %d, IS_TND: %d, IS_SPARSE: %d\n", SPARSE_MODE, IS_TND, IS_SPARSE)
    if IS_TND == 1:
        tensor_q = pl.make_tensor(
            query,
            [tiling.t1, tiling.g, tiling.d],
            [tiling.g * tiling.d, tiling.d, 1],
            dtype=pl.DT_FP16,
        )
        tensor_k = pl.make_tensor(
            key,
            [tiling.t2, 1, tiling.d],
            [tiling.d, tiling.d, 1],
            dtype=pl.DT_FP16,
        )
        tensor_lse = pl.make_tensor(
            lse,
            [1, tiling.t1, tiling.g],
            [tiling.g, tiling.g, 1],
            dtype=pl.DT_FP32,
        )
        if IS_SPARSE == 1:
            tensor_softmax_l1_norm = pl.make_tensor(
                softmax_l1_norm,
                [tiling.t1, 1, tiling.k_length],
                [tiling.k_length, tiling.k_length, 1],
                dtype=pl.DT_FP32,
            )
            tensor_sparse_indices = pl.make_tensor(
                sparse_indices_optional,
                [tiling.t1, 1, tiling.k_length],
                [tiling.k_length, tiling.k_length, 1],
                dtype=pl.DT_INT32,
            )
        else:
            tensor_softmax_l1_norm = pl.make_tensor(
                softmax_l1_norm,
                [tiling.t1, 1, tiling.max_seqlen_k],
                [tiling.max_seqlen_k, tiling.max_seqlen_k, 1],
                dtype=pl.DT_FP32,
            )
        tensor_cu_seqlens_q = pl.make_tensor(
            cu_seqlens_q_optional,
            [tiling.b + 1,],
            [1,],
            dtype=pl.DT_INT32,
        )
        tensor_cu_seqlens_k = pl.make_tensor(
            cu_seqlens_k_optional,
            [tiling.b + 1,],
            [1,],
            dtype=pl.DT_INT32,
        )
    else:
        tensor_q = pl.make_tensor(
            query,
            [tiling.b, tiling.sq, tiling.g, tiling.d],
            [tiling.sq * tiling.g * tiling.d, tiling.g * tiling.d, tiling.d, 1],
            dtype=pl.DT_FP16,
        )
        tensor_k = pl.make_tensor(
            key,
            [tiling.b, tiling.sk, 1, tiling.d],
            [tiling.sk * tiling.d, tiling.d, tiling.d, 1],
            dtype=pl.DT_FP16,
        )
        tensor_lse = pl.make_tensor(
            lse,
            [tiling.b, tiling.sq, 1, tiling.g],
            [tiling.sq * tiling.g, tiling.g, tiling.g, 1],
            dtype=pl.DT_FP32,
        )
        if IS_SPARSE == 1:
            tensor_softmax_l1_norm = pl.make_tensor(
                softmax_l1_norm,
                [tiling.b, tiling.sq, 1, tiling.k_length],
                [tiling.sq * tiling.k_length, tiling.k_length, tiling.k_length, 1],
                dtype=pl.DT_FP32,
            )
            tensor_sparse_indices = pl.make_tensor(
                sparse_indices_optional,
                [tiling.b, tiling.sq, 1, tiling.k_length],
                [tiling.sq * tiling.k_length, tiling.k_length, tiling.k_length, 1],
                dtype=pl.DT_INT32,
            )
        else:
            tensor_softmax_l1_norm = pl.make_tensor(
                softmax_l1_norm,
                [tiling.b, tiling.sq, 1, tiling.sk],
                [tiling.sq * tiling.sk, tiling.sk, tiling.sk, 1],
                dtype=pl.DT_FP32,
            )
        
    tensor_seqused_q = pl.make_tensor(
        seqused_q_optional,
        [tiling.b,],
        [1,],
        dtype=pl.DT_INT32,
    )
    tensor_seqused_k = pl.make_tensor(
        seqused_k_optional,
        [tiling.b,],
        [1,],
        dtype=pl.DT_INT32,
    )
    tensor_cmp_residual_k = pl.make_tensor(
        cmp_residual_k_optional,
        [tiling.b,],
        [1,],
        dtype=pl.DT_INT32,
    )
    tensor_topk_length = pl.make_tensor(
        topk_length_optional,
        [tiling.b,],
        [1,],
        dtype=pl.DT_INT32,
    )
    tensor_metadata = pl.make_tensor(
        metadata_optional,
        [1, 64],
        [64, 1],
        dtype=pl.DT_INT32,
    )

    total_num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()
    sub_id = pl.get_subblock_idx()
    
    has_seqused_q = tiling.has_seqused_q
    has_seqused_k = tiling.has_seqused_k
    has_topk_length = tiling.has_topk_length
    
    sq_tiles = pl.getval(tensor_metadata, 0)
    former_core_process_num = pl.getval(tensor_metadata, 1)
    remain_core_process_num = pl.getval(tensor_metadata, 2)
    remain_core_num = pl.getval(tensor_metadata, 3)
    num_cores = pl.getval(tensor_metadata, 4)
    if core_id >= num_cores:
        current_core_proccess_num = 0
    elif (core_id < total_num_cores - remain_core_num):
        current_core_proccess_num = former_core_process_num
    else:
        current_core_proccess_num = remain_core_process_num
    
    # sq_tiles = B * Sq
    firstHalfG = (tiling.g + 2 - 1) // 2
    secondHalfG = tiling.g - firstHalfG
    currentHalfG = firstHalfG
    if sub_id == 1:
        currentHalfG = secondHalfG
    
    constInfo = pl.make_tuple(b_size=tiling.b, g_size=tiling.g, s1_size=0, s2_size=0, is_sparse=IS_SPARSE, k_length=tiling.k_length, mask_mode=SPARSE_MODE, cmp_ratio=tiling.cmp_ratio, is_tnd=IS_TND, has_seqused_q=tiling.has_seqused_q, has_seqused_k=tiling.has_seqused_k, first_g_size=firstHalfG, current_g_size=currentHalfG, softmax_l1_norm_scale=1.0/tiling.g, sub_id=sub_id, core_id=core_id, num_cores=num_cores)
    runInfos = pl.struct_array(1, "RunInfo", batch_id=0, s1_idx=0, s1_size=0, s2_size=0, used_t1_index=0, used_t1_offset=0, next_used_t1_offset=0, t1_offset=0, next_t1_offset=0, t2_offset=0, ki=0, s2_real_size=0, s2_base_size=TKV, task_id=0)
    runInfo = runInfos[0]
    if has_seqused_q:
        runInfo.next_used_t1_offset = pl.getval(tensor_seqused_q, 0)
    if IS_TND:
        runInfo.next_t1_offset = pl.getval(tensor_cu_seqlens_q, 1)

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
        runInfo.used_t1_index = used_t1_index
        get_seq_length(tensor_seqused_q, tensor_cu_seqlens_q, tensor_cu_seqlens_k, constInfo, runInfo)
        
        runInfo.s2_base_size = TKV
        
        ### 计算s2_real_size
        current_s1_size = get_seqused(runInfo.s1_size, has_seqused_q, tensor_seqused_q, runInfo.batch_id)
        current_s2_size = get_seqused(runInfo.s2_size, has_seqused_k, tensor_seqused_k, runInfo.batch_id)
        # pl.printf("runInfo.s1_size: %d, runInfo.s2_size: %d, current_s1_size: %d, current_s2_size: %d\n", runInfo.s1_size, runInfo.s2_size, current_s1_size, current_s2_size)
        s1_idx = runInfo.s1_idx
        if IS_SPARSE == 0:
            if SPARSE_MODE == 0:
                s2_real_size = current_s2_size
            else:
                if constInfo.cmp_ratio > 1:
                    ori_s2_size = current_s2_size * constInfo.cmp_ratio + pl.getval(tensor_cmp_residual_k, runInfo.batch_id)
                else:
                    ori_s2_size = current_s2_size * constInfo.cmp_ratio
                s2_real_size = pl.max((ori_s2_size - current_s1_size + s1_idx + 1) // constInfo.cmp_ratio, 0)
                # pl.printf("ori_s2_size: %d, current_s2_size: %d, constInfo.cmp_ratio: %d, s2_real_size: %d\n", ori_s2_size, current_s2_size, constInfo.cmp_ratio, s2_real_size)
        else:
            if SPARSE_MODE == 0:
                s2_real_size = pl.min(constInfo.k_length, current_s2_size)
            else:
                if constInfo.cmp_ratio > 1:
                    ori_s2_size = current_s2_size * constInfo.cmp_ratio + pl.getval(tensor_cmp_residual_k, runInfo.batch_id)
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
            pl.set_validshape(cur_q_slot, [constInfo.g_size, D_TOTAL])
        with pl.section_vector():
            lse_vec_slot = lse_vec_db.next()
            if currentHalfG > 0 and runInfo.s2_real_size > 0:
                pl.set_validshape(lse_vec_slot, [1, currentHalfG])
                if IS_TND == 1:
                    pl.load(lse_vec_slot, tensor_lse, [0, runInfo.t1_offset + runInfo.s1_idx, constInfo.sub_id * constInfo.first_g_size], tile_dims=[1, 2])
                else:
                    pl.load(lse_vec_slot, tensor_lse, [runInfo.batch_id, 0, runInfo.s1_idx, constInfo.sub_id * constInfo.first_g_size], tile_dims=[1, 3])
            # pl.system.bar_all()
        
        for ki in pl.range(0, runInfo.s2_real_size, TKV):
            runInfo.ki = ki
            if ki + TKV > runInfo.s2_real_size:
                runInfo.s2_base_size =  runInfo.s2_real_size - ki
            with pl.section_cube():
                if IS_SPARSE == 1:
                    compute_qk(tensor_q, tensor_k, cur_q_slot, k_l1_db, left_db, right_db, acc_db, qk_vec_db, constInfo, runInfo)
                else:
                    compute_qk_dense(tensor_q, tensor_k, cur_q_slot, k_l1_db, left_db, right_db, acc_db, qk_vec_db, constInfo, runInfo)
            with pl.section_vector():
                compute_softmax_l1_norm(tensor_softmax_l1_norm, qk_vec_db, lse_vec_slot, out_vec_db, constInfo, runInfo)
            tick = tick + 1
        q_count = q_count + 1


# # ================================================================
# #  Kernel — dynamic rank: inputs are raw pointers, shapes come from tiling
# # ================================================================
# @pl.jit(auto_mutex=True,
#     tiling_key=SmlaTilingKey,
#     datatype={
#         "query": "input_dtype",
#     },
#     pipeline=pl.pipeline.PipelineConfig(sync_only=True))
# def sparse_flash_mla_softmax_l1_norm(
#     query: pl.Ptr[pl.DT_UINT8],
#     key: pl.Ptr[pl.DT_UINT8],
#     lse: pl.Ptr[pl.DT_UINT8],
#     sparse_indices_optional: pl.Ptr[pl.DT_UINT8],
#     cu_seqlens_q_optional: pl.Ptr[pl.DT_UINT8],
#     cu_seqlens_k_optional: pl.Ptr[pl.DT_UINT8],
#     seqused_q_optional: pl.Ptr[pl.DT_UINT8],
#     seqused_k_optional: pl.Ptr[pl.DT_UINT8],
#     cmp_residual_k_optional: pl.Ptr[pl.DT_UINT8],
#     topk_length_optional: pl.Ptr[pl.DT_UINT8],
#     metadata_optional: pl.Ptr[pl.DT_UINT8],
#     softmax_l1_norm: pl.Ptr[pl.DT_UINT8],
#     workspace: pl.Ptr[pl.DT_UINT8],
#     tiling: OpTiling,
# ):
#     pl.printf("SPARSE_MODE: %d, IS_TND: %d, IS_SPARSE: %d\n", SPARSE_MODE, IS_TND, IS_SPARSE)
#     if IS_TND:
#         tensor_q = pl.make_tensor(
#             query,
#             [tiling.t1, tiling.g, tiling.d],
#             [tiling.g * tiling.d, tiling.d, 1],
#             dtype=pl.DT_FP16,
#         )
#         tensor_k = pl.make_tensor(
#             key,
#             [tiling.t2, 1, tiling.d],
#             [tiling.d, tiling.d, 1],
#             dtype=pl.DT_FP16,
#         )
#         tensor_lse = pl.make_tensor(
#             lse,
#             [1, tiling.t1, tiling.g],
#             [tiling.g, tiling.g, 1],
#             dtype=pl.DT_FP32,
#         )
#         if IS_SPARSE:
#             tensor_softmax_l1_norm = pl.make_tensor(
#                 softmax_l1_norm,
#                 [tiling.t1, 1, tiling.k_length],
#                 [tiling.k_length, tiling.k_length, 1],
#                 dtype=pl.DT_FP32,
#             )
#             tensor_sparse_indices = pl.make_tensor(
#                 sparse_indices_optional,
#                 [tiling.t1, 1, tiling.k_length],
#                 [tiling.k_length, tiling.k_length, 1],
#                 dtype=pl.DT_INT32,
#             )
#         else:
#             tensor_softmax_l1_norm = pl.make_tensor(
#                 softmax_l1_norm,
#                 [tiling.t1, 1, tiling.max_seqlen_k],
#                 [tiling.max_seqlen_k, tiling.max_seqlen_k, 1],
#                 dtype=pl.DT_FP32,
#             )
#         tensor_cu_seqlens_q = pl.make_tensor(
#             cu_seqlens_q_optional,
#             [tiling.b + 1,],
#             [1,],
#             dtype=pl.DT_INT32,
#         )
#         tensor_cu_seqlens_k = pl.make_tensor(
#             cu_seqlens_k_optional,
#             [tiling.b + 1,],
#             [1,],
#             dtype=pl.DT_INT32,
#         )
#     else:
#         tensor_q = pl.make_tensor(
#             query,
#             [tiling.b, tiling.sq, tiling.g, tiling.d],
#             [tiling.sq * tiling.g * tiling.d, tiling.g * tiling.d, tiling.d, 1],
#             dtype=pl.DT_FP16,
#         )
#         tensor_k = pl.make_tensor(
#             key,
#             [tiling.b, tiling.sk, 1, tiling.d],
#             [tiling.sk * tiling.d, tiling.d, tiling.d, 1],
#             dtype=pl.DT_FP16,
#         )
#         tensor_lse = pl.make_tensor(
#             lse,
#             [tiling.b, tiling.sq, 1, tiling.g],
#             [tiling.sq * tiling.g, tiling.g, tiling.g, 1],
#             dtype=pl.DT_FP32,
#         )
#         if IS_SPARSE:
#             tensor_softmax_l1_norm = pl.make_tensor(
#                 softmax_l1_norm,
#                 [tiling.b, tiling.sq, 1, tiling.k_length],
#                 [tiling.sq * tiling.k_length, tiling.k_length, tiling.k_length, 1],
#                 dtype=pl.DT_FP32,
#             )
#             tensor_sparse_indices = pl.make_tensor(
#                 sparse_indices_optional,
#                 [tiling.b, tiling.sq, 1, tiling.k_length],
#                 [tiling.sq * tiling.k_length, tiling.k_length, tiling.k_length, 1],
#                 dtype=pl.DT_INT32,
#             )
#         else:
#             tensor_softmax_l1_norm = pl.make_tensor(
#                 softmax_l1_norm,
#                 [tiling.b, tiling.sq, 1, tiling.sk],
#                 [tiling.sq * tiling.sk, tiling.sk, tiling.sk, 1],
#                 dtype=pl.DT_FP32,
#             )
        
#     tensor_seqused_q = pl.make_tensor(
#         seqused_q_optional,
#         [tiling.b,],
#         [1,],
#         dtype=pl.DT_INT32,
#     )
#     tensor_seqused_k = pl.make_tensor(
#         seqused_k_optional,
#         [tiling.b,],
#         [1,],
#         dtype=pl.DT_INT32,
#     )
#     tensor_cmp_residual_k = pl.make_tensor(
#         cmp_residual_k_optional,
#         [tiling.b,],
#         [1,],
#         dtype=pl.DT_INT32,
#     )
#     tensor_topk_length = pl.make_tensor(
#         topk_length_optional,
#         [tiling.b,],
#         [1,],
#         dtype=pl.DT_INT32,
#     )
#     tensor_metadata = pl.make_tensor(
#         metadata_optional,
#         [1, 64],
#         [64, 1],
#         dtype=pl.DT_INT32,
#     )

#     if not IS_TND:
#         core_id = pl.get_block_idx()
#         sub_id = pl.get_subblock_idx()
#         Sq_dim = tiling.sq
#         skv_dim = tiling.sk
#         g_dim = tiling.g
#         firstHalfG = (g_dim + 2 - 1) // 2
#         secondHalfG = g_dim - firstHalfG
#         currentHalfG = firstHalfG
#         if sub_id == 1:
#             currentHalfG = secondHalfG

#         softmax_l1_norm_scale = 1.0 / g_dim

#         totalNum = pl.getval(tensor_metadata, 0)
#         formerCoreProcessNum = pl.getval(tensor_metadata, 1)
#         remainCoreProcessNum = pl.getval(tensor_metadata, 2)
#         remainCoreNum = pl.getval(tensor_metadata, 3)
#         totalCoreNum = pl.getval(tensor_metadata, 4)

#         former_core_num = totalCoreNum - remainCoreNum
#         if core_id < former_core_num:
#             qi_start = core_id * formerCoreProcessNum
#             qi_count = formerCoreProcessNum
#         elif core_id < totalCoreNum:
#             qi_start = former_core_num * formerCoreProcessNum + (core_id - former_core_num) * remainCoreProcessNum
#             qi_count = remainCoreProcessNum
#         else:
#             qi_start = 0
#             qi_count = 0

#         qk_vec_db = pl.make_tile_group(
#             type=pl.TileType(shape=[TG_HALF, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=VA0, mutex_ids=[0,1],
#             fwd_ids=QK_READY_FORWARD_IDS, bwd_ids=QK_READY_BARKWARD_IDS
#         )

#         with pl.section_cube():
#             q_mat_db = pl.make_tile_group(type=pl.TileType(
#                     shape=[TG, D_TOTAL], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ, valid_shape=[-1, -1], compact=1), addrs=MA0, mutex_ids=[2,3])
#             k_mat_db = pl.make_tile_group(type=pl.TileType(
#                     shape=[TD, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN), addrs=MA1, mutex_ids=[4,5,12,13])
#             left_db = pl.make_tile_group(
#                 type=pl.TileType(shape=[TG, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1), addrs=LA0, mutex_ids=[6,7])
#             right_db = pl.make_tile_group(
#                 type=pl.TileType(shape=[TD, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right), addrs=RA0, mutex_ids=[8,9]
#             )
#             acc_db = pl.make_tile_group(
#                 type=pl.TileType(shape=[TG, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1], compact=1), addrs=CA0, mutex_ids=[10,11]
#             )

#         with pl.section_vector():
#             gather_vec_db = pl.make_tile_group(
#                 type=pl.TileType(shape=[GATHER_ROW_NUM, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec), addrs=VA1, mutex_ids=[2,3]
#             )
#             lse_vec_db = pl.make_tile_group(
#                 type=pl.TileType(shape=[1, TG], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=VA2, mutex_ids=[4,5]
#             )
#             out_vec_db = pl.make_tile_group(
#                 type=pl.TileType(shape=[1, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=VA3, mutex_ids=[6,7]
#             )
#         tick = 0
#         q_count = 0
#         for qi in pl.range(qi_start, qi_start + qi_count):
#             batch_id = qi // Sq_dim
#             with pl.section_cube():
#                 cur_q_slot = q_mat_db.next()
#                 pl.set_validshape(cur_q_slot, [g_dim, D_TOTAL])
#             with pl.section_vector():
#                 lse_vec_slot = lse_vec_db.next()
#                 if currentHalfG > 0:
#                     pl.set_validshape(lse_vec_slot, [1, currentHalfG])
#                     pl.load(lse_vec_slot, tensor_lse, [batch_id, 0, qi % Sq_dim, sub_id * firstHalfG], tile_dims=[1, 3])
#             s2_real_size = TKV
#             for ki in pl.range(0, skv_dim, TKV):
#                 if ki + TKV > skv_dim:
#                     s2_real_size = skv_dim - ki
#                 with pl.section_cube():
#                     compute_qk(g_dim, ki, qi, tick, tensor_q, tensor_k, cur_q_slot, q_mat_db, k_mat_db, left_db, right_db, acc_db, qk_vec_db, Sq_dim, s2_real_size)
#                 with pl.section_vector():
#                     compute_softmax_l1_norm(tensor_softmax_l1_norm, qi, ki, currentHalfG, s2_real_size, softmax_l1_norm_scale, qk_vec_db, lse_vec_slot, out_vec_db, sub_id, Sq_dim)
#                 tick = tick + 1
#             q_count = q_count + 1
