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
from pypto_pro.runtime.tilingkey import tilingkey_field


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

Q_F16 = TG * D_TOTAL * 2
KT_F16 = TKV * TD * 2
QK_F32 = TKV * TS * 4

MA0 = 0
MA1 = Q_F16 * 2

LA0 = 0
LA1 = KT_F16
RA0 = 0
RA1 = Q_F16
CA0 = 0
CA1 = QK_F32

CUBE_BUFFER_NUM = 2
VEC_BUFFER_NUM = 2
GATHER_ROW_NUM = 32
VB4_KV = TG_HALF * TKV * 4 * VEC_BUFFER_NUM
VB4_GATHER = GATHER_ROW_NUM * TD * 2 * VEC_BUFFER_NUM
VB4_LSE = TG * 4 * VEC_BUFFER_NUM
VB4_OUT = TKV * 4 * VEC_BUFFER_NUM

QK_READY_FORWARD_IDS = (0, 1)
QK_READY_BARKWARD_IDS = (2, 3)
QK_READY_BARKWARD_IDS2 = (4, 5)


def _align_up(value, align=1024):
    return ((value + align - 1) // align) * align


VA0 = 0
VA1 = _align_up(VA0 + VB4_KV)
VA2 = _align_up(VA1 + VB4_GATHER)
VA3 = _align_up(VA2 + VB4_LSE)
VA4 = _align_up(VA2 + VB4_OUT)

EVENT_IDS_01 = (0, 1)
EVENT_IDS_23 = (2, 3)
QK_READY_IDS = (0, 1)
P_READY_IDS = (2, 3)
PV_READY_IDS = (4, 5)

B = pl.Var("B")
N = pl.Var("N")
Sq = pl.Var('Sq')
Sk = pl.Var('Sk')
G = pl.Var('G')
D = pl.Var('D')


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


# ================================================================
#  TilingKey
# ================================================================
class SmlaTilingKey:
    def is_valid(self, key):
        return True


# ================================================================
#  Pipeline stages and VF helpers
# ================================================================

@pl.pipeline.stage
def compute_qk(G, ki, qi, tick, q, k, cur_q_slot, q_l1_db, k_l1_db, left_db, right_db, acc_db, qk_vec_db, Sq_dim, s2_real_size, qkGm):
    batch_id = qi // Sq_dim
    qk_acc = acc_db.next()
    G_align2 = (G + 1) // 2 * 2
    pl.set_validshape(qk_acc, [G_align2, TKV])

    if ki == 0:
        pl.load(cur_q_slot, q, [batch_id, qi % Sq_dim, 0, 0], tile_dims=[2, 3])

    for d_offset in pl.range(0, D_TOTAL, TD):
        cur_k_slot = k_l1_db.next()
        pl.set_validshape(cur_k_slot, [TD, s2_real_size])

        pl.load(cur_k_slot, k, [batch_id, ki, 0, d_offset], is_transpose=True, tile_dims=[1, 3])
        qk_left = left_db.next()
        pl.set_validshape(qk_left, [G_align2, TD])
        qk_right = right_db.next()

        pl.move(qk_left, cur_q_slot, [0, d_offset])
        pl.move(qk_right, cur_k_slot)
        if d_offset == 0:
            pl.matmul(qk_acc, qk_left, qk_right)
        else:
            pl.matmul_acc(qk_acc, qk_acc, qk_left, qk_right)
    qk_vec_slot = qk_vec_db.next()

    pl.move(qk_vec_slot, qk_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


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

    vreg_reduce_sum_muls = vf.duplicate(0.0, preg_full)
    vreg_reduce_sum_muls_tail = vf.duplicate(0.0, preg_tail)

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
def compute_softmax_l1_norm(softmax_l1_norm, qi, ki, currentHalfG, s2_real_size, softmax_l1_norm_scale, qk_vec_db, lse_vec_slot, out_vec_db, sub_id, Sq_dim, qk):
    qk_vec_slot = qk_vec_db.next()
    out_vec_slot = out_vec_db.next()

    if currentHalfG == 0:
        return

    batch_id = qi // Sq_dim
    s1_idx = qi % Sq_dim

    muls_sel_vf_pse_type1_multi_tile(
        qk_vec_slot,
        lse_vec_slot,
        out_vec_slot,
        SCALE,
        currentHalfG,
        s2_real_size,
        softmax_l1_norm_scale,
    )

    pl.set_validshape(out_vec_slot, [1, s2_real_size])
    pl.store(softmax_l1_norm, out_vec_slot, [batch_id, s1_idx, ki], atomic=pl.AtomicType.AtomicAdd)


# ================================================================
#  Inner kernel — receives typed tensor views + workspace
# ================================================================

def sparse_flash_mla_softmax_l1_norm_inner(
    q, k, softmax_lse, softmax_l1_norm, qk, tiling
):
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()
    sub_id = pl.get_subblock_idx()

    Sq_dim = tiling.sq
    skv_dim = tiling.sk
    g_dim = tiling.g
    sq_tiles = tiling.b * Sq_dim
    firstHalfG = (g_dim + 2 - 1) // 2
    secondHalfG = g_dim - firstHalfG
    currentHalfG = firstHalfG
    if sub_id == 1:
        currentHalfG = secondHalfG

    softmax_l1_norm_scale = 1.0 / g_dim

    qk_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TG_HALF, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=VA0, mutex_ids=[0,1],
        fwd_ids=QK_READY_FORWARD_IDS, bwd_ids=QK_READY_BARKWARD_IDS
    )

    with pl.section_cube():
        q_mat_db = pl.make_tile_group(type=pl.TileType(
                shape=[TG, D_TOTAL], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ, valid_shape=[-1, -1], compact=1), addrs=MA0, mutex_ids=[2,3])
        k_mat_db = pl.make_tile_group(type=pl.TileType(
                shape=[TD, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN), addrs=MA1, mutex_ids=[4,5,12,13])
        left_db = pl.make_tile_group(
            type=pl.TileType(shape=[TG, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1), addrs=LA0, mutex_ids=[6,7])
        right_db = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right), addrs=RA0, mutex_ids=[8,9]
        )
        acc_db = pl.make_tile_group(
            type=pl.TileType(shape=[TG, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1], compact=1), addrs=CA0, mutex_ids=[10,11]
        )

    with pl.section_vector():
        gather_vec_db = pl.make_tile_group(
            type=pl.TileType(shape=[GATHER_ROW_NUM, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec), addrs=VA1, mutex_ids=[2,3]
        )
        lse_vec_db = pl.make_tile_group(
            type=pl.TileType(shape=[1, TG], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=VA2, mutex_ids=[4,5]
        )
        out_vec_db = pl.make_tile_group(
            type=pl.TileType(shape=[1, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=VA3, mutex_ids=[6,7]
        )
    tick = 0
    q_count = 0
    for qi in pl.range(core_id, sq_tiles, num_cores):
        batch_id = qi // Sq_dim
        with pl.section_cube():
            cur_q_slot = q_mat_db.next()
            pl.set_validshape(cur_q_slot, [g_dim, D_TOTAL])
        with pl.section_vector():
            lse_vec_slot = lse_vec_db.next()
            if currentHalfG > 0:
                pl.set_validshape(lse_vec_slot, [1, currentHalfG])
                pl.load(lse_vec_slot, softmax_lse, [batch_id, 0, qi % Sq_dim, sub_id * firstHalfG], tile_dims=[1, 3])
        s2_real_size = TKV
        for ki in pl.range(0, skv_dim, TKV):
            if ki + TKV > skv_dim:
                s2_real_size = skv_dim - ki
            with pl.section_cube():
                compute_qk(g_dim, ki, qi, tick, q, k, cur_q_slot, q_mat_db, k_mat_db, left_db, right_db, acc_db, qk_vec_db, Sq_dim, s2_real_size, qk)
            with pl.section_vector():
                compute_softmax_l1_norm(softmax_l1_norm, qi, ki, currentHalfG, s2_real_size, softmax_l1_norm_scale, qk_vec_db, lse_vec_slot, out_vec_db, sub_id, Sq_dim, qk)
            tick = tick + 1
        q_count = q_count + 1
    return


# ================================================================
#  Kernel — dynamic rank: inputs are raw pointers, shapes come from tiling
# ================================================================
@pl.jit(auto_mutex=True, tiling_key=SmlaTilingKey, pipeline=pl.pipeline.PipelineConfig(sync_only=True))
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
        [tiling.b, 1, tiling.sq, tiling.g],
        [tiling.sq * tiling.g, tiling.sq * tiling.g, tiling.g, 1],
        dtype=pl.DT_FP32,
    )
    tensor_softmax_l1_norm = pl.make_tensor(
        softmax_l1_norm,
        [tiling.b, tiling.sq, tiling.sk],
        [tiling.sq * tiling.sk, tiling.sk, 1],
        dtype=pl.DT_FP32,
    )
    tensor_qk = pl.make_tensor(
        workspace,
        [tiling.g, tiling.sk],
        [tiling.sk, 1],
        dtype=pl.DT_FP32,
    )

    sparse_flash_mla_softmax_l1_norm_inner(
        tensor_q,
        tensor_k,
        tensor_lse,
        tensor_softmax_l1_norm,
        tensor_qk,
        tiling,
    )
