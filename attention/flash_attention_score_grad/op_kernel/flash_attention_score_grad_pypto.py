# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FlashAttentionScoreGrad kernel (PyPTO Pro).

Minimal viable version:
  - FP16 input/output, FP32 accumulation
  - BSND layout: [B, S, N, D]
  - No atten_mask / pse / dropout / rope / sink / deter
  - D = 128, TILE_M = 128 (S1), TILE_N = 128 (S2)
  - Multi-core split over (B * N * S1_outer)
  - Cube+Vector fusion with manual cross-core sync (INTRA_BLOCK)

Algorithm (backward of O = softmax(Q @ K^T * scale) @ V):
  MM2: S   = Q @ K^T                    (Cube)  -> move to UB
  V2:  P   = softmax(S*scale, max, sum) (Vector)-> store P to GM
  MM1: dP  = dO @ V^T                   (Cube)  -> move to UB
  V3:  dS  = softmax_grad(dP, P)*scale  (Vector)-> store dS to GM
  MM3: dQ += dS @ K                     (Cube, atomic add)
  MM4: dK += dS^T @ Q                   (Cube, atomic add)
  MM5: dV += P^T @ dO                   (Cube, atomic add)
"""

from dataclasses import dataclass

import pypto_pro.language as pl


TILE_M = 128
TILE_N = 128
TILE_D = 128

EVT_C2V_S  = 0
EVT_V2C_P  = 1
EVT_C2V_DP = 2
EVT_V2C_DS = 3

L1_Q     = 0x00000
L1_K     = 0x08000
L1_K_N   = 0x10000
L1_V     = 0x18000
L1_DO    = 0x20000
L1_DS    = 0x28000
L1_DS_T  = 0x30000
L1_P_T   = 0x38000

L0A_ADDR = 0x0000
L0B_ADDR = 0x0000
L0C_ADDR = 0x0000

UB_QK    = 0x00000
UB_P     = 0x10000
UB_DS    = 0x18000
UB_TMP   = 0x20000
UB_MAX   = 0x30000
UB_SUM   = 0x30800
UB_DPSUM = 0x31000


@dataclass
class FagTiling:
    b: int
    n: int
    s1: int
    s2: int
    d: int
    scale_value: float


# ================================================================
#  SetConstInfo — populate constInfo from tiling (like AscendC)
# ================================================================
def set_const_info(tiling):
    constInfo = pl.struct("FagConstInfo",
        bSize=tiling.b,
        n2Size=tiling.n,
        s1Size=tiling.s1,
        s2Size=tiling.s2,
        dSize=tiling.d,
        scaleValue=tiling.scale_value,
        s1Outer=(tiling.s1 + TILE_M - 1) // TILE_M,
        s2Outer=(tiling.s2 + TILE_N - 1) // TILE_N,
        nd=tiling.n * TILE_D,
        bnTotal=tiling.b * tiling.n,
        s1s2=tiling.s1 * tiling.s2,
        totalWork=tiling.b * tiling.n * ((tiling.s1 + TILE_M - 1) // TILE_M),
    )
    return constInfo


# ================================================================
#  SetRunInfo — populate runInfo for a given work_id
# ================================================================
def set_run_info(constInfo, work_id):
    runInfo = pl.struct("FagRunInfo",
        bIdx=work_id // (constInfo.n2Size * constInfo.s1Outer),
        nIdx=(work_id // constInfo.s1Outer) % constInfo.n2Size,
        s1oIdx=work_id % constInfo.s1Outer,
        s1Off=(work_id % constInfo.s1Outer) * TILE_M,
        workId=work_id,
        smaxOff=(work_id // (constInfo.n2Size * constInfo.s1Outer)) * constInfo.n2Size
                + ((work_id // constInfo.s1Outer) % constInfo.n2Size),
    )
    return runInfo


# ================================================================
#  IterateMmQK — MM2: S = Q @ K^T  (cube stage)
# ================================================================
def iterate_mm_qk(constInfo, runInfo, tensor_q, tensor_k,
                  q_l1, k_l1_t, left, right, acc, qk_ub):
    cur_q = q_l1.current()
    cur_k_t = k_l1_t.current()
    pl.load(cur_q, tensor_q, [runInfo.bIdx, runInfo.s1Off, runInfo.nIdx, 0], order=[1, 3])
    pl.load(cur_k_t, tensor_k, [runInfo.bIdx, 0, runInfo.nIdx, 0], order=[3, 1])
    pl.move(left.current(), cur_q)
    pl.move(right.current(), cur_k_t)
    pl.matmul(acc.current(), left.current(), right.current())
    pl.move(qk_ub.current(), acc.current(),
            acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
    pl.system.set_cross_core(
        pipe=pl.PipeType.FIX, event_id=EVT_C2V_S,
        sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)


# ================================================================
#  IterateMmDyV — MM1: dP = dO @ V^T  (cube stage)
# ================================================================
def iterate_mm_dyv(constInfo, runInfo, tensor_dy, tensor_v,
                   do_l1, v_l1_t, left, right, acc, qk_ub):
    cur_do = do_l1.current()
    cur_v_t = v_l1_t.current()
    pl.load(cur_do, tensor_dy, [runInfo.bIdx, runInfo.s1Off, runInfo.nIdx, 0], order=[1, 3])
    pl.load(cur_v_t, tensor_v, [runInfo.bIdx, 0, runInfo.nIdx, 0], order=[3, 1])
    pl.move(left.current(), cur_do)
    pl.move(right.current(), cur_v_t)
    pl.matmul(acc.current(), left.current(), right.current())
    pl.move(qk_ub.current(), acc.current(),
            acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
    pl.system.set_cross_core(
        pipe=pl.PipeType.FIX, event_id=EVT_C2V_DP,
        sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)


# ================================================================
#  IterateMmDsK — MM3: dQ += dS @ K  (cube stage)
# ================================================================
def iterate_mm_dsk(constInfo, runInfo, s2_off,
                   tensor_ds_ws, tensor_k, tensor_dq_ws,
                   ds_l1, k_l1_n, left, right, acc_d):
    cur_ds = ds_l1.current()
    cur_k_n = k_l1_n.current()
    pl.load(cur_ds, tensor_ds_ws, [runInfo.bIdx, runInfo.nIdx, runInfo.s1Off, s2_off])
    pl.load(cur_k_n, tensor_k, [runInfo.bIdx, s2_off, runInfo.nIdx, 0], order=[1, 3])
    pl.move(left.current(), cur_ds)
    pl.move(right.current(), cur_k_n)
    pl.matmul(acc_d.current(), left.current(), right.current())
    pl.store(tensor_dq_ws, acc_d.current(),
             [runInfo.bIdx, runInfo.s1Off, runInfo.nIdx, 0],
             atomic=pl.AtomicType.AtomicAdd)


# ================================================================
#  IterateMmDsQ — MM4: dK += dS^T @ Q  (cube stage)
# ================================================================
def iterate_mm_dsq(constInfo, runInfo, s2_off,
                   tensor_ds_ws, tensor_q, tensor_dk_ws,
                   ds_t_l1, q_l1, left, right, acc_d):
    cur_ds_t = ds_t_l1.current()
    cur_q = q_l1.current()
    pl.load(cur_ds_t, tensor_ds_ws,
            [runInfo.bIdx, runInfo.nIdx, runInfo.s1Off, s2_off], order=[3, 2])
    pl.load(cur_q, tensor_q, [runInfo.bIdx, runInfo.s1Off, runInfo.nIdx, 0], order=[1, 3])
    pl.move(left.current(), cur_ds_t)
    pl.move(right.current(), cur_q)
    pl.matmul(acc_d.current(), left.current(), right.current())
    pl.store(tensor_dk_ws, acc_d.current(),
             [runInfo.bIdx, s2_off, runInfo.nIdx, 0],
             atomic=pl.AtomicType.AtomicAdd)


# ================================================================
#  IterateMmPDy — MM5: dV += P^T @ dO  (cube stage)
# ================================================================
def iterate_mm_pdy(constInfo, runInfo, s2_off,
                   tensor_p_ws, tensor_dy, tensor_dv_ws,
                   p_t_l1, do_l1, left, right, acc_d):
    cur_p_t = p_t_l1.current()
    cur_do = do_l1.current()
    pl.load(cur_p_t, tensor_p_ws,
            [runInfo.bIdx, runInfo.nIdx, runInfo.s1Off, s2_off], order=[3, 2])
    pl.load(cur_do, tensor_dy, [runInfo.bIdx, runInfo.s1Off, runInfo.nIdx, 0], order=[1, 3])
    pl.move(left.current(), cur_p_t)
    pl.move(right.current(), cur_do)
    pl.matmul(acc_d.current(), left.current(), right.current())
    pl.store(tensor_dv_ws, acc_d.current(),
             [runInfo.bIdx, s2_off, runInfo.nIdx, 0],
             atomic=pl.AtomicType.AtomicAdd)


# ================================================================
#  ProcessVec2 — V2: P = softmax(S*scale, max, sum)  (vector stage)
# ================================================================
def process_vec2(constInfo, runInfo, s2_off,
                 tensor_smax, tensor_ssum, tensor_p_ws,
                 qk_ub, p_ub, tmp_ub, max_ub, sum_ub):
    cur_qk = qk_ub.current()
    cur_p = p_ub.current()
    cur_tmp = tmp_ub.current()
    cur_max = max_ub.current()
    cur_sum = sum_ub.current()

    pl.load(cur_max, tensor_smax, [runInfo.s1Off, runInfo.smaxOff])
    pl.load(cur_sum, tensor_ssum, [runInfo.s1Off, runInfo.smaxOff])

    pl.mul(cur_qk, cur_qk, constInfo.scaleValue)
    pl.expand_sub(cur_qk, cur_qk, cur_max, dim=0)
    pl.exp(cur_tmp, cur_qk)
    pl.expand_div(cur_tmp, cur_tmp, cur_sum, dim=0)
    pl.cast(cur_p, cur_tmp, mode=pl.RoundMode.CAST_ROUND)
    pl.store(tensor_p_ws, cur_p, [runInfo.bIdx, runInfo.nIdx, runInfo.s1Off, s2_off])

    pl.system.set_cross_core(
        pipe=pl.PipeType.MTE3, event_id=EVT_V2C_P,
        sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)


# ================================================================
#  ProcessVec3 — V3: dS = softmax_grad(dP, P) * scale  (vector stage)
# ================================================================
def process_vec3(constInfo, runInfo, s2_off,
                 tensor_p_ws, tensor_ds_ws,
                 qk_ub, p_ub, ds_ub, tmp_ub, dpsum_ub):
    cur_qk = qk_ub.current()
    cur_p = p_ub.current()
    cur_ds = ds_ub.current()
    cur_tmp = tmp_ub.current()
    cur_dpsum = dpsum_ub.current()

    pl.load(cur_p, tensor_p_ws, [runInfo.bIdx, runInfo.nIdx, runInfo.s1Off, s2_off])

    pl.cast(cur_tmp, cur_p, mode=pl.RoundMode.CAST_NONE)
    pl.mul(cur_tmp, cur_qk, cur_tmp)
    pl.sum(cur_dpsum, cur_tmp, cur_qk, dim=0)
    pl.cast(cur_qk, cur_p, mode=pl.RoundMode.CAST_NONE)
    pl.expand_mul(cur_qk, cur_qk, cur_dpsum, dim=0)
    pl.sub(cur_tmp, cur_tmp, cur_qk)
    pl.mul(cur_tmp, cur_tmp, constInfo.scaleValue)
    pl.cast(cur_ds, cur_tmp, mode=pl.RoundMode.CAST_ROUND)
    pl.store(tensor_ds_ws, cur_ds, [runInfo.bIdx, runInfo.nIdx, runInfo.s1Off, s2_off])

    pl.system.set_cross_core(
        pipe=pl.PipeType.MTE3, event_id=EVT_V2C_DS,
        sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)


# ================================================================
#  Kernel
# ================================================================
@pl.jit(auto_mutex=True)
def flash_attention_score_grad(
    query: pl.Ptr[pl.DT_UINT8],
    key: pl.Ptr[pl.DT_UINT8],
    value: pl.Ptr[pl.DT_UINT8],
    dy: pl.Ptr[pl.DT_UINT8],
    softmax_max: pl.Ptr[pl.DT_UINT8],
    softmax_sum: pl.Ptr[pl.DT_UINT8],
    dq_workspace: pl.Ptr[pl.DT_UINT8],
    dk_workspace: pl.Ptr[pl.DT_UINT8],
    dv_workspace: pl.Ptr[pl.DT_UINT8],
    ds_workspace: pl.Ptr[pl.DT_UINT8],
    p_workspace: pl.Ptr[pl.DT_UINT8],
    tiling: FagTiling,
):
    # ---- SetConstInfo ----
    constInfo = set_const_info(tiling)

    # ---- Reconstruct GM tensor views (BSND layout) ----
    tensor_q = pl.make_tensor(query, [constInfo.bSize, constInfo.s1Size, constInfo.n2Size, TILE_D],
                              [constInfo.s1Size * constInfo.nd, constInfo.nd, TILE_D, 1], dtype=pl.DT_FP16)
    tensor_k = pl.make_tensor(key, [constInfo.bSize, constInfo.s2Size, constInfo.n2Size, TILE_D],
                              [constInfo.s2Size * constInfo.nd, constInfo.nd, TILE_D, 1], dtype=pl.DT_FP16)
    tensor_v = pl.make_tensor(value, [constInfo.bSize, constInfo.s2Size, constInfo.n2Size, TILE_D],
                              [constInfo.s2Size * constInfo.nd, constInfo.nd, TILE_D, 1], dtype=pl.DT_FP16)
    tensor_dy = pl.make_tensor(dy, [constInfo.bSize, constInfo.s1Size, constInfo.n2Size, TILE_D],
                               [constInfo.s1Size * constInfo.nd, constInfo.nd, TILE_D, 1], dtype=pl.DT_FP16)
    tensor_smax = pl.make_tensor(softmax_max, [constInfo.s1Size, constInfo.bnTotal],
                                 [1, constInfo.s1Size], dtype=pl.DT_FP32)
    tensor_ssum = pl.make_tensor(softmax_sum, [constInfo.s1Size, constInfo.bnTotal],
                                 [1, constInfo.s1Size], dtype=pl.DT_FP32)
    tensor_dq_ws = pl.make_tensor(dq_workspace, [constInfo.bSize, constInfo.s1Size, constInfo.n2Size, TILE_D],
                                  [constInfo.s1Size * constInfo.nd, constInfo.nd, TILE_D, 1], dtype=pl.DT_FP32)
    tensor_dk_ws = pl.make_tensor(dk_workspace, [constInfo.bSize, constInfo.s2Size, constInfo.n2Size, TILE_D],
                                  [constInfo.s2Size * constInfo.nd, constInfo.nd, TILE_D, 1], dtype=pl.DT_FP32)
    tensor_dv_ws = pl.make_tensor(dv_workspace, [constInfo.bSize, constInfo.s2Size, constInfo.n2Size, TILE_D],
                                  [constInfo.s2Size * constInfo.nd, constInfo.nd, TILE_D, 1], dtype=pl.DT_FP32)
    tensor_ds_ws = pl.make_tensor(ds_workspace, [constInfo.bSize, constInfo.n2Size, constInfo.s1Size, constInfo.s2Size],
                                  [constInfo.n2Size * constInfo.s1s2, constInfo.s1s2, constInfo.s2Size, 1], dtype=pl.DT_FP16)
    tensor_p_ws = pl.make_tensor(p_workspace, [constInfo.bSize, constInfo.n2Size, constInfo.s1Size, constInfo.s2Size],
                                 [constInfo.n2Size * constInfo.s1s2, constInfo.s1s2, constInfo.s2Size, 1], dtype=pl.DT_FP16)

    core_id = pl.get_block_idx()
    num_cores = pl.get_block_num()

    # --- Shared UB TileGroups (used by both cube and vector sections) ---
    qk_ub = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_QK, mutex_ids=[10])
    p_ub = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_P, mutex_ids=[11])
    ds_ub = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_DS, mutex_ids=[15])
    tmp_ub = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_TMP, mutex_ids=[14])
    max_ub = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, 1], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, layout=pl.DN),
        addrs=UB_MAX, mutex_ids=[12])
    sum_ub = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, 1], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, layout=pl.DN),
        addrs=UB_SUM, mutex_ids=[13])
    dpsum_ub = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, 1], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, layout=pl.DN),
        addrs=UB_DPSUM, mutex_ids=[16])

    # ==================================================================
    #  CUBE SECTION — Process()
    # ==================================================================
    with pl.section_cube():
        q_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_M, TILE_D], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, compact=1),
            addrs=L1_Q, mutex_ids=[0])
        k_l1_t = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_D, TILE_N], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZN, compact=1),
            addrs=L1_K, mutex_ids=[1])
        k_l1_n = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_N, TILE_D], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, compact=1),
            addrs=L1_K_N, mutex_ids=[17])
        v_l1_t = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_D, TILE_N], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZN, compact=1),
            addrs=L1_V, mutex_ids=[6])
        do_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_M, TILE_D], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, compact=1),
            addrs=L1_DO, mutex_ids=[5])
        ds_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, compact=1),
            addrs=L1_DS, mutex_ids=[7])
        ds_t_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_N, TILE_M], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZN, compact=1),
            addrs=L1_DS_T, mutex_ids=[8])
        p_t_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_N, TILE_M], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZN, compact=1),
            addrs=L1_P_T, mutex_ids=[9])
        left = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Left, compact=1),
            addrs=L0A_ADDR, mutex_ids=[2])
        right = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_N, TILE_D], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Right, compact=1),
            addrs=L0B_ADDR, mutex_ids=[3])
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, compact=1),
            addrs=L0C_ADDR, mutex_ids=[4])
        acc_d = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_M, TILE_D], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, compact=1),
            addrs=L0C_ADDR, mutex_ids=[4])

        for work_id in pl.range(core_id, constInfo.totalWork, num_cores):
            runInfo = set_run_info(constInfo, work_id)

            for s2_o in pl.range(0, constInfo.s2Outer):
                s2_off = s2_o * TILE_N

                # ---- MM2: S = Q @ K^T ----
                iterate_mm_qk(constInfo, runInfo, tensor_q, tensor_k,
                              q_l1, k_l1_t, left, right, acc, qk_ub)

                # Wait for V2 to finish (UB is free for dP)
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.FIX, event_id=EVT_V2C_P,
                    sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

                # ---- MM1: dP = dO @ V^T ----
                iterate_mm_dyv(constInfo, runInfo, tensor_dy, tensor_v,
                               do_l1, v_l1_t, left, right, acc, qk_ub)

                # Wait for V3 to finish (dS and P are in GM)
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE2, event_id=EVT_V2C_DS,
                    sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

                # ---- MM3: dQ += dS @ K ----
                iterate_mm_dsk(constInfo, runInfo, s2_off,
                               tensor_ds_ws, tensor_k, tensor_dq_ws,
                               ds_l1, k_l1_n, left, right, acc_d)

                # ---- MM4: dK += dS^T @ Q ----
                iterate_mm_dsq(constInfo, runInfo, s2_off,
                               tensor_ds_ws, tensor_q, tensor_dk_ws,
                               ds_t_l1, q_l1, left, right, acc_d)

                # ---- MM5: dV += P^T @ dO ----
                iterate_mm_pdy(constInfo, runInfo, s2_off,
                               tensor_p_ws, tensor_dy, tensor_dv_ws,
                               p_t_l1, do_l1, left, right, acc_d)

    # ==================================================================
    #  VECTOR SECTION — Process()
    # ==================================================================
    with pl.section_vector():
        for work_id in pl.range(core_id, constInfo.totalWork, num_cores):
            runInfo = set_run_info(constInfo, work_id)

            for s2_o in pl.range(0, constInfo.s2Outer):
                s2_off = s2_o * TILE_N

                # ---- V2: P = softmax(S * scale, fwd_max, fwd_sum) ----
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V, event_id=EVT_C2V_S,
                    sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

                process_vec2(constInfo, runInfo, s2_off,
                             tensor_smax, tensor_ssum, tensor_p_ws,
                             qk_ub, p_ub, tmp_ub, max_ub, sum_ub)

                # ---- V3: dS = softmax_grad(dP, P) * scale ----
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V, event_id=EVT_C2V_DP,
                    sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

                process_vec3(constInfo, runInfo, s2_off,
                             tensor_p_ws, tensor_ds_ws,
                             qk_ub, p_ub, ds_ub, tmp_ub, dpsum_ub)
