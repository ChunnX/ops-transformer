# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FlashAttention performance kernel (DN mode) — *dynamic rank* variant with BSND inputs.

Kernel signature matches ops-transformer's FlashAttentionScore (23 params + workspace + tiling):
    query, key, value, real_shift, drop_mask, padding_mask, atten_mask,
    prefix, actual_seq_qlen, actual_seq_kvlen, q_start_idx, kv_start_idx,
    d_scale_q, d_scale_k, d_scale_v, query_rope, key_rope, sink, p_scale,
    softmax_max, softmax_sum, softmax_out, attention_out, workspace, tiling

All tensor params are declared as typed pointers (pl.Ptr[pl.FP16] etc).
Unused optional params are declared but not accessed in the kernel body.
The actual shape/stride is supplied at runtime through a *tiling* dataclass:

    @dataclass
    class OpTiling:
        b: int
        sq: int
        n: int
        skv: int
        d: int

The kernel reconstructs typed tensor views via ``pl.make_tensor``.
Work distribution: total work ``B * N`` is flattened and distributed across cores
using ``pl.get_block_num()`` and ``pl.get_block_idx()``.

Attention mask: a fixed [FIXED_MASK_S, FIXED_MASK_S] UINT8 causal mask (1=mask out, 0=keep),
loaded into VEC UB and applied during softmax on diagonal KV tiles (ki==skv_tiles-1).
"""

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import pypto_pro.runtime as fe
import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import tilingkey_field
logging.basicConfig(level=logging.INFO)
# ================================================================
#  Configuration — change QK_PRELOAD to tune pre-compute depth
# ================================================================
QK_PRELOAD = 2          # How many KV tiles to pre-compute QK ahead
FIFO_SIZE = QK_PRELOAD + 1  # Exp-corr FIFO depth (avoids read/write collision)

# ================================================================
#  Tile dimensions and constants
# ================================================================
TS = 128
TKV = 128
TD = 128
TS_HALF = TS // 2
SCALE = 1.0 / math.sqrt(TD)
FLOAT_REP_SIZE = 64  # elements per fp32 register
D_LOOPS = TD // FLOAT_REP_SIZE
REDUCE_SIZE = 1
minValue = -1e9
block_stride_dn = TKV >> 1 | 0x1
REPEAT_STRIDE_DN = 1
vec1_s2_stride_dn = TKV * 8

# Cube tile byte sizes
Q_F16 = TS * TD * 2
KT_F16 = TKV * TD * 2
V_F16 = TKV * TD * 2
P_F16 = TS * TKV * 2
QK_HALF_F32 = TKV * TS * 4
PV_HALF_F32 = TS * TD * 4

# ---- MAT (512KB) ----
MA0 = 0
MA0_PONG = MA0 + Q_F16
MA1 = Q_F16 * 2
MA1_PONG = MA1 + KT_F16
MA2 = MA1 + KT_F16 * 2
MA2_PONG = MA2 + P_F16
MA2_PONG2 = MA2_PONG + P_F16
MA3 = MA2 + P_F16 * 3
MA3_PONG = MA3 + V_F16
LA0 = 0
LA1 = KT_F16
RA0 = 0
RA1 = Q_F16
CA0 = 0
CA1 = QK_HALF_F32
CA2 = QK_HALF_F32 * 2
CA3 = QK_HALF_F32 * 3

# ---- VEC addresses (248KB on a5) ----
VB4_KV = TS_HALF * TKV * 4
VB2_KV = TS_HALF * 2 * (TKV // 2 + 1) * 2
VB2_KV_HALF = VB2_KV // 2
VB4 = TS_HALF * TD * 4
VB2 = TS_HALF * TD * 2
VB_RED = TS_HALF * 1 * 4

VA0 = 0
VA2 = VA0 + 2 * VB4_KV
VA2_DN = VA2 + VB2_KV_HALF
VA2B = VA2 + VB2_KV
VA2B_DN = VA2B + VB2_KV_HALF
VA8 = VA2 + 2 * VB2_KV
VA7 = VA8 + 2 * VB4
VA9 = VA7 + VB4
VA_GMAX0 = VA9 + VB2
VA_GMAX1 = VA_GMAX0 + VB_RED
VA_GMAX2 = VA_GMAX1 + VB_RED
VA_GSUM0 = VA_GMAX2 + VB_RED
VA_GSUM1 = VA_GSUM0 + VB_RED
VA_GSUM2 = VA_GSUM1 + VB_RED
VA_EC0 = VA_GSUM2 + VB_RED
VA_EC1 = VA_EC0 + VB_RED
VA_EC2 = VA_EC1 + VB_RED
VB_MASK = TKV * TS_HALF
VA_MASK = VA_EC2 + VB_RED
VA_END = VA_MASK + 2 * VB_MASK
assert VA_END <= 248 * 1024, f"VEC overflow: {VA_END} > {248 * 1024}"

FIXED_MASK_S = 2048

MaskS = pl.DynVar('MaskS')
NumRanges = pl.DynVar('NumRanges')
B2 = pl.DynVar('B')
N2 = pl.DynVar('N')
Sq2 = pl.DynVar('Sq')
Skv2 = pl.DynVar('Skv')
D2 = pl.DynVar('D')

# Cross-core event IDs
QK_READY_FORWARD_IDS = (0, 1)
QK_READY_BARKWARD_IDS = (2, 3)
P_READY_FORWARD_IDS = (4, 5, 6)
PV_READY_FORWARD_IDS = (7, 8)
PV_READY_BARKWARD_IDS = (9, 10)

QK_FORWARD_MAX_EID = 2
QK_BARKWARD_MAX_EID = 4
P_FORWARD_MAX_EID = 7
PV_FORWARD_MAX_EID = 9
PV_BARKWARD_MAX_EID = 11

PV_CORE_STRIDE = 2 * FIFO_SIZE * TS


# ================================================================
#  Tiling data — describes the runtime shape of the (rank-erased) BSND inputs
# ================================================================
@dataclass
class OpTiling:
    b: int
    sq: int
    n: int
    skv: int
    d: int


# ================================================================
#  TilingKey — compile-time template specialization (matches ops-transformer)
# ================================================================
_FA_TILING_KEY_LAUNCH = {
    "KernelTypeKey": 0, "ImplMode": 0, "Layout": 1,
    "S1TemplateType": 128, "S2TemplateType": 128,
    "DTemplateType": 128, "DvTemplateType": 128,
    "PseMode": 9, "HasAtten": 0,
    "HasDrop": 0, "HasRope": 0,
    "OutDtype": 0, "Regbase": 1, "OptionalDn": 0,
}


class FaTilingKey:
    KernelTypeKey = tilingkey_field(bits=2, values=[0, 1])
    ImplMode = tilingkey_field(bits=2, values=[0, 1, 2])
    Layout = tilingkey_field(bits=4, values=[0, 1, 2, 3, 4])
    S1TemplateType = tilingkey_field(bits=10, values=[0, 16, 64, 128, 256])
    S2TemplateType = tilingkey_field(bits=10, values=[0, 16, 32, 64, 128, 256, 512])
    DTemplateType = tilingkey_field(bits=12, values=[0, 16, 32, 48, 64, 80, 96, 128, 160, 192, 256, 768])
    DvTemplateType = tilingkey_field(bits=12, values=[0, 16, 32, 48, 64, 80, 96, 128, 160, 192, 256, 768])
    PseMode = tilingkey_field(bits=4, values=[0, 1, 2, 3, 4, 9])
    HasAtten = tilingkey_field(bits=1, values=[0, 1])
    HasDrop = tilingkey_field(bits=1, values=[0, 1])
    HasRope = tilingkey_field(bits=1, values=[0, 1])
    OutDtype = tilingkey_field(bits=2, values=[0, 1, 2])
    Regbase = tilingkey_field(bits=1, values=[0, 1])
    OptionalDn = tilingkey_field(bits=1, values=[0, 1])

    def is_valid(self, key):
        (KernelTypeKey, ImplMode, Layout, S1TemplateType, S2TemplateType,
         DTemplateType, DvTemplateType, PseMode, HasAtten, HasDrop,
         HasRope, OutDtype, Regbase, OptionalDn) = key
        if KernelTypeKey != 0: return False
        if ImplMode != 0: return False
        if Layout != 1: return False
        if S1TemplateType != 128: return False
        if S2TemplateType != 128: return False
        if DTemplateType != 128: return False
        if DvTemplateType != 128: return False
        if PseMode != 9: return False
        if HasDrop != 0: return False
        if HasRope != 0: return False
        if OutDtype != 0: return False
        if Regbase != 1: return False
        if OptionalDn != 0: return False
        return True


# ================================================================
#  VF softmax functions (merged no-update / update with is_update flag)
# ================================================================
def process_vec1_dn_vf(input_tile, x_exp_tile, max_tile, exp_max_tile, sum_tile, is_update,
                       mask_tile, need_mask):
    """Softmax DN VF kernel — merged no-update / update / causal-mask versions."""
    with pl.section_vf():
        src0 = vf.RegTensor(dtype=pl.FP32)
        src1 = vf.RegTensor(dtype=pl.FP32)
        src2 = vf.RegTensor(dtype=pl.FP32)
        src3 = vf.RegTensor(dtype=pl.FP32)
        max0 = vf.RegTensor(dtype=pl.FP32)
        max1 = vf.RegTensor(dtype=pl.FP32)
        max2 = vf.RegTensor(dtype=pl.FP32)
        max3 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_f32_0 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_f32_1 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_f32_2 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_f32_3 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_exp_0 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_exp_1 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_exp_2 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_exp_3 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_sum_0 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_sum_1 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_sum_2 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_sum_3 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_sum0 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_sum1 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_max_f32_b = vf.RegTensor(dtype=pl.FP32)
        vreg_l0 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_exp_even_f16 = vf.RegTensor(dtype=pl.FP16)
        vreg_x_exp_odd_f16 = vf.RegTensor(dtype=pl.FP16)
        vreg_x_exp_even_f16_1 = vf.RegTensor(dtype=pl.FP16)
        vreg_x_exp_odd_f16_1 = vf.RegTensor(dtype=pl.FP16)
        vreg_x_exp_f16_pack = vf.RegTensor(dtype=pl.FP16)
        vreg_x_exp_f16_packa = vf.RegTensor(dtype=pl.FP16)
        vreg_x_exp_f16_1_pack = vf.RegTensor(dtype=pl.FP16)
        vreg_x_exp_f16_1_packa = vf.RegTensor(dtype=pl.FP16)
        preg_108 = vf.create_mask(pattern="ALL", dtype=pl.FP32)
        preg_136 = vf.update_mask(128, dtype=pl.FP16)
        preg_compare0 = vf.create_mask(pattern="ALL", dtype=pl.UINT32)
        preg_compare1 = vf.create_mask(pattern="ALL", dtype=pl.UINT32)
        preg_compare2 = vf.create_mask(pattern="ALL", dtype=pl.UINT32)
        preg_compare3 = vf.create_mask(pattern="ALL", dtype=pl.UINT32)
        vreg_min = vf.RegTensor(dtype=pl.FP32)
        src_ub0 = input_tile
        src_ub1 = input_tile + TS_HALF
        src_ub2 = input_tile + TS_HALF * 2
        src_ub3 = input_tile + TS_HALF * 3
        x_exp_1 = x_exp_tile + TKV * 4
        vf.duplicate(max0, minValue)
        vf.duplicate(max1, minValue)
        vf.duplicate(max2, minValue)
        vf.duplicate(max3, minValue)
        vf.duplicate(vreg_min, minValue)
        for iter_m in pl.range(0, TKV // 4):
            vf.load_align(src0, src_ub0, iter_m * TS_HALF * 4)
            vf.load_align(src1, src_ub1, iter_m * TS_HALF * 4)
            vf.load_align(src2, src_ub2, iter_m * TS_HALF * 4)
            vf.load_align(src3, src_ub3, iter_m * TS_HALF * 4)
            if need_mask:
                vf.load_align(preg_compare0, mask_tile, iter_m * TS_HALF * 4, dist="DS")
                vf.load_align(preg_compare1, mask_tile, TS_HALF + iter_m * TS_HALF * 4, dist="DS")
                vf.load_align(preg_compare2, mask_tile, TS_HALF * 2 + iter_m * TS_HALF * 4, dist="DS")
                vf.load_align(preg_compare3, mask_tile, TS_HALF * 3 + iter_m * TS_HALF * 4, dist="DS")
                vf.select(src0, vreg_min, src0, preg_compare0)
                vf.select(src1, vreg_min, src1, preg_compare1)
                vf.select(src2, vreg_min, src2, preg_compare2)
                vf.select(src3, vreg_min, src3, preg_compare3)
                vf.store_align(input_tile + (iter_m * TS_HALF * 4), src0, preg_108)
                vf.store_align(input_tile + (TS_HALF + iter_m * TS_HALF * 4), src1, preg_108)
                vf.store_align(input_tile + (TS_HALF * 2 + iter_m * TS_HALF * 4), src2, preg_108)
                vf.store_align(input_tile + (TS_HALF * 3 + iter_m * TS_HALF * 4), src3, preg_108)
            vf.max(max0, max0, src0, preg_108)
            vf.max(max1, max1, src1, preg_108)
            vf.max(max2, max2, src2, preg_108)
            vf.max(max3, max3, src3, preg_108)
        if is_update:
            vf.load_align(vreg_x_max_f32_b, max_tile, 0)
        vf.max(max0, max0, max2, preg_108)
        vf.max(max1, max1, max3, preg_108)
        vf.max(max0, max0, max1, preg_108)
        vf.muls(max0, max0, SCALE, preg_108)
        if is_update:
            vf.max(max0, max0, vreg_x_max_f32_b, preg_108)
            vf.fused_exp_sub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_108)
        vf.store_align(max_tile, max0, preg_108)
        if is_update:
            vf.store_align(exp_max_tile, vreg_x_max_f32_b, preg_108)
        vf.duplicate(vreg_x_sum_0, 0.0, preg_108)
        vf.duplicate(vreg_x_sum_1, 0.0, preg_108)
        vf.duplicate(vreg_x_sum_2, 0.0, preg_108)
        vf.duplicate(vreg_x_sum_3, 0.0, preg_108)
        if need_mask:
            vf.MemBar(mode="VST_VLD")
        for i0 in pl.range(0, TKV // 4):
            vf.load_align(vreg_x_f32_0, input_tile, i0 * TS_HALF)
            vf.load_align(vreg_x_f32_1, input_tile, TKV * TS_HALF // 4 + i0 * TS_HALF)
            vf.load_align(vreg_x_f32_2, input_tile, TKV * TS_HALF // 2 + i0 * TS_HALF)
            vf.load_align(vreg_x_f32_3, input_tile, TKV * TS_HALF // 2 + TKV * TS_HALF // 4 + i0 * TS_HALF)
            vf.muls(vreg_x_f32_0, vreg_x_f32_0, SCALE, preg_108)
            vf.muls(vreg_x_f32_1, vreg_x_f32_1, SCALE, preg_108)
            vf.muls(vreg_x_f32_2, vreg_x_f32_2, SCALE, preg_108)
            vf.muls(vreg_x_f32_3, vreg_x_f32_3, SCALE, preg_108)
            vf.fused_exp_sub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_108)
            vf.fused_exp_sub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_108)
            vf.fused_exp_sub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_108)
            vf.fused_exp_sub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_108)
            vf.cast(vreg_x_exp_even_f16, vreg_x_exp_0, preg_108, layout="ZERO")
            vf.cast(vreg_x_exp_odd_f16, vreg_x_exp_2, preg_108, layout="ZERO")
            vf.de_interleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16)
            vf.cast(vreg_x_exp_even_f16_1, vreg_x_exp_1, preg_108, layout="ZERO")
            vf.cast(vreg_x_exp_odd_f16_1, vreg_x_exp_3, preg_108, layout="ZERO")
            vf.de_interleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1)
            vf.store_align(x_exp_tile, vreg_x_exp_f16_pack, preg_136,
                          block_stride=block_stride_dn, repeat_stride=REPEAT_STRIDE_DN,
                          data_copy_mode="DATA_BLOCK_COPY", post_update=True)
            vf.add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_108)
            vf.add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_108)
            vf.store_align(x_exp_1, vreg_x_exp_f16_1_pack, preg_136,
                          block_stride=block_stride_dn, repeat_stride=REPEAT_STRIDE_DN,
                          data_copy_mode="DATA_BLOCK_COPY", post_update=True)
            vf.add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_108)
            vf.add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_108)
        vf.add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_108)
        vf.add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_108)
        vf.add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_108)
        if is_update:
            vf.load_align(vreg_l0, sum_tile, 0)
            vf.mul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_108)
            vf.add(vreg_l0, vreg_l0, vreg_x_sum0, preg_108)
            vf.store_align(sum_tile, vreg_l0, preg_108)
        else:
            vf.store_align(sum_tile, vreg_x_sum0, preg_108)


def flash_update_basic_vf(dst_tile, cur_tile, pre_tile, exp_max_tile):
    """FlashUpdateBasicVF: dst = pre * exp_max + cur"""
    with pl.section_vf():
        vreg_exp_max = vf.RegTensor(dtype=pl.FP32)
        vreg_input_pre = vf.RegTensor(dtype=pl.FP32)
        vreg_input_cur = vf.RegTensor(dtype=pl.FP32)
        vreg_mul = vf.RegTensor(dtype=pl.FP32)
        vreg_add = vf.RegTensor(dtype=pl.FP32)
        preg_all = vf.create_mask(pattern="ALL", dtype=pl.FP32)
        for i in pl.range(0, TS_HALF):
            vf.load_align(vreg_exp_max, exp_max_tile, i * REDUCE_SIZE, dist="BRC_B32")
            for j in pl.range(0, D_LOOPS):
                vf.load_align(vreg_input_pre, pre_tile, i * TD + j * FLOAT_REP_SIZE)
                vf.load_align(vreg_input_cur, cur_tile, i * TD + j * FLOAT_REP_SIZE)
                vf.mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_all)
                vf.add(vreg_add, vreg_mul, vreg_input_cur, preg_all)
                vf.store_align(dst_tile[i * TD + j * FLOAT_REP_SIZE], vreg_add, preg_all)


def flash_update_last_basic_vf(dst_tile, cur_tile, pre_tile, exp_max_tile, exp_sum_tile):
    """FlashUpdateLastBasicVF: dst = (pre * exp_max + cur) / exp_sum"""
    with pl.section_vf():
        vreg_exp_max = vf.RegTensor(dtype=pl.FP32)
        vreg_exp_sum = vf.RegTensor(dtype=pl.FP32)
        vreg_input_pre = vf.RegTensor(dtype=pl.FP32)
        vreg_input_cur = vf.RegTensor(dtype=pl.FP32)
        vreg_mul = vf.RegTensor(dtype=pl.FP32)
        vreg_add = vf.RegTensor(dtype=pl.FP32)
        vreg_div = vf.RegTensor(dtype=pl.FP32)
        preg_all = vf.create_mask(pattern="ALL", dtype=pl.FP32)
        for i in pl.range(0, TS_HALF):
            vf.load_align(vreg_exp_max, exp_max_tile, i * REDUCE_SIZE, dist="BRC_B32")
            vf.load_align(vreg_exp_sum, exp_sum_tile, i * REDUCE_SIZE, dist="BRC_B32")
            for j in pl.range(0, D_LOOPS):
                vf.load_align(vreg_input_pre, pre_tile, i * TD + j * FLOAT_REP_SIZE)
                vf.load_align(vreg_input_cur, cur_tile, i * TD + j * FLOAT_REP_SIZE)
                vf.mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_all)
                vf.add(vreg_add, vreg_mul, vreg_input_cur, preg_all)
                vf.div(vreg_div, vreg_add, vreg_exp_sum, preg_all)
                vf.store_align(dst_tile[i * TD + j * FLOAT_REP_SIZE], vreg_div, preg_all)


def last_div_vf(dst_tile, cur_tile, exp_sum_tile):
    """LastDivNewVF: dst = cur / exp_sum"""
    with pl.section_vf():
        vreg_input_cur = vf.RegTensor(dtype=pl.FP32)
        vreg_div = vf.RegTensor(dtype=pl.FP32)
        vreg_exp_sum = vf.RegTensor(dtype=pl.FP32)
        preg_all = vf.create_mask(pattern="ALL", dtype=pl.FP32)
        for i in pl.range(0, TS_HALF):
            vf.load_align(vreg_exp_sum, exp_sum_tile, i * REDUCE_SIZE, dist="BRC_B32")
            for j in pl.range(0, D_LOOPS):
                vf.load_align(vreg_input_cur, cur_tile, i * TD + j * FLOAT_REP_SIZE)
                vf.div(vreg_div, vreg_input_cur, vreg_exp_sum, preg_all)
                vf.store_align(dst_tile[i * TD + j * FLOAT_REP_SIZE], vreg_div, preg_all)


def compute_qk(b_idx, n_idx, ki, qi, cur_q_slot, q, k, k_l1_db, left_db, right_db, acc_db, qk_vec_db, task_id):
    if ki == 0:
        pl.load_tile(cur_q_slot, q, [b_idx, qi, n_idx, 0], layout="dn", tile_dims=[1, 3])
    cur_k_slot = k_l1_db.next()
    pl.load_tile(cur_k_slot, k, [b_idx, ki, n_idx, 0], tile_dims=[1, 3])
    qk_left = left_db.next()
    qk_right = right_db.next()
    qk_acc = acc_db.next()
    pl.move(qk_left, cur_k_slot)
    pl.move(qk_right, cur_q_slot)
    pl.matmul(qk_acc, qk_left, qk_right)
    qk_dst = qk_vec_db.next()
    pl.system.wait_cross_core(
        pipe=pl.PipeType.FIX,
        event_id=QK_READY_BARKWARD_IDS[task_id % 2],
    )
    pl.move(qk_dst, qk_acc, acc_to_vec_mode="dual_split_n")
    pl.system.set_cross_core(
        pipe=pl.PipeType.FIX,
        event_id=QK_READY_FORWARD_IDS[task_id % 2],
    )


def compute_pv(b_idx, n_idx, ctx, v_l1_db, p_mat_db, left_db, right_db, acc_db, pv_vec_db, v):
    pl.system.wait_cross_core(
        pipe=pl.PipeType.MTE1,
        event_id=P_READY_FORWARD_IDS[ctx.task_id_mod3],
    )
    cur_v_slot = v_l1_db.next()
    cur_p_slot = p_mat_db.next()
    pv_left = left_db.next()
    pv_right = right_db.next()
    pv_acc = acc_db.next()
    pl.load_tile(cur_v_slot, v, [b_idx, ctx.ki, n_idx, 0], tile_dims=[1, 3])
    pl.move(pv_left, cur_p_slot)
    pl.move(pv_right, cur_v_slot)
    pl.matmul(pv_acc, pv_left, pv_right)
    pv_dst = pv_vec_db.next()
    pl.system.wait_cross_core(
        pipe=pl.PipeType.FIX,
        event_id=PV_READY_BARKWARD_IDS[ctx.task_id_mod2],
    )
    pl.move(pv_dst, pv_acc, acc_to_vec_mode="dual_split_m")
    pl.system.set_cross_core(
        pipe=pl.PipeType.FIX,
        event_id=PV_READY_FORWARD_IDS[ctx.task_id_mod2],
    )


def compute_p(p_ctx, global_max_rm_buf, global_sum_rm_buf, exp_corr_rm_fifo, qk_vec_db, p_f16_db, p_f16_main_db,
    p_f16_back_db, p_mat_db, sub_id, attn_mask, mask_db, need_mask):
    pl.system.wait_cross_core(
        pipe=pl.PipeType.V,
        event_id=QK_READY_FORWARD_IDS[p_ctx.task_id_mod2],
    )
    global_max_rm_cur = global_max_rm_buf[p_ctx.q_count_mod3]
    global_sum_rm_cur = global_sum_rm_buf[p_ctx.q_count_mod3]
    exp_corr_rm_cur = exp_corr_rm_fifo[p_ctx.task_id_mod3]
    qk_slot = qk_vec_db.next()
    p_f16_slot = p_f16_db.next()
    p_f16_main_slot = p_f16_main_db.next()
    p_f16_back_slot = p_f16_back_db.next()
    p_mat_slot = p_mat_db.next()
    if need_mask:
        if p_ctx.ki == p_ctx.skv_tiles - 1:
            mask_slot = mask_db.next()
            pl.load(mask_slot, attn_mask, [0, sub_id * TS_HALF])
            if p_ctx.ki == 0:
                process_vec1_dn_vf(qk_slot, p_f16_slot, global_max_rm_cur,
                                   exp_corr_rm_cur, global_sum_rm_cur, 0, mask_slot, 1)
            else:
                process_vec1_dn_vf(qk_slot, p_f16_slot, global_max_rm_cur,
                                   exp_corr_rm_cur, global_sum_rm_cur, 1, mask_slot, 1)
        else:
            if p_ctx.ki == 0:
                process_vec1_dn_vf(qk_slot, p_f16_slot, global_max_rm_cur,
                                   exp_corr_rm_cur, global_sum_rm_cur, 0, qk_slot, 0)
            else:
                process_vec1_dn_vf(qk_slot, p_f16_slot, global_max_rm_cur,
                                   exp_corr_rm_cur, global_sum_rm_cur, 1, qk_slot, 0)
    else:
        if p_ctx.ki == 0:
            process_vec1_dn_vf(qk_slot, p_f16_slot, global_max_rm_cur,
                               exp_corr_rm_cur, global_sum_rm_cur, 0, qk_slot, 0)
        else:
            process_vec1_dn_vf(qk_slot, p_f16_slot, global_max_rm_cur,
                               exp_corr_rm_cur, global_sum_rm_cur, 1, qk_slot, 0)
    pl.system.set_cross_core(
        pipe=pl.PipeType.V,
        event_id=QK_READY_BARKWARD_IDS[p_ctx.task_id_mod2],
    )
    pl.insert(p_mat_slot, p_f16_main_slot, index_col=TS_HALF * sub_id)
    pl.insert(p_mat_slot, p_f16_back_slot, index_row=TKV // 2, index_col=TS_HALF * sub_id)
    pl.system.set_cross_core(
        pipe=pl.PipeType.MTE3,
        event_id=P_READY_FORWARD_IDS[p_ctx.task_id_mod3],
    )


def compute_gu(b_idx, n_idx, g_ctx, sub_id, pv_vec_db, global_sum_rm_buf, exp_corr_rm_fifo,
    running_o_buf, o_f16_buf, o):
    pv_slot = pv_vec_db.next()
    gsum_gu = global_sum_rm_buf[g_ctx.q_count_mod3]
    exp_corr_gu = exp_corr_rm_fifo[g_ctx.task_id_mod3]
    pl.system.wait_cross_core(
        pipe=pl.PipeType.V,
        event_id=PV_READY_FORWARD_IDS[g_ctx.task_id_mod2],
    )
    if g_ctx.ki == 0:
        pl.move(running_o_buf, pv_slot)
    else:
        if g_ctx.ki < g_ctx.skv_tiles - 1:
            flash_update_basic_vf(running_o_buf, pv_slot, running_o_buf, exp_corr_gu)
        else:
            flash_update_last_basic_vf(
                running_o_buf,
                pv_slot,
                running_o_buf,
                exp_corr_gu,
                gsum_gu,
            )
            pl.cast(o_f16_buf, running_o_buf, target_type=pl.FP16, mode="round")
            pl.store_tile(o, o_f16_buf, [b_idx, g_ctx.qi * 2 + sub_id, n_idx, 0], tile_dims=[1, 3])
    pl.system.set_cross_core(
        pipe=pl.PipeType.V,
        event_id=PV_READY_BARKWARD_IDS[g_ctx.task_id_mod2],
    )
    if g_ctx.ki == g_ctx.skv_tiles - 1:
        if g_ctx.ki == 0:
            last_div_vf(running_o_buf, running_o_buf, gsum_gu)
            pl.cast(o_f16_buf, running_o_buf, target_type=pl.FP16, mode="round")
            pl.store_tile(o, o_f16_buf, [b_idx, g_ctx.qi * 2 + sub_id, n_idx, 0], tile_dims=[1, 3])


# ================================================================
#  Kernel — dynamic rank: inputs are raw pointers, shapes come from tiling
# ================================================================
@fe.jit(codegen_mode="cce", auto_mutex=True, tiling_key=FaTilingKey, timeout=300)
def flash_attention_score(
    query: pl.Ptr[pl.UINT8],
    key: pl.Ptr[pl.UINT8],
    value: pl.Ptr[pl.UINT8],
    real_shift: pl.Ptr[pl.UINT8],
    drop_mask: pl.Ptr[pl.UINT8],
    padding_mask: pl.Ptr[pl.UINT8],
    atten_mask: pl.Ptr[pl.UINT8],
    prefix: pl.Ptr[pl.UINT8],
    actual_seq_qlen: pl.Ptr[pl.UINT8],
    actual_seq_kvlen: pl.Ptr[pl.UINT8],
    q_start_idx: pl.Ptr[pl.UINT8],
    kv_start_idx: pl.Ptr[pl.UINT8],
    d_scale_q: pl.Ptr[pl.UINT8],
    d_scale_k: pl.Ptr[pl.UINT8],
    d_scale_v: pl.Ptr[pl.UINT8],
    query_rope: pl.Ptr[pl.UINT8],
    key_rope: pl.Ptr[pl.UINT8],
    sink: pl.Ptr[pl.UINT8],
    p_scale: pl.Ptr[pl.UINT8],
    softmax_max: pl.Ptr[pl.UINT8],
    softmax_sum: pl.Ptr[pl.UINT8],
    softmax_out: pl.Ptr[pl.UINT8],
    attention_out: pl.Ptr[pl.UINT8],
    workspace: pl.Ptr[pl.UINT8],
    tiling: OpTiling,
) -> None:

    tensor_q = pl.make_tensor(query, [tiling.b, tiling.sq, tiling.n, tiling.d],
                              [tiling.sq * tiling.n * tiling.d, tiling.n * tiling.d, tiling.d, 1], dtype=pl.FP16)
    tensor_k = pl.make_tensor(key, [tiling.b, tiling.skv, tiling.n, tiling.d],
                              [tiling.skv * tiling.n * tiling.d, tiling.n * tiling.d, tiling.d, 1], dtype=pl.FP16)
    tensor_v = pl.make_tensor(value, [tiling.b, tiling.skv, tiling.n, tiling.d],
                              [tiling.skv * tiling.n * tiling.d, tiling.n * tiling.d, tiling.d, 1], dtype=pl.FP16)
    tensor_o = pl.make_tensor(attention_out, [tiling.b, tiling.sq, tiling.n, tiling.d],
                              [tiling.sq * tiling.n * tiling.d, tiling.n * tiling.d, tiling.d, 1], dtype=pl.FP16)
    tensor_attn_mask = pl.make_tensor(atten_mask, [FIXED_MASK_S, FIXED_MASK_S], [FIXED_MASK_S, 1])

    n_dim = tiling.n
    sq_dim = tiling.sq
    skv_dim = tiling.skv
    sq_tiles = (sq_dim + (TS - 1)) // TS
    skv_tiles = (skv_dim + (TKV - 1)) // TKV
    core_id = pl.get_block_idx()
    num_cores = pl.get_block_num()
    total_work = tiling.b * tiling.n
    work_per_core = (total_work + num_cores - 1) // num_cores
    work_start = core_id * work_per_core
    work_end = work_start + work_per_core
    if work_end > total_work:
        work_end = total_work

    # ===== Cross-core shared buffers =====
    qk_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TKV, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA0, mutex_ids=[14, 15])
    pv_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TS_HALF, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA8, mutex_ids=[16, 17])
    p_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[TS, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat,
                         blayout=1, slayout=2),
        addrs=MA2, mutex_ids=[18, 19, 20])

    # =================== CUBE SECTION ===================
    with pl.section_cube():
        q_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TS], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat,
                             blayout=1, slayout=2),
            addrs=MA0, mutex_ids=[0, 1])
        k_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat),
            addrs=MA1, mutex_ids=[2, 3])
        v_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat),
            addrs=MA3, mutex_ids=[4, 5])
        left_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Left),
            addrs=LA0, mutex_ids=[6, 7])
        right_db = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TS], dtype=pl.FP16, target_memory=pl.MemorySpace.Right),
            addrs=RA0, mutex_ids=[8, 9])
        acc_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TS], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc),
            addrs=CA0, mutex_ids=[10, 11, 12, 13])

        task_id = 0
        ctx_arr = pl.struct_array(4, "CubeCtx", b_idx=0, n_idx=0, ki=0, task_id_mod2=0, task_id_mod3=0)
        for work_id in pl.range(work_start, work_end):
            b_idx = work_id // n_dim
            n_idx = work_id % n_dim
            for qi in pl.range(0, sq_tiles):
                cur_q_slot = q_l1_db.next()
                causal_skv = HasAtten * (qi + 1) + (1 - HasAtten) * skv_tiles  # noqa: F821
                ki_end = causal_skv
                if work_id + 1 >= work_end:
                    if qi + 1 >= sq_tiles:
                        ki_end = causal_skv + QK_PRELOAD
                for ki in pl.range(0, ki_end):
                    if ki < causal_skv:
                        ctx_curr = ctx_arr[task_id % 4]
                        ctx_curr.b_idx = b_idx
                        ctx_curr.n_idx = n_idx
                        ctx_curr.ki = ki
                        ctx_curr.task_id_mod2 = task_id % 2
                        ctx_curr.task_id_mod3 = task_id % 3
                        compute_qk(b_idx, n_idx, ki, qi, cur_q_slot, tensor_q, tensor_k, k_l1_db, left_db, right_db,
                                   acc_db, qk_vec_db, task_id)

                    if task_id > 1:
                        prev = ctx_arr[(task_id + 2) % 4]
                        compute_pv(prev.b_idx, prev.n_idx, prev, v_l1_db, p_mat_db, left_db, right_db, acc_db,
                                   pv_vec_db, tensor_v)
                    task_id = task_id + 1

    # =================== VECTOR SECTION ===================
    with pl.section_vector():
        p_f16_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV // 2 + 1, TS_HALF * 2], dtype=pl.FP16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[VA2, VA2B], mutex_ids=[0, 1])
        p_f16_main_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV // 2 + 1, TS_HALF], dtype=pl.FP16,
                             target_memory=pl.MemorySpace.Vec,
                             valid_shape=[TKV // 2, TS_HALF], blayout=2, slayout=1),
            addrs=[VA2, VA2B], mutex_ids=[0, 1])
        p_f16_back_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV // 2 + 1, TS_HALF], dtype=pl.FP16,
                             target_memory=pl.MemorySpace.Vec,
                             valid_shape=[TKV // 2, TS_HALF], blayout=2, slayout=1),
            addrs=[VA2_DN, VA2B_DN], mutex_ids=[0, 1])
        red_rm_type = pl.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        gmax_rm_0 = pl.make_tile(red_rm_type, addr=VA_GMAX0, size=VB_RED)
        gmax_rm_1 = pl.make_tile(red_rm_type, addr=VA_GMAX1, size=VB_RED)
        gmax_rm_2 = pl.make_tile(red_rm_type, addr=VA_GMAX2, size=VB_RED)
        global_max_rm_buf = (gmax_rm_0, gmax_rm_1, gmax_rm_2)
        gsum_rm_0 = pl.make_tile(red_rm_type, addr=VA_GSUM0, size=VB_RED)
        gsum_rm_1 = pl.make_tile(red_rm_type, addr=VA_GSUM1, size=VB_RED)
        gsum_rm_2 = pl.make_tile(red_rm_type, addr=VA_GSUM2, size=VB_RED)
        global_sum_rm_buf = (gsum_rm_0, gsum_rm_1, gsum_rm_2)
        ec_rm_0 = pl.make_tile(red_rm_type, addr=VA_EC0, size=VB_RED)
        ec_rm_1 = pl.make_tile(red_rm_type, addr=VA_EC1, size=VB_RED)
        ec_rm_2 = pl.make_tile(red_rm_type, addr=VA_EC2, size=VB_RED)
        exp_corr_rm_fifo = (ec_rm_0, ec_rm_1, ec_rm_2)
        o_f16_g = pl.make_tile_group(
            type=pl.TileType(shape=[TS_HALF, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
            addrs=VA9, mutex_ids=[11])
        o_f16_buf = o_f16_g.next()
        running_o_g = pl.make_tile_group(
            type=pl.TileType(shape=[TS_HALF, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
            addrs=VA7, mutex_ids=[12])
        running_o_buf = running_o_g.next()
        mask_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TS_HALF], dtype=pl.UINT8, target_memory=pl.MemorySpace.Vec),
            addrs=VA_MASK, mutex_ids=[8, 9])

        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[1])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[1])

        sub_id = pl.index_cast(pl.get_subblock_idx())
        task_id = 0
        q_count = 0
        ctx_arr = pl.struct_array(4, "VecCtx", b_idx=0, n_idx=0, qi=0, ki=0, skv_tiles=0,
                                  task_id_mod2=0, task_id_mod3=0, q_count_mod3=0)
        for work_id in pl.range(work_start, work_end):
            b_idx = work_id // n_dim
            n_idx = work_id % n_dim
            for qi in pl.range(0, sq_tiles):
                causal_skv = HasAtten * (qi + 1) + (1 - HasAtten) * skv_tiles  # noqa: F821
                ki_end = causal_skv
                if work_id + 1 >= work_end:
                    if qi + 1 >= sq_tiles:
                        ki_end = causal_skv + 3
                for ki in pl.range(0, ki_end):
                    if ki < causal_skv:
                        ctx_curr = ctx_arr[task_id % 4]
                        ctx_curr.b_idx = b_idx
                        ctx_curr.n_idx = n_idx
                        ctx_curr.qi = qi
                        ctx_curr.ki = ki
                        ctx_curr.skv_tiles = causal_skv
                        ctx_curr.task_id_mod2 = task_id % 2
                        ctx_curr.task_id_mod3 = task_id % 3
                        ctx_curr.q_count_mod3 = q_count % 3
                    if task_id > 0:
                        if ki < causal_skv + 1:
                            p_ctx = ctx_arr[(task_id + 3) % 4]
                            compute_p(p_ctx, global_max_rm_buf, global_sum_rm_buf, exp_corr_rm_fifo,
                                qk_vec_db, p_f16_db, p_f16_main_db,
                                p_f16_back_db, p_mat_db, sub_id, tensor_attn_mask, mask_db, HasAtten)  # noqa: F821
                    if task_id > 2:
                        g_ctx = ctx_arr[(task_id + 1) % 4]
                        compute_gu(g_ctx.b_idx, g_ctx.n_idx, g_ctx, sub_id, pv_vec_db, global_sum_rm_buf,
                                   exp_corr_rm_fifo, running_o_buf, o_f16_buf, tensor_o)
                    task_id = task_id + 1
                q_count = q_count + 1

