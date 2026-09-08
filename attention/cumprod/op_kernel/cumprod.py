# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cumprod kernel (PyPTO-Pro DSL) — cumulative product along an axis.

沿 dim 的累积乘积 y[i] = prod(x[..., 0..i])，支持 exclusive / reverse 两个运行时属性
（4 种组合共享同一编译产物）；fp16/bf16 在 fp32 域累乘、逐元素写回收窄（每元素一次舍入）。

ops-transformer 接入约定（op_host/CMakeLists.txt 的 enable_pypto_kernel(cumprod) 在
configure 阶段对本文件做 codegen，接入方式参考 dense_lightning_indexer_softmax_lse_v2）：
  * CumprodTiling_tiling.h        —— host tiling 直接使用的 tiling 结构体（force-include）
  * CumprodTilingKey_tilingkey.h  —— ASCENDC_TPL tiling key 声明
  * cumprod_pypto_infer.cpp       —— 供 .i infer 使用的自包含源码

kernel 参数按算子原型顺序排列：输入（x、axis）、输出（y）、workspace、tiling。
axis 为 0-D 标量张量输入，其值在 host 侧 tiling 阶段经 TilingInputsDataDependency 读取
（三折轴 outer/scan/inner 已折算进 tiling），kernel 本体不访问该指针。
离线（OPC）编译下 kernel 参数用 pl.Ptr + pl.make_tensor，从 tiling 取 shape，
不使用 pl.Tensor[[pl.DYNAMIC, ...]] 动态维度参数。

TilingKey（对应方式参考 DenseLiLseTilingKey）：bit0 = mode（0=FOLD inner>1 折叠路径；
1=SCAN inner==1 扫描路径），host tiling 按 "TilingKey: bit0=mode" 位布局手工打包后
经 context->SetTilingKey() 下发。
exclusive / reverse 保持 TilingData 运行时 bool 字段（四组合零额外编译变体）。
"""

from dataclasses import dataclass

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.tilingkey import TilingKeyField


# #############################################################################
# 算子关键常量（与已验收设计一致；fp32 / fp16+bf16 两 class 的 tile 几何）
# #############################################################################
ROWS32 = 64        # fp32：SCAN 每 job 扫描线数（=寄存器 lane 数）
ROWS16 = 128       # fp16/bf16：SCAN 每 job 扫描线数
W32    = 64        # fp32：SCAN 列 chunk 宽（1 寄存器）
W16    = 128       # 窄：SCAN 列 chunk 宽（1 寄存器）
WP32   = 72        # fp32 gather/scatter tile 物理行距：72×4B=288B=9×32B（奇数倍，防 bank 冲突）
WP16   = 144       # 窄 gather/scatter tile 物理行距：144×2B=288B=9×32B（奇数倍）
WS_P32 = 72        # ws 前缀 tile 行距（fp32 class，容纳 ROWS32=64）
WS_P16 = 136       # ws 前缀 tile 行距（窄 class，容纳 ROWS16=128）：136×4B=544B=17×32B
MAXP   = 64        # 每扫描线最大分段数（ws tile 行数上限）
TILE_S = 8         # FOLD：每 tile 扫描行数
CHUNK32 = 64       # FOLD：inner chunk（fp32，1 寄存器）
CHUNK16 = 128      # FOLD：inner chunk（窄，1 寄存器）
EVT_AGGREADY = 0   # pass A→pass B 跨核阶段屏障事件号（[0,16)，全 kernel 唯一事件）

# 片上地址映射（UB 布局；exclusive/reverse 不新增 tile）
# fp32 class
A32_SCAN_IN0, A32_SCAN_IN1 = 0x00000, 0x04800
A32_SCAN_OUT0, A32_SCAN_OUT1 = 0x09000, 0x0D800
A32_WS = 0x12000
A32_CARRY = 0x16800
A32_FOLD_IN0, A32_FOLD_IN1 = 0x16900, 0x17100
A32_FOLD_OUT0, A32_FOLD_OUT1 = 0x17900, 0x18100
# fp16/bf16 class
A16_SCAN_IN0, A16_SCAN_IN1 = 0x00000, 0x09000
A16_SCAN_OUT0, A16_SCAN_OUT1 = 0x12000, 0x1B000
A16_WS = 0x24000
A16_CARRY = 0x2C800
A16_FOLD_IN0, A16_FOLD_IN1 = 0x2CA00, 0x2D200
A16_FOLD_OUT0, A16_FOLD_OUT1 = 0x2DA00, 0x2E200


# ================================================================
#  Tiling data —— host 侧 tiling（cumprod_tiling.cpp）按同名字段填充；
#  字段布局与 op_host 中 REGISTER_TILING_DATA_CLASS(Cumprod, CumprodTilingData) 一一对应
# ================================================================
@dataclass
class CumprodTiling:
    """outer/scan/inner 为三折轴规范视图（axis 已在 host 侧折算）；
    exclusive / reverse 为 bool 运行时字段，四 (exclusive, reverse) 组合
    共享同一编译产物（不进 TilingKey）。
    """
    outer: int
    scan: int
    inner: int
    seg_count: int
    seg_len: int
    exclusive: bool
    reverse: bool


# ================================================================
#  TilingKey（对应方式参考 DenseLiLseTilingKey）——
#  bit0：mode（0=FOLD，1=SCAN），host tiling（cumprod_tiling.cpp）按
#  "TilingKey: bit0=mode" 的位布局手工打包后经 context->SetTilingKey() 选择
# ================================================================
class CumprodTilingKey:
    mode = TilingKeyField(bits=1, values=[0, 1])


# #############################################################################
# VF 函数（io_dtype 为 kernel datatype 类型变量，编译期常量；cast 分支 trace 期裁剪）
# 属性参数：exclusive / reverse 以 pl.DT_INT64（0/1）运行时标量传入
# fold_rows_vf / scan_cols_vf；agg_cols_vf 仅 reverse（pass A 无输出、无位移语义）。
#
# 实现备注（已验收工程规避——非设计偏离；目标版本 bisheng 在 VF 指令选择阶段对
# 两类 reverse 索引构造崩溃，经 4 轮消融定位）：i64 派生标量作 vadds 操作数、
# 运行时 if 分支 phi 构造 gather/scatter 索引。规避写法（数学语义/指令条数/
# 标量非负域不变）：
#   - 方向经循环边界算术承载：forward 体 bound = valid-valid*reverse（reverse=1 时
#     0 次），reverse 体 bound = valid*reverse——每列恰执行一个方向体，无条件索引构造；
#   - vcm1/vrm1 单次迭代循环（pl.range(valid-1, valid)）借边界 uint16 化把
#     valid-1 窄化为 i32，vadds 标量与 FOLD 行偏移全部停留在 i32 域；
#   - FOLD 行偏移仅进入指针算术（load_align(in_tile + rr*CHUNK, 0)），
#     不作为 VF 内在函数操作数。
# #############################################################################
@pl.vector_function
def fold_rows_vf(in_tile, out_tile, carry_tile,
                 valid_rows: pl.DT_INT64, valid_cols: pl.DT_INT64,
                 init_mode: pl.DT_INT64,
                 exclusive: pl.DT_INT64, reverse: pl.DT_INT64):
    """FOLD（inner>1）：lane=inner，逐扫描行 acc=acc*x_row fold，无跨 lane 移动。

    init_mode: 0=种子 1.0（乘法单位元）；1=从 carry tile 续接上一 tile 的累乘状态。
    reverse:   行迭代顺序反向（扫描坐标行 r ↔ tile 内存行 rr=valid_rows-1-r；
               方向经循环边界算术承载）。
    exclusive: store-before-mul——先写出 acc（全部前序元素之积；扫描起点为
               种子 1.0）再 acc=acc*x；inclusive 先乘再写。末端 inclusive 值
               永不写出（末端丢弃）。
    carry 演化与 exclusive/inclusive 无关（恒为扫描方向已处理元素之积，含当前行）。
    """
    preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    if io_dtype == pl.DT_FP32:
        # ── fp32 class：直乘，无 cast ──
        if init_mode == 0:
            acc = vf.full(1.0, preg32, dtype=pl.DT_FP32)
        else:
            acc = vf.load_align(carry_tile, 0)
        for vrm1 in pl.range(valid_rows - 1, valid_rows):
            for r in pl.range(0, valid_rows - valid_rows * reverse):
                x_reg = vf.load_align(in_tile + r * CHUNK32, 0)
                if exclusive == 1:
                    vf.store_align(out_tile + r * CHUNK32, acc, preg32)
                    acc = vf.mul(acc, x_reg, preg32)
                else:
                    acc = vf.mul(acc, x_reg, preg32)
                    vf.store_align(out_tile + r * CHUNK32, acc, preg32)
            for r in pl.range(0, valid_rows * reverse):
                rr = vrm1 - r                        # 降序行（i32 标量，仅指针算术）
                x_reg = vf.load_align(in_tile + rr * CHUNK32, 0)
                if exclusive == 1:
                    vf.store_align(out_tile + rr * CHUNK32, acc, preg32)
                    acc = vf.mul(acc, x_reg, preg32)
                else:
                    acc = vf.mul(acc, x_reg, preg32)
                    vf.store_align(out_tile + rr * CHUNK32, acc, preg32)
        vf.store_align(carry_tile, acc, preg32)
    else:
        # ── fp16/bf16 class：加载后即拓宽 fp32，链式累乘，写回单次收窄 ──
        preg_n = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)
        if init_mode == 0:
            acc_lo = vf.full(1.0, preg32, dtype=pl.DT_FP32)
            acc_hi = vf.full(1.0, preg32, dtype=pl.DT_FP32)
        else:
            acc_lo = vf.load_align(carry_tile, 0)
            acc_hi = vf.load_align(carry_tile, 64)
        for vrm1 in pl.range(valid_rows - 1, valid_rows):
            for r in pl.range(0, valid_rows - valid_rows * reverse):
                x_n = vf.load_align(in_tile + r * CHUNK16, 0)
                lo = vf.astype(x_n, preg_n, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
                hi = vf.astype(x_n, preg_n, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
                if exclusive == 1:
                    out_lo = vf.astype(acc_lo, preg32, dtype=io_dtype,
                                       layout=pl.CastLayout.ZERO,
                                       round_mode=pl.VFRoundMode.CAST_RINT,
                                       saturate=pl.SaturateMode.OFF)
                    out_hi = vf.astype(acc_hi, preg32, dtype=io_dtype,
                                       layout=pl.CastLayout.ONE,
                                       round_mode=pl.VFRoundMode.CAST_RINT,
                                       saturate=pl.SaturateMode.OFF)
                    out_n = vf.xor(out_lo, out_hi, preg_n)   # 零位互补合并
                    vf.store_align(out_tile + r * CHUNK16, out_n, preg_n)
                    acc_lo = vf.mul(acc_lo, lo, preg32)
                    acc_hi = vf.mul(acc_hi, hi, preg32)
                else:
                    acc_lo = vf.mul(acc_lo, lo, preg32)
                    acc_hi = vf.mul(acc_hi, hi, preg32)
                    out_lo = vf.astype(acc_lo, preg32, dtype=io_dtype,
                                       layout=pl.CastLayout.ZERO,
                                       round_mode=pl.VFRoundMode.CAST_RINT,
                                       saturate=pl.SaturateMode.OFF)
                    out_hi = vf.astype(acc_hi, preg32, dtype=io_dtype,
                                       layout=pl.CastLayout.ONE,
                                       round_mode=pl.VFRoundMode.CAST_RINT,
                                       saturate=pl.SaturateMode.OFF)
                    out_n = vf.xor(out_lo, out_hi, preg_n)   # 零位互补合并
                    vf.store_align(out_tile + r * CHUNK16, out_n, preg_n)
            for r in pl.range(0, valid_rows * reverse):
                rr = vrm1 - r                        # 降序行（i32 标量，仅指针算术）
                x_n = vf.load_align(in_tile + rr * CHUNK16, 0)
                lo = vf.astype(x_n, preg_n, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
                hi = vf.astype(x_n, preg_n, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
                if exclusive == 1:
                    out_lo = vf.astype(acc_lo, preg32, dtype=io_dtype,
                                       layout=pl.CastLayout.ZERO,
                                       round_mode=pl.VFRoundMode.CAST_RINT,
                                       saturate=pl.SaturateMode.OFF)
                    out_hi = vf.astype(acc_hi, preg32, dtype=io_dtype,
                                       layout=pl.CastLayout.ONE,
                                       round_mode=pl.VFRoundMode.CAST_RINT,
                                       saturate=pl.SaturateMode.OFF)
                    out_n = vf.xor(out_lo, out_hi, preg_n)   # 零位互补合并
                    vf.store_align(out_tile + rr * CHUNK16, out_n, preg_n)
                    acc_lo = vf.mul(acc_lo, lo, preg32)
                    acc_hi = vf.mul(acc_hi, hi, preg32)
                else:
                    acc_lo = vf.mul(acc_lo, lo, preg32)
                    acc_hi = vf.mul(acc_hi, hi, preg32)
                    out_lo = vf.astype(acc_lo, preg32, dtype=io_dtype,
                                       layout=pl.CastLayout.ZERO,
                                       round_mode=pl.VFRoundMode.CAST_RINT,
                                       saturate=pl.SaturateMode.OFF)
                    out_hi = vf.astype(acc_hi, preg32, dtype=io_dtype,
                                       layout=pl.CastLayout.ONE,
                                       round_mode=pl.VFRoundMode.CAST_RINT,
                                       saturate=pl.SaturateMode.OFF)
                    out_n = vf.xor(out_lo, out_hi, preg_n)   # 零位互补合并
                    vf.store_align(out_tile + rr * CHUNK16, out_n, preg_n)
        # 跨 VF 调用的 carry：lo→[0:64]、hi→[64:128]（写读同布局，scratch 语义）
        vf.store_align(carry_tile, acc_lo, preg32)
        vf.store_align(carry_tile + 64, acc_hi, preg32)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def scan_cols_vf(in_tile, out_tile, carry_tile, ws_tile,
                 valid_rows: pl.DT_INT64, valid_cols: pl.DT_INT64,
                 init_mode: pl.DT_INT64, seg_idx: pl.DT_INT64,
                 exclusive: pl.DT_INT64, reverse: pl.DT_INT64):
    """SCAN（inner==1）：lane=扫描线，逐列 gather→(widen)→mul→(narrow)→scatter。

    init_mode: 0=种子 1.0；1=从 carry tile 续接；2=从 ws tile 逐段 gather 累乘出
    exclusive 段前缀积（k<seg_idx 的段聚合值连乘，扫描坐标段序，四组合同一——
    reverse 不改段序，exclusive 同前缀）作 carry 初值。
    reverse:   列处理顺序反向（扫描坐标列 j ↔ tile 内存列 col_off=valid_cols-1-j，
               索引 idx = g + col_off，vf.adds 标量非负，每列 1 条）。
    exclusive: scatter-before-mul（每列先散布 acc 再累乘；扫描起点散布种子 1.0，
               末端 inclusive 值永不散布）；carry 演化与 inclusive 同一。
    """
    if io_dtype == pl.DT_FP32:
        preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
        preg_u = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
        g = vf.arange(0, dtype=pl.DT_UINT32)
        g = vf.muls(g, WP32, preg_u)                 # lane L → 行 L 起始（元素）
        if init_mode == 0:
            acc = vf.full(1.0, preg32, dtype=pl.DT_FP32)
        elif init_mode == 1:
            acc = vf.load_align(carry_tile, 0)
        else:
            # 段前 exclusive 前缀积：lane L 读 ws_tile[k, L]，k<seg_idx 连乘
            g_ws = vf.arange(0, dtype=pl.DT_UINT32)
            acc = vf.full(1.0, preg32, dtype=pl.DT_FP32)
            for k in pl.range(0, seg_idx):
                idx_ws = vf.adds(g_ws, k * WS_P32, preg_u)
                c = vf.gather(ws_tile, idx_ws, preg32)
                acc = vf.mul(acc, c, preg32)
        for vcm1 in pl.range(valid_cols - 1, valid_cols):
            for j in pl.range(0, valid_cols - valid_cols * reverse):
                idx = vf.adds(g, j, preg_u)          # 升序（i32 标量）
                x_r = vf.gather(in_tile, idx, preg32)
                if exclusive == 1:
                    vf.scatter(out_tile, acc, idx, preg32)
                    acc = vf.mul(acc, x_r, preg32)
                else:
                    acc = vf.mul(acc, x_r, preg32)
                    vf.scatter(out_tile, acc, idx, preg32)
            for j in pl.range(0, valid_cols * reverse):
                idx = vf.adds(g, vcm1 - j, preg_u)   # 降序（i32 标量）
                x_r = vf.gather(in_tile, idx, preg32)
                if exclusive == 1:
                    vf.scatter(out_tile, acc, idx, preg32)
                    acc = vf.mul(acc, x_r, preg32)
                else:
                    acc = vf.mul(acc, x_r, preg32)
                    vf.scatter(out_tile, acc, idx, preg32)
        vf.store_align(carry_tile, acc, preg32)
    else:
        preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
        preg_u32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
        preg_n = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)
        preg_u = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
        g = vf.arange(0, dtype=pl.DT_UINT16)
        g = vf.muls(g, WP16, preg_u)                 # lane L → 行 L 起始（元素）
        if init_mode == 0:
            acc_lo = vf.full(1.0, preg32, dtype=pl.DT_FP32)
            acc_hi = vf.full(1.0, preg32, dtype=pl.DT_FP32)
        elif init_mode == 1:
            acc_lo = vf.load_align(carry_tile, 0)
            acc_hi = vf.load_align(carry_tile, 64)
        else:
            # ws 为 fp32：偶/奇扫描线两次 gather（lane j ↔ 行 2j / 2j+1）
            g_ws = vf.arange(0, dtype=pl.DT_UINT32)
            g_ws = vf.muls(g_ws, 2, preg_u32)
            acc_lo = vf.full(1.0, preg32, dtype=pl.DT_FP32)
            acc_hi = vf.full(1.0, preg32, dtype=pl.DT_FP32)
            for k in pl.range(0, seg_idx):
                idx_lo = vf.adds(g_ws, k * WS_P16, preg_u32)
                c_lo = vf.gather(ws_tile, idx_lo, preg32)
                acc_lo = vf.mul(acc_lo, c_lo, preg32)
                idx_hi = vf.adds(g_ws, k * WS_P16 + 1, preg_u32)
                c_hi = vf.gather(ws_tile, idx_hi, preg32)
                acc_hi = vf.mul(acc_hi, c_hi, preg32)
        for vcm1 in pl.range(valid_cols - 1, valid_cols):
            for j in pl.range(0, valid_cols - valid_cols * reverse):
                idx = vf.adds(g, j, preg_u)              # 升序（i32 标量）
                x_n = vf.gather(in_tile, idx, preg_n)
                lo = vf.astype(x_n, preg_n, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
                hi = vf.astype(x_n, preg_n, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
                if exclusive == 1:
                    out_lo = vf.astype(acc_lo, preg32, dtype=io_dtype,
                                       layout=pl.CastLayout.ZERO,
                                       round_mode=pl.VFRoundMode.CAST_RINT,
                                       saturate=pl.SaturateMode.OFF)
                    out_hi = vf.astype(acc_hi, preg32, dtype=io_dtype,
                                       layout=pl.CastLayout.ONE,
                                       round_mode=pl.VFRoundMode.CAST_RINT,
                                       saturate=pl.SaturateMode.OFF)
                    out_n = vf.xor(out_lo, out_hi, preg_n)   # 零位互补合并
                    vf.scatter(out_tile, out_n, idx, preg_n)
                    acc_lo = vf.mul(acc_lo, lo, preg32)
                    acc_hi = vf.mul(acc_hi, hi, preg32)
                else:
                    acc_lo = vf.mul(acc_lo, lo, preg32)
                    acc_hi = vf.mul(acc_hi, hi, preg32)
                    out_lo = vf.astype(acc_lo, preg32, dtype=io_dtype,
                                       layout=pl.CastLayout.ZERO,
                                       round_mode=pl.VFRoundMode.CAST_RINT,
                                       saturate=pl.SaturateMode.OFF)
                    out_hi = vf.astype(acc_hi, preg32, dtype=io_dtype,
                                       layout=pl.CastLayout.ONE,
                                       round_mode=pl.VFRoundMode.CAST_RINT,
                                       saturate=pl.SaturateMode.OFF)
                    out_n = vf.xor(out_lo, out_hi, preg_n)   # 零位互补合并
                    vf.scatter(out_tile, out_n, idx, preg_n)
            for j in pl.range(0, valid_cols * reverse):
                idx = vf.adds(g, vcm1 - j, preg_u)       # 降序（i32 标量）
                x_n = vf.gather(in_tile, idx, preg_n)
                lo = vf.astype(x_n, preg_n, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
                hi = vf.astype(x_n, preg_n, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
                if exclusive == 1:
                    out_lo = vf.astype(acc_lo, preg32, dtype=io_dtype,
                                       layout=pl.CastLayout.ZERO,
                                       round_mode=pl.VFRoundMode.CAST_RINT,
                                       saturate=pl.SaturateMode.OFF)
                    out_hi = vf.astype(acc_hi, preg32, dtype=io_dtype,
                                       layout=pl.CastLayout.ONE,
                                       round_mode=pl.VFRoundMode.CAST_RINT,
                                       saturate=pl.SaturateMode.OFF)
                    out_n = vf.xor(out_lo, out_hi, preg_n)   # 零位互补合并
                    vf.scatter(out_tile, out_n, idx, preg_n)
                    acc_lo = vf.mul(acc_lo, lo, preg32)
                    acc_hi = vf.mul(acc_hi, hi, preg32)
                else:
                    acc_lo = vf.mul(acc_lo, lo, preg32)
                    acc_hi = vf.mul(acc_hi, hi, preg32)
                    out_lo = vf.astype(acc_lo, preg32, dtype=io_dtype,
                                       layout=pl.CastLayout.ZERO,
                                       round_mode=pl.VFRoundMode.CAST_RINT,
                                       saturate=pl.SaturateMode.OFF)
                    out_hi = vf.astype(acc_hi, preg32, dtype=io_dtype,
                                       layout=pl.CastLayout.ONE,
                                       round_mode=pl.VFRoundMode.CAST_RINT,
                                       saturate=pl.SaturateMode.OFF)
                    out_n = vf.xor(out_lo, out_hi, preg_n)   # 零位互补合并
                    vf.scatter(out_tile, out_n, idx, preg_n)
        vf.store_align(carry_tile, acc_lo, preg32)
        vf.store_align(carry_tile + 64, acc_hi, preg32)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def agg_cols_vf(in_tile, carry_tile,
                valid_rows: pl.DT_INT64, valid_cols: pl.DT_INT64,
                init_mode: pl.DT_INT64, is_final: pl.DT_INT64,
                reverse: pl.DT_INT64):
    """pass A 段聚合（inner==1 且 seg_count>1，kernel 内 SCAN 前段，无独立 launch）：
    段内自 1 顺序累乘（沿扫描方向）；无 scatter、无输出 tile、无 exclusive 语义
    （段聚合=段元素之积，与位移无关——故本函数不接收 exclusive）。

    init_mode: 0=种子 1.0；1=从 carry tile 续接。is_final=1（job 末 chunk）时
    窄 class 以 INTLV_B32 双搬出把 acc_lo/acc_hi 按扫描线顺序写入 stage（=carry
    tile），供 pl.store 直接写 GM ws[s, r0+L]；fp32 class 单 acc 天然行序，无需区分。
    reverse: 列迭代顺序反向（方向经循环边界算术承载）；段聚合的乘法顺序沿扫描
    方向（与 golden 顺序递推同向）。段聚合全程 fp32（跨核 carry 不收窄）。
    """
    if io_dtype == pl.DT_FP32:
        preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
        preg_u = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
        g = vf.arange(0, dtype=pl.DT_UINT32)
        g = vf.muls(g, WP32, preg_u)
        if init_mode == 0:
            acc = vf.full(1.0, preg32, dtype=pl.DT_FP32)
        else:
            acc = vf.load_align(carry_tile, 0)
        for vcm1 in pl.range(valid_cols - 1, valid_cols):
            for j in pl.range(0, valid_cols - valid_cols * reverse):
                idx = vf.adds(g, j, preg_u)              # 升序（i32 标量）
                x_r = vf.gather(in_tile, idx, preg32)
                acc = vf.mul(acc, x_r, preg32)
            for j in pl.range(0, valid_cols * reverse):
                idx = vf.adds(g, vcm1 - j, preg_u)       # 降序（i32 标量）
                x_r = vf.gather(in_tile, idx, preg32)
                acc = vf.mul(acc, x_r, preg32)
        vf.store_align(carry_tile, acc, preg32)      # lane L = 行 L，行序
    else:
        preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
        preg_n = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)
        preg_u = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
        g = vf.arange(0, dtype=pl.DT_UINT16)
        g = vf.muls(g, WP16, preg_u)
        if init_mode == 0:
            acc_lo = vf.full(1.0, preg32, dtype=pl.DT_FP32)
            acc_hi = vf.full(1.0, preg32, dtype=pl.DT_FP32)
        else:
            acc_lo = vf.load_align(carry_tile, 0)
            acc_hi = vf.load_align(carry_tile, 64)
        for vcm1 in pl.range(valid_cols - 1, valid_cols):
            for j in pl.range(0, valid_cols - valid_cols * reverse):
                idx = vf.adds(g, j, preg_u)              # 升序（i32 标量）
                x_n = vf.gather(in_tile, idx, preg_n)
                lo = vf.astype(x_n, preg_n, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
                hi = vf.astype(x_n, preg_n, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
                acc_lo = vf.mul(acc_lo, lo, preg32)
                acc_hi = vf.mul(acc_hi, hi, preg32)
            for j in pl.range(0, valid_cols * reverse):
                idx = vf.adds(g, vcm1 - j, preg_u)       # 降序（i32 标量）
                x_n = vf.gather(in_tile, idx, preg_n)
                lo = vf.astype(x_n, preg_n, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
                hi = vf.astype(x_n, preg_n, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
                acc_lo = vf.mul(acc_lo, lo, preg32)
                acc_hi = vf.mul(acc_hi, hi, preg32)
        if is_final == 1:
            # 交错双搬出：stage[2j]=acc_lo[j]（行 2j），stage[2j+1]=acc_hi[j]（行 2j+1）
            vf.store_align(carry_tile, acc_lo, acc_hi, preg32,
                           dist=pl.StoreDist.INTLV_B32)
        else:
            vf.store_align(carry_tile, acc_lo, preg32)
            vf.store_align(carry_tile + 64, acc_hi, preg32)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


# #############################################################################
# Kernel —— 单一 @pl.jit（函数名须与op_file名一致，即enable_pypto_kernel与opFile.value的"cumprod"）；
# FOLD / SCAN 两 mode 为 TilingKey bit0 的编译期特化
# （host tiling 经 SetTilingKey 选择变体）；SCAN 分段路径 = 同一次 launch 内
# pass A → 跨核阶段屏障 → pass B
# #############################################################################
@pl.jit(auto_mutex=True, tiling_key=CumprodTilingKey, datatype={"x": "io_dtype", "y": "io_dtype"})
def cumprod(
    x: pl.Ptr[pl.DT_UINT8],
    axis: pl.Ptr[pl.DT_INT64],
    y: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: CumprodTiling,
):
    # axis 为 0-D 标量张量输入（值已在 host tiling 阶段读取并折算为三折轴），
    # kernel 本体不访问该指针，仅按算子原型占位。
    # io_dtype 编译期常量：tile 尺寸/地址分支在 trace 期裁剪（死分支不进 IR）
    if io_dtype == pl.DT_FP32:
        ROWS = ROWS32
        WP = WP32
        WS_P = WS_P32
        W = W32
        CHUNK = CHUNK32
        A_SI0, A_SI1 = A32_SCAN_IN0, A32_SCAN_IN1
        A_SO0, A_SO1 = A32_SCAN_OUT0, A32_SCAN_OUT1
        A_WST = A32_WS
        A_CR = A32_CARRY
        A_FI0, A_FI1 = A32_FOLD_IN0, A32_FOLD_IN1
        A_FO0, A_FO1 = A32_FOLD_OUT0, A32_FOLD_OUT1
    else:
        ROWS = ROWS16
        WP = WP16
        WS_P = WS_P16
        W = W16
        CHUNK = CHUNK16
        A_SI0, A_SI1 = A16_SCAN_IN0, A16_SCAN_IN1
        A_SO0, A_SO1 = A16_SCAN_OUT0, A16_SCAN_OUT1
        A_WST = A16_WS
        A_CR = A16_CARRY
        A_FI0, A_FI1 = A16_FOLD_IN0, A16_FOLD_IN1
        A_FO0, A_FO1 = A16_FOLD_OUT0, A16_FOLD_OUT1

    # GM 视图（2D，元素 stride；host tiling 传入三折轴整数与两个 bool 属性）
    x2 = pl.make_tensor(x, [tiling.outer * tiling.scan, tiling.inner], dtype=io_dtype)
    y2 = pl.make_tensor(y, [tiling.outer * tiling.scan, tiling.inner], dtype=io_dtype)
    x1 = pl.make_tensor(x, [tiling.outer, tiling.scan], dtype=io_dtype)
    y1 = pl.make_tensor(y, [tiling.outer, tiling.scan], dtype=io_dtype)
    ws = pl.make_tensor(workspace, [tiling.seg_count, tiling.outer], dtype=pl.DT_FP32)

    # TileGroup（kernel 顶层声明；全 make_tile_group + auto_mutex，无 make_tile；
    # pass A 与 pass B 同 launch 内顺序共用 scan_in group，轮转游标跨 pass 连续
    # 推进，lifetime 由 auto_mutex 顺序化；carry 单 buffer 绝不轮转）
    io_tt = pl.TileType(shape=[ROWS, WP], dtype=io_dtype,
                        target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    scan_in = pl.make_tile_group(type=io_tt, addrs=[A_SI0, A_SI1], mutex_ids=[0, 1])
    scan_out = pl.make_tile_group(type=io_tt, addrs=[A_SO0, A_SO1], mutex_ids=[2, 3])
    ws_group = pl.make_tile_group(
        type=pl.TileType(shape=[MAXP, WS_P], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=[A_WST], mutex_ids=[4])
    carry_group = pl.make_tile_group(                 # 单 buffer，绝不 next() 轮转
        type=pl.TileType(shape=[1, ROWS], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=[A_CR], mutex_ids=[5])
    fold_tt = pl.TileType(shape=[TILE_S, CHUNK], dtype=io_dtype,
                          target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    fold_in = pl.make_tile_group(type=fold_tt, addrs=[A_FI0, A_FI1], mutex_ids=[6, 7])
    fold_out = pl.make_tile_group(type=fold_tt, addrs=[A_FO0, A_FO1], mutex_ids=[8, 9])

    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()
        carry_slot = carry_group.current()

        # TilingData bool 字段 → VF 标量参数（pl.DT_INT64，0/1；
        # tiling.exclusive / tiling.reverse 为 TilingData 常量，全核同值）
        excl = 0
        if tiling.exclusive:
            excl = 1
        rev = 0
        if tiling.reverse:
            rev = 1

        if mode == 0:  # noqa: F821  (injected by the tiling key)
            # ===================== mode=FOLD（inner>1） =====================
            n_chunk = (tiling.inner + CHUNK - 1) // CHUNK
            total_jobs = tiling.outer * n_chunk
            for job in pl.range(core_id, total_jobs, num_cores):
                o = job // n_chunk
                c0 = (job % n_chunk) * CHUNK
                valid_cols = pl.min(CHUNK, tiling.inner - c0)
                for s0 in pl.range(0, tiling.scan, TILE_S):
                    valid_rows = pl.min(TILE_S, tiling.scan - s0)
                    # reverse 方向：扫描坐标窗口 [s0, s0+valid_rows) 映射内存
                    # 连续窗口 [scan-s0-valid_rows, scan-s0)（零物理翻转，
                    # 2D DMA 不拆分；tile 行序=内存序，方向由 VF 行序承载）
                    row_base = o * tiling.scan + s0
                    if tiling.reverse:
                        row_base = o * tiling.scan + tiling.scan - s0 - valid_rows
                    in_slot = fold_in.next()
                    pl.set_validshape(in_slot, [valid_rows, valid_cols])
                    pl.load(in_slot, x2, [row_base, c0])
                    out_slot = fold_out.next()
                    pl.set_validshape(out_slot, [valid_rows, valid_cols])
                    init_mode = 1
                    if s0 == 0:
                        init_mode = 0
                    fold_rows_vf(in_slot, out_slot, carry_slot,
                                 valid_rows, valid_cols, init_mode, excl, rev)
                    # 同址写回（tile 行序=内存序，扫描坐标 t 的输出落内存 rev?scan-1-t:t）
                    pl.store(y2, out_slot, [row_base, c0])
        else:
            # ============ mode=SCAN（inner==1；seg_count>1 时含 kernel 内
            # ============ pass A → 跨核阶段屏障 → pass B） ============
            row_blocks = (tiling.outer + ROWS - 1) // ROWS
            total_jobs = row_blocks * tiling.seg_count

            if tiling.seg_count > 1:
                # ────── pass A（段聚合；与 pass B 同一 job 空间/核间分配；
                #         段界=扫描坐标（四组合同一）；exclusive 不影响本段） ──────
                for job in pl.range(core_id, total_jobs, num_cores):
                    rb = job // tiling.seg_count
                    s = job % tiling.seg_count
                    r0 = rb * ROWS
                    seg_beg = s * tiling.seg_len
                    seg_end = pl.min(tiling.scan, seg_beg + tiling.seg_len)
                    valid_rows = pl.min(ROWS, tiling.outer - r0)
                    for c0 in pl.range(seg_beg, seg_end, W):
                        valid_cols = pl.min(W, seg_end - c0)
                        # reverse 方向：扫描坐标列窗 [c0, c0+valid_cols) →
                        # 内存连续窗 [scan-c0-valid_cols, scan-c0)
                        col_base = c0
                        if tiling.reverse:
                            col_base = tiling.scan - c0 - valid_cols
                        in_slot = scan_in.next()      # pass A/B 共用同一 group
                        pl.set_validshape(in_slot, [valid_rows, valid_cols])
                        pl.load(in_slot, x1, [r0, col_base])
                        init_mode = 1
                        if c0 == seg_beg:
                            init_mode = 0
                        is_final = 0
                        if c0 + W >= seg_end:
                            is_final = 1
                        agg_cols_vf(in_slot, carry_slot, valid_rows, valid_cols,
                                    init_mode, is_final, rev)
                    # 段聚合 → GM workspace seg_agg[s, r0:r0+valid_rows]
                    # （fp32，扫描坐标段序行序——段序与 reverse 无关）
                    pl.set_validshape(carry_slot, [1, valid_rows])
                    pl.store(ws, carry_slot, [s, r0])

                # ────── 跨核阶段屏障（完整协议）──────
                # set/wait 位于一切运行时可变循环之外（零 job 核也执行）；
                # seg_count/exclusive/reverse 均为 TilingData 常量、全核同值 →
                # 每核恰好 1 set + 1 wait，参与者完备（block_dim=vector_core_num，
                # host tiling 侧保证）；四组合共用同一屏障。
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3, event_id=EVT_AGGREADY,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK)
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE2, event_id=EVT_AGGREADY,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK)
                pl.system.dcci(ws)                    # 清理+失效数据缓存（防御性）

            # ────── pass B（主路径，无条件执行；单段 carry=1 无 ws） ──────
            for job in pl.range(core_id, total_jobs, num_cores):
                rb = job // tiling.seg_count
                s = job % tiling.seg_count
                r0 = rb * ROWS
                seg_beg = s * tiling.seg_len
                seg_end = pl.min(tiling.scan, seg_beg + tiling.seg_len)
                valid_rows = pl.min(ROWS, tiling.outer - r0)
                ws_slot = ws_group.current()
                if tiling.seg_count > 1:
                    # 屏障 wait 之后的首个 ws 读（MTE2 被 wait 门控到信号之后）
                    pl.set_validshape(ws_slot, [tiling.seg_count, valid_rows])
                    pl.load(ws_slot, ws, [0, r0])
                for c0 in pl.range(seg_beg, seg_end, W):
                    valid_cols = pl.min(W, seg_end - c0)
                    # reverse 方向（同 pass A 换算；输出与输入同窗口）
                    col_base = c0
                    if tiling.reverse:
                        col_base = tiling.scan - c0 - valid_cols
                    in_slot = scan_in.next()
                    pl.set_validshape(in_slot, [valid_rows, valid_cols])
                    pl.load(in_slot, x1, [r0, col_base])
                    out_slot = scan_out.next()
                    pl.set_validshape(out_slot, [valid_rows, valid_cols])
                    init_mode = 1
                    if c0 == seg_beg:
                        init_mode = 0
                        if tiling.seg_count > 1:
                            init_mode = 2             # 从 ws 折段前缀（重读 x 重算）
                    scan_cols_vf(in_slot, out_slot, carry_slot, ws_slot,
                                 valid_rows, valid_cols, init_mode, s, excl, rev)
                    pl.store(y1, out_slot, [r0, col_base])
