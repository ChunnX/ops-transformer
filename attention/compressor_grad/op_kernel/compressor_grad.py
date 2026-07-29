#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

"""
compressor_grad_kernel — Compressor 反向传播 NPU Kernel（统一入口）。

TilingKey:  Coff, Layout, DataType               — 编译期常量
TilingData: batchSize, tokenSize, seqSize,       — 运行时参数
            cmpRatio, hiddenSize, headDim

三阶段流水线（同 backward_design.md）:
  Phase 1: Vec Scatter  — 每压缩块计算 d_kv, d_score, APE 累加
  Phase 2: Cube Matmul  — d_new_kv @ wkv, d_new_score @ wgate, 产出 partial
  Phase 3: Vec Reduce   — 跨 D 组归约 + 跨轮 RMW

当前状态: Phase 1 已实现, Phase 2/3 待实现
"""

from dataclasses import dataclass
import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField as tilingkey_field

# ================================================================
#  Compile-time constants
# ================================================================
FLOAT_REG_ELEMENTS = 64   # FP32 VF register holds 64 elements

# ================================================================
#  TilingKey
# ================================================================
class CompressorGradTilingKey:
    Coff     = tilingkey_field(bits=2, values=[1, 2])
    Layout   = tilingkey_field(bits=1, values=[0, 1])
    DataType = tilingkey_field(bits=2, values=[0, 1])

    def is_valid(self, key):
        coff, layout, dtype = key
        return coff in (1, 2) and layout in (0, 1) and dtype in (0, 1)


# ================================================================
#  TilingData
# ================================================================
@dataclass
class CompressorGradTiling:
    batchSize:        int = 1
    tokenSize:        int = 0
    seqSize:          int = 0
    cmpRatio:         int = 4
    hiddenSize:       int = 512
    headDim:          int = 128
    featureDim:       int = 0
    scKvRowCount:     int = 0
    outStride:        int = 0
    groupSize:        int = 0
    dBaseSize:        int = 0
    nCorePerGroup:    int = 0
    mPerRoundPerGroup:int = 0
    mBaseSize:        int = 0
    mPerRoundAll:     int = 0
    dbRows:           int = 0
    ape_ws_elems:     int = 0
    dx_per_group:     int = 0
    dx_stride:        int = 0
    dx_ws_elems:      int = 0
    d_x_rows:         int = 0
    outputRows:       int = 0
    rgChunkNum:       int = 0
    rgPerCore:        int = 0
    rgRemain:         int = 0


# ================================================================
#  VF: 反向 scatter — d_kv = dC * softmax, d_score = softmax * (dsw - col_sum)
# ================================================================
@pl.vector_function
def _vf_scatter_backward(
    dc_f32_tile,       # dC: (n_blocks, D) FP32 (pre-cast outside VF)
    kv_tile,           # kv_2d: (n_blocks*N, D) fp32 — kv → dsw
    softmax_tile,      # softmax: (n_blocks*N, D) fp32 — sm → dsb
    temp_tile,         # temp: (n_blocks*N, D) fp32 — dkv output
    n_blocks: pl.DT_INT64,
    n_rows_per_block: pl.DT_INT64,
    dDealSize: pl.DT_INT64,
):
    dsw_tile = kv_tile
    dsb_tile = softmax_tile
    dkv_tile = temp_tile

    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)

    halves = dDealSize // FLOAT_REG_ELEMENTS
    row_stride = dDealSize
    block_stride = n_rows_per_block * row_stride

    for half in pl.range(0, halves):
        half_off = half * FLOAT_REG_ELEMENTS

        for k in pl.range(0, n_blocks):
            block_off = k * block_stride
            dc_off = k * dDealSize + half_off

            dc_reg = vf.load_align(dc_f32_tile, dc_off)

            col_sum = vf.full(0.0, dtype=pl.DT_FP32)
            for m in pl.range(0, n_rows_per_block):
                row_off = block_off + m * row_stride + half_off
                sm_reg = vf.load_align(softmax_tile, row_off)
                kv_reg = vf.load_align(kv_tile, row_off)
                dkv_reg = vf.mul(dc_reg, sm_reg, preg)
                vf.store_align(dkv_tile + row_off, dkv_reg, preg)
                dsw_reg = vf.mul(dc_reg, kv_reg, preg)
                vf.store_align(dsw_tile + row_off, dsw_reg, preg)
                prod_reg = vf.mul(sm_reg, dsw_reg, preg)
                col_sum = vf.add(col_sum, prod_reg, preg)

            vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
            for m in pl.range(0, n_rows_per_block):
                row_off = block_off + m * row_stride + half_off
                sm_reg = vf.load_align(softmax_tile, row_off)
                dsw_reg = vf.load_align(dsw_tile, row_off)
                tmp_reg = vf.sub(dsw_reg, col_sum, preg)
                dsb_reg = vf.mul(sm_reg, tmp_reg, preg)
                vf.store_align(dsb_tile + row_off, dsb_reg, preg)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


# ================================================================
#  VF: 清零 tile
# ================================================================
@pl.vector_function
def _vf_tile_zero(
    tile,
    n_rows: pl.DT_INT64,
    n_cols: pl.DT_INT64,
):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    halves = n_cols // FLOAT_REG_ELEMENTS
    row_stride = n_cols
    zero_reg = vf.full(0.0, dtype=pl.DT_FP32)
    for half in pl.range(0, halves):
        half_off = half * FLOAT_REG_ELEMENTS
        for m in pl.range(0, n_rows):
            vf.store_align(tile + m * row_stride + half_off, zero_reg, preg)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


# ================================================================
#  VF: 清零 tile (FP16)
# ================================================================
@pl.vector_function
def _vf_tile_zero_f16(
    tile,
    n_rows: pl.DT_INT64,
    n_cols: pl.DT_INT64,
):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    halves = n_cols // 128
    row_stride = n_cols
    zero_reg = vf.full(0.0, dtype=pl.DT_FP16)
    for half in pl.range(0, halves):
        half_off = half * 128
        for m in pl.range(0, n_rows):
            vf.store_align(tile + m * row_stride + half_off, zero_reg, preg)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


# ================================================================
#  VF: 清零 tile (BF16)
# ================================================================
@pl.vector_function
def _vf_tile_zero_bf16(
    tile,
    n_rows: pl.DT_INT64,
    n_cols: pl.DT_INT64,
):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    halves = n_cols // 128
    row_stride = n_cols
    zero_reg = vf.full(0.0, dtype=pl.DT_BF16)
    for half in pl.range(0, halves):
        half_off = half * 128
        for m in pl.range(0, n_rows):
            vf.store_align(tile + m * row_stride + half_off, zero_reg, preg)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


# ================================================================
#  VF: 掩码清零 — 将非有效 token 对应的 d_kv/d_score 行置零
# ================================================================
@pl.vector_function
def _vf_mask_cache_fill(
    dsb_tile,
    dkv_tile,
    cache_line_fill: pl.DT_INT64,
    dDealSize: pl.DT_INT64,
    d_stride: pl.DT_INT64,
):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    halves = dDealSize // FLOAT_REG_ELEMENTS
    row_stride = d_stride
    zero_reg = vf.full(0.0, dtype=pl.DT_FP32)
    for half in pl.range(0, halves):
        half_off = half * FLOAT_REG_ELEMENTS
        for m in pl.range(0, cache_line_fill):
            vf.store_align(dsb_tile + m * row_stride + half_off, zero_reg, preg)
            vf.store_align(dkv_tile + m * row_stride + half_off, zero_reg, preg)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


# ================================================================
#  VF: 跨 block 归约 — d_ape[m,:] += Σ_k d_score[k,m,:]（per-round 调用）
# ================================================================
@pl.vector_function
def _vf_reduce_dscore_to_ape(
    dsb_tile,
    ape_tile,
    n_blocks: pl.DT_INT64,
    n_rows_per_block: pl.DT_INT64,
    dDealSize: pl.DT_INT64,
    d_stride: pl.DT_INT64,
):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    halves = dDealSize // FLOAT_REG_ELEMENTS
    row_stride     = d_stride
    block_stride   = n_rows_per_block * row_stride
    ape_row_stride = d_stride
    for half in pl.range(0, halves):
        half_off = half * FLOAT_REG_ELEMENTS
        for m in pl.range(0, n_rows_per_block):
            ape_off = m * ape_row_stride + half_off
            ape_reg = vf.full(0.0, dtype=pl.DT_FP32)
            for k in pl.range(0, n_blocks):
                block_off = k * block_stride
                row_off = block_off + m * row_stride + half_off
                dsb_reg = vf.load_align(dsb_tile + row_off, 0)
                ape_reg = vf.add(ape_reg, dsb_reg, preg)
            vf.store_align(ape_tile + ape_off, ape_reg, preg)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


# ================================================================
#  Helper: get start position / seq used / seq length
# ================================================================
def _get_start_pos(bIdx, start_pos, ctx):
    result = 0
    if start_pos is not None:
        result = pl.getval(pl.make_tensor(start_pos, [ctx.batchSize], [1], dtype=pl.DT_INT32), bIdx)
    return result


def _get_seq_length(bIdx, cu_seqlens, ctx):
    result = ctx.seqSize
    if ctx.layout == 0:  # TH
        result = pl.getval(pl.make_tensor(cu_seqlens, [ctx.batchSize + 1], [1], dtype=pl.DT_INT32), bIdx + 1) - \
                 pl.getval(pl.make_tensor(cu_seqlens, [ctx.batchSize + 1], [1], dtype=pl.DT_INT32), bIdx)
    return result


def _get_seq_used(bIdx, seqused, cu_seqlens, ctx):
    if seqused is not None:
        result = pl.getval(pl.make_tensor(seqused, [ctx.batchSize], [1], dtype=pl.DT_INT32), bIdx)
    else:
        result = _get_seq_length(bIdx, cu_seqlens, ctx)
    return result


# ================================================================
#  Helper: 将 (b, s) 映射为全局 token 索引 tIdx，BSH/TH 统一
# ================================================================
def _get_token_idx(bIdx, sIdx, cu_seqlens, ctx):
    result = bIdx * ctx.seqSize + sIdx
    if ctx.layout == 0:
        result = pl.getval(pl.make_tensor(cu_seqlens, [ctx.batchSize + 1], [1], dtype=pl.DT_INT32), bIdx) + sIdx
    return result


# ================================================================
#  Helper: 计算 (bStart,sStart) 到 (bEnd,sEnd) 之间的行数，sEnd 左闭右开
# ================================================================
def _get_row_count(bStart, sStart, bEnd, sEnd, cu_seqlens, ctx):
    result = _get_token_idx(bEnd, sEnd, cu_seqlens, ctx) - _get_token_idx(bStart, sStart, cu_seqlens, ctx)
    return result


# ================================================================
#  Helper: 计算位置 (bIdx, sIdx) 之前的压缩块总数
# ================================================================
def _get_cmp_block_count(bIdx, sIdx, start_pos, seqused, cu_seqlens, ctx):
    result = 0
    for bb in pl.range(0, bIdx):
        sp = _get_start_pos(bb, start_pos, ctx)
        sq = _get_seq_used(bb, seqused, cu_seqlens, ctx)
        cmpEnd = (sp + sq) // ctx.cmpRatio
        if cmpEnd > sp // ctx.cmpRatio:
            result += cmpEnd - (sp // ctx.cmpRatio)
    sp = _get_start_pos(bIdx, start_pos, ctx)
    result += (sIdx + sp) // ctx.cmpRatio - sp // ctx.cmpRatio
    return result


# ================================================================
#  Helper: 将全局 token 索引 tIdx 反向映射为 (bIdx, sIdx)
# ================================================================
def _get_pos_from_token_idx(tIdx, cu_seqlens, ctx):
    resultB = tIdx // ctx.seqSize
    resultS = tIdx % ctx.seqSize
    if ctx.layout == 0:
        resultB = 0
        while resultB < ctx.batchSize:
            if tIdx < pl.getval(pl.make_tensor(cu_seqlens, [ctx.batchSize + 1], [1], dtype=pl.DT_INT32), resultB + 1):
                resultS = tIdx - pl.getval(pl.make_tensor(cu_seqlens, [ctx.batchSize + 1], [1], dtype=pl.DT_INT32), resultB)
                break
            resultB += 1
    return resultB, resultS


# ================================================================
#  Helper: advance block traversal (used by Segment 1 and 3)
# ================================================================
def _skip_blocks(blockCount, bIdx, scIdx,
                 start_pos, seqused, cu_seqlens, ctx):
    skipped = 0
    while skipped < blockCount:
        spVal = _get_start_pos(bIdx, start_pos, ctx)
        sqVal = _get_seq_used(bIdx, seqused, cu_seqlens, ctx)
        cmpLimit = (spVal + sqVal) // ctx.cmpRatio * ctx.cmpRatio
        blkInBatch = (cmpLimit - spVal) // ctx.cmpRatio
        remaining = blkInBatch - scIdx
        if remaining > 0:
            taken = min(blockCount - skipped, remaining)
            skipped += taken
            scIdx += taken
        if scIdx >= blkInBatch:
            scIdx = 0
            bIdx += 1
        if bIdx >= ctx.batchSize:
            break
    return bIdx, scIdx


# ================================================================
#  Unified Compressor Grad Kernel
# ================================================================
@pl.jit(tiling_key=CompressorGradTilingKey, auto_mutex=True, arch="a5")
def compressor_grad(
    # ── 输入（前向参数） ──
    x:              pl.Ptr[pl.DT_UINT8],     # (T, hiddenSize)      
    wkv:            pl.Ptr[pl.DT_UINT8],     # (C, hiddenSize)       
    wgate:          pl.Ptr[pl.DT_UINT8],     # (C, hiddenSize)       

    # ── 输入（前向中间变量 / 上游梯度） ──
    d_cmp_kv:       pl.Ptr[pl.DT_UINT8],     # (outputRows, headDim)           
    kv:             pl.Ptr[pl.DT_UINT8],     # (outputRows, scKvRowCount, headDim) FP32
    softmax_score:  pl.Ptr[pl.DT_UINT8],     # (outputRows, scKvRowCount, headDim) FP32

    # ── 位置 / 序列参数 ──
    cu_seqlens:     pl.Ptr[pl.DT_UINT8],
    seqused:        pl.Ptr[pl.DT_UINT8],
    start_pos:      pl.Ptr[pl.DT_UINT8],

    # ── 输出（最终梯度） ──
    d_x:            pl.Ptr[pl.DT_UINT8],     # (T, hiddenSize) FP32 partials → Phase 3 reduce
    d_wkv:          pl.Ptr[pl.DT_UINT8],     # (C, hiddenSize) FP32
    d_wgate:        pl.Ptr[pl.DT_UINT8],     # (C, hiddenSize) FP32
    d_ape:          pl.Ptr[pl.DT_UINT8],     # APE workspace / output

    # ── 工作空间 ──
    workspace:      pl.Ptr[pl.DT_UINT8],     # Phase 2 partials

    # ── Tiling ──
    tiling:         CompressorGradTiling,
):
    batchSize  = tiling.batchSize
    tokenSize  = tiling.tokenSize
    seqSize    = tiling.seqSize
    cmpRatio   = tiling.cmpRatio
    headDim    = tiling.headDim
    featureDim = Coff * headDim
    scKvRowCount = Coff * cmpRatio
    outStride    = (seqSize + cmpRatio - 1) // cmpRatio
    ioDType      = pl.DT_BF16 if DataType == 0 else pl.DT_FP16

    # ── Sequence context ──
    seq_ctx = pl.struct("SeqCtx",
        batchSize=batchSize,
        cmpRatio=cmpRatio,
        layout=Layout,
        seqSize=seqSize,
    )

    # ── Compute total compression blocks ──
    totalBlocks = 0
    for bi in pl.range(0, batchSize):
        sp = _get_start_pos(bi, start_pos, seq_ctx)
        sq = _get_seq_used(bi, seqused, cu_seqlens, seq_ctx)
        cmpLimit = (sp + sq) // cmpRatio * cmpRatio
        if cmpLimit > sp:
            totalBlocks = totalBlocks + (cmpLimit - sp) // cmpRatio

    if Layout == 0:
        outputRows = min(tokenSize, (tokenSize // cmpRatio) + batchSize)
    else:
        outputRows = batchSize * ((seqSize + cmpRatio - 1) // cmpRatio)


    # ── Tensor views ──
    tensorDCmpKv = pl.make_tensor(d_cmp_kv, [outputRows, headDim],
                                   [headDim, 1], dtype=ioDType)
    tensorKv      = pl.make_tensor(kv, [outputRows, scKvRowCount, headDim],
                                   [scKvRowCount * headDim, headDim, 1], dtype=pl.DT_FP32)
    tensorSoftmax = pl.make_tensor(softmax_score, [outputRows, scKvRowCount, headDim],
                                   [scKvRowCount * headDim, headDim, 1], dtype=pl.DT_FP32)

    hiddenSize = tiling.hiddenSize

    # ── 共享 round 参数（Phase 1 + Phase 2 都需要）──
    cubeCoreNum = pl.get_block_num()
    groupSize         = headDim // 128
    dBaseSize         = 128 // Coff
    groupNum     = cubeCoreNum // groupSize
    mPerRoundPerGroup = 128 * 2 // Coff // cmpRatio
    mBaseSize         = 128 // cmpRatio
    mPerRoundAll      = groupNum * mPerRoundPerGroup
    roundCnt          = (totalBlocks + mPerRoundAll - 1) // mPerRoundAll
    dbRows            = mPerRoundAll * cmpRatio

    # ── L1A CV 通道: 4 个独立 tile_group [128,128], double-buffer ──
    l1_nz_type = pl.TileType(shape=[128 * Coff, 128 // Coff], dtype=ioDType,
                              target_memory=pl.MemorySpace.Mat, layout=pl.NZ, compact=1)
    l1_zn_type = pl.TileType(shape=[128 // Coff, 128 * Coff], dtype=ioDType,
                              target_memory=pl.MemorySpace.Mat, layout=pl.ZN, compact=1)

    l1_kv0_db    = pl.make_tile_group(type=l1_nz_type, addrs=[0x00000, 0x10000], mutex_ids=[10, 11])
    l1_kv1_db    = pl.make_tile_group(type=l1_nz_type, addrs=[0x08000, 0x18000], mutex_ids=[12, 13])
    l1_sb0_db    = pl.make_tile_group(type=l1_nz_type, addrs=[0x20000, 0x30000], mutex_ids=[14, 15])
    l1_sb1_db    = pl.make_tile_group(type=l1_nz_type, addrs=[0x28000, 0x38000], mutex_ids=[16, 17])

    l1_kv0_db_t  = pl.make_tile_group(type=l1_zn_type, addrs=[0x00000, 0x10000], mutex_ids=[10, 11])
    l1_kv1_db_t  = pl.make_tile_group(type=l1_zn_type, addrs=[0x08000, 0x18000], mutex_ids=[12, 13])
    l1_sb0_db_t  = pl.make_tile_group(type=l1_zn_type, addrs=[0x20000, 0x30000], mutex_ids=[14, 15])
    l1_sb1_db_t  = pl.make_tile_group(type=l1_zn_type, addrs=[0x28000, 0x38000], mutex_ids=[16, 17])

    # ── Workspace 分块: APE 区域 + d_x 区域──
    ape_ws_elems   = 2 * dbRows * headDim
    dx_per_group   = 256 * hiddenSize
    dx_stride      = cubeCoreNum * dx_per_group
    dx_ws_elems    = 2 * dx_stride
    dw_ws_elems    = 2 * groupNum * featureDim * hiddenSize
    x_ws_elems    = 2 * groupNum * (256 // Coff) * hiddenSize

    ws_f32 = pl.make_ptr(workspace, dtype=pl.DT_FP32)
    dx_ptr = pl.addptr(ws_f32, ape_ws_elems)       # dkv@wkv + dsb@wgate partials
    dwkv_ptr = pl.addptr(dx_ptr, dx_ws_elems)   
    dwgate_ptr = pl.addptr(dwkv_ptr, dw_ws_elems)   
    x_ptr = pl.addptr(dwgate_ptr, dw_ws_elems)   

    # ── d_x output + workspace views (Phase 3) ──
    if Layout == 0:
        d_x_rows = tokenSize
    else:
        d_x_rows = batchSize * seqSize
    d_x_out = pl.make_tensor(d_x, [d_x_rows, hiddenSize],
                              [hiddenSize, 1], dtype=ioDType)
    d_wkv_out = pl.make_tensor(d_wkv, [featureDim, hiddenSize],
                              [hiddenSize, 1], dtype=ioDType)
    d_wgate_out = pl.make_tensor(d_wgate, [featureDim, hiddenSize],
                              [hiddenSize, 1], dtype=ioDType)
    d_x_ws = pl.make_tensor(dx_ptr,
        [dx_ws_elems // hiddenSize, hiddenSize], [hiddenSize, 1], dtype=pl.DT_FP32)
    d_wkv_ws = pl.make_tensor(dwkv_ptr,
        [dw_ws_elems // hiddenSize, hiddenSize], [hiddenSize, 1], dtype=pl.DT_FP32)
    d_wgate_ws = pl.make_tensor(dwgate_ptr,
        [dw_ws_elems // hiddenSize, hiddenSize], [hiddenSize, 1], dtype=pl.DT_FP32)
    x_t = pl.make_tensor(x, [tokenSize, hiddenSize],
                            [hiddenSize, 1], dtype=ioDType)
    x_ws = pl.make_tensor(x_ptr,
        [x_ws_elems // hiddenSize, hiddenSize], [hiddenSize, 1], dtype=ioDType)

    # ── Per-round shared state (persists across rounds) ──
    bIdx              = 0
    scIdx             = 0
    p3bIdx            = 0
    p3scIdx           = 0
    mxbIdx            = 0
    mxscIdx           = 0

    # ════════════════════════════════════════════════════════════
    #  Vec tile declarations (once)
    # ════════════════════════════════════════════════════════════
    with pl.section_vector():
        coreNum   = cubeCoreNum * pl.get_subblock_num()
        coreIdx   = pl.get_block_idx()
        # pl.printf("coreNum:%d, coreIdx:%d\n", coreNum, coreIdx)

        # ── UB tile type declarations ──
        tile_type_dc_f16 = pl.TileType(
            shape=[64, 128], dtype=ioDType,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])

        tile_type_cast_out = pl.TileType(
            shape=[128 * Coff, 128 // Coff], dtype=ioDType,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])

        tile_type_cast_nz = pl.TileType(
            shape=[128 * Coff, 128 // Coff], dtype=ioDType,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1],
            layout=pl.NZ, compact=1)

        tile_type_work = pl.TileType(
            shape=[128 * Coff, 128 // Coff], dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1],
            pad=pl.TilePad.zero)
        
        
        tile_type_work3 = pl.TileType(
            shape=[1, 128 * 128], dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1],
            pad=pl.TilePad.zero)

        tile_type_move_x = pl.TileType(
            shape=[128, 256], dtype=ioDType,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1],
            pad=pl.TilePad.zero)

        dc_type_f32 = pl.TileType(shape=[64, 128 // Coff], dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1,-1], pad=pl.TilePad.zero)

        # ── UB tile_group allocation (auto_mutex for shared addresses) ──
        #          | group             | addr      | size   | mutex |
        #          | dc_f16            | 0x00000   | 16384  | [0]   |
        #          | dc_f32            | 0x04000   | 32768  | [1]   |
        #          | kv / dsw          | 0x0C000   | 65536  | [2]   |
        #          | softmax          | 0x1C000   | 65536  | [3]   |
        #          | dkv (temp)       | 0x2C000   | 65536  | [4]   |
        #          | dsb_cast         | 0x1C000   | 32768  | [3]   |
        #          | dsb_nz           | 0x24000   | 32768  | [5]   |
        #          | dkv_cast         | 0x2C000   | 32768  | [4]   |
        #          | dkv_nz           | 0x34000   | 32768  | [6]   |
        # Total: 16K + 32K + 64K + 64K + 64K = 240KB < 248KB
        dc_group      = pl.make_tile_group(type=tile_type_dc_f16, addrs=[0x00000], mutex_ids=[0])
        dc_f32_group  = pl.make_tile_group(type=dc_type_f32,     addrs=[0x04000], mutex_ids=[1])
        kv_group      = pl.make_tile_group(type=tile_type_work,  addrs=[0x0C000], mutex_ids=[2])
        sm_group      = pl.make_tile_group(type=tile_type_work,  addrs=[0x1C000], mutex_ids=[3])
        dkv_group     = pl.make_tile_group(type=tile_type_work,  addrs=[0x2C000], mutex_ids=[4])
        dsb_cast_group= pl.make_tile_group(type=tile_type_cast_out, addrs=[0x1C000], mutex_ids=[3])
        dkv_cast_group= pl.make_tile_group(type=tile_type_cast_out, addrs=[0x2C000], mutex_ids=[4])
        ape_local_group = pl.make_tile_group(type=tile_type_work, addrs=[0x0C000], mutex_ids=[2])
        dsb_nz_group = pl.make_tile_group(type=tile_type_cast_nz, addrs=[0x24000], mutex_ids=[3])
        dkv_nz_group = pl.make_tile_group(type=tile_type_cast_nz, addrs=[0x34000], mutex_ids=[4])
        movex_group = pl.make_tile_group(type=tile_type_move_x,  addrs=[0x1C000, 0x2C000], mutex_ids=[3, 4])

        # Phase 3 reduce: 复用 0x0C000/0x1C000/0x2C000 三个 64KB 区域
        reduce_acc_group = pl.make_tile_group(type=tile_type_work3, addrs=[0x0C000], mutex_ids=[2])
        reduce_ld_group = pl.make_tile_group(type=tile_type_work3, addrs=[0x1C000, 0x2C000], mutex_ids=[3, 4])

        # Phase 3 d_x reduce: acc/ld 复用 reduce_acc/reduce_ld0，仅新增 cast
        p3_tile_type_cast = pl.TileType(
            shape=[1, 128 * 128], dtype=ioDType,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
        p3Cast_group = pl.make_tile_group(type=p3_tile_type_cast, addrs=[0x2C000], mutex_ids=[4])

        tile_dc_f16   = dc_group.next()
        tile_dc_f32   = dc_f32_group.next()
        tile_kv       = kv_group.next()
        tile_softmax  = sm_group.next()
        tile_dsb_cast = dsb_cast_group.next()
        tile_temp     = dkv_group.next()
        tile_dkv_cast = dkv_cast_group.next()
        tile_ape_local = ape_local_group.next()
        tile_dkv_nz  = dkv_nz_group.next()
        tile_dsb_nz  = dsb_nz_group.next()

        reduce_acc = reduce_acc_group.next()

        p3_cast = p3Cast_group.next()

        # ── D group division ──
        cubeCoreIdx      = coreIdx // 2
        groupIdx         = cubeCoreIdx // groupSize
        intraGroupIdx    = cubeCoreIdx % groupSize

        nStart = intraGroupIdx * 128 + (coreIdx % Coff) * dBaseSize
        nEnd   = nStart + dBaseSize

        tensorApe = pl.make_tensor(ws_f32,
            [2 * dbRows, headDim],
            [headDim, 1], dtype=pl.DT_FP32)


        tensorDApeWsFlat = pl.make_tensor(ws_f32,
            [2 * mPerRoundAll // Coff, Coff * cmpRatio * headDim],
            [Coff * cmpRatio * headDim, 1], dtype=pl.DT_FP32)

        tensorDWKvWsFlat = pl.make_tensor(dwkv_ptr,
            [dw_ws_elems // (featureDim * hiddenSize), featureDim * hiddenSize],
            [featureDim * hiddenSize, 1], dtype=pl.DT_FP32)
            
        tensorDWGateWsFlat = pl.make_tensor(dwgate_ptr,
            [dw_ws_elems // (featureDim * hiddenSize), featureDim * hiddenSize],
            [featureDim * hiddenSize, 1], dtype=pl.DT_FP32)

        tensorDApeFlat = pl.make_tensor(d_ape, [1, Coff * cmpRatio * headDim],
                                         [Coff * cmpRatio * headDim, 1], dtype=pl.DT_FP32)
        tensorDWKvFlat = pl.make_tensor(d_wkv, [1, featureDim * hiddenSize],
                                         [featureDim * hiddenSize, 1], dtype=pl.DT_FP32)
        tensorDWGateFlat = pl.make_tensor(d_wgate, [1, featureDim * hiddenSize],
                                         [featureDim * hiddenSize, 1], dtype=pl.DT_FP32)

    # ═════════════════════════════════════════════════════════════════════════
    #  Cube tile declarations (once)
    # ═════════════════════════════════════════════════════════════════════════
    with pl.section_cube():
        cubeCoreIdx     = pl.get_block_idx()
        # pl.printf("coreNum:%d, cubeCoreIdx:%d\n", pl.get_block_num(), cubeCoreIdx)
        cube_group_idx = cubeCoreIdx // groupNum
        groupIdx         = cubeCoreIdx // groupSize
        intraGroupIdx    = cubeCoreIdx % groupSize

        nStart = intraGroupIdx * 128
        nEnd   = nStart + dBaseSize
        

        # L1A CV 数据: 复用主函数体 tile_group, 通过 .next() db

        # L1W: wkv/wgate chunk 加载（double-buffer, 0x40000-0x4FFFF, 64KB）
        l1_w_db = pl.make_tile_group(
            type=pl.TileType(shape=[128, 256], dtype=ioDType,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ, compact=1),
            addrs=[0x40000, 0x48000], mutex_ids=[20, 21])

        # L0 tiles
        cv_l0a = pl.make_tile_group(
            type=pl.TileType(shape=[128 * Coff, 128 // Coff], dtype=ioDType,
                             target_memory=pl.MemorySpace.Left, layout=pl.NZ, compact=1),
            addrs=[0x00000, 0x08000], mutex_ids=[18, 19])
        cv_l0b = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=ioDType,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN, compact=1),
            addrs=[0x00000, 0x08000], mutex_ids=[22, 23])
        cv_l0c = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024, compact=1),
            addrs=0x00000, mutex_ids=[24, 25, 26, 27])

        # GM tensor views
        wkv_t = pl.make_tensor(wkv, [featureDim, hiddenSize],
                               [hiddenSize, 1], dtype=ioDType)
        wgate_t = pl.make_tensor(wgate, [featureDim, hiddenSize],
                                 [hiddenSize, 1], dtype=ioDType)

        # d_x partial workspace (dkv@wkv) + dsb@wgate workspace


    # ════════════════════════════════════════════════════════════
    #  Phase 1+3: Vec Scatter & Reduce (per-round compute)
    # ════════════════════════════════════════════════════════════
    for roundIdx in pl.range(0, roundCnt):
        roundBlocks = min(mPerRoundAll,
                          totalBlocks - roundIdx * mPerRoundAll)
        slot_id = roundIdx % 2
        
        with pl.section_vector():
            preDealTcSize = groupIdx * mPerRoundPerGroup
            if Coff == 1 and coreIdx % 2 == 1:
                preDealTcSize += mBaseSize
            dealTcSize   = min(mBaseSize,
                               max(0, roundBlocks - preDealTcSize))
            blocksAfter  = max(0, roundBlocks - preDealTcSize - dealTcSize)

            if preDealTcSize < roundBlocks:
                # Segment 1: skip
                bIdx, scIdx = _skip_blocks(preDealTcSize, bIdx, scIdx,
                                           start_pos, seqused, cu_seqlens, seq_ctx)

                # Segment 2: process
                processed = 0
                while processed < dealTcSize:
                    spVal = _get_start_pos(bIdx, start_pos, seq_ctx)
                    sqVal = _get_seq_used(bIdx, seqused, cu_seqlens, seq_ctx)
                    cmpLimit = (spVal + sqVal) // cmpRatio * cmpRatio
                    blkInBatch = (cmpLimit - spVal) // cmpRatio
                    remaining = blkInBatch - scIdx
                    if remaining > 0:
                        taken = min(dealTcSize - processed, remaining)
                        if Layout == 0:
                            globalBlkIdx = roundIdx * mPerRoundAll + preDealTcSize + processed
                        else:
                            globalBlkIdx = bIdx * outStride + scIdx

                        nTileRows = taken * scKvRowCount
                        dDealSize = dBaseSize

                        dcTile  = tile_dc_f16[processed:, :]
                        dcF32   = tile_dc_f32[processed:, :]
                        kvTile  = tile_kv[processed * scKvRowCount:, :]
                        smTile  = tile_softmax[processed * scKvRowCount:, :]
                        dkvTile = tile_temp[processed * scKvRowCount:, :]

                        pl.set_validshape(dcTile,  [taken, dDealSize])
                        pl.set_validshape(kvTile,  [nTileRows, dDealSize])
                        pl.set_validshape(smTile,  [nTileRows, dDealSize])

                        pl.load(dcTile, tensorDCmpKv, [globalBlkIdx, nStart])
                        pl.load(kvTile, tensorKv,      [globalBlkIdx, 0, nStart])
                        pl.load(smTile, tensorSoftmax, [globalBlkIdx, 0, nStart])

                        pl.set_validshape(dcF32, [taken, dDealSize])
                        pl.cast(dcF32, dcTile,
                                mode=pl.RoundMode.CAST_ROUND)

                        _vf_scatter_backward(
                            dcF32, kvTile, smTile, dkvTile,
                            taken, scKvRowCount, dDealSize,
                        )

                        if Coff == 1:
                            cache_fill_rows = spVal % cmpRatio if scIdx == 0 else 0
                            _vf_mask_cache_fill(
                                smTile,
                                dkvTile,
                                cache_fill_rows, dDealSize, dDealSize * Coff,
                            )
                        else:
                            cache_fill_rows = 0
                            if scIdx == 0:
                                cache_fill_rows += spVal % cmpRatio + cmpRatio if taken >= 2 else cmpRatio
                            elif scIdx == 1:
                                cache_fill_rows += spVal % cmpRatio
                            _vf_mask_cache_fill(
                                smTile,
                                dkvTile,
                                cache_fill_rows, dDealSize, dDealSize * Coff,
                            )
                            cache_fill_rows = spVal % cmpRatio if scIdx == 0 else 0
                            _vf_mask_cache_fill(
                                smTile[1:,:],
                                dkvTile[1:,:],
                                cache_fill_rows, dDealSize, dDealSize * Coff,
                            )

                        processed += taken
                        scIdx += taken
                    if scIdx >= blkInBatch:
                        scIdx = 0
                        bIdx += 1
                    if bIdx >= batchSize:
                        break

                nTileRows = dealTcSize * scKvRowCount

                # APE: local_ape 累加 + 写 workspace
                pl.set_validshape(tile_ape_local, [Coff * cmpRatio, dBaseSize])
                _vf_reduce_dscore_to_ape(
                    tile_softmax, tile_ape_local,
                    dealTcSize, Coff * cmpRatio, dBaseSize, dBaseSize,
                )

                # ── Cast dkv/dsb FP32→BF16/FP16, move ND→NZ for CV ──
                # pl.dump_data(tile_temp, offsets=[0, 0], shapes=[1, 128])
                pl.set_validshape(tile_dkv_cast, [nTileRows, dBaseSize])
                pl.set_validshape(tile_temp,     [nTileRows, dBaseSize])
                pl.cast(tile_dkv_cast, tile_temp, mode=pl.RoundMode.CAST_ROUND)
                pl.set_validshape(tile_dkv_nz, [nTileRows, dBaseSize])
                pl.set_validshape(tile_dkv_cast, [nTileRows, dBaseSize])
                pl.move(tile_dkv_nz, tile_dkv_cast)

                # pl.dump_data(tile_softmax, offsets=[0, 0], shapes=[1, 128])
                pl.set_validshape(tile_dsb_cast, [nTileRows, dBaseSize])
                pl.set_validshape(tile_softmax,  [nTileRows, dBaseSize])
                pl.cast(tile_dsb_cast, tile_softmax, mode=pl.RoundMode.CAST_ROUND)
                # pl.printf("nTileRows:%d\n", nTileRows)
                pl.set_validshape(tile_dsb_nz, [nTileRows, dBaseSize])
                pl.set_validshape(tile_dsb_cast, [nTileRows, dBaseSize])
                # pl.dump_data(tile_dsb_cast, offsets=[0, 0],shapes=[1, 128])
                # pl.dump_data(tile_dsb_cast, offsets=[15, 0],shapes=[1, 128])
                # pl.dump_data(tile_dsb_cast, offsets=[16, 0],shapes=[1, 128])
                pl.move(tile_dsb_nz, tile_dsb_cast)
                # pl.dump_data(tile_dsb_nz, offsets=[0, 0],shapes=[1, 128])
                # pl.dump_data(tile_dsb_nz, offsets=[15, 0],shapes=[1, 128])
                # pl.dump_data(tile_dsb_nz, offsets=[16, 0],shapes=[1, 128])

                dbBase = slot_id * dbRows
                localSlot = groupIdx * 2 * cmpRatio
                if Coff == 1 and coreIdx % 2 == 1:
                    localSlot += cmpRatio

                pl.store(tensorApe, tile_ape_local, [dbBase + localSlot, nStart])

                # ── CV 通路: Vec UB(NZ) → L1 Mat(NZ) (coff=1 only) ──
                sub_idx = coreIdx % 2
                l1_kv = l1_kv0_db.next() if sub_idx == 0 else l1_kv1_db.next()
                l1_sb = l1_sb0_db.next() if sub_idx == 0 else l1_sb1_db.next()
                pl.set_validshape(l1_kv, [nTileRows, dBaseSize])
                pl.set_validshape(l1_sb, [nTileRows, dBaseSize])
                pl.insert(l1_kv, tile_dkv_nz, [0, 0])
                pl.insert(l1_sb, tile_dsb_nz, [0, 0])

                # 搬运x
                mxbIdx, mxscIdx = _skip_blocks(preDealTcSize, mxbIdx, mxscIdx,
                                            start_pos, seqused, cu_seqlens, seq_ctx)

                processed = 0
                while processed < dealTcSize:   
                    spVal = _get_start_pos(mxbIdx, start_pos, seq_ctx)
                    sqVal = _get_seq_used(mxbIdx, seqused, cu_seqlens, seq_ctx)
                    cmpLimit = (spVal + sqVal) // cmpRatio * cmpRatio
                    blkInBatch = (cmpLimit - spVal) // cmpRatio
                    remaining = blkInBatch - mxscIdx
                    if remaining > 0:
                        taken = min(dealTcSize - processed, remaining)
                        dstIdx = slot_id * groupNum * (256 // Coff) + (preDealTcSize + processed) * cmpRatio
                        mxsIdx = (0 if mxscIdx == 0
                                                else mxscIdx * cmpRatio - (spVal % cmpRatio))
                        srcIdx = _get_token_idx(mxbIdx, mxsIdx, cu_seqlens, seq_ctx)
                        mxRows = taken * cmpRatio
                        if mxsIdx == 0:
                            mxRows -= spVal % cmpRatio
                            dstIdx += spVal % cmpRatio
                        # pl.printf("srcIdx:%d, dstIdx:%d, mxRows:%d", srcIdx, dstIdx, mxRows)
                        for hi in pl.range(0, hiddenSize, 256):
                            mxTie = movex_group.next()
                            pl.set_validshape(mxTie,  [mxRows, 256])
                            pl.load(mxTie, x_t, [srcIdx, hi])
                            pl.store(x_ws, mxTie, [dstIdx, hi])

                        processed += taken
                        mxscIdx += taken
                    if mxscIdx >= blkInBatch:
                        mxscIdx = 0
                        mxbIdx += 1
                    if mxbIdx >= batchSize:
                        break

                mxbIdx, mxscIdx = _skip_blocks(blocksAfter, mxbIdx, mxscIdx,
                                        start_pos, seqused, cu_seqlens, seq_ctx)

            # ── CV 通路: 所有 Vec 每轮必发信号（避免死锁）──
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=slot_id,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)
            # pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=slot_id,
            #     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)


            
            # ── Phase 3: d_ape cross-core reduce + RMW ──
            # d_ape: [cmpRatio, Coff*headDim]. Process per row-group of Coff rows.
            flatRowBase = slot_id * (mPerRoundAll // Coff)
            rgChunkNum = (Coff * cmpRatio * headDim) // 128
            rgPerCore = rgChunkNum // coreNum 
            rgRemain = rgChunkNum % coreNum
            rgStart = coreIdx * rgPerCore + (coreIdx if coreIdx < rgRemain else rgRemain)
            rgEnd = rgStart + rgPerCore + (1 if coreIdx < rgRemain else 0)
            # pl.printf("[Mo]Coff:%d, cmpRatio:%d, headDim:%d, coreNum:%d, coreIdx:%d\n", Coff, cmpRatio, headDim, coreNum,coreIdx)
            # pl.printf("[Mo]rgChunkNum:%d, rgPerCore:%d, rgRemain:%d\n", rgChunkNum, rgPerCore, rgRemain)
            rgStart = rgStart * 128
            rgEnd = rgEnd * 128
            validRows = (roundBlocks + mPerRoundPerGroup - 1) // mPerRoundPerGroup * (2 // Coff)
            pl.set_validshape(reduce_acc, [1, rgEnd - rgStart])
            # pl.printf("[Mo]rgStart:%d, rgEnd:%d, rgEnd - rgStart:%d\n", rgStart, rgEnd, rgEnd - rgStart)
            if roundIdx == 0:
                _vf_tile_zero(reduce_acc, 1, (rgEnd - rgStart))
            else:
                pl.load(reduce_acc, tensorDApeFlat, [0, rgStart])
            for r in pl.range(0, validRows):
                reduce_ld = reduce_ld_group.next()
                pl.set_validshape(reduce_ld, [1, rgEnd - rgStart]) 
                pl.load(reduce_ld, tensorDApeWsFlat, [flatRowBase + r, rgStart])
                pl.add(reduce_acc, reduce_acc, reduce_ld)
            pl.store(tensorDApeFlat, reduce_acc, [0, rgStart])

            # ── Phase 3: d_x reduce ──
            p3bIdxEnd, p3scIdxEnd = _skip_blocks(roundBlocks, p3bIdx, p3scIdx,
                                           start_pos, seqused, cu_seqlens, seq_ctx)

            bStart = p3bIdx
            bEnd   = p3bIdxEnd - 1 if p3scIdxEnd == 0 else p3bIdxEnd
            sStart = (0 if p3scIdx == 0
                      else p3scIdx * cmpRatio - (_get_start_pos(bStart, start_pos, seq_ctx) % cmpRatio))
            sEnd   = (_get_seq_length(bEnd, cu_seqlens, seq_ctx) if p3scIdxEnd == 0
                      else p3scIdxEnd * cmpRatio - (_get_start_pos(bEnd, start_pos, seq_ctx) % cmpRatio))

            totalRows = _get_row_count(bStart, sStart, bEnd, sEnd, cu_seqlens, seq_ctx)

            rowsPerCore = totalRows // coreNum
            rowsRemain  = totalRows % coreNum
            myRowS = coreIdx * rowsPerCore + (coreIdx if coreIdx < rowsRemain else rowsRemain)
            myRowE = myRowS + rowsPerCore + (1 if coreIdx < rowsRemain else 0)

            if myRowS < myRowE:
                # ── 从 (bStart, sStart) 偏移 myRowS 行到达本核起始位置 ──
                startTidx = _get_token_idx(bStart, sStart, cu_seqlens, seq_ctx)
                b, localS = _get_pos_from_token_idx(startTidx + myRowS, cu_seqlens, seq_ctx)
                tIdx = startTidx + myRowS
                sp = _get_start_pos(b, start_pos, seq_ctx)
                sq = _get_seq_used(b, seqused, cu_seqlens, seq_ctx)
                cmpLimit = (sp + sq) // cmpRatio * cmpRatio - sp
                scIdx = _get_cmp_block_count(b, localS, start_pos, seqused, cu_seqlens, seq_ctx) - \
                             _get_cmp_block_count(p3bIdx, sStart, start_pos, seqused, cu_seqlens, seq_ctx)
                scInnerIdx = (localS + sp) % cmpRatio

                for rowIdx in pl.range(myRowS, myRowE):
                    # ── 判断是否参与压缩 ──
                    inCompress = localS < cmpLimit

                    if inCompress:
                        ws_offset = slot_id * cubeCoreNum * 256 + (scIdx // mPerRoundPerGroup) * groupSize * (256 // Coff) + (scIdx % mPerRoundPerGroup) * cmpRatio + scInnerIdx
                        pl.set_validshape(reduce_acc, [1, hiddenSize])
                        pl.load(reduce_acc, d_x_ws, [ws_offset, 0])
                        if groupSize > 1:
                            for r in pl.range(1, groupSize):
                                reduce_ld = reduce_ld_group.next()
                                pl.set_validshape(reduce_ld, [1, hiddenSize]) 
                                pl.load(reduce_ld, d_x_ws, [ws_offset + r * (256 // Coff), 0])
                                pl.add(reduce_acc, reduce_acc, reduce_ld)

                        pl.set_validshape(p3_cast, [1, hiddenSize])
                        pl.cast(p3_cast, reduce_acc, mode=pl.RoundMode.CAST_ROUND)
                        pl.store(d_x_out, p3_cast, [tIdx, 0])

                    else:
                        # ── 不参与压缩: p3_cast 直接刷零，store ──
                        pl.set_validshape(p3_cast, [1, hiddenSize])
                        if DataType == 0:
                            _vf_tile_zero_bf16(p3_cast, 1, hiddenSize)
                        else:
                            _vf_tile_zero_f16(p3_cast, 1, hiddenSize)
                        pl.store(d_x_out, p3_cast, [tIdx, 0])

                    # ── 推进到下一行 ──
                    tIdx += 1
                    localS += 1
                    seqLen = _get_seq_length(b, cu_seqlens, seq_ctx)
                    if localS >= seqLen:
                        localS = 0
                        b += 1
                        if b >= batchSize:
                            break
                        sp = _get_start_pos(b, start_pos, seq_ctx)
                        sq = _get_seq_used(b, seqused, cu_seqlens, seq_ctx)
                        cmpLimit = (sp + sq) // cmpRatio * cmpRatio - sp
                    scInnerIdx = (localS + sp) % cmpRatio
                    if (localS < cmpLimit) and (localS == 0 or scInnerIdx == 0):
                        scIdx += 1

            # pl.printf("@!#\n")
            # ── Phase 3: d_w reduce ──
            flatRowBase = slot_id * (groupNum * featureDim)
            rgChunkNum = (featureDim * hiddenSize) // 128
            rgPerCore = rgChunkNum // coreNum 
            rgRemain = rgChunkNum % coreNum
            rgStart = coreIdx * rgPerCore + (coreIdx if coreIdx < rgRemain else rgRemain)
            rgEnd = rgStart + rgPerCore + (1 if coreIdx < rgRemain else 0)
            # pl.printf("[Mo]Coff:%d, cmpRatio:%d, headDim:%d, coreNum:%d, coreIdx:%d\n", Coff, cmpRatio, headDim, coreNum,coreIdx)
            # pl.printf("[Mo]rgChunkNum:%d, rgPerCore:%d, rgRemain:%d\n", rgChunkNum, rgPerCore, rgRemain)
            rgStart = rgStart * 128
            rgEnd = rgEnd * 128
            rgNum = rgEnd - rgStart
            validRows = (roundBlocks + mPerRoundPerGroup - 1) // mPerRoundPerGroup
            # pl.printf("roundBlocks:%d, mPerRoundPerGroup:%d\n", roundBlocks, mPerRoundPerGroup)
            for curRgStart in pl.range(rgStart, rgEnd, (128 * 128)):
                curRgNum = min(128 * 128, rgEnd - curRgStart)
                curRgEnd = curRgStart + curRgNum
                # pl.printf("curRgStart:%d, curRgNum:%d", curRgStart, curRgNum)
                pl.set_validshape(reduce_acc, [1, curRgNum])
                # pl.printf("[Mo] validRows:%d, rgStart:%d, rgEnd:%d, rgEnd - rgStart:%d\n", validRows, rgStart, rgEnd, rgEnd - rgStart)

                # d_wkv
                if roundIdx == 0:
                    _vf_tile_zero(reduce_acc, 1, curRgNum)
                else:
                    pl.load(reduce_acc, tensorDWKvFlat, [0, curRgStart])
                for r in pl.range(0, validRows):
                    # reduce_ld = reduce_ld0 if r % 2 == 0 else reduce_ld1
                    if r % 2 == 0:
                        reduce_ld = reduce_ld_group.next()
                        pl.set_validshape(reduce_ld, [1, curRgNum]) 
                        pl.load(reduce_ld, tensorDWKvWsFlat, [flatRowBase + r, curRgStart])
                        pl.add(reduce_acc, reduce_acc, reduce_ld)
                pl.store(tensorDWKvFlat, reduce_acc, [0, curRgStart])
                # pl.dump_data(tensorDWKvFlat, offsets=[0, curRgStart], shapes=[1, 128])

                # d_wgate
                if roundIdx == 0:
                    _vf_tile_zero(reduce_acc, 1, curRgNum)
                else:
                    pl.load(reduce_acc, tensorDWGateFlat, [0, curRgStart])
                for r in pl.range(0, validRows):
                    reduce_ld = reduce_ld_group.next()
                    pl.set_validshape(reduce_ld, [1, curRgNum]) 
                    pl.load(reduce_ld, tensorDWGateWsFlat, [flatRowBase + r, curRgStart])
                    pl.add(reduce_acc, reduce_acc, reduce_ld)
                pl.store(tensorDWGateFlat, reduce_acc, [0, curRgStart])
            
        # ════════════════════════════════════════════════════════════
        #  Phase 2: Cube Matmul — dkv@wkv + dsb@wgate → d_x_partial
        # ════════════════════════════════════════════════════════════
        with pl.section_cube():
            preDealTcSize = groupIdx * mPerRoundPerGroup
            dealTcSize   = min(mPerRoundPerGroup,
                               max(0, roundBlocks - preDealTcSize))
            pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE1, event_id=slot_id,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            if  Coff != 1:
                # coff=2 CV 路径 L1 布局不同，暂跳过
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
                # pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=slot_id + 2,
                # sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK)
                # pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=slot_id + 2,
                # sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK)
                # pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=slot_id,
                # sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                
                continue
            
            if preDealTcSize >= roundBlocks:
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
                continue


            l1_kv0 = l1_kv0_db.next()
            l1_kv1 = l1_kv1_db.next()
            l1_sb0 = l1_sb0_db.next()
            l1_sb1 = l1_sb1_db.next()

            l1_kv0_t = l1_kv0_db_t.next()
            l1_kv1_t = l1_kv1_db_t.next()
            l1_sb0_t = l1_sb0_db_t.next()
            l1_sb1_t = l1_sb1_db_t.next()

            # d_x partial 写入行偏移
            dx_row_base = slot_id * 256 * cubeCoreNum + cubeCoreIdx * 256

            dealTcSize0 = min(dealTcSize, mBaseSize)
            dealTcSize1 = max(dealTcSize - mBaseSize, 0)
            nTileRows0 = dealTcSize0 * scKvRowCount
            nTileRows1 = dealTcSize1 * scKvRowCount
            # pl.printf("nStart :%d\n", nStart)
            # ── Matmul #1: dkv @ wkv → d_x partial ──
            for ni in pl.range(0, hiddenSize, 256):
                for mi in pl.range(0, 4):
                    nTileRows = nTileRows0 if mi % 2 == 0 else nTileRows1
                    # pl.printf("mi:%d, nTileRows:%d\n", mi, nTileRows)
                    m_off = (mi % 2) * 128
                    l0a = cv_l0a.next()
                    pl.set_validshape(l0a, [nTileRows, dBaseSize])
                    w_l1 = l1_w_db.next() if mi % 2 == 0 else l1_w_db.current()
                    if mi == 0:
                        pl.set_validshape(l1_kv0, [nTileRows, dBaseSize])
                        pl.move(l0a, l1_kv0)
                        pl.set_validshape(w_l1, [128, 256])
                        pl.load(w_l1, wkv_t, [nStart, ni])
                    elif mi == 1:
                        pl.set_validshape(l1_kv1, [nTileRows, dBaseSize])
                        pl.move(l0a, l1_kv1)
                    elif mi == 2:
                        pl.set_validshape(l1_sb0, [nTileRows, dBaseSize])
                        pl.move(l0a, l1_sb0)
                        pl.set_validshape(w_l1, [128, 256])
                        pl.load(w_l1, wgate_t, [nStart, ni])
                    else:
                        pl.set_validshape(l1_sb1, [nTileRows, dBaseSize])
                        pl.move(l0a, l1_sb1)

                    for nii in pl.range(0, 256, 128):
                        l0b = cv_l0b.next()
                        l0c = cv_l0c.next()
                        pl.set_validshape(l0b, [dBaseSize, 128])
                        pl.set_validshape(l0c, [nTileRows, 128])
                        pl.move(l0b, w_l1, offset=[0, nii])
                        if nTileRows == 0:
                            continue
                        if mi // 2 == 0:
                            pl.matmul(l0c, l0a, l0b)
                        else:
                            pl.matmul_acc(l0c, l0c, l0a, l0b)
                            pl.store(d_x_ws, l0c, [dx_row_base + m_off, ni+nii])

            dw_row_base = slot_id * groupNum * featureDim + groupIdx * featureDim + intraGroupIdx * dBaseSize
            mStart = slot_id * groupNum * (256 // Coff) + groupIdx * (256 // Coff)
            # ── Matmul #2: dkv @ x → d_wkv & dscore @ x → d_wgate partial ──
            for ni in pl.range(0, hiddenSize, 256):
                for ii in pl.range(0, 8):
                    ki = ii // 4 # 0, 0, 0, 0, 1, 1, 1, 1
                    nii = (ii % 4 // 2) * 128 # 0, 0, 128, 128, 0, 0, 128, 128
                    mi = ii % 2 # 0, 1, 0, 1, 0, 1, 0, 1

                    nTileRows = nTileRows0 if ki == 0 else nTileRows1
                    l0a = cv_l0a.next()
                    pl.set_validshape(l0a, [dBaseSize, nTileRows])

                    # m_off = (mi % 2) * 128
                    w_l1 = l1_w_db.next() if ii % 4 == 0 else l1_w_db.current()

                    if ii == 0:
                        pl.set_validshape(l1_kv0_t, [dBaseSize, nTileRows])
                        pl.move(l0a, l1_kv0_t)
                        pl.set_validshape(w_l1, [nTileRows, 256])
                        pl.load(w_l1, x_ws, [mStart, ni])

                    elif ii == 1:
                        pl.set_validshape(l1_sb0_t, [dBaseSize, nTileRows])
                        pl.move(l0a, l1_sb0_t)
                    elif ii == 4:
                        pl.set_validshape(l1_kv1_t, [dBaseSize, nTileRows])
                        pl.move(l0a, l1_kv1_t)
                        pl.set_validshape(w_l1, [nTileRows, 256])
                        pl.load(w_l1, x_ws, [mStart + 128, ni])
                    elif ii == 5:
                        pl.set_validshape(l1_sb1_t, [dBaseSize, nTileRows])
                        pl.move(l0a, l1_sb1_t)

                    l0b = cv_l0b.next() if mi == 0 else cv_l0b.current()
                    l0c = cv_l0c.next()
                    pl.set_validshape(l0b, [nTileRows, 128])
                    pl.set_validshape(l0c, [dBaseSize, 128])
                    pl.move(l0b, w_l1, offset=[0, nii])
                    if nTileRows == 0:
                        continue
                    if ki == 0:
                        pl.matmul(l0c, l0a, l0b)
                    else:
                        pl.matmul_acc(l0c, l0c, l0a, l0b)
                    if mi == 0:
                        pl.store(d_wkv_ws, l0c, [dw_row_base, ni+nii])
                    else:
                        pl.store(d_wgate_ws, l0c, [dw_row_base, ni+nii])
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
