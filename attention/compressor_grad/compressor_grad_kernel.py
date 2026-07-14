#!/usr/bin/env python3
"""
compressor_grad_kernel — Compressor 反向传播 NPU Kernel（统一入口）。

TilingKey:  Coff, Layout, DataType               — 编译期常量
TilingData: batchSize, tokenSize, seqSize,       — 运行时参数
            cmpRatio, hiddenSize, headDim

三阶段流水线（同 backward_design.md）:
  Phase 1: Vec Scatter  — 每压缩块计算 d_kv, d_score, APE 累加
  Phase 2: Cube Matmul  — d_new_kv^T @ x → d_wkv 已实现; d_x/d_wgate 待实现
  Phase 3: Vec Reduce   — 跨 D 组归约 + 跨轮 RMW

当前状态: Phase 1 已实现, Phase 2 d_wkv 已实现 (从 block_cube.d_wkv 合并),
         Phase 2 d_x/d_wgate 及 Phase 3 待实现
"""

from __future__ import annotations
from dataclasses import dataclass
import torch, torch_npu
import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField as tilingkey_field
from pypto_pro.runtime.platform import get_platform_info

# ================================================================
#  Compile-time constants
# ================================================================
FLOAT_REG_ELEMENTS = 64   # FP32 VF register holds 64 elements

# ================================================================
#  TilingKey
# ================================================================
class CompressorTilingKey:
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
class CompressorTiling:
    batchSize:  int = 1
    tokenSize:  int = 0
    seqSize:    int = 0
    cmpRatio:   int = 4
    hiddenSize: int = 512
    headDim:    int = 128


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
def _get_start_pos(bIdx, start_pos, batchSize):
    result = 0
    if start_pos is not None:
        result = pl.getval(pl.make_tensor(start_pos, [batchSize], [1], dtype=pl.DT_INT32), bIdx)
    return result


def _get_seq_length(bIdx, cu_seqlens, batchSize, Layout, seqSize):
    result = seqSize
    if Layout == 0:  # TH
        result = pl.getval(pl.make_tensor(cu_seqlens, [batchSize + 1], [1], dtype=pl.DT_INT32), bIdx + 1) - \
                 pl.getval(pl.make_tensor(cu_seqlens, [batchSize + 1], [1], dtype=pl.DT_INT32), bIdx)
    return result


def _get_seq_used(bIdx, seqused, cu_seqlens, batchSize, Layout, seqSize):
    if seqused is not None:
        result = pl.getval(pl.make_tensor(seqused, [batchSize], [1], dtype=pl.DT_INT32), bIdx)
    else:
        result = _get_seq_length(bIdx, cu_seqlens, batchSize, Layout, seqSize)
    return result


# ================================================================
#  Helper: advance block traversal (used by Segment 1 and 3)
# ================================================================
def _skip_blocks(blockCount, bIdx, scIdx,
                 start_pos, seqused, cu_seqlens, batchSize, cmpRatio, Layout, seqSize):
    skipped = 0
    while skipped < blockCount:
        spVal = _get_start_pos(bIdx, start_pos, batchSize)
        sqVal = _get_seq_used(bIdx, seqused, cu_seqlens, batchSize, Layout, seqSize)
        cmpLimit = (spVal + sqVal) // cmpRatio * cmpRatio
        blkInBatch = (cmpLimit - spVal) // cmpRatio
        remaining = blkInBatch - scIdx
        if remaining > 0:
            taken = min(blockCount - skipped, remaining)
            skipped += taken
            scIdx += taken
        if scIdx >= blkInBatch:
            scIdx = 0
            bIdx += 1
        if bIdx >= batchSize:
            break
    return bIdx, scIdx


# ================================================================
#  Unified Compressor Grad Kernel
# ================================================================
@pl.jit(tiling_key=CompressorTilingKey, auto_mutex=True)
def compressor_grad_kernel(
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
    tiling:         CompressorTiling,
):
    batchSize  = tiling.batchSize
    tokenSize  = tiling.tokenSize
    seqSize    = tiling.seqSize
    cmpRatio   = tiling.cmpRatio
    headDim    = tiling.headDim
    hiddenSize = tiling.hiddenSize
    featureDim = Coff * headDim
    scKvRowCount = Coff * cmpRatio
    outStride    = (seqSize + cmpRatio - 1) // cmpRatio
    ioDType      = pl.DT_BF16 if DataType == 0 else pl.DT_FP16

    # ── Compute total compression blocks ──
    totalBlocks = 0
    for bi in pl.range(0, batchSize):
        sp = _get_start_pos(bi, start_pos, batchSize)
        sq = _get_seq_used(bi, seqused, cu_seqlens, batchSize, Layout, seqSize)
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
    # d_new_kv GM view: vec 阶段写, cube 阶段读. 形状 [T, C], ioDType.
    # 反步 3 路由: 每个 cmp 块的 d_kv_block 按 overlap_id 写到对应 token 行和
    # overlap_id*headDim + nStart 列. 累加 (atomic add) 因不同 core 可能写到同
    # 一 GM cell (跨 round / 跨 batch overlap).
    tensorDNewKv = pl.make_tensor(workspace, [tokenSize, featureDim],
                                  [featureDim, 1], dtype=ioDType)
    # ════════════════════════════════════════════════════════════
    #  Phase 1: Vec Scatter
    # ════════════════════════════════════════════════════════════
    with pl.section_vector():
        coreNum   = pl.get_block_num()
        coreIdx   = pl.get_block_idx()


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
            layout=pl.NZ)

        tile_type_work = pl.TileType(
            shape=[128 * Coff, 128 // Coff], dtype=pl.DT_FP32,
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
        ape_local_group = pl.make_tile_group(type=tile_type_work, addrs=[0x0C000], mutex_ids=[7])
        dkv_nz_group = pl.make_tile_group(type=tile_type_cast_nz, addrs=[0x34000], mutex_ids=[6])
        dsb_nz_group = pl.make_tile_group(type=tile_type_cast_nz, addrs=[0x24000], mutex_ids=[5])

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

        # ── D group division ──
        groupSize        = headDim // 128
        dBaseSize        = 128 // Coff
        nCorePerGroup    = (coreNum // 2) // groupSize
        cubeCoreIdx      = coreIdx // 2
        groupIdx         = cubeCoreIdx // groupSize
        intraGroupIdx    = cubeCoreIdx % groupSize

        nStart = intraGroupIdx * 128 + (coreIdx % Coff) * dBaseSize
        nEnd   = nStart + dBaseSize

        # ── M axis round division ──
        mPerRoundPerGroup = 128 * 2 // Coff // cmpRatio
        mBaseSize         = 128 // cmpRatio
        mPerRoundAll      = nCorePerGroup * mPerRoundPerGroup
        roundCnt          = (totalBlocks + mPerRoundAll - 1) // mPerRoundAll
        bIdx              = 0
        scIdx             = 0


        wsRows = roundCnt * mPerRoundAll * cmpRatio
        tensorApe = pl.make_tensor(d_ape,
            [wsRows, headDim],
            [headDim, 1], dtype=pl.DT_FP32)


        for roundIdx in pl.range(0, roundCnt):
            roundBlocks = min(mPerRoundAll,
                              totalBlocks - roundIdx * mPerRoundAll)
            preDealTcSize = groupIdx * mPerRoundPerGroup
            if Coff == 1 and coreIdx % 2 == 1:
                preDealTcSize += mBaseSize
            if preDealTcSize >= roundBlocks:
                break
            dealTcSize   = min(mBaseSize,
                               max(0, roundBlocks - preDealTcSize))
            blocksAfter  = max(0, roundBlocks - preDealTcSize - dealTcSize)

            # Segment 1: skip
            bIdx, scIdx = _skip_blocks(preDealTcSize, bIdx, scIdx,
                                       start_pos, seqused, cu_seqlens,
                                       batchSize, cmpRatio, Layout, seqSize)

            # Segment 2: process
            if Layout == 0:
                storeStart = roundIdx * mPerRoundAll + preDealTcSize
            else:
                storeStart = bIdx * outStride + scIdx
            processed = 0
            while processed < dealTcSize:
                spVal = _get_start_pos(bIdx, start_pos, batchSize)
                sqVal = _get_seq_used(bIdx, seqused, cu_seqlens, batchSize, Layout, seqSize)
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

                    # 2D slice subview (parser requires tile[r:r2, c:c2] form,
                    # not 1D flat offset tile[flat]).  Each tile's column count
                    # equals dDealSize (the per-deal D slice), and rows advance
                    # by `processed * <rows per block>` along the M axis.
                    dcRowOff  = processed * dDealSize
                    dcTile    = tile_dc_f16[dcRowOff : dcRowOff + taken, 0 : dDealSize]
                    dcF32     = tile_dc_f32[dcRowOff : dcRowOff + taken, 0 : dDealSize]
                    kvRowOff  = processed * scKvRowCount
                    kvTile    = tile_kv[kvRowOff : kvRowOff + nTileRows, 0 : dDealSize]
                    smTile    = tile_softmax[kvRowOff : kvRowOff + nTileRows, 0 : dDealSize]
                    dkvTile   = tile_temp[kvRowOff : kvRowOff + nTileRows, 0 : dDealSize]

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
                            smTile[dDealSize : dDealSize + nTileRows, 0 : dDealSize],
                            dkvTile[dDealSize : dDealSize + nTileRows, 0 : dDealSize],
                            cache_fill_rows, dDealSize, dDealSize * Coff,
                        )

                    # ── 反步 3: route dkv → GM d_new_kv (workspace) ──
                    # d_new_kv[T, C] += d_kv_block[overlap_id*cmpRatio + cache_rows :
                    #                              overlap_id*cmpRatio + cmpRatio, :]
                    # at GM rows [flat_off_s:flat_off_e], cols [overlap_id*headDim + nStart
                    #             : overlap_id*headDim + nStart + dDealSize].
                    # Coff==1: only overlap_id=0 (cur). Coff==2: overlap_id=0 (prev) + 1 (cur).
                    # RMW via atomic add: a token at batch boundary can be both the cur
                    # of one cmp block and the prev of the next (overlap), so累加是必要的.
                    block_seq_start = spVal + (scIdx - taken) * cmpRatio if Layout == 0 \
                                       else _get_seq_used(bIdx, seqused, cu_seqlens, batchSize, Layout, seqSize)  # placeholder
                    # Recompute per-block flat offsets and route each overlap slot.
                    for b in pl.range(0, taken):
                        # 当前 block 在 batch 内的相对位置
                        blk_scIdx = scIdx - taken + b
                        # block 全局序列起止
                        blk_seq_s = spVal + blk_scIdx * cmpRatio
                        blk_seq_e = blk_seq_s + cmpRatio
                        if Layout == 0:
                            flat_base = bIdx * seqSize
                        else:
                            flat_base = _get_start_pos(bIdx, start_pos, batchSize) + \
                                        blk_scIdx * cmpRatio - spVal + spVal
                            flat_base = bIdx * seqSize
                        fos = flat_base + (blk_seq_s - spVal)
                        foe = flat_base + (blk_seq_e - spVal)

                        # UB tile rows for this block: b*scKvRowCount .. (b+1)*scKvRowCount
                        blk_row_s = b * scKvRowCount
                        # ── current overlap (overlap_id = Coff - 1) ──
                        cur_id = Coff - 1
                        cur_cache = 0
                        if spVal + blk_scIdx * cmpRatio == spVal and blk_scIdx == 0:
                            cur_cache = spVal % cmpRatio
                        cur_row_s = blk_row_s + cur_id * cmpRatio + cur_cache
                        cur_row_e = blk_row_s + cur_id * cmpRatio + cmpRatio
                        cur_col_s = cur_id * headDim + nStart
                        cur_sub = dkvTile[cur_row_s : cur_row_e, 0 : dDealSize]
                        pl.set_validshape(cur_sub, [cur_row_e - cur_row_s, dDealSize])
                        pl.store(tensorDNewKv, cur_sub, [fos + cur_cache, cur_col_s],
                                 atomic=pl.AtomicType.AtomicAdd)

                        # ── prev overlap (overlap_id = 0, coff == 2 only) ──
                        if Coff == 2:
                            prev_cache = 0
                            if blk_scIdx == 0:
                                prev_cache = cmpRatio
                            elif blk_seq_s - cmpRatio < spVal:
                                prev_cache = spVal % cmpRatio
                            if prev_cache < cmpRatio:
                                prev_row_s = blk_row_s + prev_cache
                                prev_row_e = blk_row_s + cmpRatio
                                prev_col_s = 0 * headDim + nStart
                                prev_fos = fos - (cmpRatio - prev_cache)
                                prev_foe = fos
                                prev_sub = dkvTile[prev_row_s : prev_row_e, 0 : dDealSize]
                                pl.set_validshape(prev_sub, [prev_row_e - prev_row_s, dDealSize])
                                pl.store(tensorDNewKv, prev_sub,
                                         [prev_fos, prev_col_s],
                                         atomic=pl.AtomicType.AtomicAdd)

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

            slotRow = roundIdx * mPerRoundAll * cmpRatio + groupIdx * mPerRoundPerGroup * cmpRatio

            if Coff == 1 and coreIdx % 2 == 1:
                slotRow += mBaseSize * cmpRatio

            pl.store(tensorApe, tile_ape_local, [slotRow, nStart])

            # ── Cast dkv/dsb FP32→BF16/FP16, move ND→NZ for CV ──
            pl.set_validshape(tile_dkv_cast, [nTileRows, dBaseSize])
            pl.set_validshape(tile_temp,     [nTileRows, dBaseSize])
            pl.cast(tile_dkv_cast, tile_temp, mode=pl.RoundMode.CAST_ROUND)
            pl.set_validshape(tile_dkv_nz, [nTileRows, dBaseSize])
            pl.move(tile_dkv_nz, tile_dkv_cast)

            pl.set_validshape(tile_dsb_cast, [nTileRows, dBaseSize])
            pl.set_validshape(tile_softmax,  [nTileRows, dBaseSize])
            pl.cast(tile_dsb_cast, tile_softmax, mode=pl.RoundMode.CAST_ROUND)
            pl.set_validshape(tile_dsb_nz, [nTileRows, dBaseSize])
            pl.move(tile_dsb_nz, tile_dsb_cast)

            # Segment 3: skip
            bIdx, scIdx = _skip_blocks(blocksAfter, bIdx, scIdx,
                                       start_pos, seqused, cu_seqlens,
                                       batchSize, cmpRatio, Layout, seqSize)

    # ════════════════════════════════════════════════════════════
    #  Phase 2: Cube Matmul — d_wkv  (merged from block_cube.d_wkv)
    # ════════════════════════════════════════════════════════════
    # d_wkv[C, H] = d_new_kv.T[C, T] @ x_flat[T, H]
    #
    # 输入:
    #   workspace  — Phase 1 产出的 d_new_kv [T, C] (ioDType, BF16/FP16)
    #   x          — 原始输入 [T, H] (ioDType) (BSH 视为 [T, H] 内存连续)
    # 输出:
    #   d_wkv      — [C, H] FP32 (MUST be zero-init, cross-core atomic add)
    #
    # 多核策略: K-split (K=T 切分到各核), per-core K 超过单 tile 时通过
    # matmul / matmul_acc (phase Partial/Final) 在 L0C 内 K-累加, 然后跨核
    # atomic add 到 GM. Transpose-load (order=[1,0]) 把 d_new_kv[T,C] 在片内
    # 转置成 [C,T] 左操作数 (Mat layout=ZN). 详见 block_cube.d_wkv docstring.
    with pl.section_cube():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()
        k_per_core = tokenSize // num_cores
        k_start = core_id * k_per_core

        # Coff-dependent tiling (compile-time constants per TilingKey)
        if Coff == 1:
            wkv_tile_m = 128
            wkv_tile_k = 64
            wkv_tile_n = 128
        else:  # Coff == 2
            wkv_tile_m = 256
            wkv_tile_k = 32
            wkv_tile_n = 64

        # L1/L0 tile groups (与 block_cube.d_wkv 同布局, mutex_ids 0-9)
        a_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[wkv_tile_m, wkv_tile_k], dtype=ioDType,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
            addrs=0x00000, mutex_ids=[0, 1],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[wkv_tile_k, wkv_tile_n], dtype=ioDType,
                             target_memory=pl.MemorySpace.Mat),
            addrs=0x08000, mutex_ids=[2, 3],
        )
        a_left = pl.make_tile_group(
            type=pl.TileType(shape=[wkv_tile_m, wkv_tile_k], dtype=ioDType,
                             target_memory=pl.MemorySpace.Left),
            addrs=0x00000, mutex_ids=[4, 5],
        )
        b_right = pl.make_tile_group(
            type=pl.TileType(shape=[wkv_tile_k, wkv_tile_n], dtype=ioDType,
                             target_memory=pl.MemorySpace.Right),
            addrs=0x00000, mutex_ids=[6, 7],
        )
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[wkv_tile_m, wkv_tile_n], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, fractal=1024),
            addrs=0x00000, mutex_ids=[8, 9],
        )

        # GM 视图: workspace 作为 d_new_kv [T, C] (ioDType); x 作为 [T, H] (ioDType)
        d_new_kv_t = pl.make_tensor(workspace, [tokenSize, featureDim],
                                    [featureDim, 1], dtype=ioDType)
        x_flat = pl.make_tensor(x, [tokenSize, hiddenSize],
                                [hiddenSize, 1], dtype=ioDType)
        d_wkv_out = pl.make_tensor(d_wkv, [featureDim, hiddenSize],
                                   [hiddenSize, 1], dtype=pl.DT_FP32)

        needs_k_acc = k_per_core > wkv_tile_k
        if needs_k_acc:
            pl.system.set_mm_layout_transform(enabled=True)

        for mi in pl.range(0, featureDim, wkv_tile_m):
            for ni in pl.range(0, hiddenSize, wkv_tile_n):
                if needs_k_acc:
                    ac = acc.current()
                    for ki_off in pl.range(0, k_per_core, wkv_tile_k):
                        ki = k_start + ki_off
                        cur_a = a_l1.next()
                        cur_b = b_l1.next()
                        al = a_left.next()
                        br = b_right.next()
                        pl.load(cur_a, d_new_kv_t, [ki, mi], order=[1, 0])
                        pl.load(cur_b, x_flat, [ki, ni])
                        pl.move(al, cur_a)
                        pl.move(br, cur_b)
                        is_first = (ki_off == 0)
                        is_last = (ki_off + wkv_tile_k >= k_per_core)
                        if is_first:
                            pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
                        elif is_last:
                            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
                        else:
                            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
                    pl.store(d_wkv_out, ac, [mi, ni],
                             atomic=pl.AtomicType.AtomicAdd,
                             phase=pl.STPhase.Final)
                else:
                    cur_a = a_l1.next()
                    cur_b = b_l1.next()
                    al = a_left.next()
                    br = b_right.next()
                    ac = acc.next()
                    pl.load(cur_a, d_new_kv_t, [k_start, mi], order=[1, 0])
                    pl.load(cur_b, x_flat, [k_start, ni])
                    pl.move(al, cur_a)
                    pl.move(br, cur_b)
                    pl.matmul(ac, al, br)
                    pl.store(d_wkv_out, ac, [mi, ni],
                             atomic=pl.AtomicType.AtomicAdd)

        if needs_k_acc:
            pl.system.set_mm_layout_transform(enabled=False)

    # ════════════════════════════════════════════════════════════
    #  Phase 2: Cube Matmul  (TODO: d_wgate / d_x)
    # ════════════════════════════════════════════════════════════
    # Phase 2 职责（per round）:
    #   ① d_new_kv @ wkv           → d_x_partial 累加
    #   ② d_new_score @ wgate      → d_x_partial 累加
    #   ③ d_new_kv^T @ x           → d_wkv_partial   (DONE above)
    #   ④ d_new_score^T @ x        → d_wgate_partial
    #
    # 输入（CV 通路，Phase 1 产出）:
    #   本轮 d_kv (BF16/FP16, M×K) via L1A
    #   本轮 d_score (BF16/FP16, M×K) via L1A
    #
    # 输入（GM 流式加载）:
    #   wkv/wgate (K×H) via L1W
    #   x 子块 (M_block×H) via L1W
    #
    # 输出（写 workspace 供 Phase 3 reduce）:
    #   d_x_partial:     [D_groups][T_round][H] FP32
    #   d_wkv_partial:   [MBlocks_round][C][H] FP32   (DONE above)
    #   d_wgate_partial: [MBlocks_round][C][H] FP32
    #
    # Cube 分核: 按 D 组划分，每组 nCorePerGroup 个 Cube
    # L1/L0 布局: 见 docs/backward_design.md §5

    # ════════════════════════════════════════════════════════════
    #  Phase 3: Vec Reduce  (TODO)
    # ════════════════════════════════════════════════════════════
    # Phase 3 职责（per round，sync_all 后）:
    #   d_x:    Σ_g d_x_partial[g, :, :]     → d_x[T_round, :]
    #   d_wkv:  prev + Σ_m d_wkv_partial[m, :, :]   → d_wkv[:, :] (RMW)
    #   d_wgate: prev + Σ_m d_wgate_partial[m, :, :] → d_wgate[:, :] (RMW)
    #   d_ape:  prev + Σ_slot d_ape_ws[slot, :, :]  → d_ape[:, :] (RMW)
    #
    # 分核方案: 统一 flat 128 元素分块 + 跨步轮转（见 phase3_core_split_design.md）
