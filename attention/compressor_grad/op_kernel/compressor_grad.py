#!/usr/bin/env python3
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
D_BASE_SIZE = 128
M_BASE_SIZE = 128

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


def _ceil_div(dividend, divisor):
    return 0 if divisor == 0 else ((dividend) + (divisor) - 1) // (divisor)

def _align(num, rnd):
    return 0 if (rnd) == 0 else (((num) + (rnd)-1) // (rnd) * (rnd))

def _trunc(num, rnd):
    return 0 if (rnd) == 0 else (((num) // (rnd) * (rnd)))

# ================================================================
#  VF: 反向 scatter — d_kv = dC * softmax, d_score = softmax * (dsw - col_sum)
# ================================================================
@pl.vector_function
def _vf_scatter_backward(
    dCmpKvTile,       # dC: (n_blocks, D) FP32 (pre-cast outside VF)
    kvTile,           # kv_2d: (n_blocks*N, D) fp32 — kv → dsw
    softmaxScoreTile,      # softmax: (n_blocks*N, D) fp32 — sm → dsb
    tempTile,         # temp: (n_blocks*N, D) fp32 — dkv output
    scNum: pl.DT_INT64,
    cmpRowCnt: pl.DT_INT64, # cmpRatio * coff
    dDealSize: pl.DT_INT64,
):
    dSoftmaxScoreTile = kvTile
    dScoreTile = softmaxScoreTile
    dKvTile = tempTile

    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)

    halves = dDealSize // FLOAT_REG_ELEMENTS
    cmpSize = cmpRowCnt * dDealSize

    for dIdx in pl.range(0, halves):
        dOffset = dIdx * FLOAT_REG_ELEMENTS

        for scIdx in pl.range(0, scNum):
            scOffset = scIdx * cmpSize
            dCmpKvOffset = scIdx * dDealSize + dOffset

            vregDCmpKv = vf.load_align(dCmpKvTile, dCmpKvOffset)

            vregSum = vf.full(0.0, dtype=pl.DT_FP32)
            for tIdx in pl.range(0, cmpRowCnt):
                offset = scOffset + tIdx * dDealSize + dOffset
                vregSoftmaxScore = vf.load_align(softmaxScoreTile, offset)
                vregKv = vf.load_align(kvTile, offset)
                vregDKv = vf.mul(vregDCmpKv, vregSoftmaxScore, mask)
                vf.store_align(dKvTile + offset, vregDKv, mask)
                vregDSoftmaxScore = vf.mul(vregDCmpKv, vregKv, mask)
                vf.store_align(dSoftmaxScoreTile + offset, vregDSoftmaxScore, mask)
                vregDSoftmaxScore = vf.mul(vregSoftmaxScore, vregDSoftmaxScore, mask)
                vregSum = vf.add(vregSum, vregDSoftmaxScore, mask)

            vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
            for tIdx in pl.range(0, cmpRowCnt):
                offset = scOffset + tIdx * dDealSize + dOffset
                vregSoftmaxScore = vf.load_align(softmaxScoreTile, offset)
                vregDSoftmaxScore = vf.load_align(dSoftmaxScoreTile, offset)
                vregDSoftmaxScore = vf.sub(vregDSoftmaxScore, vregSum, mask)
                vregDScore = vf.mul(vregSoftmaxScore, vregDSoftmaxScore, mask)
                vf.store_align(dScoreTile + offset, vregDScore, mask)
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
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    halves = n_cols // FLOAT_REG_ELEMENTS
    row_stride = n_cols
    zero_reg = vf.full(0.0, dtype=pl.DT_FP32)
    for half in pl.range(0, halves):
        half_off = half * FLOAT_REG_ELEMENTS
        for m in pl.range(0, n_rows):
            vf.store_align(tile + m * row_stride + half_off, zero_reg, mask)
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
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    halves = n_cols // 128
    row_stride = n_cols
    zero_reg = vf.full(0.0, dtype=pl.DT_FP16)
    for half in pl.range(0, halves):
        half_off = half * 128
        for m in pl.range(0, n_rows):
            vf.store_align(tile + m * row_stride + half_off, zero_reg, mask)
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
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    halves = n_cols // 128
    row_stride = n_cols
    zero_reg = vf.full(0.0, dtype=pl.DT_BF16)
    for half in pl.range(0, halves):
        half_off = half * 128
        for m in pl.range(0, n_rows):
            vf.store_align(tile + m * row_stride + half_off, zero_reg, mask)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


# ================================================================
#  VF: 掩码清零 — 将非有效 token 对应的 d_kv/d_score 行置零
# ================================================================
@pl.vector_function
def _vf_mask_cache_fill(
    dScoreTile,
    dkvTile,
    row: pl.DT_INT64,
    dDealSize: pl.DT_INT64,
    dStride: pl.DT_INT64,
):
    if row > 0:
        mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
        halves = dDealSize // FLOAT_REG_ELEMENTS
        vregZero = vf.full(0.0, dtype=pl.DT_FP32)
        for hIdx in pl.range(0, halves):
            dOffset = hIdx * FLOAT_REG_ELEMENTS
            for rIdx in pl.range(0, row):
                vf.store_align(dScoreTile + rIdx * dStride + dOffset, vregZero, mask)
                vf.store_align(dkvTile + rIdx * dStride + dOffset, vregZero, mask)
        vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


# ================================================================
#  VF: 跨 block 归约 — d_ape[m,:] += Σ_k d_score[k,m,:]（per-round 调用）
# ================================================================
@pl.vector_function
def _vf_reduce_dscore_to_ape(
    dScoreTile,
    ape_tile,
    n_blocks: pl.DT_INT64,
    n_rows_per_block: pl.DT_INT64,
    dDealSize: pl.DT_INT64,
    d_stride: pl.DT_INT64,
):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
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
                vregDScore = vf.load_align(dScoreTile + row_off, 0)
                ape_reg = vf.add(ape_reg, vregDScore, mask)
            vf.store_align(ape_tile + ape_off, ape_reg, mask)
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
    if ctx.layout == 1:  # TH (1=TH)
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
    if ctx.layout == 1:
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
        bStartPos = _get_start_pos(bb, start_pos, ctx)
        bSeqLength = _get_seq_used(bb, seqused, cu_seqlens, ctx)
        cmpEnd = (bStartPos + bSeqLength) // ctx.cmpRatio
        if cmpEnd > bStartPos // ctx.cmpRatio:
            result += cmpEnd - (bStartPos // ctx.cmpRatio)
    bStartPos = _get_start_pos(bIdx, start_pos, ctx)
    lastS = _trunc(_get_seq_used(bIdx, seqused, cu_seqlens, ctx) + bStartPos, ctx.cmpRatio) - bStartPos
    result += (min(lastS, sIdx) + bStartPos) // ctx.cmpRatio - bStartPos // ctx.cmpRatio
    return result


# ================================================================
#  Helper: 将全局 token 索引 tIdx 反向映射为 (bIdx, sIdx)
# ================================================================
def _get_pos_from_token_idx(tIdx, cu_seqlens, ctx):
    resultB = tIdx // ctx.seqSize
    resultS = tIdx % ctx.seqSize
    if ctx.layout == 1:
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
        bStartPos = _get_start_pos(bIdx, start_pos, ctx)
        sqVal = _get_seq_used(bIdx, seqused, cu_seqlens, ctx)
        cmpLimit = (bStartPos + sqVal) // ctx.cmpRatio * ctx.cmpRatio
        blkInBatch = (cmpLimit - bStartPos) // ctx.cmpRatio
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
    d_cmp_kv:       pl.Ptr[pl.DT_UINT8],     # (cmpKvRows, headDim)           
    kv:             pl.Ptr[pl.DT_UINT8],     # (cmpKvRows, cmpRowCnt, headDim) FP32
    softmax_score:  pl.Ptr[pl.DT_UINT8],     # (cmpKvRows, cmpRowCnt, headDim) FP32

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
    seqSize    = tiling.seqSize
    tokenSize  = tiling.tokenSize if Layout == 1 else batchSize * seqSize
    cmpRatio   = tiling.cmpRatio
    headDim    = tiling.headDim
    hiddenSize = tiling.hiddenSize
    totalHeadDim = Coff * headDim
    cmpRowCnt = Coff * cmpRatio
    cmpSize = Coff * cmpRatio * headDim
    cmpKvBatchStride    = _ceil_div(seqSize, cmpRatio)
    ioDType      = pl.DT_BF16 if DataType == 0 else pl.DT_FP16
    dbRatio = 2 # tiling传入
    dDealSize = D_BASE_SIZE // Coff # 不从tiling 传入，作为编译期常数
    mDealSize = M_BASE_SIZE * Coff
    l1Addr = 0
    ubAddr = 0

    # ── Sequence context ──
    seq_ctx = pl.struct("SeqCtx",
        batchSize=batchSize,
        cmpRatio=cmpRatio,
        layout=Layout,
        seqSize=seqSize,
    )

    # ── Compute total compression blocks ──
    compressedCnt = 0
    for bIdx in pl.range(0, batchSize):
        bStartPos = _get_start_pos(bIdx, start_pos, seq_ctx)
        bSeqUsed = _get_seq_used(bIdx, seqused, cu_seqlens, seq_ctx)
        compressSeqIdx = _trunc(bStartPos + bSeqUsed, cmpRatio)
        if compressSeqIdx > bStartPos:
            compressedCnt = compressedCnt + (compressSeqIdx - bStartPos) // cmpRatio

    if Layout == 1:
        cmpKvRows = min(tokenSize, (tokenSize // cmpRatio) + batchSize)
    else:
        cmpKvRows = batchSize * cmpKvBatchStride


    # ── Gm Tensor ──
    dCmpKvGm = pl.make_tensor(d_cmp_kv, [cmpKvRows, headDim],
                                   [headDim, 1], dtype=ioDType)
    kvGm      = pl.make_tensor(kv, [cmpKvRows, cmpRowCnt, headDim],
                                   [cmpSize, headDim, 1], dtype=pl.DT_FP32)
    softmaxScoreGm = pl.make_tensor(softmax_score, [cmpKvRows, cmpRowCnt, headDim],
                                   [cmpSize, headDim, 1], dtype=pl.DT_FP32)
    
    xGm = pl.make_tensor(x, [tokenSize, hiddenSize],
                            [hiddenSize, 1], dtype=ioDType) #TODO:bsh时，tokenSize=b*s

    xRows = tokenSize if Layout == 1 else batchSize * seqSize
    dXOutGm = pl.make_tensor(d_x, [xRows, hiddenSize],
                              [hiddenSize, 1], dtype=ioDType)
    dWKvOutGm = pl.make_tensor(d_wkv, [totalHeadDim, hiddenSize],
                              [hiddenSize, 1], dtype=ioDType)
    dWGateOutGm = pl.make_tensor(d_wgate, [totalHeadDim, hiddenSize],
                              [hiddenSize, 1], dtype=ioDType)


    # ── 共享 round 参数（Phase 1 + Phase 2 都需要）──
    cubeCoreNum = pl.get_block_num() #TODO:改成tiling传入
    groupSize         = headDim // D_BASE_SIZE
    groupNum     = cubeCoreNum // groupSize
    coffCoef =  2 // Coff
    cubeMBaseSize = M_BASE_SIZE * coffCoef # coff = 1, vec子核分M轴；coff = 2, vec子核分coff
    groupDealScNum = cubeMBaseSize // cmpRatio
    dealScNum         = M_BASE_SIZE // cmpRatio
    totalScNumPerRound      = groupNum * groupDealScNum
    roundCnt          = _ceil_div(compressedCnt, totalScNumPerRound)
    totalRowsPerRound            = totalScNumPerRound * cmpRowCnt

    # ── L1A CV 通道: 4 个独立 tile_group [128,128], double-buffer ──
    L1_NZ_TILE_TYPE = pl.TileType(shape=[M_BASE_SIZE, D_BASE_SIZE], dtype=ioDType,
                              target_memory=pl.MemorySpace.Mat, layout=pl.NZ, compact=1)
    L1_ZN_TILE_TYPE = pl.TileType(shape=[M_BASE_SIZE, D_BASE_SIZE], dtype=ioDType,
                              target_memory=pl.MemorySpace.Mat, layout=pl.ZN, compact=1)

    L1Kv0TileGroupNz    = pl.make_tile_group(type=L1_NZ_TILE_TYPE, addrs=0x00000, mutex_ids=[10, 11]) # 128 * 128 * 2 = 0x10000
    L1Kv1TileGroupNz    = pl.make_tile_group(type=L1_NZ_TILE_TYPE, addrs=0x10000, mutex_ids=[12, 13])
    L1Score0TileGroupNz    = pl.make_tile_group(type=L1_NZ_TILE_TYPE, addrs=0x20000, mutex_ids=[14, 15])
    L1Score1TileGroupNz    = pl.make_tile_group(type=L1_NZ_TILE_TYPE, addrs=0x30000, mutex_ids=[16, 17])

    L1Kv0TileGroupZn  = pl.make_tile_group(type=L1_ZN_TILE_TYPE, addrs=0x00000, mutex_ids=[10, 11])
    L1Kv1TileGroupZn  = pl.make_tile_group(type=L1_ZN_TILE_TYPE, addrs=0x10000, mutex_ids=[12, 13])
    L1Score0TileGroupZn  = pl.make_tile_group(type=L1_ZN_TILE_TYPE, addrs=0x20000, mutex_ids=[14, 15])
    L1Score1TileGroupZn  = pl.make_tile_group(type=L1_ZN_TILE_TYPE, addrs=0x30000, mutex_ids=[16, 17])

    # ── Workspace 分块: APE 区域 + d_x 区域──
    apeWorkSpaceSize   = dbRatio * groupNum * cmpSize * coffCoef
    dXWorkSpaceSize    = dbRatio * cubeCoreNum * (M_BASE_SIZE * 2) * hiddenSize
    dWeightWorkSpaceSize    = dbRatio * groupNum * totalHeadDim * hiddenSize
    xWorkSpaceSize    = dbRatio * (groupNum * cubeMBaseSize + (Coff - 1) * cmpRatio) * hiddenSize
    dXCacheWorkSpaceSize    = dbRatio * cmpRatio * hiddenSize

    workSpacePtr = pl.make_ptr(workspace, dtype=pl.DT_FP32)
    dXWorkSpacePtr = pl.addptr(workSpacePtr, apeWorkSpaceSize)       
    dWKvWorkSpacePtr = pl.addptr(dXWorkSpacePtr, dXWorkSpaceSize)   
    dWGateWorkSpacePtr = pl.addptr(dWKvWorkSpacePtr, dWeightWorkSpaceSize)   
    xWorkSpacePtr = pl.addptr(dWGateWorkSpacePtr, dWeightWorkSpaceSize)   
    dXCachePtr = pl.addptr(xWorkSpacePtr, xWorkSpaceSize)   

    # ── d_x output + workspace views (Phase 3) ──
    dXResultGm = pl.make_tensor(dXWorkSpacePtr,
        [Coff, dXWorkSpaceSize // hiddenSize // Coff, hiddenSize], [dXWorkSpaceSize // Coff, hiddenSize, 1], dtype=pl.DT_FP32)
    dWkvResultGm = pl.make_tensor(dWKvWorkSpacePtr,
        [dWeightWorkSpaceSize // hiddenSize, hiddenSize], [hiddenSize, 1], dtype=pl.DT_FP32)
    dWGateResultGm = pl.make_tensor(dWGateWorkSpacePtr,
        [dWeightWorkSpaceSize // hiddenSize, hiddenSize], [hiddenSize, 1], dtype=pl.DT_FP32)
    xArrangeGm = pl.make_tensor(xWorkSpacePtr,
        [xWorkSpaceSize // hiddenSize, hiddenSize], [hiddenSize, 1], dtype=ioDType)
    dXCacheGm = pl.make_tensor(dXCachePtr,
        [dXCacheWorkSpaceSize // hiddenSize, hiddenSize], [hiddenSize, 1], dtype=pl.DT_FP32)

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
        subCoreIdx = pl.get_subblock_idx()
        # pl.printf("subCoreIdx:%d\n", subCoreIdx)
        # pl.printf("coreNum:%d, coreIdx:%d\n", coreNum, coreIdx)

        # ── UB tile type declarations ──
        D_CMP_KV_TILE_TYPE = pl.TileType(
            shape=[mDealSize // 2, dDealSize], dtype=ioDType,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])

        D_CMP_KV_FP32_TILE_TYPE = pl.TileType(shape=[mDealSize // 2, dDealSize], dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1,-1], pad=pl.TilePad.zero)
        
        NORMEL_TILE_TYPE = pl.TileType(
            shape=[mDealSize, dDealSize], dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1],
            pad=pl.TilePad.zero, compact=1)
        
        CAST_TILE_TYPE = pl.TileType(
            shape=[mDealSize, dDealSize], dtype=ioDType,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1)

        CAST_MD_TILE_TYPE = pl.TileType(
            shape=[M_BASE_SIZE, D_BASE_SIZE], dtype=ioDType,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1)

        NZ_TILE_TYPE = pl.TileType(
            shape=[mDealSize, dDealSize], dtype=ioDType,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1],
            layout=pl.NZ, compact=1)
        
        NZ_MD_TILE_TYPE = pl.TileType(
            shape=[M_BASE_SIZE, dDealSize], dtype=ioDType,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1],
            layout=pl.NZ, compact=1)

        MOVE_X_TILE_TYPE = pl.TileType(
            shape=[M_BASE_SIZE, D_BASE_SIZE * 2], dtype=ioDType,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1],
            pad=pl.TilePad.zero)
        
        FLAT_TILE_TYPE = pl.TileType(
            shape=[1, M_BASE_SIZE * D_BASE_SIZE], dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1],
            pad=pl.TilePad.zero)
        
        FLAT_CAST_TILE_TYPE = pl.TileType(
            shape=[1, M_BASE_SIZE * D_BASE_SIZE], dtype=ioDType,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
        
        TEMP_TILE_TYPE = pl.TileType(
            shape=[M_BASE_SIZE, D_BASE_SIZE // 2], dtype=ioDType,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1)

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
        # TODO:地址之后用常量
        dCmpKvTileGroup      = pl.make_tile_group(type=D_CMP_KV_TILE_TYPE, addrs=[0x00000], mutex_ids=[0])
        dCmpKvFP32TileGroup  = pl.make_tile_group(type=D_CMP_KV_FP32_TILE_TYPE,     addrs=[0x04000], mutex_ids=[1]) # 16kB = 0x4000

        kvTileGroup      = pl.make_tile_group(type=NORMEL_TILE_TYPE,  addrs=[0x0C000], mutex_ids=[2]) # 32kB = 0x8000
        softmaxScoreTileGroup      = pl.make_tile_group(type=NORMEL_TILE_TYPE,  addrs=[0x1C000], mutex_ids=[3]) # 64kb = 0x10000
        dKvTileGroup     = pl.make_tile_group(type=NORMEL_TILE_TYPE,  addrs=[0x2C000], mutex_ids=[4])

        dScoreCastTileGroup = pl.make_tile_group(type=CAST_TILE_TYPE, addrs=[0x1C000], mutex_ids=[3])
        dKvCastTileGroup = pl.make_tile_group(type=CAST_TILE_TYPE, addrs=[0x2C000], mutex_ids=[4])

        apeTileGroup = pl.make_tile_group(type=NORMEL_TILE_TYPE, addrs=[0x0C000], mutex_ids=[2]) # TODO:可以直接tile复用

        dScoreCastMDTileGroup = pl.make_tile_group(type=CAST_MD_TILE_TYPE, addrs=[0x1C000], mutex_ids=[3])
        dKvCastMDTileGroup = pl.make_tile_group(type=CAST_MD_TILE_TYPE, addrs=[0x2C000], mutex_ids=[4])

        dScoreNzTileGroup = pl.make_tile_group(type=NZ_TILE_TYPE, addrs=[0x24000], mutex_ids=[3])
        dKvNzTileGroup = pl.make_tile_group(type=NZ_TILE_TYPE, addrs=[0x34000], mutex_ids=[4])

        dScoreNzMDPrevTileGroup = pl.make_tile_group(type=NZ_MD_TILE_TYPE, addrs=[0x24000], mutex_ids=[3])
        dKvNzMDPrevTileGroup = pl.make_tile_group(type=NZ_MD_TILE_TYPE, addrs=[0x34000], mutex_ids=[4])
        dScoreNzMDCurTileGroup = pl.make_tile_group(type=NZ_MD_TILE_TYPE, addrs=[0x28000], mutex_ids=[3])
        dKvNzMDCurTileGroup = pl.make_tile_group(type=NZ_MD_TILE_TYPE, addrs=[0x38000], mutex_ids=[4])

        tempTileGroup  = pl.make_tile_group(type=TEMP_TILE_TYPE, addrs=0x00000, mutex_ids=[0, 1]) # 16kB = 0x4000

        moveXTileGroup = pl.make_tile_group(type=MOVE_X_TILE_TYPE,  addrs=[0x1C000, 0x2C000], mutex_ids=[3, 4])

        # Phase 3 reduce: 复用 0x0C000/0x1C000/0x2C000 三个 64KB 区域
        reduceAccTileGroup = pl.make_tile_group(type=FLAT_TILE_TYPE, addrs=[0x0C000], mutex_ids=[2])
        reduceLoadTileGroup = pl.make_tile_group(type=FLAT_TILE_TYPE, addrs=[0x1C000, 0x2C000], mutex_ids=[3, 4])

        # Phase 3 d_x reduce: acc/ld 复用 reduce_acc/reduce_ld0，仅新增 cast
        reduceResultTileGroup = pl.make_tile_group(type=FLAT_CAST_TILE_TYPE, addrs=[0x0C000], mutex_ids=[2])

        tile_dc_f16   = dCmpKvTileGroup.next()
        tile_dc_f32   = dCmpKvFP32TileGroup.next()
        tile_kv       = kvTileGroup.next()
        tile_softmax  = softmaxScoreTileGroup.next()
        tile_dsb_cast = dScoreCastTileGroup.next()
        tile_dsb_cast_md = dScoreCastMDTileGroup.next()
        tile_temp     = dKvTileGroup.next()
        tile_dkv_cast = dKvCastTileGroup.next()
        tile_dkv_cast_md = dKvCastMDTileGroup.next()
        tile_ape_local = apeTileGroup.next()
        tile_dkv_nz  = dKvNzTileGroup.next()
        tile_dsb_nz  = dScoreNzTileGroup.next()

        tile_dkv_nz_md_prev  = dKvNzMDPrevTileGroup.next()
        tile_dsb_nz_md_prev  = dScoreNzMDPrevTileGroup.next()
        tile_dkv_nz_md_cur  = dKvNzMDCurTileGroup.next()
        tile_dsb_nz_md_cur  = dScoreNzMDCurTileGroup.next()

        reduce_acc = reduceAccTileGroup.next()

        p3_cast = reduceResultTileGroup.next()

        # ── D group division ──
        cubeCoreIdx      = coreIdx // 2
        groupIdx         = cubeCoreIdx // groupSize
        intraGroupIdx    = cubeCoreIdx % groupSize

        nStart = intraGroupIdx * D_BASE_SIZE + (subCoreIdx % Coff) * dDealSize

        tensorApe = pl.make_tensor(workSpacePtr,
            [2 * groupNum * coffCoef * cmpRowCnt,  headDim],
            [headDim, 1], dtype=pl.DT_FP32)


        tensorDApeWsFlat = pl.make_tensor(workSpacePtr,
            [2 * groupNum * coffCoef, cmpSize],
            [cmpSize, 1], dtype=pl.DT_FP32)

        tensorDWKvWsFlat = pl.make_tensor(dWKvWorkSpacePtr,
            [dWeightWorkSpaceSize // (totalHeadDim * hiddenSize), totalHeadDim * hiddenSize],
            [totalHeadDim * hiddenSize, 1], dtype=pl.DT_FP32)
            
        tensorDWGateWsFlat = pl.make_tensor(dWGateWorkSpacePtr,
            [dWeightWorkSpaceSize // (totalHeadDim * hiddenSize), totalHeadDim * hiddenSize],
            [totalHeadDim * hiddenSize, 1], dtype=pl.DT_FP32)

        tensorDApeFlat = pl.make_tensor(d_ape, [1, cmpSize],
                                         [cmpSize, 1], dtype=pl.DT_FP32)
        tensorDWKvFlat = pl.make_tensor(d_wkv, [1, totalHeadDim * hiddenSize],
                                         [totalHeadDim * hiddenSize, 1], dtype=pl.DT_FP32)
        tensorDWGateFlat = pl.make_tensor(d_wgate, [1, totalHeadDim * hiddenSize],
                                         [totalHeadDim * hiddenSize, 1], dtype=pl.DT_FP32)

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
        # nEnd   = nStart + D_BASE_SIZE
        

        # L1A CV 数据: 复用主函数体 tile_group, 通过 .next() db

        # L1W: wkv/wgate chunk 加载（double-buffer, 0x40000-0x4FFFF, 64KB）
        l1_w_db = pl.make_tile_group(
            type=pl.TileType(shape=[128, D_BASE_SIZE * 2], dtype=ioDType,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ, compact=1),
            addrs=0x40000, mutex_ids=[20, 21])

        # L0 tiles
        cv_l0a = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=ioDType,
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
        wkv_t = pl.make_tensor(wkv, [totalHeadDim, hiddenSize],
                               [hiddenSize, 1], dtype=ioDType)
        wgate_t = pl.make_tensor(wgate, [totalHeadDim, hiddenSize],
                                 [hiddenSize, 1], dtype=ioDType)

        # d_x partial workspace (dkv@wkv) + dsb@wgate workspace


    # ════════════════════════════════════════════════════════════
    #  Phase 1+3: Vec Scatter & Reduce (per-round compute)
    # ════════════════════════════════════════════════════════════
    for roundIdx in pl.range(0, roundCnt):
        roundBlocks = min(totalScNumPerRound,
                          compressedCnt - roundIdx * totalScNumPerRound)
        dbIdx = roundIdx % 2
        
        with pl.section_vector():
            preDealTcSize = groupIdx * groupDealScNum
            if Coff == 1 and coreIdx % 2 == 1:
                preDealTcSize += dealScNum
            dealTcSize   = min(dealScNum,
                               max(0, roundBlocks - preDealTcSize))
            blocksAfter  = max(0, roundBlocks - preDealTcSize - dealTcSize)

            if preDealTcSize < roundBlocks:
                # Segment 1: skip
                bIdx, scIdx = _skip_blocks(preDealTcSize, bIdx, scIdx,
                                           start_pos, seqused, cu_seqlens, seq_ctx)

                # Segment 2: process
                processed = 0
                while processed < dealTcSize:
                    bStartPos = _get_start_pos(bIdx, start_pos, seq_ctx)
                    sqVal = _get_seq_used(bIdx, seqused, cu_seqlens, seq_ctx)
                    cmpLimit = (bStartPos + sqVal) // cmpRatio * cmpRatio
                    blkInBatch = (cmpLimit - bStartPos) // cmpRatio
                    remaining = blkInBatch - scIdx
                    if remaining > 0:
                        taken = min(dealTcSize - processed, remaining)
                        if Layout == 1:
                            globalBlkIdx = roundIdx * totalScNumPerRound + preDealTcSize + processed
                        else:
                            globalBlkIdx = bIdx * cmpKvBatchStride + scIdx

                        nTileRows = taken * cmpRowCnt

                        dcTile  = tile_dc_f16[processed:, :]
                        dcF32   = tile_dc_f32[processed:, :]
                        kvTile  = tile_kv[processed * cmpRowCnt:, :]
                        smTile  = tile_softmax[processed * cmpRowCnt:, :]
                        dkvTile = tile_temp[processed * cmpRowCnt:, :]

                        pl.set_validshape(dcTile,  [taken, dDealSize])
                        pl.set_validshape(kvTile,  [nTileRows, dDealSize])
                        pl.set_validshape(smTile,  [nTileRows, dDealSize])

                        
                        pl.load(dcTile, dCmpKvGm, [globalBlkIdx, nStart])
                        pl.load(kvTile, kvGm,      [globalBlkIdx, 0, nStart])
                        pl.load(smTile, softmaxScoreGm, [globalBlkIdx, 0, nStart])
                        # pl.printf("globalBlkIdx:%d, nStart:%d\n", globalBlkIdx, nStart)
                        # pl.dump_data(kvTile, offsets=[0, 0], shapes=[1, 64])
                        # pl.dump_data(smTile, offsets=[0, 0], shapes=[1, 64])
                        # pl.dump_data(dcTile, offsets=[0, 0], shapes=[1, 64])

                        pl.set_validshape(dcF32, [taken, dDealSize])
                        pl.cast(dcF32, dcTile, mode=pl.RoundMode.CAST_ROUND)

                        _vf_scatter_backward(
                            dcF32, kvTile, smTile, dkvTile,
                            taken, cmpRowCnt, dDealSize,
                        )

                        if Coff == 1:
                            cacheFillRows = bStartPos % cmpRatio if scIdx == 0 else 0
                            _vf_mask_cache_fill(
                                smTile,
                                dkvTile,
                                cacheFillRows, dDealSize, D_BASE_SIZE,
                            )
                        else:
                            if scIdx == 0:
                                cacheFillRows = bStartPos % cmpRatio + cmpRatio if taken >= 2 else cmpRatio
                            elif scIdx == 1:
                                cacheFillRows = bStartPos % cmpRatio
                            else:
                                cacheFillRows = 0
                            # pl.printf("cacheFillRows:%d\n", cacheFillRows)
                            _vf_mask_cache_fill(
                                smTile,
                                dkvTile,
                                cacheFillRows, dDealSize, D_BASE_SIZE,
                            )
                            cacheFillRows = bStartPos % cmpRatio if scIdx == 0 else 0
                            # pl.printf("cacheFillRows:%d\n", cacheFillRows)
                            _vf_mask_cache_fill(
                                smTile[1:, :],
                                dkvTile[1:, :],
                                cacheFillRows, dDealSize, D_BASE_SIZE,
                            )

                        processed += taken
                        scIdx += taken
                    if scIdx >= blkInBatch:
                        scIdx = 0
                        bIdx += 1
                    if bIdx >= batchSize:
                        break

                nTileRows = dealTcSize * cmpRowCnt

                # pl.dump_data(tile_softmax, offsets=[1, 0], shapes=[1, 64])
                # pl.dump_data(tile_temp, offsets=[1, 0], shapes=[1, 64])
                # APE: local_ape 累加 + 写 workspace
                pl.set_validshape(tile_ape_local, [cmpRowCnt, dDealSize])
                _vf_reduce_dscore_to_ape(
                    tile_softmax, tile_ape_local,
                    dealTcSize, cmpRowCnt, dDealSize, dDealSize,
                )
                # pl.dump_data(tile_ape_local, offsets=[0, 0], shapes=[1, 64])
                # pl.dump_data(tile_softmax, offsets=[4, 0], shapes=[1, 64])

                # ── Cast dkv/dsb FP32→BF16/FP16, move ND→NZ for CV ──
                # pl.dump_data(tile_temp, offsets=[0, 0], shapes=[1, 128])
                # pl.dump_data(tile_temp, offsets=[4, 0], shapes=[1, 128])
                pl.set_validshape(tile_dkv_cast, [nTileRows, dDealSize])
                pl.set_validshape(tile_temp,     [nTileRows, dDealSize])
                pl.cast(tile_dkv_cast, tile_temp, mode=pl.RoundMode.CAST_ROUND)
                if Coff == 1:
                    pl.set_validshape(tile_dkv_cast, [nTileRows, dDealSize])
                    pl.set_validshape(tile_dkv_nz, [nTileRows, dDealSize])
                    pl.move(tile_dkv_nz, tile_dkv_cast)
                else:
                    # pass
                    tempTile = tempTileGroup.next()
                    pl.set_validshape(tile_dkv_cast_md, [dealTcSize * cmpRatio, dDealSize])
                    pl.set_validshape(tile_dkv_nz_md_prev, [dealTcSize * cmpRatio, dDealSize])
                    pl.set_validshape(tempTile, [dealTcSize * cmpRatio, dDealSize])
                    # pl.printf("dealTcSize * cmpRatio:%d\n", dealTcSize * cmpRatio)
                    pl.move(tempTile, tile_dkv_cast_md, offset=[0, 0])
                    # pl.dump_data(tempTile, offsets=[4, 0], shapes=[1, 64])
                    pl.move(tile_dkv_nz_md_prev, tempTile)
                    tempTile = tempTileGroup.next()
                    pl.set_validshape(tile_dkv_cast_md[:, dDealSize:], [dealTcSize * cmpRatio, dDealSize])
                    pl.set_validshape(tile_dkv_nz_md_cur, [dealTcSize * cmpRatio, dDealSize])
                    pl.set_validshape(tempTile, [dealTcSize * cmpRatio, dDealSize])
                    pl.move(tempTile, tile_dkv_cast_md, offset=[0, dDealSize])
                    # pl.dump_data(tempTile, offsets=[0, 0], shapes=[1, 64])
                    pl.move(tile_dkv_nz_md_cur, tempTile)


                # tile_dkv_nz_md_prev  = dKvNzMDPrevTileGroup.next()
                # tile_dsb_nz_md_prev  = dScoreNzMDPrevTileGroup.next()
                # tile_dkv_nz_md_cur  = dKvNzMDCurTileGroup.next()
                # tile_dsb_nz_md_cur  = dScoreNzMDCurTileGroup.next()

                pl.set_validshape(tile_dsb_cast, [nTileRows, dDealSize])
                pl.set_validshape(tile_softmax,  [nTileRows, dDealSize])
                pl.cast(tile_dsb_cast, tile_softmax, mode=pl.RoundMode.CAST_ROUND)
                if Coff == 1:
                    pl.set_validshape(tile_dsb_nz, [nTileRows, dDealSize])
                    pl.set_validshape(tile_dsb_cast, [nTileRows, dDealSize])
                    pl.move(tile_dsb_nz, tile_dsb_cast)
                else:
                    # pass
                    tempTile = tempTileGroup.next()
                    pl.set_validshape(tile_dsb_cast_md, [dealTcSize * cmpRatio, dDealSize])
                    pl.set_validshape(tile_dsb_nz_md_prev, [dealTcSize * cmpRatio, dDealSize])
                    pl.set_validshape(tempTile, [dealTcSize * cmpRatio, dDealSize])
                    pl.move(tempTile, tile_dsb_cast_md, offset=[0, 0])
                    # pl.dump_data(tempTile, offsets=[4, 0], shapes=[1, 64])
                    pl.move(tile_dsb_nz_md_prev, tempTile)
                    tempTile = tempTileGroup.next()
                    pl.set_validshape(tile_dsb_cast_md[:, dDealSize:], [dealTcSize * cmpRatio, dDealSize])
                    pl.set_validshape(tile_dsb_nz_md_cur, [dealTcSize * cmpRatio, dDealSize])
                    pl.set_validshape(tempTile, [dealTcSize * cmpRatio, dDealSize])
                    pl.move(tempTile, tile_dsb_cast_md, offset=[0, dDealSize])
                    # pl.dump_data(tempTile, offsets=[0, 0], shapes=[1, 64])
                    pl.move(tile_dsb_nz_md_cur, tempTile)


                dbBase = dbIdx * groupNum * coffCoef * cmpRowCnt
                localSlot = groupIdx * coffCoef * cmpRowCnt
                if Coff == 1 and coreIdx % 2 == 1:
                    localSlot += cmpRowCnt

                pl.store(tensorApe, tile_ape_local, [dbBase + localSlot, nStart])
                # pl.printf("dbBase:%d, localSlot:%d, nStart:%d\n", dbBase, localSlot, nStart)
                # pl.dump_data(tensorApe, offsets=[dbBase + localSlot, nStart], shapes=[1, 128])
                # ── CV 通路: Vec UB(NZ) → L1 Mat(NZ) (coff=1 only) ──
                sub_idx = coreIdx % 2
                if Coff == 1:
                    l1_kv = L1Kv0TileGroupNz.next() if sub_idx == 0 else L1Kv1TileGroupNz.next()
                    l1_sb = L1Score0TileGroupNz.next() if sub_idx == 0 else L1Score1TileGroupNz.next()
                    pl.set_validshape(l1_kv, [nTileRows, dDealSize])
                    pl.set_validshape(l1_sb, [nTileRows, dDealSize])
                    pl.insert(l1_kv, tile_dkv_nz, [0, 0])
                    pl.insert(l1_sb, tile_dsb_nz, [0, 0])
                else:
                    pass
                    l1_kv0 = L1Kv0TileGroupNz.next()
                    l1_kv1 = L1Kv1TileGroupNz.next()
                    l1_sb0 = L1Score0TileGroupNz.next()
                    l1_sb1 = L1Score1TileGroupNz.next()
                    colOffset = 0 if sub_idx == 0 else dDealSize
                    pl.set_validshape(l1_kv0, [dealTcSize * cmpRatio, D_BASE_SIZE])
                    pl.set_validshape(l1_kv1, [dealTcSize * cmpRatio, D_BASE_SIZE])
                    pl.set_validshape(l1_sb0, [dealTcSize * cmpRatio, D_BASE_SIZE])
                    pl.set_validshape(l1_sb1, [dealTcSize * cmpRatio, D_BASE_SIZE])
                    pl.insert(l1_kv0, tile_dkv_nz_md_prev, [0, colOffset])
                    pl.insert(l1_kv1, tile_dkv_nz_md_cur, [0, colOffset])
                    pl.insert(l1_sb0, tile_dsb_nz_md_prev, [0, colOffset])
                    pl.insert(l1_sb1, tile_dsb_nz_md_cur, [0, colOffset])

                bIdx, scIdx = _skip_blocks(blocksAfter, bIdx, scIdx,
                                        start_pos, seqused, cu_seqlens, seq_ctx)


                # 搬运x
                mxbIdxStart = mxbIdx
                mxscIdxStart = mxscIdx
                # pl.printf("mxbIdxStart:%d, mxscIdxStart:%d\n", mxbIdxStart, mxscIdxStart)
                mxbIdx, mxscIdx = _skip_blocks(preDealTcSize, mxbIdx, mxscIdx,
                                            start_pos, seqused, cu_seqlens, seq_ctx)
                # pl.printf("mxbIdxStart:%d, mxscIdxStart:%d\n", mxbIdx, mxscIdx)

                # pl.printf("mxbIdxStart:%d, mxscIdxStart:%d\n", mxbIdxStart, mxscIdxStart)
                dbRowCnt = groupNum * cubeMBaseSize + (Coff - 1) * cmpRatio
                if Coff == 2 and groupIdx == 0 and mxscIdx != 0:
                    firstSrcIdx = ((dbIdx + dbRatio - 1) % dbRatio) * dbRowCnt + dbRowCnt - cmpRatio
                    firstDstIdx = dbIdx * dbRowCnt
                    for hi in pl.range(0, hiddenSize, 256):
                        mxTie = moveXTileGroup.next()
                        pl.set_validshape(mxTie,  [cmpRatio, 256])
                        pl.load(mxTie, xArrangeGm, [firstSrcIdx, hi])
                        pl.store(xArrangeGm, mxTie, [firstDstIdx, hi])


                processed = 0
                while processed < dealTcSize:   
                    bStartPos = _get_start_pos(mxbIdx, start_pos, seq_ctx)
                    bSeqUsed = _get_seq_used(mxbIdx, seqused, cu_seqlens, seq_ctx)
                    cmpLimit = (bStartPos + bSeqUsed) // cmpRatio * cmpRatio
                    blkInBatch = (cmpLimit - bStartPos) // cmpRatio
                    remaining = blkInBatch - mxscIdx
                    if remaining > 0:
                        taken = min(dealTcSize - processed, remaining)
                        dstIdx = dbIdx * dbRowCnt + (preDealTcSize + processed) * cmpRatio + (Coff - 1) * cmpRatio
                        mxsIdx = (0 if mxscIdx == 0
                                                else mxscIdx * cmpRatio - (bStartPos % cmpRatio))
                        srcIdx = _get_token_idx(mxbIdx, mxsIdx, cu_seqlens, seq_ctx)
                        mxRows = taken * cmpRatio
                        if mxsIdx == 0:
                            mxRows -= bStartPos % cmpRatio
                            dstIdx += bStartPos % cmpRatio
                        # pl.printf("srcIdx:%d, dstIdx:%d, mxRows:%d", srcIdx, dstIdx, mxRows)
                        for hi in pl.range(0, hiddenSize, 256):
                            mxTie = moveXTileGroup.next()
                            pl.set_validshape(mxTie,  [mxRows, 256])
                            pl.load(mxTie, xGm, [srcIdx, hi])
                            pl.store(xArrangeGm, mxTie, [dstIdx, hi])

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
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=dbIdx,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)


            
            # ── Phase 3: d_ape cross-core reduce + RMW ──
            # d_ape: [cmpRatio, Coff*headDim]. Process per row-group of Coff rows.
            flatRowBase = dbIdx * groupNum * (2 // Coff) 
            rgChunkNum = cmpSize // 128 # 按128分
            rgPerCore = rgChunkNum // coreNum 
            rgRemain = rgChunkNum % coreNum
            rgStart = coreIdx * rgPerCore + (coreIdx if coreIdx < rgRemain else rgRemain)
            rgEnd = rgStart + rgPerCore + (1 if coreIdx < rgRemain else 0)
            # pl.printf("[Mo]Coff:%d, cmpRatio:%d, headDim:%d, coreNum:%d, coreIdx:%d\n", Coff, cmpRatio, headDim, coreNum,coreIdx)
            # pl.printf("[Mo]rgChunkNum:%d, rgPerCore:%d, rgRemain:%d\n", rgChunkNum, rgPerCore, rgRemain)
            rgStart = rgStart * 128
            rgEnd = rgEnd * 128
            validRows = (roundBlocks + groupDealScNum - 1) // groupDealScNum * (2 // Coff)
            pl.set_validshape(reduce_acc, [1, rgEnd - rgStart])
            # pl.printf("[Mo]rgStart:%d, rgEnd:%d, rgEnd - rgStart:%d\n", rgStart, rgEnd, rgEnd - rgStart)
            if roundIdx == 0:
                _vf_tile_zero(reduce_acc, 1, (rgEnd - rgStart))
            else:
                pl.load(reduce_acc, tensorDApeFlat, [0, rgStart])
            for r in pl.range(0, validRows):
                reduce_ld = reduceLoadTileGroup.next()
                pl.set_validshape(reduce_ld, [1, rgEnd - rgStart]) 
                pl.load(reduce_ld, tensorDApeWsFlat, [flatRowBase + r, rgStart])
                # pl.dump_data(tensorDApeWsFlat, offsets=[flatRowBase + r, rgStart], shapes=[1, 128])
                # pl.printf("[Mo]flatRowBase:%d, r:%d, rgStart:%d\n", flatRowBase, r, rgStart)
                
                pl.add(reduce_acc, reduce_acc, reduce_ld)
            pl.store(tensorDApeFlat, reduce_acc, [0, rgStart])

            # ── Phase 3: d_x reduce ──
            p3bIdxEnd, p3scIdxEnd = _skip_blocks(roundBlocks, p3bIdx, p3scIdx,
                                           start_pos, seqused, cu_seqlens, seq_ctx)
            p3bIdxStart, p3scIdxStart = p3bIdx, p3scIdx
            p3bIdx, p3scIdx = p3bIdxEnd, p3scIdxEnd
            prevScIdx = 0
            if Coff == 2 and p3scIdxStart != 0:
                p3scIdxStart -= 1
                prevScIdx += 1
            
            bStart = p3bIdxStart
            bEnd   = p3bIdxEnd - 1 if p3scIdxEnd == 0 else p3bIdxEnd
            sStart = (0 if p3scIdxStart == 0
                      else p3scIdxStart * cmpRatio - (_get_start_pos(bStart, start_pos, seq_ctx) % cmpRatio))
            sEnd   = (_get_seq_length(bEnd, cu_seqlens, seq_ctx) if p3scIdxEnd == 0
                      else p3scIdxEnd * cmpRatio - (_get_start_pos(bEnd, start_pos, seq_ctx) % cmpRatio))
            if roundIdx == roundCnt - 1:
                bEnd = batchSize - 1
                sEnd = _get_seq_length(bEnd, cu_seqlens, seq_ctx)
            totalRows = _get_row_count(bStart, sStart, bEnd, sEnd, cu_seqlens, seq_ctx)

            rowsPerCore = totalRows // coreNum
            rowsRemain  = totalRows % coreNum
            myRowS = coreIdx * rowsPerCore + (coreIdx if coreIdx < rowsRemain else rowsRemain)
            myRowE = myRowS + rowsPerCore + (1 if coreIdx < rowsRemain else 0)
            roundEndScIdx = _get_cmp_block_count(bEnd, sEnd, start_pos, seqused, cu_seqlens, seq_ctx) - \
                             _get_cmp_block_count(bStart, sStart, start_pos, seqused, cu_seqlens, seq_ctx) - 1

            # pl.printf("myRowS:%d, myRowE:%d\n", myRowS, myRowE)
            if myRowS < myRowE:
                # ── 从 (bStart, sStart) 偏移 myRowS 行到达本核起始位置 ──
                startTidx = _get_token_idx(bStart, sStart, cu_seqlens, seq_ctx)
                b, localS = _get_pos_from_token_idx(startTidx + myRowS, cu_seqlens, seq_ctx)
                tIdx = startTidx + myRowS
                bStartPos = _get_start_pos(b, start_pos, seq_ctx)
                bSeqUsed = _get_seq_used(b, seqused, cu_seqlens, seq_ctx)
                bSeqLength = _get_seq_length(b, cu_seqlens, seq_ctx)
                cmpLimit = (bStartPos + bSeqUsed) // cmpRatio * cmpRatio - bStartPos
                p3lScIdx = _get_cmp_block_count(b, localS, start_pos, seqused, cu_seqlens, seq_ctx) - \
                             _get_cmp_block_count(bStart, sStart, start_pos, seqused, cu_seqlens, seq_ctx)  
                if localS >= cmpLimit and cmpLimit > 0:                                                                                                                                                                                                                                                         
                    p3lScIdx -= 1 
                batchEndScIdx = _get_cmp_block_count(b, bSeqUsed, start_pos, seqused, cu_seqlens, seq_ctx) - \
                             _get_cmp_block_count(bStart, sStart, start_pos, seqused, cu_seqlens, seq_ctx) - 1
                scInnerIdx = (localS + bStartPos) % cmpRatio

                for rowIdx in pl.range(myRowS, myRowE):
                    # ── 判断是否参与压缩 ──
                    inCompress = localS < cmpLimit

                    if inCompress:
                        if p3lScIdx < prevScIdx: # TODO：只处理prev，需要从额外的cache里读取
                            ws_offset = dbIdx * cubeCoreNum * cubeMBaseSize + scInnerIdx
                            pl.set_validshape(reduce_acc, [1, hiddenSize])
                            pl.load(reduce_acc, dXCacheGm, [dbIdx * cmpRatio + scInnerIdx, 0])
                            for r in pl.range(0, groupSize):
                                reduce_ld = reduceLoadTileGroup.next()
                                pl.set_validshape(reduce_ld, [1, hiddenSize]) 
                                pl.load(reduce_ld, dXResultGm, [0, ws_offset + r * cubeMBaseSize, 0])
                                pl.add(reduce_acc, reduce_acc, reduce_ld)
                        else:
                            ws_offset = dbIdx * cubeCoreNum * cubeMBaseSize + ((p3lScIdx - prevScIdx) // groupDealScNum) * groupSize * cubeMBaseSize + ((p3lScIdx - prevScIdx) % groupDealScNum) * cmpRatio + scInnerIdx
                            pl.set_validshape(reduce_acc, [1, hiddenSize])
                            pl.load(reduce_acc, dXResultGm, [Coff - 1, ws_offset, 0])
                            if groupSize > 1:
                                for r in pl.range(1, groupSize):
                                    reduce_ld = reduceLoadTileGroup.next()
                                    pl.set_validshape(reduce_ld, [1, hiddenSize]) 
                                    pl.load(reduce_ld, dXResultGm, [Coff - 1, ws_offset + r * cubeMBaseSize, 0])
                                    pl.add(reduce_acc, reduce_acc, reduce_ld)

                            if Coff == 2 and p3lScIdx != roundEndScIdx and p3lScIdx != batchEndScIdx:
                                pl.set_validshape(reduce_acc, [1, hiddenSize])
                                next_ws_offset = dbIdx * cubeCoreNum * cubeMBaseSize + ((p3lScIdx + 1 - prevScIdx) // groupDealScNum) * groupSize * cubeMBaseSize + ((p3lScIdx + 1 - prevScIdx) % groupDealScNum) * cmpRatio + scInnerIdx
                                for r in pl.range(0, groupSize):
                                    reduce_ld = reduceLoadTileGroup.next()
                                    pl.set_validshape(reduce_ld, [1, hiddenSize]) 
                                    pl.load(reduce_ld, dXResultGm, [0, next_ws_offset + r * cubeMBaseSize, 0])
                                    pl.add(reduce_acc, reduce_acc, reduce_ld)

                        if Coff == 2 and p3lScIdx == roundEndScIdx and p3lScIdx != batchEndScIdx:
                            pl.store(dXCacheGm, reduce_acc, [(dbIdx + 1) % dbRatio * cmpRatio + scInnerIdx, 0])

                        else:
                            pl.set_validshape(p3_cast, [1, hiddenSize])
                            pl.cast(p3_cast, reduce_acc, mode=pl.RoundMode.CAST_ROUND)
                            pl.store(dXOutGm, p3_cast, [tIdx, 0])

                    else:
                        if p3lScIdx < prevScIdx:
                            continue
                        # ── 不参与压缩: p3_cast 直接刷零，store ──
                        pl.set_validshape(p3_cast, [1, hiddenSize])
                        if DataType == 0:
                            _vf_tile_zero_bf16(p3_cast, 1, hiddenSize)
                        else:
                            _vf_tile_zero_f16(p3_cast, 1, hiddenSize)
                        pl.store(dXOutGm, p3_cast, [tIdx, 0])

                    # ── 推进到下一行 ──
                    tIdx += 1
                    localS += 1
                    if localS >= bSeqLength:
                        localS = 0
                        b += 1
                        if b >= batchSize:
                            break
                        bStartPos = _get_start_pos(b, start_pos, seq_ctx)
                        bSeqUsed = _get_seq_used(b, seqused, cu_seqlens, seq_ctx)
                        bSeqLength = _get_seq_length(b, cu_seqlens, seq_ctx)
                        cmpLimit = (bStartPos + bSeqUsed) // cmpRatio * cmpRatio - bStartPos
                        batchEndScIdx = _get_cmp_block_count(b, bSeqUsed, start_pos, seqused, cu_seqlens, seq_ctx) - \
                             _get_cmp_block_count(bStart, sStart, start_pos, seqused, cu_seqlens, seq_ctx) - 1
                    scInnerIdx = (localS + bStartPos) % cmpRatio
                    if (localS < cmpLimit) and (localS == 0 or scInnerIdx == 0):
                        p3lScIdx += 1

            # pl.printf("@!#\n")
            # ── Phase 3: d_w reduce ──
            flatRowBase = dbIdx * groupNum
            rgChunkNum = (totalHeadDim * hiddenSize) // 128
            rgPerCore = rgChunkNum // coreNum 
            rgRemain = rgChunkNum % coreNum
            rgStart = coreIdx * rgPerCore + (coreIdx if coreIdx < rgRemain else rgRemain)
            rgEnd = rgStart + rgPerCore + (1 if coreIdx < rgRemain else 0)
            # pl.printf("[Mo]Coff:%d, cmpRatio:%d, headDim:%d, coreNum:%d, coreIdx:%d\n", Coff, cmpRatio, headDim, coreNum,coreIdx)
            # pl.printf("[Mo]rgChunkNum:%d, rgPerCore:%d, rgRemain:%d\n", rgChunkNum, rgPerCore, rgRemain)
            rgStart = rgStart * 128
            rgEnd = rgEnd * 128
            rgNum = rgEnd - rgStart
            validRows = (roundBlocks + groupDealScNum - 1) // groupDealScNum
            # pl.printf("roundBlocks:%d, groupDealScNum:%d\n", roundBlocks, groupDealScNum)
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
                    reduce_ld = reduceLoadTileGroup.next()
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
                    reduce_ld = reduceLoadTileGroup.next()
                    pl.set_validshape(reduce_ld, [1, curRgNum]) 
                    pl.load(reduce_ld, tensorDWGateWsFlat, [flatRowBase + r, curRgStart])
                    pl.add(reduce_acc, reduce_acc, reduce_ld)
                pl.store(tensorDWGateFlat, reduce_acc, [0, curRgStart])
            
        # ════════════════════════════════════════════════════════════
        #  Phase 2: Cube Matmul — dkv@wkv + dsb@wgate → d_x_partial
        # ════════════════════════════════════════════════════════════
        with pl.section_cube():
            preDealTcSize = groupIdx * groupDealScNum
            dealTcSize   = min(groupDealScNum,
                               max(0, roundBlocks - preDealTcSize))
            pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE1, event_id=dbIdx,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            if preDealTcSize >= roundBlocks:
                # coff=2 CV 路径 L1 布局不同，暂跳过
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
                continue


            l1_kv0 = L1Kv0TileGroupNz.next()
            l1_kv1 = L1Kv1TileGroupNz.next()
            l1_sb0 = L1Score0TileGroupNz.next()
            l1_sb1 = L1Score1TileGroupNz.next()

            l1_kv0_t = L1Kv0TileGroupZn.next()
            l1_kv1_t = L1Kv1TileGroupZn.next()
            l1_sb0_t = L1Score0TileGroupZn.next()
            l1_sb1_t = L1Score1TileGroupZn.next()

            # d_x partial 写入行偏移
            dx_row_base = (dbIdx * cubeCoreNum + cubeCoreIdx) * cubeMBaseSize

            if Coff == 1:
                dealTcSize0 = min(dealTcSize, dealScNum)
                dealTcSize1 = max(dealTcSize - dealScNum, 0)
                nTileRows0 = dealTcSize0 * cmpRatio
                nTileRows1 = dealTcSize1 * cmpRatio
            else:
                nTileRows0 = dealTcSize * cmpRatio
                nTileRows1 = dealTcSize * cmpRatio

            # pl.printf("nStart :%d\n", nStart)
            # ── Matmul #1: dkv @ wkv → d_x partial ──
            if Coff == 1:
                for ni in pl.range(0, hiddenSize, 256):
                    for mi in pl.range(0, 4):
                        nTileRows = nTileRows0 if mi % 2 == 0 else nTileRows1
                        # pl.printf("mi:%d, nTileRows:%d\n", mi, nTileRows)
                        m_off = (mi % 2) * 128
                        l0a = cv_l0a.next()
                        pl.set_validshape(l0a, [nTileRows, D_BASE_SIZE])
                        w_l1 = l1_w_db.next() if (mi % 2 == 0 or Coff == 2) else l1_w_db.current()
                        dOffset = 0 if Coff == 1 else headDim 
                        if mi == 0:
                            pl.set_validshape(l1_kv0, [nTileRows, D_BASE_SIZE])
                            pl.move(l0a, l1_kv0)
                            pl.set_validshape(w_l1, [128, 256])
                            pl.load(w_l1, wkv_t, [nStart, ni])

                            # pl.system.bar_all()
                        elif mi == 1:
                            pl.set_validshape(l1_kv1, [nTileRows, D_BASE_SIZE])
                            pl.move(l0a, l1_kv1)
                        elif mi == 2:
                            pl.set_validshape(l1_sb0, [nTileRows, D_BASE_SIZE])
                            pl.move(l0a, l1_sb0)
                            pl.set_validshape(w_l1, [128, 256])
                            # pl.system.bar_all()
                            pl.load(w_l1, wgate_t, [nStart + dOffset, ni])
                            # pl.system.bar_all()
                        else:
                            pl.set_validshape(l1_sb1, [nTileRows, D_BASE_SIZE])
                            pl.move(l0a, l1_sb1)

                        # pl.system.bar_all()
                        for nii in pl.range(0, 256, 128):
                            l0b = cv_l0b.next()
                            l0c = cv_l0c.next()
                            pl.set_validshape(l0b, [D_BASE_SIZE, 128])
                            pl.set_validshape(l0c, [nTileRows, 128])
                            pl.move(l0b, w_l1, offset=[0, nii])
                            if mi // 2 == 0:
                                pl.matmul(l0c, l0a, l0b)
                            else:
                                if nTileRows > 0:
                                    pl.matmul_acc(l0c, l0c, l0a, l0b)
                                pl.store(dXResultGm, l0c, [0, dx_row_base + m_off, ni+nii])

            else:
                for ni in pl.range(0, hiddenSize, 256):
                    for mi in pl.range(0, 4):
                        nTileRows = nTileRows0 if mi % 2 == 0 else nTileRows1
                        l0a = cv_l0a.next()
                        pl.set_validshape(l0a, [nTileRows, D_BASE_SIZE])
                        w_l1 = l1_w_db.next() 
                        dOffset = headDim 
                        if mi == 0:
                            pl.set_validshape(l1_kv0, [nTileRows, D_BASE_SIZE])
                            pl.move(l0a, l1_kv0)

                            # pl.system.bar_all()
                        elif mi == 1:
                            pl.set_validshape(l1_kv1, [nTileRows, D_BASE_SIZE])
                            pl.move(l0a, l1_kv1)
                        elif mi == 2:
                            pl.set_validshape(l1_sb0, [nTileRows, D_BASE_SIZE])
                            pl.move(l0a, l1_sb0)
                            # pl.system.bar_all()
                            # pl.system.bar_all()
                        else:
                            pl.set_validshape(l1_sb1, [nTileRows, D_BASE_SIZE])
                            pl.move(l0a, l1_sb1)
                            # pl.set_validshape(w_l1, [128, 128])
                            # pl.load(w_l1, wgate_t, [nStart + headDim, ni])

                        pl.set_validshape(w_l1, [128, 256])
                        if mi // 2 == 0:
                            pl.load(w_l1, wkv_t, [nStart + (mi % 2) * headDim, ni])
                        else:
                            pl.load(w_l1, wgate_t, [nStart + (mi % 2) * headDim, ni])
                        for nii in pl.range(0, 256, 128):
                            l0b = cv_l0b.next()
                            l0c = cv_l0c.next()
                            pl.set_validshape(l0b, [D_BASE_SIZE, 128])
                            pl.set_validshape(l0c, [nTileRows, 128])
                            pl.move(l0b, w_l1, offset=[0, nii])
                            if mi // 2 == 0:
                                pl.matmul(l0c, l0a, l0b)
                            else:
                                pl.matmul_acc(l0c, l0c, l0a, l0b)
                                pl.store(dXResultGm, l0c, [mi % 2, dx_row_base, ni+nii])
                                # if ni + nii == 128 and mi % 2 == 1:
                                #     pl.dump_data(dXResultGm, offsets=[mi % 2, dx_row_base, ni+nii], shapes = [1, 5, 128])
                                # elif ni + nii == 0 and mi % 2 == 0:
                                #     pl.dump_data(dXResultGm, offsets=[mi % 2, dx_row_base, ni+nii], shapes = [1, 5, 128])


            if Coff == 1:
                dw_row_base = dbIdx * groupNum * totalHeadDim + groupIdx * totalHeadDim + intraGroupIdx * D_BASE_SIZE
                mStart = dbIdx * groupNum * cubeMBaseSize + groupIdx * cubeMBaseSize
                # ── Matmul #2: dkv @ x → d_wkv & dscore @ x → d_wgate partial ──
                for ni in pl.range(0, hiddenSize, 256):
                    for ii in pl.range(0, 8):
                        ki = ii // 4 # 0, 0, 0, 0, 1, 1, 1, 1
                        nii = (ii % 4 // 2) * 128 # 0, 0, 128, 128, 0, 0, 128, 128
                        mi = ii % 2 # 0, 1, 0, 1, 0, 1, 0, 1

                        nTileRows = nTileRows0 if ki == 0 else nTileRows1
                        l0a = cv_l0a.next()
                        pl.set_validshape(l0a, [D_BASE_SIZE, nTileRows])

                        # m_off = (mi % 2) * 128
                        w_l1 = l1_w_db.next() if ii % 4 == 0 else l1_w_db.current()

                        if ii == 0:
                            pl.set_validshape(l1_kv0_t, [D_BASE_SIZE, nTileRows])
                            pl.move(l0a, l1_kv0_t)
                            pl.set_validshape(w_l1, [nTileRows, 256])
                            pl.load(w_l1, xArrangeGm, [mStart, ni])

                        elif ii == 1:
                            pl.set_validshape(l1_sb0_t, [D_BASE_SIZE, nTileRows])
                            pl.move(l0a, l1_sb0_t)
                        elif ii == 4:
                            pl.set_validshape(l1_kv1_t, [D_BASE_SIZE, nTileRows])
                            pl.move(l0a, l1_kv1_t)
                            pl.set_validshape(w_l1, [nTileRows, 256])
                            pl.load(w_l1, xArrangeGm, [mStart + 128, ni])
                        elif ii == 5:
                            pl.set_validshape(l1_sb1_t, [D_BASE_SIZE, nTileRows])
                            pl.move(l0a, l1_sb1_t)

                        l0b = cv_l0b.next() if mi == 0 else cv_l0b.current()
                        l0c = cv_l0c.next()
                        pl.set_validshape(l0b, [nTileRows, 128])
                        pl.set_validshape(l0c, [D_BASE_SIZE, 128])
                        pl.move(l0b, w_l1, offset=[0, nii])
                        if ki == 0:
                            pl.matmul(l0c, l0a, l0b)
                        else:
                            if nTileRows > 0:
                                pl.matmul_acc(l0c, l0c, l0a, l0b)
                            if mi == 0:
                                pl.store(dWkvResultGm, l0c, [dw_row_base, ni+nii])
                            else:
                                pl.store(dWGateResultGm, l0c, [dw_row_base, ni+nii])

            else:
                dw_row_base = dbIdx * groupNum * totalHeadDim + groupIdx * totalHeadDim + intraGroupIdx * D_BASE_SIZE
                dbRowCnt = groupNum * cubeMBaseSize + (Coff - 1) * cmpRatio
                mStart = dbIdx * dbRowCnt + groupIdx * cubeMBaseSize + cmpRatio
                # ── Matmul #2: dkv @ x → d_wkv & dscore @ x → d_wgate partial ──
                for ni in pl.range(0, hiddenSize, 256):
                    for ii in pl.range(0, 8):
                        ki = ii // 4 # 0, 0, 0, 0, 1, 1, 1, 1
                        nii = (ii % 4 // 2) * 128 # 0, 0, 128, 128, 0, 0, 128, 128
                        mi = ii % 2 # 0, 1, 0, 1, 0, 1, 0, 1

                        nTileRows = nTileRows0 if ki == 0 else nTileRows1
                        l0a = cv_l0a.next()
                        pl.set_validshape(l0a, [D_BASE_SIZE, nTileRows])

                        # m_off = (mi % 2) * 128
                        w_l1 = l1_w_db.next() if ii % 4 == 0 else l1_w_db.current()

                        if ii == 0:
                            pl.set_validshape(l1_kv0_t, [D_BASE_SIZE, nTileRows])
                            pl.move(l0a, l1_kv0_t)
                            pl.set_validshape(w_l1, [nTileRows, 256])
                            pl.load(w_l1, xArrangeGm, [mStart - cmpRatio, ni])

                        elif ii == 1:
                            pl.set_validshape(l1_sb0_t, [D_BASE_SIZE, nTileRows])
                            pl.move(l0a, l1_sb0_t)
                        elif ii == 4:
                            pl.set_validshape(l1_kv1_t, [D_BASE_SIZE, nTileRows])
                            pl.move(l0a, l1_kv1_t)
                            pl.set_validshape(w_l1, [nTileRows, 256])
                            pl.load(w_l1, xArrangeGm, [mStart, ni])
                        elif ii == 5:
                            pl.set_validshape(l1_sb1_t, [D_BASE_SIZE, nTileRows])
                            pl.move(l0a, l1_sb1_t)

                        l0b = cv_l0b.next() if mi == 0 else cv_l0b.current()
                        l0c = cv_l0c.next()
                        pl.set_validshape(l0b, [nTileRows, 128])
                        pl.set_validshape(l0c, [D_BASE_SIZE, 128])
                        pl.move(l0b, w_l1, offset=[0, nii])
                        pl.matmul(l0c, l0a, l0b)
                        if mi == 0:
                            pl.store(dWkvResultGm, l0c, [dw_row_base + ki * headDim, ni+nii])
                        else:
                            pl.store(dWGateResultGm, l0c, [dw_row_base + ki * headDim, ni+nii])

            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
