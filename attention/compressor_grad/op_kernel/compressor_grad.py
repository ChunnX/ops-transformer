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
import torch, torch_npu
import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField as tilingkey_field

# ================================================================
#  Compile-time constants
# ================================================================
FLOAT_REG_ELEMENTS = 64   # FP32 VF register holds 64 elements
D_BASE_SIZE = 128
H_L1_BASE_SIZE = 256   # mm1/mm2 hidden 维 L1 分块
BIT16_REG_ELEMENTS = 128   # FP16/BF16 VF 寄存器元素数
BASE_SIZE = 128            # L0/L1 通用基础维度
H_BASE_SIZE = 128      # hidden 维子块
M_BASE_SIZE = 128

# ================================================================
#  UB / L1 地址常量（tile_group 声明）
# ================================================================
UB_DC_F16_ADDR         = 0x00000   # dc_f16 / temp
UB_DC_F32_ADDR         = 0x04000   # dc_f32
UB_KV_APE_ADDR         = 0x0C000   # kv / ape_local / reduce_acc / p3_cast
UB_SM_DSB_ADDR         = 0x1C000   # softmax / dsb_cast / moveX0 / reduce_ld0
UB_DSB_NZ_ADDR         = 0x24000   # dsb_nz / dsb_nz_md_prev
UB_DSB_NZ_MD_CUR_ADDR  = 0x28000   # dsb_nz_md_cur
UB_DKV_ADDR            = 0x2C000   # dkv / dkv_cast / moveX1 / reduce_ld1
UB_DKV_NZ_ADDR         = 0x34000   # dkv_nz / dkv_nz_md_prev
UB_DKV_NZ_MD_CUR_ADDR  = 0x38000   # dkv_nz_md_cur
L1_KV0_ADDR            = 0x00000   # L1Kv0 (NZ/ZN)
L1_KV1_ADDR            = 0x10000   # L1Kv1 (NZ/ZN)
L1_SB0_ADDR            = 0x20000   # L1Score0 (NZ/ZN)
L1_SB1_ADDR            = 0x30000   # L1Score1 (NZ/ZN)
L1_W_ADDR              = 0x40000   # l1_w_db

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
    batchSize:  int = 1
    tokenSize:  int = 0
    seqSize:    int = 0
    cmpRatio:   int = 4
    hiddenSize: int = 512
    headDim:    int = 128
    # ── 以下为 host 计算的派生配置（kernel 运行期不变，单一来源）──
    cubeCoreNum:        int = 0     # Cube 核数 = launch block_dim（替代 pl.get_block_num()）
    coreNum:            int = 0     # Vec 逻辑核数 = cubeCoreNum * subBlockNum(2)
    totalHeadDim:       int = 0     # Coff * headDim
    cmpRowCnt:          int = 0     # Coff * cmpRatio
    cmpSize:            int = 0     # Coff * cmpRatio * headDim
    cmpKvBatchStride:   int = 0     # ceil_div(seqSize, cmpRatio)
    cmpKvRows:          int = 0     # dCmpKv/kv/sm 行数（= host outputRows）
    xRows:              int = 0     # x/dX 行数
    groupSize:          int = 1     # headDim // D_BASE_SIZE
    groupNum:           int = 0     # cubeCoreNum // groupSize
    groupDealScNum:     int = 0     # cubeMBaseSize // cmpRatio
    dealScNum:          int = 0     # M_BASE_SIZE // cmpRatio
    totalScNumPerRound: int = 0     # groupNum * groupDealScNum
    dbRowCnt:           int = 0     # groupNum * groupRowStride（xArrangeGm 每 db 区域行数）
    groupRowStride:     int = 0     # 每 group 实际行数 = groupDealScNum*cmpRatio + (Coff-1)*cmpRatio
    # ── 编译期派生（TilingKey 折叠值，host 同步传入；TileType 静态 shape 仍用内联算术）──
    coffCoef:           int = 0     # 2 // Coff
    cubeMBaseSize:      int = 0     # M_BASE_SIZE * coffCoef
    dDealSize:          int = 0     # D_BASE_SIZE // Coff
    mDealSize:          int = 0     # M_BASE_SIZE * Coff
    # ── workspace 分区大小（FP32 元素数，dbRatio=2 双缓冲）──
    dapeWsSize:         int = 0     # dbRatio * groupNum * cmpSize * coffCoef
    dXWsSize:           int = 0     # dbRatio * cubeCoreNum * (M_BASE_SIZE*2) * hiddenSize
    dWWeightWsSize:     int = 0     # dbRatio * groupNum * totalHeadDim * hiddenSize
    xWsSize:            int = 0     # dbRatio * groupNum * groupRowStride * hiddenSize
    dXCacheWsSize:      int = 0     # dbRatio * cmpRatio * hiddenSize


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
    halves = n_cols // BIT16_REG_ELEMENTS
    row_stride = n_cols
    zero_reg = vf.full(0.0, dtype=pl.DT_FP16)
    for half in pl.range(0, halves):
        half_off = half * BIT16_REG_ELEMENTS
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
    halves = n_cols // BIT16_REG_ELEMENTS
    row_stride = n_cols
    zero_reg = vf.full(0.0, dtype=pl.DT_BF16)
    for half in pl.range(0, halves):
        half_off = half * BIT16_REG_ELEMENTS
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
#  VF: 跨 block 归约 — dApe[m,:] += Σ_k d_score[k,m,:]（per-round 调用）
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
            # 累加起点 = ape_tile 当前值（round0 已清零 / round>0 已 load ws 旧值）
            ape_reg = vf.load_align(ape_tile + ape_off, 0)
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
def _get_start_pos(bIdx, startPos, ctx):
    result = 0
    if startPos is not None:
        result = pl.getval(pl.make_tensor(startPos, [ctx.batchSize], [1], dtype=pl.DT_INT32), bIdx)
    return result


def _get_seq_length(bIdx, cuSeqlens, ctx):
    result = ctx.seqSize
    if ctx.layout == 1:  # TH (1=TH)
        result = pl.getval(pl.make_tensor(cuSeqlens, [ctx.batchSize + 1], [1], dtype=pl.DT_INT32), bIdx + 1) - \
                 pl.getval(pl.make_tensor(cuSeqlens, [ctx.batchSize + 1], [1], dtype=pl.DT_INT32), bIdx)
    return result


def _get_seq_used(bIdx, seqUsed, cuSeqlens, ctx):
    if seqUsed is not None:
        result = pl.getval(pl.make_tensor(seqUsed, [ctx.batchSize], [1], dtype=pl.DT_INT32), bIdx)
    else:
        result = _get_seq_length(bIdx, cuSeqlens, ctx)
    return result


# ================================================================
#  Helper: 将 (b, s) 映射为全局 token 索引 tIdx，BSH/TH 统一
# ================================================================
def _get_token_idx(bIdx, sIdx, cuSeqlens, ctx):
    result = bIdx * ctx.seqSize + sIdx
    if ctx.layout == 1:
        result = pl.getval(pl.make_tensor(cuSeqlens, [ctx.batchSize + 1], [1], dtype=pl.DT_INT32), bIdx) + sIdx
    return result


# ================================================================
#  Helper: 计算 (bStart,sStart) 到 (bEnd,sEnd) 之间的行数，sEnd 左闭右开
# ================================================================
def _get_row_count(bStart, sStart, bEnd, sEnd, cuSeqlens, ctx):
    result = _get_token_idx(bEnd, sEnd, cuSeqlens, ctx) - _get_token_idx(bStart, sStart, cuSeqlens, ctx)
    return result


# ================================================================
#  Helper: 计算位置 (bIdx, sIdx) 之前的压缩块总数
# ================================================================
def _get_cmp_block_count(bIdx, sIdx, startPos, seqUsed, cuSeqlens, ctx):
    """位置 (bIdx, sIdx) 之前的压缩块总数（按对齐位置计数）。

    语义与 blkInBatch 的 _ceil_div(cmpLimit-startPos, cmpRatio) 等价：
    startPos%cmpRatio!=0 时首块（部分块）计 1 块，末尾不足一块不计。
    修改时须与 _skip_blocks/_process_scatter/_arrange_x 的 blkInBatch 保持一致。
    """
    result = 0
    for bb in pl.range(0, bIdx):
        bStartPos = _get_start_pos(bb, startPos, ctx)
        bSeqLength = _get_seq_used(bb, seqUsed, cuSeqlens, ctx)
        cmpEnd = (bStartPos + bSeqLength) // ctx.cmpRatio
        if cmpEnd > bStartPos // ctx.cmpRatio:
            result += cmpEnd - (bStartPos // ctx.cmpRatio)
    bStartPos = _get_start_pos(bIdx, startPos, ctx)
    lastS = _trunc(_get_seq_used(bIdx, seqUsed, cuSeqlens, ctx) + bStartPos, ctx.cmpRatio) - bStartPos
    result += (min(lastS, sIdx) + bStartPos) // ctx.cmpRatio - bStartPos // ctx.cmpRatio
    return result


# ================================================================
#  Helper: 将全局 token 索引 tIdx 反向映射为 (bIdx, sIdx)
# ================================================================
def _get_pos_from_token_idx(tIdx, cuSeqlens, ctx):
    resultB = tIdx // ctx.seqSize
    resultS = tIdx % ctx.seqSize
    if ctx.layout == 1:
        resultB = 0
        while resultB < ctx.batchSize:
            if tIdx < pl.getval(pl.make_tensor(cuSeqlens, [ctx.batchSize + 1], [1], dtype=pl.DT_INT32), resultB + 1):
                resultS = tIdx - pl.getval(pl.make_tensor(cuSeqlens, [ctx.batchSize + 1], [1], dtype=pl.DT_INT32), resultB)
                break
            resultB += 1
    return resultB, resultS


# ================================================================
#  Helper: advance block traversal (used by Segment 1 and 3)
# ================================================================
def _skip_blocks(blockCount, bIdx, scIdx,
                 startPos, seqUsed, cuSeqlens, ctx):
    skipped = 0
    while skipped < blockCount:
        bStartPos = _get_start_pos(bIdx, startPos, ctx)
        sqVal = _get_seq_used(bIdx, seqUsed, cuSeqlens, ctx)
        cmpLimit = (bStartPos + sqVal) // ctx.cmpRatio * ctx.cmpRatio
        # startPos%cmpRatio!=0 时首块为部分块但仍计 1 块 → 向上取整（_ceil_div）
        blkInBatch = _ceil_div(cmpLimit - bStartPos, ctx.cmpRatio)
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


# ================================================================
#  Phase 2 matmul sub-functions（从 kernel 提取，coff=1/2 统一）
# ================================================================
def _compute_dx_partial(matmulCtx, dbIdx, cvL0a, cvL0b, cvL0c, l1WDb,
                        l1Kv0, l1Kv1, l1Sb0, l1Sb1,
                        wkvT, wgateT, dXResultGm):
    """Matmul #1: dkv@wkv + dsb@wgate → dX partial（h×wIdx×subVecIdx 三重循环）"""
    cubeCoreIdx = pl.get_block_idx()
    groupSize = matmulCtx.groupSize
    groupNum = matmulCtx.groupNum
    groupIdx = cubeCoreIdx // groupSize
    intraGroupIdx = cubeCoreIdx % groupSize
    nStart = intraGroupIdx * D_BASE_SIZE
    dxRowBase = (dbIdx * matmulCtx.cubeCoreNum + cubeCoreIdx) * matmulCtx.cubeMBaseSize
    for h in pl.range(0, matmulCtx.hiddenSize, H_L1_BASE_SIZE):
        for wIdx in pl.range(0, 2):
            for subVecIdx in pl.range(0, 2):
                nTileRows = matmulCtx.nTileRows0 if subVecIdx == 0 else matmulCtx.nTileRows1
                # coff=1 M 轴子槽行距 = dealScNum*cmpRatio（cr|128 时 = D_BASE_SIZE，与旧值一致）
                mOff    = subVecIdx * matmulCtx.dealScNum * matmulCtx.cmpRatio if Coff == 1 else 0
                dCoffIdx = 0 if Coff == 1 else subVecIdx
                l0a = cvL0a.next()
                pl.set_validshape(l0a, [nTileRows, D_BASE_SIZE])
                wL1 = l1WDb.next() if (subVecIdx == 0 or Coff == 2) else l1WDb.current()
                srcL1 = (l1Kv0 if subVecIdx == 0 else l1Kv1) if wIdx == 0 else \
                        (l1Sb0 if subVecIdx == 0 else l1Sb1)
                pl.set_validshape(srcL1, [nTileRows, D_BASE_SIZE])
                pl.move(l0a, srcL1)
                pl.set_validshape(wL1, [D_BASE_SIZE, H_L1_BASE_SIZE])
                if subVecIdx == 0 or Coff == 2:
                    wRowOff = 0 if Coff == 1 else subVecIdx * matmulCtx.headDim
                    if wIdx == 0:
                        pl.load(wL1, wkvT, [nStart + wRowOff, h])
                    else:
                        pl.load(wL1, wgateT, [nStart + wRowOff, h])
                for nL0 in pl.range(0, H_L1_BASE_SIZE, H_BASE_SIZE):
                    l0b = cvL0b.next()
                    l0c = cvL0c.next()
                    pl.set_validshape(l0b, [D_BASE_SIZE, H_BASE_SIZE])
                    pl.set_validshape(l0c, [nTileRows, H_BASE_SIZE])
                    pl.move(l0b, wL1, offset=[0, nL0])
                    if wIdx == 0:
                        # 尾轮 nTileRows=0（coff=1 subVecIdx=1）时跳过 0 行 matmul（aMatrixRow 下限 1）
                        if nTileRows > 0:
                            pl.matmul(l0c, l0a, l0b)
                    else:
                        if nTileRows > 0:
                            pl.matmul_acc(l0c, l0c, l0a, l0b)
                        pl.store(dXResultGm, l0c, [dCoffIdx, dxRowBase + mOff, h + nL0])


def _compute_dw_partial(matmulCtx, roundIdx, dbIdx, cvL0a, cvL0b, cvL0c, l1WDb,
                        l1Kv0T, l1Kv1T, l1Sb0T, l1Sb1T,
                        xArrangeGm, dWkvResultGm, dWGateResultGm):
    """Matmul #2: dkv@x + dsb@x → dWkv/dWgate partial（h×subVecIdx×nL0×wIdx 循环）

    末轮归约方案：round 0 覆盖写初始化固定槽，round 1+ FIX 原子加（跨轮累加），
    ws 单缓冲（无 dbIdx），v2 末轮一次跨核归约。
    """
    cubeCoreIdx = pl.get_block_idx()
    groupSize = matmulCtx.groupSize
    groupNum = matmulCtx.groupNum
    groupIdx = cubeCoreIdx // groupSize
    intraGroupIdx = cubeCoreIdx % groupSize
    totalHeadDim = matmulCtx.totalHeadDim
    dwRowBase = groupIdx * totalHeadDim + intraGroupIdx * D_BASE_SIZE
    dbRowCnt = matmulCtx.dbRowCnt
    # group 区域步长用实际值 groupRowStride（与 _arrange_x 写方逐字节对齐）
    mStart = dbIdx * dbRowCnt + groupIdx * matmulCtx.groupRowStride + (Coff - 1) * matmulCtx.cmpRatio
    for h in pl.range(0, matmulCtx.hiddenSize, H_L1_BASE_SIZE):
        for subVecIdx in pl.range(0, 2):
            for nL0 in pl.range(0, H_L1_BASE_SIZE, H_BASE_SIZE):
                for wIdx in pl.range(0, 2):
                    nTileRows = matmulCtx.nTileRows0 if subVecIdx == 0 else matmulCtx.nTileRows1
                    l0a = cvL0a.next()
                    pl.set_validshape(l0a, [D_BASE_SIZE, nTileRows])
                    wL1 = l1WDb.next() if (nL0 == 0 and wIdx == 0) else l1WDb.current()
                    if nL0 == 0:
                        srcL1 = (l1Kv0T if subVecIdx == 0 else l1Kv1T) if wIdx == 0 else \
                                (l1Sb0T if subVecIdx == 0 else l1Sb1T)
                        pl.set_validshape(srcL1, [D_BASE_SIZE, nTileRows])
                        pl.move(l0a, srcL1)
                    if nL0 == 0 and wIdx == 0:
                        xOff = mStart + subVecIdx * matmulCtx.dealScNum * matmulCtx.cmpRatio if Coff == 1 else mStart
                        if Coff == 2 and subVecIdx == 0:
                            xOff = xOff - matmulCtx.cmpRatio
                        pl.set_validshape(wL1, [nTileRows, H_L1_BASE_SIZE])
                        pl.load(wL1, xArrangeGm, [xOff, h])
                    l0b = cvL0b.next() if wIdx == 0 else cvL0b.current()
                    l0c = cvL0c.next()
                    pl.set_validshape(l0b, [nTileRows, H_BASE_SIZE])
                    pl.set_validshape(l0c, [D_BASE_SIZE, H_BASE_SIZE])
                    pl.move(l0b, wL1, offset=[0, nL0])
                    if Coff == 1 and subVecIdx == 1:
                        if nTileRows > 0:
                            pl.matmul_acc(l0c, l0c, l0a, l0b)
                    else:
                        pl.matmul(l0c, l0a, l0b)
                    if subVecIdx == 1 or Coff == 2:
                        dwRowOff = 0 if Coff == 1 else subVecIdx * matmulCtx.headDim
                        if wIdx == 0:
                            if roundIdx == 0:
                                pl.store(dWkvResultGm, l0c, [dwRowBase + dwRowOff, h + nL0])
                            else:
                                pl.store(dWkvResultGm, l0c, [dwRowBase + dwRowOff, h + nL0],
                                         atomic=pl.AtomicType.AtomicAdd)
                        else:
                            if roundIdx == 0:
                                pl.store(dWGateResultGm, l0c, [dwRowBase + dwRowOff, h + nL0])
                            else:
                                pl.store(dWGateResultGm, l0c, [dwRowBase + dwRowOff, h + nL0],
                                         atomic=pl.AtomicType.AtomicAdd)




# ================================================================
#  Phase 3 reduce sub-functions（从 kernel 提取）
# ================================================================
def _reduce_ape(vecCtx, coreIdx, groupIdx, dbIdx, roundBlocks,
                reduceAcc, reduceLdGroup, tensorDApeFlat, tensorDApeWsFlat):
    """Phase 3: dApe 跨核归约 + RMW（每核处理 rgStart..rgEnd 的 chunk）"""
    coreNum = vecCtx.coreNum
    groupNum = vecCtx.groupNum
    flatRowBase = dbIdx * groupNum * (2 // Coff)
    rgChunkNum = vecCtx.cmpSize // D_BASE_SIZE
    rgPerCore = rgChunkNum // coreNum
    rgRemain = rgChunkNum % coreNum
    rgStart = coreIdx * rgPerCore + (coreIdx if coreIdx < rgRemain else rgRemain)
    rgEnd = rgStart + rgPerCore + (1 if coreIdx < rgRemain else 0)
    rgStart = rgStart * D_BASE_SIZE
    rgEnd = rgEnd * D_BASE_SIZE
    # validRows：实际写入 tensorApe 的行数（只读被写过的行，与 workspace 初始内容无关）
    # coff=1：写行号连续为 0..min(roundBlocks, 2*groupNum)-1——sub1 子核槽位写入条件
    #   含额外 +dealScNum（groupIdx*gds + dealScNum < roundBlocks），原 ceil 公式在
    #   roundBlocks 非 groupDealScNum 整数倍时高估，会读入未写槽位（未清零→垃圾）
    # coff=2：每 group 仅 1 行（coffCoef=1），ceil 公式精确
    if Coff == 1:
        validRows = min(roundBlocks, groupNum * 2)
    else:
        validRows = (roundBlocks + vecCtx.groupDealScNum - 1) // vecCtx.groupDealScNum
    pl.set_validshape(reduceAcc, [1, rgEnd - rgStart])
    # 末轮一次性归约：累加器从零开始（输出 tensor 不保证清零，不能读入做 RMW）
    _vf_tile_zero(reduceAcc, 1, (rgEnd - rgStart))
    for r in pl.range(0, validRows):
        reduceLd = reduceLdGroup.next()
        pl.set_validshape(reduceLd, [1, rgEnd - rgStart])
        pl.load(reduceLd, tensorDApeWsFlat, [flatRowBase + r, rgStart])
        pl.add(reduceAcc, reduceAcc, reduceLd)
    pl.store(tensorDApeFlat, reduceAcc, [0, rgStart])



def _reduce_d_weight(vecCtx, coreIdx, groupIdx, dbIdx, roundBlocks,
                     reduceAcc, reduceLdGroup, p3Cast,
                     tensorDWKvFlat, tensorDWKvWsFlat, tensorDWGateFlat, tensorDWGateWsFlat):
    """Phase 3: dWkv/dWgate 跨核归约 + RMW（每核处理 rgStart..rgEnd 的 chunk）"""
    coreNum = vecCtx.coreNum
    groupNum = vecCtx.groupNum
    totalHeadDim = vecCtx.totalHeadDim
    flatRowBase = dbIdx * groupNum
    rgChunkNum = (totalHeadDim * vecCtx.hiddenSize) // D_BASE_SIZE
    rgPerCore = rgChunkNum // coreNum
    rgRemain = rgChunkNum % coreNum
    rgStart = coreIdx * rgPerCore + (coreIdx if coreIdx < rgRemain else rgRemain)
    rgEnd = rgStart + rgPerCore + (1 if coreIdx < rgRemain else 0)
    rgStart = rgStart * D_BASE_SIZE
    rgEnd = rgEnd * D_BASE_SIZE
    validRows = (roundBlocks + vecCtx.groupDealScNum - 1) // vecCtx.groupDealScNum
    for curRgStart in pl.range(rgStart, rgEnd, (BASE_SIZE * BASE_SIZE)):
        curRgNum = min(BASE_SIZE * BASE_SIZE, rgEnd - curRgStart)
        pl.set_validshape(reduceAcc, [1, curRgNum])
        # dWkv：末轮一次性归约，累加器从零开始（输出 tensor 不保证清零，不能读入做 RMW）
        _vf_tile_zero(reduceAcc, 1, curRgNum)
        for r in pl.range(0, validRows):
            reduceLd = reduceLdGroup.next()
            pl.set_validshape(reduceLd, [1, curRgNum])
            pl.load(reduceLd, tensorDWKvWsFlat, [flatRowBase + r, curRgStart])
            pl.add(reduceAcc, reduceAcc, reduceLd)
        pl.set_validshape(p3Cast, [1, curRgNum])
        pl.cast(p3Cast, reduceAcc, mode=pl.RoundMode.CAST_ROUND)
        pl.store(tensorDWKvFlat, p3Cast, [0, curRgStart])
        # dWgate：同上，从零开始
        _vf_tile_zero(reduceAcc, 1, curRgNum)
        for r in pl.range(0, validRows):
            reduceLd = reduceLdGroup.next()
            pl.set_validshape(reduceLd, [1, curRgNum])
            pl.load(reduceLd, tensorDWGateWsFlat, [flatRowBase + r, curRgStart])
            pl.add(reduceAcc, reduceAcc, reduceLd)
        pl.set_validshape(p3Cast, [1, curRgNum])
        pl.cast(p3Cast, reduceAcc, mode=pl.RoundMode.CAST_ROUND)
        pl.store(tensorDWGateFlat, p3Cast, [0, curRgStart])



# ================================================================
#  无压缩块：四输出显式刷 0（totalValid=0 / compressedCnt=0）
# ================================================================
def _zero_outputs(coreIdx, coreNum, xRows, hiddenSize, totalHeadDim, cmpSize,
                  dXOutGm, tensorDWKvFlat, tensorDWGateFlat, tensorDApeFlat,
                  reduceAcc, zeroCast):
    """无压缩块时对四个输出 GM 显式刷 0，不依赖归约路径的隐式行为。

    各核按 D_BASE_SIZE 块/行分片（与 _reduce_ape/_reduce_d_weight/_reduce_dx
    的分核模式一致）；零 tile 仅初始化一次，循环复用 store 到 GM。
    """
    # ── 零 tile 初始化（一次）：ioDType 直接清零，FP32 仅供 d_ape ──
    pl.set_validshape(zeroCast, [1, hiddenSize])
    if DataType == 0:                              # BF16
        _vf_tile_zero_bf16(zeroCast, 1, hiddenSize)
    else:                                          # FP16
        _vf_tile_zero_f16(zeroCast, 1, hiddenSize)
    pl.set_validshape(reduceAcc, [1, hiddenSize])
    _vf_tile_zero(reduceAcc, 1, hiddenSize)        # FP32 全零（d_ape 用）

    # ── d_x [xRows, hiddenSize]：按行分片（_reduce_dx 同款）──
    rowsPerCore = xRows // coreNum
    rowsRemain  = xRows % coreNum
    rS = coreIdx * rowsPerCore + (coreIdx if coreIdx < rowsRemain else rowsRemain)
    rE = rS + rowsPerCore + (1 if coreIdx < rowsRemain else 0)
    for r in pl.range(rS, rE):
        pl.store(dXOutGm, zeroCast, [r, 0])

    # ── d_wkv / d_wgate [totalHeadDim * hiddenSize]：按 D_BASE_SIZE 块分片 ──
    #    （与 _reduce_d_weight 同款：扁平 view + 块分片）
    dwChunkNum = (totalHeadDim * hiddenSize) // D_BASE_SIZE
    dwPerCore  = dwChunkNum // coreNum
    dwRemain   = dwChunkNum % coreNum
    wS = coreIdx * dwPerCore + (coreIdx if coreIdx < dwRemain else dwRemain)
    wE = wS + dwPerCore + (1 if coreIdx < dwRemain else 0)
    wS = wS * D_BASE_SIZE
    wE = wE * D_BASE_SIZE
    for curWStart in pl.range(wS, wE, BASE_SIZE * BASE_SIZE):
        curWNum = min(BASE_SIZE * BASE_SIZE, wE - curWStart)
        pl.set_validshape(zeroCast, [1, curWNum])
        pl.store(tensorDWKvFlat, zeroCast, [0, curWStart])
        pl.store(tensorDWGateFlat, zeroCast, [0, curWStart])

    # ── d_ape [1, cmpSize] FP32：按 D_BASE_SIZE 块分片（_reduce_ape 同款）──
    rgChunkNum = cmpSize // D_BASE_SIZE
    rgPerCore  = rgChunkNum // coreNum
    rgRemain   = rgChunkNum % coreNum
    rgStart = coreIdx * rgPerCore + (coreIdx if coreIdx < rgRemain else rgRemain)
    rgEnd   = rgStart + rgPerCore + (1 if coreIdx < rgRemain else 0)
    rgStart = rgStart * D_BASE_SIZE
    rgEnd   = rgEnd * D_BASE_SIZE
    for curRgStart in pl.range(rgStart, rgEnd, BASE_SIZE * BASE_SIZE):
        curRgNum = min(BASE_SIZE * BASE_SIZE, rgEnd - curRgStart)
        pl.set_validshape(reduceAcc, [1, curRgNum])
        pl.store(tensorDApeFlat, reduceAcc, [0, curRgStart])


def _cast_dkv_dsb_to_l1(vecCtx, nTileRows, dealTcSize, subIdx,
                        tileTemp, tileDkvCast, tileDkvCastMd, tileDkvNz,
                        tileDkvNzMdPrev, tileDkvNzMdCur,
                        tileDsbCast, tileDsbCastMd, tileDsbNz, tileSoftmax,
                        tileDsbNzMdPrev, tileDsbNzMdCur,
                        tempTileGroup, l1Kv0G, l1Kv1G, l1Sb0G, l1Sb1G):
    """Phase 1: dkv/dsb FP32→FP16 cast + ND→NZ + insert L1（coff=1/2 统一）"""
    dDealSize = vecCtx.dDealSize
    cmpRatio = vecCtx.cmpRatio
    # dkv 通路
    pl.set_validshape(tileDkvCast, [nTileRows, dDealSize])
    pl.set_validshape(tileTemp, [nTileRows, dDealSize])
    pl.cast(tileDkvCast, tileTemp, mode=pl.RoundMode.CAST_ROUND)
    if Coff == 1:
        pl.set_validshape(tileDkvNz, [nTileRows, dDealSize])
        pl.move(tileDkvNz, tileDkvCast)
    else:
        tempTile = tempTileGroup.next()
        pl.set_validshape(tileDkvCastMd, [dealTcSize * cmpRatio, dDealSize])
        pl.set_validshape(tileDkvNzMdPrev, [dealTcSize * cmpRatio, dDealSize])
        pl.set_validshape(tempTile, [dealTcSize * cmpRatio, dDealSize])
        pl.move(tempTile, tileDkvCastMd, offset=[0, 0])
        pl.move(tileDkvNzMdPrev, tempTile)
        tempTile = tempTileGroup.next()
        pl.set_validshape(tileDkvCastMd[:, dDealSize:], [dealTcSize * cmpRatio, dDealSize])
        pl.set_validshape(tileDkvNzMdCur, [dealTcSize * cmpRatio, dDealSize])
        pl.set_validshape(tempTile, [dealTcSize * cmpRatio, dDealSize])
        pl.move(tempTile, tileDkvCastMd, offset=[0, dDealSize])
        pl.move(tileDkvNzMdCur, tempTile)
    # dsb 通路
    pl.set_validshape(tileDsbCast, [nTileRows, dDealSize])
    pl.set_validshape(tileSoftmax, [nTileRows, dDealSize])
    pl.cast(tileDsbCast, tileSoftmax, mode=pl.RoundMode.CAST_ROUND)
    if Coff == 1:
        pl.set_validshape(tileDsbNz, [nTileRows, dDealSize])
        pl.move(tileDsbNz, tileDsbCast)
    else:
        tempTile = tempTileGroup.next()
        pl.set_validshape(tileDsbCastMd, [dealTcSize * cmpRatio, dDealSize])
        pl.set_validshape(tileDsbNzMdPrev, [dealTcSize * cmpRatio, dDealSize])
        pl.set_validshape(tempTile, [dealTcSize * cmpRatio, dDealSize])
        pl.move(tempTile, tileDsbCastMd, offset=[0, 0])
        pl.move(tileDsbNzMdPrev, tempTile)
        tempTile = tempTileGroup.next()
        pl.set_validshape(tileDsbCastMd[:, dDealSize:], [dealTcSize * cmpRatio, dDealSize])
        pl.set_validshape(tileDsbNzMdCur, [dealTcSize * cmpRatio, dDealSize])
        pl.set_validshape(tempTile, [dealTcSize * cmpRatio, dDealSize])
        pl.move(tempTile, tileDsbCastMd, offset=[0, dDealSize])
        pl.move(tileDsbNzMdCur, tempTile)
    # insert L1
    if Coff == 1:
        l1Kv = l1Kv0G.next() if subIdx == 0 else l1Kv1G.next()
        l1Sb = l1Sb0G.next() if subIdx == 0 else l1Sb1G.next()
        pl.set_validshape(l1Kv, [nTileRows, dDealSize])
        pl.set_validshape(l1Sb, [nTileRows, dDealSize])
        pl.insert(l1Kv, tileDkvNz, [0, 0])
        pl.insert(l1Sb, tileDsbNz, [0, 0])
    else:
        l1Kv0 = l1Kv0G.next()
        l1Kv1 = l1Kv1G.next()
        l1Sb0 = l1Sb0G.next()
        l1Sb1 = l1Sb1G.next()
        colOffset = 0 if subIdx == 0 else dDealSize
        pl.set_validshape(l1Kv0, [dealTcSize * cmpRatio, D_BASE_SIZE])
        pl.set_validshape(l1Kv1, [dealTcSize * cmpRatio, D_BASE_SIZE])
        pl.set_validshape(l1Sb0, [dealTcSize * cmpRatio, D_BASE_SIZE])
        pl.set_validshape(l1Sb1, [dealTcSize * cmpRatio, D_BASE_SIZE])
        pl.insert(l1Kv0, tileDkvNzMdPrev, [0, colOffset])
        pl.insert(l1Kv1, tileDkvNzMdCur, [0, colOffset])
        pl.insert(l1Sb0, tileDsbNzMdPrev, [0, colOffset])
        pl.insert(l1Sb1, tileDsbNzMdCur, [0, colOffset])



def _reduce_dx(vecCtx, coreIdx, dbIdx, roundIdx,
               bIdxStart, scIdxStart, bIdxEnd, scIdxEnd,
               reduceAcc, reduceLdGroup, p3Cast,
               dXCacheGm, dXResultGm, dXOutGm,
               startPos, seqUsed, cuSeqlens, seqCtx):
    """Phase 3: dX 跨核归约（token 行遍历 + coff=2 dXCache 缓存逻辑）"""
    coreNum = vecCtx.coreNum
    cubeCoreNum = vecCtx.cubeCoreNum
    roundCnt = vecCtx.roundCnt
    cmpRatio = vecCtx.cmpRatio
    hiddenSize = vecCtx.hiddenSize
    batchSize = vecCtx.batchSize
    groupSize = vecCtx.groupSize
    dbRatio = vecCtx.dbRatio
    # start 节点局部副本（避免别名修改传入的 scIdxStart）
    startScIdx = scIdxStart
    prevScIdx = 0
    if Coff == 2 and startScIdx != 0:
        startScIdx -= 1
        prevScIdx += 1
    bStart = bIdxStart
    bEnd   = bIdxEnd - 1 if scIdxEnd == 0 else bIdxEnd
    sStart = (0 if startScIdx == 0
              else startScIdx * cmpRatio - (_get_start_pos(bStart, startPos, seqCtx) % cmpRatio))
    sEnd   = (_get_seq_length(bEnd, cuSeqlens, seqCtx) if scIdxEnd == 0
              else scIdxEnd * cmpRatio - (_get_start_pos(bEnd, startPos, seqCtx) % cmpRatio))
    if roundIdx == roundCnt - 1:
        bEnd = batchSize - 1
        sEnd = _get_seq_length(bEnd, cuSeqlens, seqCtx)
    totalRows = _get_row_count(bStart, sStart, bEnd, sEnd, cuSeqlens, seqCtx)
    rowsPerCore = totalRows // coreNum
    rowsRemain  = totalRows % coreNum
    myRowS = coreIdx * rowsPerCore + (coreIdx if coreIdx < rowsRemain else rowsRemain)
    myRowE = myRowS + rowsPerCore + (1 if coreIdx < rowsRemain else 0)
    roundEndScIdx = _get_cmp_block_count(bEnd, sEnd, startPos, seqUsed, cuSeqlens, seqCtx) - \
                     _get_cmp_block_count(bStart, sStart, startPos, seqUsed, cuSeqlens, seqCtx) - 1
    if myRowS < myRowE:
        startTidx = _get_token_idx(bStart, sStart, cuSeqlens, seqCtx)
        b, localS = _get_pos_from_token_idx(startTidx + myRowS, cuSeqlens, seqCtx)
        tIdx = startTidx + myRowS
        bStartPos = _get_start_pos(b, startPos, seqCtx)
        bSeqUsed = _get_seq_used(b, seqUsed, cuSeqlens, seqCtx)
        bSeqLength = _get_seq_length(b, cuSeqlens, seqCtx)
        cmpLimit = (bStartPos + bSeqUsed) // cmpRatio * cmpRatio - bStartPos
        p3lScIdx = _get_cmp_block_count(b, localS, startPos, seqUsed, cuSeqlens, seqCtx) - \
                     _get_cmp_block_count(bStart, sStart, startPos, seqUsed, cuSeqlens, seqCtx)
        if localS >= cmpLimit and cmpLimit > 0:
            p3lScIdx -= 1
        batchEndScIdx = _get_cmp_block_count(b, bSeqUsed, startPos, seqUsed, cuSeqlens, seqCtx) - \
                     _get_cmp_block_count(bStart, sStart, startPos, seqUsed, cuSeqlens, seqCtx) - 1
        scInnerIdx = (localS + bStartPos) % cmpRatio
        for rowIdx in pl.range(myRowS, myRowE):
            inCompress = localS < cmpLimit
            if inCompress:
                if p3lScIdx < prevScIdx:
                    ws_offset = dbIdx * cubeCoreNum * vecCtx.cubeMBaseSize + scInnerIdx
                    pl.set_validshape(reduceAcc, [1, hiddenSize])
                    pl.load(reduceAcc, dXCacheGm, [dbIdx * cmpRatio + scInnerIdx, 0])
                    for r in pl.range(0, groupSize):
                        reduceLd = reduceLdGroup.next()
                        pl.set_validshape(reduceLd, [1, hiddenSize])
                        pl.load(reduceLd, dXResultGm, [0, ws_offset + r * vecCtx.cubeMBaseSize, 0])
                        pl.add(reduceAcc, reduceAcc, reduceLd)
                else:
                    ws_offset = dbIdx * cubeCoreNum * vecCtx.cubeMBaseSize + ((p3lScIdx - prevScIdx) // vecCtx.groupDealScNum) * groupSize * vecCtx.cubeMBaseSize + ((p3lScIdx - prevScIdx) % vecCtx.groupDealScNum) * cmpRatio + scInnerIdx
                    pl.set_validshape(reduceAcc, [1, hiddenSize])
                    pl.load(reduceAcc, dXResultGm, [Coff - 1, ws_offset, 0])
                    if groupSize > 1:
                        for r in pl.range(1, groupSize):
                            reduceLd = reduceLdGroup.next()
                            pl.set_validshape(reduceLd, [1, hiddenSize])
                            pl.load(reduceLd, dXResultGm, [Coff - 1, ws_offset + r * vecCtx.cubeMBaseSize, 0])
                            pl.add(reduceAcc, reduceAcc, reduceLd)
                    if Coff == 2 and p3lScIdx != roundEndScIdx and p3lScIdx != batchEndScIdx:
                        pl.set_validshape(reduceAcc, [1, hiddenSize])
                        next_ws_offset = dbIdx * cubeCoreNum * vecCtx.cubeMBaseSize + ((p3lScIdx + 1 - prevScIdx) // vecCtx.groupDealScNum) * groupSize * vecCtx.cubeMBaseSize + ((p3lScIdx + 1 - prevScIdx) % vecCtx.groupDealScNum) * cmpRatio + scInnerIdx
                        for r in pl.range(0, groupSize):
                            reduceLd = reduceLdGroup.next()
                            pl.set_validshape(reduceLd, [1, hiddenSize])
                            pl.load(reduceLd, dXResultGm, [0, next_ws_offset + r * vecCtx.cubeMBaseSize, 0])
                            pl.add(reduceAcc, reduceAcc, reduceLd)
                if Coff == 2 and p3lScIdx == roundEndScIdx and p3lScIdx != batchEndScIdx:
                    pl.store(dXCacheGm, reduceAcc, [(dbIdx + 1) % dbRatio * cmpRatio + scInnerIdx, 0])
                else:
                    pl.set_validshape(p3Cast, [1, hiddenSize])
                    pl.cast(p3Cast, reduceAcc, mode=pl.RoundMode.CAST_ROUND)
                    pl.store(dXOutGm, p3Cast, [tIdx, 0])
            else:
                if p3lScIdx < prevScIdx:
                    continue
                pl.set_validshape(p3Cast, [1, hiddenSize])
                if DataType == 0:
                    _vf_tile_zero_bf16(p3Cast, 1, hiddenSize)
                else:
                    _vf_tile_zero_f16(p3Cast, 1, hiddenSize)
                pl.store(dXOutGm, p3Cast, [tIdx, 0])
            tIdx += 1
            localS += 1
            if localS >= bSeqLength:
                # ── 跨 batch 转移：压缩块索引全局连续，新批首块是前序批末块的后继。
                # 前序批有压缩块（cmpLimit>0，此刻 cmpLimit 仍是前序批的值）→ +1；
                # 前序批 0 块 → 索引不变（块索引跨批不连续，不能无条件 +1）。
                if cmpLimit > 0:
                    p3lScIdx += 1
                localS = 0
                b += 1
                if b >= batchSize:
                    break
                bStartPos = _get_start_pos(b, startPos, seqCtx)
                bSeqUsed = _get_seq_used(b, seqUsed, cuSeqlens, seqCtx)
                bSeqLength = _get_seq_length(b, cuSeqlens, seqCtx)
                cmpLimit = (bStartPos + bSeqUsed) // cmpRatio * cmpRatio - bStartPos
                batchEndScIdx = _get_cmp_block_count(b, bSeqUsed, startPos, seqUsed, cuSeqlens, seqCtx) - \
                     _get_cmp_block_count(bStart, sStart, startPos, seqUsed, cuSeqlens, seqCtx) - 1
            else:
                # ── 批内块边界（localS 为 cr 倍数，进入下一块）→ +1
                # else 分支 localS≥1；(localS+bStartPos)%cmpRatio==0 即块边界；
                # localS<cmpLimit 保证仍在压缩区内（尾区不增量）。
                if (localS < cmpLimit) and ((localS + bStartPos) % cmpRatio == 0):
                    p3lScIdx += 1
            scInnerIdx = (localS + bStartPos) % cmpRatio



def _process_scatter(vecCtx, roundIdx, totalScNumPerRound, preDealTcSize, dealTcSize,
                     bIdx, scIdx,
                     tileDcF16, tileDcF32, tileKv, tileSoftmax, tileTemp,
                     dCmpKvGm, kvGm, softmaxScoreGm, nStart,
                     startPos, seqUsed, cuSeqlens, seqCtx):
    """Phase 1: scatter（load dc/kv/sm → cast → VF 反向 → mask cache fill）"""
    cmpRatio = vecCtx.cmpRatio
    cmpRowCnt = vecCtx.cmpRowCnt
    dDealSize = vecCtx.dDealSize
    batchSize = vecCtx.batchSize
    processed = 0
    while processed < dealTcSize:
        bStartPos = _get_start_pos(bIdx, startPos, seqCtx)
        sqVal = _get_seq_used(bIdx, seqUsed, cuSeqlens, seqCtx)
        cmpLimit = (bStartPos + sqVal) // cmpRatio * cmpRatio
        # startPos%cmpRatio!=0 时首块为部分块但仍计 1 块 → 向上取整（_ceil_div）
        blkInBatch = _ceil_div(cmpLimit - bStartPos, cmpRatio)
        remaining = blkInBatch - scIdx
        if remaining > 0:
            taken = min(dealTcSize - processed, remaining)
            if Layout == 1:
                globalBlkIdx = roundIdx * totalScNumPerRound + preDealTcSize + processed
            else:
                globalBlkIdx = bIdx * vecCtx.cmpKvBatchStride + scIdx
            nTileRows = taken * cmpRowCnt
            dcTile = tileDcF16[processed:, :]
            dcF32 = tileDcF32[processed:, :]
            kvTile = tileKv[processed * cmpRowCnt:, :]
            smTile = tileSoftmax[processed * cmpRowCnt:, :]
            dkvTile = tileTemp[processed * cmpRowCnt:, :]
            pl.set_validshape(dcTile, [taken, dDealSize])
            pl.set_validshape(kvTile, [nTileRows, dDealSize])
            pl.set_validshape(smTile, [nTileRows, dDealSize])
            pl.load(dcTile, dCmpKvGm, [globalBlkIdx, nStart])
            pl.load(kvTile, kvGm, [globalBlkIdx, 0, nStart])
            pl.load(smTile, softmaxScoreGm, [globalBlkIdx, 0, nStart])
            pl.set_validshape(dcF32, [taken, dDealSize])
            pl.cast(dcF32, dcTile, mode=pl.RoundMode.CAST_ROUND)
            _vf_scatter_backward(
                dcF32, kvTile, smTile, dkvTile,
                taken, cmpRowCnt, dDealSize,
            )
            if Coff == 1:
                cacheFillRows = bStartPos % cmpRatio if scIdx == 0 else 0
                _vf_mask_cache_fill(smTile, dkvTile, cacheFillRows, dDealSize, D_BASE_SIZE)
            else:
                if scIdx == 0:
                    cacheFillRows = bStartPos % cmpRatio + cmpRatio if taken >= 2 else cmpRatio
                elif scIdx == 1:
                    cacheFillRows = bStartPos % cmpRatio
                else:
                    cacheFillRows = 0
                _vf_mask_cache_fill(smTile, dkvTile, cacheFillRows, dDealSize, D_BASE_SIZE)
                cacheFillRows = bStartPos % cmpRatio if scIdx == 0 else 0
                _vf_mask_cache_fill(smTile[1:, :], dkvTile[1:, :], cacheFillRows, dDealSize, D_BASE_SIZE)
            processed += taken
            scIdx += taken
        if scIdx >= blkInBatch:
            scIdx = 0
            bIdx += 1
        if bIdx >= batchSize:
            break



def _arrange_x(vecCtx, dbIdx, groupIdx, dbRowCnt, dealTcSize,
               bIdx, scIdx, subSlot, moveXTiles, xGm, xArrangeGm,
               startPos, seqUsed, cuSeqlens, seqCtx):
    """Phase 1: x 搬运（cache copy + 块排列 + coff=2 头部补写）"""
    cmpRatio = vecCtx.cmpRatio
    hiddenSize = vecCtx.hiddenSize
    batchSize = vecCtx.batchSize
    dbRatio = vecCtx.dbRatio
    # cache copy（coff=2, group 0 跨 db 头部）：源 = 上一 db 区域最后一个块的 cr 行。
    # 紧凑布局下区域排满，末块起点恰好 = dbRowCnt − cmpRatio。
    if Coff == 2 and groupIdx == 0 and scIdx != 0:
        firstSrcIdx = ((dbIdx + dbRatio - 1) % dbRatio) * dbRowCnt + dbRowCnt - cmpRatio
        firstDstIdx = dbIdx * dbRowCnt
        for hi in pl.range(0, hiddenSize, D_BASE_SIZE * 2):
            mxTie = moveXTiles.next()
            pl.set_validshape(mxTie, [cmpRatio, D_BASE_SIZE * 2])
            pl.load(mxTie, xArrangeGm, [firstSrcIdx, hi])
            pl.store(xArrangeGm, mxTie, [firstDstIdx, hi])
    processed = 0
    while processed < dealTcSize:
        bStartPos = _get_start_pos(bIdx, startPos, seqCtx)
        bSeqUsed = _get_seq_used(bIdx, seqUsed, cuSeqlens, seqCtx)
        cmpLimit = (bStartPos + bSeqUsed) // cmpRatio * cmpRatio
        # startPos%cmpRatio!=0 时首块为部分块但仍计 1 块 → 向上取整（_ceil_div）
        blkInBatch = _ceil_div(cmpLimit - bStartPos, cmpRatio)
        remaining = blkInBatch - scIdx
        if remaining > 0:
            taken = min(dealTcSize - processed, remaining)
            # 块槽位相对化：group 区域步长用实际值 groupRowStride（= gs*cr + coff=2 头部 cr），
            # 与 mm2 mStart 逐字节对齐；槽位 = subSlot(coff=1 子核偏移) + processed(块序)
            dstIdx = dbIdx * dbRowCnt + groupIdx * vecCtx.groupRowStride \
                     + (subSlot + processed) * cmpRatio + (Coff - 1) * cmpRatio
            mxsIdx = (0 if scIdx == 0
                      else scIdx * cmpRatio - (bStartPos % cmpRatio))
            srcIdx = _get_token_idx(bIdx, mxsIdx, cuSeqlens, seqCtx)
            mxRows = taken * cmpRatio
            if mxsIdx == 0:
                mxRows -= bStartPos % cmpRatio
                dstIdx += bStartPos % cmpRatio
            if Coff == 2 and processed == 0 and groupIdx > 0:
                for hi in pl.range(0, hiddenSize, D_BASE_SIZE * 2):
                    mxTie = moveXTiles.next()
                    pl.set_validshape(mxTie, [cmpRatio, D_BASE_SIZE * 2])
                    pl.load(mxTie, xGm, [srcIdx - cmpRatio, hi])
                    pl.store(xArrangeGm, mxTie, [dbIdx * dbRowCnt + groupIdx * vecCtx.groupRowStride, hi])
            for hi in pl.range(0, hiddenSize, D_BASE_SIZE * 2):
                mxTie = moveXTiles.next()
                pl.set_validshape(mxTie, [mxRows, D_BASE_SIZE * 2])
                pl.load(mxTie, xGm, [srcIdx, hi])
                pl.store(xArrangeGm, mxTie, [dstIdx, hi])
            processed += taken
            scIdx += taken
        if scIdx >= blkInBatch:
            scIdx = 0
            bIdx += 1
        if bIdx >= batchSize:
            break

@pl.jit(tiling_key=CompressorGradTilingKey, auto_mutex=True, arch="a5")
def compressor_grad(
    # ── 输入（前向参数） ──
    x:              pl.Ptr[pl.DT_UINT8],     # (T, hiddenSize)      
    wkv:            pl.Ptr[pl.DT_UINT8],     # (C, hiddenSize)       
    wgate:          pl.Ptr[pl.DT_UINT8],     # (C, hiddenSize)       

    # ── 输入（前向中间变量 / 上游梯度） ──
    dCmpKv:       pl.Ptr[pl.DT_UINT8],     # (cmpKvRows, headDim)
    softmaxScore: pl.Ptr[pl.DT_UINT8],     # (cmpKvRows, cmpRowCnt, headDim) FP32
    kv:             pl.Ptr[pl.DT_UINT8],     # (cmpKvRows, cmpRowCnt, headDim) FP32

    # ── 位置 / 序列参数 ──
    cuSeqlens:     pl.Ptr[pl.DT_UINT8],
    seqUsed:        pl.Ptr[pl.DT_UINT8],
    startPos:      pl.Ptr[pl.DT_UINT8],

    # ── 输出（最终梯度） ──
    dX:            pl.Ptr[pl.DT_UINT8],     # (T, hiddenSize) FP32 partials → Phase 3 reduce
    dWkv:          pl.Ptr[pl.DT_UINT8],     # (C, hiddenSize) FP32
    dWgate:        pl.Ptr[pl.DT_UINT8],     # (C, hiddenSize) FP32
    dApe:          pl.Ptr[pl.DT_UINT8],     # APE workspace / output

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
    totalHeadDim = tiling.totalHeadDim
    cmpRowCnt = tiling.cmpRowCnt
    cmpSize = tiling.cmpSize
    cmpKvBatchStride    = tiling.cmpKvBatchStride
    ioDType      = pl.DT_BF16 if DataType == 0 else pl.DT_FP16
    dbRatio = 2 # tiling传入
    dDealSize = tiling.dDealSize
    mDealSize = tiling.mDealSize

    # ── Sequence context ──
    seq_ctx = pl.struct("SeqCtx",
        batchSize=batchSize,
        cmpRatio=cmpRatio,
        layout=Layout,
        seqSize=seqSize,
    )

    # ── Compressed rows / block counts ──
    # compressedCnt 依赖 startPos/seqUsed 数据内容（op_host tiling 阶段读不到），kernel 内算
    cmpKvRows = tiling.cmpKvRows
    compressedCnt = 0
    for bIdx in pl.range(0, batchSize):
        bStartPos = _get_start_pos(bIdx, startPos, seq_ctx)
        bSeqUsed = _get_seq_used(bIdx, seqUsed, cuSeqlens, seq_ctx)
        compressSeqIdx = _trunc(bStartPos + bSeqUsed, cmpRatio)
        if compressSeqIdx > bStartPos:
            # startPos%cmpRatio!=0 时首块为部分块但仍计 1 块 → 向上取整（_ceil_div）
            compressedCnt = compressedCnt + _ceil_div(compressSeqIdx - bStartPos, cmpRatio)


    # ── Gm Tensor ──
    dCmpKvGm = pl.make_tensor(dCmpKv, [cmpKvRows, headDim],
                                   [headDim, 1], dtype=ioDType)
    kvGm      = pl.make_tensor(kv, [cmpKvRows, cmpRowCnt, headDim],
                                   [cmpSize, headDim, 1], dtype=pl.DT_FP32)
    softmaxScoreGm = pl.make_tensor(softmaxScore, [cmpKvRows, cmpRowCnt, headDim],
                                   [cmpSize, headDim, 1], dtype=pl.DT_FP32)
    
    xGm = pl.make_tensor(x, [tokenSize, hiddenSize],
                            [hiddenSize, 1], dtype=ioDType) #TODO:bsh时，tokenSize=b*s

    xRows = tiling.xRows
    dXOutGm = pl.make_tensor(dX, [xRows, hiddenSize],
                              [hiddenSize, 1], dtype=ioDType)
    dWKvOutGm = pl.make_tensor(dWkv, [totalHeadDim, hiddenSize],
                              [hiddenSize, 1], dtype=ioDType)
    dWGateOutGm = pl.make_tensor(dWgate, [totalHeadDim, hiddenSize],
                              [hiddenSize, 1], dtype=ioDType)


    # ── 共享 round 参数（Phase 1 + Phase 2 都需要）──
    cubeCoreNum = tiling.cubeCoreNum
    groupSize   = tiling.groupSize
    groupNum    = tiling.groupNum
    coffCoef = tiling.coffCoef
    cubeMBaseSize = tiling.cubeMBaseSize # coff = 1, vec子核分M轴；coff = 2, vec子核分coff
    groupDealScNum = tiling.groupDealScNum
    dealScNum      = tiling.dealScNum
    totalScNumPerRound      = tiling.totalScNumPerRound
    roundCnt          = _ceil_div(compressedCnt, totalScNumPerRound)
    # 单轮实际块数上限 = 首轮块数（各轮最大值）：归约侧有效行数依据
    #（不满轮时空闲 group 的 ws 区域未写，归约只读写过数据的行）
    maxRoundBlocks    = min(totalScNumPerRound, compressedCnt)

    # ── Vec 子函数共享 ctx（int64 稳定值，派生配置经 tiling 传入）──
    vecCtx = pl.struct("VecCtx",
        hiddenSize=hiddenSize, batchSize=batchSize, cmpRatio=cmpRatio, headDim=headDim,
        cmpRowCnt=cmpRowCnt, cmpSize=cmpSize,
        dDealSize=dDealSize, mDealSize=mDealSize,
        dbRatio=dbRatio, coffCoef=coffCoef,
        cubeMBaseSize=cubeMBaseSize, groupDealScNum=groupDealScNum,
        dealScNum=dealScNum, cmpKvBatchStride=cmpKvBatchStride,
        cubeCoreNum=cubeCoreNum, coreNum=tiling.coreNum,
        groupSize=groupSize, groupNum=groupNum,
        totalHeadDim=totalHeadDim, dbRowCnt=tiling.dbRowCnt, groupRowStride=tiling.groupRowStride, roundCnt=roundCnt)

    # ── L1A CV 通道: 4 个独立 tile_group [128,128], double-buffer ──
    L1_NZ_TILE_TYPE = pl.TileType(shape=[M_BASE_SIZE, D_BASE_SIZE], dtype=ioDType,
                              target_memory=pl.MemorySpace.Mat, layout=pl.NZ, compact=1)
    L1_ZN_TILE_TYPE = pl.TileType(shape=[M_BASE_SIZE, D_BASE_SIZE], dtype=ioDType,
                              target_memory=pl.MemorySpace.Mat, layout=pl.ZN, compact=1)

    L1Kv0TileGroupNz    = pl.make_tile_group(type=L1_NZ_TILE_TYPE, addrs=L1_KV0_ADDR, mutex_ids=[10, 11]) # 128 * 128 * 2 = 0x10000
    L1Kv1TileGroupNz    = pl.make_tile_group(type=L1_NZ_TILE_TYPE, addrs=L1_KV1_ADDR, mutex_ids=[12, 13])
    L1Score0TileGroupNz    = pl.make_tile_group(type=L1_NZ_TILE_TYPE, addrs=L1_SB0_ADDR, mutex_ids=[14, 15])
    L1Score1TileGroupNz    = pl.make_tile_group(type=L1_NZ_TILE_TYPE, addrs=L1_SB1_ADDR, mutex_ids=[16, 17])

    L1Kv0TileGroupZn  = pl.make_tile_group(type=L1_ZN_TILE_TYPE, addrs=L1_KV0_ADDR, mutex_ids=[10, 11])
    L1Kv1TileGroupZn  = pl.make_tile_group(type=L1_ZN_TILE_TYPE, addrs=L1_KV1_ADDR, mutex_ids=[12, 13])
    L1Score0TileGroupZn  = pl.make_tile_group(type=L1_ZN_TILE_TYPE, addrs=L1_SB0_ADDR, mutex_ids=[14, 15])
    L1Score1TileGroupZn  = pl.make_tile_group(type=L1_ZN_TILE_TYPE, addrs=L1_SB1_ADDR, mutex_ids=[16, 17])

    # ── Workspace 分块: APE 区域 + dX 区域──
    apeWorkSpaceSize   = tiling.dapeWsSize
    dXWorkSpaceSize    = tiling.dXWsSize
    dWeightWorkSpaceSize    = tiling.dWWeightWsSize
    xWorkSpaceSize    = tiling.xWsSize
    dXCacheWorkSpaceSize    = tiling.dXCacheWsSize

    workSpacePtr = pl.make_ptr(workspace, dtype=pl.DT_FP32)
    dXWorkSpacePtr = pl.addptr(workSpacePtr, apeWorkSpaceSize)       
    dWKvWorkSpacePtr = pl.addptr(dXWorkSpacePtr, dXWorkSpaceSize)   
    dWGateWorkSpacePtr = pl.addptr(dWKvWorkSpacePtr, dWeightWorkSpaceSize)   
    xWorkSpacePtr = pl.addptr(dWGateWorkSpacePtr, dWeightWorkSpaceSize)   
    dXCachePtr = pl.addptr(xWorkSpacePtr, xWorkSpaceSize)   

    # ── dX output + workspace views (Phase 3) ──
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
    # ── Prev-round snapshot（preload：v2(i-1) 在下一轮执行，使用上一轮存档）──
    prevDbIdx         = 0
    prevBIdxStart     = 0
    prevScIdxStart    = 0
    prevBIdxEnd       = 0
    prevScIdxEnd      = 0
    # ── Current-round snapshot（V1(i) 写入；V2(i-1) 结束后轮换为 prev）──
    curDbIdx          = 0
    curBIdxStart      = 0
    curScIdxStart     = 0
    curBIdxEnd        = 0
    curScIdxEnd       = 0

    # ════════════════════════════════════════════════════════════
    #  Vec tile declarations (once)
    # ════════════════════════════════════════════════════════════
    with pl.section_vector():
        coreNum   = tiling.coreNum
        coreIdx   = pl.get_block_idx()
        subCoreIdx = pl.get_subblock_idx()

        # ── UB tile type declarations ──
        # TileType 静态 shape 需编译期立即量 → 用内联算术（值恒等于 tiling.dDealSize/mDealSize）
        D_CMP_KV_TILE_TYPE = pl.TileType(
            shape=[M_BASE_SIZE * Coff // 2, D_BASE_SIZE // Coff], dtype=ioDType,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])

        D_CMP_KV_FP32_TILE_TYPE = pl.TileType(shape=[M_BASE_SIZE * Coff // 2, D_BASE_SIZE // Coff], dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1,-1], pad=pl.TilePad.zero)
        
        NORMEL_TILE_TYPE = pl.TileType(
            shape=[M_BASE_SIZE * Coff, D_BASE_SIZE // Coff], dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1],
            pad=pl.TilePad.zero, compact=1)
        
        CAST_TILE_TYPE = pl.TileType(
            shape=[M_BASE_SIZE * Coff, D_BASE_SIZE // Coff], dtype=ioDType,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1)

        CAST_MD_TILE_TYPE = pl.TileType(
            shape=[M_BASE_SIZE, D_BASE_SIZE], dtype=ioDType,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1)

        NZ_TILE_TYPE = pl.TileType(
            shape=[M_BASE_SIZE * Coff, D_BASE_SIZE // Coff], dtype=ioDType,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1],
            layout=pl.NZ, compact=1)
        
        NZ_MD_TILE_TYPE = pl.TileType(
            shape=[M_BASE_SIZE, D_BASE_SIZE // Coff], dtype=ioDType,
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
        dCmpKvTileGroup      = pl.make_tile_group(type=D_CMP_KV_TILE_TYPE, addrs=[UB_DC_F16_ADDR], mutex_ids=[0])
        dCmpKvFP32TileGroup  = pl.make_tile_group(type=D_CMP_KV_FP32_TILE_TYPE,     addrs=[UB_DC_F32_ADDR], mutex_ids=[1]) # 16kB = 0x4000

        kvTileGroup      = pl.make_tile_group(type=NORMEL_TILE_TYPE,  addrs=[UB_KV_APE_ADDR], mutex_ids=[2]) # 32kB = 0x8000
        softmaxScoreTileGroup      = pl.make_tile_group(type=NORMEL_TILE_TYPE,  addrs=[UB_SM_DSB_ADDR], mutex_ids=[3]) # 64kb = 0x10000
        dKvTileGroup     = pl.make_tile_group(type=NORMEL_TILE_TYPE,  addrs=[UB_DKV_ADDR], mutex_ids=[4])

        dScoreCastTileGroup = pl.make_tile_group(type=CAST_TILE_TYPE, addrs=[UB_SM_DSB_ADDR], mutex_ids=[3])
        dKvCastTileGroup = pl.make_tile_group(type=CAST_TILE_TYPE, addrs=[UB_DKV_ADDR], mutex_ids=[4])

        apeTileGroup = pl.make_tile_group(type=NORMEL_TILE_TYPE, addrs=[UB_KV_APE_ADDR], mutex_ids=[2]) # TODO:可以直接tile复用

        dScoreCastMDTileGroup = pl.make_tile_group(type=CAST_MD_TILE_TYPE, addrs=[UB_SM_DSB_ADDR], mutex_ids=[3])
        dKvCastMDTileGroup = pl.make_tile_group(type=CAST_MD_TILE_TYPE, addrs=[UB_DKV_ADDR], mutex_ids=[4])

        dScoreNzTileGroup = pl.make_tile_group(type=NZ_TILE_TYPE, addrs=[UB_DSB_NZ_ADDR], mutex_ids=[3])
        dKvNzTileGroup = pl.make_tile_group(type=NZ_TILE_TYPE, addrs=[UB_DKV_NZ_ADDR], mutex_ids=[4])

        dScoreNzMDPrevTileGroup = pl.make_tile_group(type=NZ_MD_TILE_TYPE, addrs=[UB_DSB_NZ_ADDR], mutex_ids=[3])
        dKvNzMDPrevTileGroup = pl.make_tile_group(type=NZ_MD_TILE_TYPE, addrs=[UB_DKV_NZ_ADDR], mutex_ids=[4])
        dScoreNzMDCurTileGroup = pl.make_tile_group(type=NZ_MD_TILE_TYPE, addrs=[UB_DSB_NZ_MD_CUR_ADDR], mutex_ids=[3])
        dKvNzMDCurTileGroup = pl.make_tile_group(type=NZ_MD_TILE_TYPE, addrs=[UB_DKV_NZ_MD_CUR_ADDR], mutex_ids=[4])

        tempTileGroup  = pl.make_tile_group(type=TEMP_TILE_TYPE, addrs=UB_DC_F16_ADDR, mutex_ids=[0, 1]) # 16kB = 0x4000

        moveXTileGroup = pl.make_tile_group(type=MOVE_X_TILE_TYPE,  addrs=[UB_SM_DSB_ADDR, UB_DKV_ADDR], mutex_ids=[3, 4])

        # Phase 3 reduce: 复用 0x0C000/0x1C000/0x2C000 三个 64KB 区域
        reduceAccTileGroup = pl.make_tile_group(type=FLAT_TILE_TYPE, addrs=[UB_KV_APE_ADDR], mutex_ids=[2])
        reduceLoadTileGroup = pl.make_tile_group(type=FLAT_TILE_TYPE, addrs=[UB_SM_DSB_ADDR, UB_DKV_ADDR], mutex_ids=[3, 4])

        # Phase 3 dX reduce: acc/ld 复用 reduce_acc/reduce_ld0，仅新增 cast
        reduceResultTileGroup = pl.make_tile_group(type=FLAT_CAST_TILE_TYPE, addrs=[UB_KV_APE_ADDR], mutex_ids=[2])

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
            [groupNum * coffCoef * cmpRowCnt,  headDim],
            [headDim, 1], dtype=pl.DT_FP32)


        tensorDApeWsFlat = pl.make_tensor(workSpacePtr,
            [groupNum * coffCoef, cmpSize],
            [cmpSize, 1], dtype=pl.DT_FP32)

        tensorDWKvWsFlat = pl.make_tensor(dWKvWorkSpacePtr,
            [dWeightWorkSpaceSize // (totalHeadDim * hiddenSize), totalHeadDim * hiddenSize],
            [totalHeadDim * hiddenSize, 1], dtype=pl.DT_FP32)
            
        tensorDWGateWsFlat = pl.make_tensor(dWGateWorkSpacePtr,
            [dWeightWorkSpaceSize // (totalHeadDim * hiddenSize), totalHeadDim * hiddenSize],
            [totalHeadDim * hiddenSize, 1], dtype=pl.DT_FP32)

        tensorDApeFlat = pl.make_tensor(dApe, [1, cmpSize],
                                         [cmpSize, 1], dtype=pl.DT_FP32)
        # dW 输出为 FP16/BF16（与 op 注册一致），归约在 FP32 完成、写回前 cast
        tensorDWKvFlat = pl.make_tensor(dWkv, [1, totalHeadDim * hiddenSize],
                                         [totalHeadDim * hiddenSize, 1], dtype=ioDType)
        tensorDWGateFlat = pl.make_tensor(dWgate, [1, totalHeadDim * hiddenSize],
                                         [totalHeadDim * hiddenSize, 1], dtype=ioDType)



    # ═════════════════════════════════════════════════════════════════════════
    #  Cube tile declarations (once)
    # ═════════════════════════════════════════════════════════════════════════
    with pl.section_cube():
        cubeCoreIdx     = pl.get_block_idx()
        groupIdx         = cubeCoreIdx // groupSize
        intraGroupIdx    = cubeCoreIdx % groupSize

        nStart = intraGroupIdx * D_BASE_SIZE
        # nEnd   = nStart + D_BASE_SIZE
        

        # L1A CV 数据: 复用主函数体 tile_group, 通过 .next() db

        # L1W: wkv/wgate chunk 加载（double-buffer, 0x40000-0x4FFFF, 64KB）
        l1_w_db = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_SIZE, H_L1_BASE_SIZE], dtype=ioDType,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ, compact=1),
            addrs=L1_W_ADDR, mutex_ids=[20, 21])

        # L0 tiles
        cv_l0a = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_SIZE, BASE_SIZE], dtype=ioDType,
                             target_memory=pl.MemorySpace.Left, layout=pl.NZ, compact=1),
            addrs=[0x00000, 0x08000], mutex_ids=[18, 19])
        cv_l0b = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_SIZE, H_BASE_SIZE], dtype=ioDType,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN, compact=1),
            addrs=[0x00000, 0x08000], mutex_ids=[22, 23])
        cv_l0c = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_SIZE, H_BASE_SIZE], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024, compact=1),
            addrs=0x00000, mutex_ids=[24, 25, 26, 27])

        # GM tensor views
        wkv_t = pl.make_tensor(wkv, [totalHeadDim, hiddenSize],
                               [hiddenSize, 1], dtype=ioDType)
        wgate_t = pl.make_tensor(wgate, [totalHeadDim, hiddenSize],
                                 [hiddenSize, 1], dtype=ioDType)

        # dX partial workspace (dkv@wkv) + dsb@wgate workspace



    # ═════════════════════════════════════════════════════════════════════════
    #  无压缩块（totalValid=0 / compressedCnt=0）特判：放所有 section 外部，
    #  主函数支持多个 return——vec 侧四输出刷 0 后返回，cube 侧无代码直接返回，
    #  均不进 round 循环（d_ape/d_wkv/d_wgate 归约在 validRows=0 时虽隐式写 0，
    #  d_x 的 _reduce_dx 用回绕节点不可依赖——统一在此显式处理）
    # ═════════════════════════════════════════════════════════════════════════
    if compressedCnt == 0:
        with pl.section_vector():
            _zero_outputs(coreIdx, vecCtx.coreNum, xRows, hiddenSize, totalHeadDim,
                        tiling.cmpSize, dXOutGm, tensorDWKvFlat, tensorDWGateFlat,
                        tensorDApeFlat, reduce_acc, p3_cast)
        return
    # ════════════════════════════════════════════════════════════
    #  Phase 1+3: Vec Scatter & Reduce (per-round compute)
    # ════════════════════════════════════════════════════════════
    for roundIdx in pl.range(0, roundCnt + 1):
        roundBlocks = min(totalScNumPerRound,
                          compressedCnt - roundIdx * totalScNumPerRound)
        dbIdx = roundIdx % 2
        
        with pl.section_vector():
            if roundIdx < roundCnt:
                preDealTcSize = groupIdx * groupDealScNum
                if Coff == 1 and coreIdx % 2 == 1:
                    preDealTcSize += dealScNum
                dealTcSize   = min(dealScNum,
                                   max(0, roundBlocks - preDealTcSize))
                blocksAfter  = max(0, roundBlocks - preDealTcSize - dealTcSize)

                # ── 块遍历节点规划（所有核：scatter/x 用 before/after，dX reduce 用 start/end）──
                bIdxStart, scIdxStart   = bIdx, scIdx
                bIdxBefore, scIdxBefore = _skip_blocks(preDealTcSize, bIdxStart, scIdxStart, startPos, seqUsed, cuSeqlens, seq_ctx)
                bIdxAfter, scIdxAfter   = _skip_blocks(dealTcSize,  bIdxBefore, scIdxBefore, startPos, seqUsed, cuSeqlens, seq_ctx)
                bIdxEnd, scIdxEnd       = _skip_blocks(blocksAfter, bIdxAfter, scIdxAfter,   startPos, seqUsed, cuSeqlens, seq_ctx)

                if preDealTcSize < roundBlocks:
                    # Segment 2: process（scatter，before → after）
                    bIdx, scIdx = bIdxBefore, scIdxBefore
                    _process_scatter(vecCtx, roundIdx, totalScNumPerRound, preDealTcSize, dealTcSize,
                                     bIdx, scIdx,
                                     tile_dc_f16, tile_dc_f32, tile_kv, tile_softmax, tile_temp,
                                     dCmpKvGm, kvGm, softmaxScoreGm, nStart,
                                     startPos, seqUsed, cuSeqlens, seq_ctx)

                    nTileRows = dealTcSize * cmpRowCnt

                    # APE: local_ape 累加 + 写 workspace（round0 清零 / round>0 读旧值，VF 内累加，一次 MTE3）
                    localSlot = groupIdx * coffCoef * cmpRowCnt
                    if Coff == 1 and coreIdx % 2 == 1:
                        localSlot += cmpRowCnt
                    pl.set_validshape(tile_ape_local, [cmpRowCnt, dDealSize])
                    if roundIdx == 0:
                        _vf_tile_zero(tile_ape_local, cmpRowCnt, dDealSize)
                    else:
                        pl.load(tile_ape_local, tensorApe, [localSlot, nStart])
                    _vf_reduce_dscore_to_ape(
                        tile_softmax, tile_ape_local,
                        dealTcSize, cmpRowCnt, dDealSize, dDealSize,
                    )
                    pl.store(tensorApe, tile_ape_local, [localSlot, nStart])
                    sub_idx = coreIdx % 2
                    # ── Cast dkv/dsb FP32→BF16/FP16 + ND→NZ + insert L1（子函数）──
                    _cast_dkv_dsb_to_l1(vecCtx, nTileRows, dealTcSize, sub_idx,
                                        tile_temp, tile_dkv_cast, tile_dkv_cast_md, tile_dkv_nz,
                                        tile_dkv_nz_md_prev, tile_dkv_nz_md_cur,
                                        tile_dsb_cast, tile_dsb_cast_md, tile_dsb_nz, tile_softmax,
                                        tile_dsb_nz_md_prev, tile_dsb_nz_md_cur,
                                        tempTileGroup, L1Kv0TileGroupNz, L1Kv1TileGroupNz, L1Score0TileGroupNz, L1Score1TileGroupNz)



                    dbRowCnt = tiling.dbRowCnt
                    # 搬运x（subSlot: coff=1 的 M 轴子核槽位偏移；coff=2 恒 0）
                    bIdx, scIdx = bIdxBefore, scIdxBefore
                    subSlot = subCoreIdx * dealScNum if Coff == 1 else 0
                    _arrange_x(vecCtx, dbIdx, groupIdx, dbRowCnt, dealTcSize,
                               bIdx, scIdx, subSlot, moveXTileGroup, xGm, xArrangeGm,
                               startPos, seqUsed, cuSeqlens, seq_ctx)

                # ── CV 通路: 本轮 v1 完成即发信号（供 cube c1(i) 消费）──
                bIdx, scIdx = bIdxEnd, scIdxEnd
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=dbIdx,
                    sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

                # ── 本轮快照存入 cur（V2(i-1) 结束后才轮换为 prev，避免被本轮 V2 读到）──
                curDbIdx       = dbIdx
                curBIdxStart   = bIdxStart
                curScIdxStart  = scIdxStart
                curBIdxEnd     = bIdxEnd
                curScIdxEnd    = scIdxEnd
                if roundIdx == 0:
                    prevDbIdx       = curDbIdx
                    prevBIdxStart   = curBIdxStart
                    prevScIdxStart  = curScIdxStart
                    prevBIdxEnd     = curBIdxEnd
                    prevScIdxEnd    = curScIdxEnd

            if roundIdx > 0:
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)

                # ── Phase 3: dX reduce（上一轮 start/end 节点，每轮执行）──
                _reduce_dx(vecCtx, coreIdx, prevDbIdx, roundIdx - 1,
                           prevBIdxStart, prevScIdxStart, prevBIdxEnd, prevScIdxEnd,
                           reduce_acc, reduceLoadTileGroup, p3_cast,
                           dXCacheGm, dXResultGm, dXOutGm,
                           startPos, seqUsed, cuSeqlens, seq_ctx)

                # ── 末轮：dApe/dW 一次性跨核归约（生产者侧已按轮累加至 ws 固定槽）──
                # roundBlocks 传 maxRoundBlocks：validRows 只覆盖写过数据的行
                #（不满轮时 groupIdx >= ceil(maxRoundBlocks/groupDealScNum) 的核未执行
                #  Phase 1/2 写入，其 ws 区域是初始值，不能参与归约）
                if roundIdx == roundCnt:
                    _reduce_ape(vecCtx, coreIdx, groupIdx, 0, maxRoundBlocks,
                                reduce_acc, reduceLoadTileGroup, tensorDApeFlat, tensorDApeWsFlat)
                    _reduce_d_weight(vecCtx, coreIdx, groupIdx, 0, maxRoundBlocks,
                                     reduce_acc, reduceLoadTileGroup, p3_cast,
                                     tensorDWKvFlat, tensorDWKvWsFlat, tensorDWGateFlat, tensorDWGateWsFlat)

                # ── V2 结束后轮换：本轮快照 cur → prev（供下一轮 V2(i) 使用）──
                prevDbIdx       = curDbIdx
                prevBIdxStart   = curBIdxStart
                prevScIdxStart  = curScIdxStart
                prevBIdxEnd     = curBIdxEnd
                prevScIdxEnd    = curScIdxEnd
            
        # ════════════════════════════════════════════════════════════
        #  Phase 2: Cube Matmul — dkv@wkv + dsb@wgate → d_x_partial
        # ════════════════════════════════════════════════════════════
        with pl.section_cube():
            if roundIdx < roundCnt:
                preDealTcSize = groupIdx * groupDealScNum
                dealTcSize   = min(groupDealScNum,
                                   max(0, roundBlocks - preDealTcSize))
                pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE1, event_id=dbIdx,
                    sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                if preDealTcSize < roundBlocks:
                    l1_kv0 = L1Kv0TileGroupNz.next()
                    l1_kv1 = L1Kv1TileGroupNz.next()
                    l1_sb0 = L1Score0TileGroupNz.next()
                    l1_sb1 = L1Score1TileGroupNz.next()

                    l1_kv0_t = L1Kv0TileGroupZn.next()
                    l1_kv1_t = L1Kv1TileGroupZn.next()
                    l1_sb0_t = L1Score0TileGroupZn.next()
                    l1_sb1_t = L1Score1TileGroupZn.next()


                    nTileRows0 = min(dealTcSize, dealScNum) * cmpRatio if Coff == 1 else dealTcSize * cmpRatio
                    nTileRows1 = max(dealTcSize - dealScNum, 0) * cmpRatio if Coff == 1 else dealTcSize * cmpRatio

                    matmulCtx = pl.struct("MatmulCtx",
                        hiddenSize=hiddenSize, nTileRows0=nTileRows0, nTileRows1=nTileRows1,
                        headDim=headDim, dealScNum=dealScNum,
                        cubeMBaseSize=cubeMBaseSize, cmpRatio=cmpRatio,
                        cubeCoreNum=tiling.cubeCoreNum, groupSize=tiling.groupSize,
                        groupNum=tiling.groupNum, totalHeadDim=tiling.totalHeadDim,
                        dbRowCnt=tiling.dbRowCnt, groupRowStride=tiling.groupRowStride)

                    # ── Matmul #1/#2：提取子函数（coff=1/2 统一）──
                    _compute_dx_partial(matmulCtx, dbIdx, cv_l0a, cv_l0b, cv_l0c, l1_w_db,
                                        l1_kv0, l1_kv1, l1_sb0, l1_sb1,
                                        wkv_t, wgate_t, dXResultGm)
                    _compute_dw_partial(matmulCtx, roundIdx, dbIdx, cv_l0a, cv_l0b, cv_l0c, l1_w_db,
                                        l1_kv0_t, l1_kv1_t, l1_sb0_t, l1_sb1_t,
                                        xArrangeGm, dWkvResultGm, dWGateResultGm)

                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
