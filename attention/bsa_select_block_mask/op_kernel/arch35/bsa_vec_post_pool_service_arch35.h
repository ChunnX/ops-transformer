/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bsa_vec_post_pool_service_arch35.h
 * \brief Post-Softmax Mean Pooling Service（softmax 融合版）
 *        在 SoftmaxSecondPassAndCast 归一化后的 FP32 tile 上直接完成 postXBlock × postYBlock
 *        二维均值折叠并写出 pooledScore，消除 attnScore 中转显存与独立 pooling 相位。
 *
 * 折叠流程（每 kChunk tile、每行切片）：
 *   1) 行折叠：ReduceSum(RA) 把 postXBlock 行折叠为列和 rowSumUb（超过单切片行数的单元
 *      跨切片折叠，Add 向量累计）
 *   2) 列折叠：ReduceSum(AR) 把 postYBlock 列段折叠为组值
 *   3) 跨 kChunk 的列组 → pendingPostYGroupUb / stagePendingUb（UB 内驻留，任意时刻至多 1 个列组在飞；
 *      postYBlock 大于 tile 列宽时整 tile 累计 pending，最后一个 tile 收尾）
 *      跨 qChunk 的行组 → postXGroupStagingGm（GM 部分和；片元固定归 aiv0，同核跨 chunk 合并）
 *   4) 分段除 count（标量 invCount + Muls）→ Cast FP16 → 分段写 GM
 *
 * 对齐约束：所有 VEC 访问 32B 对齐——
 *   - 单值量（lead/trail/pending/count 修正）一律放入 8-float 对齐槽，操作按 8 lane 进行，
 *     垃圾数据限制在 padding lane 内
 *   - emit 行布局分段对齐：[lead 槽@0 | fulls@8 | trail 槽@8+align8(nFull)]，
 *     AR 批量折叠永远落在 8 对齐段首
 */
#ifndef BSA_VEC_POST_POOL_SERVICE_ARCH35_H
#define BSA_VEC_POST_POOL_SERVICE_ARCH35_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bsa_select_block_mask_common_arch35.h"
#include "bsa_select_block_mask_tiling_data_arch35.h"

// ---- 融合 post pool 容量推导（全部由 tiling 常量推导，Q_CHUNK_SIZE/K_CHUNK_SIZE 调整自动适应）----
// 单个 kChunk tile 的列宽（k 方向不核间切分）
static constexpr uint32_t BSA_POST_POOL_COLS_PER_LOOP = CV_EXEC_RATIO * K_CHUNK_SIZE;
// postXBlock/postYBlock 的最小粒度（host 侧 BLOCK_SHAPE_ALIGN 对齐保证）
static constexpr uint32_t BSA_POST_POOL_MIN_GROUP_LEN = 8;
// 单个行切片的最大行数（等于 scoreUb 的行容量 Q_CHUNK_SIZE/2）
static constexpr uint32_t BSA_POST_POOL_ROWS_PER_SLICE = Q_CHUNK_SIZE / 2;
// 单值对齐槽宽度（32B / sizeof(float)），所有单值量的操作粒度
static constexpr uint32_t BSA_POST_POOL_SLOT_LANES = FLOAT_DATA_BLOCK_NUM;
// 每 AIV 每 chunk 的 emit 行数上限：lead(1) + 完整组(≤ 半 chunk/最小组) + 余量
static constexpr uint32_t BSA_POST_POOL_EMIT_ROWS_MAX = Q_CHUNK_SIZE / 2 / BSA_POST_POOL_MIN_GROUP_LEN + 2;
// 每 tile 的 emit 行跨距上限：lead 槽(8) + 完整列组(≤ 列宽/最小组) + trail 槽(8)
static constexpr uint32_t BSA_POST_POOL_EMIT_STRIDE_MAX =
    2 * BSA_POST_POOL_SLOT_LANES + BSA_POST_POOL_COLS_PER_LOOP / BSA_POST_POOL_MIN_GROUP_LEN;
// 每 chunk 的行处理单元上限：完整组 + lead/trail 片元 + 余量
static constexpr uint32_t BSA_POST_POOL_UNITS_MAX = Q_CHUNK_SIZE / BSA_POST_POOL_MIN_GROUP_LEN + 4;

template <typename BSAT>
class BSAVecPostPoolService {
public:
    using T = float;    ///< 折叠累计类型（与 softmax 归一化 tile 一致，FP32）
    using OUT_T = half; ///< pooledScore 输出类型

    /// softmax 两趟与 post pool 共用的行切片描述（chunk 内行号 + 本 AIV 局部行号）
    struct PostRowSlice {
        uint32_t chunkRowStart; ///< chunk 内起始行（scoreFp32 scratch 的行号）
        uint32_t localRowStart; ///< 本 AIV 局部起始行（softmax per-row UB 数组的下标）
        uint32_t rowCount;
    };

    __aicore__ inline BSAVecPostPoolService(){};

    __aicore__ inline void InitParams(const BSAConstInfo &constInfo,
                                      const optiling::BSASelectBlockMaskTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers(TBuf<> *uBuf, uint32_t offset, LocalTensor<uint8_t> sharedTemp);
    __aicore__ inline void InitGM(GlobalTensor<OUT_T> &pooledScoreGm, GlobalTensor<T> &stagingGm);

    /// 本 AIV 在 chunk 行区间 [qChunkStart, qChunkStart+chunkRows) 内的行切片划分。
    /// 非 post 模式退化为经典对半切分（aiv0 取 ceil 半）；post 模式按 postXBlock 组对齐，
    /// 跨 chunk 组片元固定归 aiv0（与 staging 归宿一致），保证 softmax 两趟切分完全一致。
    __aicore__ inline uint32_t GetChunkRowSlices(uint32_t qChunkStart, uint32_t chunkRows, uint32_t validXBlocks,
                                                 PostRowSlice *slices);

    /// chunk 生命周期开始：解析行处理单元、emit/staging 归属，重置 pending 状态
    __aicore__ inline void BeginChunk(uint32_t batchIdx, uint32_t headIdx, uint32_t qChunkStart, uint32_t chunkRows,
                                      uint32_t validXBlocks, uint32_t validYBlocks);
    /// 归一化 tile 的一个行切片折叠（tile 为 scoreUb 的 [slice.rowCount × kColsAlign] 视图）
    __aicore__ inline void FoldNormalizedSlice(LocalTensor<T> &tile, const PostRowSlice &slice, uint32_t kChunkStart,
                                               uint32_t kCols, uint32_t kColsAlign, bool isLastKTile);
    /// 本 tile 全部切片折叠完成：分段除 count、Cast、写出 pooledScore
    __aicore__ inline void FlushTile();

private:
    /// 行方向处理单元（一个 postXBlock 组或其跨 chunk 片元）
    struct PostXUnit {
        uint32_t chunkRowStart;  ///< chunk 内起始行
        uint32_t rowCount;       ///< 本 chunk 内行数
        uint32_t postXGroupIdx;  ///< 全局 px
        uint8_t fragmentKind;    ///< 0=chunk 内完整组 1=lead 片元 2=trail 片元
        uint8_t completesHere;   ///< 片元是否在本 chunk 内收尾（完整组恒为 1）
        uint8_t emitRowSlot;     ///< emit 单元在本 chunk 的行槽（仅 emit 单元有效）
        uint8_t stagePendingIdx; ///< staging 单元的列组 pending 槽（仅 staging 单元有效）
    };

    __aicore__ inline uint32_t ComputeChunkUnits(uint32_t qChunkStart, uint32_t chunkRows, uint32_t validXBlocks,
                                                 PostXUnit *units);
    __aicore__ inline void ComputeMineFlags(const PostXUnit *units, uint32_t unitNum, uint8_t *mineFlags);
    __aicore__ inline void FoldUnitColumns(uint32_t unitIdx, uint32_t kChunkStart, uint32_t kCols, uint32_t kColsAlign,
                                           bool isLastKTile);
    __aicore__ inline void DispatchUnitResult(uint32_t unitIdx, uint32_t gStart);
    __aicore__ inline void SegmentMulsCastOut(LocalTensor<T> rowBase, LocalTensor<OUT_T> fp16Base, uint32_t segOffset,
                                              uint32_t segLen, uint32_t gmWriteLen, float invCount, uint64_t gmColBase);
    __aicore__ inline void StagingReadSegment(uint32_t stageRow, uint32_t colGroupStart, uint32_t colCount,
                                              LocalTensor<T> mergeSeg);
    __aicore__ inline void StagingWriteSegment(uint32_t stageRow, uint32_t colGroupStart, uint32_t colCount,
                                               LocalTensor<T> srcSeg);
    __aicore__ inline uint32_t StageRowOf(uint32_t unitIdx);
    __aicore__ inline void InitEmitTile(uint32_t nFull);

    BSAConstInfo constInfo;
    const optiling::BSASelectBlockMaskTilingData *__restrict tilingData = nullptr;

    GlobalTensor<OUT_T> pooledScoreGmTensor;
    GlobalTensor<T> stagingGmTensor; ///< 跨 chunk 行组部分和 [aivNum*2][postYBlocks]

    LocalTensor<uint8_t> reduceSharedTemp; ///< 借用 softmax 相位的 reduce 临时空间

    // ---- UB 折叠缓冲 ----
    LocalTensor<T> rowSumUb;       ///< 行折叠输出 [BSA_POST_POOL_COLS_PER_LOOP]
    LocalTensor<T> rowSumTmpUb;    ///< 跨切片累计临时
    LocalTensor<T> colFoldTmpUb;   ///< lead/trail 单值列折叠临时（8 lane 对齐槽）
    LocalTensor<T> stagingMergeUb; ///< staging 读回合并（与 emit 行同布局）
    LocalTensor<T> emitSumUb;      ///< emit 矩阵（FP32，分段对齐行布局）
    LocalTensor<OUT_T> emitFp16Ub; ///< emit 矩阵 FP16（段偏移 = 2×fp32 段偏移，保证 16-half/32B 对齐）
    LocalTensor<T> pendingPostYGroupUb; ///< emit 行的列组跨 tile 部分和 [EMIT_ROWS_MAX × 8 槽]
    LocalTensor<T> stageColSumUb;       ///< staging 单元本 tile 折叠结果（与 emit 行同布局）
    LocalTensor<T> stagePendingUb;      ///< staging 单元的列组跨 tile 部分和 [2 × 8 槽]

    // ---- chunk 级标量状态（索引簿记，不参与 UB 张量计算）----
    PostXUnit units[BSA_POST_POOL_UNITS_MAX];
    uint8_t unitIsMine[BSA_POST_POOL_UNITS_MAX];
    uint32_t unitFoldedRows[BSA_POST_POOL_UNITS_MAX];
    uint32_t unitNum = 0;
    uint32_t emitRowCount = 0;     ///< 本 chunk 本 AIV 的 emit 行数
    uint32_t emitFirstPx = 0;      ///< emit 首行全局 px
    uint32_t emitTileInited = 0;   ///< 本 tile 布局是否已初始化
    uint32_t emitTileFullsOff = 0; ///< fulls 段起始（恒 8）
    uint32_t emitTileTrailOff = 0; ///< trail 槽起始（8 + align8(nFullMain)）
    uint32_t emitTileStride = 0;   ///< 行跨距
    uint32_t emitTileLeadCol = 0;  ///< lead 段是否有产出（0/1）
    uint32_t emitTileFullsLen = 0; ///< fulls 段产出列数
    uint32_t emitTileTrailCol = 0; ///< trail 段是否有产出（0/1）
    uint32_t emitTileColStart = 0; ///< 本 tile emit 起始全局列组
    uint32_t validXBlocksCur = 0;
    uint32_t validYBlocksCur = 0;
    uint32_t validPostXCur = 0;
    uint32_t validPostYCur = 0;
    uint64_t pooledHeadOffset = 0;
    uint32_t chunkParity = 0; ///< qChunkStart / qChunkSize 的奇偶（staging 槽位推导）

    event_t eventPost = (event_t)2; ///< post 折叠相位 V↔MTE 同步（pool:0/1, softmax:3 之外）
};

template <typename BSAT>
__aicore__ inline void BSAVecPostPoolService<BSAT>::InitParams(
    const BSAConstInfo &constInfo, const optiling::BSASelectBlockMaskTilingData *__restrict tilingData)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;
}

template <typename BSAT>
__aicore__ inline void BSAVecPostPoolService<BSAT>::InitBuffers(TBuf<> *uBuf, uint32_t offset,
                                                                LocalTensor<uint8_t> sharedTemp)
{
    // 在 softmax 缓冲之后继续 carve，合计约 12KB，叠加 softmax ~162KB 后仍在 192KB 内
    reduceSharedTemp = sharedTemp;
    uint32_t ubOffset = offset;

    rowSumUb = uBuf->GetWithOffset<T>(BSA_POST_POOL_COLS_PER_LOOP, ubOffset);
    ubOffset += BSA_POST_POOL_COLS_PER_LOOP * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    rowSumTmpUb = uBuf->GetWithOffset<T>(BSA_POST_POOL_COLS_PER_LOOP, ubOffset);
    ubOffset += BSA_POST_POOL_COLS_PER_LOOP * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    stagingMergeUb = uBuf->GetWithOffset<T>(BSA_POST_POOL_EMIT_STRIDE_MAX, ubOffset);
    stageColSumUb =
        uBuf->GetWithOffset<T>(BSA_POST_POOL_EMIT_STRIDE_MAX, ubOffset + BSA_POST_POOL_EMIT_STRIDE_MAX * sizeof(T));
    ubOffset += 2 * BSA_POST_POOL_EMIT_STRIDE_MAX * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    emitSumUb = uBuf->GetWithOffset<T>(BSA_POST_POOL_EMIT_ROWS_MAX * BSA_POST_POOL_EMIT_STRIDE_MAX, ubOffset);
    ubOffset += BSA_POST_POOL_EMIT_ROWS_MAX * BSA_POST_POOL_EMIT_STRIDE_MAX * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    emitFp16Ub = uBuf->GetWithOffset<OUT_T>(BSA_POST_POOL_EMIT_ROWS_MAX * BSA_POST_POOL_EMIT_STRIDE_MAX * 2, ubOffset);
    ubOffset += BSA_POST_POOL_EMIT_ROWS_MAX * BSA_POST_POOL_EMIT_STRIDE_MAX * 2 * sizeof(OUT_T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    pendingPostYGroupUb = uBuf->GetWithOffset<T>(BSA_POST_POOL_EMIT_ROWS_MAX * BSA_POST_POOL_SLOT_LANES, ubOffset);
    ubOffset += BSA_POST_POOL_EMIT_ROWS_MAX * BSA_POST_POOL_SLOT_LANES * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    stagePendingUb = uBuf->GetWithOffset<T>(2 * BSA_POST_POOL_SLOT_LANES, ubOffset);
    ubOffset += 2 * BSA_POST_POOL_SLOT_LANES * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    colFoldTmpUb = uBuf->GetWithOffset<T>(BSA_POST_POOL_SLOT_LANES, ubOffset);
}

template <typename BSAT>
__aicore__ inline void BSAVecPostPoolService<BSAT>::InitGM(GlobalTensor<OUT_T> &pooledScoreGm,
                                                           GlobalTensor<T> &stagingGm)
{
    this->pooledScoreGmTensor = pooledScoreGm;
    this->stagingGmTensor = stagingGm;
}

/**
 * @brief 解析 chunk 的行处理单元（lead 片元 / 完整组 / trail 片元，行区间并集恰为 chunk 行区间）
 */
template <typename BSAT>
__aicore__ inline uint32_t BSAVecPostPoolService<BSAT>::ComputeChunkUnits(uint32_t qChunkStart, uint32_t chunkRows,
                                                                          uint32_t validXBlocks, PostXUnit *units)
{
    const uint32_t postXBlock = constInfo.postBlockShapeX;
    uint32_t unitNum = 0;

    // lead 片元：前一 chunk 遗留组在本 chunk 的剩余行（chunk 起点与组边界对齐时 leadLen==0，无片元）
    uint32_t leadLen = (postXBlock - qChunkStart % postXBlock) % postXBlock;
    uint32_t leadPx = qChunkStart / postXBlock;
    uint32_t leadEndEff = BSAMin((leadPx + 1) * postXBlock, validXBlocks); // 组真实结束行（尾组受 validX 截断）
    leadLen = BSAMin(leadLen, leadEndEff - qChunkStart); // 仅 clamp：尾组受 validX 截断时缩短
    uint32_t leadRowsInChunk = BSAMin(leadLen, chunkRows);
    uint32_t leadCompletes = (qChunkStart + leadRowsInChunk >= leadEndEff) ? 1 : 0;

    if (leadRowsInChunk > 0) {
        units[unitNum].chunkRowStart = 0;
        units[unitNum].rowCount = leadRowsInChunk;
        units[unitNum].postXGroupIdx = leadPx;
        units[unitNum].fragmentKind = 1;
        units[unitNum].completesHere = static_cast<uint8_t>(leadCompletes);
        unitNum++;
    }

    // 完整组：chunk 内整组开始的 postXBlock 行
    uint32_t fullStart = leadRowsInChunk;
    uint32_t remaining = chunkRows - fullStart;
    uint32_t trailLen = remaining % postXBlock;
    uint32_t nFull = remaining / postXBlock;
    uint32_t fullPx = (qChunkStart + fullStart) / postXBlock;
    for (uint32_t i = 0; i < nFull; i++) {
        units[unitNum].chunkRowStart = fullStart + i * postXBlock;
        units[unitNum].rowCount = postXBlock;
        units[unitNum].postXGroupIdx = fullPx + i;
        units[unitNum].fragmentKind = 0;
        units[unitNum].completesHere = 1;
        unitNum++;
    }

    // trail 片元：本 chunk 起始、延伸到后续 chunk 的组头（末组受 validX 截断时在本 chunk 收尾）
    if (trailLen > 0) {
        uint32_t trailStart = fullStart + nFull * postXBlock;
        uint32_t trailPx = (qChunkStart + trailStart) / postXBlock;
        uint32_t trailEndEff = BSAMin((trailPx + 1) * postXBlock, validXBlocks);
        uint32_t trailCompletes = (qChunkStart + trailStart + trailLen >= trailEndEff) ? 1 : 0;
        units[unitNum].chunkRowStart = trailStart;
        units[unitNum].rowCount = trailLen;
        units[unitNum].postXGroupIdx = trailPx;
        units[unitNum].fragmentKind = 2;
        units[unitNum].completesHere = static_cast<uint8_t>(trailCompletes);
        unitNum++;
    }
    return unitNum;
}

/**
 * @brief 单元 AIV 归属：片元归 aiv0；完整组前段 aiv0 / 后段 aiv1（按负载均衡）。
 *        纯函数，不改服务状态；GetChunkRowSlices / BeginChunk 共用，保证归属一致。
 */
template <typename BSAT>
__aicore__ inline void BSAVecPostPoolService<BSAT>::ComputeMineFlags(const PostXUnit *units, uint32_t unitNum,
                                                                     uint8_t *mineFlags)
{
    const uint32_t postXBlock = constInfo.postBlockShapeX;
    uint32_t fragRows = 0;
    uint32_t nFull = 0;
    uint32_t firstFullIdx = 0;
    for (uint32_t i = 0; i < unitNum; i++) {
        if (units[i].fragmentKind != 0) {
            fragRows += units[i].rowCount;
        } else {
            if (nFull == 0) {
                firstFullIdx = i;
            }
            nFull++;
        }
    }
    // aiv0 目标完整组数：补齐 aiv0 片元负载后与 aiv1 负载均衡
    uint32_t nFullToAiv0 = 0;
    if (nFull > 0) {
        uint32_t budget =
            2 * BSA_POST_POOL_ROWS_PER_SLICE > fragRows ? (2 * BSA_POST_POOL_ROWS_PER_SLICE - fragRows) / 2 : 0;
        nFullToAiv0 = BSAMin(nFull, budget / postXBlock);
    }
    for (uint32_t i = 0; i < unitNum; i++) {
        if (units[i].fragmentKind != 0) {
            mineFlags[i] = static_cast<uint8_t>(constInfo.subBlockIdx == 0);
        } else {
            mineFlags[i] = static_cast<uint8_t>(constInfo.subBlockIdx == 0 ? (i < firstFullIdx + nFullToAiv0) :
                                                                             (i >= firstFullIdx + nFullToAiv0));
        }
    }
}

template <typename BSAT>
__aicore__ inline uint32_t BSAVecPostPoolService<BSAT>::GetChunkRowSlices(uint32_t qChunkStart, uint32_t chunkRows,
                                                                          uint32_t validXBlocks, PostRowSlice *slices)
{
    if (chunkRows == 0) {
        return 0;
    }

    if (!constInfo.usePostBlockShape) {
        // 非 post 模式：经典对半切分（与原 softmax 切分保持一致）
        uint32_t half = chunkRows / 2 + chunkRows % 2;
        if (constInfo.subBlockIdx == 0) {
            slices[0] = {0, 0, half};
            return half > 0 ? 1 : 0;
        }
        if (half >= chunkRows) {
            return 0; // aiv1 无行
        }
        slices[0] = {half, 0, chunkRows - half};
        return 1;
    }

    // post 模式：解析单元并归属本 AIV，贪心打包成 ≤ BSA_POST_POOL_ROWS_PER_SLICE 行的切片
    PostXUnit localUnits[BSA_POST_POOL_UNITS_MAX];
    uint8_t localMine[BSA_POST_POOL_UNITS_MAX];
    uint32_t totalUnits = ComputeChunkUnits(qChunkStart, chunkRows, validXBlocks, localUnits);
    ComputeMineFlags(localUnits, totalUnits, localMine);

    // 贪心打包：整单元入当前切片，超行上限则新开切片；超大单元按行上限拆分（折叠侧跨切片累计）
    uint32_t sliceNum = 0;
    uint32_t localRow = 0;
    for (uint32_t i = 0; i < totalUnits; i++) {
        if (!localMine[i]) {
            continue;
        }
        uint32_t rowsLeft = localUnits[i].rowCount;
        uint32_t unitRow = localUnits[i].chunkRowStart;
        while (rowsLeft > 0) {
            uint32_t rows = BSAMin(rowsLeft, BSA_POST_POOL_ROWS_PER_SLICE);
            slices[sliceNum] = {unitRow, localRow, rows};
            sliceNum++;
            localRow += rows;
            unitRow += rows;
            rowsLeft -= rows;
        }
    }
    return sliceNum;
}

template <typename BSAT>
__aicore__ inline void BSAVecPostPoolService<BSAT>::BeginChunk(uint32_t batchIdx, uint32_t headIdx,
                                                               uint32_t qChunkStart, uint32_t chunkRows,
                                                               uint32_t validXBlocks, uint32_t validYBlocks)
{
    const uint32_t postXBlock = constInfo.postBlockShapeX;
    const uint32_t postYBlock = constInfo.postBlockShapeY;

    validXBlocksCur = validXBlocks;
    validYBlocksCur = validYBlocks;
    validPostXCur = BSACeilDiv(validXBlocks, postXBlock);
    validPostYCur = BSACeilDiv(validYBlocks, postYBlock);
    pooledHeadOffset = (static_cast<uint64_t>(batchIdx) * constInfo.numHeads + headIdx) * constInfo.postXBlocks *
                       constInfo.postYBlocks;
    chunkParity = (qChunkStart / constInfo.qChunkSize) & 1;

    unitNum = ComputeChunkUnits(qChunkStart, chunkRows, validXBlocks, units);
    ComputeMineFlags(units, unitNum, unitIsMine);

    // 槽位分配：emit 行槽按 px 升序；staging 单元的列组 pending 槽至多 2
    emitRowCount = 0;
    emitFirstPx = 0;
    uint32_t stagePendingNext = 0;
    bool emitFirstSet = false;
    for (uint32_t i = 0; i < unitNum; i++) {
        unitFoldedRows[i] = 0;
        units[i].emitRowSlot = 0;
        units[i].stagePendingIdx = 0;
        if (!unitIsMine[i]) {
            continue;
        }
        // emit 单元：完整组或在本地收尾的 lead 片元；staging 单元：trail 头与未收尾的中段片元
        bool isEmit = (units[i].fragmentKind == 0) || (units[i].completesHere != 0);
        if (isEmit) {
            units[i].emitRowSlot = static_cast<uint8_t>(emitRowCount);
            if (!emitFirstSet) {
                emitFirstPx = units[i].postXGroupIdx;
                emitFirstSet = true;
            }
            emitRowCount++;
        } else {
            units[i].stagePendingIdx = static_cast<uint8_t>(stagePendingNext);
            stagePendingNext++;
        }
    }

    // 列组 pending 生命周期从零开始（tile 0 无 lead，trail 从零累计）
    AscendC::Duplicate(pendingPostYGroupUb, 0.0f, BSA_POST_POOL_EMIT_ROWS_MAX * BSA_POST_POOL_SLOT_LANES);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Duplicate(stagePendingUb, 0.0f, 2 * BSA_POST_POOL_SLOT_LANES);
    AscendC::PipeBarrier<PIPE_V>();

    emitTileInited = 0;
}

/**
 * @brief 归一化 tile 的一个行切片折叠：行折叠（RA）→ 单元收尾时列折叠
 */
template <typename BSAT>
__aicore__ inline void BSAVecPostPoolService<BSAT>::FoldNormalizedSlice(LocalTensor<T> &tile, const PostRowSlice &slice,
                                                                        uint32_t kChunkStart, uint32_t kCols,
                                                                        uint32_t kColsAlign, bool isLastKTile)
{
    const uint32_t sliceEnd = slice.chunkRowStart + slice.rowCount;
    for (uint32_t i = 0; i < unitNum; i++) {
        if (!unitIsMine[i]) {
            continue;
        }
        uint32_t unitEnd = units[i].chunkRowStart + units[i].rowCount;
        uint32_t overlapStart = BSAMax(units[i].chunkRowStart, slice.chunkRowStart);
        uint32_t overlapEnd = BSAMin(unitEnd, sliceEnd);
        if (overlapStart >= overlapEnd) {
            continue;
        }

        uint32_t rows = overlapEnd - overlapStart;
        uint32_t tileRowOffset = overlapStart - slice.chunkRowStart;
        uint32_t rowFoldShape[] = {rows, kColsAlign};
        if (unitFoldedRows[i] == 0) {
            AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, true>(rowSumUb, tile[tileRowOffset * kColsAlign],
                                                                      reduceSharedTemp, rowFoldShape, true);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, true>(rowSumTmpUb, tile[tileRowOffset * kColsAlign],
                                                                      reduceSharedTemp, rowFoldShape, true);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(rowSumUb, rowSumUb, rowSumTmpUb, kColsAlign);
            AscendC::PipeBarrier<PIPE_V>();
        }
        unitFoldedRows[i] += rows;

        if (unitFoldedRows[i] >= units[i].rowCount) {
            FoldUnitColumns(i, kChunkStart, kCols, kColsAlign, isLastKTile);
            unitFoldedRows[i] = 0; // 每 tile 完整折叠一次：完成后清零，供下一 tile 重新累计
        }
    }
}

/**
 * @brief 初始化本 tile 的 emit 行布局（tile 级一次，全部单元共用）
 */
template <typename BSAT>
__aicore__ inline void BSAVecPostPoolService<BSAT>::InitEmitTile(uint32_t nFullMain)
{
    if (emitTileInited != 0) {
        return;
    }
    emitTileInited = 1;
    emitTileFullsOff = BSA_POST_POOL_SLOT_LANES;
    emitTileTrailOff = emitTileFullsOff +
                       (nFullMain + BSA_POST_POOL_SLOT_LANES - 1) / BSA_POST_POOL_SLOT_LANES * BSA_POST_POOL_SLOT_LANES;
    emitTileStride = emitTileTrailOff + BSA_POST_POOL_SLOT_LANES;
    emitTileLeadCol = 0;
    emitTileFullsLen = 0;
    emitTileTrailCol = 0;
}

/**
 * @brief 单元列折叠：
 *        leadClip ≥ kCols：整 tile 在同一列组内 → pending 累计（最后一个 tile 单列收尾）
 *        否则：lead 列段（pending 合并，落 lead 槽）→ 完整列组（AR 批量，落 fulls 段）
 *              → trail 列段（pending 累计 / 最后 tile 收尾，落 trail 槽）
 *        validY 尾列组恒走 trail 槽（与 trail 收尾互斥），保证 fulls 段 count 均一
 */
template <typename BSAT>
__aicore__ inline void BSAVecPostPoolService<BSAT>::FoldUnitColumns(uint32_t unitIdx, uint32_t kChunkStart,
                                                                    uint32_t kCols, uint32_t kColsAlign,
                                                                    bool isLastKTile)
{
    const PostXUnit &unit = units[unitIdx];
    const uint32_t postYBlock = constInfo.postBlockShapeY;
    const uint32_t c0 = kChunkStart;

    // 等待上一单元/上一 tile 的 MTE3 写出完成，才能覆写折叠结果缓冲
    SetFlag<HardEvent::MTE3_V>(eventPost);
    WaitFlag<HardEvent::MTE3_V>(eventPost);

    // emit 单元：完整组或在本 chunk 收尾的 lead 片元；staging 单元：trail 头 / 未收尾的中段片元
    bool isEmit = (unit.fragmentKind == 0) || (unit.completesHere != 0);
    LocalTensor<T> pendUb = pendingPostYGroupUb[unit.emitRowSlot * BSA_POST_POOL_SLOT_LANES];
    if (!isEmit) {
        pendUb = stagePendingUb[unit.stagePendingIdx * BSA_POST_POOL_SLOT_LANES];
    }

    uint32_t leadLen = (postYBlock - c0 % postYBlock) % postYBlock;
    uint32_t leadClip = BSAMin(leadLen, kCols);

    // 行布局目标（emit 行 / staging 行共用同布局）；布局在确定 nFullMain 后由 InitEmitTile 建立
    LocalTensor<T> rowBase = isEmit ? emitSumUb[unit.emitRowSlot * BSA_POST_POOL_EMIT_STRIDE_MAX] : stageColSumUb;

    if (leadClip >= kCols) {
        // 整 tile 落在同一列组内（postYBlock > tile 列宽）：整行累计 pending；最后一个 tile 单列收尾
        uint32_t wholeColShape[] = {1, kColsAlign};
        AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR, true>(colFoldTmpUb, rowSumUb[0], reduceSharedTemp,
                                                                  wholeColShape, true);
        AscendC::PipeBarrier<PIPE_V>();
        if (!isLastKTile) {
            AscendC::Add(pendUb, colFoldTmpUb, pendUb, BSA_POST_POOL_SLOT_LANES);
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }
        InitEmitTile(0);
        emitTileColStart = c0 / postYBlock;
        emitTileLeadCol = 1;
        AscendC::Add(rowBase[0], colFoldTmpUb, pendUb, BSA_POST_POOL_SLOT_LANES);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(pendUb, 0.0f, BSA_POST_POOL_SLOT_LANES);
        AscendC::PipeBarrier<PIPE_V>();
        DispatchUnitResult(unitIdx, emitTileColStart);
        return;
    }

    // 常规分解：lead 收尾（leadLen < kCols 时必收尾）→ 完整列组 → trail 列段
    bool leadCompletesHere = leadLen > 0;
    uint32_t fullStart = leadClip;
    uint32_t nFull = (kCols - fullStart) / postYBlock;
    uint32_t trailStart = fullStart + nFull * postYBlock;
    bool hasTrail = trailStart < kCols;
    // validY 尾列组若落在完整列组区间内，单独折入 trail 槽（此时无 trail 列段，互斥）
    uint32_t gFirstFull = (c0 + fullStart) / postYBlock;
    bool lastFullIsTail = (nFull > 0) && (gFirstFull + nFull == validPostYCur);
    uint32_t nFullMain = nFull - (lastFullIsTail ? 1 : 0);

    if (!leadCompletesHere && nFullMain == 0 && !hasTrail && !(isLastKTile && lastFullIsTail)) {
        // 本 tile 无产出列组：仅累计 trail pending（postYBlock 不整除列宽的中段 tile）
        if (hasTrail) {
            uint32_t trailColShape[] = {1, kColsAlign - trailStart};
            AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR, true>(colFoldTmpUb, rowSumUb[trailStart],
                                                                      reduceSharedTemp, trailColShape, true);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(pendUb, colFoldTmpUb, pendUb, BSA_POST_POOL_SLOT_LANES);
            AscendC::PipeBarrier<PIPE_V>();
        }
        return;
    }

    // 首个到达的单元以实际 nFullMain 定 tile 布局
    InitEmitTile(nFullMain);
    emitTileColStart = leadCompletesHere ? (c0 / postYBlock) : gFirstFull;

    if (leadCompletesHere) {
        // lead 列段：前一 tile pending + 本 tile 列段 → lead 槽（段偏移恒 0）
        uint32_t leadColShape[] = {1, leadClip};
        AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR, true>(colFoldTmpUb, rowSumUb[0], reduceSharedTemp,
                                                                  leadColShape, true);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(rowBase[0], colFoldTmpUb, pendUb, BSA_POST_POOL_SLOT_LANES);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(pendUb, 0.0f, BSA_POST_POOL_SLOT_LANES);
        AscendC::PipeBarrier<PIPE_V>();
        emitTileLeadCol = 1;
    }

    if (nFullMain > 0) {
        // 完整列组：AR 批量折叠，直接落 fulls 段（count 均一 postYBlock）
        uint32_t fullGroupShape[] = {nFullMain, postYBlock};
        AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR, true>(rowBase[emitTileFullsOff], rowSumUb[fullStart],
                                                                  reduceSharedTemp, fullGroupShape, true);
        AscendC::PipeBarrier<PIPE_V>();
        emitTileFullsLen = nFullMain;
    }

    if (lastFullIsTail) {
        // validY 尾列组（完整列组区间内的最后一个）：单独折入 trail 槽
        uint32_t tailColShape[] = {1, postYBlock};
        AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR, true>(rowBase[emitTileTrailOff],
                                                                  rowSumUb[fullStart + nFullMain * postYBlock],
                                                                  reduceSharedTemp, tailColShape, true);
        AscendC::PipeBarrier<PIPE_V>();
        emitTileTrailCol = 1;
    } else if (hasTrail) {
        // trail 列段：最后一个 tile 收尾（含对齐补零）落 trail 槽；否则累计 pending
        uint32_t trailColShape[] = {1, kColsAlign - trailStart};
        AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR, true>(colFoldTmpUb, rowSumUb[trailStart], reduceSharedTemp,
                                                                  trailColShape, true);
        AscendC::PipeBarrier<PIPE_V>();
        if (isLastKTile) {
            AscendC::Add(rowBase[emitTileTrailOff], colFoldTmpUb, pendUb, BSA_POST_POOL_SLOT_LANES);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(pendUb, 0.0f, BSA_POST_POOL_SLOT_LANES);
            AscendC::PipeBarrier<PIPE_V>();
            emitTileTrailCol = 1;
        } else {
            AscendC::Add(pendUb, colFoldTmpUb, pendUb, BSA_POST_POOL_SLOT_LANES);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    DispatchUnitResult(unitIdx, emitTileColStart);
}

/**
 * @brief 单元折叠结果的归宿分发：
 *        emit + lead 片元 → 读回 staging 合并进 emit 行；
 *        staging + lead 中段片元 → 读回 staging、合并、写回（RMW）；
 *        staging + trail 头片元 → 直接写 staging（下一 chunk 的 lead 合并）
 *        分段进行（lead 槽 / fulls 段 / trail 槽），全部对齐访问
 */
template <typename BSAT>
__aicore__ inline void BSAVecPostPoolService<BSAT>::DispatchUnitResult(uint32_t unitIdx, uint32_t gStart)
{
    const PostXUnit &unit = units[unitIdx];
    bool isEmit = (unit.fragmentKind == 0) || (unit.completesHere != 0);
    LocalTensor<T> rowBase = isEmit ? emitSumUb[unit.emitRowSlot * BSA_POST_POOL_EMIT_STRIDE_MAX] : stageColSumUb;

    // 分段表：{段偏移, 段长, 全局列组偏移}；lead/trail 槽真实内容恰 1 列，按精确长度搬运
    uint32_t segOff[3];
    uint32_t segLen[3];
    uint32_t segCol[3];
    uint32_t segNum = 0;
    if (emitTileLeadCol != 0) {
        segOff[segNum] = 0;
        segLen[segNum] = 1;
        segCol[segNum] = 0;
        segNum++;
    }
    if (emitTileFullsLen > 0) {
        segOff[segNum] = emitTileFullsOff;
        segLen[segNum] = emitTileFullsLen;
        segCol[segNum] = emitTileLeadCol;
        segNum++;
    }
    if (emitTileTrailCol != 0) {
        segOff[segNum] = emitTileTrailOff;
        segLen[segNum] = 1;
        segCol[segNum] = emitTileLeadCol + emitTileFullsLen;
        segNum++;
    }

    if (isEmit) {
        if (unit.fragmentKind == 1) {
            // lead 片元在本 chunk 收尾：合并前一 chunk 的 staging 部分和后进入 emit
            for (uint32_t s = 0; s < segNum; s++) {
                StagingReadSegment(StageRowOf(unitIdx), gStart + segCol[s], segLen[s], stagingMergeUb[segOff[s]]);
                AscendC::Add(rowBase[segOff[s]], rowBase[segOff[s]], stagingMergeUb[segOff[s]], segLen[s]);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        return;
    }
    if (unit.fragmentKind == 1) {
        // 中段片元（组跨 ≥3 个 chunk）：staging 读回 + 本 tile 折叠 + 写回（RMW）
        for (uint32_t s = 0; s < segNum; s++) {
            StagingReadSegment(StageRowOf(unitIdx), gStart + segCol[s], segLen[s], stagingMergeUb[segOff[s]]);
            AscendC::Add(rowBase[segOff[s]], rowBase[segOff[s]], stagingMergeUb[segOff[s]], segLen[s]);
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t s = 0; s < segNum; s++) {
            StagingWriteSegment(StageRowOf(unitIdx), gStart + segCol[s], segLen[s], rowBase[segOff[s]]);
        }
        return;
    }
    // trail 头片元：组的首个贡献，直接写 staging（下一 chunk 的 lead 以相同槽位合并）
    for (uint32_t s = 0; s < segNum; s++) {
        StagingWriteSegment(StageRowOf(unitIdx), gStart + segCol[s], segLen[s], rowBase[segOff[s]]);
    }
}

template <typename BSAT>
__aicore__ inline uint32_t BSAVecPostPoolService<BSAT>::StageRowOf(uint32_t unitIdx)
{
    // staging 行 = aiv0 编号 * 2 + 槽位奇偶：
    //   lead/中段片元读写 slot = chunkParity（前一 chunk 的 trail 以 chunkParity^1 写入，恰为本 chunk 奇偶）
    //   trail 头片元写 slot = chunkParity^1（下一 chunk 的 lead 以其 chunkParity 读，恰为本奇偶取反）
    const PostXUnit &unit = units[unitIdx];
    uint32_t aiv0Idx = constInfo.aivIdx & ~1u;
    uint32_t slot = (unit.fragmentKind == 2) ? (chunkParity ^ 1) : chunkParity;
    return aiv0Idx * 2 + slot;
}

template <typename BSAT>
__aicore__ inline void BSAVecPostPoolService<BSAT>::StagingReadSegment(uint32_t stageRow, uint32_t colGroupStart,
                                                                       uint32_t colCount, LocalTensor<T> mergeSeg)
{
    SetFlag<HardEvent::MTE3_MTE2>(eventPost);
    WaitFlag<HardEvent::MTE3_MTE2>(eventPost);
    SetFlag<HardEvent::V_MTE2>(eventPost);
    WaitFlag<HardEvent::V_MTE2>(eventPost);

    uint64_t offset = static_cast<uint64_t>(stageRow) * constInfo.postYBlocks + colGroupStart;
    DataCopyPad(mergeSeg, stagingGmTensor[offset],
                DataCopyExtParams(1, static_cast<uint32_t>(colCount * sizeof(T)), 0, 0, 0),
                DataCopyPadExtParams<T>{true, 0, 0, 0});

    SetFlag<HardEvent::MTE2_V>(eventPost);
    WaitFlag<HardEvent::MTE2_V>(eventPost);
}

template <typename BSAT>
__aicore__ inline void BSAVecPostPoolService<BSAT>::StagingWriteSegment(uint32_t stageRow, uint32_t colGroupStart,
                                                                        uint32_t colCount, LocalTensor<T> srcSeg)
{
    SetFlag<HardEvent::V_MTE3>(eventPost);
    WaitFlag<HardEvent::V_MTE3>(eventPost);

    uint64_t offset = static_cast<uint64_t>(stageRow) * constInfo.postYBlocks + colGroupStart;
    DataCopyPad(stagingGmTensor[offset], srcSeg,
                DataCopyExtParams(1, static_cast<uint32_t>(colCount * sizeof(T)), 0, 0, 0));
}

/**
 * @brief 分段收尾：标量 invCount 缩放 → Cast FP16 → 写 GM（GM 侧任意偏移由 DataCopyPad 处理）
 */
template <typename BSAT>
__aicore__ inline void BSAVecPostPoolService<BSAT>::SegmentMulsCastOut(LocalTensor<T> rowBase,
                                                                       LocalTensor<OUT_T> fp16Base, uint32_t segOffset,
                                                                       uint32_t segLen, uint32_t gmWriteLen,
                                                                       float invCount, uint64_t gmColBase)
{
    AscendC::Muls(rowBase[segOffset], rowBase[segOffset], invCount, segLen);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(fp16Base[segOffset * 2], rowBase[segOffset], AscendC::RoundMode::CAST_ROUND, segLen);
    AscendC::PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE3>(eventPost);
    WaitFlag<HardEvent::V_MTE3>(eventPost);
    // 计算按对齐 segLen（8 lane），GM 写出按精确 gmWriteLen（lead/trail 槽仅 1 列，避免溢出砸邻行）
    DataCopyPad(pooledScoreGmTensor[gmColBase], fp16Base[segOffset * 2],
                DataCopyExtParams(1, static_cast<uint32_t>(gmWriteLen * sizeof(OUT_T)), 0, 0, 0));
}

/**
 * @brief 本 tile 收尾：逐 emit 行、逐段除 count（标量 invCount）→ Cast → 写 GM
 *        count = rowsInPostXBlock(px) × colsInPostYBlock(全局列组)
 */
template <typename BSAT>
__aicore__ inline void BSAVecPostPoolService<BSAT>::FlushTile()
{
    if (emitRowCount == 0 || emitTileInited == 0) {
        emitTileInited = 0;
        return;
    }

    const uint32_t postXBlock = constInfo.postBlockShapeX;
    const uint32_t postYBlock = constInfo.postBlockShapeY;
    const uint32_t tailRowGroup = validPostXCur - 1;
    const uint32_t tailColGroup = validPostYCur - 1;

    // 等待上一 tile 的 MTE3 写出完成，才能覆写 emit 缓冲
    SetFlag<HardEvent::MTE3_V>(eventPost);
    WaitFlag<HardEvent::MTE3_V>(eventPost);

    for (uint32_t r = 0; r < emitRowCount; r++) {
        uint32_t px = emitFirstPx + r;
        uint32_t rowsInPx = (px == tailRowGroup) ? (validXBlocksCur - tailRowGroup * postXBlock) : postXBlock;
        float invRows = 1.0f / static_cast<float>(static_cast<int32_t>(rowsInPx));

        LocalTensor<T> rowBase = emitSumUb[r * BSA_POST_POOL_EMIT_STRIDE_MAX];
        LocalTensor<OUT_T> fp16Base = emitFp16Ub[r * BSA_POST_POOL_EMIT_STRIDE_MAX * 2];
        uint64_t gmRowBase = pooledHeadOffset + static_cast<uint64_t>(px) * validPostYCur + emitTileColStart;

        if (emitTileLeadCol != 0) {
            uint32_t g = emitTileColStart;
            uint32_t colsInG = (g == tailColGroup) ? (validYBlocksCur - tailColGroup * postYBlock) : postYBlock;
            float invCount = invRows / static_cast<float>(static_cast<int32_t>(colsInG));
            SegmentMulsCastOut(rowBase, fp16Base, 0, BSA_POST_POOL_SLOT_LANES, 1, invCount, gmRowBase);
        }
        if (emitTileFullsLen > 0) {
            float invCount = invRows / static_cast<float>(static_cast<int32_t>(postYBlock));
            SegmentMulsCastOut(rowBase, fp16Base, emitTileFullsOff, emitTileFullsLen, emitTileFullsLen, invCount,
                               gmRowBase + emitTileLeadCol);
        }
        if (emitTileTrailCol != 0) {
            uint32_t g = emitTileColStart + emitTileLeadCol + emitTileFullsLen;
            uint32_t colsInG = (g == tailColGroup) ? (validYBlocksCur - tailColGroup * postYBlock) : postYBlock;
            float invCount = invRows / static_cast<float>(static_cast<int32_t>(colsInG));
            SegmentMulsCastOut(rowBase, fp16Base, emitTileTrailOff, BSA_POST_POOL_SLOT_LANES, 1, invCount,
                               gmRowBase + emitTileLeadCol + emitTileFullsLen);
        }
    }

    emitTileInited = 0;
}

#endif // BSA_VEC_POST_POOL_SERVICE_ARCH35_H
