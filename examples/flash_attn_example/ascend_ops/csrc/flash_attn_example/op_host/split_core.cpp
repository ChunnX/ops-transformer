/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file split_core.cpp
 * \brief Split core v2 implementation for GQA non-quant tiling.
 *        Ported from ops-transformer split_core.cpp.
 */

#include "split_core.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

namespace split_core {

uint32_t GetS1SeqSize(uint32_t bIdx, const BaseInfo &baseInfo)
{
    if (baseInfo.actualSeqS1Size.empty()) {
        return baseInfo.s1Size;
    }

    if (baseInfo.actualLenQDims == 1U) {
        return static_cast<uint32_t>(baseInfo.actualSeqS1Size[0]);
    }

    if (!baseInfo.isAccumSeqS1) {
        return static_cast<uint32_t>(baseInfo.actualSeqS1Size[bIdx]);
    }

    return (bIdx == 0) ? static_cast<uint32_t>(baseInfo.actualSeqS1Size[bIdx]) :
                         static_cast<uint32_t>(baseInfo.actualSeqS1Size[bIdx] - baseInfo.actualSeqS1Size[bIdx - 1U]);
}

uint32_t GetS2SeqSize(uint32_t bIdx, const BaseInfo &baseInfo)
{
    if (baseInfo.actualSeqS2Size.empty()) {
        return baseInfo.s2Size;
    }

    if (baseInfo.actualLenKvDims == 1U) {
        return static_cast<uint32_t>(baseInfo.actualSeqS2Size[0]);
    }

    if (!baseInfo.isAccumSeqS2) {
        return static_cast<uint32_t>(baseInfo.actualSeqS2Size[bIdx]);
    }

    return (bIdx == 0) ?
               static_cast<uint32_t>(baseInfo.actualSeqS2Size[bIdx]) :
               static_cast<uint32_t>(baseInfo.actualSeqS2Size[bIdx] - baseInfo.actualSeqS2Size[bIdx - 1U]);
}

int64_t CalcPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const BaseInfo &baseInfo)
{
    auto mode = static_cast<SparseMode>(baseInfo.sparseMode);
    if (mode == SparseMode::BAND) {
        return static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + baseInfo.preToken;
    }
    return baseInfo.preToken;
}

int64_t CalcNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const BaseInfo &baseInfo)
{
    auto mode = static_cast<SparseMode>(baseInfo.sparseMode);
    switch (mode) {
        case SparseMode::DEFAULT_MASK:
        case SparseMode::ALL_MASK:
        case SparseMode::LEFT_UP_CAUSAL:
            return baseInfo.nextToken;
        case SparseMode::RIGHT_DOWN_CAUSAL:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        case SparseMode::BAND:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + baseInfo.nextToken;
        default:
            return baseInfo.nextToken;
    }
}

int64_t CalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t alignCoefM = 16U;
    uint32_t alignCoefS2 = 64U;
    uint32_t alignBasicM = (basicM + alignCoefM - 1U) >> 4U;
    uint32_t alignBasicS2 = (basicS2 + alignCoefS2 - 1U) >> 6U;
    return static_cast<int64_t>(6U * alignBasicM + 10U * alignBasicS2);
}

BlockCost<int64_t> CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize, uint32_t s1GTailSize,
                                 uint32_t s2TailSize)
{
    BlockCost<int64_t> typeCost{};
    typeCost[NORMAL_BLOCK][NORMAL_BLOCK] = CalcCost(s1NormalSize, s2NormalSize);
    typeCost[TAIL_BLOCK][NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2NormalSize);
    typeCost[NORMAL_BLOCK][TAIL_BLOCK] = (s2TailSize == 0U) ? 0U : CalcCost(s1NormalSize, s2TailSize);
    typeCost[TAIL_BLOCK][TAIL_BLOCK] = (s1GTailSize == 0U || s2TailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2TailSize);
    return typeCost;
}

Range<uint32_t> CalcS2Range(uint32_t s1GIdx, const BaseInfo &baseInfo, const SplitParam &splitParam,
                            const BatchCache &batchCache)
{
    uint32_t s2Start = 0U;
    uint32_t s2End = 0U;

    if (batchCache.s1Size == 0U || batchCache.s2Size == 0U) {
        return std::make_pair(s2Start, s2End);
    }

    if (!baseInfo.attnMaskFlag) {
        s2Start = 0U;
        s2End = (batchCache.s2Size + splitParam.s2BaseSize - 1U) / splitParam.s2BaseSize;
        return std::make_pair(s2Start, s2End);
    }

    int64_t s1GFirstToken = static_cast<int64_t>(s1GIdx) * static_cast<int64_t>(splitParam.mBaseSize);
    int64_t s1GLastToken = std::min(s1GFirstToken + static_cast<int64_t>(splitParam.mBaseSize),
                                    static_cast<int64_t>(batchCache.s1Size) * static_cast<int64_t>(baseInfo.gSize)) -
                           1;

    int64_t s1FirstToken = 0;
    int64_t s1LastToken = 0;
    if (baseInfo.isS1G) {
        s1FirstToken = s1GFirstToken / static_cast<int64_t>(baseInfo.gSize);
        s1LastToken = s1GLastToken / static_cast<int64_t>(baseInfo.gSize);
    } else {
        if (s1GFirstToken / batchCache.s1Size == s1GLastToken / batchCache.s1Size) {
            s1FirstToken = s1GFirstToken % static_cast<int64_t>(batchCache.s1Size);
            s1LastToken = s1GLastToken % static_cast<int64_t>(batchCache.s1Size);
        } else {
            s1FirstToken = 0;
            s1LastToken = batchCache.s1Size;
        }
    }

    int64_t s2FirstToken = s1FirstToken - batchCache.preTokenLeftUp;
    int64_t s2LastToken = s1LastToken + batchCache.nextTokenLeftUp;

    if (s2FirstToken >= batchCache.s2Size || s2LastToken < 0 || s2LastToken < s2FirstToken) {
        s2Start = 0U;
        s2End = 0U;
        return std::make_pair(s2Start, s2End);
    }

    s2FirstToken = Clip(s2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
    s2LastToken = Clip(s2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
    s2Start = static_cast<uint32_t>(s2FirstToken) / splitParam.s2BaseSize;
    s2End = static_cast<uint32_t>(s2LastToken) / splitParam.s2BaseSize + 1U;

    return std::make_pair(s2Start, s2End);
}

void CalcSplitInfo(SplitContext &splitContext)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitParam &splitParam = splitContext.splitParam;

    SplitInfo &splitInfo = splitContext.splitInfo;
    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        uint32_t s1Size = GetS1SeqSize(bIdx, baseInfo);
        uint32_t s2Size = GetS2SeqSize(bIdx, baseInfo);

        splitInfo.s1GBaseNum[bIdx] = (s1Size * baseInfo.gSize + (splitParam.mBaseSize - 1U)) / splitParam.mBaseSize;
        splitInfo.s1GTailSize[bIdx] = (s1Size * baseInfo.gSize) % splitParam.mBaseSize;
        splitInfo.s2BaseNum[bIdx] = (s2Size + splitParam.s2BaseSize - 1U) / splitParam.s2BaseSize;
        splitInfo.s2TailSize[bIdx] = s2Size % splitParam.s2BaseSize;
        if (splitInfo.s1GBaseNum[bIdx] != 0U && splitInfo.s2BaseNum[bIdx] != 0U) {
            splitInfo.isKvSeqAllZero = false;
        }
    }
}

void CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitParam &splitParam = splitContext.splitParam;
    const SplitInfo &splitInfo = splitContext.splitInfo;

    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1SeqSize(bIdx, baseInfo);
    batchCache.s2Size = GetS2SeqSize(bIdx, baseInfo);
    batchCache.preTokenLeftUp = CalcPreTokenLeftUp(batchCache.s1Size, batchCache.s2Size, baseInfo);
    batchCache.nextTokenLeftUp = CalcNextTokenLeftUp(batchCache.s1Size, batchCache.s2Size, baseInfo);
    batchCache.typeCost = CalcCostTable(splitParam.mBaseSize, splitParam.s2BaseSize, splitInfo.s1GTailSize[bIdx],
                                        splitInfo.s2TailSize[bIdx]);
}

void CalcS1GCache(uint32_t s1GIdx, const SplitContext &splitContext, const BatchCache &batchCache, S1GCache &s1GCache)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitParam &splitParam = splitContext.splitParam;
    const SplitInfo &splitInfo = splitContext.splitInfo;

    s1GCache.bIdx = batchCache.bIdx;
    s1GCache.s1GIdx = s1GIdx;

    auto s2Range = CalcS2Range(s1GIdx, baseInfo, splitParam, batchCache);
    s1GCache.s2Start = s2Range.first;
    s1GCache.s2End = s2Range.second;

    if (s1GCache.s2Start >= s1GCache.s2End) {
        s1GCache.s1GBlock = 0;
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
        return;
    }

    s1GCache.s1GBlock = s1GCache.s2End - s1GCache.s2Start;
    uint32_t curTailS2Num =
        (splitInfo.s2TailSize[batchCache.bIdx] != 0U && s1GCache.s2End == splitInfo.s2BaseNum[batchCache.bIdx]) ? 1U :
                                                                                                                  0U;
    uint32_t curNormalS2Num = s1GCache.s1GBlock - curTailS2Num;
    if (splitInfo.s1GBaseNum[batchCache.bIdx] == 0) {
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
    } else if (s1GIdx == (splitInfo.s1GBaseNum[batchCache.bIdx] - 1U) && splitInfo.s1GTailSize[batchCache.bIdx] != 0U) {
        s1GCache.s1GCost = batchCache.typeCost[TAIL_BLOCK][NORMAL_BLOCK] * curNormalS2Num +
                           batchCache.typeCost[TAIL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GLastBlockCost = curTailS2Num > 0U ? batchCache.typeCost[TAIL_BLOCK][TAIL_BLOCK] :
                                                        batchCache.typeCost[TAIL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GNormalBlockCost = batchCache.typeCost[TAIL_BLOCK][NORMAL_BLOCK];
    } else {
        s1GCache.s1GCost = batchCache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK] * curNormalS2Num +
                           batchCache.typeCost[NORMAL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GLastBlockCost = curTailS2Num > 0U ? batchCache.typeCost[NORMAL_BLOCK][TAIL_BLOCK] :
                                                        batchCache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GNormalBlockCost = batchCache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
    }
}

void CopyTmpResult(FAMetaData &tmpRes, FAMetaData &splitRes)
{
    uint64_t len = tmpRes.bN2End.size();
    splitRes.usedCoreNum = tmpRes.usedCoreNum;
    splitRes.maxCost = tmpRes.maxCost;
    splitRes.combineRes.combineNum = tmpRes.combineRes.combineNum;

    for (size_t i = 0; i < len; ++i) {
        splitRes.bN2End[i] = tmpRes.bN2End[i];
        splitRes.mEnd[i] = tmpRes.mEnd[i];
        splitRes.s2End[i] = tmpRes.s2End[i];
        splitRes.combineRes.combineBN2Idx[i] = tmpRes.combineRes.combineBN2Idx[i];
        splitRes.combineRes.combineMIdx[i] = tmpRes.combineRes.combineMIdx[i];
        splitRes.combineRes.combineS2SplitNum[i] = tmpRes.combineRes.combineS2SplitNum[i];
        splitRes.combineRes.combineWorkspaceIdx[i] = tmpRes.combineRes.combineWorkspaceIdx[i];
        splitRes.firstCombineDataWorkspaceIdx[i] = tmpRes.firstCombineDataWorkspaceIdx[i];
        splitRes.combineRes.mSize[i] = tmpRes.combineRes.mSize[i];
    }
}

void ClearTmpResult(FAMetaData &tmpResult)
{
    uint64_t len = tmpResult.bN2End.size();
    tmpResult.usedCoreNum = 0U;
    tmpResult.maxCost = 0;
    tmpResult.combineRes.combineNum = 0U;
    tmpResult.combineRes.combineUsedVecNum = 0U;

    for (size_t i = 0; i < len; ++i) {
        tmpResult.bN2End[i] = 0U;
        tmpResult.mEnd[i] = 0U;
        tmpResult.s2End[i] = 0U;
        tmpResult.combineRes.combineBN2Idx[i] = 0U;
        tmpResult.combineRes.combineMIdx[i] = 0U;
        tmpResult.combineRes.combineS2SplitNum[i] = 0U;
        tmpResult.combineRes.combineWorkspaceIdx[i] = 0U;
        tmpResult.firstCombineDataWorkspaceIdx[i] = 0U;
        tmpResult.combineRes.mSize[i] = 0U;
    }
}

void CalcBatchCost(uint32_t bIdx, const SplitContext &splitContext, CostInfo &costInfo)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;

    costInfo.bN2CostOfEachBatch[bIdx] = 0;
    costInfo.bN2BlockOfEachBatch[bIdx] = 0U;
    costInfo.bN2LastBlockCostOfEachBatch[bIdx] = 0U;

    if (GetS1SeqSize(bIdx, baseInfo) == 0U || GetS2SeqSize(bIdx, baseInfo) == 0U) {
        return;
    }

    BatchCache bCache;
    S1GCache s1GCache;
    CalcBatchCache(bIdx, splitContext, bCache);
    for (uint32_t s1GIdx = 0; s1GIdx < splitInfo.s1GBaseNum[bIdx]; s1GIdx++) {
        CalcS1GCache(s1GIdx, splitContext, bCache, s1GCache);
        costInfo.bN2CostOfEachBatch[bIdx] += s1GCache.s1GCost;
        costInfo.bN2BlockOfEachBatch[bIdx] += s1GCache.s1GBlock;
        if (s1GCache.s1GCost > costInfo.maxS1GCost) {
            costInfo.maxS1GCost = s1GCache.s1GCost;
        }
        if (s1GCache.s1GBlock > 0) {
            costInfo.bN2LastBlockCostOfEachBatch[bIdx] = s1GCache.s1GLastBlockCost;
        }
    }
}

void CalcCostInfo(SplitContext &splitContext)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;
    CostInfo &costInfo = splitContext.costInfo;

    if (splitInfo.isKvSeqAllZero) {
        costInfo.totalCost = 0;
        costInfo.totalBlockNum = 0U;
        return;
    }

    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        CalcBatchCost(bIdx, splitContext, costInfo);
        costInfo.totalCost += costInfo.bN2CostOfEachBatch[bIdx] * baseInfo.n2Size;
        costInfo.totalBlockNum += costInfo.bN2BlockOfEachBatch[bIdx] * baseInfo.n2Size;
    }
}

void UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;
    const CostInfo &costInfo = splitContext.costInfo;

    bool UpdateS1G = false;
    bool UpdateBatch = false;

    if (assignContext.curS2Idx >= assignContext.s1GCache.s2End) {
        assignContext.curS2Idx = 0U;
        assignContext.curS1GIdx++;
        UpdateS1G = true;
    }

    if (assignContext.curS1GIdx >= splitInfo.s1GBaseNum[assignContext.curBIdx]) {
        assignContext.curS1GIdx = 0U;
        assignContext.curBN2Idx++;
    }

    if (assignContext.curBN2Idx == baseInfo.bSize * baseInfo.n2Size) {
        assignContext.curS1GIdx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.isFinished = true;
        return;
    }

    if (assignContext.curBN2Idx / baseInfo.n2Size != assignContext.curBIdx) {
        assignContext.curBIdx = assignContext.curBN2Idx / baseInfo.n2Size;
        assignContext.curS1GIdx = 0U;
        UpdateBatch = true;
        UpdateS1G = true;
    }

    if (UpdateBatch) {
        CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    }
    if (UpdateS1G) {
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    const BaseInfo &baseInfo = splitContext.baseInfo;
    const CostInfo &costInfo = splitContext.costInfo;

    while (assignContext.bN2Cost == 0 ||
           IsWithinTolerance(assignContext.coreCache.costLimit,
                             costInfo.bN2LastBlockCostOfEachBatch[assignContext.curBIdx] / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + assignContext.bN2Cost)) {
        assignContext.coreCache.cost += assignContext.bN2Cost;
        assignContext.coreCache.block += assignContext.bN2Block;
        assignContext.curBN2Idx++;

        if (assignContext.curBN2Idx == baseInfo.bSize * baseInfo.n2Size) {
            assignContext.curS1GIdx = 0U;
            assignContext.curS2Idx = 0U;
            assignContext.isFinished = true;
            return;
        }

        if (assignContext.curBN2Idx / baseInfo.n2Size != assignContext.curBIdx) {
            assignContext.curBIdx = assignContext.curBN2Idx / baseInfo.n2Size;
            CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        }

        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
        assignContext.curS1GIdx = 0U;
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void AssignByRow(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit,
                             assignContext.s1GCache.s1GLastBlockCost / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + assignContext.s1GCache.s1GCost)) {
        assignContext.coreCache.cost += assignContext.s1GCache.s1GCost;
        assignContext.coreCache.block += assignContext.s1GCache.s1GBlock;

        assignContext.bN2Cost = assignContext.bN2Cost > assignContext.s1GCache.s1GCost ?
                                    assignContext.bN2Cost - assignContext.s1GCache.s1GCost :
                                    0;
        assignContext.bN2Block = assignContext.bN2Block > assignContext.s1GCache.s1GBlock ?
                                     assignContext.bN2Block - assignContext.s1GCache.s1GBlock :
                                     0U;
        do {
            assignContext.curS1GIdx++;
            CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        } while (assignContext.s1GCache.s1GBlock == 0);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void AssignByBlock(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished || !splitContext.splitParam.streamK) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit, curCost / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + curCost)) {
        assignContext.coreCache.cost += curCost;
        assignContext.coreCache.block++;
        assignContext.curS2Idx++;
        assignContext.bN2Cost = assignContext.bN2Cost - curCost;
        assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
        assignContext.bN2Block--;
        assignContext.s1GCache.s1GBlock--;
    }
}

void ForceAssign(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }

    assignContext.coreCache.cost += curCost;
    assignContext.coreCache.block++;
    assignContext.curS2Idx++;
    assignContext.bN2Cost = assignContext.bN2Cost - curCost;
    assignContext.bN2Block--;
    assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
    assignContext.s1GCache.s1GBlock--;
    UpdateCursor(splitContext, assignContext);
}

bool IsNeedRecordCombineInfo(const AssignContext &assignContext, const FAMetaData &splitRes)
{
    if (assignContext.curCoreIdx == 0U) {
        return false;
    }
    if (assignContext.curKvSplitPart <= 1U) {
        return false;
    }
    if (assignContext.curBN2Idx == splitRes.bN2End[assignContext.curCoreIdx - 1U] &&
        assignContext.curS1GIdx == splitRes.mEnd[assignContext.curCoreIdx - 1U]) {
        return false;
    }
    return true;
}

void RecordCombineInfo(const SplitContext &splitContext, const AssignContext &assignContext, FAMetaData &result)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitParam &splitParam = splitContext.splitParam;
    const SplitInfo &splitInfo = splitContext.splitInfo;
    uint32_t splitBIdx = result.bN2End[assignContext.curCoreIdx - 1U] / baseInfo.n2Size;
    uint32_t splitS1GIdx = result.mEnd[assignContext.curCoreIdx - 1U];
    uint32_t s1Size = GetS1SeqSize(splitBIdx, baseInfo);

    uint32_t curCombineS1gSize = (splitS1GIdx == splitInfo.s1GBaseNum[splitBIdx] - 1U) ?
                                (s1Size * baseInfo.gSize - splitS1GIdx * splitParam.mBaseSize) :
                                splitParam.mBaseSize;
    uint32_t curCombineS1gSplitPart = (curCombineS1gSize + splitParam.gS1BaseSizeOfCombine - 1U) / splitParam.gS1BaseSizeOfCombine;
    uint32_t curCombineS1gLastPartSize = curCombineS1gSize - (splitParam.gS1BaseSizeOfCombine * (curCombineS1gSplitPart - 1U));

    result.combineRes.combineBN2Idx[result.combineRes.combineNum] = result.bN2End[assignContext.curCoreIdx - 1U];
    result.combineRes.combineMIdx[result.combineRes.combineNum] = result.mEnd[assignContext.curCoreIdx - 1U];
    result.combineRes.combineS2SplitNum[result.combineRes.combineNum] = assignContext.curKvSplitPart;
    result.combineRes.combineWorkspaceIdx[result.combineRes.combineNum] = assignContext.preCombineDataNUM;
    result.combineRes.mSize[result.combineRes.combineNum] = curCombineS1gSize;
    result.combineRes.combineNum++;
}

void CalcSplitPlan(uint32_t coreNum, int64_t costLimit, const SplitContext &splitContext, FAMetaData &result)
{
    const CostInfo &costInfo = splitContext.costInfo;

    if (coreNum == 0U) {
        return;
    }
    result.maxCost = 0U;
    result.usedCoreNum = 0U;

    AssignContext assignContext{};
    assignContext.curBIdx = 0U;
    assignContext.curS1GIdx = 0U;
    assignContext.unassignedCost = costInfo.totalCost;
    assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
    assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
    CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
    assignContext.curS2Idx = assignContext.s1GCache.s2Start;

    for (uint32_t i = 0; i < coreNum; ++i) {
        if (result.maxCost > costLimit) {
            return;
        }
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
            break;
        }

        assignContext.curCoreIdx = i;
        result.firstCombineDataWorkspaceIdx[assignContext.curCoreIdx] =
            assignContext.preCombineDataNUM + assignContext.curKvSplitPart - 1U;

        int64_t avgCost = assignContext.unassignedCost / (coreNum - assignContext.curCoreIdx);
        assignContext.coreCache = {};
        if (!splitContext.splitParam.streamK) {
            assignContext.coreCache.costLimit = std::max(avgCost, costInfo.maxS1GCost);
        } else {
            assignContext.coreCache.costLimit = avgCost;
        }

        AssignByBatch(splitContext, assignContext);
        AssignByRow(splitContext, assignContext);
        AssignByBlock(splitContext, assignContext);
        if (assignContext.coreCache.block == 0 && splitContext.splitParam.streamK) {
            ForceAssign(splitContext, assignContext);
        }

        result.bN2End[i] = assignContext.curBN2Idx;
        result.mEnd[i] = assignContext.curS1GIdx;
        result.s2End[i] = assignContext.curS2Idx;
        result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);

        assignContext.unassignedCost -= assignContext.coreCache.cost;

        if (IsNeedRecordCombineInfo(assignContext, result)) {
            RecordCombineInfo(splitContext, assignContext, result);
            assignContext.preCombineDataNUM += assignContext.curKvSplitPart;
            assignContext.curKvSplitPart = 1U;
        }

        if (assignContext.curS2Idx > assignContext.s1GCache.s2Start &&
            assignContext.curS2Idx <= assignContext.s1GCache.s2End) {
            assignContext.curKvSplitPart++;
        }
    }

    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

void SplitCombine(FAMetaData &result)
{
    CombineResult &combineRes = result.combineRes;
    uint64_t totalCombineLoad = 0;
    for (uint32_t i = 0; i < result.combineRes.combineNum; i++) {
        totalCombineLoad += combineRes.combineS2SplitNum[i] * combineRes.mSize[i];
    }
    uint32_t maxVectorNum = result.usedCoreNum * result.vecCubeRatio;
    uint64_t averageLoad = (totalCombineLoad + maxVectorNum - 1U) / maxVectorNum;
    uint32_t curCoreIndex = 0;
    for (uint32_t i = 0; i < result.combineRes.combineNum; i++) {
        uint32_t curCombineVectorNum =
            std::max(combineRes.combineS2SplitNum[i] * combineRes.mSize[i] / averageLoad, 1UL);
        uint32_t curAvgMSize = (combineRes.mSize[i] + curCombineVectorNum - 1U) / curCombineVectorNum;
        curCombineVectorNum = (combineRes.mSize[i] + curAvgMSize - 1U) / curAvgMSize;
        for (uint32_t vid = 0; vid < curCombineVectorNum; vid++) {
            combineRes.taskIdx[curCoreIndex] = i;
            combineRes.mStart[curCoreIndex] = vid * curAvgMSize;
            combineRes.mLen[curCoreIndex] = (vid < curCombineVectorNum - 1) ? curAvgMSize : (combineRes.mSize[i] - vid * curAvgMSize);
            curCoreIndex++;
        }
    }
    combineRes.combineUsedVecNum = curCoreIndex;
}

void SplitCore(uint32_t coreNum, const BaseInfo &baseInfo, const SplitParam &param, FAMetaData &result)
{
    SplitContext splitContext(baseInfo, param);
    CalcSplitInfo(splitContext);
    if (splitContext.splitInfo.isKvSeqAllZero) {
        result.usedCoreNum = 1U;
        result.bN2End[0] = baseInfo.bSize * baseInfo.n2Size;
        result.mEnd[0] = 0U;
        result.s2End[0] = 0U;
        return;
    }
    CalcCostInfo(splitContext);

    uint32_t maxCore = std::min(coreNum, splitContext.costInfo.totalBlockNum);
    uint32_t minCore =
        static_cast<uint32_t>(std::sqrt(static_cast<float>(splitContext.costInfo.totalBlockNum) + 0.25f) + 0.5f);
    minCore = std::min(minCore, maxCore);

    result.maxCost = INT64_MAX;
    result.usedCoreNum = 1U;

    FAMetaData tmpResult{coreNum, result.vecCubeRatio};
    for (uint32_t i = minCore; i <= maxCore; ++i) {
        CalcSplitPlan(i, result.maxCost, splitContext, tmpResult);
        if (tmpResult.maxCost < result.maxCost) {
            CopyTmpResult(tmpResult, result);
        }
        ClearTmpResult(tmpResult);
    }

    if (result.combineRes.combineNum > 0U) {
        SplitCombine(result);
    }

    result.usedCoreNum = std::max(result.usedCoreNum, 1U);
}

} // namespace split_core