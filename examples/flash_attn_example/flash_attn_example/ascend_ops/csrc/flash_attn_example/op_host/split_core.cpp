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
 * \brief BN1->S1 two-level core splitting implementation.
 */

#include "split_core.h"
#include <cmath>
#include <algorithm>

namespace split_core {

uint32_t GetS1SeqSize(uint32_t bIdx, const BaseInfo &baseInfo)
{
    return baseInfo.s1Size;
}

uint32_t GetS2SeqSize(uint32_t bIdx, const BaseInfo &baseInfo)
{
    return baseInfo.s2Size;
}

int64_t CalcPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const BaseInfo &baseInfo)
{
    return baseInfo.preToken;
}

int64_t CalcNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const BaseInfo &baseInfo)
{
    return baseInfo.nextToken;
}

int64_t CalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t alignCoefM = 16U;
    uint32_t alignCoefS2 = 64U;
    uint32_t alignBasicM = (basicM + alignCoefM - 1U) >> 4U;
    uint32_t alignBasicS2 = (basicS2 + alignCoefS2 - 1U) >> 6U;
    return static_cast<int64_t>(6U * alignBasicM + 10U * alignBasicS2);
}

BlockCost<int64_t> CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize, uint32_t s1TailSize,
                                 uint32_t s2TailSize)
{
    BlockCost<int64_t> typeCost{};
    typeCost[NORMAL_BLOCK][NORMAL_BLOCK] = CalcCost(s1NormalSize, s2NormalSize);
    typeCost[TAIL_BLOCK][NORMAL_BLOCK] = (s1TailSize == 0U) ? 0 : CalcCost(s1TailSize, s2NormalSize);
    typeCost[NORMAL_BLOCK][TAIL_BLOCK] = (s2TailSize == 0U) ? 0 : CalcCost(s1NormalSize, s2TailSize);
    typeCost[TAIL_BLOCK][TAIL_BLOCK] = (s1TailSize == 0U || s2TailSize == 0U) ? 0 : CalcCost(s1TailSize, s2TailSize);
    return typeCost;
}

Range<uint32_t> CalcS2Range(uint32_t s1Start, const BaseInfo &baseInfo, const SplitParam &splitParam,
                            const BN1Cache &bn1Cache)
{
    uint32_t s2Start = 0U;
    uint32_t s2End = 0U;

    if (bn1Cache.s1Size == 0U || bn1Cache.s2Size == 0U) {
        return std::make_pair(s2Start, s2End);
    }

    if (!baseInfo.attnMaskFlag) {
        s2Start = 0U;
        s2End = (bn1Cache.s2Size + splitParam.s2BaseSize - 1U) / splitParam.s2BaseSize;
        return std::make_pair(s2Start, s2End);
    }

    int64_t s1FirstToken = s1Start;
    int64_t s1LastToken = std::min(s1FirstToken + static_cast<int64_t>(splitParam.mBaseSize),
                                   static_cast<int64_t>(bn1Cache.s1Size)) - 1;

    int64_t s2FirstToken = s1FirstToken - bn1Cache.preTokenLeftUp;
    int64_t s2LastToken = s1LastToken + bn1Cache.nextTokenLeftUp;

    if (s2FirstToken >= static_cast<int64_t>(bn1Cache.s2Size) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
        s2Start = 0U;
        s2End = 0U;
        return std::make_pair(s2Start, s2End);
    }

    s2FirstToken = Clip(s2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(bn1Cache.s2Size - 1U));
    s2LastToken = Clip(s2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(bn1Cache.s2Size - 1U));
    s2Start = static_cast<uint32_t>(s2FirstToken) / splitParam.s2BaseSize;
    s2End = static_cast<uint32_t>(s2LastToken) / splitParam.s2BaseSize + 1U;

    return std::make_pair(s2Start, s2End);
}

void CalcSplitInfo(SplitContext &splitContext)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitParam &splitParam = splitContext.splitParam;
    SplitInfo &splitInfo = splitContext.splitInfo;

    uint32_t totalBN1 = baseInfo.bSize * baseInfo.n1Size;

    for (uint32_t bn1Idx = 0; bn1Idx < totalBN1; bn1Idx++) {
        uint32_t bIdx = bn1Idx / baseInfo.n1Size;
        uint32_t s1Size = GetS1SeqSize(bIdx, baseInfo);
        uint32_t s2Size = GetS2SeqSize(bIdx, baseInfo);

        splitInfo.s1BlockCount[bn1Idx] = (s1Size + splitParam.mBaseSize - 1U) / splitParam.mBaseSize;
        splitInfo.s1TailCount[bn1Idx] = s1Size % splitParam.mBaseSize;
        splitInfo.s2BlockCount[bn1Idx] = (s2Size + splitParam.s2BaseSize - 1U) / splitParam.s2BaseSize;
        splitInfo.s2TailCount[bn1Idx] = s2Size % splitParam.s2BaseSize;
        if (splitInfo.s1BlockCount[bn1Idx] != 0U && splitInfo.s2BlockCount[bn1Idx] != 0U) {
            splitInfo.isKvSeqAllZero = false;
        }
    }
}

void CalcBN1Cache(uint32_t bn1Idx, const SplitContext &splitContext, BN1Cache &bn1Cache)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitParam &splitParam = splitContext.splitParam;
    const SplitInfo &splitInfo = splitContext.splitInfo;

    bn1Cache.bn1Idx = bn1Idx;
    bn1Cache.bIdx = bn1Idx / baseInfo.n1Size;
    bn1Cache.n1Head = bn1Idx % baseInfo.n1Size;
    bn1Cache.s1Size = GetS1SeqSize(bn1Cache.bIdx, baseInfo);
    bn1Cache.s2Size = GetS2SeqSize(bn1Cache.bIdx, baseInfo);
    bn1Cache.s1BlockCount = splitInfo.s1BlockCount[bn1Idx];
    bn1Cache.s2BlockCount = splitInfo.s2BlockCount[bn1Idx];
    bn1Cache.preTokenLeftUp = CalcPreTokenLeftUp(bn1Cache.s1Size, bn1Cache.s2Size, baseInfo);
    bn1Cache.nextTokenLeftUp = CalcNextTokenLeftUp(bn1Cache.s1Size, bn1Cache.s2Size, baseInfo);
    bn1Cache.typeCost = CalcCostTable(splitParam.mBaseSize, splitParam.s2BaseSize,
                                       splitInfo.s1TailCount[bn1Idx],
                                       splitInfo.s2TailCount[bn1Idx]);
}

void CalcS1BlockCache(uint32_t s1BlockIdx, const SplitContext &splitContext, const BN1Cache &bn1Cache,
                      S1BlockCache &s1BlockCache)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitParam &splitParam = splitContext.splitParam;
    const SplitInfo &splitInfo = splitContext.splitInfo;

    s1BlockCache.bn1Idx = bn1Cache.bn1Idx;
    s1BlockCache.s1BlockIdx = s1BlockIdx;

    if (s1BlockIdx >= bn1Cache.s1BlockCount) {
        s1BlockCache.s2Start = 0U;
        s1BlockCache.s2End = 0U;
        s1BlockCache.s1BlockCost = 0;
        return;
    }

    uint32_t s1BlockStart = s1BlockIdx * splitParam.mBaseSize;
    auto s2Range = CalcS2Range(s1BlockStart, baseInfo, splitParam, bn1Cache);
    s1BlockCache.s2Start = s2Range.first;
    s1BlockCache.s2End = s2Range.second;

    if (s1BlockCache.s2Start >= s1BlockCache.s2End) {
        s1BlockCache.s1BlockCost = 0;
        return;
    }

    uint32_t s2BlockCount = s1BlockCache.s2End - s1BlockCache.s2Start;
    uint32_t curTailS2Num =
        (splitInfo.s2TailCount[bn1Cache.bn1Idx] != 0U &&
         s1BlockCache.s2End == splitInfo.s2BlockCount[bn1Cache.bn1Idx]) ? 1U : 0U;
    uint32_t curNormalS2Num = s2BlockCount - curTailS2Num;

    if (s1BlockIdx == (bn1Cache.s1BlockCount - 1U) && splitInfo.s1TailCount[bn1Cache.bn1Idx] != 0U) {
        s1BlockCache.s1BlockCost = bn1Cache.typeCost[TAIL_BLOCK][NORMAL_BLOCK] * curNormalS2Num +
                                   bn1Cache.typeCost[TAIL_BLOCK][TAIL_BLOCK] * curTailS2Num;
    } else {
        s1BlockCache.s1BlockCost = bn1Cache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK] * curNormalS2Num +
                                   bn1Cache.typeCost[NORMAL_BLOCK][TAIL_BLOCK] * curTailS2Num;
    }
}

void CopyTmpResult(FAMetaData &tmpRes, FAMetaData &splitRes)
{
    uint64_t len = tmpRes.bn1End.size();
    splitRes.usedCoreNum = tmpRes.usedCoreNum;
    splitRes.maxCost = tmpRes.maxCost;

    for (size_t i = 0; i < len; ++i) {
        splitRes.bn1End[i] = tmpRes.bn1End[i];
        splitRes.s1End[i] = tmpRes.s1End[i];
    }
}

void ClearTmpResult(FAMetaData &tmpResult)
{
    uint64_t len = tmpResult.bn1End.size();
    tmpResult.usedCoreNum = 0U;
    tmpResult.maxCost = 0;

    for (size_t i = 0; i < len; ++i) {
        tmpResult.bn1End[i] = 0U;
        tmpResult.s1End[i] = 0U;
    }
}

void CalcBN1Cost(uint32_t bn1Idx, SplitContext &splitContext)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    CostInfo &costInfo = splitContext.costInfo;

    costInfo.bn1Cost[bn1Idx] = 0;
    costInfo.bn1S1BlockCount[bn1Idx] = 0U;
    costInfo.bn1LastS1BlockCost[bn1Idx] = 0;

    uint32_t bIdx = bn1Idx / baseInfo.n1Size;
    if (GetS1SeqSize(bIdx, baseInfo) == 0U || GetS2SeqSize(bIdx, baseInfo) == 0U) {
        return;
    }

    BN1Cache bn1Cache;
    S1BlockCache s1BlockCache;
    CalcBN1Cache(bn1Idx, splitContext, bn1Cache);

    for (uint32_t s1BlockIdx = 0; s1BlockIdx < bn1Cache.s1BlockCount; s1BlockIdx++) {
        CalcS1BlockCache(s1BlockIdx, splitContext, bn1Cache, s1BlockCache);
        costInfo.bn1Cost[bn1Idx] += s1BlockCache.s1BlockCost;
        if (s1BlockCache.s2End > s1BlockCache.s2Start) {
            costInfo.bn1S1BlockCount[bn1Idx] += 1;
            costInfo.bn1LastS1BlockCost[bn1Idx] = s1BlockCache.s1BlockCost;
        }
        if (s1BlockCache.s1BlockCost > costInfo.maxS1BlockCost) {
            costInfo.maxS1BlockCost = s1BlockCache.s1BlockCost;
        }
    }
}

void CalcCostInfo(SplitContext &splitContext)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    CostInfo &costInfo = splitContext.costInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;

    if (splitInfo.isKvSeqAllZero) {
        costInfo.totalCost = 0;
        costInfo.totalBlockNum = 0U;
        return;
    }

    uint32_t totalBN1 = baseInfo.bSize * baseInfo.n1Size;
    for (uint32_t bn1Idx = 0; bn1Idx < totalBN1; bn1Idx++) {
        CalcBN1Cost(bn1Idx, splitContext);
        costInfo.totalCost += costInfo.bn1Cost[bn1Idx];
        costInfo.totalBlockNum += costInfo.bn1S1BlockCount[bn1Idx];
    }
}

void UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;
    const CostInfo &costInfo = splitContext.costInfo;

    uint32_t totalBN1 = baseInfo.bSize * baseInfo.n1Size;

    assignContext.curS1BlockIdx++;

    if (assignContext.curBN1Idx < totalBN1 &&
        assignContext.curS1BlockIdx >= splitInfo.s1BlockCount[assignContext.curBN1Idx]) {
        assignContext.curBN1Idx++;
        assignContext.curS1BlockIdx = 0;
    }

    if (assignContext.curBN1Idx >= totalBN1) {
        assignContext.isFinished = true;
        return;
    }

    if (assignContext.curS1BlockIdx == 0) {
        CalcBN1Cache(assignContext.curBN1Idx, splitContext, assignContext.bn1Cache);
        assignContext.bn1Cost = costInfo.bn1Cost[assignContext.curBN1Idx];
        assignContext.bn1S1BlockCount = costInfo.bn1S1BlockCount[assignContext.curBN1Idx];
    }

    CalcS1BlockCache(assignContext.curS1BlockIdx, splitContext, assignContext.bn1Cache,
                     assignContext.s1BlockCache);
}

void AssignByBN1(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    const BaseInfo &baseInfo = splitContext.baseInfo;
    const CostInfo &costInfo = splitContext.costInfo;

    while (assignContext.bn1Cost == 0 ||
           IsWithinTolerance(assignContext.coreCache.costLimit,
                             costInfo.bn1LastS1BlockCost[assignContext.curBN1Idx] / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + assignContext.bn1Cost)) {
        assignContext.coreCache.cost += assignContext.bn1Cost;
        assignContext.coreCache.block += assignContext.bn1S1BlockCount;
        assignContext.curBN1Idx++;

        uint32_t totalBN1 = baseInfo.bSize * baseInfo.n1Size;
        if (assignContext.curBN1Idx >= totalBN1) {
            assignContext.curS1BlockIdx = 0;
            assignContext.isFinished = true;
            return;
        }

        CalcBN1Cache(assignContext.curBN1Idx, splitContext, assignContext.bn1Cache);
        assignContext.bn1Cost = costInfo.bn1Cost[assignContext.curBN1Idx];
        assignContext.bn1S1BlockCount = costInfo.bn1S1BlockCount[assignContext.curBN1Idx];
        assignContext.curS1BlockIdx = 0;
        CalcS1BlockCache(assignContext.curS1BlockIdx, splitContext, assignContext.bn1Cache,
                         assignContext.s1BlockCache);
    }
}

void AssignByS1Block(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    while (assignContext.s1BlockCache.s1BlockCost > 0 &&
           IsWithinTolerance(assignContext.coreCache.costLimit,
                             assignContext.s1BlockCache.s1BlockCost / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + assignContext.s1BlockCache.s1BlockCost)) {
        assignContext.coreCache.cost += assignContext.s1BlockCache.s1BlockCost;
        assignContext.coreCache.block++;
        assignContext.curS1BlockIdx++;
        assignContext.bn1Cost = std::max(static_cast<int64_t>(0),
                                         assignContext.bn1Cost - assignContext.s1BlockCache.s1BlockCost);
        assignContext.bn1S1BlockCount--;

        CalcS1BlockCache(assignContext.curS1BlockIdx, splitContext, assignContext.bn1Cache,
                         assignContext.s1BlockCache);
    }
}

void ForceAssign(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    assignContext.coreCache.cost += assignContext.s1BlockCache.s1BlockCost;
    assignContext.coreCache.block++;
    assignContext.curS1BlockIdx++;
    assignContext.bn1Cost = std::max(static_cast<int64_t>(0),
                                     assignContext.bn1Cost - assignContext.s1BlockCache.s1BlockCost);
    assignContext.bn1S1BlockCount--;
    UpdateCursor(splitContext, assignContext);
}

void CalcSplitPlan(uint32_t coreNum, int64_t costLimit, const SplitContext &splitContext, FAMetaData &result)
{
    const CostInfo &costInfo = splitContext.costInfo;

    if (coreNum == 0U) {
        return;
    }
    result.maxCost = 0U;
    result.usedCoreNum = 0U;

    uint32_t totalBN1 = splitContext.baseInfo.bSize * splitContext.baseInfo.n1Size;

    AssignContext assignContext{};
    assignContext.curBN1Idx = 0U;
    assignContext.curS1BlockIdx = 0U;
    assignContext.unassignedCost = costInfo.totalCost;
    assignContext.bn1Cost = costInfo.bn1Cost[assignContext.curBN1Idx];
    assignContext.bn1S1BlockCount = costInfo.bn1S1BlockCount[assignContext.curBN1Idx];
    CalcBN1Cache(assignContext.curBN1Idx, splitContext, assignContext.bn1Cache);
    CalcS1BlockCache(assignContext.curS1BlockIdx, splitContext, assignContext.bn1Cache,
                     assignContext.s1BlockCache);

    for (uint32_t i = 0; i < coreNum; ++i) {
        if (result.maxCost > costLimit) {
            return;
        }
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
            break;
        }

        assignContext.curCoreIdx = i;

        int64_t avgCost = assignContext.unassignedCost / static_cast<int64_t>(coreNum - assignContext.curCoreIdx);
        assignContext.coreCache = {};
        assignContext.coreCache.costLimit = avgCost;

        AssignByBN1(splitContext, assignContext);
        AssignByS1Block(splitContext, assignContext);
        if (assignContext.coreCache.block == 0) {
            ForceAssign(splitContext, assignContext);
        }

        result.bn1End[i] = assignContext.curBN1Idx;
        result.s1End[i] = assignContext.curS1BlockIdx;
        result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);

        assignContext.unassignedCost -= assignContext.coreCache.cost;
    }

    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

void SplitCore(uint32_t coreNum, const BaseInfo &baseInfo, const SplitParam &param, FAMetaData &result)
{
    SplitContext splitContext(baseInfo, param);
    CalcSplitInfo(splitContext);
    if (splitContext.splitInfo.isKvSeqAllZero) {
        result.usedCoreNum = 1U;
        uint32_t totalBN1 = baseInfo.bSize * baseInfo.n1Size;
        result.bn1End[0] = totalBN1;
        result.s1End[0] = 0U;
        return;
    }
    CalcCostInfo(splitContext);

    uint32_t maxCore = std::min(coreNum, splitContext.costInfo.totalBlockNum);
    uint32_t minCore =
        static_cast<uint32_t>(std::sqrt(static_cast<float>(splitContext.costInfo.totalBlockNum) + 0.25f) + 0.5f);
    minCore = std::min(minCore, maxCore);

    result.maxCost = INT64_MAX;
    result.usedCoreNum = 1U;

    FAMetaData tmpResult{coreNum};
    for (uint32_t i = minCore; i <= maxCore; ++i) {
        CalcSplitPlan(i, result.maxCost, splitContext, tmpResult);
        if (tmpResult.maxCost < result.maxCost) {
            CopyTmpResult(tmpResult, result);
        }
        ClearTmpResult(tmpResult);
    }

    result.usedCoreNum = std::max(result.usedCoreNum, 1U);
}

} // namespace split_core
