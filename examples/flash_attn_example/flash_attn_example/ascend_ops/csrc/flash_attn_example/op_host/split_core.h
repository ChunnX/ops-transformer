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
 * \file split_core.h
 * \brief BN1->S1 two-level core splitting (no GS1 fusion, no combine).
 */

#ifndef SPLIT_CORE_H
#define SPLIT_CORE_H

#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>

namespace split_core {
constexpr int64_t FA_TOLERANCE_RATIO = 2;

enum BlockType : uint32_t {
    NORMAL_BLOCK = 0,
    TAIL_BLOCK,
    BLOCK_MAX_TYPE
};

template <class T>
using Range = std::pair<T, T>;

template <class T>
using BlockCost = std::array<std::array<T, static_cast<size_t>(BLOCK_MAX_TYPE)>, static_cast<size_t>(BLOCK_MAX_TYPE)>;

template <typename T>
T Clip(T value, T minValue, T maxValue)
{
    if (value < minValue) {
        return minValue;
    }
    if (value > maxValue) {
        return maxValue;
    }
    return value;
}

template <typename T>
inline bool IsWithinTolerance(T limit, T tolerance, T value)
{
    return limit + tolerance >= value;
}

struct BaseInfo {
    uint32_t bSize{0U};
    uint32_t n1Size{0U};
    uint32_t n2Size{0U};
    uint32_t gSize{0U};
    uint32_t s1Size{0U};
    uint32_t s2Size{0U};
    bool isAccumSeqS1{false};
    bool isAccumSeqS2{false};
    bool attnMaskFlag{false};
    int32_t sparseMode{0U};
    int64_t preToken{0};
    int64_t nextToken{0};
};

struct SplitParam {
    uint32_t mBaseSize{1U};
    uint32_t s2BaseSize{1U};
};

struct SplitInfo {
    std::vector<uint32_t> s1BlockCount{};
    std::vector<uint32_t> s1TailCount{};
    std::vector<uint32_t> s2BlockCount{};
    std::vector<uint32_t> s2TailCount{};
    bool isKvSeqAllZero{true};

    explicit SplitInfo(uint32_t totalBN1)
        : s1BlockCount(totalBN1), s1TailCount(totalBN1),
          s2BlockCount(totalBN1), s2TailCount(totalBN1)
    {
    }
};

struct CostInfo {
    std::vector<int64_t> bn1Cost{};
    std::vector<uint32_t> bn1S1BlockCount{};
    std::vector<int64_t> bn1LastS1BlockCost{};
    uint32_t totalBlockNum{0U};
    int64_t totalCost{0};
    int64_t maxS1BlockCost{0};

    explicit CostInfo(uint32_t totalBN1)
        : bn1Cost(totalBN1), bn1S1BlockCount(totalBN1), bn1LastS1BlockCost(totalBN1)
    {
    }
};

struct SplitContext {
    BaseInfo baseInfo{};
    SplitParam splitParam{};
    SplitInfo splitInfo{0U};
    CostInfo costInfo{0U};

    explicit SplitContext(const BaseInfo &info, SplitParam param)
        : baseInfo(info), splitParam(param), splitInfo(info.bSize * info.n1Size), costInfo(info.bSize * info.n1Size)
    {
    }
};

struct BN1Cache {
    uint32_t bn1Idx{0U};
    uint32_t bIdx{0U};
    uint32_t n1Head{0U};
    uint32_t s1Size{0U};
    uint32_t s2Size{0U};
    int64_t preTokenLeftUp{0};
    int64_t nextTokenLeftUp{0};
    uint32_t s1BlockCount{0U};
    uint32_t s2BlockCount{0U};
    BlockCost<int64_t> typeCost{};
};

struct S1BlockCache {
    uint32_t bn1Idx{0U};
    uint32_t s1BlockIdx{0U};
    uint32_t s2Start{0U};
    uint32_t s2End{0U};
    int64_t s1BlockCost{0};
};

struct FAMetaData {
    uint32_t usedCoreNum{0U};
    std::vector<uint32_t> bn1End{};
    std::vector<uint32_t> s1End{};
    int64_t maxCost{0};

    explicit FAMetaData(uint32_t coreNum)
        : bn1End(coreNum), s1End(coreNum) {};
};

struct AssignContext {
    uint32_t curBN1Idx{0U};
    uint32_t curS1BlockIdx{0U};
    uint32_t curCoreIdx{0U};
    int64_t unassignedCost{0};
    uint32_t usedCoreNum{0U};

    int64_t bn1Cost{0};
    uint32_t bn1S1BlockCount{0U};
    bool isFinished{false};
    BN1Cache bn1Cache{};
    S1BlockCache s1BlockCache{};
    struct { int64_t costLimit{0}; int64_t cost{0}; uint32_t block{0U}; } coreCache;
};

uint32_t GetS1SeqSize(uint32_t bIdx, const BaseInfo &baseInfo);
uint32_t GetS2SeqSize(uint32_t bIdx, const BaseInfo &baseInfo);
int64_t CalcPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const BaseInfo &baseInfo);
int64_t CalcNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const BaseInfo &baseInfo);
Range<uint32_t> CalcS2Range(uint32_t s1Start, const BaseInfo &baseInfo, const SplitParam &splitParam,
                            const BN1Cache &bn1Cache);
int64_t CalcCost(uint32_t basicM, uint32_t basicS2);
BlockCost<int64_t> CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize, uint32_t s1TailSize,
                                 uint32_t s2TailSize);

void CalcBN1Cache(uint32_t bn1Idx, const SplitContext &splitContext, BN1Cache &bn1Cache);
void CalcS1BlockCache(uint32_t s1BlockIdx, const SplitContext &splitContext, const BN1Cache &bn1Cache,
                      S1BlockCache &s1BlockCache);
void CopyTmpResult(FAMetaData &tmpRes, FAMetaData &splitRes);
void ClearTmpResult(FAMetaData &tmpResult);

void CalcSplitInfo(SplitContext &splitContext);
void CalcBN1Cost(uint32_t bn1Idx, SplitContext &splitContext);
void CalcCostInfo(SplitContext &splitContext);

void UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext);
void AssignByBN1(const SplitContext &splitContext, AssignContext &assignContext);
void AssignByS1Block(const SplitContext &splitContext, AssignContext &assignContext);
void ForceAssign(const SplitContext &splitContext, AssignContext &assignContext);

void CalcSplitPlan(uint32_t coreNum, int64_t costLimit, const SplitContext &splitContext, FAMetaData &result);
void SplitCore(uint32_t coreNum, const BaseInfo &baseInfo, const SplitParam &splitParam, FAMetaData &result);
} // namespace split_core
#endif
