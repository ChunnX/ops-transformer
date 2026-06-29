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
 * \brief Split core v2 module for GQA non-quant tiling.
 *        Ported from ops-transformer split_core.h.
 */

#ifndef SPLIT_CORE_H
#define SPLIT_CORE_H

#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>

namespace split_core {
constexpr int64_t FA_TOLERANCE_RATIO = 2;
constexpr uint32_t FD_TOLERANCE_RATIO = 2U;

enum BlockType : uint32_t {
    NORMAL_BLOCK = 0,
    TAIL_BLOCK,
    BLOCK_MAX_TYPE
};

enum class SparseMode : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    SPARSE_BUTT,
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
    uint32_t n2Size{0U};
    uint32_t gSize{0U};
    uint32_t s1Size{0U};
    uint32_t s2Size{0U};
    bool isS1G{true};
    bool isAccumSeqS1{false};
    bool isAccumSeqS2{false};
    std::vector<int64_t> actualSeqS1Size{};
    std::vector<int64_t> actualSeqS2Size{};
    uint32_t actualLenQDims{0U};
    uint32_t actualLenKvDims{0U};
    bool attnMaskFlag{false};
    int32_t sparseMode{0U};
    int64_t preToken{0};
    int64_t nextToken{0};
    int64_t actualSeqPrefixSize{0};
};

struct SplitParam {
    uint32_t mBaseSize{1U};
    uint32_t s2BaseSize{1U};
    uint32_t gS1BaseSizeOfCombine{8U};
    bool streamK = true;
};

struct CombineResult {
    uint32_t combineNum{0U};
    std::vector<uint32_t> combineBN2Idx{};
    std::vector<uint32_t> combineMIdx{};
    std::vector<uint32_t> combineS2SplitNum{};
    std::vector<uint32_t> combineWorkspaceIdx{};
    uint32_t combineUsedVecNum{0U};
    std::vector<uint32_t> mSize{};
    std::vector<uint32_t> taskIdx{};
    std::vector<uint32_t> mStart{};
    std::vector<uint32_t> mLen{};
    CombineResult(uint32_t coreNum, uint32_t vecCubeRatio)
        : combineBN2Idx(coreNum), combineMIdx(coreNum), combineS2SplitNum(coreNum), combineWorkspaceIdx(coreNum), mSize(coreNum),
          taskIdx(coreNum * vecCubeRatio), mStart(coreNum * vecCubeRatio), mLen(coreNum * vecCubeRatio)
    {
    }
};

struct FAMetaData {
    uint32_t usedCoreNum{0U};
    uint32_t vecCubeRatio{0U};
    std::vector<uint32_t> bN2End{};
    std::vector<uint32_t> mEnd{};
    std::vector<uint32_t> s2End{};
    int64_t maxCost{0};
    std::vector<uint32_t> firstCombineDataWorkspaceIdx{};
    CombineResult combineRes{0U, 0U};
    FAMetaData(uint32_t coreNum, uint32_t ratio)
        : vecCubeRatio(ratio), bN2End(coreNum), mEnd(coreNum), s2End(coreNum), firstCombineDataWorkspaceIdx(coreNum),
          combineRes(coreNum, ratio){};
};

struct SplitInfo {
    std::vector<uint32_t> s1GBaseNum{};
    std::vector<uint32_t> s2BaseNum{};
    std::vector<uint32_t> s1GTailSize{};
    std::vector<uint32_t> s2TailSize{};
    bool isKvSeqAllZero{true};

    explicit SplitInfo(uint32_t batchSize)
        : s1GBaseNum(batchSize), s2BaseNum(batchSize), s1GTailSize(batchSize), s2TailSize(batchSize)
    {
    }
};

struct CostInfo {
    std::vector<int64_t> bN2CostOfEachBatch{};
    std::vector<uint32_t> bN2BlockOfEachBatch{};
    std::vector<int64_t> bN2LastBlockCostOfEachBatch{};
    uint32_t totalBlockNum{0U};
    int64_t totalCost{0};
    int64_t maxS1GCost{0};

    explicit CostInfo(uint32_t batchSize)
        : bN2CostOfEachBatch(batchSize), bN2BlockOfEachBatch(batchSize), bN2LastBlockCostOfEachBatch(batchSize)
    {
    }
};

struct SplitContext {
    const BaseInfo &baseInfo{0U};
    SplitParam splitParam{};
    SplitInfo splitInfo{0U};
    CostInfo costInfo{0U};

    explicit SplitContext(const BaseInfo &info, SplitParam param)
        : baseInfo(info), splitParam(param), splitInfo(info.bSize), costInfo(info.bSize)
    {
    }
};

struct BatchCache {
    uint32_t bIdx{0U};
    uint32_t s1Size{0U};
    uint32_t s2Size{0U};
    int64_t preTokenLeftUp{0};
    int64_t nextTokenLeftUp{0};
    BlockCost<int64_t> typeCost{};
};

struct S1GCache {
    uint32_t bIdx{0U};
    uint32_t s1GIdx{0U};
    uint32_t s2Start{0U};
    uint32_t s2End{0U};
    int64_t s1GCost{0};
    int64_t s1GLastBlockCost{0};
    uint32_t s1GBlock{0U};
    int64_t s1GNormalBlockCost{0};
};

struct CoreCache {
    int64_t costLimit{0};
    int64_t cost{0};
    uint32_t block{0U};
};

struct AssignContext {
    uint32_t curBIdx{0U};
    uint32_t curBN2Idx{0U};
    uint32_t curS1GIdx{0U};
    uint32_t curS2Idx{0U};
    uint32_t curCoreIdx{0U};
    int64_t unassignedCost{0};
    uint32_t usedCoreNum{0U};
    uint32_t curKvSplitPart{1U};
    uint32_t preCombineDataNUM{0U};

    int64_t bN2Cost{0};
    uint32_t bN2Block{0U};
    bool isFinished{false};
    BatchCache batchCache{};
    S1GCache s1GCache{};
    CoreCache coreCache{};
};

uint32_t GetS1SeqSize(uint32_t bIdx, const BaseInfo &baseInfo);
uint32_t GetS2SeqSize(uint32_t bIdx, const BaseInfo &baseInfo);
int64_t CalcPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const BaseInfo &baseInfo);
int64_t CalcNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const BaseInfo &baseInfo);
Range<uint32_t> CalcS2Range(uint32_t s1GIdx, const BaseInfo &baseInfo, const SplitParam &splitParam,
                            const BatchCache &batchCache);
int64_t CalcCost(uint32_t basicM, uint32_t basicS2);
BlockCost<int64_t> CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize, uint32_t s1GTailSize,
                                 uint32_t s2TailSize);

void CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache);
void CalcS1GCache(uint32_t s1GIdx, const SplitContext &splitContext, const BatchCache &batchCache, S1GCache &s1GCache);
void CopyTmpResult(FAMetaData &tmpRes, FAMetaData &splitRes);
void ClearTmpResult(FAMetaData &tmpResult);

void CalcSplitInfo(SplitContext &splitContext);
void CalcBatchCost(uint32_t bIdx, const SplitContext &splitContext, CostInfo &costInfo);
void CalcCostInfo(SplitContext &splitContext);

void UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext);
void AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext);
void AssignByRow(const SplitContext &splitContext, AssignContext &assignContext);
void AssignByBlock(const SplitContext &splitContext, AssignContext &assignContext);
void ForceAssign(const SplitContext &splitContext, AssignContext &assignContext);

bool IsNeedRecordCombineInfo(const AssignContext &assignContext, const FAMetaData &splitRes);
void RecordCombineInfo(const SplitContext &splitContext, const AssignContext &assignContext, FAMetaData &result);

void SplitCombine(FAMetaData &result);
void CalcSplitPlan(uint32_t coreNum, int64_t costLimit, const SplitContext &splitContext, FAMetaData &result);
void SplitCore(uint32_t coreNum, const BaseInfo &baseInfo, const SplitParam &splitParam, FAMetaData &result);
} // namespace split_core
#endif