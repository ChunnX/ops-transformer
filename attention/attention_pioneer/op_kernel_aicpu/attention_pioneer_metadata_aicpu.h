/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATTENTION_PIONEER_METADATA_AICPU_H
#define ATTENTION_PIONEER_METADATA_AICPU_H

#include <array>
#include <string>
#include <vector>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"

namespace aicpu {

constexpr int64_t AP_FA_TOLERANCE_RATIO = 2;
constexpr uint32_t AP_FD_TOLERANCE_RATIO = 2U;

constexpr uint32_t AP_BYTE_BLOCK = 32;
constexpr uint32_t AP_S_INNER_SIZE_512 = 512;
constexpr uint32_t AP_S_INNER_SIZE_1024 = 1024;

constexpr uint32_t AP_S1_SIZE_16 = 16;
constexpr uint32_t AP_MAX_SPLIT_SIZE = 8192;

constexpr uint32_t AP_M_BASE_SIZE_32 = 32;
constexpr uint32_t AP_M_BASE_SIZE_64 = 64;
constexpr uint32_t AP_M_BASE_SIZE_128 = 128;
constexpr uint32_t AP_M_BASE_SIZE_256 = 256;
constexpr uint32_t AP_M_BASE_SIZE_512 = 512;

constexpr uint32_t AP_S_INNER_SIZE_ALIGN_512 = 512;
constexpr uint32_t AP_S_INNER_SIZE_ALIGN_1024 = 1024;
constexpr uint32_t AP_S_INNER_SIZE_ALIGN_2048 = 2048;
constexpr uint32_t AP_S_INNER_SIZE_ALIGN_4096 = 4096;

enum APBlockType : uint32_t {
    AP_NORMAL_BLOCK = 0,
    AP_TAIL_BLOCK,
    AP_BLOCK_MAX_TYPE
};

enum class APSparseMode : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
};

template<class T>
using APRange = std::pair<T, T>;

template<class T>
using APBlockCost = std::array<std::array<T, static_cast<size_t>(AP_BLOCK_MAX_TYPE)>, static_cast<size_t>(AP_BLOCK_MAX_TYPE)>;

template<typename T>
T APClip(T value, T minValue, T maxValue)
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
inline auto APAlign(T num, T rnd) -> T
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

template<typename T>
inline bool APIsWithinTolerance(T limit, T tolerance, T value)
{
    return limit + tolerance >= value;
}

// Split core output: FD info
struct APFlashDecodeResult {
    std::vector<uint32_t> bN2IdxOfFdHead {};
    std::vector<uint32_t> gS1IdxOfFdHead {};
    std::vector<uint32_t> s2SplitNumOfFdHead {};
    std::vector<uint32_t> gS1SplitNumOfFdHead {};
    std::vector<uint32_t> gS1LastPartSizeOfFdHead {};
    std::vector<uint32_t> gS1IdxEndOfFdHead {};
    std::vector<uint32_t> gS1IdxEndOfFdHeadSplit {};
    std::vector<uint32_t> s2SplitStartIdxOfCore {};

    APFlashDecodeResult(uint32_t coreNum, uint32_t vecCubeRatio) :
        bN2IdxOfFdHead(coreNum),
        gS1IdxOfFdHead(coreNum),
        s2SplitNumOfFdHead(coreNum),
        gS1SplitNumOfFdHead(coreNum),
        gS1LastPartSizeOfFdHead(coreNum),
        gS1IdxEndOfFdHead(coreNum * vecCubeRatio),
        gS1IdxEndOfFdHeadSplit(coreNum * vecCubeRatio),
        s2SplitStartIdxOfCore(coreNum) {}
};

// Split core output: FA split result
struct APSplitResult {
    uint32_t usedCoreNum { 0U };
    uint32_t vecCubeRatio { 0U };
    std::vector<uint32_t> bN2End {};
    std::vector<uint32_t> gS1End {};
    std::vector<uint32_t> s2End {};
    int64_t maxCost { 0 };
    uint32_t numOfFdHead { 0U };
    uint32_t maxS2SplitNum { 0U };
    uint32_t usedVecNumOfFd { 0U };
    APFlashDecodeResult fdRes { 0U, 0U };

    APSplitResult(uint32_t coreNum, uint32_t ratio) :
        bN2End(coreNum),
        vecCubeRatio(ratio),
        gS1End(coreNum),
        s2End(coreNum),
        fdRes(coreNum, ratio) {};
};

// Split core input: base info
struct APBaseInfo {
    uint32_t bSize { 0U };
    uint32_t n2Size { 0U };
    uint32_t gSize { 0U };
    uint32_t s1Size { 0U };
    uint32_t s2Size { 0U };
    bool isS1G { true };
    bool isAccumSeqS1 { false };
    bool isAccumSeqS2 { false };
    std::vector<int64_t> actualSeqS1Size {};
    std::vector<int64_t> actualSeqS2Size {};
    uint32_t actualLenQDims { 0U };
    uint32_t actualLenKvDims { 0U };
    bool attenMaskFlag { false };
    int32_t sparseMode { 0U };
    int64_t preToken { 0 };
    int64_t nextToken { 0 };
    int64_t actualSeqPrefixSize { 0 };
    int64_t keySinkNumber { 0 };
    bool supportFD { false };
};

// Split core input: split parameters
struct APSplitParam {
    uint32_t mBaseSize { 1U };
    uint32_t s2BaseSize { 1U };
    uint32_t gS1BaseSizeOfFd { 8U };
};

// Internal: split info
struct APSplitInfo {
    std::vector<uint32_t> s1GBaseNum {};
    std::vector<uint32_t> s2BaseNum {};
    std::vector<uint32_t> s1GTailSize {};
    std::vector<uint32_t> s2TailSize {};
    bool isKvSeqAllZero { true };

    explicit APSplitInfo(uint32_t batchSize) :
        s1GBaseNum(batchSize),
        s2BaseNum(batchSize),
        s1GTailSize(batchSize),
        s2TailSize(batchSize) {}
};

// Internal: cost info
struct APCostInfo {
    std::vector<int64_t> bN2CostOfEachBatch {};
    std::vector<uint32_t> bN2BlockOfEachBatch {};
    std::vector<int64_t> bN2LastBlockCostOfEachBatch {};
    uint32_t totalBlockNum { 0U };
    int64_t totalCost { 0 };

    explicit APCostInfo(uint32_t batchSize) :
        bN2CostOfEachBatch(batchSize),
        bN2BlockOfEachBatch(batchSize),
        bN2LastBlockCostOfEachBatch(batchSize) {}
};

// Internal: split context
struct APSplitContext {
    const APBaseInfo &baseInfo;
    const APSplitParam &splitParam;
    APSplitInfo splitInfo { 0U };
    APCostInfo costInfo { 0U };

    explicit APSplitContext(const APBaseInfo &info, const APSplitParam &param) :
        baseInfo(info),
        splitParam(param),
        splitInfo(info.bSize),
        costInfo(info.bSize) {}
};

// Internal: batch cache
struct APBatchCache {
    uint32_t bIdx { 0U };
    uint32_t s1Size { 0U };
    uint32_t s2Size { 0U };
    int64_t preTokenLeftUp { 0 };
    int64_t nextTokenLeftUp { 0 };
    APBlockCost<int64_t> typeCost {};
};

// Internal: S1G cache
struct APS1GCache {
    uint32_t bIdx { 0U };
    uint32_t s1GIdx { 0U };
    uint32_t s2Start { 0U };
    uint32_t s2End { 0U };
    int64_t s1GCost { 0 };
    int64_t s1GLastBlockCost { 0 };
    uint32_t s1GBlock { 0U };
    int64_t s1GNormalBlockCost { 0 };
    int64_t firstTokenWithSinkNumber { 0 };
};

// Internal: core cache
struct APCoreCache {
    int64_t costLimit { 0 };
    int64_t cost { 0 };
    uint32_t block { 0U };
};

// Internal: assign context
struct APAssignContext {
    uint32_t curBIdx { 0U };
    uint32_t curBN2Idx { 0U };
    uint32_t curS1GIdx { 0U };
    uint32_t curS2Idx { 0U };
    uint32_t curCoreIdx { 0U };
    int64_t unassignedCost { 0 };
    uint32_t usedCoreNum { 0U };
    uint32_t curKvSplitPart { 1U };

    int64_t bN2Cost { 0 };
    uint32_t bN2Block { 0U };
    bool isFinished { false };
    APBatchCache batchCache {};
    APS1GCache s1GCache {};
    APCoreCache coreCache {};
};

class AttentionPioneerMetadataKernel : public CpuKernel {
public:
    AttentionPioneerMetadataKernel() = default;
    ~AttentionPioneerMetadataKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    bool Prepare(CpuKernelContext &ctx);
    bool ParamsParse();
    bool CheckSingleParam();
    bool ParseAxisInfo();
    bool GetS1Size();
    bool GetS2Size();
    bool GetBatchSize();
    bool GetActualSeqLenSize(uint32_t &size);
    bool GetActualSeqKVLenSize(uint32_t &size);
    bool CheckSingleParaInputLayout();
    bool CheckSingleParaNumHeads();
    bool GetPreNextToken();
    bool GetActualSeqInfo();
    bool ParamsCheck();
    void CheckIsMla();
    void CalcInnerSizeMla(uint32_t seqSize);
    void CalcMBaseSizeMla();
    void CreateSplitInputMla(APBaseInfo &baseInfo, APSplitParam &splitParam) const;
    bool DealSameSeqEachBatch();
    void CalcInnerSizeGqa(uint32_t seqSize);
    void CalcMBaseSizeGqa();
    void CreateSplitInputGqa(APBaseInfo &baseInfo, APSplitParam &splitParam) const;
    bool CheckFeature();
    bool CheckFeatureActualSeqLensQData();
    bool CheckFeatureActualSeqLensKvData();
    bool CheckFeatureAxisInfo() const;
    bool CheckMultiParaConsistency();
    bool CheckActualSeqLensQ() const;
    bool CheckActualSeqLensKv() const;
    bool GenMetaData(const APSplitResult &splitRes);

    // util
    uint32_t GetS1SeqSize(uint32_t bIdx, const APBaseInfo &baseInfo);
    uint32_t GetS2SeqSize(uint32_t bIdx, const APBaseInfo &baseInfo);
    int64_t CalcPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const APBaseInfo &baseInfo);
    int64_t CalcNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const APBaseInfo &baseInfo);
    APRange<uint32_t> CalcS2Range(uint32_t s1GIdx, const APBaseInfo &baseInfo, const APSplitParam &splitParam,
        const APBatchCache &batchCache);
    int64_t CalcCost(uint32_t basicM, uint32_t basicS2);
    APBlockCost<int64_t> CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize, uint32_t s1GTailSize,
        uint32_t s2TailSize);

    // cache calculation
    void CalcBatchCache(uint32_t bIdx, const APSplitContext &splitContext, APBatchCache &batchCache);
    void CalcS1GCache(uint32_t s1GIdx, const APSplitContext &splitContext, const APBatchCache &batchCache, APS1GCache &s1GCache);
    void CopyTmpResult(APSplitResult &tmpRes, APSplitResult &splitRes);
    void ClearTmpResult(APSplitResult &tmpResult);
    void CalcS1GCacheWithSinkNumber(uint32_t s1GIdx, const APSplitContext &splitContext, const APBatchCache &batchCache, APS1GCache &s1GCache, uint32_t sinkNumber);
    void UpdateCurS2IdxWithSinkNumber(const APSplitContext &splitContext, APAssignContext &assignContext, uint32_t sinkNumber);

    // preprocess
    void CalcSplitInfo(APSplitContext &splitContext);
    void CalcBatchCost(uint32_t bIdx, const APSplitContext &splitContext, APCostInfo &costInfo);
    void CalcCostInfo(APSplitContext &splitContext);

    // assign
    void UpdateCursor(const APSplitContext &splitContext, APAssignContext &assignContext);
    void AssignByBatch(const APSplitContext &splitContext, APAssignContext &assignContext);
    void AssignByRow(const APSplitContext &splitContext, APAssignContext &assignContext);
    void AssignByBlock(const APSplitContext &splitContext, APAssignContext &assignContext);
    void ForceAssign(const APSplitContext &splitContext, APAssignContext &assignContext);

    // FD
    bool IsNeedRecordFDInfo(const APAssignContext &assignContext, const APSplitResult &splitRes);
    void RecordFDInfo(const APSplitContext &splitContext, const APAssignContext &assignContext, APSplitResult &result);

    // main
    void SplitFD(APSplitResult &result);
    void CalcSplitPlan(uint32_t coreNum, int64_t costLimit, const APSplitContext &splitContext, APSplitResult &result);
    void SplitCore(uint32_t coreNum, const APBaseInfo &baseInfo, const APSplitParam &splitParam, APSplitResult &result);

private:
    enum class ParamId : uint32_t {
        actualSeqLengthsOptional = 0,
        actualSeqLengthsKvOptional = 1,
        metaData = 0,
    };

private:
    CpuKernelContext *context_ = nullptr;

    // input
    Tensor *actualSeqLengths_ = nullptr;
    Tensor *actualSeqLengthsKv_ = nullptr;

    // output
    Tensor *metaData_ = nullptr;

    // attributes
    uint32_t numHeadsQ_ = 0;
    uint32_t numHeadsKv_ = 1;
    uint32_t headDimQK_ = 0;
    uint32_t headDimV_ = 0;
    uint32_t ropeHeadDim_ = 0;
    uint32_t batchSize_ = 0;
    uint32_t sparseMode_ = 0;
    int64_t preTokens_ = 2147483647;
    int64_t nextTokens_ = 2147483647;
    int64_t blockSize_ = 0;
    std::string inputLayout_ = "TND";
    std::string inputLayoutKv_ = "TND";
    int64_t kSinkNum_ = 0;
    std::string socVersion_ = "ascend950";
    uint32_t aicCoreNum_ = 32U;
    uint32_t aivCoreNum_ = 64U;
    uint32_t cvRatio_ = 2;

    // derived
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t gSize = 0;
    int64_t s2Size = 0;
    uint32_t s1Size = 0;
    uint32_t bSize = 0;
    uint32_t actualLenQDims = 0;
    uint32_t actualLenKvDims = 0;
    int64_t preToken = 0;
    int64_t nextToken = 0;
    bool attenMaskFlag = false;
    bool isMlaSink = false;
    bool isAccumQSeq = false;
    bool isAccumKVSeq = false;
    bool needInit = false;
    bool isSameActualseq = true;
    bool splitKVFlag_ = false;
    bool softmaxWithBrcbFlag_ = false;
    bool batchContinuousFlag = false;
    bool actualSeqLenFlag = false;
    bool isSameSeqAllKVTensor = true;
    int64_t maxActualseq = 0;
    uint32_t sInnerLoopTimes_ = 0;
    uint32_t sInnerSizeTail_ = 0;
    uint32_t sInnerSizeAlign_ = 0;
    uint32_t actualSeqLengthsQSize_ = 0;
    uint32_t actualSeqLengthsKvSize_ = 0;
    uint32_t mBaseSize_ = 256;
    uint32_t sInnerSize_ = 0;
    uint32_t mFdBaseSize_ = 8;
    uint32_t numOfFdHead = 0;
    uint32_t usedCoreNum = 0;
    uint32_t usedVecNumOfFd = 0;
    std::vector<int64_t> kvListSeqLens {};
    std::vector<uint32_t> qSize = {};
    std::vector<uint32_t> kvSize = {};
};

} // namespace aicpu

#endif // ATTENTION_PIONEER_METADATA_AICPU_H
