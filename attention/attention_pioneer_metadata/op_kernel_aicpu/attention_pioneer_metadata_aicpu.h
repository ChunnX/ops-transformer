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

constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;

enum class APSparseMode : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
};

template<typename T>
T APClip(T value, T minValue, T maxValue)
{
    if (value < minValue) return minValue;
    if (value > maxValue) return maxValue;
    return value;
}

template <typename T>
inline auto APAlign(T num, T rnd) -> T
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

template <typename T>
inline T CeilDiv(T a, T b)
{
    return (a + b - 1) / b;
}

class AttentionPioneerMetadataKernel : public CpuKernel {
public:
    AttentionPioneerMetadataKernel() = default;
    ~AttentionPioneerMetadataKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    // Prepare & validation
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
    bool CheckFeature();
    bool CheckFeatureActualSeqLensQData();
    bool CheckFeatureActualSeqLensKvData();
    bool CheckFeatureAxisInfo() const;
    bool CheckMultiParaConsistency();
    bool CheckActualSeqLensQ() const;
    bool CheckActualSeqLensKv() const;

    // Inner size calculation
    void CalcInnerSizeMla(uint32_t seqSize);
    void CalcMBaseSizeMla();
    void CalcInnerSizeGqa(uint32_t seqSize);
    void CalcMBaseSizeGqa();
    bool DealSameSeqEachBatch();

    // A5 greedy split-core algorithm (ported from PromptFlashAttentionPioneerSplitNBSeq)
    void ComputeSplitNBSeq();
    uint32_t GetQSeqSize(uint32_t bIdx) const;
    uint32_t GetKVSeqSize(uint32_t bIdx) const;
    void GetPreNextTokensLeftUp(int64_t actualSeqLength, int64_t actualSeqLengthKV,
        int64_t &preTokensLeftUp, int64_t &nextTokensLeftUp) const;
    void FixParamWithRowInvalid(int64_t &actualSeqLength, int64_t actualSeqLengthKV,
        int64_t &preTokensLeftUp, int64_t &nextTokensLeftUp) const;
    static int64_t ClipSInnerToken(int64_t sInnerToken, int64_t minValue, int64_t maxValue);
    int64_t GetActualInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd,
        int64_t innerBlockNums) const;
    int64_t GetCalcBlockNumsOneHead(int64_t actualSeqLength, int64_t actualSeqLengthKV,
        uint32_t sOuterSize, uint32_t sInnerSize,
        int64_t preTokensLeftUp, int64_t nextTokensLeftUp) const;
    int64_t GetCalcBlockNumsForS1G(int64_t actualSeqLength, int64_t actualSeqLengthKV,
        uint32_t sOuterSize, uint32_t sInnerSize,
        int64_t preTokensLeftUp, int64_t nextTokensLeftUp) const;
    int64_t SinkBlockCount(int64_t actualSeqLength, uint32_t sOuterSize, uint32_t sInnerSize) const;

    // Output generation
    bool GenMetaData();

    // Layout helpers: check if Q-side layout has TND/NTD semantics
    bool IsQLayoutTND() const {
        return inputLayout_ == "TND" || inputLayout_ == "TND_NTD";
    }
    bool IsQLayoutNTD() const {
        return inputLayout_ == "NTD" || inputLayout_ == "NTD_TND";
    }
    bool IsQLayoutTNDOrNTD() const {
        return IsQLayoutTND() || IsQLayoutNTD();
    }

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
    std::vector<int64_t> kvListSeqLens {};
    std::vector<uint32_t> qSize = {};
    std::vector<uint32_t> kvSize = {};

    // Split-core result (written by ComputeSplitNBSeq, read by GenMetaData)
    std::vector<int32_t> splitBnStartIdx_ {};
    std::vector<int32_t> splitGS1StartIdx_ {};
    uint32_t splitUsedCoreNum_ = 0;
    uint32_t splitS1OuterSize_ = 0;
};

} // namespace aicpu

#endif // ATTENTION_PIONEER_METADATA_AICPU_H
