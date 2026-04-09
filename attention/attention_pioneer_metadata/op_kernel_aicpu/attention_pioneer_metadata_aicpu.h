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

#include <string>
#include <vector>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"

namespace aicpu {

constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;

enum class APSparseMode : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
};

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
    // === Stage 1: Prepare ===
    bool Prepare(CpuKernelContext &ctx);

    // === Stage 2: CheckIsDecode ===
    void CheckIsDecode();

    // === Stage 3: ParseAxisInfo ===
    bool ParseAxisInfo();

    // === Stage 4: Tiling ===
    void SetTilingParams();

    // === Stage 5: ComputeSplitNBSeq ===
    void ComputeSplitNBSeq();

    // === Stage 6: GenMetaData ===
    bool GenMetaData();

    // Split-core helpers
    void GetPreNextTokensLeftUp(int64_t actualSeqLength, int64_t actualSeqLengthKV,
        int64_t &preTokensLeftUp, int64_t &nextTokensLeftUp) const;
    void FixParamWithRowInvalid(int64_t &actualSeqLength, int64_t actualSeqLengthKV,
        int64_t &preTokensLeftUp, int64_t &nextTokensLeftUp) const;
    int64_t GetCalcBlockNumsOneHead(int64_t actualSeqLength, int64_t actualSeqLengthKV,
        int64_t preTokensLeftUp, int64_t nextTokensLeftUp) const;
    int64_t GetCalcBlockNumsForS1G(int64_t actualSeqLength, int64_t actualSeqLengthKV,
        int64_t preTokensLeftUp, int64_t nextTokensLeftUp) const;
    int64_t GetActualInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd,
        int64_t innerBlockNums) const;
    int64_t SinkBlockCount(int64_t actualSeqLength) const;
    static int64_t ClipSInnerToken(int64_t val, int64_t lo, int64_t hi);

private:
    enum class ParamId : uint32_t {
        actualSeqLengthsOptional = 0,
        actualSeqLengthsKvOptional = 1,
        metaData = 0,
    };

    // Input/Output tensors
    Tensor *actualSeqLengths_ = nullptr;
    Tensor *actualSeqLengthsKv_ = nullptr;
    Tensor *metaData_ = nullptr;

    // Attributes
    uint32_t numHeadsQ_ = 0;
    uint32_t numHeadsKv_ = 1;
    uint32_t headDimQK_ = 0;
    uint32_t headDimV_ = 0;
    uint32_t ropeHeadDim_ = 0;
    uint32_t sparseMode_ = 0;
    int64_t preTokens_ = SPARSE_MODE_INT_MAX;
    int64_t nextTokens_ = SPARSE_MODE_INT_MAX;
    std::string inputLayout_ = "TND";
    int64_t kSinkNum_ = 0;
    int64_t blockSize_ = 0;
    std::string socVersion_ = "ascend950";
    uint32_t aicCoreNum_ = 32U;
    uint32_t aivCoreNum_ = 64U;

    // Derived parameters
    bool isDecode_ = false;
    uint32_t n1Size_ = 0;       // numHeadsQ
    uint32_t n2Size_ = 0;       // numHeadsKv
    uint32_t gSize_ = 1;        // n1Size / n2Size
    uint32_t bSize_ = 0;        // batch size (from actualSeqLengths shape)
    uint32_t s1Size_ = 0;       // max Q seq length (D-side already * G)
    int64_t s2Size_ = 0;        // max KV seq length
    int64_t preToken_ = 0;
    int64_t nextToken_ = 0;
    bool attenMaskFlag_ = false;

    // Tiling parameters (fixed per P/D side)
    uint32_t sOuterSize_ = 128;
    uint32_t sInnerSize_ = 128;

    // Per-batch seq lengths (non-incremental)
    std::vector<int64_t> qSeqLens_;
    std::vector<int64_t> kvSeqLens_;

    // Split-core result
    std::vector<int32_t> splitBnStartIdx_;
    std::vector<int32_t> splitGS1StartIdx_;
    uint32_t splitUsedCoreNum_ = 0;
    uint32_t splitS1OuterSize_ = 0;
};

} // namespace aicpu

#endif // ATTENTION_PIONEER_METADATA_AICPU_H
