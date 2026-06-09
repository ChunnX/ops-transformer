/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_block_sparse_attention_metadata_aicpu.h
 * \brief
 */

#ifndef QUANT_BLOCK_SPARSE_ATTENTION_METADATA_AICPU_H
#define QUANT_BLOCK_SPARSE_ATTENTION_METADATA_AICPU_H

#include <array>
#include <string>
#include <vector>
#include <limits>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"
#include "quant_block_sparse_attention_metadata.h"
#include "../../common/op_kernel/aicpu_common.h"

using namespace optiling;
using namespace std;

namespace aicpu {

static const int64_t NUM_8192 = 8192L;
static const int64_t NUM_4096 = 4096L;
static const int64_t NUM_2048 = 2048L;
static const int64_t NUM_1024 = 1024L;
static const int64_t NUM_512 = 512L;
static const int64_t NUM_256 = 256L;
static const int64_t NUM_128 = 128L;
static const int64_t NUM_64 = 64L;
static const int64_t NUM_32 = 32L;

struct SplitInfo {
    std::vector<std::vector<uint32_t>> sparseSeqLen {};
    std::vector<uint64_t> bN2BlockOfEachBatch {};
    uint64_t totalBlock { 0U };
    bool isAllZero { true };
};

struct CoreCache {
    uint64_t blockLimit { 0U };
    uint64_t block { 0U };
};

struct AssignContext {
    uint32_t curCoreIdx { 0U };
    uint32_t curBIdx { 0U };
    uint32_t curBN2Idx { 0U };
    uint32_t curS1GIdx { 0U };
    uint32_t curS2Idx { 0U };
    uint64_t unassignedBlock { 0U };
    uint64_t bN2Block { 0U };
    CoreCache coreCache {};
    bool isFinished { false };
};

struct SplitResult {
    uint32_t usedCoreNum { 0U };
    uint64_t maxBlock { 0U };
    std::vector<uint32_t> bN2End {};
    std::vector<uint32_t> gS1End {};
    std::vector<uint32_t> s2End {};
    std::vector<uint32_t> firstFdDataWorkspaceIdx {};

    explicit SplitResult(uint32_t aicNum)
        : bN2End(aicNum),
          gS1End(aicNum),
          s2End(aicNum),
          firstFdDataWorkspaceIdx(aicNum) {}
};

class QuantBlockSparseAttentionMetadataCpuKernel : public CpuKernel {
public:
    QuantBlockSparseAttentionMetadataCpuKernel() = default;
    ~QuantBlockSparseAttentionMetadataCpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    bool Prepare(CpuKernelContext &ctx);
    bool ParamsCheck();
    bool CheckTensorData();
    bool ParamsInit();
    bool BalanceSchedule(SplitResult &splitRes);
    bool GenMetaData(const SplitResult &splitRes);
    uint32_t GetRowBlockNum(const SplitInfo &splitInfo, uint32_t bIdx, uint32_t s1Idx) const;
    void CalcSplitInfo(SplitInfo &splitInfo);
    void UpdateCurrentBN2Block(const SplitInfo &splitInfo, AssignContext &assignContext) const;
    void UpdateCursor(const SplitInfo &splitInfo, AssignContext &assignContext) const;
    void AssignByBatch(const SplitInfo &splitInfo, AssignContext &assignContext);
    void AssignByRow(const SplitInfo &splitInfo, AssignContext &assignContext);
    void ForceAssign(const SplitInfo &splitInfo, AssignContext &assignContext);
    void AssignBlocksToCore(const SplitInfo &splitInfo, AssignContext &assignContext, SplitResult &result);
    void CalcSplitPlan(const SplitInfo &splitInfo, SplitResult &result);

private:
    CpuKernelContext *context_ = nullptr;
    // input tensor
    Tensor *sparseSeqLen_ = nullptr;
    Tensor *cuSeqlensQ_ = nullptr;
    Tensor *cuSeqlensKv_ = nullptr;
    Tensor *sequsedQ_ = nullptr;
    Tensor *sequsedKv_ = nullptr;
    // output tensor
    Tensor *metaData_ = nullptr;

    // attr
    int32_t batchSize_ = 0;
    int32_t numHeadsQ_ = 0;
    int32_t numHeadsKv_ = 0;
    int32_t headDim_ = 0;
    int32_t sparseBlockSizeQ_ = 128;
    int32_t sparseBlockSizeKv_ = 128;
    int32_t quantMode_ = 0; // TODO: confirm default value for quant_mode
    int32_t maskMode_ = 3;
    std::string layoutQ_ = "BSND"; // TODO: confirm default value for layout_q
    std::string layoutKv_ = "PA_BNSD";
    std::string layoutSparseIndices_ = "B_N_Qb_Kb";

    // platform
    std::string socVersion_ = "";
    int32_t aicCoreNum_ = 36;
    int32_t aivCoreNum_ = 72;

    // SplitParams
    uint32_t groupSize_ = 0;
    uint32_t Qbmax_ = 0;
    uint32_t mBaseSize_ = NUM_128;
    uint32_t s2BaseSize_ = NUM_128;

private:
    enum class ParamId : uint32_t {
        // input
        sparseSeqLen = 0,
        cuSeqlensQ = 1,
        cuSeqlensKv = 2,
        sequsedQ = 3,
        sequsedKv = 4,
        // output
        metaData = 0,
    };
};
} // namespace aicpu

#endif
