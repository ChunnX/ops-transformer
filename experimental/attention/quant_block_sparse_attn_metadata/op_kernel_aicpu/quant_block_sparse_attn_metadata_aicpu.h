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
 * \file quant_block_sparse_attn_metadata_aicpu.h
 * \brief
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_METADATA_AICPU_H
#define QUANT_BLOCK_SPARSE_ATTN_METADATA_AICPU_H

#include <array>
#include <string>
#include <vector>
#include <limits>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"
#include "quant_block_sparse_attn_metadata.h"
#include "../../../../attention/common/op_kernel/aicpu_common.h"

namespace aicpu {

static const int64_t NUM_128 = 128L;

struct CoreCache {
    uint64_t blockLimit { 0U };
    uint64_t block { 0U };
};

struct AssignContext {
    uint32_t curCoreIdx { 0U };
    uint32_t curBIdx { 0U };
    uint32_t curN1Idx { 0U };
    uint32_t curBN1Idx { 0U };
    uint32_t curS1Idx { 0U };
    uint32_t curS2Idx { 0U };
    uint64_t unassignedBlock { 0U };
    uint64_t bN1Block { 0U };
    CoreCache coreCache {};
    bool isFinished { false };
};

struct SplitResult {
    uint32_t usedCoreNum { 0U };
    uint64_t maxBlock { 0U };
    std::vector<uint32_t> bN1End {};
    std::vector<uint32_t> s1End {};
    std::vector<uint32_t> s2End {};
    std::vector<uint32_t> firstFdDataWorkspaceIdx {};

    explicit SplitResult(uint32_t aicNum)
        : bN1End(aicNum),
          s1End(aicNum),
          s2End(aicNum),
          firstFdDataWorkspaceIdx(aicNum) {}
};

class QuantBlockSparseAttnMetadataCpuKernel : public CpuKernel {
public:
    QuantBlockSparseAttnMetadataCpuKernel() = default;
    ~QuantBlockSparseAttnMetadataCpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    bool Prepare(CpuKernelContext &ctx);
    bool ParamsCheck();
    bool CheckTensorData();
    bool ParamsInit();
    bool ScheduleSection(SplitResult &splitRes);
    bool GenMetadata(const SplitResult &splitRes);
    uint32_t GetRowBlockNum(uint32_t bIdx, uint32_t n1Idx, uint32_t s1Idx) const;
    uint64_t GetBN1BlockNum(uint32_t bIdx, uint32_t n1Idx) const;
    uint64_t CalcBlockNum() const;
    bool AdvanceToValidRow(AssignContext &assignContext) const;
    void AssignByBatch(AssignContext &assignContext) const;
    void AssignByRow(AssignContext &assignContext) const;
    void ScheduleFa(uint64_t blockNum, SplitResult &result);

private:
    CpuKernelContext *context_ = nullptr;
    // input tensor
    Tensor *sparseSeqLen_ = nullptr;
    Tensor *cuSeqlensQ_ = nullptr;
    Tensor *cuSeqlensKv_ = nullptr;
    Tensor *sequsedQ_ = nullptr;
    Tensor *sequsedKv_ = nullptr;
    // output tensor
    Tensor *metadata_ = nullptr;

    // attr
    int32_t batchSize_ = 0;
    int32_t numHeadsQ_ = 0;
    int32_t numHeadsKv_ = 0;
    int32_t headDim_ = 0;
    int32_t sparseBlockSizeQ_ = 128;
    int32_t sparseBlockSizeK_ = 128;
    int32_t quantMode_ = 1;
    int32_t maskMode_ = 3;
    std::string layoutQ_ = "TND";
    std::string layoutKv_ = "PA_BNSD";
    std::string layoutSparseIndices_ = "B_N_Qb_Kb";

    // platform
    std::string socVersion_ = "";
    int32_t aicCoreNum_ = 36;
    int32_t aivCoreNum_ = 72;

    // SplitParams
    uint32_t sparseSeqLenDimNum_ = 0;
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
        metadata = 0,
    };
};
} // namespace aicpu

#endif
