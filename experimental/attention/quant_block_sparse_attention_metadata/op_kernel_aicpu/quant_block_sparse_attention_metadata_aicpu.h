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
#include "../../../../attention/common/op_kernel/load_balance/block_balance/block_balance.h"
#include "../../common/op_kernel/aicpu_common.h"

namespace aicpu {

static const int64_t NUM_128 = 128L;

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
    bool BalanceSchedule(load_balance::BlockBalanceResult &splitRes);
    bool GenMetaData(const load_balance::BlockBalanceResult &splitRes);

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
    load_balance::BlockBalanceParam param;

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
