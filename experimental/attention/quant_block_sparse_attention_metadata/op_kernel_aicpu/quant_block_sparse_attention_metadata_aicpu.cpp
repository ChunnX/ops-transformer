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
 * \file quant_block_sparse_attention_metadata_aicpu.cpp
 * \brief
 */

#include "log.h"
#include "status.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include "quant_block_sparse_attention_metadata_aicpu.h"

#define KERNEL_STATUS_OK 0
#define KERNEL_STATUS_PARAM_INVALID 1

namespace aicpu {
uint32_t QuantBlockSparseAttentionMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    load_balance::BlockBalanceResult splitRes(static_cast<uint32_t>(aicCoreNum_));
    success = BalanceSchedule(splitRes) && GenMetaData(splitRes);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool QuantBlockSparseAttentionMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    context_ = &ctx;
    // input
    sparseSeqLen_ = ctx.Input(static_cast<uint32_t>(ParamId::sparseSeqLen));
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
    cuSeqlensKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensKv));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    sequsedKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedKv));
    // output
    metaData_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    bool requiredAttrs =
        GetAttrValue(ctx, "soc_version", socVersion_) &&
        GetAttrValue(ctx, "aic_core_num", aicCoreNum_) &&
        GetAttrValue(ctx, "aiv_core_num", aivCoreNum_);
    if (!requiredAttrs) {
        return false;
    }
    // attributes optional
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "num_heads_q", numHeadsQ_);
    GetAttrValueOpt(ctx, "num_heads_kv", numHeadsKv_);
    GetAttrValueOpt(ctx, "head_dim", headDim_);
    GetAttrValueOpt(ctx, "sparse_block_size_q", sparseBlockSizeQ_);
    GetAttrValueOpt(ctx, "sparse_block_size_kv", sparseBlockSizeKv_);
    GetAttrValueOpt(ctx, "quant_mode", quantMode_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "layout_kv", layoutKv_);
    GetAttrValueOpt(ctx, "layout_sparse_indices", layoutSparseIndices_);
    return (ParamsCheck() && ParamsInit());
}

bool QuantBlockSparseAttentionMetadataCpuKernel::ParamsCheck()
{
    return CheckTensorData();
}

bool QuantBlockSparseAttentionMetadataCpuKernel::CheckTensorData()
{
    // REQUIRED_INPUT tensor check
    if (sparseSeqLen_ == nullptr) {
        KERNEL_LOG_ERROR("sparseSeqLen is nullptr");
        return false;
    }
    if (sparseSeqLen_->GetTensorShape() == nullptr || sparseSeqLen_->GetTensorShape()->GetDimSize() != 2) {
        KERNEL_LOG_ERROR("sparseSeqLen must be 2D tensor");
        return false;
    }
    if (sparseSeqLen_->GetTensorShape()->GetDimSize() > 0 && sparseSeqLen_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("sparseSeqLen data is nullptr");
        return false;
    }
    // OPTIONAL_INPUT tensor check
    if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetTensorShape()->GetDimSize() > 0 &&
        cuSeqlensQ_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("cuSeqlensQ data is nullptr");
        return false;
    }
    if (cuSeqlensKv_ != nullptr && cuSeqlensKv_->GetTensorShape()->GetDimSize() > 0 &&
        cuSeqlensKv_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("cuSeqlensKv data is nullptr");
        return false;
    }
    if (sequsedQ_ != nullptr && sequsedQ_->GetTensorShape()->GetDimSize() > 0 && sequsedQ_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("sequsedQ data is nullptr");
        return false;
    }
    if (sequsedKv_ != nullptr && sequsedKv_->GetTensorShape()->GetDimSize() > 0 && sequsedKv_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("sequsedKv data is nullptr");
        return false;
    }
    // OUTPUT tensor check
    if (metaData_ == nullptr || metaData_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("metaData is empty");
        return false;
    }
    return true;
}

bool QuantBlockSparseAttentionMetadataCpuKernel::ParamsInit()
{
    int64_t sparseSeqLenBatchSize = sparseSeqLen_->GetTensorShape()->GetDimSize(0);
    if (batchSize_ <= 0) {
        KERNEL_LOG_ERROR("invalid batch size, batchSize=%d", batchSize_);
        return false;
    }
    if (sparseSeqLenBatchSize != static_cast<int64_t>(batchSize_)) {
        KERNEL_LOG_ERROR("sparseSeqLen shape[0] must be batchSize, got shape[0]=%ld, batchSize=%d",
                         sparseSeqLenBatchSize, batchSize_);
        return false;
    }
    if (numHeadsKv_ == 0) {
        numHeadsKv_ = numHeadsQ_;
        groupSize_ = 1;
    } else {
        groupSize_ = numHeadsQ_ / numHeadsKv_;
    }
    if (groupSize_ == 0U) {
        KERNEL_LOG_ERROR("invalid group size, numHeadsQ=%d, numHeadsKv=%d", numHeadsQ_, numHeadsKv_);
        return false;
    }
    mBaseSize_ = static_cast<uint32_t>(std::max(sparseBlockSizeQ_, 1));
    s2BaseSize_ = static_cast<uint32_t>(std::max(sparseBlockSizeKv_, 1));
    Qbmax_ = static_cast<uint32_t>(sparseSeqLen_->GetTensorShape()->GetDimSize(1));
    if (numHeadsKv_ <= 0 || aicCoreNum_ <= 0 || Qbmax_ == 0U) {
        KERNEL_LOG_ERROR("invalid split params, batchSize=%d, numHeadsKv=%d, aicCoreNum=%d, Qbmax=%u",
                         batchSize_, numHeadsKv_, aicCoreNum_, Qbmax_);
        return false;
    }
    param.batchSize = static_cast<uint32_t>(batchSize_);
    param.kvHeadNum = static_cast<uint32_t>(numHeadsKv_);
    param.aicCoreNum = static_cast<uint32_t>(aicCoreNum_);
    param.qBlockMax = Qbmax_;
    param.sparseSeqLen = static_cast<const int32_t *>(sparseSeqLen_->GetData());
    return true;
}

bool QuantBlockSparseAttentionMetadataCpuKernel::BalanceSchedule(load_balance::BlockBalanceResult &splitRes)
{
    return load_balance::BlockBalance::Compute(param, splitRes) == load_balance::BLOCK_BALANCE_SUCCESS;
}

bool QuantBlockSparseAttentionMetadataCpuKernel::GenMetaData(const load_balance::BlockBalanceResult &splitRes)
{
    if (metaData_ == nullptr || metaData_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("metadata is empty");
        return false;
    }
    optiling::detail::FaMetaData faMetadata(metaData_->GetData());
    uint32_t *ptr = static_cast<uint32_t *>(metaData_->GetData());
    ptr[1] = 0U;
    ptr[2] = mBaseSize_;
    ptr[3] = s2BaseSize_;
    for (uint32_t i = 0; i < optiling::AIC_CORE_NUM; i++) {
        faMetadata.setFaMetadata(i, optiling::FA_CORE_ENABLE_INDEX, 0U);
        faMetadata.setFaMetadata(i, optiling::FA_BN1_START_INDEX, 0U);
        faMetadata.setFaMetadata(i, optiling::FA_M_START_INDEX, 0U);
        faMetadata.setFaMetadata(i, optiling::FA_S2_START_INDEX, 0U);
        faMetadata.setFaMetadata(i, optiling::FA_BN1_END_INDEX, 0U);
        faMetadata.setFaMetadata(i, optiling::FA_M_END_INDEX, 0U);
        faMetadata.setFaMetadata(i, optiling::FA_S2_END_INDEX, 0U);
        faMetadata.setFaMetadata(i, optiling::FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, 0U);
    }
    for (uint32_t i = 0; i < optiling::AIV_CORE_NUM; i++) {
        for (uint32_t j = 0; j < optiling::FD_METADATA_SIZE; j++) {
            faMetadata.fdMetadata[i * optiling::FD_METADATA_SIZE + j] = 0U;
        }
    }
    for (uint32_t i = 0; i < splitRes.usedCoreNum; i++) {
        faMetadata.setFaMetadata(i, optiling::FA_CORE_ENABLE_INDEX, 1U);
        // 分核内部使用BN2，metadata输出需要BN1，因此乘以groupSize转换。
        uint32_t bn1Start = (i == 0) ? 0U : splitRes.bN2End[i - 1] * groupSize_;
        uint32_t bn1End = splitRes.bN2End[i] * groupSize_;
        if (i > 0) {
            faMetadata.setFaMetadata(i, optiling::FA_M_START_INDEX, splitRes.s1End[i - 1]);
            faMetadata.setFaMetadata(i, optiling::FA_S2_START_INDEX, splitRes.s2End[i - 1]);
        }
        faMetadata.setFaMetadata(i, optiling::FA_BN1_START_INDEX, bn1Start);
        faMetadata.setFaMetadata(i, optiling::FA_BN1_END_INDEX, bn1End);
        faMetadata.setFaMetadata(i, optiling::FA_M_END_INDEX, splitRes.s1End[i]);
        faMetadata.setFaMetadata(i, optiling::FA_S2_END_INDEX, splitRes.s2End[i]);
        faMetadata.setFaMetadata(i, optiling::FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
                                 splitRes.firstFdDataWorkspaceIdx[i]);
    }
    return true;
}

namespace {
static const char *kernelType = "QuantBlockSparseAttentionMetadata";
REGISTER_CPU_KERNEL(kernelType, QuantBlockSparseAttentionMetadataCpuKernel);
} // namespace

} // namespace aicpu
