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
    SplitResult splitRes(static_cast<uint32_t>(aicCoreNum_));
    success = BalanceSchedule(splitRes) && GenMetadata(splitRes);
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
    batchSize_ = sparseSeqLen_->GetTensorShape()->GetDimSize(0);
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
    if (batchSize_ <= 0 || numHeadsKv_ <= 0 || aicCoreNum_ <= 0 || Qbmax_ == 0U) {
        KERNEL_LOG_ERROR("invalid split params, batchSize=%d, numHeadsKv=%d, aicCoreNum=%d, Qbmax=%u",
                         batchSize_, numHeadsKv_, aicCoreNum_, Qbmax_);
        return false;
    }
    return true;
}

uint32_t QuantBlockSparseAttentionMetadataCpuKernel::GetRowBlockNum(const SplitInfo &splitInfo,
                                                                    uint32_t bIdx, uint32_t s1Idx) const
{
    if (bIdx >= static_cast<uint32_t>(batchSize_) || s1Idx >= Qbmax_) {
        return 0U;
    }
    if (bIdx >= splitInfo.sparseSeqLen.size() || s1Idx >= splitInfo.sparseSeqLen[bIdx].size()) {
        return 0U;
    }
    return splitInfo.sparseSeqLen[bIdx][s1Idx];
}

void QuantBlockSparseAttentionMetadataCpuKernel::CalcSplitInfo(SplitInfo &splitInfo)
{
    uint32_t sparseSeqLenSize = static_cast<uint32_t>(batchSize_) * Qbmax_;
    const int32_t *sparseSeqLenData = static_cast<const int32_t *>(sparseSeqLen_->GetData());
    uint32_t batchSize = static_cast<uint32_t>(batchSize_);
    splitInfo.sparseSeqLen.assign(batchSize, std::vector<uint32_t>(Qbmax_, 0U));
    splitInfo.bN2BlockOfEachBatch.assign(batchSize, 0U);
    splitInfo.totalBlock = 0U;
    splitInfo.isAllZero = true;

    for (uint32_t bIdx = 0; bIdx < batchSize; ++bIdx) {
        uint64_t batchBlock = 0U;
        for (uint32_t s1Idx = 0; s1Idx < Qbmax_; ++s1Idx) {
            uint32_t idx = bIdx * Qbmax_ + s1Idx;
            uint32_t rowBlock = 0U;
            if (idx < sparseSeqLenSize && sparseSeqLenData[idx] > 0) {
                rowBlock = static_cast<uint32_t>(sparseSeqLenData[idx]);
            }
            splitInfo.sparseSeqLen[bIdx][s1Idx] = rowBlock;
            batchBlock += rowBlock;
        }
        splitInfo.bN2BlockOfEachBatch[bIdx] = batchBlock;
        splitInfo.totalBlock += batchBlock * static_cast<uint32_t>(numHeadsKv_);
    }
    splitInfo.isAllZero = (splitInfo.totalBlock == 0U);
}

void QuantBlockSparseAttentionMetadataCpuKernel::UpdateCurrentBN2Block(const SplitInfo &splitInfo,
                                                                       AssignContext &assignContext) const
{
    if (assignContext.curBN2Idx >= static_cast<uint32_t>(batchSize_ * numHeadsKv_)) {
        assignContext.bN2Block = 0U;
        return;
    }
    assignContext.curBIdx = assignContext.curBN2Idx / static_cast<uint32_t>(numHeadsKv_);
    assignContext.bN2Block = splitInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
}

void QuantBlockSparseAttentionMetadataCpuKernel::UpdateCursor(const SplitInfo &splitInfo,
                                                                 AssignContext &assignContext) const
{
    while (assignContext.curBN2Idx < static_cast<uint32_t>(batchSize_ * numHeadsKv_)) {
        assignContext.curBIdx = assignContext.curBN2Idx / static_cast<uint32_t>(numHeadsKv_);
        while (assignContext.curS1GIdx < Qbmax_ &&
               GetRowBlockNum(splitInfo, assignContext.curBIdx, assignContext.curS1GIdx) == 0U) {
            assignContext.curS1GIdx++;
        }
        if (assignContext.curS1GIdx < Qbmax_) {
            assignContext.curS2Idx = 0U;
            UpdateCurrentBN2Block(splitInfo, assignContext);
            return;
        }
        assignContext.curBN2Idx++;
        assignContext.curS1GIdx = 0U;
        assignContext.curS2Idx = 0U;
    }
    assignContext.curS1GIdx = 0U;
    assignContext.curS2Idx = 0U;
    assignContext.isFinished = true;
    UpdateCurrentBN2Block(splitInfo, assignContext);
}

void QuantBlockSparseAttentionMetadataCpuKernel::AssignByBatch(const SplitInfo &splitInfo,
                                                               AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }
    UpdateCursor(splitInfo, assignContext);
    while (!assignContext.isFinished &&
           assignContext.curBN2Idx < static_cast<uint32_t>(batchSize_ * numHeadsKv_) &&
           assignContext.curS1GIdx == 0U) {
        UpdateCurrentBN2Block(splitInfo, assignContext);
        if (assignContext.bN2Block == 0U) {
            assignContext.curBN2Idx++;
            UpdateCursor(splitInfo, assignContext);
            continue;
        }
        if (assignContext.coreCache.block + assignContext.bN2Block > assignContext.coreCache.blockLimit) {
            break;
        }
        assignContext.coreCache.block += assignContext.bN2Block;
        assignContext.curBN2Idx++;
        assignContext.curS1GIdx = 0U;
        assignContext.curS2Idx = 0U;
        UpdateCursor(splitInfo, assignContext);
    }
}

void QuantBlockSparseAttentionMetadataCpuKernel::AssignByRow(const SplitInfo &splitInfo,
                                                             AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }
    UpdateCursor(splitInfo, assignContext);
    while (!assignContext.isFinished &&
           assignContext.curBN2Idx < static_cast<uint32_t>(batchSize_ * numHeadsKv_)) {
        uint32_t rowBlock = GetRowBlockNum(splitInfo, assignContext.curBIdx, assignContext.curS1GIdx);
        if (rowBlock == 0U) {
            assignContext.curS1GIdx++;
            UpdateCursor(splitInfo, assignContext);
            continue;
        }
        if (assignContext.coreCache.block + rowBlock > assignContext.coreCache.blockLimit) {
            break;
        }
        assignContext.coreCache.block += rowBlock;
        if (assignContext.bN2Block > rowBlock) {
            assignContext.bN2Block -= rowBlock;
        } else {
            assignContext.bN2Block = 0U;
        }
        assignContext.curS1GIdx++;
        UpdateCursor(splitInfo, assignContext);
    }
}

void QuantBlockSparseAttentionMetadataCpuKernel::ForceAssign(const SplitInfo &splitInfo,
                                                             AssignContext &assignContext)
{
    if (assignContext.isFinished || assignContext.coreCache.block != 0U) {
        return;
    }
    UpdateCursor(splitInfo, assignContext);
    if (assignContext.isFinished) {
        return;
    }
    uint32_t rowBlock = GetRowBlockNum(splitInfo, assignContext.curBIdx, assignContext.curS1GIdx);
    assignContext.coreCache.block += rowBlock;
    if (assignContext.bN2Block > rowBlock) {
        assignContext.bN2Block -= rowBlock;
    } else {
        assignContext.bN2Block = 0U;
    }
    assignContext.curS1GIdx++;
    UpdateCursor(splitInfo, assignContext);
}

void QuantBlockSparseAttentionMetadataCpuKernel::AssignBlocksToCore(const SplitInfo &splitInfo,
                                                                    AssignContext &assignContext, SplitResult &result)
{
    assignContext.coreCache = {};
    uint32_t remainingCore = static_cast<uint32_t>(aicCoreNum_) - assignContext.curCoreIdx;
    assignContext.coreCache.blockLimit =
        (assignContext.unassignedBlock + remainingCore - 1U) / remainingCore;

    AssignByBatch(splitInfo, assignContext);
    AssignByRow(splitInfo, assignContext);
    ForceAssign(splitInfo, assignContext);

    result.bN2End[assignContext.curCoreIdx] = assignContext.curBN2Idx;
    result.gS1End[assignContext.curCoreIdx] = assignContext.curS1GIdx;
    result.s2End[assignContext.curCoreIdx] = 0U;
    result.firstFdDataWorkspaceIdx[assignContext.curCoreIdx] = 0U;
    result.maxBlock = std::max(result.maxBlock, assignContext.coreCache.block);
    if (assignContext.unassignedBlock > assignContext.coreCache.block) {
        assignContext.unassignedBlock -= assignContext.coreCache.block;
    } else {
        assignContext.unassignedBlock = 0U;
    }
}

void QuantBlockSparseAttentionMetadataCpuKernel::CalcSplitPlan(const SplitInfo &splitInfo, SplitResult &result)
{
    AssignContext assignContext {};
    assignContext.unassignedBlock = splitInfo.totalBlock;
    UpdateCurrentBN2Block(splitInfo, assignContext);
    UpdateCursor(splitInfo, assignContext);

    for (uint32_t i = 0; i < static_cast<uint32_t>(aicCoreNum_); i++) {
        if (assignContext.isFinished || assignContext.unassignedBlock == 0U) {
            break;
        }
        assignContext.curCoreIdx = i;
        AssignBlocksToCore(splitInfo, assignContext, result);
        result.usedCoreNum = i + 1U;
    }
    result.usedCoreNum = std::max(result.usedCoreNum, 1U);
}

bool QuantBlockSparseAttentionMetadataCpuKernel::BalanceSchedule(SplitResult &splitRes)
{
    SplitInfo splitInfo;
    CalcSplitInfo(splitInfo);
    if (splitInfo.isAllZero) {
        splitRes.usedCoreNum = 1U;
        splitRes.bN2End[0] = static_cast<uint32_t>(batchSize_ * numHeadsKv_);
        splitRes.gS1End[0] = 0U;
        splitRes.s2End[0] = 0U;
        splitRes.firstFdDataWorkspaceIdx[0] = 0U;
        return true;
    }
    CalcSplitPlan(splitInfo, splitRes);
    return true;
}

bool QuantBlockSparseAttentionMetadataCpuKernel::GenMetadata(const SplitResult &splitRes)
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
        uint32_t bn1Start = (i == 0) ? 0U : splitRes.bN2End[i - 1] * groupSize_;
        uint32_t bn1End = splitRes.bN2End[i] * groupSize_;
        if (i > 0) {
            faMetadata.setFaMetadata(i, optiling::FA_M_START_INDEX, splitRes.gS1End[i - 1]);
            faMetadata.setFaMetadata(i, optiling::FA_S2_START_INDEX, splitRes.s2End[i - 1]);
        }
        faMetadata.setFaMetadata(i, optiling::FA_BN1_START_INDEX, bn1Start);
        faMetadata.setFaMetadata(i, optiling::FA_BN1_END_INDEX, bn1End);
        faMetadata.setFaMetadata(i, optiling::FA_M_END_INDEX, splitRes.gS1End[i]);
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
