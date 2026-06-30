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
 * \file quant_block_sparse_attn_metadata_aicpu.cpp
 * \brief
 */

#include "log.h"
#include "status.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include "quant_block_sparse_attn_metadata_aicpu.h"

#define KERNEL_STATUS_OK 0
#define KERNEL_STATUS_PARAM_INVALID 1

namespace aicpu {
uint32_t QuantBlockSparseAttnMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    SplitResult splitRes(static_cast<uint32_t>(aicCoreNum_));
    success = ScheduleSection(splitRes) && GenMetadata(splitRes);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool QuantBlockSparseAttnMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    context_ = &ctx;
    // input
    sparseSeqLen_ = ctx.Input(static_cast<uint32_t>(ParamId::sparseSeqLen));
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
    cuSeqlensKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensKv));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    sequsedKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedKv));
    // output
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metadata));

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
    GetAttrValueOpt(ctx, "sparse_block_size_k", sparseBlockSizeK_);
    GetAttrValueOpt(ctx, "quant_mode", quantMode_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "layout_kv", layoutKv_);
    GetAttrValueOpt(ctx, "layout_sparse_indices", layoutSparseIndices_);
    return (ParamsCheck() && ParamsInit());
}

bool QuantBlockSparseAttnMetadataCpuKernel::ParamsCheck()
{
    return CheckTensorData();
}

bool QuantBlockSparseAttnMetadataCpuKernel::CheckTensorData()
{
    // REQUIRED_INPUT tensor check
    if (sparseSeqLen_ == nullptr) {
        KERNEL_LOG_ERROR("sparseSeqLen is nullptr");
        return false;
    }
    if (sparseSeqLen_->GetTensorShape() == nullptr ||
        (sparseSeqLen_->GetTensorShape()->GetDims() != 2 && sparseSeqLen_->GetTensorShape()->GetDims() != 3)) {
        KERNEL_LOG_ERROR("sparseSeqLen must be 2D or 3D tensor");
        return false;
    }
    if (sparseSeqLen_->GetTensorShape()->GetDims() > 0 && sparseSeqLen_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("sparseSeqLen data is nullptr");
        return false;
    }
    // OUTPUT tensor check
    if (metadata_ == nullptr || metadata_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("metadata is empty");
        return false;
    }
    return true;
}

bool QuantBlockSparseAttnMetadataCpuKernel::ParamsInit()
{
    if (sparseSeqLen_ == nullptr || sparseSeqLen_->GetTensorShape() == nullptr || sparseSeqLen_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("sparseSeqLen must be a valid tensor");
        return false;
    }
    auto sparseSeqLenShape = sparseSeqLen_->GetTensorShape();
    sparseSeqLenDimNum_ = static_cast<uint32_t>(sparseSeqLenShape->GetDims());
    int64_t sparseSeqLenBatchSize = sparseSeqLenShape->GetDimSize(0);
    if (batchSize_ <= 0) {
        KERNEL_LOG_ERROR("invalid batch size, batchSize=%d", batchSize_);
        return false;
    }
    if (sparseSeqLenBatchSize != static_cast<int64_t>(batchSize_)) {
        KERNEL_LOG_ERROR("sparseSeqLen shape[0] must be batchSize, got shape[0]=%ld, batchSize=%d",
                         sparseSeqLenBatchSize, batchSize_);
        return false;
    }
    mBaseSize_ = static_cast<uint32_t>(std::max(sparseBlockSizeQ_, 1));
    s2BaseSize_ = static_cast<uint32_t>(std::max(sparseBlockSizeK_, 1));
    if (numHeadsKv_ == 0) {
        numHeadsKv_ = numHeadsQ_;
    }
    if (numHeadsKv_ <= 0 || numHeadsQ_ % numHeadsKv_ != 0) {
        KERNEL_LOG_ERROR("numHeadsQ must be divisible by numHeadsKv, numHeadsQ=%d, numHeadsKv=%d",
                         numHeadsQ_, numHeadsKv_);
        return false;
    }
    if (sparseSeqLenDimNum_ == 3U) {
        uint32_t sparseN1Size = static_cast<uint32_t>(sparseSeqLenShape->GetDimSize(1));
        Qbmax_ = static_cast<uint32_t>(sparseSeqLenShape->GetDimSize(2));
        if (sparseN1Size != static_cast<uint32_t>(numHeadsQ_)) {
            KERNEL_LOG_ERROR("sparseSeqLen shape must be [B,N1,Qb], got N1=%u, expected N1=%d",
                             sparseN1Size, numHeadsQ_);
            return false;
        }
    } else if (sparseSeqLenDimNum_ == 2U) {
        Qbmax_ = static_cast<uint32_t>(sparseSeqLenShape->GetDimSize(1));
    } else {
        KERNEL_LOG_ERROR("sparseSeqLen must be 3D [B,N1,Qb] or legacy 2D [B,Qb], but got %u dims",
                         sparseSeqLenDimNum_);
        return false;
    }
    if (numHeadsQ_ <= 0 || aicCoreNum_ <= 0 || Qbmax_ == 0U) {
        KERNEL_LOG_ERROR("invalid split params, batchSize=%d, numHeadsQ=%d, numHeadsKv=%d, aicCoreNum=%d, Qbmax=%u",
                         batchSize_, numHeadsQ_, numHeadsKv_, aicCoreNum_, Qbmax_);
        return false;
    }
    return true;
}

uint32_t QuantBlockSparseAttnMetadataCpuKernel::GetRowBlockNum(uint32_t bIdx, uint32_t n1Idx, uint32_t s1Idx) const
{
    if (bIdx >= static_cast<uint32_t>(batchSize_) || n1Idx >= static_cast<uint32_t>(numHeadsQ_) ||
        s1Idx >= Qbmax_ || sparseSeqLen_ == nullptr || sparseSeqLen_->GetData() == nullptr) {
        return 0U;
    }
    const int32_t *sparseSeqLenData = static_cast<const int32_t *>(sparseSeqLen_->GetData());
    uint64_t idx = 0U;
    if (sparseSeqLenDimNum_ == 3U) {
        idx = (static_cast<uint64_t>(bIdx) * static_cast<uint32_t>(numHeadsQ_) + n1Idx) * Qbmax_ + s1Idx;
    } else {
        idx = static_cast<uint64_t>(bIdx) * Qbmax_ + s1Idx;
    }
    return sparseSeqLenData[idx] > 0 ? static_cast<uint32_t>(sparseSeqLenData[idx]) : 0U;
}

uint64_t QuantBlockSparseAttnMetadataCpuKernel::GetBN1BlockNum(uint32_t bIdx, uint32_t n1Idx) const
{
    uint64_t bN1BlockNum = 0U;
    for (uint32_t s1Idx = 0U; s1Idx < Qbmax_; ++s1Idx) {
        bN1BlockNum += GetRowBlockNum(bIdx, n1Idx, s1Idx);
    }
    return bN1BlockNum;
}

uint64_t QuantBlockSparseAttnMetadataCpuKernel::CalcBlockNum() const
{
    uint64_t blockNum = 0U;
    for (uint32_t bIdx = 0U; bIdx < static_cast<uint32_t>(batchSize_); ++bIdx) {
        for (uint32_t n1Idx = 0U; n1Idx < static_cast<uint32_t>(numHeadsQ_); ++n1Idx) {
            blockNum += GetBN1BlockNum(bIdx, n1Idx);
        }
    }
    return blockNum;
}

bool QuantBlockSparseAttnMetadataCpuKernel::AdvanceToValidRow(AssignContext &assignContext) const
{
    if (assignContext.isFinished) {
        return false;
    }

    uint32_t totalBN1 = static_cast<uint32_t>(batchSize_) * static_cast<uint32_t>(numHeadsQ_);
    while (assignContext.curBN1Idx < totalBN1) {
        assignContext.curBIdx = assignContext.curBN1Idx / static_cast<uint32_t>(numHeadsQ_);
        assignContext.curN1Idx = assignContext.curBN1Idx % static_cast<uint32_t>(numHeadsQ_);
        if (assignContext.curS1Idx == 0U && assignContext.bN1Block == 0U) {
            assignContext.bN1Block = GetBN1BlockNum(assignContext.curBIdx, assignContext.curN1Idx);
        }
        while (assignContext.curS1Idx < Qbmax_ &&
               GetRowBlockNum(assignContext.curBIdx, assignContext.curN1Idx, assignContext.curS1Idx) == 0U) {
            assignContext.curS1Idx++;
        }
        if (assignContext.curS1Idx < Qbmax_) {
            return true;
        }
        assignContext.curBN1Idx++;
        assignContext.curN1Idx = 0U;
        assignContext.curS1Idx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.bN1Block = 0U;
    }
    assignContext.isFinished = true;
    return false;
}

void QuantBlockSparseAttnMetadataCpuKernel::AssignByBatch(AssignContext &assignContext) const
{
    if (assignContext.isFinished) {
        return;
    }

    const uint32_t totalBN1 = static_cast<uint32_t>(batchSize_) * static_cast<uint32_t>(numHeadsQ_);
    while (assignContext.curBN1Idx < totalBN1) {
        if (!AdvanceToValidRow(assignContext)) {
            return;
        }
        if (assignContext.bN1Block == 0U) {
            assignContext.curBN1Idx++;
            assignContext.curN1Idx = 0U;
            assignContext.curS1Idx = 0U;
            assignContext.curS2Idx = 0U;
            continue;
        }
        if (assignContext.coreCache.block + assignContext.bN1Block > assignContext.coreCache.blockLimit) {
            return;
        }

        assignContext.coreCache.block += assignContext.bN1Block;
        assignContext.curBN1Idx++;
        assignContext.curN1Idx = 0U;
        assignContext.curS1Idx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.bN1Block = 0U;
    }

    assignContext.isFinished = true;
}

void QuantBlockSparseAttnMetadataCpuKernel::AssignByRow(AssignContext &assignContext) const
{
    while (!assignContext.isFinished) {
        if (!AdvanceToValidRow(assignContext)) {
            break;
        }
        uint32_t rowBlock =
            GetRowBlockNum(assignContext.curBIdx, assignContext.curN1Idx, assignContext.curS1Idx);
        if (assignContext.coreCache.block > 0U &&
            assignContext.coreCache.block + rowBlock > assignContext.coreCache.blockLimit) {
            return;
        }
        assignContext.coreCache.block += rowBlock;
        assignContext.bN1Block = assignContext.bN1Block > rowBlock ? assignContext.bN1Block - rowBlock : 0U;
        assignContext.curS1Idx++;
    }
}

void QuantBlockSparseAttnMetadataCpuKernel::ScheduleFa(uint64_t blockNum, SplitResult &result)
{
    AssignContext assignContext {};
    assignContext.unassignedBlock = blockNum;
    assignContext.curBIdx = 0U;
    assignContext.curN1Idx = 0U;
    assignContext.curBN1Idx = 0U;
    assignContext.curS1Idx = 0U;
    assignContext.curS2Idx = 0U;
    assignContext.bN1Block = 0U;

    for (uint32_t i = 0; i < static_cast<uint32_t>(aicCoreNum_); i++) {
        if (assignContext.isFinished || assignContext.unassignedBlock == 0U) {
            break;
        }
        if (!AdvanceToValidRow(assignContext)) {
            break;
        }
        assignContext.curCoreIdx = i;

        assignContext.coreCache = {};
        uint32_t remainingCore = static_cast<uint32_t>(aicCoreNum_) - i;
        assignContext.coreCache.blockLimit =
            (assignContext.unassignedBlock + remainingCore - 1U) / remainingCore;
        uint32_t curRowBlock =
            GetRowBlockNum(assignContext.curBIdx, assignContext.curN1Idx, assignContext.curS1Idx);
        assignContext.coreCache.blockLimit =
            std::max(assignContext.coreCache.blockLimit, static_cast<uint64_t>(curRowBlock));

        AssignByBatch(assignContext);
        AssignByRow(assignContext);

        result.bN1End[i] = assignContext.curBN1Idx;
        result.s1End[i] = assignContext.curS1Idx;
        result.s2End[i] = 0U;
        result.firstFdDataWorkspaceIdx[i] = 0U;
        result.maxBlock = std::max(result.maxBlock, assignContext.coreCache.block);
        if (assignContext.unassignedBlock > assignContext.coreCache.block) {
            assignContext.unassignedBlock -= assignContext.coreCache.block;
        } else {
            assignContext.unassignedBlock = 0U;
        }
        result.usedCoreNum = i + 1U;
    }
    result.usedCoreNum = std::max(result.usedCoreNum, 1U);
}

bool QuantBlockSparseAttnMetadataCpuKernel::ScheduleSection(SplitResult &splitRes)
{
    uint64_t blockNum = CalcBlockNum();
    if (blockNum == 0U) {
        splitRes.usedCoreNum = 1U;
        splitRes.bN1End[0] = static_cast<uint32_t>(batchSize_) * static_cast<uint32_t>(numHeadsQ_);
        splitRes.s1End[0] = 0U;
        splitRes.s2End[0] = 0U;
        splitRes.firstFdDataWorkspaceIdx[0] = 0U;
        return true;
    }
    ScheduleFa(blockNum, splitRes);
    return true;
}

bool QuantBlockSparseAttnMetadataCpuKernel::GenMetadata(const SplitResult &splitRes)
{
    if (metadata_ == nullptr || metadata_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("metadata is empty");
        return false;
    }
    optiling::detail::qbsaMetaData qbsaMetadata(metadata_->GetData());
    for (uint32_t i = 0; i < optiling::AIC_CORE_NUM; i++) {
        qbsaMetadata.setQbsaMetadata(i, optiling::QBSA_CORE_ENABLE_INDEX, 0U);
        qbsaMetadata.setQbsaMetadata(i, optiling::QBSA_BN1_START_INDEX, 0U);
        qbsaMetadata.setQbsaMetadata(i, optiling::QBSA_M_START_INDEX, 0U);
        qbsaMetadata.setQbsaMetadata(i, optiling::QBSA_S2_START_INDEX, 0U);
        qbsaMetadata.setQbsaMetadata(i, optiling::QBSA_BN1_END_INDEX, 0U);
        qbsaMetadata.setQbsaMetadata(i, optiling::QBSA_M_END_INDEX, 0U);
        qbsaMetadata.setQbsaMetadata(i, optiling::QBSA_S2_END_INDEX, 0U);
        qbsaMetadata.setQbsaMetadata(i, optiling::QBSA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, 0U);
    }
    for (uint32_t i = 0; i < optiling::AIV_CORE_NUM; i++) {
        for (uint32_t j = 0; j < optiling::FD_METADATA_SIZE; j++) {
            qbsaMetadata.fdMetadata[i * optiling::FD_METADATA_SIZE + j] = 0U;
        }
    }
    for (uint32_t i = 0; i < splitRes.usedCoreNum; i++) {
        qbsaMetadata.setQbsaMetadata(i, optiling::QBSA_CORE_ENABLE_INDEX, 1U);
        uint32_t bn1Start = (i == 0) ? 0U : splitRes.bN1End[i - 1];
        uint32_t bn1End = splitRes.bN1End[i];
        if (i > 0) {
            qbsaMetadata.setQbsaMetadata(i, optiling::QBSA_M_START_INDEX, splitRes.s1End[i - 1]);
            qbsaMetadata.setQbsaMetadata(i, optiling::QBSA_S2_START_INDEX, splitRes.s2End[i - 1]);
        }
        qbsaMetadata.setQbsaMetadata(i, optiling::QBSA_BN1_START_INDEX, bn1Start);
        qbsaMetadata.setQbsaMetadata(i, optiling::QBSA_BN1_END_INDEX, bn1End);
        qbsaMetadata.setQbsaMetadata(i, optiling::QBSA_M_END_INDEX, splitRes.s1End[i]);
        qbsaMetadata.setQbsaMetadata(i, optiling::QBSA_S2_END_INDEX, splitRes.s2End[i]);
        qbsaMetadata.setQbsaMetadata(i, optiling::QBSA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
                                 splitRes.firstFdDataWorkspaceIdx[i]);
    }
    return true;
}

namespace {
static const char *kernelType = "QuantBlockSparseAttnMetadata";
REGISTER_CPU_KERNEL(kernelType, QuantBlockSparseAttnMetadataCpuKernel);
} // namespace

} // namespace aicpu
