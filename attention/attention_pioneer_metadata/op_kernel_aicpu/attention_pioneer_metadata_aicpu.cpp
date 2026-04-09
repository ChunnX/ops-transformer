/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cmath>
#include "../../attention_pioneer/op_host/common/attention_pioneer_metadata.h"
#include "cpu_context_util.h"
#include "attention_pioneer_metadata_aicpu.h"

using namespace optiling;

namespace aicpu {
namespace {
    const char* const kAttentionPioneerMetadata = "AttentionPioneerMetadata";
} // namespace

uint32_t AttentionPioneerMetadataKernel::Compute(CpuKernelContext &ctx) {
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    CheckIsMla();
    cvRatio_ = aivCoreNum_ / aicCoreNum_;
    if (s2Size == 0) {
        s2Size = 1024;
    }
    if (isMlaSink) {
        CalcInnerSizeMla(static_cast<uint32_t>(s2Size));
    } else {
        CalcInnerSizeGqa(static_cast<uint32_t>(s2Size));
    }
    ComputeSplitNBSeq();
    if (splitUsedCoreNum_ > aicCoreNum_) {
        KERNEL_LOG_ERROR("used_core_num %u exceeds aic_num %u", splitUsedCoreNum_, aicCoreNum_);
    }
    success = GenMetaData();
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool AttentionPioneerMetadataKernel::GenMetaData() {
    optiling::detail::APMetaData* metaDataPtr =
        reinterpret_cast<optiling::detail::APMetaData*>(metaData_->GetData());
    // Per-core split arrays
    for (uint32_t i = 0; i < AP_SPLIT_ARRAY_LEN; ++i) {
        metaDataPtr->bnStartIdx[i] = splitBnStartIdx_[i];
        metaDataPtr->gS1StartIdx[i] = splitGS1StartIdx_[i];
    }
    // Base metadata
    metaDataPtr->baseMetadata[AP_USED_CORE_NUM_INDEX] = static_cast<int32_t>(splitUsedCoreNum_);
    metaDataPtr->baseMetadata[AP_M_BASE_SIZE_INDEX] = static_cast<int32_t>(mBaseSize_);
    metaDataPtr->baseMetadata[AP_S_INNER_SIZE_INDEX] = static_cast<int32_t>(sInnerSize_);
    metaDataPtr->baseMetadata[AP_S1_OUTER_SIZE_INDEX] = static_cast<int32_t>(splitS1OuterSize_);
    metaDataPtr->baseMetadata[AP_S1_SIZE_INDEX] = static_cast<int32_t>(s1Size);
    metaDataPtr->baseMetadata[AP_S2_SIZE_INDEX] = static_cast<int32_t>(s2Size);
    metaDataPtr->baseMetadata[AP_ACTUAL_LEN_Q_DIM_INDEX] = static_cast<int32_t>(actualLenQDims);
    metaDataPtr->baseMetadata[AP_ACTUAL_LEN_KV_DIM_INDEX] = static_cast<int32_t>(actualLenKvDims);
    return true;
}

void AttentionPioneerMetadataKernel::CalcInnerSizeMla(uint32_t seqSize)
{
    constexpr uint32_t MLA_D_SIDE_QK_HEAD_DIM = 512U;
    constexpr uint32_t MLA_D_SIDE_ROPE_HEAD_DIM = 64U;
    constexpr uint32_t MLA_D_SIDE_S_OUTER_SIZE = AP_M_BASE_SIZE_64;
    constexpr uint32_t MLA_D_SIDE_S_INNER_SIZE = 128U;
    constexpr uint32_t MLA_P_SIDE_S_OUTER_SIZE = AP_M_BASE_SIZE_128;
    constexpr uint32_t MLA_P_SIDE_S_INNER_SIZE = 128U;

    bool isDSideConfig = (headDimQK_ == MLA_D_SIDE_QK_HEAD_DIM) &&
        (headDimV_ == MLA_D_SIDE_QK_HEAD_DIM) &&
        (ropeHeadDim_ == MLA_D_SIDE_ROPE_HEAD_DIM);
    sInnerSize_ = isDSideConfig ? MLA_D_SIDE_S_INNER_SIZE : MLA_P_SIDE_S_INNER_SIZE;
    mBaseSize_ = isDSideConfig ? MLA_D_SIDE_S_OUTER_SIZE : MLA_P_SIDE_S_OUTER_SIZE;

    if (seqSize < sInnerSize_) {
        sInnerSize_ = seqSize;
    }
    sInnerLoopTimes_ = (seqSize + sInnerSize_ - 1U) / sInnerSize_;
    sInnerSizeTail_ = seqSize - (sInnerLoopTimes_ - 1U) * sInnerSize_;
    sInnerSizeAlign_ = APAlign(sInnerSize_, AP_BYTE_BLOCK);
    softmaxWithBrcbFlag_ = (mBaseSize_ <= AP_M_BASE_SIZE_128);
}

void AttentionPioneerMetadataKernel::CalcMBaseSizeMla()
{
    uint32_t bN2 = bSize * n2Size;
    uint32_t s1G = s1Size * gSize;
    if (bN2 == 0U || aicCoreNum_ == 0U) {
        return;
    }
    if (bN2 % aicCoreNum_ == 0U || !DealSameSeqEachBatch()) {
        return;
    }

    uint32_t originalA = bN2;
    uint32_t originalB = aicCoreNum_;
    while (originalB != 0U) {
        originalA %= originalB;
        std::swap(originalA, originalB);
    }

    uint32_t lcm = (bN2 / originalA) * aicCoreNum_;
    uint32_t splitOfS1G = lcm / bN2;
    if (splitOfS1G == 0U || s1G % splitOfS1G != 0U) {
        return;
    }
    uint32_t mBaseSizeTmp = s1G / splitOfS1G;
    if (mBaseSizeTmp > mBaseSize_ || mBaseSizeTmp < 128U) {
        return;
    }
    mBaseSize_ = mBaseSizeTmp;
}


void AttentionPioneerMetadataKernel::CalcInnerSizeGqa(uint32_t s2SizeVal)
{
    if (IsQLayoutTNDOrNTD()) {
        sInnerSize_ = AP_S_INNER_SIZE_512;
        if (kSinkNum_ > 0) {
            sInnerSize_ = 256;
        }
    } else {
        if (s1Size <= AP_S1_SIZE_16) {
            sInnerSize_ = AP_MAX_SPLIT_SIZE;
            uint32_t sInnerSizeArr[3U] = {8192U, 4096U, 2048U};
            uint32_t idx = std::min(gSize / 5U, 2U);
            sInnerSize_ = sInnerSizeArr[idx];
        } else {
            sInnerSize_ = AP_S_INNER_SIZE_512;
        }
    }
    if (attenMaskFlag && (sparseMode_ == static_cast<uint32_t>(APSparseMode::LEFT_UP_CAUSAL) ||
            sparseMode_ == static_cast<uint32_t>(APSparseMode::RIGHT_DOWN_CAUSAL) ||
            sparseMode_ == static_cast<uint32_t>(APSparseMode::BAND))) {
        sInnerSize_ = std::min(sInnerSize_, AP_S_INNER_SIZE_1024);
    }

    if (blockSize_ != 0) {
        uint32_t blockSize = static_cast<uint32_t>(blockSize_);
        if ((sInnerSize_ > blockSize) && (sInnerSize_ % blockSize != 0)) {
            sInnerSize_ = (sInnerSize_ / blockSize) * blockSize;
        }
    }
    sInnerLoopTimes_ = (s2SizeVal + sInnerSize_ - static_cast<uint32_t>(1)) / sInnerSize_;
    sInnerSizeTail_ = s2SizeVal - (sInnerLoopTimes_ - static_cast<uint32_t>(1)) * sInnerSize_;
    if (sInnerSize_ > s2SizeVal) {
        sInnerSize_ = s2SizeVal;
    }
    sInnerSizeAlign_ = APAlign(sInnerSize_, AP_BYTE_BLOCK);
    CalcMBaseSizeGqa();
}

void AttentionPioneerMetadataKernel::CalcMBaseSizeGqa()
{
    if (IsQLayoutTNDOrNTD()) {
        mBaseSize_ = AP_M_BASE_SIZE_512;
    } else {
        if (s1Size <= AP_S1_SIZE_16) {
            if (sInnerSizeAlign_ <= AP_S_INNER_SIZE_ALIGN_512) {
                mBaseSize_ = AP_M_BASE_SIZE_512;
            } else if (sInnerSizeAlign_ <= AP_S_INNER_SIZE_ALIGN_1024) {
                mBaseSize_ = AP_M_BASE_SIZE_256;
            } else if (sInnerSizeAlign_ <= AP_S_INNER_SIZE_ALIGN_2048) {
                mBaseSize_ = AP_M_BASE_SIZE_128;
            } else if (sInnerSizeAlign_ <= AP_S_INNER_SIZE_ALIGN_4096) {
                mBaseSize_ = AP_M_BASE_SIZE_64;
            } else {
                mBaseSize_ = AP_M_BASE_SIZE_32;
            }
        } else {
            mBaseSize_ = AP_M_BASE_SIZE_512;
        }
    }
    softmaxWithBrcbFlag_ = (mBaseSize_ <= AP_M_BASE_SIZE_128);
}

bool AttentionPioneerMetadataKernel::DealSameSeqEachBatch()
{
    batchContinuousFlag = blockSize_ == 0 ? true : false;
    if (batchContinuousFlag) {
        return actualSeqLenFlag ? isSameActualseq : isSameSeqAllKVTensor;
    }
    return isSameActualseq;
}

bool AttentionPioneerMetadataKernel::Prepare(CpuKernelContext &ctx) {
    actualSeqLengths_ = ctx.Input(static_cast<uint32_t>(ParamId::actualSeqLengthsOptional));
    actualSeqLengthsKv_ = ctx.Input(static_cast<uint32_t>(ParamId::actualSeqLengthsKvOptional));
    metaData_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    bool requiredAttrs = GetAttrValue(ctx, "num_heads_q", numHeadsQ_) &&
                        GetAttrValue(ctx, "num_heads_kv", numHeadsKv_) &&
                        GetAttrValue(ctx, "head_dim_qk", headDimQK_) &&
                        GetAttrValue(ctx, "head_dim_v", headDimV_);
                        GetAttrValueOpt(ctx, "soc_version", socVersion_);
                        GetAttrValueOpt(ctx, "aic_core_num", aicCoreNum_);
                        GetAttrValueOpt(ctx, "aiv_core_num", aivCoreNum_);
    if (!requiredAttrs) {
        return false;
    }

    GetAttrValueOpt(ctx, "rope_head_dim", ropeHeadDim_);
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "sparse_mode", sparseMode_);
    GetAttrValueOpt(ctx, "pre_tokens", preTokens_);
    GetAttrValueOpt(ctx, "next_tokens", nextTokens_);
    GetAttrValueOpt(ctx, "input_layout", inputLayout_);
    GetAttrValueOpt(ctx, "k_sink_num", kSinkNum_);

    GetAttrValueOpt(ctx, "block_size", blockSize_);

    return (ParamsParse() && ParamsCheck());
}

bool AttentionPioneerMetadataKernel::ParamsParse() {
    return (CheckSingleParam() && ParseAxisInfo());
}

bool AttentionPioneerMetadataKernel::CheckSingleParam() {
    KERNEL_CHECK_NULLPTR(metaData_, false, "metadata is null");
    auto metaShape = metaData_->GetTensorShape();
    KERNEL_CHECK_NULLPTR(metaShape, false, "shape of metadata is null");
    KERNEL_CHECK_NULLPTR(metaData_->GetData(), false, "data of metadata is null");
    if (aicCoreNum_ == 0 || aivCoreNum_ == 0 || (aivCoreNum_ % aicCoreNum_ != 0)) {
        KERNEL_LOG_ERROR("Core num invalid: aic:%u, aiv:%u", aicCoreNum_, aivCoreNum_);
        return false;
    }
    if (batchSize_ < 0) {
        KERNEL_LOG_ERROR("batchSize is %d, it should be greater than 0.", batchSize_);
        return false;
    }
    constexpr int32_t BLOCK_SIZE_ALIGN_SIZE = 16;
    constexpr int32_t BLOCK_SIZE_MAX_SIZE = 1024;
    if (blockSize_ % BLOCK_SIZE_ALIGN_SIZE != 0) {
        KERNEL_LOG_ERROR("blockSize should aligned to 16, but got %d.", blockSize_);
        return false;
    }
    if (blockSize_ > BLOCK_SIZE_MAX_SIZE) {
        KERNEL_LOG_ERROR("blockSize should less equal than 1024, but got %d.", blockSize_);
        return false;
    }
    if ((sparseMode_ != static_cast<uint32_t>(APSparseMode::DEFAULT_MASK)) &&
        (sparseMode_ != static_cast<uint32_t>(APSparseMode::ALL_MASK)) &&
        (sparseMode_ != static_cast<uint32_t>(APSparseMode::LEFT_UP_CAUSAL)) &&
        (sparseMode_ != static_cast<uint32_t>(APSparseMode::RIGHT_DOWN_CAUSAL)) &&
        (sparseMode_ != static_cast<uint32_t>(APSparseMode::BAND))) {
        KERNEL_LOG_ERROR("sparseMode only supports 0/1/2/3/4, but got %u.", sparseMode_);
        return false;
    }
    if (headDimQK_ <= 0) {
        KERNEL_LOG_ERROR("qk head dim is %d, it should be greater than 0.", headDimQK_);
        return false;
    }
    if (headDimV_ <= 0) {
        KERNEL_LOG_ERROR("value head dim is %d, it should be greater than 0.", headDimV_);
        return false;
    }
    return CheckSingleParaInputLayout();
}

bool AttentionPioneerMetadataKernel::CheckSingleParaInputLayout() {
    const std::vector<std::string> inputLayoutList = {
        "BSH", "BSND", "BNSD", "TND", "NTD",
        "BSH_NBSD", "BSND_NBSD", "BNSD_NBSD", "TND_NTD", "NTD_TND",
        "BSH_BNSD", "BSND_BNSD", "BNSD_BSND"
    };
    if (std::find(inputLayoutList.begin(), inputLayoutList.end(), inputLayout_) == inputLayoutList.end()) {
        KERNEL_LOG_ERROR("input layout not supported: %s.", inputLayout_.c_str());
        return false;
    }
    if (IsQLayoutTNDOrNTD()) {
        KERNEL_CHECK_NULLPTR(actualSeqLengths_, false,
            "When inputLayout is TND or NTD, actualSeqQLengths should not be null.");
        KERNEL_CHECK_NULLPTR(actualSeqLengthsKv_, false,
            "When inputLayout is TND or NTD, actualSeqKVLengths should not be null.");
    }
    return true;
}

bool AttentionPioneerMetadataKernel::ParseAxisInfo() {
    return (CheckSingleParaNumHeads() && GetBatchSize() && GetS1Size() && GetS2Size() && GetPreNextToken() && GetActualSeqInfo());
}

bool AttentionPioneerMetadataKernel::CheckSingleParaNumHeads() {
    if (numHeadsQ_ <= 0) {
        KERNEL_LOG_ERROR("the query's heads num(%u) should be greater than 0", numHeadsQ_);
        return false;
    }
    if (numHeadsKv_ <= 0) {
        KERNEL_LOG_ERROR("the kv's heads num(%u) should be greater than 0", numHeadsKv_);
        return false;
    }
    n1Size = static_cast<uint32_t>(numHeadsQ_);
    n2Size = (numHeadsKv_ == 0) ? n1Size : static_cast<uint32_t>(numHeadsKv_);
    if (n1Size % n2Size != 0U) {
        KERNEL_LOG_ERROR("the query's heads num(%u) should be a multiple of the key/value's heads num(%u)",
            n1Size, n2Size);
        return false;
    }
    gSize = n1Size / n2Size;
    return true;
}

bool AttentionPioneerMetadataKernel::GetBatchSize()
{
    if (IsQLayoutTNDOrNTD()) {
        return GetActualSeqLenSize(bSize);
    } else {
        bSize = batchSize_;
        return true;
    }
}

bool AttentionPioneerMetadataKernel::GetS1Size()
{
    if (IsQLayoutTNDOrNTD()) {
        const int64_t *actualSeqQ = static_cast<const int64_t*>(actualSeqLengths_->GetData());
        if (actualSeqQ == nullptr) {
            return false;
        }
        uint32_t b = 0;
        if(!GetActualSeqLenSize(b)) {
            return false;
        }
        int64_t qActualSeqMax = 0;
        for (uint32_t i = 0; i < b; i++) {
            int64_t tmpS1 = (i == 0U) ? actualSeqQ[0] : (actualSeqQ[i] - actualSeqQ[i - 1U]);
            if (tmpS1 > qActualSeqMax) {
                qActualSeqMax = tmpS1;
            }
        }
        s1Size = static_cast<uint32_t>(qActualSeqMax);
    }
    return true;
}

bool AttentionPioneerMetadataKernel::GetActualSeqLenSize(uint32_t &size)
{
    int64_t shapeSize = actualSeqLengths_->GetTensorShape()->GetDimSize(0);
    if (shapeSize <= 0) {
        KERNEL_LOG_ERROR("actualSeqLengths's shape size is %ld, it should be greater than 0.", shapeSize);
        return false;
    }
    size = static_cast<uint32_t>(shapeSize);
    return true;
}

bool AttentionPioneerMetadataKernel::GetActualSeqKVLenSize(uint32_t &size)
{
    int64_t shapeSize = actualSeqLengthsKv_->GetTensorShape()->GetDimSize(0);
    if (shapeSize <= 0) {
        KERNEL_LOG_ERROR("actualSeqKVLengths's shape size is %ld, it should be greater than 0.", shapeSize);
        return false;
    }
    size = static_cast<uint32_t>(shapeSize);
    return true;
}

bool AttentionPioneerMetadataKernel::GetS2Size()
{
    const int64_t *actualLenData = static_cast<const int64_t*>(actualSeqLengthsKv_->GetData());
    if (actualLenData == nullptr) {
        return false;
    }
    uint32_t loop = std::min(static_cast<uint32_t>(actualSeqLengthsKv_->GetTensorShape()->GetDimSize(0)), bSize);
    if (blockSize_ == 0 && IsQLayoutTNDOrNTD()) {
        for (uint32_t i = 0; i < loop; i++) {
            int64_t tmpS = (i == 0U) ? actualLenData[0] : (actualLenData[i] - actualLenData[i - 1U]);
            s2Size = std::max(s2Size, tmpS);
        }
        if (kSinkNum_ > 0) {
            s2Size += kSinkNum_;
        }
        return true;
    } else {
        for (uint32_t i = 0; i < loop; i++) {
            s2Size = std::max(s2Size, actualLenData[i]);
        }
        if (kSinkNum_ > 0) {
            s2Size += kSinkNum_;
        }
        return true;
    }
    return true;
}

bool AttentionPioneerMetadataKernel::GetPreNextToken()
{
    static int64_t SPARSE_MODE_INT_MAX = 2147483647;
    if (sparseMode_ != static_cast<uint32_t>(APSparseMode::DEFAULT_MASK)) {
        attenMaskFlag = true;
    }
    if (sparseMode_ == static_cast<uint32_t>(APSparseMode::ALL_MASK)) {
        preToken = SPARSE_MODE_INT_MAX;
        nextToken = SPARSE_MODE_INT_MAX;
    } else if (sparseMode_ == static_cast<uint32_t>(APSparseMode::LEFT_UP_CAUSAL) ||
                sparseMode_ == static_cast<uint32_t>(APSparseMode::RIGHT_DOWN_CAUSAL)) {
        nextToken = 0;
        preToken = SPARSE_MODE_INT_MAX;
    } else {
        preToken = preTokens_ == SPARSE_MODE_INT_MAX ? 0 : preTokens_;
        nextToken = nextTokens_ == SPARSE_MODE_INT_MAX ? 0 : nextTokens_;
    }

    if (preToken > SPARSE_MODE_INT_MAX) {
        preToken = SPARSE_MODE_INT_MAX;
    } else if (preToken < -(SPARSE_MODE_INT_MAX)) {
        preToken = -(SPARSE_MODE_INT_MAX);
    }
    if (nextToken > SPARSE_MODE_INT_MAX) {
        nextToken = SPARSE_MODE_INT_MAX;
    } else if (nextToken < -(SPARSE_MODE_INT_MAX)) {
        nextToken = -(SPARSE_MODE_INT_MAX);
    }

    if (preToken < 0 && nextToken < 0) {
        KERNEL_LOG_ERROR("preTokens(%ld) and nextTokens(%ld) cannot neither be negative number.", preToken, nextToken);
        return false;
    }
    if (nextToken * (-1) > preToken) {
        KERNEL_LOG_ERROR("nextToken line(%ld) should be higher than preToken line(%ld).", preToken, nextToken);
        return false;
    }
    return true;
}

bool AttentionPioneerMetadataKernel::GetActualSeqInfo()
{
    maxActualseq = s2Size;
    if (actualSeqLengthsKv_ != nullptr && actualSeqLengthsKv_->GetData() != nullptr) {
        const int64_t *actualLenData = static_cast<const int64_t*>(actualSeqLengthsKv_->GetData());
        actualLenKvDims = actualSeqLengthsKv_->GetTensorShape()->GetDimSize(0);
        if (IsQLayoutTNDOrNTD()) {
            isAccumQSeq = true;
        }
        uint32_t loop = ((actualLenKvDims == 1) && (kvListSeqLens.size() == 1)) ? 1 : bSize;
        loop = std::min(loop, actualLenKvDims);
        for (uint32_t i = 0; i < loop; i++) {
            int64_t actLen = actualLenData[i];
            needInit = (actLen == 0);
            maxActualseq = maxActualseq < actLen ? actLen : maxActualseq;
            if (actualLenData[i] != actualLenData[0]) {
                isSameActualseq = false;
                break;
            }
        }
    }

    if (actualSeqLengths_ != nullptr && actualSeqLengths_->GetData() != nullptr) {
        const int64_t *actualLenQData = static_cast<const int64_t*>(actualSeqLengths_->GetData());
        actualLenQDims = actualSeqLengths_->GetTensorShape()->GetDimSize(0);
        if (blockSize_ == 0 && IsQLayoutTNDOrNTD()) {
            isAccumKVSeq = true;
        }
        uint32_t loop = std::min(bSize, actualLenQDims);
        for (uint32_t i = 0; i < loop; i++) {
            if (actualLenQData[i] != static_cast<int64_t>(s1Size)) {
                needInit = true;
                break;
            }
        }
    }
    return true;
}

bool AttentionPioneerMetadataKernel::ParamsCheck()
{
    return (CheckFeature() && CheckMultiParaConsistency());
}

void AttentionPioneerMetadataKernel::CheckIsMla()
{
    if (ropeHeadDim_ == 0U) {
        isMlaSink = false;
        return;
    }
    constexpr uint32_t QK_HEAD_DIM_512 = 512U;
    constexpr uint32_t ROPE_HEAD_DIM_64 = 64U;
    if ((headDimQK_ != headDimV_) || (headDimQK_ != QK_HEAD_DIM_512) || (ropeHeadDim_ != ROPE_HEAD_DIM_64)) {
        isMlaSink = false;
        return;
    }
    if (n2Size != 1U) {
        isMlaSink = false;
        return;
    }
    const std::vector<std::string> layoutSupportList = {
        "BSH", "BSND", "BNSD", "TND", "TND_NTD"
    };
    if (std::find(layoutSupportList.begin(), layoutSupportList.end(), inputLayout_) == layoutSupportList.end()) {
        isMlaSink = false;
        return;
    }
    if ((sparseMode_ != static_cast<uint32_t>(APSparseMode::DEFAULT_MASK)) &&
        (sparseMode_ != static_cast<uint32_t>(APSparseMode::RIGHT_DOWN_CAUSAL)) &&
        (sparseMode_ != static_cast<uint32_t>(APSparseMode::BAND))) {
        isMlaSink = false;
        return;
    }
    std::vector<uint32_t> gSizeSupportList = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128};
    if (std::find(gSizeSupportList.begin(), gSizeSupportList.end(), gSize) == gSizeSupportList.end()) {
        isMlaSink = false;
        return;
    }
    isMlaSink = true;
}

bool AttentionPioneerMetadataKernel::CheckFeature()
{
    return (CheckFeatureActualSeqLensQData() && CheckFeatureActualSeqLensKvData() && CheckFeatureAxisInfo());
}

bool AttentionPioneerMetadataKernel::CheckFeatureAxisInfo() const
{
    constexpr uint32_t MAX_ACTUAL_SEQ_LEN_BYTE = 64U * 1024U;
    if (actualSeqLengthsQSize_ > MAX_ACTUAL_SEQ_LEN_BYTE) {
        KERNEL_LOG_ERROR("actual sequence length q should be smaller or equal to 64K, but got %u", actualSeqLengthsQSize_);
        return false;
    }
    if (actualSeqLengthsKvSize_ > MAX_ACTUAL_SEQ_LEN_BYTE) {
        KERNEL_LOG_ERROR("actual sequence length kv should be smaller or equal to 64K, but got %u", actualSeqLengthsKvSize_);
        return false;
    }
    return true;
}

bool AttentionPioneerMetadataKernel::CheckFeatureActualSeqLensQData()
{
    if (actualSeqLengths_ == nullptr) {
        qSize.push_back(s1Size);
        return true;
    }
    const int64_t *actualSeq = static_cast<const int64_t*>(actualSeqLengths_->GetData());
    if (actualSeq == nullptr) {
        qSize.push_back(s1Size);
        return true;
    }
    if (!GetActualSeqLenSize(actualSeqLengthsQSize_)) {
        return false;
    }
    uint32_t loop = std::min(actualSeqLengthsQSize_, bSize);
    for (uint32_t i = 0; i < loop; i++) {
        int64_t tmpS1 = 0;
        if (IsQLayoutTNDOrNTD()) {
            if (actualSeq[i] < 0) {
                KERNEL_LOG_ERROR("When q's Layout is TND or NTD, actualSeqQLengths[%u] should not be negative, but got %ld.", i, actualSeq[i]);
                return false;
            }
            if (i > 0U && actualSeq[i] < actualSeq[i - 1U]) {
                KERNEL_LOG_ERROR("When q's Layout is TND or NTD, actualSeqQLengths[%u](%ld) < actualSeqQLengths[%u](%ld).",
                    i, actualSeq[i], (i - 1U), actualSeq[i - 1U]);
                return false;
            }
            tmpS1 = (i == 0U) ? actualSeq[0] : (actualSeq[i] - actualSeq[i - 1U]);
        } else {
            tmpS1 = actualSeq[i];
        }
        if (tmpS1 > static_cast<int64_t>(s1Size) || tmpS1 < 0) {
            KERNEL_LOG_ERROR("actualSeqQLengths[%u] computed is %ld, should be in [0, Q_S(%u)].", i, tmpS1, s1Size);
            return false;
        }
        qSize.push_back(tmpS1);
    }
    return true;
}

bool AttentionPioneerMetadataKernel::CheckFeatureActualSeqLensKvData()
{
    if (actualSeqLengthsKv_ == nullptr) {
        kvSize.push_back(s2Size);
        return true;
    }
    actualSeqLenFlag = true;
    const int64_t *actualSeq = static_cast<const int64_t*>(actualSeqLengthsKv_->GetData());
    if (actualSeq == nullptr) {
        kvSize.push_back(s2Size);
        return true;
    }
    if(!GetActualSeqKVLenSize(actualSeqLengthsKvSize_)) {
        return false;
    }
    uint32_t loop = std::min(actualSeqLengthsKvSize_, bSize);
    for (uint32_t i = 0; i < loop; i++) {
        int64_t tmpS2 = 0;
        if (blockSize_ == 0 && IsQLayoutTNDOrNTD()) {
            if (actualSeq[i] < 0) {
                KERNEL_LOG_ERROR("When kv's Layout is TND or NTD, actualSeqKVLengths[%u] should not be negative, but got %ld.", i, actualSeq[i]);
                return false;
            }
            if (i > 0U && actualSeq[i] < actualSeq[i - 1U]) {
                KERNEL_LOG_ERROR("When kv's Layout is TND or NTD, actualSeqKVLengths[%u](%ld) < actualSeqKVLengths[%u](%ld).",
                    i, actualSeq[i], (i - 1U), actualSeq[i - 1U]);
                return false;
            }
            tmpS2 = (i == 0U) ? actualSeq[0] : (actualSeq[i] - actualSeq[i - 1U]);
        } else {
            tmpS2 = actualSeq[i];
        }
        if (tmpS2 < 0 || tmpS2 > s2Size) {
            KERNEL_LOG_ERROR("actualSeqKVLengths[%u] computed is %ld, should be in [0, KV_S(%ld)].", i, tmpS2, s2Size);
            return false;
        }
        kvSize.push_back(tmpS2);
    }
    return true;
}

bool AttentionPioneerMetadataKernel::CheckMultiParaConsistency()
{
    return (CheckActualSeqLensQ() && CheckActualSeqLensKv());
}

bool AttentionPioneerMetadataKernel::CheckActualSeqLensQ() const
{
    if ((actualSeqLengths_ == nullptr) || (actualSeqLengths_->GetData() == nullptr)) {
        return true;
    }
    if (IsQLayoutTND()) {
        if (actualSeqLengthsQSize_ != bSize && actualSeqLengthsQSize_ != 1U) {
            KERNEL_LOG_ERROR("actual seq q len shape size is %u, should be batch size(%u) or 1.", actualSeqLengthsQSize_, bSize);
            return false;
        }
    } else {
        if (actualSeqLengthsQSize_ < bSize && actualSeqLengthsQSize_ != 1U) {
            KERNEL_LOG_ERROR("actual seq q len shape size is %u, should be >= batch size(%u) or 1.", actualSeqLengthsQSize_, bSize);
            return false;
        }
    }
    return true;
}

bool AttentionPioneerMetadataKernel::CheckActualSeqLensKv() const
{
    if ((actualSeqLengthsKv_ == nullptr) || (actualSeqLengthsKv_->GetData() == nullptr)) {
        return true;
    }
    if (IsQLayoutTND()) {
        if (actualSeqLengths_ != nullptr && actualSeqLengths_->GetData() != nullptr &&
            actualSeqLengthsKvSize_ != actualSeqLengthsQSize_) {
            KERNEL_LOG_ERROR("actual seq kv len shape size(%u) should equal actual seq q len shape size(%u).",
                actualSeqLengthsKvSize_, actualSeqLengthsQSize_);
            return false;
        }
        if (actualSeqLengthsKvSize_ != bSize && actualSeqLengthsKvSize_ != 1U) {
            KERNEL_LOG_ERROR("actual seq kv len shape size is %u, should be batch size(%u) or 1.", actualSeqLengthsKvSize_, bSize);
            return false;
        }
    } else {
        if (actualSeqLengthsKvSize_ < bSize && actualSeqLengthsKvSize_ != 1U) {
            KERNEL_LOG_ERROR("actual seq kv len shape size is %u, should be >= batch size(%u) or 1.", actualSeqLengthsKvSize_, bSize);
            return false;
        }
    }
    return true;
}

// ============================= A5 Greedy Split Core Algorithm ===========================

// Get per-batch Q sequence length
uint32_t AttentionPioneerMetadataKernel::GetQSeqSize(uint32_t bIdx) const
{
    if (qSize.empty()) return s1Size;
    if (qSize.size() == 1U) return qSize[0];
    return qSize[bIdx];
}

// Get per-batch KV sequence length
uint32_t AttentionPioneerMetadataKernel::GetKVSeqSize(uint32_t bIdx) const
{
    if (kvSize.empty()) return static_cast<uint32_t>(s2Size);
    if (kvSize.size() == 1U) return kvSize[0];
    return kvSize[bIdx];
}

// Ported from A5: GetPreNextTokensLeftUp
void AttentionPioneerMetadataKernel::GetPreNextTokensLeftUp(
    int64_t actualSeqLength, int64_t actualSeqLengthKV,
    int64_t &preTokensLeftUp, int64_t &nextTokensLeftUp) const
{
    auto mode = static_cast<APSparseMode>(sparseMode_);
    bool isS1G = (IsQLayoutTND() || inputLayout_ == "BSH" || inputLayout_ == "BSND");
    if (mode == APSparseMode::RIGHT_DOWN_CAUSAL) {
        preTokensLeftUp = SPARSE_MODE_INT_MAX;
        if (isMlaSink && isS1G) {
            nextTokensLeftUp = actualSeqLengthKV * gSize - actualSeqLength;
        } else {
            nextTokensLeftUp = actualSeqLengthKV - actualSeqLength;
        }
    } else if (mode == APSparseMode::BAND) {
        if (isMlaSink && isS1G) {
            preTokensLeftUp = preToken * gSize - actualSeqLengthKV * gSize + actualSeqLength;
            nextTokensLeftUp = nextToken * gSize + actualSeqLengthKV * gSize - actualSeqLength;
        } else {
            preTokensLeftUp = preToken - actualSeqLengthKV + actualSeqLength;
            nextTokensLeftUp = nextToken + actualSeqLengthKV - actualSeqLength;
        }
    } else {
        if (isMlaSink && isS1G) {
            preTokensLeftUp = preToken * gSize;
            nextTokensLeftUp = nextToken * gSize;
        } else {
            preTokensLeftUp = preToken;
            nextTokensLeftUp = nextToken;
        }
    }
}

// Ported from A5: FixParamWithRowInvalid
void AttentionPioneerMetadataKernel::FixParamWithRowInvalid(
    int64_t &actualSeqLength, int64_t actualSeqLengthKV,
    int64_t &preTokensLeftUp, int64_t &nextTokensLeftUp) const
{
    int64_t nextTokensError = (nextTokensLeftUp < 0) ? -nextTokensLeftUp : 0;
    nextTokensError = std::min(nextTokensError, actualSeqLength);
    int64_t preTokensError = 0;
    if (isMlaSink) {
        preTokensError = (actualSeqLength > actualSeqLengthKV * gSize + preTokensLeftUp) ?
            (actualSeqLength - actualSeqLengthKV * gSize - preTokensLeftUp) : 0;
    } else {
        preTokensError = (actualSeqLength > actualSeqLengthKV + preTokensLeftUp) ?
            (actualSeqLength - actualSeqLengthKV - preTokensLeftUp) : 0;
    }
    preTokensError = std::min(preTokensError, actualSeqLength);
    nextTokensLeftUp += nextTokensError;
    preTokensLeftUp -= nextTokensError;
    actualSeqLength -= nextTokensError;
    actualSeqLength -= preTokensError;
}

int64_t AttentionPioneerMetadataKernel::ClipSInnerToken(int64_t val, int64_t lo, int64_t hi)
{
    return (val < lo) ? lo : ((val > hi) ? hi : val);
}

int64_t AttentionPioneerMetadataKernel::GetActualInnerBlockNums(
    int64_t sInnerIndexStart, int64_t sInnerIndexEnd, int64_t innerBlockNums) const
{
    if (sInnerIndexEnd < 0) return 0;
    if (sInnerIndexEnd < innerBlockNums) {
        return (sInnerIndexStart < 0) ? (sInnerIndexEnd + 1) : (sInnerIndexEnd - sInnerIndexStart + 1);
    }
    return (sInnerIndexStart < 0) ? innerBlockNums : (innerBlockNums - sInnerIndexStart);
}

// MLA-specific block count for one head
int64_t AttentionPioneerMetadataKernel::GetCalcBlockNumsForS1G(
    int64_t actualSeqLength, int64_t actualSeqLengthKV,
    uint32_t sOuterSize, uint32_t sInnerSize,
    int64_t preTokensLeftUp, int64_t nextTokensLeftUp) const
{
    int64_t s2BlockNum = 0;
    int64_t outerBlockNums = CeilDiv(actualSeqLength, static_cast<int64_t>(sOuterSize));
    for (int64_t i = 0; i < outerBlockNums; i++) {
        int64_t cubeSOuterOffset = i * static_cast<int64_t>(sOuterSize);
        int64_t sInnerFirstToken = ClipSInnerToken(
            (cubeSOuterOffset - preTokensLeftUp) / gSize, 0, actualSeqLengthKV);
        int64_t sInnerLastToken = ClipSInnerToken(
            CeilDiv(cubeSOuterOffset + nextTokensLeftUp + sInnerSize,
                    static_cast<int64_t>(gSize)),
            0, actualSeqLengthKV);
        s2BlockNum += CeilDiv(sInnerLastToken, static_cast<int64_t>(sInnerSize)) -
                      sInnerFirstToken / static_cast<int64_t>(sInnerSize);
    }
    return s2BlockNum;
}

// Total blocks for one head across all sOuter blocks
int64_t AttentionPioneerMetadataKernel::GetCalcBlockNumsOneHead(
    int64_t actualSeqLength, int64_t actualSeqLengthKV,
    uint32_t sOuterSize, uint32_t sInnerSize,
    int64_t preTokensLeftUp, int64_t nextTokensLeftUp) const
{
    int64_t outerBlockNums = CeilDiv(actualSeqLength, static_cast<int64_t>(sOuterSize));
    int64_t innerBlockNums = CeilDiv(actualSeqLengthKV, static_cast<int64_t>(sInnerSize));
    if (!attenMaskFlag) {
        return innerBlockNums * outerBlockNums;
    }
    if (isMlaSink) {
        return GetCalcBlockNumsForS1G(actualSeqLength, actualSeqLengthKV,
            sOuterSize, sInnerSize, preTokensLeftUp, nextTokensLeftUp);
    }
    int64_t totalBlockNums = 0;
    int64_t curPreTokensLeftUp = preTokensLeftUp;
    int64_t curNextTokensLeftUp = nextTokensLeftUp;
    for (int64_t sOuterIndex = 0; sOuterIndex < outerBlockNums; sOuterIndex++) {
        int64_t sInnerIndexStart = -(curPreTokensLeftUp > 0 ?
            CeilDiv(curPreTokensLeftUp, static_cast<int64_t>(sInnerSize)) :
            curPreTokensLeftUp / static_cast<int64_t>(sInnerSize));
        int64_t sInnerIndexEnd = curNextTokensLeftUp > 0 ?
            CeilDiv(curNextTokensLeftUp, static_cast<int64_t>(sInnerSize)) :
            curNextTokensLeftUp / static_cast<int64_t>(sInnerSize);
        totalBlockNums += GetActualInnerBlockNums(sInnerIndexStart, sInnerIndexEnd, innerBlockNums);
        curPreTokensLeftUp -= static_cast<int64_t>(sOuterSize);
        curNextTokensLeftUp += static_cast<int64_t>(sOuterSize);
    }
    return totalBlockNums;
}

int64_t AttentionPioneerMetadataKernel::SinkBlockCount(
    int64_t actualSeqLength, uint32_t sOuterSize, uint32_t sInnerSize) const
{
    int64_t outerBlockNums = CeilDiv(actualSeqLength, static_cast<int64_t>(sOuterSize));
    int64_t sinkBlockPerS1g = CeilDiv(kSinkNum_, static_cast<int64_t>(sInnerSize));
    return sinkBlockPerS1g * outerBlockNums;
}

// Main A5 greedy split-core algorithm
// Ported from: PromptFlashAttentionPioneerTilingV2::ComputeSplitNBSeq + PromptFlashAttentionPioneerSplitNBSeq
void AttentionPioneerMetadataKernel::ComputeSplitNBSeq()
{
    uint32_t sOuterSize = mBaseSize_;
    uint32_t sInnerSize = sInnerSize_;
    uint32_t headNumSize = n2Size;
    uint32_t curCoreNum = aicCoreNum_;

    // Build per-batch actual seq lengths
    std::vector<int64_t> actualSeqLengths(bSize, static_cast<int64_t>(s1Size));
    std::vector<int64_t> actualSeqLengthsKV(bSize, s2Size);
    for (uint32_t i = 0; i < bSize; i++) {
        actualSeqLengths[i] = static_cast<int64_t>(GetQSeqSize(i));
        actualSeqLengthsKV[i] = static_cast<int64_t>(GetKVSeqSize(i));
    }

    // Phase 1: compute totalBlockNumsOneHead (total work across all batches for one head)
    int64_t totalBlockNumsOneHead = 0;
    int64_t sinkBlockNumberPerS1g = 0;
    if (kSinkNum_ > 0) {
        sinkBlockNumberPerS1g = CeilDiv(kSinkNum_, static_cast<int64_t>(sInnerSize));
    }

    for (uint32_t sIdx = 0; sIdx < bSize; sIdx++) {
        int64_t actualSeqLengthsTmp = actualSeqLengths[sIdx];
        int64_t preTokensLeftUp = 0;
        int64_t nextTokensLeftUp = 0;
        GetPreNextTokensLeftUp(actualSeqLengths[sIdx], actualSeqLengthsKV[sIdx],
            preTokensLeftUp, nextTokensLeftUp);
        FixParamWithRowInvalid(actualSeqLengthsTmp, actualSeqLengthsKV[sIdx],
            preTokensLeftUp, nextTokensLeftUp);

        totalBlockNumsOneHead += GetCalcBlockNumsOneHead(actualSeqLengthsTmp, actualSeqLengthsKV[sIdx],
            sOuterSize, sInnerSize, preTokensLeftUp, nextTokensLeftUp);
        if (kSinkNum_ > 0) {
            totalBlockNumsOneHead += SinkBlockCount(actualSeqLengthsTmp, sOuterSize, sInnerSize);
        }
    }

    // coreWightTarget: average work per core
    double coreWightTarget = static_cast<double>(totalBlockNumsOneHead * headNumSize) /
                             static_cast<double>(curCoreNum);

    splitS1OuterSize_ = static_cast<uint32_t>(CeilDiv(static_cast<int64_t>(s1Size),
                                                       static_cast<int64_t>(sOuterSize)));

    // Phase 2: greedy allocation across batch x head x sOuter
    const uint32_t arrayLen = curCoreNum + 1;
    splitBnStartIdx_.assign(arrayLen, 0);
    splitGS1StartIdx_.assign(arrayLen, 0);

    int64_t curWight = 0;
    uint32_t curCore = 0;
    uint32_t tmpCoreNidEnd = 0;
    uint32_t tmpCoreSidEnd = 0;
    uint32_t tmpCoreSposEnd = 0;

    for (uint32_t sIdx = 0; sIdx < bSize; sIdx++) {
        for (uint32_t headNum = 0; headNum < headNumSize; headNum++) {
            int64_t preTokensLeftUp = 0;
            int64_t nextTokensLeftUp = 0;
            GetPreNextTokensLeftUp(actualSeqLengths[sIdx], actualSeqLengthsKV[sIdx],
                preTokensLeftUp, nextTokensLeftUp);
            int64_t actualSeqLength = actualSeqLengths[sIdx];
            int64_t actualSeqLengthKV = actualSeqLengthsKV[sIdx];
            FixParamWithRowInvalid(actualSeqLength, actualSeqLengthKV,
                preTokensLeftUp, nextTokensLeftUp);

            int64_t outerBlockNums = CeilDiv(actualSeqLength, static_cast<int64_t>(sOuterSize));
            int64_t innerBlockNums = CeilDiv(actualSeqLengthKV, static_cast<int64_t>(sInnerSize));

            for (int64_t sOuterIndex = 0; sOuterIndex < outerBlockNums; sOuterIndex++) {
                int64_t dif = static_cast<int64_t>(coreWightTarget * static_cast<double>(curCore + 1)) - curWight;

                // Compute actualInnerBlockNums for this sOuter row
                int64_t actualInnerBlockNums = 0;
                if (isMlaSink) {
                    int64_t cubeSOuterOffset = sOuterIndex * static_cast<int64_t>(sOuterSize);
                    int64_t sInnerFirstToken = ClipSInnerToken(
                        (cubeSOuterOffset - preTokensLeftUp) / gSize, 0, actualSeqLengthKV)
                        / static_cast<int64_t>(sInnerSize);
                    int64_t sInnerLastToken = ClipSInnerToken(
                        CeilDiv(cubeSOuterOffset + nextTokensLeftUp + sOuterSize,
                                static_cast<int64_t>(gSize)),
                        0, actualSeqLengthKV);
                    actualInnerBlockNums = CeilDiv(sInnerLastToken, static_cast<int64_t>(sInnerSize)) - sInnerFirstToken;
                    if (kSinkNum_ > 0) {
                        actualInnerBlockNums += sinkBlockNumberPerS1g;
                    }
                } else {
                    int64_t preTokensNoPrefix = preTokensLeftUp;
                    int64_t nextTokensNoPrefix = nextTokensLeftUp;
                    int64_t sInnerIndexStart = -(preTokensNoPrefix > 0 ?
                        CeilDiv(preTokensNoPrefix, static_cast<int64_t>(sInnerSize)) :
                        preTokensNoPrefix / static_cast<int64_t>(sInnerSize));
                    int64_t sInnerIndexEnd = nextTokensNoPrefix > 0 ?
                        CeilDiv(nextTokensNoPrefix, static_cast<int64_t>(sInnerSize)) :
                        nextTokensNoPrefix / static_cast<int64_t>(sInnerSize);
                    actualInnerBlockNums = GetActualInnerBlockNums(sInnerIndexStart, sInnerIndexEnd, innerBlockNums);
                    if (kSinkNum_ > 0) {
                        actualInnerBlockNums += sinkBlockNumberPerS1g;
                    }
                }

                // Greedy: if adding this block exceeds target more than not adding, start new core
                if (actualInnerBlockNums - dif > dif &&
                    !(tmpCoreNidEnd == 0 && tmpCoreSidEnd == 0 && tmpCoreSposEnd == 0)) {
                    curCore += 1;
                    if (curCore < curCoreNum) {
                        splitBnStartIdx_[curCore] = static_cast<int32_t>(sIdx * headNumSize + headNum);
                        splitGS1StartIdx_[curCore] = static_cast<int32_t>(sOuterIndex);
                    }
                }
                tmpCoreNidEnd = headNum + 1;
                tmpCoreSidEnd = sIdx + 1;
                tmpCoreSposEnd = static_cast<uint32_t>(sOuterIndex + 1);

                curWight += actualInnerBlockNums;
                preTokensLeftUp -= static_cast<int64_t>(sOuterSize);
                nextTokensLeftUp += static_cast<int64_t>(sOuterSize);
            }
        }
    }

    // Sentinel values for the last core + 1
    uint32_t lastIdx = std::min(curCore + 1, curCoreNum);
    splitBnStartIdx_[lastIdx] = static_cast<int32_t>(bSize * headNumSize);
    splitGS1StartIdx_[lastIdx] = static_cast<int32_t>(tmpCoreSposEnd);

    // Actual cube cores used (capped at aicCoreNum_)
    splitUsedCoreNum_ = std::min(curCore + 1, curCoreNum);

    // Pad remaining entries to arrayLen
    splitBnStartIdx_.resize(AP_SPLIT_ARRAY_LEN, splitBnStartIdx_.back());
    splitGS1StartIdx_.resize(AP_SPLIT_ARRAY_LEN, splitGS1StartIdx_.back());
}

REGISTER_CPU_KERNEL(kAttentionPioneerMetadata, AttentionPioneerMetadataKernel);
} // namespace aicpu
