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
#include "../op_host/common/attention_pioneer_metadata.h"
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
    APBaseInfo baseInfo {};
    APSplitParam splitParam {};
    if (isMlaSink) {
        CalcInnerSizeMla(static_cast<uint32_t>(s2Size));
        CreateSplitInputMla(baseInfo, splitParam);
    } else {
        CalcInnerSizeGqa(static_cast<uint32_t>(s2Size));
        CreateSplitInputGqa(baseInfo, splitParam);
    }
    APSplitResult res {aicCoreNum_, cvRatio_};
    SplitCore(aicCoreNum_, baseInfo, splitParam, res);
    if (res.numOfFdHead > aicCoreNum_ || res.usedCoreNum > aicCoreNum_ || res.maxS2SplitNum > aicCoreNum_ + 1) {
        KERNEL_LOG_ERROR("used_core_num: %u, num_of_fd_head: %u, max_s2_split_num: %u, aic_num: %u",
            res.usedCoreNum, res.numOfFdHead, res.maxS2SplitNum, aicCoreNum_);
    }
    success = GenMetaData(res);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool AttentionPioneerMetadataKernel::GenMetaData(const APSplitResult &res) {
    optiling::detail::APMetaData* metaDataPtr = (optiling::detail::APMetaData*)metaData_->GetData();
    // aic Metadata Generate
    for (uint32_t i = 0; i < aicCoreNum_; ++i) {
        if (i >= res.usedCoreNum) {
            metaDataPtr->aicMetadata[i][AP_AIC_CORE_ENABLE_INDEX] = 0;
            continue;
        }
        metaDataPtr->aicMetadata[i][AP_AIC_CORE_ENABLE_INDEX] = 1;
        metaDataPtr->aicMetadata[i][AP_BN2_END_PTR_INDEX] = res.bN2End[i];
        metaDataPtr->aicMetadata[i][AP_GS1_END_PTR_INDEX] = res.gS1End[i];
        metaDataPtr->aicMetadata[i][AP_S2_END_PTR_INDEX] = res.s2End[i];
        metaDataPtr->aicMetadata[i][AP_BN2_IDX_OF_FD_HEAD_INDEX] = res.fdRes.bN2IdxOfFdHead[i];
        metaDataPtr->aicMetadata[i][AP_GS1_IDX_OF_FD_HEAD_INDEX] = res.fdRes.gS1IdxOfFdHead[i];
        metaDataPtr->aicMetadata[i][AP_S2_SPLIT_NUM_OF_FD_HEAD_INDEX] = res.fdRes.s2SplitNumOfFdHead[i];
        metaDataPtr->aicMetadata[i][AP_S2_SPLIT_START_IDX_OF_CORE_INDEX] = res.fdRes.s2SplitStartIdxOfCore[i];
        metaDataPtr->aicMetadata[i][AP_GS1_SPLIT_NUM_OF_FD_HEAD_INDEX] = res.fdRes.gS1SplitNumOfFdHead[i];
        metaDataPtr->aicMetadata[i][AP_GS1_LAST_PART_SIZE_OF_FD_HEAD_INDEX] = res.fdRes.gS1LastPartSizeOfFdHead[i];
    }

    // aiv Metadata Generate
    for (uint32_t i = 0; i < aicCoreNum_ * cvRatio_; ++i) {
        if (i >= (res.usedCoreNum * cvRatio_)) {
            metaDataPtr->aivMetadata[i][AP_AIV_CORE_ENABLE_INDEX] = 0;
            continue;
        }
        metaDataPtr->aivMetadata[i][AP_AIV_CORE_ENABLE_INDEX] = 1;
        metaDataPtr->aivMetadata[i][AP_GS1_IDX_END_OF_FD_HEAD_INDEX] = res.fdRes.gS1IdxEndOfFdHead[i];
        metaDataPtr->aivMetadata[i][AP_GS1_IDX_END_OF_FD_HEAD_SPLIT_INDEX] = res.fdRes.gS1IdxEndOfFdHeadSplit[i];
    }

    // base Metadata Generate
    metaDataPtr->baseMetadata[AP_M_BASE_SIZE_INDEX] = mBaseSize_;
    metaDataPtr->baseMetadata[AP_S_INNER_SIZE_INDEX] = sInnerSize_;
    metaDataPtr->baseMetadata[AP_M_FD_BASE_SIZE_INDEX] = mFdBaseSize_;
    metaDataPtr->baseMetadata[AP_NUM_OF_FD_INDEX] = res.numOfFdHead;
    metaDataPtr->baseMetadata[AP_USED_CORE_NUM_INDEX] = res.usedCoreNum;
    metaDataPtr->baseMetadata[AP_USED_VEC_NUM_OF_FD_INDEX] = res.usedVecNumOfFd;
    metaDataPtr->baseMetadata[AP_S1_SIZE_INDEX] = s1Size;
    metaDataPtr->baseMetadata[AP_S2_SIZE_INDEX] = s2Size;
    metaDataPtr->baseMetadata[AP_ACTUAL_LEN_Q_DIM_INDEX] = actualLenQDims;
    metaDataPtr->baseMetadata[AP_ACTUAL_LEN_KV_DIM_INDEX] = actualLenKvDims;
    return true;
}

void AttentionPioneerMetadataKernel::CalcInnerSizeMla(uint32_t seqSize)
{
    sInnerSize_ = 512U;
    uint32_t s1G = s1Size * gSize;
    if (sinkNum_ > 0 && sinkNum_ <= 256 && sparseMode_ == static_cast<uint32_t>(APSparseMode::BAND) && s1G >= 128) {
        sInnerSize_ = 256;
    }

    if (kSinkNum_ > 0 && sparseMode_ == static_cast<uint32_t>(APSparseMode::BAND) && bSize <= 6 &&
        (preToken + nextToken < 512)) {
        sInnerSize_ = 256;
    }

    if (splitKVFlag_ && inputLayout_ != "TND") {
        if (seqSize == 256U) {
            sInnerSize_ = 128U;
        } else if (seqSize > 256U && seqSize <= sInnerSize_) {
            sInnerSize_ = (sInnerSize_ + 1U) / 2U;
        }
    }

    if (blockSize_ != 0) {
        uint32_t blockSize = static_cast<uint32_t>(blockSize_);
        if ((sInnerSize_ > blockSize) && (sInnerSize_ % blockSize != 0)) {
            sInnerSize_ = (sInnerSize_ / blockSize) * blockSize;
        }
    }

    sInnerLoopTimes_ = (seqSize + sInnerSize_ - 1U) / sInnerSize_;
    sInnerSizeTail_ = seqSize - (sInnerLoopTimes_ - 1U) * sInnerSize_;
    if (sInnerSize_ > seqSize) {
        sInnerSize_ = seqSize;
    }
    sInnerSizeAlign_ = APAlign(sInnerSize_, AP_BYTE_BLOCK);
    CalcMBaseSizeMla();
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

void AttentionPioneerMetadataKernel::CreateSplitInputMla(APBaseInfo &baseInfo, APSplitParam &splitParam) const
{
    baseInfo.bSize = bSize;
    baseInfo.n2Size = n2Size;
    baseInfo.gSize = gSize;
    baseInfo.s2Size = static_cast<uint32_t>(s2Size);
    baseInfo.s1Size = s1Size;
    baseInfo.actualLenQDims = actualLenQDims;
    baseInfo.actualLenKvDims = actualLenKvDims;
    baseInfo.preToken = preToken;
    baseInfo.nextToken = nextToken;
    baseInfo.isS1G = inputLayout_ == "TND" || inputLayout_ == "BSH_BSND";
    baseInfo.sparseMode = sparseMode_;
    baseInfo.attenMaskFlag = attenMaskFlag;
    baseInfo.sinkNumber = sinkNum_;
    baseInfo.keySinkNumber = kSinkNum_;
    baseInfo.supportFD = true;
    if (actualSeqLengths_ != nullptr) {
        baseInfo.isAccumSeqS1 = isAccumQSeq;
        baseInfo.actualSeqS1Size.reserve(bSize);
        const int64_t *s1Ptr = static_cast<const int64_t*>(actualSeqLengths_->GetData());
        for (uint32_t i = 0; i < bSize; ++i) {
            baseInfo.actualSeqS1Size.emplace_back(s1Ptr[i]);
        }
    }
    if (actualSeqLengthsKv_ != nullptr) {
        baseInfo.isAccumSeqS2 = isAccumKVSeq;
        baseInfo.actualSeqS2Size.reserve(bSize);
        const int64_t *s2Ptr = static_cast<const int64_t*>(actualSeqLengthsKv_->GetData());
        for (uint32_t i = 0; i < bSize; ++i) {
            baseInfo.actualSeqS2Size.emplace_back(s2Ptr[i]);
        }
    }
    splitParam.mBaseSize = mBaseSize_;
    splitParam.s2BaseSize = sInnerSize_;
    splitParam.gS1BaseSizeOfFd = mFdBaseSize_;
}

bool AttentionPioneerMetadataKernel::DealSameSeqEachBatch()
{
    batchContinuousFlag = blockSize_ == 0 ? true : false;
    if (batchContinuousFlag) {
        if (actualSeqLenFlag) {
            return isSameActualseq;
        } else {
            return isSameSeqAllKVTensor;
        }
    } else {
        return isSameActualseq;
    }
}

void AttentionPioneerMetadataKernel::CalcInnerSizeGqa(uint32_t s2SizeVal)
{
    if (inputLayout_ == "TND" || inputLayout_ == "NTD") {
        sInnerSize_ = AP_S_INNER_SIZE_512;
        if ((sinkNum_ > 0 && sinkNum_ <= 256 && s1Size > AP_S1_SIZE_16
            && sparseMode_ == static_cast<uint32_t>(APSparseMode::BAND)) || (kSinkNum_ > 0)) {
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
    if (inputLayout_ == "TND" || inputLayout_ == "NTD") {
        mBaseSize_ = AP_M_BASE_SIZE_512;
        if (sinkNum_ > 0 && sinkNum_ <= 256 && s1Size > AP_S1_SIZE_16
            && sparseMode_ == static_cast<uint32_t>(APSparseMode::BAND) && nextToken == 0
            && preToken % 512 >= sinkNum_) {
            mBaseSize_ = AP_M_BASE_SIZE_256;
        }
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

void AttentionPioneerMetadataKernel::CreateSplitInputGqa(APBaseInfo &baseInfo, APSplitParam &splitParam) const
{
    baseInfo.bSize = bSize;
    baseInfo.n2Size = n2Size;
    baseInfo.gSize = gSize;
    baseInfo.s2Size = s2Size;
    baseInfo.s1Size = s1Size;
    baseInfo.actualLenQDims = actualLenQDims;
    baseInfo.actualLenKvDims = actualLenKvDims;
    baseInfo.preToken = preToken;
    baseInfo.nextToken = nextToken;
    baseInfo.isS1G = inputLayout_ == "TND" || inputLayout_ == "BSH_BSND";
    baseInfo.sparseMode = sparseMode_;
    baseInfo.attenMaskFlag = attenMaskFlag;
    baseInfo.sinkNumber = sinkNum_;
    baseInfo.keySinkNumber = kSinkNum_;
    baseInfo.supportFD = baseInfo.keySinkNumber ? false : true;
    if (actualSeqLengths_ != nullptr) {
        baseInfo.isAccumSeqS1 = isAccumQSeq;
        baseInfo.actualSeqS1Size.reserve(bSize);
        const int64_t *s1Ptr = static_cast<const int64_t*>(actualSeqLengths_->GetData());
        for (uint32_t i = 0; i < bSize; ++i) {
            baseInfo.actualSeqS1Size.emplace_back(s1Ptr[i]);
        }
    }
    if (actualSeqLengthsKv_ != nullptr) {
        baseInfo.isAccumSeqS2 = isAccumKVSeq;
        const int64_t *s2Ptr = static_cast<const int64_t*>(actualSeqLengthsKv_->GetData());
        for (uint32_t i = 0; i < bSize; ++i) {
            baseInfo.actualSeqS2Size.emplace_back(s2Ptr[i]);
        }
    }
    splitParam.mBaseSize = mBaseSize_;
    splitParam.s2BaseSize = sInnerSize_;
    splitParam.gS1BaseSizeOfFd = mFdBaseSize_;
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
    GetAttrValueOpt(ctx, "input_layout_kv", inputLayoutKv_);
    GetAttrValueOpt(ctx, "sink_num", sinkNum_);
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
        "BSH", "BSND", "BNSD", "TND", "NTD", "BSH_NBSD", "BSND_NBSD", "BNSD_NBSD", "TND_NTD", "NTD_TND", "BSH_BNSD", "BSND_BNSD", "BNSD_BSND"
    };
    if (std::find(inputLayoutList.begin(), inputLayoutList.end(), inputLayout_) == inputLayoutList.end()) {
        KERNEL_LOG_ERROR("input layout not supported: %s.", inputLayout_.c_str());
        return false;
    }
    const std::vector<std::string> inputLayoutKvList = {
        "BSH", "BSND", "BNSD", "TND", "NTD", "NZ", "BnBsH", "BnNBsD", "PA"
    };
    if (std::find(inputLayoutKvList.begin(), inputLayoutKvList.end(), inputLayoutKv_) == inputLayoutKvList.end()) {
        KERNEL_LOG_ERROR("input layout kv not supported: %s.", inputLayoutKv_.c_str());
        return false;
    }
    if (inputLayout_ == "TND" || inputLayout_ == "NTD") {
        KERNEL_CHECK_NULLPTR(actualSeqLengths_, false,
            "When inputLayout is TND or NTD, actualSeqQLengths should not be null.");
        KERNEL_CHECK_NULLPTR(actualSeqLengthsKv_, false,
            "When inputLayout is TND or NTD, actualSeqKVLengths should not be null.");
    }
    if (blockSize_ == 0) {
        if (inputLayoutKv_ != "BSH" && inputLayoutKv_ != "BSND" && inputLayoutKv_ != "BNSD" &&
            inputLayoutKv_ != "TND" && inputLayoutKv_ != "NTD") {
            KERNEL_LOG_ERROR("In BATCH_CONTINUOUS situation, key/value's layout not supported: %s", inputLayoutKv_.c_str());
            return false;
        }
        if (inputLayoutKv_ != inputLayout_) {
            KERNEL_LOG_ERROR("In BATCH_CONTINUOUS situation, key/value's layout and query's layout should be same.");
            return false;
        }
    } else {
        if ((inputLayoutKv_ == "BnBsH" || inputLayoutKv_ == "BnNBsD" || inputLayoutKv_ == "NZ" || inputLayoutKv_ == "PA") && (inputLayout_ != "BSH" && inputLayout_ != "BSND" &&
                        inputLayout_ != "BNSD" && inputLayout_ != "TND" && inputLayout_ != "NTD")) {
            KERNEL_LOG_ERROR("In PAGE_ATTENTION situation, query layout not supported: %s", inputLayout_.c_str());
            return false;
        }
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
    if (inputLayout_ == "TND" || inputLayout_ == "NTD") {
        return GetActualSeqLenSize(bSize);
    } else {
        bSize = batchSize_;
        return true;
    }
}

bool AttentionPioneerMetadataKernel::GetS1Size()
{
    if (inputLayout_ == "TND" || inputLayout_ == "NTD") {
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
    if (inputLayoutKv_ == "TND" || inputLayoutKv_ == "NTD") {
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
        if (inputLayout_ == "TND" || inputLayout_ == "NTD") {
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
        if (inputLayoutKv_ == "TND" || inputLayoutKv_ == "NTD") {
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
        "BSH", "BSND", "BNSD", "TND", "BSH_NBSD", "BSND_NBSD", "BNSD_NBSD", "TND_NTD"
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
        if (inputLayout_ == "TND" || inputLayout_ == "NTD") {
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
        if (inputLayoutKv_ == "TND" || inputLayoutKv_ == "NTD") {
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
    return (CheckActualSeqLensQ() && CheckActualSeqLensKv() && CheckSinkNumber());
}

bool AttentionPioneerMetadataKernel::CheckSinkNumber()
{
    if(sinkNum_ > 0 && kSinkNum_ > 0) {
        KERNEL_LOG_ERROR("sinkNum and kSinkNum cannot coexist");
        return false;
    }
    int64_t maxSinkNumber = 512;
    if (sinkNum_ == 0) {
        return true;
    }
    if (sinkNum_ < 0 || sinkNum_ > maxSinkNumber || sinkNum_ % 2 == 1) {
        KERNEL_LOG_ERROR("invalid Parameter sinkNumber");
        return false;
    }
    if (sinkNum_ > 0 && inputLayout_ != "TND") {
        KERNEL_LOG_ERROR("when sinkNumber > 0, qLayout should TND");
        return false;
    }
    uint64_t minActualKvSeqLen = 1000000000000;
    const int64_t *actualKvLenData = static_cast<const int64_t*>(actualSeqLengthsKv_->GetData());
    if (actualKvLenData != nullptr) {
        uint32_t loop = actualSeqLengthsKv_->GetTensorShape()->GetDimSize(0);
        for (uint32_t i = 0; i < loop; i++) {
            int64_t tmpS = 0;
            if (blockSize_ != 0) {
                tmpS = actualKvLenData[i];
            } else {
                tmpS = (i == 0U) ? actualKvLenData[0] : (actualKvLenData[i] - actualKvLenData[i - 1U]);
            }
            minActualKvSeqLen = minActualKvSeqLen < static_cast<uint64_t>(tmpS) ? minActualKvSeqLen : static_cast<uint64_t>(tmpS);
        }
    }
    if (minActualKvSeqLen < static_cast<uint64_t>(sinkNum_)) {
        KERNEL_LOG_ERROR("when sinkNumber > 0, sinknumber(%ld) should <= minActualKvSeqLen(%lu)",
                sinkNum_, minActualKvSeqLen);
        return false;
    }
    return CheckMlaSinkNumber();
}

bool AttentionPioneerMetadataKernel::CheckMlaSinkNumber()
{
    if (ropeHeadDim_ == 0U || headDimV_ != 512U) {
        return true;
    }
    if (sparseMode_ != static_cast<uint32_t>(APSparseMode::BAND)) {
        KERNEL_LOG_ERROR("sparseMode only support 4 when sinkNumber(%ld) exist and D is 512", sinkNum_);
        return false;
    }
    const int64_t *actualSeqLensDataQ = static_cast<const int64_t*>(actualSeqLengths_->GetData());
    const int64_t *actualSeqLensData = static_cast<const int64_t*>(actualSeqLengthsKv_->GetData());
    int64_t tmpS1Size = 0;
    int64_t tmpS2Size = 0;
    for (uint32_t i = 0; i < actualSeqLengthsQSize_; i++) {
        tmpS1Size = (i == 0U) ? actualSeqLensDataQ[0] : (actualSeqLensDataQ[i] - actualSeqLensDataQ[i - 1U]);
        if (blockSize_ != 0) {
            tmpS2Size = actualSeqLensData[i];
        } else {
            tmpS2Size = (i == 0U) ? actualSeqLensData[0] : (actualSeqLensData[i] - actualSeqLensData[i - 1U]);
        }
        if (tmpS2Size == sinkNum_) {
            continue;
        }
        if (preToken < 0 || (nextToken < (tmpS1Size - tmpS2Size + sinkNum_))) {
            KERNEL_LOG_ERROR("preToken < 0 or nextToken < (s1 - s2 + sinkNumber) is not supported when sinkNumber(%ld) exist and D is 512", sinkNum_);
            return false;
        }
    }
    return true;
}

bool AttentionPioneerMetadataKernel::CheckActualSeqLensQ() const
{
    if ((actualSeqLengths_ == nullptr) || (actualSeqLengths_->GetData() == nullptr)) {
        return true;
    }
    if (inputLayout_ == "TND") {
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
    if (inputLayout_ == "TND") {
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

// ============================= Split Core Functions ===========================

uint32_t AttentionPioneerMetadataKernel::GetS1SeqSize(uint32_t bIdx, const APBaseInfo &baseInfo)
{
    if (baseInfo.actualSeqS1Size.empty()) return baseInfo.s1Size;
    if (baseInfo.actualLenQDims == 1U) return static_cast<uint32_t>(baseInfo.actualSeqS1Size[0]);
    if (!baseInfo.isAccumSeqS1) return static_cast<uint32_t>(baseInfo.actualSeqS1Size[bIdx]);
    return (bIdx == 0) ? static_cast<uint32_t>(baseInfo.actualSeqS1Size[bIdx]) :
           static_cast<uint32_t>(baseInfo.actualSeqS1Size[bIdx] - baseInfo.actualSeqS1Size[bIdx - 1U]);
}

uint32_t AttentionPioneerMetadataKernel::GetS2SeqSize(uint32_t bIdx, const APBaseInfo &baseInfo)
{
    uint32_t prefix = static_cast<uint32_t>(baseInfo.actualSeqPrefixSize);
    if (baseInfo.actualSeqS2Size.empty()) return prefix + baseInfo.s2Size;
    if (baseInfo.actualLenKvDims == 1U) return prefix + static_cast<uint32_t>(baseInfo.actualSeqS2Size[0]);
    if (!baseInfo.isAccumSeqS2) return prefix + static_cast<uint32_t>(baseInfo.actualSeqS2Size[bIdx]);
    return (bIdx == 0) ? prefix + static_cast<uint32_t>(baseInfo.actualSeqS2Size[bIdx]) :
           prefix + static_cast<uint32_t>(baseInfo.actualSeqS2Size[bIdx] - baseInfo.actualSeqS2Size[bIdx - 1U]);
}

int64_t AttentionPioneerMetadataKernel::CalcPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const APBaseInfo &baseInfo)
{
    if (static_cast<APSparseMode>(baseInfo.sparseMode) == APSparseMode::BAND) {
        return static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + baseInfo.preToken;
    }
    return baseInfo.preToken;
}

int64_t AttentionPioneerMetadataKernel::CalcNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const APBaseInfo &baseInfo)
{
    auto mode = static_cast<APSparseMode>(baseInfo.sparseMode);
    switch (mode) {
        case APSparseMode::DEFAULT_MASK:
        case APSparseMode::ALL_MASK:
        case APSparseMode::LEFT_UP_CAUSAL:
            return baseInfo.nextToken;
        case APSparseMode::RIGHT_DOWN_CAUSAL:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        case APSparseMode::BAND:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + baseInfo.nextToken;
        default:
            return baseInfo.nextToken;
    }
}

int64_t AttentionPioneerMetadataKernel::CalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t alignBasicM = (basicM + 15U) >> 4U;
    uint32_t alignBasicS2 = (basicS2 + 63U) >> 6U;
    return static_cast<int64_t>(6U * alignBasicM + 10U * alignBasicS2);
}

APBlockCost<int64_t> AttentionPioneerMetadataKernel::CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize, uint32_t s1GTailSize, uint32_t s2TailSize)
{
    APBlockCost<int64_t> typeCost {};
    typeCost[AP_NORMAL_BLOCK][AP_NORMAL_BLOCK] = CalcCost(s1NormalSize, s2NormalSize);
    typeCost[AP_TAIL_BLOCK][AP_NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2NormalSize);
    typeCost[AP_NORMAL_BLOCK][AP_TAIL_BLOCK] = (s2TailSize == 0U) ? 0U : CalcCost(s1NormalSize, s2TailSize);
    typeCost[AP_TAIL_BLOCK][AP_TAIL_BLOCK] = (s1GTailSize == 0U || s2TailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2TailSize);
    return typeCost;
}

APRange<uint32_t> AttentionPioneerMetadataKernel::CalcS2Range(uint32_t s1GIdx, const APBaseInfo &baseInfo, const APSplitParam &splitParam, const APBatchCache &batchCache)
{
    uint32_t s2Start = 0U;
    uint32_t s2End = 0U;
    if (batchCache.s1Size == 0U || batchCache.s2Size == 0U) {
        return std::make_pair(s2Start, s2End);
    }
    if (!baseInfo.attenMaskFlag) {
        s2Start = 0U;
        s2End = (batchCache.s2Size + splitParam.s2BaseSize - 1U) / splitParam.s2BaseSize;
        return std::make_pair(s2Start, s2End);
    }
    int64_t s1GFirstToken = static_cast<int64_t>(s1GIdx) * static_cast<int64_t>(splitParam.mBaseSize);
    int64_t s1GLastToken = std::min(s1GFirstToken + static_cast<int64_t>(splitParam.mBaseSize),
        static_cast<int64_t>(batchCache.s1Size) * static_cast<int64_t>(baseInfo.gSize)) - 1;
    int64_t s1FirstToken = 0;
    int64_t s1LastToken = 0;
    if (baseInfo.isS1G) {
        s1FirstToken = s1GFirstToken / static_cast<int64_t>(baseInfo.gSize);
        s1LastToken = s1GLastToken / static_cast<int64_t>(baseInfo.gSize);
    } else {
        if (s1GFirstToken / batchCache.s1Size == s1GLastToken / batchCache.s1Size) {
            s1FirstToken = s1GFirstToken % static_cast<int64_t>(batchCache.s1Size);
            s1LastToken = s1GLastToken % static_cast<int64_t>(batchCache.s1Size);
        } else {
            s1FirstToken = 0;
            s1LastToken = batchCache.s1Size;
        }
    }
    int64_t s2FirstToken = s1FirstToken - batchCache.preTokenLeftUp;
    int64_t s2LastToken = s1LastToken + batchCache.nextTokenLeftUp;
    if (s2FirstToken >= batchCache.s2Size || s2LastToken < 0 || s2LastToken < s2FirstToken) {
        return std::make_pair(0U, 0U);
    }
    s2FirstToken = APClip(s2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
    s2LastToken = APClip(s2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
    s2Start = static_cast<uint32_t>(s2FirstToken) / splitParam.s2BaseSize;
    s2End = static_cast<uint32_t>(s2LastToken) / splitParam.s2BaseSize + 1U;
    return std::make_pair(s2Start, s2End);
}

void AttentionPioneerMetadataKernel::CalcSplitInfo(APSplitContext &splitContext)
{
    const APBaseInfo &baseInfo = splitContext.baseInfo;
    const APSplitParam &splitParam = splitContext.splitParam;
    APSplitInfo &splitInfo = splitContext.splitInfo;
    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        uint32_t s1Val = GetS1SeqSize(bIdx, baseInfo);
        uint32_t s2Val;
        if (baseInfo.keySinkNumber && baseInfo.supportFD) {
            s2Val = baseInfo.keySinkNumber + GetS2SeqSize(bIdx, baseInfo);
        } else {
            s2Val = GetS2SeqSize(bIdx, baseInfo);
        }
        splitInfo.s1GBaseNum[bIdx] = (s1Val * baseInfo.gSize + (splitParam.mBaseSize - 1U)) / splitParam.mBaseSize;
        splitInfo.s1GTailSize[bIdx] = (s1Val * baseInfo.gSize) % splitParam.mBaseSize;
        splitInfo.s2BaseNum[bIdx] = (s2Val + splitParam.s2BaseSize - 1U) / splitParam.s2BaseSize;
        splitInfo.s2TailSize[bIdx] = s2Val % splitParam.s2BaseSize;
        if (splitInfo.s1GBaseNum[bIdx] != 0U && splitInfo.s2BaseNum[bIdx] != 0U) {
            splitInfo.isKvSeqAllZero = false;
        }
    }
}

void AttentionPioneerMetadataKernel::CalcBatchCache(uint32_t bIdx, const APSplitContext &splitContext, APBatchCache &batchCache)
{
    const APBaseInfo &baseInfo = splitContext.baseInfo;
    const APSplitParam &splitParam = splitContext.splitParam;
    const APSplitInfo &splitInfo = splitContext.splitInfo;
    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1SeqSize(bIdx, baseInfo);
    if (baseInfo.keySinkNumber && baseInfo.supportFD) {
        batchCache.s2Size = GetS2SeqSize(bIdx, baseInfo) + baseInfo.keySinkNumber;
    } else {
        batchCache.s2Size = GetS2SeqSize(bIdx, baseInfo);
    }
    batchCache.preTokenLeftUp = CalcPreTokenLeftUp(batchCache.s1Size, batchCache.s2Size, baseInfo);
    batchCache.nextTokenLeftUp = CalcNextTokenLeftUp(batchCache.s1Size, batchCache.s2Size, baseInfo);
    batchCache.typeCost = CalcCostTable(splitParam.mBaseSize, splitParam.s2BaseSize, splitInfo.s1GTailSize[bIdx], splitInfo.s2TailSize[bIdx]);
}

void AttentionPioneerMetadataKernel::CalcS1GCacheWithSinkNumber(uint32_t s1GIdx, const APSplitContext &splitContext, const APBatchCache &batchCache, APS1GCache &s1GCache, uint32_t sinkNumber)
{
    s1GCache.s2Start = 0;
    if (s1GCache.s2End == 0) return;
    if (s1GCache.firstTokenWithSinkNumber == 0) return;
    const APSplitParam &splitParam = splitContext.splitParam;
    const APSplitInfo &splitInfo = splitContext.splitInfo;
    uint32_t sinkBlockLastIndex = (static_cast<uint32_t>(sinkNumber) + splitParam.s2BaseSize - 1) / splitParam.s2BaseSize - 1;
    if (s1GCache.firstTokenWithSinkNumber > sinkBlockLastIndex) {
        s1GCache.s1GBlock += sinkBlockLastIndex + 1;
        if (s1GIdx == (splitInfo.s1GBaseNum[batchCache.bIdx] - 1U) && splitInfo.s1GTailSize[batchCache.bIdx] != 0U) {
            s1GCache.s1GCost += batchCache.typeCost[AP_TAIL_BLOCK][AP_NORMAL_BLOCK] * (sinkBlockLastIndex + 1);
        } else {
            s1GCache.s1GCost += batchCache.typeCost[AP_NORMAL_BLOCK][AP_NORMAL_BLOCK] * (sinkBlockLastIndex + 1);
        }
    } else {
        s1GCache.s1GBlock += s1GCache.firstTokenWithSinkNumber;
        if (s1GIdx == (splitInfo.s1GBaseNum[batchCache.bIdx] - 1U) && splitInfo.s1GTailSize[batchCache.bIdx] != 0U) {
            s1GCache.s1GCost += batchCache.typeCost[AP_TAIL_BLOCK][AP_NORMAL_BLOCK] * s1GCache.firstTokenWithSinkNumber;
        } else {
            s1GCache.s1GCost += batchCache.typeCost[AP_NORMAL_BLOCK][AP_NORMAL_BLOCK] * s1GCache.firstTokenWithSinkNumber;
        }
    }
}

void AttentionPioneerMetadataKernel::CalcS1GCache(uint32_t s1GIdx, const APSplitContext &splitContext, const APBatchCache &batchCache, APS1GCache &s1GCache)
{
    const APBaseInfo &baseInfo = splitContext.baseInfo;
    const APSplitParam &splitParam = splitContext.splitParam;
    const APSplitInfo &splitInfo = splitContext.splitInfo;
    s1GCache.bIdx = batchCache.bIdx;
    s1GCache.s1GIdx = s1GIdx;
    auto s2Range = CalcS2Range(s1GIdx, baseInfo, splitParam, batchCache);
    s1GCache.s2Start = s2Range.first;
    s1GCache.s2End = s2Range.second;
    s1GCache.firstTokenWithSinkNumber = s2Range.first;
    if (s1GCache.s2Start >= s1GCache.s2End) {
        s1GCache.s1GBlock = 0;
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
        return;
    }
    s1GCache.s1GBlock = s1GCache.s2End - s1GCache.s2Start;
    uint32_t curTailS2Num = (splitInfo.s2TailSize[batchCache.bIdx] != 0U &&
        s1GCache.s2End == splitInfo.s2BaseNum[batchCache.bIdx]) ? 1U : 0U;
    uint32_t curNormalS2Num = s1GCache.s1GBlock - curTailS2Num;
    if (splitInfo.s1GBaseNum[batchCache.bIdx] == 0) {
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
    } else if (s1GIdx == (splitInfo.s1GBaseNum[batchCache.bIdx] - 1U) && splitInfo.s1GTailSize[batchCache.bIdx] != 0U) {
        s1GCache.s1GCost = batchCache.typeCost[AP_TAIL_BLOCK][AP_NORMAL_BLOCK] * curNormalS2Num +
            batchCache.typeCost[AP_TAIL_BLOCK][AP_TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GLastBlockCost = curTailS2Num > 0U ? batchCache.typeCost[AP_TAIL_BLOCK][AP_TAIL_BLOCK] :
                                 batchCache.typeCost[AP_TAIL_BLOCK][AP_NORMAL_BLOCK];
        s1GCache.s1GNormalBlockCost = batchCache.typeCost[AP_TAIL_BLOCK][AP_NORMAL_BLOCK];
    } else {
        s1GCache.s1GCost = batchCache.typeCost[AP_NORMAL_BLOCK][AP_NORMAL_BLOCK] * curNormalS2Num +
            batchCache.typeCost[AP_NORMAL_BLOCK][AP_TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GLastBlockCost = curTailS2Num > 0U ? batchCache.typeCost[AP_NORMAL_BLOCK][AP_TAIL_BLOCK] :
                                 batchCache.typeCost[AP_NORMAL_BLOCK][AP_NORMAL_BLOCK];
        s1GCache.s1GNormalBlockCost = batchCache.typeCost[AP_NORMAL_BLOCK][AP_NORMAL_BLOCK];
    }
    uint32_t sinkNumber = (baseInfo.keySinkNumber && baseInfo.supportFD) ? baseInfo.keySinkNumber : baseInfo.sinkNumber;
    if (sinkNumber > 0 && baseInfo.sparseMode == 4U) {
        CalcS1GCacheWithSinkNumber(s1GIdx, splitContext, batchCache, s1GCache, sinkNumber);
    }
}

void AttentionPioneerMetadataKernel::CopyTmpResult(APSplitResult &tmpRes, APSplitResult &splitRes)
{
    uint64_t len = tmpRes.bN2End.size();
    splitRes.usedCoreNum = tmpRes.usedCoreNum;
    splitRes.maxCost = tmpRes.maxCost;
    splitRes.numOfFdHead = tmpRes.numOfFdHead;
    splitRes.maxS2SplitNum = tmpRes.maxS2SplitNum;
    for (size_t i = 0; i < len; ++i) {
        splitRes.bN2End[i] = tmpRes.bN2End[i];
        splitRes.gS1End[i] = tmpRes.gS1End[i];
        splitRes.s2End[i] = tmpRes.s2End[i];
        splitRes.fdRes.bN2IdxOfFdHead[i] = tmpRes.fdRes.bN2IdxOfFdHead[i];
        splitRes.fdRes.gS1IdxOfFdHead[i] = tmpRes.fdRes.gS1IdxOfFdHead[i];
        splitRes.fdRes.s2SplitNumOfFdHead[i] = tmpRes.fdRes.s2SplitNumOfFdHead[i];
        splitRes.fdRes.s2SplitStartIdxOfCore[i] = tmpRes.fdRes.s2SplitStartIdxOfCore[i];
        splitRes.fdRes.gS1SplitNumOfFdHead[i] = tmpRes.fdRes.gS1SplitNumOfFdHead[i];
        splitRes.fdRes.gS1LastPartSizeOfFdHead[i] = tmpRes.fdRes.gS1LastPartSizeOfFdHead[i];
    }
}

void AttentionPioneerMetadataKernel::ClearTmpResult(APSplitResult &tmpResult)
{
    uint64_t len = tmpResult.bN2End.size();
    tmpResult.usedCoreNum = 0U;
    tmpResult.maxCost = 0;
    tmpResult.numOfFdHead = 0U;
    tmpResult.maxS2SplitNum = 0U;
    tmpResult.usedVecNumOfFd = 0U;
    for (size_t i = 0; i < len; ++i) {
        tmpResult.bN2End[i] = 0U;
        tmpResult.gS1End[i] = 0U;
        tmpResult.s2End[i] = 0U;
        tmpResult.fdRes.bN2IdxOfFdHead[i] = 0U;
        tmpResult.fdRes.gS1IdxOfFdHead[i] = 0U;
        tmpResult.fdRes.s2SplitNumOfFdHead[i] = 0U;
        tmpResult.fdRes.s2SplitStartIdxOfCore[i] = 0U;
        tmpResult.fdRes.gS1SplitNumOfFdHead[i] = 0U;
        tmpResult.fdRes.gS1LastPartSizeOfFdHead[i] = 0U;
    }
}

void AttentionPioneerMetadataKernel::CalcBatchCost(uint32_t bIdx, const APSplitContext &splitContext, APCostInfo &costInfo)
{
    const APBaseInfo &baseInfo = splitContext.baseInfo;
    const APSplitInfo &splitInfo = splitContext.splitInfo;
    costInfo.bN2CostOfEachBatch[bIdx] = 0;
    costInfo.bN2BlockOfEachBatch[bIdx] = 0U;
    costInfo.bN2LastBlockCostOfEachBatch[bIdx] = 0U;
    if (GetS1SeqSize(bIdx, baseInfo) == 0U || GetS2SeqSize(bIdx, baseInfo) == 0U) return;
    APBatchCache bCache;
    APS1GCache s1GCacheLocal;
    CalcBatchCache(bIdx, splitContext, bCache);
    for (uint32_t s1GIdx = 0; s1GIdx < splitInfo.s1GBaseNum[bIdx]; s1GIdx++) {
        CalcS1GCache(s1GIdx, splitContext, bCache, s1GCacheLocal);
        costInfo.bN2CostOfEachBatch[bIdx] += s1GCacheLocal.s1GCost;
        costInfo.bN2BlockOfEachBatch[bIdx] += s1GCacheLocal.s1GBlock;
        if (s1GCacheLocal.s1GBlock > 0) {
            costInfo.bN2LastBlockCostOfEachBatch[bIdx] = s1GCacheLocal.s1GLastBlockCost;
        }
    }
}

void AttentionPioneerMetadataKernel::CalcCostInfo(APSplitContext &splitContext)
{
    const APBaseInfo &baseInfo = splitContext.baseInfo;
    const APSplitInfo &splitInfo = splitContext.splitInfo;
    APCostInfo &costInfo = splitContext.costInfo;
    if (splitInfo.isKvSeqAllZero) {
        costInfo.totalCost = 0;
        costInfo.totalBlockNum = 0U;
        return;
    }
    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        CalcBatchCost(bIdx, splitContext, costInfo);
        costInfo.totalCost += costInfo.bN2CostOfEachBatch[bIdx] * baseInfo.n2Size;
        costInfo.totalBlockNum += costInfo.bN2BlockOfEachBatch[bIdx] * baseInfo.n2Size;
    }
}

void AttentionPioneerMetadataKernel::UpdateCursor(const APSplitContext &splitContext, APAssignContext &assignContext)
{
    const APBaseInfo &baseInfo = splitContext.baseInfo;
    const APSplitInfo &splitInfo = splitContext.splitInfo;
    const APCostInfo &costInfo = splitContext.costInfo;
    bool updateS1G = false;
    bool updateBatch = false;
    if (assignContext.curS2Idx >= assignContext.s1GCache.s2End) {
        assignContext.curS2Idx = 0U;
        assignContext.curS1GIdx++;
        updateS1G = true;
    }
    if (assignContext.curS1GIdx >= splitInfo.s1GBaseNum[assignContext.curBIdx]) {
        assignContext.curS1GIdx = 0U;
        assignContext.curBN2Idx++;
    }
    if (assignContext.curBN2Idx == baseInfo.bSize * baseInfo.n2Size) {
        assignContext.curS1GIdx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.isFinished = true;
        return;
    }
    if (assignContext.curBN2Idx / baseInfo.n2Size != assignContext.curBIdx) {
        assignContext.curBIdx = assignContext.curBN2Idx / baseInfo.n2Size;
        assignContext.curS1GIdx = 0U;
        updateBatch = true;
        updateS1G = true;
    }
    if (updateBatch) {
        CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    }
    if (updateS1G) {
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void AttentionPioneerMetadataKernel::AssignByBatch(const APSplitContext &splitContext, APAssignContext &assignContext)
{
    if (assignContext.isFinished) return;
    const APBaseInfo &baseInfo = splitContext.baseInfo;
    const APCostInfo &costInfo = splitContext.costInfo;
    while (assignContext.bN2Cost == 0 || APIsWithinTolerance(assignContext.coreCache.costLimit,
        costInfo.bN2LastBlockCostOfEachBatch[assignContext.curBIdx] / AP_FA_TOLERANCE_RATIO,
        assignContext.coreCache.cost + assignContext.bN2Cost)) {
        assignContext.coreCache.cost += assignContext.bN2Cost;
        assignContext.coreCache.block += assignContext.bN2Block;
        assignContext.curBN2Idx++;
        if (assignContext.curBN2Idx == baseInfo.bSize * baseInfo.n2Size) {
            assignContext.curS1GIdx = 0U;
            assignContext.curS2Idx = 0U;
            assignContext.isFinished = true;
            return;
        }
        if (assignContext.curBN2Idx / baseInfo.n2Size != assignContext.curBIdx) {
            assignContext.curBIdx = assignContext.curBN2Idx / baseInfo.n2Size;
            CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        }
        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
        assignContext.curS1GIdx = 0U;
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void AttentionPioneerMetadataKernel::AssignByRow(const APSplitContext &splitContext, APAssignContext &assignContext)
{
    if (assignContext.isFinished) return;
    while (APIsWithinTolerance(assignContext.coreCache.costLimit,
        assignContext.s1GCache.s1GLastBlockCost / AP_FA_TOLERANCE_RATIO,
        assignContext.coreCache.cost + assignContext.s1GCache.s1GCost)) {
        assignContext.coreCache.cost += assignContext.s1GCache.s1GCost;
        assignContext.coreCache.block += assignContext.s1GCache.s1GBlock;
        assignContext.bN2Cost = assignContext.bN2Cost > assignContext.s1GCache.s1GCost ?
                                assignContext.bN2Cost - assignContext.s1GCache.s1GCost : 0;
        assignContext.bN2Block = assignContext.bN2Block > assignContext.s1GCache.s1GBlock ?
                                 assignContext.bN2Block - assignContext.s1GCache.s1GBlock : 0U;
        do {
            assignContext.curS1GIdx++;
            CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        } while (assignContext.s1GCache.s1GBlock == 0);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void AttentionPioneerMetadataKernel::UpdateCurS2IdxWithSinkNumber(const APSplitContext &splitContext, APAssignContext &assignContext, uint32_t sinkNumber)
{
    const APSplitParam &splitParam = splitContext.splitParam;
    if (assignContext.s1GCache.s2End == 0) return;
    if (assignContext.s1GCache.firstTokenWithSinkNumber == 0) {
        assignContext.curS2Idx++;
        return;
    }
    uint32_t sinkBlockLastIndex = (static_cast<uint32_t>(sinkNumber) + splitParam.s2BaseSize - 1) / splitParam.s2BaseSize - 1;
    if (assignContext.curS2Idx < sinkBlockLastIndex || assignContext.curS2Idx >= assignContext.s1GCache.firstTokenWithSinkNumber) {
        assignContext.curS2Idx++;
        return;
    }
    if (assignContext.s1GCache.firstTokenWithSinkNumber <= sinkBlockLastIndex) {
        assignContext.curS2Idx++;
    } else {
        assignContext.curS2Idx = assignContext.s1GCache.firstTokenWithSinkNumber;
    }
}

void AttentionPioneerMetadataKernel::AssignByBlock(const APSplitContext &splitContext, APAssignContext &assignContext)
{
    if (assignContext.isFinished) return;
    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }
    uint32_t sinkNumber = (splitContext.baseInfo.keySinkNumber && splitContext.baseInfo.supportFD) ? splitContext.baseInfo.keySinkNumber : splitContext.baseInfo.sinkNumber;
    while (APIsWithinTolerance(assignContext.coreCache.costLimit, curCost / AP_FA_TOLERANCE_RATIO,
            assignContext.coreCache.cost + curCost)) {
        assignContext.coreCache.cost += curCost;
        assignContext.coreCache.block++;
        if (sinkNumber > 0 && splitContext.baseInfo.sparseMode == 4U) {
            UpdateCurS2IdxWithSinkNumber(splitContext, assignContext, sinkNumber);
        } else {
            assignContext.curS2Idx++;
        }
        assignContext.bN2Cost = assignContext.bN2Cost - curCost;
        assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
        assignContext.bN2Block--;
        assignContext.s1GCache.s1GBlock--;
    }
}

void AttentionPioneerMetadataKernel::ForceAssign(const APSplitContext &splitContext, APAssignContext &assignContext)
{
    if (assignContext.isFinished) return;
    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }
    assignContext.coreCache.cost += curCost;
    assignContext.coreCache.block++;
    uint32_t sinkNumber = (splitContext.baseInfo.keySinkNumber && splitContext.baseInfo.supportFD) ? splitContext.baseInfo.keySinkNumber : splitContext.baseInfo.sinkNumber;
    if (sinkNumber > 0 && splitContext.baseInfo.sparseMode == 4U) {
        UpdateCurS2IdxWithSinkNumber(splitContext, assignContext, sinkNumber);
    } else {
        assignContext.curS2Idx++;
    }
    assignContext.bN2Cost = assignContext.bN2Cost - curCost;
    assignContext.bN2Block--;
    assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
    assignContext.s1GCache.s1GBlock--;
    UpdateCursor(splitContext, assignContext);
}

bool AttentionPioneerMetadataKernel::IsNeedRecordFDInfo(const APAssignContext &assignContext, const APSplitResult &splitRes)
{
    if (assignContext.curCoreIdx == 0U) return false;
    if (assignContext.curKvSplitPart <= 1U) return false;
    if (assignContext.curBN2Idx == splitRes.bN2End[assignContext.curCoreIdx - 1U] &&
        assignContext.curS1GIdx == splitRes.gS1End[assignContext.curCoreIdx - 1U]) {
        return false;
    }
    return true;
}

void AttentionPioneerMetadataKernel::RecordFDInfo(const APSplitContext &splitContext, const APAssignContext &assignContext, APSplitResult &result)
{
    const APBaseInfo &baseInfo = splitContext.baseInfo;
    const APSplitParam &splitParam = splitContext.splitParam;
    const APSplitInfo &splitInfo = splitContext.splitInfo;
    uint32_t splitBIdx = result.bN2End[assignContext.curCoreIdx - 1U] / baseInfo.n2Size;
    uint32_t splitS1GIdx = result.gS1End[assignContext.curCoreIdx - 1U];
    uint32_t s1Val = GetS1SeqSize(splitBIdx, baseInfo);
    uint32_t curFdS1gSize = (splitS1GIdx == splitInfo.s1GBaseNum[splitBIdx] - 1U) ?
                            (s1Val * baseInfo.gSize - splitS1GIdx * splitParam.mBaseSize) : splitParam.mBaseSize;
    uint32_t curFdS1gSplitPart = (curFdS1gSize + splitParam.gS1BaseSizeOfFd - 1U) / splitParam.gS1BaseSizeOfFd;
    uint32_t curFdS1gLastPartSize = curFdS1gSize - (splitParam.gS1BaseSizeOfFd * (curFdS1gSplitPart - 1U));
    result.maxS2SplitNum = std::max(result.maxS2SplitNum, assignContext.curKvSplitPart);
    result.fdRes.bN2IdxOfFdHead[result.numOfFdHead] = result.bN2End[assignContext.curCoreIdx - 1U];
    result.fdRes.gS1IdxOfFdHead[result.numOfFdHead] = result.gS1End[assignContext.curCoreIdx - 1U];
    result.fdRes.s2SplitNumOfFdHead[result.numOfFdHead] = assignContext.curKvSplitPart;
    result.fdRes.gS1SplitNumOfFdHead[result.numOfFdHead] = curFdS1gSplitPart;
    result.fdRes.gS1LastPartSizeOfFdHead[result.numOfFdHead] = curFdS1gLastPartSize;
    result.numOfFdHead++;
}

void AttentionPioneerMetadataKernel::CalcSplitPlan(uint32_t coreNum, int64_t costLimit, const APSplitContext &splitContext, APSplitResult &result)
{
    const APCostInfo &costInfo = splitContext.costInfo;
    const APBaseInfo &baseInfo = splitContext.baseInfo;
    if (coreNum == 0U) return;
    result.maxCost = 0U;
    result.usedCoreNum = 0U;
    APAssignContext assignContext {};
    assignContext.curBIdx = 0U;
    assignContext.curS1GIdx = 0U;
    assignContext.unassignedCost = costInfo.totalCost;
    assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
    assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
    CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
    assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    for (uint32_t i = 0; i < coreNum; ++i) {
        if (result.maxCost > costLimit) return;
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) break;
        assignContext.curCoreIdx = i;
        result.fdRes.s2SplitStartIdxOfCore[assignContext.curCoreIdx] = assignContext.curKvSplitPart - 1U;
        assignContext.coreCache = {};
        assignContext.coreCache.costLimit = assignContext.unassignedCost / (coreNum - assignContext.curCoreIdx);
        AssignByBatch(splitContext, assignContext);
        AssignByRow(splitContext, assignContext);
        if (baseInfo.supportFD) {
            AssignByBlock(splitContext, assignContext);
            if (assignContext.coreCache.block == 0) {
                ForceAssign(splitContext, assignContext);
            }
        }
        result.bN2End[i] = assignContext.curBN2Idx;
        result.gS1End[i] = assignContext.curS1GIdx;
        result.s2End[i] = assignContext.curS2Idx;
        result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);
        assignContext.unassignedCost -= assignContext.coreCache.cost;
        if (baseInfo.supportFD) {
            if (IsNeedRecordFDInfo(assignContext, result)) {
                RecordFDInfo(splitContext, assignContext, result);
                assignContext.curKvSplitPart = 1U;
            }
            if (assignContext.curS2Idx > assignContext.s1GCache.s2Start &&
                assignContext.curS2Idx <= assignContext.s1GCache.s2End) {
                assignContext.curKvSplitPart++;
            }
        }
    }
    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

void AttentionPioneerMetadataKernel::SplitFD(APSplitResult &result)
{
    uint32_t totalFDLoad = 0;
    uint32_t totalFDHeadSplit = 0;
    for (uint32_t i = 0; i < result.numOfFdHead; i++) {
        totalFDLoad += result.fdRes.s2SplitNumOfFdHead[i] * result.fdRes.gS1SplitNumOfFdHead[i];
        totalFDHeadSplit += result.fdRes.gS1SplitNumOfFdHead[i];
    }
    uint32_t maxVectorNum = std::min(totalFDHeadSplit, result.usedCoreNum * result.vecCubeRatio);
    double loadThrOfVector = static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum);
    int64_t loadOfCurVector = 0;
    uint32_t curCoreIndex = 0;
    uint32_t preTmpFDIndexEndOfFdHead = 0;
    uint32_t preTmpFDIndexEndOfFdHeadSplit = 0;
    for (uint32_t i = 0; i < result.numOfFdHead; i++) {
        uint32_t fDKVSplitNum = result.fdRes.s2SplitNumOfFdHead[i];
        for (uint32_t gS1SplitIdx = 0; gS1SplitIdx < result.fdRes.gS1SplitNumOfFdHead[i]; gS1SplitIdx++) {
            double remainSpace = loadThrOfVector - static_cast<double>(loadOfCurVector);
            if (fDKVSplitNum > remainSpace * AP_FD_TOLERANCE_RATIO) {
                result.fdRes.gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
                result.fdRes.gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
                curCoreIndex += 1U;
                totalFDLoad -= static_cast<uint32_t>(loadOfCurVector);
                loadThrOfVector = static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum - curCoreIndex);
                loadOfCurVector = 0;
            }
            loadOfCurVector += fDKVSplitNum;
            preTmpFDIndexEndOfFdHead = i;
            preTmpFDIndexEndOfFdHeadSplit = gS1SplitIdx;
        }
    }
    result.fdRes.gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
    result.fdRes.gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
    result.usedVecNumOfFd = curCoreIndex + 1;
}

void AttentionPioneerMetadataKernel::SplitCore(uint32_t coreNum, const APBaseInfo &baseInfo, const APSplitParam &param, APSplitResult &result)
{
    APSplitContext splitContext(baseInfo, param);
    CalcSplitInfo(splitContext);
    if (splitContext.splitInfo.isKvSeqAllZero) {
        result.usedCoreNum = 1U;
        result.bN2End[0] = baseInfo.bSize * baseInfo.n2Size;
        result.gS1End[0] = 0U;
        result.s2End[0] = 0U;
        return;
    }
    CalcCostInfo(splitContext);
    uint32_t maxCore = std::min(coreNum, splitContext.costInfo.totalBlockNum);
    uint32_t minCore = static_cast<uint32_t>(
        std::sqrt(static_cast<float>(splitContext.costInfo.totalBlockNum) + 0.25f) + 0.5f);
    minCore = std::min(minCore, maxCore);
    result.maxCost = INT64_MAX;
    result.usedCoreNum = 1U;
    APSplitResult tmpResult {coreNum, result.vecCubeRatio};
    for (uint32_t i = minCore; i <= maxCore; ++i) {
        CalcSplitPlan(i, result.maxCost, splitContext, tmpResult);
        if (tmpResult.maxCost < result.maxCost) {
            CopyTmpResult(tmpResult, result);
        }
        ClearTmpResult(tmpResult);
    }
    if (result.numOfFdHead > 0U) {
        SplitFD(result);
    }
    result.usedCoreNum = std::max(result.usedCoreNum, 1U);
}

REGISTER_CPU_KERNEL(kAttentionPioneerMetadata, AttentionPioneerMetadataKernel);
} // namespace aicpu
