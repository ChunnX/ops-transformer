/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <algorithm>
#include "../op_host/attention_pioneer_metadata.h"
#include "cpu_context_util.h"
#include "attention_pioneer_metadata_aicpu.h"

using namespace optiling;

namespace aicpu {
namespace {
    const char* const kAttentionPioneerMetadata = "AttentionPioneerMetadata";
} // namespace

// ======================== Compute Entry ========================

uint32_t AttentionPioneerMetadataKernel::Compute(CpuKernelContext &ctx) {
    if (!Prepare(ctx)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    CheckIsDecode();
    if (!ParseAxisInfo()) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    SetTilingParams();
    ComputeSplitNBSeq();
    if (!GenMetaData()) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

// ======================== Stage 1: Prepare ========================

bool AttentionPioneerMetadataKernel::Prepare(CpuKernelContext &ctx) {
    actualSeqLengths_ = ctx.Input(static_cast<uint32_t>(ParamId::actualSeqLengthsOptional));
    actualSeqLengthsKv_ = ctx.Input(static_cast<uint32_t>(ParamId::actualSeqLengthsKvOptional));
    metaData_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    KERNEL_CHECK_NULLPTR(metaData_, false, "metadata is null");
    KERNEL_CHECK_NULLPTR(metaData_->GetData(), false, "metadata data is null");
    KERNEL_CHECK_NULLPTR(actualSeqLengths_, false, "actualSeqLengths is null");
    KERNEL_CHECK_NULLPTR(actualSeqLengthsKv_, false, "actualSeqLengthsKv is null");

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
    GetAttrValueOpt(ctx, "sparse_mode", sparseMode_);
    GetAttrValueOpt(ctx, "pre_tokens", preTokens_);
    GetAttrValueOpt(ctx, "next_tokens", nextTokens_);
    GetAttrValueOpt(ctx, "input_layout", inputLayout_);
    GetAttrValueOpt(ctx, "k_sink_num", kSinkNum_);
    GetAttrValueOpt(ctx, "block_size", blockSize_);

    if (aicCoreNum_ == 0 || aivCoreNum_ == 0) {
        KERNEL_LOG_ERROR("Core num invalid: aic:%u, aiv:%u", aicCoreNum_, aivCoreNum_);
        return false;
    }

    // Head sizes
    n1Size_ = numHeadsQ_;
    n2Size_ = (numHeadsKv_ == 0) ? n1Size_ : numHeadsKv_;
    if (n1Size_ == 0 || n2Size_ == 0 || n1Size_ % n2Size_ != 0) {
        KERNEL_LOG_ERROR("numHeadsQ(%u) must be a multiple of numHeadsKv(%u)", n1Size_, n2Size_);
        return false;
    }
    gSize_ = n1Size_ / n2Size_;
    return true;
}

// ======================== Stage 2: CheckIsDecode ========================

void AttentionPioneerMetadataKernel::CheckIsDecode() {
    constexpr uint32_t MLA_D_SIDE_QK_HEAD_DIM = 512U;
    constexpr uint32_t MLA_D_SIDE_ROPE_HEAD_DIM = 64U;
    isDecode_ = (headDimQK_ == MLA_D_SIDE_QK_HEAD_DIM) &&
                (headDimV_ == MLA_D_SIDE_QK_HEAD_DIM) &&
                (ropeHeadDim_ == MLA_D_SIDE_ROPE_HEAD_DIM);
}

// ======================== Stage 3: ParseAxisInfo ========================

bool AttentionPioneerMetadataKernel::ParseAxisInfo() {
    // batchSize from actualSeqLengths tensor shape
    int64_t bDim = actualSeqLengths_->GetTensorShape()->GetDimSize(0);
    if (bDim <= 0) {
        KERNEL_LOG_ERROR("actualSeqLengths shape dim0 is %ld, should be > 0", bDim);
        return false;
    }
    bSize_ = static_cast<uint32_t>(bDim);

    if (!ParseActualSeqLengths()) {
        return false;
    }

    // s1Size: max Qs, D-side multiplied by G (qSeqLens_ already contains per-batch values, D-side already * G)
    int64_t maxQs = 0;
    for (uint32_t i = 0; i < bSize_; i++) {
        maxQs = std::max(maxQs, qSeqLens_[i]);
    }
    s1Size_ = static_cast<uint32_t>(maxQs);

    // s2Size: max KV seq length
    int64_t maxKVs = 0;
    for (uint32_t i = 0; i < bSize_; i++) {
        maxKVs = std::max(maxKVs, kvSeqLens_[i]);
    }
    s2Size_ = maxKVs;

    // preToken / nextToken based on sparse mode
    if (sparseMode_ != static_cast<uint32_t>(APSparseMode::DEFAULT_MASK)) {
        attenMaskFlag_ = true;
    }
    if (sparseMode_ == static_cast<uint32_t>(APSparseMode::ALL_MASK)) {
        preToken_ = SPARSE_MODE_INT_MAX;
        nextToken_ = SPARSE_MODE_INT_MAX;
    } else if (sparseMode_ == static_cast<uint32_t>(APSparseMode::LEFT_UP_CAUSAL) ||
               sparseMode_ == static_cast<uint32_t>(APSparseMode::RIGHT_DOWN_CAUSAL)) {
        preToken_ = SPARSE_MODE_INT_MAX;
        nextToken_ = 0;
    } else {
        preToken_ = (preTokens_ == SPARSE_MODE_INT_MAX) ? 0 : preTokens_;
        nextToken_ = (nextTokens_ == SPARSE_MODE_INT_MAX) ? 0 : nextTokens_;
    }

    return true;
}

bool AttentionPioneerMetadataKernel::ParseActualSeqLengths() {
    const int64_t *qData = static_cast<const int64_t*>(actualSeqLengths_->GetData());
    const int64_t *kvData = static_cast<const int64_t*>(actualSeqLengthsKv_->GetData());
    KERNEL_CHECK_NULLPTR(qData, false, "actualSeqLengths data is null");
    KERNEL_CHECK_NULLPTR(kvData, false, "actualSeqLengthsKv data is null");

    uint32_t kvDim = static_cast<uint32_t>(actualSeqLengthsKv_->GetTensorShape()->GetDimSize(0));
    uint32_t kvLoop = std::min(kvDim, bSize_);
    bool isPA = (blockSize_ > 0);

    qSeqLens_.resize(bSize_);
    kvSeqLens_.resize(bSize_);

    if (IsLayoutTND()) {
        // TND layout: Q is cumulative → differential; KV depends on PA
        for (uint32_t i = 0; i < bSize_; i++) {
            // Q: cumulative → per-batch
            qSeqLens_[i] = (i == 0) ? qData[0] : (qData[i] - qData[i - 1]);
            // D-side: Q * G
            if (isDecode_) {
                qSeqLens_[i] *= static_cast<int64_t>(gSize_);
            }

            // KV
            if (i < kvLoop) {
                kvSeqLens_[i] = kvData[i];
                if (!(isDecode_ && isPA) && i >= 1) {
                    kvSeqLens_[i] = kvData[i] - kvData[i - 1];
                }
            } else {
                kvSeqLens_[i] = kvSeqLens_[kvLoop - 1];
            }
        }
    } else {
        // Non-TND layouts (BSH, BSND, BNSD): direct values
        uint32_t qDim = static_cast<uint32_t>(actualSeqLengths_->GetTensorShape()->GetDimSize(0));
        for (uint32_t i = 0; i < bSize_; i++) {
            // Q: read from tensor (use index 0 if only 1 element)
            int64_t qVal = (qDim > 1) ? qData[i] : qData[0];
            // D-side BSH/BSND: Q * G
            if (isDecode_) {
                qVal *= static_cast<int64_t>(gSize_);
            }
            qSeqLens_[i] = qVal;

            // KV: direct values
            if (i < kvLoop) {
                kvSeqLens_[i] = (kvDim > 1) ? kvData[i] : kvData[0];
            } else {
                kvSeqLens_[i] = kvSeqLens_[kvLoop - 1];
            }
        }
    }

    return true;
}

// ======================== Stage 4: Tiling ========================

void AttentionPioneerMetadataKernel::SetTilingParams() {
    // P-side: sOuterSize=128, sInnerSize=128
    // D-side: sOuterSize=64,  sInnerSize=128
    if (isDecode_) {
        sOuterSize_ = 64U;
        sInnerSize_ = 128U;
    } else {
        sOuterSize_ = 128U;
        sInnerSize_ = 128U;
    }
}

// ======================== Split-core Helpers ========================

void AttentionPioneerMetadataKernel::GetPreNextTokensLeftUp(
    int64_t actualSeqLength, int64_t actualSeqLengthKV,
    int64_t &preTokensLeftUp, int64_t &nextTokensLeftUp) const
{
    auto mode = static_cast<APSparseMode>(sparseMode_);
    if (mode == APSparseMode::RIGHT_DOWN_CAUSAL) {
        preTokensLeftUp = SPARSE_MODE_INT_MAX;
        if (isDecode_) {
            nextTokensLeftUp = actualSeqLengthKV * static_cast<int64_t>(gSize_) - actualSeqLength;
        } else {
            nextTokensLeftUp = actualSeqLengthKV - actualSeqLength;
        }
    } else if (mode == APSparseMode::BAND) {
        if (isDecode_) {
            preTokensLeftUp = preToken_ * static_cast<int64_t>(gSize_) -
                actualSeqLengthKV * static_cast<int64_t>(gSize_) + actualSeqLength;
            nextTokensLeftUp = nextToken_ * static_cast<int64_t>(gSize_) +
                actualSeqLengthKV * static_cast<int64_t>(gSize_) - actualSeqLength;
        } else {
            preTokensLeftUp = preToken_ - actualSeqLengthKV + actualSeqLength;
            nextTokensLeftUp = nextToken_ + actualSeqLengthKV - actualSeqLength;
        }
    } else {
        if (isDecode_) {
            preTokensLeftUp = preToken_ * static_cast<int64_t>(gSize_);
            nextTokensLeftUp = nextToken_ * static_cast<int64_t>(gSize_);
        } else {
            preTokensLeftUp = preToken_;
            nextTokensLeftUp = nextToken_;
        }
    }
}

void AttentionPioneerMetadataKernel::FixParamWithRowInvalid(
    int64_t &actualSeqLength, int64_t actualSeqLengthKV,
    int64_t &preTokensLeftUp, int64_t &nextTokensLeftUp) const
{
    int64_t nextTokensError = (nextTokensLeftUp < 0) ? -nextTokensLeftUp : 0;
    nextTokensError = std::min(nextTokensError, actualSeqLength);

    int64_t preTokensError = 0;
    if (isDecode_) {
        preTokensError = (actualSeqLength > actualSeqLengthKV * static_cast<int64_t>(gSize_) + preTokensLeftUp) ?
            (actualSeqLength - actualSeqLengthKV * static_cast<int64_t>(gSize_) - preTokensLeftUp) : 0;
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
    return (sInnerIndexStart < 0) ? innerBlockNums :
        (sInnerIndexStart < innerBlockNums ? innerBlockNums - sInnerIndexStart : 0);
}

int64_t AttentionPioneerMetadataKernel::GetCalcBlockNumsForS1G(
    int64_t actualSeqLength, int64_t actualSeqLengthKV,
    int64_t preTokensLeftUp, int64_t nextTokensLeftUp) const
{
    int64_t s2BlockNum = 0;
    int64_t outerBlockNums = CeilDiv(actualSeqLength, static_cast<int64_t>(sOuterSize_));
    for (int64_t i = 0; i < outerBlockNums; i++) {
        int64_t cubeSOuterOffset = i * static_cast<int64_t>(sOuterSize_);
        int64_t sInnerFirstToken = ClipSInnerToken(
            (cubeSOuterOffset - preTokensLeftUp) / static_cast<int64_t>(gSize_),
            0, actualSeqLengthKV);
        int64_t sInnerLastToken = ClipSInnerToken(
            CeilDiv(cubeSOuterOffset + nextTokensLeftUp + static_cast<int64_t>(sOuterSize_),
                    static_cast<int64_t>(gSize_)),
            0, actualSeqLengthKV);
        s2BlockNum += CeilDiv(sInnerLastToken, static_cast<int64_t>(sInnerSize_)) -
                      sInnerFirstToken / static_cast<int64_t>(sInnerSize_);
    }
    return s2BlockNum;
}

int64_t AttentionPioneerMetadataKernel::GetCalcBlockNumsOneHead(
    int64_t actualSeqLength, int64_t actualSeqLengthKV,
    int64_t preTokensLeftUp, int64_t nextTokensLeftUp) const
{
    int64_t outerBlockNums = CeilDiv(actualSeqLength, static_cast<int64_t>(sOuterSize_));
    int64_t innerBlockNums = CeilDiv(actualSeqLengthKV, static_cast<int64_t>(sInnerSize_));

    if (!attenMaskFlag_) {
        return innerBlockNums * outerBlockNums;
    }
    if (isDecode_) {
        return GetCalcBlockNumsForS1G(actualSeqLength, actualSeqLengthKV,
            preTokensLeftUp, nextTokensLeftUp);
    }
    // P-side with attenMask: iterate sOuter rows
    int64_t totalBlockNums = 0;
    int64_t curPre = preTokensLeftUp;
    int64_t curNext = nextTokensLeftUp;
    for (int64_t sOuterIndex = 0; sOuterIndex < outerBlockNums; sOuterIndex++) {
        int64_t sInnerIndexStart = -(curPre > 0 ?
            CeilDiv(curPre, static_cast<int64_t>(sInnerSize_)) :
            curPre / static_cast<int64_t>(sInnerSize_));
        int64_t sInnerIndexEnd = curNext > 0 ?
            CeilDiv(curNext, static_cast<int64_t>(sInnerSize_)) :
            curNext / static_cast<int64_t>(sInnerSize_);
        totalBlockNums += GetActualInnerBlockNums(sInnerIndexStart, sInnerIndexEnd, innerBlockNums);
        curPre -= static_cast<int64_t>(sOuterSize_);
        curNext += static_cast<int64_t>(sOuterSize_);
    }
    return totalBlockNums;
}

int64_t AttentionPioneerMetadataKernel::SinkBlockCount(int64_t actualSeqLength) const
{
    int64_t outerBlockNums = CeilDiv(actualSeqLength, static_cast<int64_t>(sOuterSize_));
    int64_t sinkBlockPerS1g = CeilDiv(kSinkNum_, static_cast<int64_t>(sInnerSize_));
    return sinkBlockPerS1g * outerBlockNums;
}

// ======================== Stage 5: ComputeSplitNBSeq ========================

void AttentionPioneerMetadataKernel::ComputeSplitNBSeq()
{
    // P-side: headNumSize = N1, D-side: headNumSize = N2
    uint32_t headNumSize = isDecode_ ? n2Size_ : n1Size_;
    uint32_t curCoreNum = aicCoreNum_;

    // Build per-batch actual seq lengths for split
    // qSeqLens_ already contains D-side * G from ParseActualSeqLengths
    std::vector<int64_t> actualSeqLengths(bSize_);
    std::vector<int64_t> actualSeqLengthsKV(bSize_);
    for (uint32_t i = 0; i < bSize_; i++) {
        actualSeqLengths[i] = qSeqLens_[i];
        actualSeqLengthsKV[i] = kvSeqLens_[i];
    }

    // Phase 1: compute totalBlockNumsOneHead
    int64_t totalBlockNumsOneHead = 0;
    int64_t sinkBlockNumberPerS1g = 0;
    if (kSinkNum_ > 0) {
        sinkBlockNumberPerS1g = CeilDiv(kSinkNum_, static_cast<int64_t>(sInnerSize_));
    }

    for (uint32_t sIdx = 0; sIdx < bSize_; sIdx++) {
        int64_t actualSeqLengthsTmp = actualSeqLengths[sIdx];
        int64_t preTokensLeftUp = 0;
        int64_t nextTokensLeftUp = 0;
        GetPreNextTokensLeftUp(actualSeqLengths[sIdx], actualSeqLengthsKV[sIdx],
            preTokensLeftUp, nextTokensLeftUp);
        FixParamWithRowInvalid(actualSeqLengthsTmp, actualSeqLengthsKV[sIdx],
            preTokensLeftUp, nextTokensLeftUp);

        totalBlockNumsOneHead += GetCalcBlockNumsOneHead(actualSeqLengthsTmp,
            actualSeqLengthsKV[sIdx], preTokensLeftUp, nextTokensLeftUp);
        if (kSinkNum_ > 0) {
            totalBlockNumsOneHead += SinkBlockCount(actualSeqLengthsTmp);
        }
    }

    // Save for GenMetaData
    totalBlockNumsOneHead_ = totalBlockNumsOneHead;

    // coreWightTarget: average work per core
    double coreWightTarget = static_cast<double>(totalBlockNumsOneHead * headNumSize) /
                             static_cast<double>(curCoreNum);

    // s1OuterSize: seqSize = s1Size (already includes G for D-side)
    splitS1OuterSize_ = static_cast<uint32_t>(
        CeilDiv(static_cast<int64_t>(s1Size_), static_cast<int64_t>(sOuterSize_)));

    // Phase 2: greedy allocation across batch x head x sOuter
    splitCoreEnable_.assign(AP_AIC_CORE_NUM, 0);
    splitBnStartIdx_.assign(AP_AIC_CORE_NUM, 0);
    splitGS1StartIdx_.assign(AP_AIC_CORE_NUM, 0);

    int64_t curWight = 0;
    uint32_t curCore = 0;
    uint32_t tmpCoreNidEnd = 0;
    uint32_t tmpCoreSidEnd = 0;
    uint32_t tmpCoreSposEnd = 0;

    // Core 0 is always enabled when there is work
    splitCoreEnable_[0] = 1;

    for (uint32_t sIdx = 0; sIdx < bSize_; sIdx++) {
        for (uint32_t headNum = 0; headNum < headNumSize; headNum++) {
            int64_t preTokensLeftUp = 0;
            int64_t nextTokensLeftUp = 0;
            GetPreNextTokensLeftUp(actualSeqLengths[sIdx], actualSeqLengthsKV[sIdx],
                preTokensLeftUp, nextTokensLeftUp);
            int64_t actualSeqLength = actualSeqLengths[sIdx];
            int64_t actualSeqLengthKV = actualSeqLengthsKV[sIdx];
            FixParamWithRowInvalid(actualSeqLength, actualSeqLengthKV,
                preTokensLeftUp, nextTokensLeftUp);

            int64_t outerBlockNums = CeilDiv(actualSeqLength, static_cast<int64_t>(sOuterSize_));
            int64_t innerBlockNums = CeilDiv(actualSeqLengthKV, static_cast<int64_t>(sInnerSize_));

            for (int64_t sOuterIndex = 0; sOuterIndex < outerBlockNums; sOuterIndex++) {
                int64_t dif = static_cast<int64_t>(
                    coreWightTarget * static_cast<double>(curCore + 1)) - curWight;

                // Compute actualInnerBlockNums for this sOuter row
                int64_t actualInnerBlockNums = 0;
                if (isDecode_) {
                    int64_t cubeSOuterOffset = sOuterIndex * static_cast<int64_t>(sOuterSize_);
                    int64_t sInnerFirstToken = ClipSInnerToken(
                        (cubeSOuterOffset - preTokensLeftUp) / static_cast<int64_t>(gSize_),
                        0, actualSeqLengthKV) / static_cast<int64_t>(sInnerSize_);
                    int64_t sInnerLastToken = ClipSInnerToken(
                        CeilDiv(cubeSOuterOffset + nextTokensLeftUp +
                                static_cast<int64_t>(sOuterSize_),
                                static_cast<int64_t>(gSize_)),
                        0, actualSeqLengthKV);
                    actualInnerBlockNums = CeilDiv(sInnerLastToken,
                        static_cast<int64_t>(sInnerSize_)) - sInnerFirstToken;
                    if (kSinkNum_ > 0) {
                        actualInnerBlockNums += sinkBlockNumberPerS1g;
                    }
                } else {
                    int64_t sInnerIndexStart = -(preTokensLeftUp > 0 ?
                        CeilDiv(preTokensLeftUp, static_cast<int64_t>(sInnerSize_)) :
                        preTokensLeftUp / static_cast<int64_t>(sInnerSize_));
                    int64_t sInnerIndexEnd = nextTokensLeftUp > 0 ?
                        CeilDiv(nextTokensLeftUp, static_cast<int64_t>(sInnerSize_)) :
                        nextTokensLeftUp / static_cast<int64_t>(sInnerSize_);
                    actualInnerBlockNums = GetActualInnerBlockNums(
                        sInnerIndexStart, sInnerIndexEnd, innerBlockNums);
                    if (kSinkNum_ > 0) {
                        actualInnerBlockNums += sinkBlockNumberPerS1g;
                    }
                }

                // Greedy: if adding this block exceeds target more than not, start new core
                if (actualInnerBlockNums - dif > dif &&
                    !(tmpCoreNidEnd == 0 && tmpCoreSidEnd == 0 && tmpCoreSposEnd == 0)) {
                    curCore += 1;
                    if (curCore < curCoreNum) {
                        splitCoreEnable_[curCore] = 1;
                        splitBnStartIdx_[curCore] =
                            static_cast<int32_t>(sIdx * headNumSize + headNum);
                        splitGS1StartIdx_[curCore] =
                            static_cast<int32_t>(sOuterIndex);
                    }
                }
                tmpCoreNidEnd = headNum + 1;
                tmpCoreSidEnd = sIdx + 1;
                tmpCoreSposEnd = static_cast<uint32_t>(sOuterIndex + 1);

                curWight += actualInnerBlockNums;
                preTokensLeftUp -= static_cast<int64_t>(sOuterSize_);
                nextTokensLeftUp += static_cast<int64_t>(sOuterSize_);
            }
        }
    }

    splitUsedCoreNum_ = std::min(curCore + 1, curCoreNum);
}

// ======================== Stage 6: GenMetaData ========================

bool AttentionPioneerMetadataKernel::GenMetaData() {
    detail::APMetaData* metaDataPtr =
        reinterpret_cast<detail::APMetaData*>(metaData_->GetData());

    // Write per-core slots
    for (uint32_t i = 0; i < AP_AIC_CORE_NUM; ++i) {
        metaDataPtr->coreSlots[i].fields[AIC_CORE_ENABLE_INDEX] = splitCoreEnable_[i];
        metaDataPtr->coreSlots[i].fields[BN2_START_PTR_INDEX] = splitBnStartIdx_[i];
        metaDataPtr->coreSlots[i].fields[GS1_START_PTR_INDEX] = splitGS1StartIdx_[i];
    }

    metaDataPtr->baseMetadata[USED_CORE_NUM_INDEX] = static_cast<int32_t>(splitUsedCoreNum_);
    metaDataPtr->baseMetadata[M_BASE_SIZE_INDEX] = static_cast<int32_t>(sOuterSize_);
    metaDataPtr->baseMetadata[S_INNER_SIZE_INDEX] = static_cast<int32_t>(sInnerSize_);
    metaDataPtr->baseMetadata[S1_OUTER_SIZE_INDEX] = static_cast<int32_t>(splitS1OuterSize_);
    metaDataPtr->baseMetadata[S1_SIZE_INDEX] = static_cast<int32_t>(s1Size_);
    metaDataPtr->baseMetadata[S2_SIZE_INDEX] = static_cast<int32_t>(s2Size_);
    metaDataPtr->baseMetadata[ACTUAL_LEN_Q_DIM_INDEX] =
        static_cast<int32_t>(actualSeqLengths_->GetTensorShape()->GetDimSize(0));
    metaDataPtr->baseMetadata[ACTUAL_LEN_KV_DIM_INDEX] =
        static_cast<int32_t>(actualSeqLengthsKv_->GetTensorShape()->GetDimSize(0));

    // multiSmaxsInnerLoopTimes: max across batches of CeilDiv(kvSeqLen, sInnerSize)
    uint32_t multiSmaxsInnerLoopTimes = 0;
    for (uint32_t i = 0; i < bSize_; i++) {
        uint32_t sInnerLoopTimes = static_cast<uint32_t>(
            CeilDiv(kvSeqLens_[i], static_cast<int64_t>(sInnerSize_)));
        multiSmaxsInnerLoopTimes = std::max(multiSmaxsInnerLoopTimes, sInnerLoopTimes);
    }
    metaDataPtr->baseMetadata[MULTI_SMAX_S_INNER_LOOP_TIMES_INDEX] =
        static_cast<int32_t>(multiSmaxsInnerLoopTimes);

    // actualCoreNums: vector cores = cube cores * CV_RATIO
    metaDataPtr->baseMetadata[ACTUAL_CORE_NUMS_INDEX] =
        static_cast<int32_t>(splitUsedCoreNum_ * AP_CV_RATIO);

    // totalSize, splitFactorSize, splitFactorTailSize
    uint32_t headNumSize = isDecode_ ? n2Size_ : n1Size_;
    int64_t sinnerBlockNum = CeilDiv(s2Size_, static_cast<int64_t>(sInnerSize_));
    int64_t totalSize = (sinnerBlockNum > 0) ?
        (totalBlockNumsOneHead_ / sinnerBlockNum) * static_cast<int64_t>(headNumSize) : 0;
    int64_t splitFactorSize = (splitUsedCoreNum_ > 0) ?
        CeilDiv(totalSize, static_cast<int64_t>(splitUsedCoreNum_)) : 0;
    int64_t splitFactorTailSize = (splitFactorSize > 0) ?
        ((totalSize % splitFactorSize != 0) ? (totalSize % splitFactorSize) : splitFactorSize) : 0;

    metaDataPtr->baseMetadata[TOTAL_SIZE_INDEX] = static_cast<int32_t>(totalSize);
    metaDataPtr->baseMetadata[SPLIT_FACTOR_SIZE_INDEX] = static_cast<int32_t>(splitFactorSize);
    metaDataPtr->baseMetadata[SPLIT_FACTOR_TAIL_SIZE_INDEX] = static_cast<int32_t>(splitFactorTailSize);

    return true;
}

REGISTER_CPU_KERNEL(kAttentionPioneerMetadata, AttentionPioneerMetadataKernel);
} // namespace aicpu
