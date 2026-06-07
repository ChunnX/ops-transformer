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
 * \file sparse_flash_mla_metadata_aicpu.cpp
 * \brief
 */

#include "log.h"
#include "status.h"
#include "cust_op/cust_cpu_utils.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include "../../sparse_flash_mla/op_kernel/sparse_flash_mla_metadata.h"
#include "sparse_flash_mla_metadata_aicpu.h"

using namespace optiling;

namespace aicpu {
uint32_t
SparseFlashMlaMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    SplitResult splitRes {aicCoreNum_, aivCoreNum_};
    success = BalanceSchedule(splitRes) && GenMetadata(splitRes);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool SparseFlashMlaMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    // input
    actSeqLenQ_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenQ));
    actSeqLenOriKv_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenOriKv));
    actSeqLenCmpKv_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenCmpKv));
    seqUsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::seqUsedQ));
    seqUsedKv_ = ctx.Input(static_cast<uint32_t>(ParamId::seqUsedKv));
    // output
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metadata));

    bool requiredAttrs = GetAttrValue(ctx, "num_heads_q", queryHeadNum_) &&
                         GetAttrValue(ctx, "num_heads_kv", kvHeadNum_) &&
                         GetAttrValue(ctx, "head_dim", headDim_);
    if (!requiredAttrs) {
        return false;
    }

    // attributes optional
    GetAttrValueOpt(ctx, "soc_version", socVersion_);
    GetAttrValueOpt(ctx, "aic_core_num", aicCoreNum_);
    GetAttrValueOpt(ctx, "aiv_core_num", aivCoreNum_);
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", querySeqSize_);
    GetAttrValueOpt(ctx, "max_seqlen_kv", kvSeqSize_);
    GetAttrValueOpt(ctx, "ori_topk", oriTopK_);
    GetAttrValueOpt(ctx, "cmp_topk", cmpTopK_);
    GetAttrValueOpt(ctx, "cmp_ratio", cmpRatio_);
    GetAttrValueOpt(ctx, "ori_mask_mode", oriMaskMode_);
    GetAttrValueOpt(ctx, "cmp_mask_mode", cmpMaskMode_);
    GetAttrValueOpt(ctx, "ori_win_left", winLeft_);
    GetAttrValueOpt(ctx, "ori_win_right", winRight_);
    GetAttrValueOpt(ctx, "layout_q", layoutQuery_);
    GetAttrValueOpt(ctx, "layout_kv", layoutKv_);
    GetAttrValueOpt(ctx, "has_ori_kv", hasOriKv_);
    GetAttrValueOpt(ctx, "has_cmp_kv", hasCmpKv_);

    return ParamsInit();
}

int32_t SparseFlashMlaMetadataCpuKernel::GetQueryBatchSize()
{
    // 1. 濡傛灉seqUsedQ_ 浼犱簡锛屼娇鐢╯eqUsedQ_鑾峰彇BatchSize
    if (seqUsedQ_ != nullptr && seqUsedQ_->GetData() != nullptr) {
        if (seqUsedQ_->GetTensorShape() != nullptr) {
            return seqUsedQ_->GetTensorShape()->GetDimSize(0);
        }
    }
    // 2. seqUsedQ_ 娌′紶锛屽垽鏂?Layout
    if (layoutQuery_ == "TND") {
        // 濡傛灉鏄?TND锛屽皾璇曚娇鐢?actSeqLenQ_鑾峰彇BatchSize
        if (actSeqLenQ_ != nullptr && actSeqLenQ_->GetData() != nullptr) {
            if (actSeqLenQ_->GetTensorShape() != nullptr) {
                return actSeqLenQ_->GetTensorShape()->GetDimSize(0) - 1;
            }
        }
    }
    // 3. 濡傛灉涓嶆槸 TND锛屾垨鑰?actSeqLenQ_ 涓虹┖锛屼娇鐢╞atchSize_
    return batchSize_;
}

bool SparseFlashMlaMetadataCpuKernel::ParamsInit()
{
    batchSize_ = GetQueryBatchSize();
    auto mode = static_cast<SparseMode>(oriMaskMode_);
    if (mode == SparseMode::DEFAULT_MASK) {
        preToken_ = INT64_MAX;
        nextToken_ = INT64_MAX;
        attentionMode_ = 0;
    } else if (mode == SparseMode::RIGHT_DOWN_CAUSAL) {
        preToken_ = INT64_MAX;
        nextToken_ = 0;
        attentionMode_ = 1;
    } else {//SparseMode = 4
        preToken_ = (winLeft_ > -1) ? winLeft_ : INT64_MAX;
        nextToken_ = 0;
        attentionMode_ = 1;
    }
    isS1G_ = (layoutQuery_ == "BSND" || layoutQuery_ == "BSH" || layoutQuery_ == "TND");
    groupSize_ = queryHeadNum_ / kvHeadNum_;
    if (hasCmpKv_) {
        if (cmpTopK_ > 0) {
            isSCFA = true;
        } else {
            isCFA = true;
        }
    }
    if (isSCFA) {
        mBaseSize_ = groupSize_;
    } else {
        mBaseSize_ = 256U;
    }
    s2BaseSize_ = 512U;
    return true;
}

uint32_t SparseFlashMlaMetadataCpuKernel::GetS1SeqSize(uint32_t bIdx)
{
    // 1. 濡傛灉 seqUsedQ_ 浼犱簡锛岀洿鎺ヤ娇鐢?
    if (seqUsedQ_ != nullptr && seqUsedQ_->GetData() != nullptr) {
        const int32_t *seqUsedPtr = static_cast<const int32_t*>(seqUsedQ_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    // 2. seqUsedQ_ 娌′紶锛屽垽鏂?Layout
    if (layoutQuery_ == "TND") {
        // 濡傛灉鏄?TND锛屽皾璇曚娇鐢?actSeqLenQ_
        if (actSeqLenQ_ != nullptr && actSeqLenQ_->GetData() != nullptr) {
            const int32_t *s1Ptr =static_cast<const int32_t*>(actSeqLenQ_->GetData());
            return static_cast<uint32_t>(s1Ptr[bIdx + 1U] - s1Ptr[bIdx]);
        }
    }
    // 3. 濡傛灉涓嶆槸 TND锛屾垨鑰?actSeqLenQ_ 涓虹┖锛屼娇鐢?querySeqSize_
    return static_cast<uint32_t>(querySeqSize_);
}

uint32_t SparseFlashMlaMetadataCpuKernel::GetS2SeqSize(uint32_t bIdx)
{
    // 1. 濡傛灉 seqUsedKv_ 浼犱簡锛岀洿鎺ヤ娇鐢?
    if (seqUsedKv_ != nullptr && seqUsedKv_->GetData() != nullptr) {
        const int32_t *seqUsedPtr = static_cast<const int32_t*>(seqUsedKv_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    // 2. seqUsedKv_ 娌′紶锛屽垽鏂?Layout
    if (layoutKv_ == "TND") {
        // 濡傛灉鏄?TND锛屽皾璇曚娇鐢?actSeqLenOriKv_
        if (actSeqLenOriKv_ != nullptr && actSeqLenOriKv_->GetData() != nullptr) {
            const int32_t *s2Ptr = static_cast<const int32_t*>(actSeqLenOriKv_->GetData());
            return static_cast<uint32_t>(s2Ptr[bIdx + 1U] - s2Ptr[bIdx]);
        }
    }
    // 3. 濡傛灉涓嶆槸 TND锛屾垨鑰?actSeqLenOriKv_ 涓虹┖锛屼娇鐢?kvSeqSize_
    return static_cast<uint32_t>(kvSeqSize_);
}

uint32_t SparseFlashMlaMetadataCpuKernel::GetS1ValidSeqSize(uint32_t bIdx)
{
    uint32_t s1Size = GetS1SeqSize(bIdx);
    uint32_t s2Size = GetS2SeqSize(bIdx);
    if (s1Size > s2Size) {
        return s2Size;
    } else {
        return s1Size;
    }
}

void SparseFlashMlaMetadataCpuKernel::CalcSplitInfo(SplitContext &splitContext)
{
    // 璁＄畻姣忎釜batch鐨勫垏鍒嗭紝缁熻鏄惁涓虹┖batch锛岃褰曟渶鍚庢湁鏁坆atch锛堟瘡涓猙atch鐨勬瘡涓狽2鍒囧垎鏄竴鏍风殑锛?
    SplitInfo &splitInfo = splitContext.splitInfo;
    for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
        uint32_t s1Size = GetS1ValidSeqSize(bIdx);
        uint32_t s2Size = GetS2SeqSize(bIdx);
        splitInfo.s1GBaseNum[bIdx] = (s1Size * groupSize_ + (mBaseSize_ - 1U)) / mBaseSize_;
        splitInfo.s1GTailSize[bIdx] = (s1Size * groupSize_) % mBaseSize_;
        splitInfo.s2BaseNum[bIdx] = (s2Size + s2BaseSize_ - 1U) / s2BaseSize_;
        splitInfo.s2TailSize[bIdx] = s2Size % s2BaseSize_;
        if (splitInfo.s1GBaseNum[bIdx] != 0U && splitInfo.s2BaseNum[bIdx] != 0U) {
            splitInfo.isKvSeqAllZero = false;
        }
    }
    return;
}

int64_t SparseFlashMlaMetadataCpuKernel::CalcPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<SparseMode>(oriMaskMode_);
    if (mode == SparseMode::BAND) {
        return static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + preToken_;
    }
    return preToken_;
}

int64_t SparseFlashMlaMetadataCpuKernel::CalcNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<SparseMode>(oriMaskMode_);
    switch (mode) {
        case SparseMode::DEFAULT_MASK:
        case SparseMode::ALL_MASK:
        case SparseMode::LEFT_UP_CAUSAL:
            return nextToken_;
        case SparseMode::RIGHT_DOWN_CAUSAL:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        case SparseMode::BAND:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + nextToken_;
        default:
            return nextToken_;
    }
}

int64_t SparseFlashMlaMetadataCpuKernel::WinCalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t winAlignCoefM = 16U;
    uint32_t winAlignCoefS2 = 64U;
    uint32_t winAlignBasicM = (basicM + winAlignCoefM - 1U) >> 4U;    // 鎸塧lignCoefM瀵归綈锛屽悜涓婂彇鏁达紝4锛氱Щ浣嶆搷浣滃疄鐜伴櫎16
    uint32_t winAlignBasicS2 = (basicS2 + winAlignCoefS2 - 1U) >> 6U; // 鎸塧lignCoefS2瀵归綈锛屽悜涓婂彇鏁达紝6锛氱Щ浣嶆搷浣滃疄鐜伴櫎64
    return static_cast<int64_t>(6U * winAlignBasicM + 10U * winAlignBasicS2);                // 6锛歁杞寸郴鏁帮紝10锛歋2杞寸郴鏁?
}

int64_t SparseFlashMlaMetadataCpuKernel::CmpCalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t cmpAlignCoefM = 16U;
    uint32_t cmpAlignCoefS2 = 64U;
    uint32_t cmpAlignBasicM = (basicM + cmpAlignCoefM - 1U) >> 4U;    // 鎸塧lignCoefM瀵归綈锛屽悜涓婂彇鏁达紝4锛氱Щ浣嶆搷浣滃疄鐜伴櫎16
    uint32_t cmpAlignBasicS2 = (basicS2 + cmpAlignCoefS2 - 1U) >> 6U; // 鎸塧lignCoefS2瀵归綈锛屽悜涓婂彇鏁达紝6锛氱Щ浣嶆搷浣滃疄鐜伴櫎64
    return static_cast<int64_t>(6U * cmpAlignBasicM + 10U * cmpAlignBasicS2);                // 6锛歁杞寸郴鏁帮紝10锛歋2杞寸郴鏁?
}

void SparseFlashMlaMetadataCpuKernel::CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize,
                                                        uint32_t s1GTailSize, uint32_t winS2TailSize,
                                                        uint32_t cmpS2TailSize)
{
    // win閮ㄥ垎cost
    typeCost_[WIN_NORMAL_BLOCK][WIN_NORMAL_BLOCK] = WinCalcCost(s1NormalSize, s2NormalSize);
    typeCost_[WIN_TAIL_BLOCK][WIN_NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : WinCalcCost(s1GTailSize, s2NormalSize);
    typeCost_[WIN_NORMAL_BLOCK][WIN_TAIL_BLOCK] = (winS2TailSize == 0U) ? 0U : WinCalcCost(s1NormalSize, winS2TailSize);
    typeCost_[WIN_TAIL_BLOCK][WIN_TAIL_BLOCK] = (s1GTailSize == 0U || winS2TailSize == 0U) ? 0U :
        WinCalcCost(s1GTailSize, winS2TailSize);
    // cmp閮ㄥ垎cost
    if (hasCmpKv_) {
        typeCost_[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK] = CmpCalcCost(s1NormalSize, s2NormalSize);
        typeCost_[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : CmpCalcCost(s1GTailSize, s2NormalSize);
        typeCost_[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK] = (cmpS2TailSize == 0U) ? 0U :
            CmpCalcCost(s1NormalSize, cmpS2TailSize);
        typeCost_[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK] = (s1GTailSize == 0U || cmpS2TailSize == 0U) ? 0U :
            CmpCalcCost(s1GTailSize, cmpS2TailSize);
    }
}

Range<int64_t> SparseFlashMlaMetadataCpuKernel::CalcS2TokenRange(uint32_t s1GIdx, const BatchCache &batchCache)
{
    // actual seq == 0
    if (batchCache.s1Size == 0U || batchCache.s2Size == 0U) {
        return std::make_pair(0, 0);
    }

    // no mask
    if (!attentionMode_) { //attentionMaskFlag ?
        return std::make_pair(0, static_cast<int64_t>(batchCache.s2Size) - 1);
    }

    // 1. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
    int64_t s1GFirstToken = static_cast<int64_t>(s1GIdx) * static_cast<int64_t>(mBaseSize_);
    int64_t s1GLastToken = std::min(s1GFirstToken + static_cast<int64_t>(mBaseSize_),
        static_cast<int64_t>(batchCache.s1Size) * static_cast<int64_t>(groupSize_)) - 1;

    int64_t s1FirstToken = 0;
    int64_t s1LastToken = 0;
    if (isS1G_) {
        s1FirstToken = s1GFirstToken / static_cast<int64_t>(groupSize_);
        s1LastToken = s1GLastToken / static_cast<int64_t>(groupSize_);
    } else {
        if (s1GFirstToken / batchCache.s1Size == s1GLastToken / batchCache.s1Size) {
            // start and end locate in one G
            s1FirstToken = s1GFirstToken % static_cast<int64_t>(batchCache.s1Size);
            s1LastToken = s1GLastToken % static_cast<int64_t>(batchCache.s1Size);
        } else {
            // start and end locate in tow or more G, but working same as crossing a complete block
            s1FirstToken = 0;
            s1LastToken = batchCache.s1Size;
        }
    }

    int64_t s2FirstToken = s1FirstToken - batchCache.preTokenLeftUp;
    int64_t s2LastToken = s1LastToken + batchCache.nextTokenLeftUp;
    return std::make_pair(s2FirstToken, s2LastToken);
}

void SparseFlashMlaMetadataCpuKernel::CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext,
                                                         BatchCache &batchCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1ValidSeqSize(bIdx);
    batchCache.s2Size = GetS2SeqSize(bIdx);
    batchCache.preTokenLeftUp = CalcPreTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
    batchCache.nextTokenLeftUp = CalcNextTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
    isHighPorfScfa_ = isSCFA &&
        (batchCache.s1Size >= batchCache.s2Size || batchCache.s2Size - batchCache.s1Size + 1 < cmpTopK_ * cmpRatio_);
}

void SparseFlashMlaMetadataCpuKernel::CalcWinS1GCache(S1GCache &s1GCache, const SplitInfo &splitInfo)
{
    // 澶勭悊win閮ㄥ垎block淇℃伅
    if (s1GCache.winS2Start >= s1GCache.winS2End) {
        // win鑼冨洿鏃犳晥, 鍒欐暣涓猻1g琛岀瓑鏁堜负绌鸿
        s1GCache.winS1GBlock = 0;
        s1GCache.winS1GCost = 0;
        s1GCache.winS1GLastBlockCost = 0;
        s1GCache.winS1GNormalBlockCost = 0;
    } else {
        //璁＄畻 Win 鏂瑰悜 Block 鏁伴噺鍙?Cost
        s1GCache.winS1GBlock = s1GCache.winS2End - s1GCache.winS2Start;
        // 鍒ゆ柇 Win S2 鏂瑰悜鏄惁鍖呭惈灏惧潡
        uint32_t curWinTailS2Num = (s1GCache.winS2TailSize != 0U) ? 1U : 0U; // Updated check using local var
        uint32_t curWinNormalS2Num = s1GCache.winS1GBlock - curWinTailS2Num;
        if (s1GCache.s1GIdx == (splitInfo.s1GBaseNum[s1GCache.bIdx] - 1U) &&
            splitInfo.s1GTailSize[s1GCache.bIdx] != 0U) {
            s1GCache.winS1GCost = typeCost_[WIN_TAIL_BLOCK][WIN_NORMAL_BLOCK] * curWinNormalS2Num +
                typeCost_[WIN_TAIL_BLOCK][WIN_TAIL_BLOCK] * curWinTailS2Num;
            s1GCache.winS1GLastBlockCost = curWinTailS2Num > 0U ? typeCost_[WIN_TAIL_BLOCK][WIN_TAIL_BLOCK] :
                                            typeCost_[WIN_TAIL_BLOCK][WIN_NORMAL_BLOCK];
            s1GCache.winS1GNormalBlockCost = typeCost_[WIN_TAIL_BLOCK][WIN_NORMAL_BLOCK];
        }
        else {
            s1GCache.winS1GCost = typeCost_[WIN_NORMAL_BLOCK][WIN_NORMAL_BLOCK] * curWinNormalS2Num +
                typeCost_[WIN_NORMAL_BLOCK][WIN_TAIL_BLOCK] * curWinTailS2Num;
            s1GCache.winS1GLastBlockCost = curWinTailS2Num > 0U ? typeCost_[WIN_NORMAL_BLOCK][WIN_TAIL_BLOCK] :
                                            typeCost_[WIN_NORMAL_BLOCK][WIN_NORMAL_BLOCK];
            s1GCache.winS1GNormalBlockCost = typeCost_[WIN_NORMAL_BLOCK][WIN_NORMAL_BLOCK];
        }
    }
}

void SparseFlashMlaMetadataCpuKernel::CalcCmpS1GCache(S1GCache &s1GCache, const SplitInfo &splitInfo)
{
    // 澶勭悊cmp閮ㄥ垎block淇℃伅
    if (s1GCache.cmpS2Start >= s1GCache.cmpS2End) {
        // Cmp鑼冨洿鏃犳晥, Cost淇濇寔涓?0
        s1GCache.cmpS1GBlock = 0;
        s1GCache.cmpS1GCost = 0;
        s1GCache.cmpS1GLastBlockCost = 0;
        s1GCache.cmpS1GNormalBlockCost = 0;
    } else {
        //璁＄畻 cmp 鏂瑰悜 Block 鏁伴噺鍙?Cost
        s1GCache.cmpS1GBlock = s1GCache.cmpS2End - s1GCache.cmpS2Start;
        // 鍒ゆ柇 Cmp S2 鏂瑰悜鏄惁鍖呭惈灏惧潡
        uint32_t curCmpTailS2Num = (s1GCache.cmpS2TailSize != 0U) ? 1U : 0U;// Updated check using local var
        uint32_t curCmpNormalS2Num = s1GCache.cmpS1GBlock - curCmpTailS2Num;
        if (s1GCache.s1GIdx == (splitInfo.s1GBaseNum[s1GCache.bIdx] - 1U) &&
            splitInfo.s1GTailSize[s1GCache.bIdx] != 0U) {
            s1GCache.cmpS1GCost = typeCost_[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK] * curCmpNormalS2Num +
                typeCost_[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK] * curCmpTailS2Num;
            s1GCache.cmpS1GLastBlockCost = curCmpTailS2Num > 0U ? typeCost_[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK] :
                                                typeCost_[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK];
            s1GCache.cmpS1GNormalBlockCost = typeCost_[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK];
        } else {
            s1GCache.cmpS1GCost = typeCost_[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK] * curCmpNormalS2Num +
                typeCost_[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK] * curCmpTailS2Num;
            s1GCache.cmpS1GLastBlockCost = curCmpTailS2Num > 0U ? typeCost_[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK] :
                                             typeCost_[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK];
            s1GCache.cmpS1GNormalBlockCost = typeCost_[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK];
        }
    }
}

void SparseFlashMlaMetadataCpuKernel::CalcBlockRangeAndTailSize(Range<int64_t> &oriS2TokenRange,
                                                                    const BatchCache &batchCache,
                                                                    S1GCache &s1GCache)
{
    int64_t oriS2FirstToken = oriS2TokenRange.first;
    int64_t oriS2LastToken = oriS2TokenRange.second;
    // win閮ㄥ垎s2璧锋鍜宼ailSize
    if (oriS2FirstToken >= static_cast<int64_t>(batchCache.s2Size) || oriS2LastToken < 0 ||
            oriS2LastToken < oriS2FirstToken) {
        oriS2FirstToken = 0;
        oriS2LastToken = 0;
        s1GCache.winS2Start = 0;
        s1GCache.winS2End = 0;
        s1GCache.winS2TailSize = 0;
    } else {
        oriS2FirstToken = Clip(oriS2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
        oriS2LastToken = Clip(oriS2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
        s1GCache.winS2Start = 0;
        s1GCache.winS2End = (oriS2LastToken - oriS2FirstToken) / s2BaseSize_ + 1U;
        s1GCache.winS2TailSize = (oriS2LastToken - oriS2FirstToken + 1) % s2BaseSize_;
    }
    // cmp閮ㄥ垎s2璧锋鍜宼ailSize
    s1GCache.cmpS2Start = s1GCache.winS2End;
    // 璁＄畻CmpS2LastToken鐨勯暱搴?
    uint32_t cmpS2LastTokenSize = hasCmpKv_ ? (oriS2LastToken + 1) / cmpRatio_ : 0;
    uint32_t actCmpS2LastTokenSize = 0;
    if (isCFA) {
        actCmpS2LastTokenSize = cmpS2LastTokenSize;
    } else if (isSCFA) {
        // CmpS2LastToken涓巘opk鍙栨渶灏?
        actCmpS2LastTokenSize = std::min(cmpS2LastTokenSize,  static_cast<uint32_t>(cmpTopK_));
    }
    // 灏唗oken闀垮害杞寲涓簍oken绱㈠紩锛岀劧鍚庣敱token绱㈠紩璁＄畻s2绱㈠紩
    s1GCache.cmpS2End = (actCmpS2LastTokenSize == 0) ? s1GCache.cmpS2Start : s1GCache.cmpS2Start +
                        (actCmpS2LastTokenSize - 1) / s2BaseSize_ + 1U;
    // 鐢眛oken闀垮害璁＄畻cmpS2TailSize
    s1GCache.cmpS2TailSize = actCmpS2LastTokenSize % s2BaseSize_;
}

void SparseFlashMlaMetadataCpuKernel::GatherWinAndCmpCache(S1GCache &s1GCache)
{
    s1GCache.s2Start = (s1GCache.winS1GBlock > 0) ? s1GCache.winS2Start : s1GCache.cmpS2Start;
    if (s1GCache.cmpS1GBlock > 0) {
        s1GCache.s1GLastBlockCost = s1GCache.cmpS1GLastBlockCost;
        s1GCache.s2End = s1GCache.cmpS2End;
    } else {
        s1GCache.s1GLastBlockCost = s1GCache.winS1GLastBlockCost;
        s1GCache.s2End = s1GCache.winS2End;
    }
    s1GCache.s1GBlock = s1GCache.winS1GBlock + s1GCache.cmpS1GBlock;
    s1GCache.s1GCost = s1GCache.winS1GCost + s1GCache.cmpS1GCost;
}

void SparseFlashMlaMetadataCpuKernel::CalcS1GCache(uint32_t s1GIdx, const SplitContext &splitContext,
                                                       const BatchCache &batchCache, S1GCache &s1GCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    // 濡傛灉s1G鏄┖琛岋紝鍒欑洿鎺ヨ繑鍥?
    if (splitInfo.s1GBaseNum[batchCache.bIdx] == 0) {
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.winS1GNormalBlockCost = 0;
        s1GCache.winS1GLastBlockCost = 0;
        s1GCache.cmpS1GNormalBlockCost = 0;
        s1GCache.cmpS1GLastBlockCost = 0;
        s1GCache.s1GBlock = 0;
        s1GCache.s2Start = 0;
        s1GCache.cmpS2Start = 0;
        s1GCache.s2End = 0;
        return;
    }
    s1GCache.bIdx = batchCache.bIdx;
    s1GCache.s1GIdx = s1GIdx;
    // 璁＄畻ori_kv鐨則oken璧锋
    auto oriS2TokenRange = CalcS2TokenRange(s1GIdx, batchCache);
    // 璁＄畻win鍜宑mp閮ㄥ垎s2璧锋鍜宼ailSize
    CalcBlockRangeAndTailSize(oriS2TokenRange, batchCache, s1GCache);
    // Calculate CostTable locally
    CalcCostTable(mBaseSize_, s2BaseSize_, splitInfo.s1GTailSize[s1GCache.bIdx], s1GCache.winS2TailSize,
                  s1GCache.cmpS2TailSize);
    // 璁＄畻win鍜宑mp閮ㄥ垎鐨刢ost, block淇℃伅
    CalcWinS1GCache(s1GCache, splitInfo);
    CalcCmpS1GCache(s1GCache, splitInfo);
    // 姹囨€粀in鍜宑mp閮ㄥ垎鐨刢ost, block淇℃伅
    GatherWinAndCmpCache(s1GCache);
    if (isHighPorfScfa_) {
        if (queryHeadNum_ == 128 && cmpTopK_ == 1024) {
            int64_t oriS2LastToken = oriS2TokenRange.second;
            oriS2LastToken = Clip(oriS2LastToken, static_cast<int64_t>(0),
                static_cast<int64_t>(batchCache.s2Size - 1U));
            oriS2LastToken = std::min(static_cast<int64_t>(cmpTopK_) * cmpRatio_ - 1U, oriS2LastToken);
            uint32_t winAlignCoefM = 16U;
            uint32_t winAlignCoefS2 = 64U;
            uint32_t winAlignBasicM = (mBaseSize_ + winAlignCoefM - 1U) >> 4U;
            uint32_t winAlignBasicS2 = (oriS2LastToken + 1U + winAlignCoefS2 - 1U) >> 6U;
            s1GCache.s1GCost = static_cast<int64_t>(15U * winAlignBasicM + 5U * winAlignBasicS2);
            if (oriS2LastToken == (static_cast<int64_t>(cmpTopK_) * cmpRatio_ - 1U)) {
                s1GCache.s1GCost -= s1GCache.s1GCost / 21U;
            }
            if (oriS2LastToken + 1U < 2048U) {
                s1GCache.s1GCost -= s1GCache.s1GCost / 18U;
            }
            if (oriS2LastToken + 1U < 3392U && oriS2LastToken + 1U > 2048U) {
                s1GCache.s1GCost += s1GCache.s1GCost / 7U;
            }
        }
    }
}

void SparseFlashMlaMetadataCpuKernel::CalcBatchCost(uint32_t bIdx, const SplitContext &splitContext,
                                                        CostInfo &costInfo)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    costInfo.bN2CostOfEachBatch[bIdx] = 0;
    costInfo.bN2BlockOfEachBatch[bIdx] = 0U;
    costInfo.bN2LastBlockCostOfEachBatch[bIdx] = 0U;

    if (GetS1ValidSeqSize(bIdx) == 0U || GetS2SeqSize(bIdx) == 0U) {
        return;
    }

    BatchCache bCache;
    S1GCache s1GCache;
    CalcBatchCache(bIdx, splitContext, bCache);
    for (uint32_t s1GIdx = 0; s1GIdx < splitInfo.s1GBaseNum[bIdx]; s1GIdx++) {
        CalcS1GCache(s1GIdx, splitContext, bCache, s1GCache);
        costInfo.bN2CostOfEachBatch[bIdx] += s1GCache.s1GCost;
        costInfo.bN2BlockOfEachBatch[bIdx] += s1GCache.s1GBlock;
        // 鏇存柊鏈€澶1G琛屽紑閿€
        if (s1GCache.s1GCost > costInfo.maxS1GCost) {
            costInfo.maxS1GCost = s1GCache.s1GCost;
        }
        if(s1GCache.s1GBlock > 0){
            costInfo.bN2LastBlockCostOfEachBatch[bIdx] = s1GCache.s1GLastBlockCost;
        }
    }
}

void SparseFlashMlaMetadataCpuKernel::CalcCostInfo(SplitContext &splitContext)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    CostInfo &costInfo = splitContext.costInfo;

    if (splitInfo.isKvSeqAllZero) {
        costInfo.totalCost = 0;
        costInfo.totalBlockNum = 0U;
        return;
    }

    // 璁＄畻batch鐨勮礋杞藉苟璁板綍锛岀敤浜庢寜batch鍒嗛厤锛岄渶瑕佹寜琛岃绠楄捣姝㈢偣锛岀粺璁″潡鏁般€佽礋杞?
    for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
        CalcBatchCost(bIdx, splitContext, costInfo);
        costInfo.totalCost += costInfo.bN2CostOfEachBatch[bIdx] * kvHeadNum_;
        costInfo.totalBlockNum += costInfo.bN2BlockOfEachBatch[bIdx] * kvHeadNum_;
    }
}

void SparseFlashMlaMetadataCpuKernel::AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }
    const CostInfo &costInfo = splitContext.costInfo;
    if (isHighPorfScfa_) {
        if (queryHeadNum_ == 128 && cmpTopK_ == 1024) {
            while (assignContext.bN2Cost == 0 ||
                   assignContext.coreCache.costLimit - assignContext.coreCache.cost >
                   0.8L * static_cast<long double>(assignContext.bN2Cost)) {
                assignContext.coreCache.cost += assignContext.bN2Cost;
                assignContext.coreCache.block += assignContext.bN2Block;
                assignContext.curBN2Idx++;

                // to the end
                if (assignContext.curBN2Idx == batchSize_ * kvHeadNum_) {
                    assignContext.curS1GIdx = 0U;
                    assignContext.curS2Idx = 0U;
                    assignContext.isFinished = true;
                    return;
                }

                // next batch
                if (assignContext.curBN2Idx / kvHeadNum_ != assignContext.curBIdx) {
                    assignContext.curBIdx = assignContext.curBN2Idx / kvHeadNum_;
                    CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
                }

                assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
                assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
                assignContext.curS1GIdx = 0U;
                CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
                assignContext.curS2Idx = assignContext.s1GCache.s2Start;
            }
            return;
        }
    }
    while (assignContext.bN2Cost == 0 || IsWithinTolerance(assignContext.coreCache.costLimit,
        costInfo.bN2LastBlockCostOfEachBatch[assignContext.curBIdx] / FA_TOLERANCE_RATIO,
        assignContext.coreCache.cost + assignContext.bN2Cost)) {
        assignContext.coreCache.cost += assignContext.bN2Cost;
        assignContext.coreCache.block += assignContext.bN2Block;
        assignContext.curBN2Idx++;

        // to the end
        if (assignContext.curBN2Idx == batchSize_ * kvHeadNum_) {
            assignContext.curS1GIdx = 0U;
            assignContext.curS2Idx = 0U;
            assignContext.isFinished = true;
            return;
        }

        // next batch
        if (assignContext.curBN2Idx / kvHeadNum_ != assignContext.curBIdx) {
            assignContext.curBIdx = assignContext.curBN2Idx / kvHeadNum_;
            CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        }

        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
        assignContext.curS1GIdx = 0U;
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void SparseFlashMlaMetadataCpuKernel::AssignByRow(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }
    if (isHighPorfScfa_) {
        if (queryHeadNum_ == 128 && cmpTopK_ == 1024) {
            while (assignContext.coreCache.costLimit - assignContext.coreCache.cost >
                   0.8L * static_cast<long double>(assignContext.s1GCache.s1GCost)) {
                assignContext.coreCache.cost += assignContext.s1GCache.s1GCost;
                assignContext.coreCache.block += assignContext.s1GCache.s1GBlock;

                // 褰撳墠batch琚垎閰嶄竴琛屽嚭鍘伙紝鏇存柊鍓╀綑璐熻浇
                assignContext.bN2Cost = assignContext.bN2Cost > assignContext.s1GCache.s1GCost ?
                                        assignContext.bN2Cost - assignContext.s1GCache.s1GCost : 0;
                assignContext.bN2Block = assignContext.bN2Block > assignContext.s1GCache.s1GBlock ?
                                        assignContext.bN2Block - assignContext.s1GCache.s1GBlock : 0U;
                // 璁＄畻鏂颁竴琛岀殑淇℃伅
                do {
                    assignContext.curS1GIdx++;
                    CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache,
                                 assignContext.s1GCache);
                } while (assignContext.s1GCache.s1GBlock == 0);
                assignContext.curS2Idx = assignContext.s1GCache.s2Start;
            }
            return;
        }
    }
    while (IsWithinTolerance(assignContext.coreCache.costLimit,
        assignContext.s1GCache.s1GLastBlockCost / FA_TOLERANCE_RATIO,
        assignContext.coreCache.cost + assignContext.s1GCache.s1GCost)) {
        assignContext.coreCache.cost += assignContext.s1GCache.s1GCost;
        assignContext.coreCache.block += assignContext.s1GCache.s1GBlock;

        // 褰撳墠batch琚垎閰嶄竴琛屽嚭鍘伙紝鏇存柊鍓╀綑璐熻浇
        assignContext.bN2Cost = assignContext.bN2Cost > assignContext.s1GCache.s1GCost ?
                                assignContext.bN2Cost - assignContext.s1GCache.s1GCost : 0;
        assignContext.bN2Block = assignContext.bN2Block > assignContext.s1GCache.s1GBlock ?
                                assignContext.bN2Block - assignContext.s1GCache.s1GBlock : 0U;
        // 璁＄畻鏂颁竴琛岀殑淇℃伅
        do{
            assignContext.curS1GIdx++;
            CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        }while(assignContext.s1GCache.s1GBlock == 0);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

int64_t SparseFlashMlaMetadataCpuKernel::CalcCurBlockCost(AssignContext &assignContext)
{
    int64_t curCost = 0;
    if (assignContext.curS2Idx < assignContext.s1GCache.cmpS2Start) {
        curCost = assignContext.s1GCache.winS1GNormalBlockCost;
        if (assignContext.curS2Idx == (assignContext.s1GCache.cmpS2Start - 1U)) {
            curCost = assignContext.s1GCache.winS1GLastBlockCost;
        }
    } else {
        curCost = assignContext.s1GCache.cmpS1GNormalBlockCost;
        if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
            curCost = assignContext.s1GCache.cmpS1GLastBlockCost;
        }
    }
    return curCost;
}

void SparseFlashMlaMetadataCpuKernel::AssignByBlock(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished || !supportFd) {
        return;
    }

    int64_t curCost = CalcCurBlockCost(assignContext);

    while (IsWithinTolerance(assignContext.coreCache.costLimit, curCost / FA_TOLERANCE_RATIO,
            assignContext.coreCache.cost + curCost)) { // (costLimit - curCostOnCore) * FA_TOLERANCE_RATIO > curCost锛涜嚦灏戝垎閰?鍧?
        assignContext.coreCache.cost += curCost;
        assignContext.coreCache.block++;
        assignContext.curS2Idx++;
        // 褰撳墠batch琚垎閰嶄竴鍧楀嚭鍘伙紝鏇存柊鍓╀綑璐熻浇
        assignContext.bN2Cost = assignContext.bN2Cost - curCost;
        // 褰撳墠琛岃鍒嗛厤涓€鍧楀嚭鍘伙紝鏇存柊鍓╀綑璐熻浇
        assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
        assignContext.bN2Block--;
        assignContext.s1GCache.s1GBlock--;
        curCost = CalcCurBlockCost(assignContext);
    }
}

bool SparseFlashMlaMetadataCpuKernel::IsNeedRecordFDInfo(const AssignContext &assignContext,
                                                             const SplitResult &splitRes)
{
    // 鍒囧垎鐐瑰ぇ姒傜巼涓嶄細鍒氬ソ鍦ㄨ灏撅紝鍥犳婊炲悗澶勭悊褰掔害淇℃伅鐨勭粺璁★紝鍒颁笅涓€涓垏鍒嗙偣鍐嶅垽鏂槸鍚﹂渶瑕佸綊绾?
    // 鏍?鏃犻渶澶勭悊
    if (assignContext.curCoreIdx == 0U) {
        return false;
    }
    // 鏃犺法鏍歌锛屾棤闇€澶勭悊
    if (assignContext.curKvSplitPart <= 1U) {
        return false;
    }
    // 闇€瑕佸綊绾︾殑琛岃繕鏈鐞嗗畬
    if (assignContext.curBN2Idx == splitRes.bN2End[assignContext.curCoreIdx - 1U] &&
        assignContext.curS1GIdx == splitRes.gS1End[assignContext.curCoreIdx - 1U]) {
        return false;
    }
    return true;
}

void SparseFlashMlaMetadataCpuKernel::RecordFDInfo(const SplitContext &splitContext,
                                                       const AssignContext &assignContext,
                                                       SplitResult &result)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    // 闇€瑕佽绾︾殑琛屾槸涓婁竴涓牳鐨勫垏鍒嗙偣鎵€鍦ㄤ綅缃?
    uint32_t splitBIdx = result.bN2End[assignContext.curCoreIdx - 1U] / kvHeadNum_;
    uint32_t splitS1GIdx = result.gS1End[assignContext.curCoreIdx - 1U];
    uint32_t s1Size = GetS1ValidSeqSize(splitBIdx);

    // 璁＄畻褰掔害鏁版嵁鐨凢D鍧囪　鍒掑垎淇℃伅
    uint32_t curFdS1gSize = (splitS1GIdx == splitInfo.s1GBaseNum[splitBIdx] - 1U) ?
                            (s1Size * groupSize_ - splitS1GIdx * mBaseSize_) : mBaseSize_;
    // 璁板綍
    result.maxS2SplitNum = std::max(result.maxS2SplitNum, assignContext.curKvSplitPart);
    // 鑻ュ瓨鍦ㄥご褰掔害锛屽垯鍒囧垎鐐逛竴瀹氫负涓婁竴涓牳缁撴潫鐨勪綅缃?
    result.fdRes.fdBN2Idx[result.numOfFdHead] = result.bN2End[assignContext.curCoreIdx - 1U];
    result.fdRes.fdMIdx[result.numOfFdHead] = result.gS1End[assignContext.curCoreIdx - 1U];
    result.fdRes.fdWorkspaceIdx[result.numOfFdHead] = assignContext.preFdDataNum;
    result.fdRes.fdS2SplitNum[result.numOfFdHead] = assignContext.curKvSplitPart;
    result.fdRes.fdMSize[result.numOfFdHead] = curFdS1gSize;
    result.numOfFdHead++;
}

void SparseFlashMlaMetadataCpuKernel::UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    const CostInfo &costInfo = splitContext.costInfo;

    bool UpdateS1G = false;
    bool UpdateBatch = false;

    // Update S2
    if (assignContext.curS2Idx >= assignContext.s1GCache.s2End) {    // 杈圭晫assignInfo.s2End鏄彇涓嶅埌鐨勫紑鍖洪棿
        assignContext.curS2Idx = 0U;
        assignContext.curS1GIdx++;
        UpdateS1G = true;
    }

    // Update S1G
    if (assignContext.curS1GIdx >= splitInfo.s1GBaseNum[assignContext.curBIdx]) {
        assignContext.curS1GIdx = 0U;
        assignContext.curBN2Idx++;
    }

    // Update Batch
    if (assignContext.curBN2Idx == batchSize_ * kvHeadNum_) {  // 鎵€鏈夎礋杞藉叏閮ㄥ垎閰嶅畬锛岃缃渶鍚庝竴涓牳鐨勫彸寮€鍖洪棿锛岃繑鍥?
        assignContext.curS1GIdx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.isFinished = true;
        return;
    }

    if (assignContext.curBN2Idx / kvHeadNum_ != assignContext.curBIdx) {
        assignContext.curBIdx = assignContext.curBN2Idx / kvHeadNum_;
        assignContext.curS1GIdx = 0U;
        UpdateBatch = true;
        UpdateS1G = true;
    }

    // Update Cache
    if (UpdateBatch) {
        CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    }
    if (UpdateS1G) {
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = (supportFd) ? assignContext.s1GCache.winS2Start : 0;
    }
}

void SparseFlashMlaMetadataCpuKernel::ForceAssign(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = CalcCurBlockCost(assignContext);

    assignContext.coreCache.cost += curCost;
    assignContext.coreCache.block++;
    assignContext.curS2Idx++;
    // 褰撳墠batch琚垎閰嶄竴鍧楀嚭鍘伙紝鏇存柊鍓╀綑璐熻浇
    assignContext.bN2Cost = assignContext.bN2Cost - curCost;
    assignContext.bN2Block--;
    // 褰撳墠琛岃鍒嗛厤涓€鍧楀嚭鍘伙紝鏇存柊鍓╀綑璐熻浇
    assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
    assignContext.s1GCache.s1GBlock--;
    UpdateCursor(splitContext, assignContext);
}

void SparseFlashMlaMetadataCpuKernel::AssignBlocksToCore(const SplitContext &splitContext,
    AssignContext &assignContext, SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;
    result.firstFdDataWorkspaceIdx[assignContext.curCoreIdx] = assignContext.preFdDataNum +
        assignContext.curKvSplitPart - 1U;
    int64_t avgCost = assignContext.unassignedCost / (aicCoreNum_ - assignContext.curCoreIdx);
    assignContext.coreCache = {};
    if (!supportFd) {
        assignContext.coreCache.costLimit = std::max(avgCost, costInfo.maxS1GCost);
    } else {
        assignContext.coreCache.costLimit = avgCost;
    }
    // 1銆佹寜鏁碽atch鍒嗛厤
    AssignByBatch(splitContext, assignContext);
    // 2銆佹寜琛屽垎閰?
    AssignByRow(splitContext, assignContext);
    // 3銆佹寜鍧楀垎閰?
    AssignByBlock(splitContext, assignContext);
    // 4銆佸己鍒跺垎閰?
    if (assignContext.coreCache.block == 0 && supportFd) {
        ForceAssign(splitContext, assignContext);
    }
    result.bN2End[assignContext.curCoreIdx] = assignContext.curBN2Idx;
    result.gS1End[assignContext.curCoreIdx] = assignContext.curS1GIdx;
    result.s2End[assignContext.curCoreIdx] = assignContext.curS2Idx;
    result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);
    assignContext.unassignedCost -= assignContext.coreCache.cost;
    // 瀵逛箣鍓嶇殑褰掔害淇℃伅杩涜璁板綍骞舵竻鐞?
    if (IsNeedRecordFDInfo(assignContext, result)) {
        RecordFDInfo(splitContext, assignContext, result);
        assignContext.preFdDataNum += assignContext.curKvSplitPart;
        assignContext.curKvSplitPart = 1U;
    }
    // 鏇存柊S2鍒囧垎淇℃伅
    if (assignContext.curS2Idx > assignContext.s1GCache.s2Start &&
        assignContext.curS2Idx <= assignContext.s1GCache.s2End) {
        assignContext.curKvSplitPart++;
    }
}

void SparseFlashMlaMetadataCpuKernel::CalcSplitPlan(int64_t costLimit, const SplitContext &splitContext,
                                                        SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;

    if (aicCoreNum_ == 0U) {
        return;
    }
    result.maxCost = 0U;
    result.usedCoreNum = 0U;

    AssignContext assignContext {};
    assignContext.curBIdx = 0U;
    assignContext.curS1GIdx = 0U;
    assignContext.unassignedCost = costInfo.totalCost;
    assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
    assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
    CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
    assignContext.curS2Idx = assignContext.s1GCache.s2Start;

    for (uint32_t i = 0; i < aicCoreNum_; ++i) {
        if (result.maxCost > costLimit) {
            return;
        }
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
            break;
        }
        assignContext.curCoreIdx = i;
        AssignBlocksToCore(splitContext, assignContext, result);
    }
    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

void SparseFlashMlaMetadataCpuKernel::SplitFD(SplitResult &splitRes)
{
    // 璁＄畻FD鐨勬€绘暟鎹噺
    uint64_t totalFDLoad = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        totalFDLoad += splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i];
    }
    // 璁＄畻姣忎釜鏍稿鐞嗙殑load
    uint64_t averageLoad = (totalFDLoad + aivCoreNum_ - 1U) / aivCoreNum_; // 鍚戜笂鍙栨暣锛岄伩鍏嶆牳璐熻浇涓?
    uint32_t curCoreIndex = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        // 璁＄畻褰撳墠褰掔害浠诲姟鎵€鐢ㄦ牳鏁帮紝鍚戜笅鍙栨暣锛岄伩鍏嶄娇鐢ㄦ牳鏁拌秴鍑烘€绘牳鏁?
        uint32_t curFDVectorNum = splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i] / averageLoad;
        curFDVectorNum = std::max(1U, curFDVectorNum);
        // 璁＄畻褰撳墠褰掔害浠诲姟姣忎釜鏍哥殑琛屾暟锛屽悜涓婂彇鏁达紝閬垮厤琛屾暟涓?
        uint32_t curAveMSize = (splitRes.fdRes.fdMSize[i] + curFDVectorNum - 1U) / curFDVectorNum;
        curFDVectorNum = (splitRes.fdRes.fdMSize[i] + curAveMSize -1U)/ curAveMSize;
        for (uint32_t vid = 0; vid < curFDVectorNum; vid++) {
            splitRes.fdRes.fdIdx[curCoreIndex] = i;
            splitRes.fdRes.fdMStart[curCoreIndex] = vid * curAveMSize;
            splitRes.fdRes.fdMNum[curCoreIndex] =
                (vid < curFDVectorNum - 1) ? curAveMSize : (splitRes.fdRes.fdMSize[i] - vid * curAveMSize);
            curCoreIndex++;
        }
    }
    splitRes.fdRes.fdUsedVecNum = curCoreIndex;
}

bool SparseFlashMlaMetadataCpuKernel::BalanceSchedule(SplitResult &splitRes)
{
    SplitContext splitContext(batchSize_);
    // 1銆佸垝鍒嗗熀鏈潡锛岀粺璁′俊鎭?
    CalcSplitInfo(splitContext);
    // 鍏ㄧ┖case
    if (splitContext.splitInfo.isKvSeqAllZero) {
        splitRes.usedCoreNum = 1U;
        splitRes.bN2End[0] = batchSize_ * kvHeadNum_;
        splitRes.gS1End[0] = 0U;
        splitRes.s2End[0] = 0U;
        return true;
    }
    CalcCostInfo(splitContext);
    // 2銆佽幏鍙栨瘡涓牳鐨勫垎閰嶆柟妗?
    splitRes.maxCost = INT64_MAX;
    splitRes.usedCoreNum = 1U;

    CalcSplitPlan(splitRes.maxCost, splitContext, splitRes);
    // 3銆佸瓨鍦‵D浠诲姟锛屽FD杩涜璐熻浇鍧囪　鍒嗛厤
    if (splitRes.numOfFdHead > 0U) {
        SplitFD(splitRes);
    }
    splitRes.usedCoreNum = std::max(splitRes.usedCoreNum, 1U);  // 鑷冲皯浣跨敤1涓猚ore
    return true;
}

bool SparseFlashMlaMetadataCpuKernel::GenMetadata(SplitResult &splitRes)
{
    auto* metadataPtr = reinterpret_cast<optiling::detail::SmlaMetaData*>(metadata_->GetData());
    if (metadataPtr == nullptr) {
        return false;
    }
    std::memset(metadataPtr, 0, sizeof(optiling::detail::SmlaMetaData));
    // FA Metadata Generate
    for (size_t i = 0; i < aicCoreNum_; ++i) {
        if (i >= splitRes.usedCoreNum) {
            metadataPtr->faMetadata[i][FA_CORE_ENABLE_INDEX] = 0; // AIC disenable
            continue;
        }
        metadataPtr->faMetadata[i][FA_CORE_ENABLE_INDEX] = 1; // AIC enable
        // FA START
        metadataPtr->faMetadata[i][FA_BN2_START_INDEX] = i == 0 ? 0 : splitRes.bN2End[i-1];
        metadataPtr->faMetadata[i][FA_M_START_INDEX] = i == 0 ? 0 : splitRes.gS1End[i-1];
        metadataPtr->faMetadata[i][FA_S2_START_INDEX] = i == 0 ? 0 : splitRes.s2End[i-1];
        // FA END
        metadataPtr->faMetadata[i][FA_BN2_END_INDEX] = splitRes.bN2End[i];
        metadataPtr->faMetadata[i][FA_M_END_INDEX] = splitRes.gS1End[i];
        metadataPtr->faMetadata[i][FA_S2_END_INDEX] = splitRes.s2End[i];
        // firstFdData
        metadataPtr->faMetadata[i][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = splitRes.firstFdDataWorkspaceIdx[i];
        metadataPtr->faMetadata[i][FA_S2_MAX_NUM] = splitRes.maxS2SplitNum;
    }

    // FD Metadata Generate
    for (size_t i = 0; i < aivCoreNum_; ++i) {
        if (i >= splitRes.fdRes.fdUsedVecNum) {
            metadataPtr->fdMetadata[i][FD_CORE_ENABLE_INDEX] = 0; // AIV disenable
            continue;
        }
        metadataPtr->fdMetadata[i][FD_CORE_ENABLE_INDEX] = 1; // AIV enable
        uint32_t curFdIdx = splitRes.fdRes.fdIdx[i];
        metadataPtr->fdMetadata[i][FD_BN2_IDX_INDEX] = splitRes.fdRes.fdBN2Idx[curFdIdx];
        metadataPtr->fdMetadata[i][FD_M_IDX_INDEX] = splitRes.fdRes.fdMIdx[curFdIdx];
        metadataPtr->fdMetadata[i][FD_WORKSPACE_IDX_INDEX] = splitRes.fdRes.fdWorkspaceIdx[curFdIdx];
        metadataPtr->fdMetadata[i][FD_WORKSPACE_NUM_INDEX] = splitRes.fdRes.fdS2SplitNum[curFdIdx];
        metadataPtr->fdMetadata[i][FD_M_START_INDEX] = splitRes.fdRes.fdMStart[i];
        metadataPtr->fdMetadata[i][FD_M_NUM_INDEX] = splitRes.fdRes.fdMNum[i];
    }
    return true;
}
namespace {
    static const char *kernelType = "SparseFlashMlaMetadata";
    REGISTER_CPU_KERNEL(kernelType, SparseFlashMlaMetadataCpuKernel);
}

}; // namespace aicpu
