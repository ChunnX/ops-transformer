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
 * \file flash_attn_metadata_aicpu.cpp
 * \brief
 */

#include "log.h"
#include "status.h"
#include "cust_op/cust_cpu_utils.h"
#include <cstdio>
#include <cmath>
// #include "../../common/aicpu/cpu_context_util.h"
#include "flash_attn_metadata_aicpu.h"

#define KERNEL_STATUS_OK 0
#define KERNEL_STATUS_PARAM_INVALID 1

namespace aicpu {
uint32_t FlashAttnMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    SplitResult splitRes{aicCoreNum_, aivCoreNum_};
    success = BalanceSchedule(splitRes) && GenMetaData(splitRes);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool FlashAttnMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    // input
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
    cuSeqlensKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensKv));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    sequsedKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedKv));
    // output
    metaData_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    bool requiredAttrs =
                GetAttrValue(ctx, "num_heads_q", numHeadsQ_) &&
                GetAttrValue(ctx, "num_heads_kv", numHeadsKv_) &&
                GetAttrValue(ctx, "head_dim", headDim_) &&
                GetAttrValue(ctx, "soc_version", socVersion_) &&
                GetAttrValue(ctx, "aic_core_num", aicCoreNum_) &&
                GetAttrValue(ctx, "aiv_core_num", aivCoreNum_);
    if (!requiredAttrs) {
        return false;
    }
    // attributes optional
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_kv", maxSeqlenKv_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "win_left", winLeft_);
    GetAttrValueOpt(ctx, "win_right", winRight_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "layout_kv", layoutKv_);
    GetAttrValueOpt(ctx, "layout_out", layoutOut_);
    return ParamsInit();
    // return true;
}

bool FlashAttnMetadataCpuKernel::ParamsInit()
{
    actualSeqLengths.resize(batchSize_, maxSeqlenQ_);
    actualSeqLengthsKV.resize(batchSize_, maxSeqlenKv_);
    auto cuSeqlensQ = (int32_t *)cuSeqlensQ_->GetData();
    auto cuSeqlensKv = (int32_t *)cuSeqlensKv_->GetData();
    if (cuSeqlensQ != nullptr) {
        for (uint32_t i = 0; i < batchSize_; ++i) {
            actualSeqLengths[i] = cuSeqlensQ[i+1] - cuSeqlensQ[i];
        }
    }
    if (cuSeqlensKv != nullptr) {
        for (uint32_t i = 0; i < batchSize_; ++i) {
            actualSeqLengthsKV[i] = cuSeqlensKv[i+1] - cuSeqlensKv[i];
        }
    }

    if (winLeft_ == -1) {
        winLeft_ = std::numeric_limits<int>::max();
    }
    if (winRight_ == -1) {
        winRight_ == std::numeric_limits<int>::max();
    }

    CalcS1S2BasicBlock();
    s1OuterSize = (maxSeqlenQ_ + s1BasicBlock - 1) / s1BasicBlock;
    if (maskMode_ == static_cast<int64_t>(MaskMode::NO_MASK)) {
        hasAttenMask = false;
    }
    return true;
}

void FlashAttnMetadataCpuKernel::GetPreNextTokensLeftUp(int64_t actualSeqLength, int64_t actualSeqLengthKV,
                                                        int64_t &winLeftLeftUp, int64_t &winRightLeftUp)
{
    if (maskMode_ == static_cast<int64_t>(MaskMode::RIGHT_DOWN_CAUSAL)) {
        winLeftLeftUp = SPARSE_MODE_INT_MAX;
        winRightLeftUp = actualSeqLengthKV - actualSeqLength;
    } else if (maskMode_ == static_cast<int64_t>(MaskMode::BAND)) {
        winLeftLeftUp = winLeft_ - actualSeqLengthKV + actualSeqLength;
        winRightLeftUp = winRight_ + actualSeqLengthKV - actualSeqLength;
    } else {
        winLeftLeftUp = winLeft_;
        winRightLeftUp = winRight_;
    }
    // winLeftLeftUp = SPARSE_MODE_INT_MAX;
    // winRightLeftUp = SPARSE_MODE_INT_MAX;
}

// 函数内部不处理prefix逻辑，prefix场景下入参需自行传入actualSeqLengthKV + prefix
void FixParamWithRowInvalid(int64_t &actualSeqLength, int64_t actualSeqLengthKV, int64_t &winLeftLeftUp,
                            int64_t &winRightLeftUp)
{
    // 若出现行无效，需要重新计算nexttokens，pretokens，actualseqlen，以便正确计算分核核数
    int64_t winRightError = (winRightLeftUp < 0) ? -winRightLeftUp : 0;
    winRightError = winRightError > actualSeqLength ? actualSeqLength : winRightError;
    int64_t winLeftError = 0;

    winLeftError = (actualSeqLength > actualSeqLengthKV + winLeftLeftUp) ?
                         (actualSeqLength - actualSeqLengthKV - winLeftLeftUp) :
                         0;

    winLeftError = winLeftError > actualSeqLength ? actualSeqLength : winLeftError;

    // 若出现上方行无效，需要重新计算nexttokens，pretokens，actualseqlen
    winRightLeftUp += winRightError;
    winLeftLeftUp -= winRightError;
    actualSeqLength -= winRightError;

    // 若出现下方行无效，需要重新计算actualseqlen
    actualSeqLength -= winLeftError;
}

int64_t SumOfArithmeticSeries(int64_t an, int64_t d)
{
    // 等差数列求和，an：等差数列第n项，d：等差数列公差
    if (d == 0) {
        return 0;
    }
    return (an > 0) ? (an % d + an) * (an / d + 1) / 2 : 0; // 2: 等差数列求和公式分母
}

int64_t GetCutBlockNums(int64_t blockSeqLengthKV, int64_t blockSeqLength, int64_t sInner, int64_t sOuter, int64_t token)
{
    // 以nextToken视角计算完全被nextToken掩盖的基本块数
    int64_t blockNums = 0;
    int64_t blockToken = token > 0 ? ((token + sInner - 1) / sInner * sInner) : (token / sInner * sInner);
    int64_t outDivIn = sOuter > sInner ? sOuter / sInner : 1;
    int64_t InDivOut = sInner > sOuter ? sInner / sOuter : 1;
    int64_t tolerance = 0;
    int64_t smallSize = 0;
    if (outDivIn >= 1) {
        tolerance = outDivIn;
        smallSize = sInner;
    } else {
        tolerance = InDivOut;
        smallSize = sOuter;
    }

    // nextToken与上边右边构成的大三角形
    int64_t innerCutBlockNums = (blockSeqLengthKV - blockToken) / smallSize - tolerance;
    blockNums += SumOfArithmeticSeries(innerCutBlockNums, tolerance);

    // nextToken与上边左边构成的左侧三角形，需要减去
    int64_t innerCutBlockLeftNums = -blockToken / smallSize - tolerance;
    blockNums -= SumOfArithmeticSeries(innerCutBlockLeftNums, tolerance);

    // nextToken与下边右边构成的下侧三角形，需要减去
    int64_t innerCutBlockDownNums = (blockSeqLengthKV - blockSeqLength - blockToken) / smallSize - tolerance;
    blockNums -= SumOfArithmeticSeries(innerCutBlockDownNums, tolerance);

    // nextToken与下边左边构成的小三角形，是前两个三角形的重叠部分，需要加上
    int64_t innerCutBlockLeftDownNums = (-blockToken - blockSeqLength) / smallSize - tolerance;
    blockNums += SumOfArithmeticSeries(innerCutBlockLeftDownNums, tolerance);
    return blockNums;
}


int64_t GetCalcBlockNumsOneHead(int64_t actualSeqLength, int64_t actualSeqLengthKV, uint32_t s1BasicBlock,
                                uint32_t s2BasicBlock, int64_t winLeftLeftUp, int64_t winRightLeftUp,
                                bool isAttenMaskUsed)
{
    if (!isAttenMaskUsed) {
        int64_t outerBlockNums = (actualSeqLength + s1BasicBlock - 1) / s1BasicBlock;
        int64_t innerBlockNums =
            (actualSeqLengthKV + s2BasicBlock - 1) / s2BasicBlock + (s2BasicBlock - 1) / s2BasicBlock;
        int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
        return toCalcBlockNums;
    } else {
        int64_t innerBlockNums =
            (actualSeqLengthKV + static_cast<int64_t>(s2BasicBlock) - 1) / static_cast<int64_t>(s2BasicBlock);
        int64_t blockSeqLengthKV = innerBlockNums * static_cast<int64_t>(s2BasicBlock);
        int64_t outerBlockNums =
            (actualSeqLength + static_cast<int64_t>(s1BasicBlock) - 1) / static_cast<int64_t>(s1BasicBlock);
        int64_t blockSeqLength = outerBlockNums * static_cast<int64_t>(s1BasicBlock);
        int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
        // 必须满足pretoken + nexttoken > 0，否则会减出小于0的块数，这里需要去除prefix影响
        toCalcBlockNums -= GetCutBlockNums(blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(s2BasicBlock),
                                           static_cast<int64_t>(s1BasicBlock), winRightLeftUp);
        toCalcBlockNums -=
            GetCutBlockNums(blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(s2BasicBlock),
                            static_cast<int64_t>(s1BasicBlock), blockSeqLengthKV - blockSeqLength + winLeftLeftUp);

        // prefix部分单独计算
        int64_t innerBlockNumsPrefix = (static_cast<int64_t>(s2BasicBlock) - 1) / static_cast<int64_t>(s2BasicBlock);
        int64_t blockSharedPrefix = innerBlockNumsPrefix * static_cast<int64_t>(s2BasicBlock);
        toCalcBlockNums += innerBlockNumsPrefix * outerBlockNums;
        toCalcBlockNums -= GetCutBlockNums(blockSharedPrefix, blockSeqLength, static_cast<int64_t>(s2BasicBlock),
                                           static_cast<int64_t>(s1BasicBlock), winRightLeftUp);
        toCalcBlockNums -=
            GetCutBlockNums(blockSharedPrefix, blockSeqLength, static_cast<int64_t>(s2BasicBlock),
                            static_cast<int64_t>(s1BasicBlock), blockSharedPrefix - blockSeqLength + winLeftLeftUp);

        return toCalcBlockNums;
    }
}

void FlashAttnMetadataCpuKernel::CalcS1S2BasicBlock()
{
    s1BasicBlock = NUM_128;
    s2BasicBlock = NUM_128;
}

void FlashAttnMetadataCpuKernel::CalcThresholdForS2Size()
{
    int64_t dBasicBlock = AlignUp(static_cast<int64_t>(headDim_), D_TEMPLATE_SPLIT_SIZE);
    int64_t dVBasicBlock = AlignUp(static_cast<int64_t>(headDim_), D_TEMPLATE_SPLIT_SIZE);

    int64_t l2CacheSizeForKandV = L2_CACHE_SIZE * NUM_1024 * NUM_1024;
    if (batchSize_ == 0 || numHeadsKv_ == 0 || dBasicBlock == 0 || dVBasicBlock == 0) {
        KERNEL_LOG_ERROR("The product of bSize[%ld], nSize[%ld], dSize[%ld] and dVSzie[%ld] cannot be zero.",
                         batchSize_, numHeadsKv_, dBasicBlock, dVBasicBlock);
        thresholdS2Size = std::numeric_limits<int64_t>::max();
        return;
    }

    int64_t typeSize = 2; // noquant
    // 当bn1比较大时，部分s比较小的场景，性能受限并非在L2cache资源的复用；
    int64_t bnSize = std::min(static_cast<int64_t>(batchSize_) * numHeadsKv_, static_cast<int64_t>(coreNum));
    thresholdS2Size = l2CacheSizeForKandV / (bnSize * (dBasicBlock + dVBasicBlock) * typeSize);
}

bool FlashAttnMetadataCpuKernel::IsUseSpliteCoreMode(MaskMode inputSparseMode)
{
    if (inputSparseMode == MaskMode::LEFT_UP_CAUSAL) {
        return std::min(maxSeqlenQ_, maxSeqlenKv_) >= thresholdS2Size;
    }

    if (inputSparseMode == MaskMode::RIGHT_DOWN_CAUSAL) {
        if (maxSeqlenQ_ <= maxSeqlenKv_) {
            return maxSeqlenKv_ >= thresholdS2Size;
        }
    }

    return false;
}

void FlashAttnMetadataCpuKernel::SetSplitCoreModeParam()
{
    splitCoreMode = SplitCoreMode::SPLIT_S1OUT_CUBE;
    return;

    splitCoreMode = SplitCoreMode::SPLIT_NBS_CUBE;

    CalcThresholdForS2Size();
    // 索引从0开始，需要将基本块个数减1
    if ((maskMode_ == static_cast<int64_t>(MaskMode::LEFT_UP_CAUSAL)) &&
        IsUseSpliteCoreMode(MaskMode::LEFT_UP_CAUSAL)) {
        firstFullLoadS1OuterIdx = CeilDivision(std::min(maxSeqlenQ_, maxSeqlenKv_), s1BasicBlock) - 1;
        splitCoreMode = SplitCoreMode::SPLIT_S1OUT_CUBE;
    } else if ((maskMode_ == static_cast<int64_t>(MaskMode::RIGHT_DOWN_CAUSAL)) &&
               IsUseSpliteCoreMode(MaskMode::RIGHT_DOWN_CAUSAL)) {
        firstFullLoadS1OuterIdx = s1OuterSize - 1;
        splitCoreMode = SplitCoreMode::SPLIT_S1OUT_CUBE;
    } else if (maskMode_ == static_cast<int64_t>(MaskMode::NO_MASK)) {
        if (!hasAttenMask && maxSeqlenKv_ >= thresholdS2Size) {
            firstFullLoadS1OuterIdx = -1;
            splitCoreMode = SplitCoreMode::SPLIT_S1OUT_CUBE;
        } else if (winLeft_ >= maxSeqlenQ_ && winRight_ == 0 && IsUseSpliteCoreMode(MaskMode::LEFT_UP_CAUSAL)) {
            firstFullLoadS1OuterIdx = CeilDivision(std::min(maxSeqlenQ_, maxSeqlenKv_), s1BasicBlock) - 1;
            splitCoreMode = SplitCoreMode::SPLIT_S1OUT_CUBE;
        }
    }
}

int64_t FlashAttnMetadataCpuKernel::CalcTotalSize()
{
    int64_t totalBlockNumsOneHead = 0; // The calculation amount of all sequences for a single head
    uint32_t multiSmaxsInnerLoopTimes = 0U;
    for (uint32_t sIdx = 0; sIdx < batchSize_; sIdx++) {
        int64_t actualSeqLengthsTmp = actualSeqLengths[sIdx]; // 用于存放减去行无效后，真实的actseqlen
        int64_t winLeftLeftUp = 0;
        int64_t winRightLeftUp = 0;
        GetPreNextTokensLeftUp(actualSeqLengths[sIdx], actualSeqLengthsKV[sIdx], winLeftLeftUp, winRightLeftUp);

        // 计算各sparse mode情况下，减去行无效后真实的actseqlen
        FixParamWithRowInvalid(actualSeqLengthsTmp, actualSeqLengthsKV[sIdx], winLeftLeftUp, winRightLeftUp);

        // sinner方向块数，prefix和origin是分开切的。
        uint32_t sInnerLoopTimes =
            (actualSeqLengthsKV[sIdx] + s2BasicBlock - 1) / s2BasicBlock + (s2BasicBlock - 1) / s2BasicBlock;
        multiSmaxsInnerLoopTimes = std::max(multiSmaxsInnerLoopTimes, sInnerLoopTimes);

        totalBlockNumsOneHead += GetCalcBlockNumsOneHead(actualSeqLengthsTmp, s2BasicBlock, s1BasicBlock, s2BasicBlock,
                                                         winLeftLeftUp, winRightLeftUp, true);


        int64_t outerBlockNums =
            (actualSeqLengthsTmp + static_cast<int64_t>(s1BasicBlock) - 1) / static_cast<int64_t>(s1BasicBlock);
        totalSize += outerBlockNums * numHeadsQ_;
        KERNEL_LOG_DEBUG("totalSize:%d, headNumSize:%d, outerBlockNums:%d, actualSeqLengthsTmp:%d, s1BasicBlock:%d, "
                         "winLeftLeftUp:%d, winRightLeftUp:%d \n",
                         totalSize, numHeadsQ_, outerBlockNums, actualSeqLengthsTmp, s1BasicBlock, winLeftLeftUp,
                         winRightLeftUp);
    }
    return totalSize;

    int64_t totalSize = batchSize_ * numHeadsQ_ * s1OuterSize;
    return totalSize;
}

bool FlashAttnMetadataCpuKernel::BalanceSchedule(SplitResult &splitRes)
{
    //// coreNum
    coreNum = aicCoreNum_;

    //// splitMode and firstFullLoadS1OuterIdx
    SetSplitCoreModeParam();

    //// totalsize
    CalcTotalSize();

    //// firstFullLoadS1OuterIdx
    if (maskMode_ == static_cast<int64_t>(MaskMode::LEFT_UP_CAUSAL)) {
        firstFullLoadS1OuterIdx = (std::min(maxSeqlenQ_, maxSeqlenKv_) + s1BasicBlock - 1) / s1BasicBlock - 1;
    } else if (maskMode_ == static_cast<int64_t>(MaskMode::RIGHT_DOWN_CAUSAL)) {
        firstFullLoadS1OuterIdx = s1OuterSize - 1;
        uint32_t s1SparseValidSize = maxSeqlenQ_;
        uint32_t s2SparseValidSize =
            std::min(AlignUp(static_cast<int32_t>(maxSeqlenKv_ - maxSeqlenQ_), static_cast<int32_t>(s1BasicBlock)),
                     static_cast<int32_t>(maxSeqlenKv_));
        KERNEL_LOG_DEBUG("maxSeqlenQ_:%d, maxSeqlenKv_:%d, s1BasicBlock:%d\n", maxSeqlenQ_, maxSeqlenKv_,
                         s1BasicBlock);
        KERNEL_LOG_DEBUG("s1SparseValidSize:%d, s2SparseValidSize:%d\n", s1SparseValidSize, s2SparseValidSize);
    } else if (maskMode_ == static_cast<int64_t>(MaskMode::NO_MASK) || static_cast<int64_t>(MaskMode::ALL_MASK)) {
        firstFullLoadS1OuterIdx = -1;
    }

    for (uint32_t cid = 0; cid < splitRes.usedCoreNum; cid++) {
        splitRes.s1out[cid] = totalSize;
    }

    return true;
}

bool FlashAttnMetadataCpuKernel::GenMetaData(SplitResult &splitRes)
{
    if (splitCoreMode == SplitCoreMode::SPLIT_S1OUT_CUBE) {
        optiling::detail::S1outFaMetaData *metaDataPtr = (optiling::detail::S1outFaMetaData *)(metaData_->GetData());
        metaDataPtr->splitCoreMode = splitCoreMode;
        for (uint32_t cid = 0; cid < aicCoreNum_; cid++) {
            if (cid >= splitRes.usedCoreNum) {
                metaDataPtr->s1outMetadata[cid][S1OUT_CORE_ENABLE_INDEX] = 0; // AIC disenable
                continue;
            }
            metaDataPtr->s1outMetadata[cid][S1OUT_CORE_ENABLE_INDEX] = 1;
            metaDataPtr->s1outMetadata[cid][S1OUT_M_STRIDE_INDEX] = splitRes.usedCoreNum;
            metaDataPtr->s1outMetadata[cid][S1OUT_M_NUM_INDEX] = splitRes.s1out[cid];
            metaDataPtr->s1outMetadata[cid][S1OUT_FIRST_FULL_LOAD_S1_OUTET_IDX_INDEX] = firstFullLoadS1OuterIdx;
        }
    }

    return true;
}

namespace {
static const char *kernelType = "FlashAttnMetadata";
REGISTER_CPU_KERNEL(kernelType, FlashAttnMetadataCpuKernel);
} // namespace

} // namespace aicpu