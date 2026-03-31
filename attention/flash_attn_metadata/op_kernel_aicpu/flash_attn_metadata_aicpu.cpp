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

#define KERNEL_STATUS_OK      0
#define KERNEL_STATUS_PARAM_INVALID 1

namespace aicpu {
uint32_t FlashAttnMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    KERNEL_LOG_DEBUG("success into Prepare");
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    SplitResult splitRes {aicCoreNum_, aivCoreNum_};
    KERNEL_LOG_DEBUG("success into BalanceSchedule");
    success = BalanceSchedule(splitRes) && GenMetaData(splitRes);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool FlashAttnMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
        // input
    actualSeqLengths_ = ctx.Input(static_cast<uint32_t>(ParamId::actualSeqLengths));
    actualSeqLengthsKV_ = ctx.Input(static_cast<uint32_t>(ParamId::actualSeqLengthsKV));
    // output
    metaData_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    bool requiredAttrs = GetAttrValue(ctx, "num_heads_q", numHeadsQ_) &&
                        GetAttrValue(ctx, "num_heads_kv", numHeadsKv_) &&
                        GetAttrValue(ctx, "head_dim", headDim_) &&
                        GetAttrValue(ctx, "sparse_mode", sparseMode_) &&
                        GetAttrValue(ctx, "soc_version", socVersion_) &&
                        GetAttrValue(ctx, "aic_core_num", aicCoreNum_) &&
                        GetAttrValue(ctx, "aiv_core_num", aivCoreNum_);
    if (!requiredAttrs) {
        return false;
    }
    KERNEL_LOG_DEBUG("success into prepare : %d %d\n",aicCoreNum_, aivCoreNum_);
    // attributes optional
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_kv", maxSeqlenKv_);
    // return (ParamsCheck() && ParamsInit());
    return ParamsInit();
    // return true;
}

bool FlashAttnMetadataCpuKernel::ParamsInit()
{
    size_t seqLens = actualSeqLengths_->GetTensorShape()->GetDimSize(0);
    actualSeqLengths = (int32_t *)actualSeqLengths_->GetData();

    size_t kvSeqLens = actualSeqLengthsKV_->GetTensorShape()->GetDimSize(0);
    actualSeqLengthsKV = (int32_t *)actualSeqLengthsKV_->GetData();

    CalcS1S2BasicBlock();
    s1OuterSize = (maxSeqlenQ_ + s1BasicBlock - 1) / s1BasicBlock;
    if (sparseMode_ == static_cast<int64_t>(SparseMode::NO_MASK)) {
        hasAttenMask = false;
    }
    return true;
}


void FlashAttnMetadataCpuKernel::GetPreNextTokensLeftUp(int64_t actualSeqLength, int64_t actualSeqLengthKV, int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp) 
{
    if (sparseMode_ == static_cast<int64_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        preTokensLeftUp = SPARSE_MODE_INT_MAX;
        nextTokensLeftUp = actualSeqLengthKV - actualSeqLength;
    } else if (sparseMode_ == static_cast<int64_t>(SparseMode::BAND)) {
        preTokensLeftUp = preTokens_ - actualSeqLengthKV + actualSeqLength;
        nextTokensLeftUp = nextTokens_ + actualSeqLengthKV - actualSeqLength;
    } else {
        preTokensLeftUp = preTokens_;
        nextTokensLeftUp = nextTokens_;
    }
    // preTokensLeftUp = SPARSE_MODE_INT_MAX;
    // nextTokensLeftUp = SPARSE_MODE_INT_MAX;
}

// 函数内部不处理prefix逻辑，prefix场景下入参需自行传入actualSeqLengthKV + prefix
void FixParamWithRowInvalid(int64_t& actualSeqLength, int64_t actualSeqLengthKV,
    int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp) 
{
    // 若出现行无效，需要重新计算nexttokens，pretokens，actualseqlen，以便正确计算分核核数
    int64_t nextTokensError = (nextTokensLeftUp < 0) ? -nextTokensLeftUp : 0;
    nextTokensError = nextTokensError > actualSeqLength ? actualSeqLength : nextTokensError;
    int64_t preTokensError = 0;

    preTokensError = (actualSeqLength > actualSeqLengthKV + preTokensLeftUp) ?
        (actualSeqLength - actualSeqLengthKV - preTokensLeftUp) : 0;

    preTokensError = preTokensError > actualSeqLength ? actualSeqLength : preTokensError;

    // 若出现上方行无效，需要重新计算nexttokens，pretokens，actualseqlen
    nextTokensLeftUp += nextTokensError;
    preTokensLeftUp -= nextTokensError;
    actualSeqLength -= nextTokensError;

    // 若出现下方行无效，需要重新计算actualseqlen
    actualSeqLength -= preTokensError;
}

int64_t SumOfArithmeticSeries(int64_t an, int64_t d)
{
    // 等差数列求和，an：等差数列第n项，d：等差数列公差
    if (d == 0) {
        return 0;
    }
    return (an > 0) ? (an % d + an) * (an / d + 1) / 2 : 0; // 2: 等差数列求和公式分母
}

int64_t GetCutBlockNums(int64_t blockSeqLengthKV, int64_t blockSeqLength,
        int64_t sInner, int64_t sOuter, int64_t token)
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


int64_t GetCalcBlockNumsOneHead(int64_t actualSeqLength, int64_t actualSeqLengthKV,
    uint32_t s1BasicBlock, uint32_t s2BasicBlock, int64_t preTokensLeftUp, int64_t nextTokensLeftUp, bool isAttenMaskUsed)
{
    if (!isAttenMaskUsed) {
        int64_t outerBlockNums = (actualSeqLength + s1BasicBlock - 1) / s1BasicBlock;
        int64_t innerBlockNums = (actualSeqLengthKV + s2BasicBlock - 1) / s2BasicBlock + (s2BasicBlock - 1) / s2BasicBlock; 
        int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
        return toCalcBlockNums;
    } else {
        int64_t innerBlockNums = (actualSeqLengthKV + static_cast<int64_t>(s2BasicBlock) - 1) /
            static_cast<int64_t>(s2BasicBlock);
        int64_t blockSeqLengthKV = innerBlockNums * static_cast<int64_t>(s2BasicBlock);
        int64_t outerBlockNums = (actualSeqLength + static_cast<int64_t>(s1BasicBlock) - 1) /
            static_cast<int64_t>(s1BasicBlock);
        int64_t blockSeqLength = outerBlockNums * static_cast<int64_t>(s1BasicBlock);
        int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
        // 必须满足pretoken + nexttoken > 0，否则会减出小于0的块数，这里需要去除prefix影响
        toCalcBlockNums -= GetCutBlockNums(blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(s2BasicBlock),
            static_cast<int64_t>(s1BasicBlock), nextTokensLeftUp);
        toCalcBlockNums -= GetCutBlockNums(blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(s2BasicBlock),
            static_cast<int64_t>(s1BasicBlock), blockSeqLengthKV - blockSeqLength + preTokensLeftUp);

        // prefix部分单独计算
        int64_t innerBlockNumsPrefix = (static_cast<int64_t>(s2BasicBlock) - 1) /
            static_cast<int64_t>(s2BasicBlock);
        int64_t blockSharedPrefix = innerBlockNumsPrefix * static_cast<int64_t>(s2BasicBlock);
        toCalcBlockNums += innerBlockNumsPrefix * outerBlockNums;
        toCalcBlockNums -= GetCutBlockNums(blockSharedPrefix, blockSeqLength, static_cast<int64_t>(s2BasicBlock),
            static_cast<int64_t>(s1BasicBlock), nextTokensLeftUp);
        toCalcBlockNums -= GetCutBlockNums(blockSharedPrefix, blockSeqLength, static_cast<int64_t>(s2BasicBlock),
            static_cast<int64_t>(s1BasicBlock), blockSharedPrefix - blockSeqLength + preTokensLeftUp);
 
        return toCalcBlockNums;
    }
}

void FlashAttnMetadataCpuKernel::CalcS1S2BasicBlock()
{
    s1BasicBlock = NUM_128;
    s2BasicBlock = NUM_128;
}

void FlashAttnMetadataCpuKernel::CalcThresholdForS2Size() {
    int64_t dBasicBlock = AlignUp(static_cast<int64_t>(headDim_) + ropeHeadDim_, D_TEMPLATE_SPLIT_SIZE);
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

bool FlashAttnMetadataCpuKernel::IsUseSpliteCoreMode(SparseMode inputSparseMode) {
    if (inputSparseMode == SparseMode::LEFT_UP_CAUSAL) {
        return std::min(maxSeqlenQ_, maxSeqlenKv_) >= thresholdS2Size;
    }

    if (inputSparseMode == SparseMode::RIGHT_DOWN_CAUSAL) {
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
    if ((sparseMode_ == static_cast<int64_t>(SparseMode::LEFT_UP_CAUSAL)) &&
        IsUseSpliteCoreMode(SparseMode::LEFT_UP_CAUSAL)) {
        firstFullLoadS1OuterIdx = CeilDivision(std::min(maxSeqlenQ_, maxSeqlenKv_), s1BasicBlock) - 1;
        splitCoreMode = SplitCoreMode::SPLIT_S1OUT_CUBE;
    } else if ((sparseMode_ == static_cast<int64_t>(SparseMode::RIGHT_DOWN_CAUSAL)) &&
        IsUseSpliteCoreMode(SparseMode::RIGHT_DOWN_CAUSAL)) {
        firstFullLoadS1OuterIdx = s1OuterSize - 1;
        splitCoreMode = SplitCoreMode::SPLIT_S1OUT_CUBE;
    } else if (sparseMode_ == static_cast<int64_t>(SparseMode::NO_MASK)) {
        if (!hasAttenMask && maxSeqlenKv_ >= thresholdS2Size) {
            firstFullLoadS1OuterIdx = -1;
            splitCoreMode = SplitCoreMode::SPLIT_S1OUT_CUBE;
        } else if (preTokens_ >= maxSeqlenQ_ && nextTokens_ == 0 && IsUseSpliteCoreMode(SparseMode::LEFT_UP_CAUSAL)) {
            firstFullLoadS1OuterIdx = CeilDivision(std::min(maxSeqlenQ_, maxSeqlenKv_), s1BasicBlock) - 1;
            splitCoreMode = SplitCoreMode::SPLIT_S1OUT_CUBE;
        }
    }
}

int64_t FlashAttnMetadataCpuKernel::CalcTotalSize()
{
    uint32_t headNumSize = numHeadsQ_;
    int64_t qS = maxSeqlenQ_;
    int64_t S2 = maxSeqlenKv_;
    int64_t totalBlockNumsOneHead = 0; // The calculation amount of all sequences for a single head
    uint32_t multiSmaxsInnerLoopTimes = 0U;
    for (uint32_t sIdx = 0; sIdx < batchSize_; sIdx++) {
        int64_t actualSeqLengthsTmp = actualSeqLengths[sIdx]; // 用于存放减去行无效后，真实的actseqlen
        int64_t preTokensLeftUp = 0;
        int64_t nextTokensLeftUp = 0;
        GetPreNextTokensLeftUp(actualSeqLengths[sIdx], actualSeqLengthsKV[sIdx],
            preTokensLeftUp, nextTokensLeftUp);

        // 计算各sparse mode情况下，减去行无效后真实的actseqlen
        FixParamWithRowInvalid(actualSeqLengthsTmp, actualSeqLengthsKV[sIdx] , preTokensLeftUp, nextTokensLeftUp);

        // sinner方向块数，prefix和origin是分开切的。
        uint32_t sInnerLoopTimes = (actualSeqLengthsKV[sIdx] + s2BasicBlock - 1) / s2BasicBlock +
            (s2BasicBlock - 1) / s2BasicBlock;
        multiSmaxsInnerLoopTimes = std::max(multiSmaxsInnerLoopTimes, sInnerLoopTimes);

        totalBlockNumsOneHead += GetCalcBlockNumsOneHead(actualSeqLengthsTmp, s2BasicBlock, s1BasicBlock,
            s2BasicBlock, preTokensLeftUp, nextTokensLeftUp, true);


        int64_t outerBlockNums = (actualSeqLengthsTmp + static_cast<int64_t>(s1BasicBlock) - 1) /
            static_cast<int64_t>(s1BasicBlock);
        totalSize += outerBlockNums * headNumSize;
        KERNEL_LOG_DEBUG("totalSize:%d, headNumSize:%d, outerBlockNums:%d, actualSeqLengthsTmp:%d, s1BasicBlock:%d, preTokensLeftUp:%d, nextTokensLeftUp:%d \n", totalSize, headNumSize, outerBlockNums, actualSeqLengthsTmp, s1BasicBlock, preTokensLeftUp, nextTokensLeftUp);
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
    if (sparseMode_ == static_cast<int64_t>(SparseMode::LEFT_UP_CAUSAL)) {
        firstFullLoadS1OuterIdx = (std::min(maxActualseqQ, maxActualseqKV) + s1BasicBlock -1 ) / s1BasicBlock - 1;
    } else if (sparseMode_ == static_cast<int64_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        firstFullLoadS1OuterIdx = s1OuterSize - 1;
        uint32_t s1SparseValidSize = maxSeqlenQ_;
        uint32_t s2SparseValidSize = std::min(AlignUp(static_cast<int32_t>(maxSeqlenKv_ - maxSeqlenQ_), static_cast<int32_t>(s1BasicBlock)), static_cast<int32_t>(maxSeqlenKv_));
        // baseParams->set_s1SparseValidSize(s1SparseValidSize);
        // baseParams->set_s2SparseValidSize(s2SparseValidSize);
        KERNEL_LOG_DEBUG("maxActualseqQ:%d, maxActualseqKV:%d, s1BasicBlock:%d\n", maxActualseqQ, maxActualseqKV, s1BasicBlock);
        KERNEL_LOG_DEBUG("s1SparseValidSize:%d, s2SparseValidSize:%d\n", s1SparseValidSize, s2SparseValidSize);
    } else if (sparseMode_ == static_cast<int64_t>(SparseMode::NO_MASK) || static_cast<int64_t>(SparseMode::ALL_MASK)) {
            firstFullLoadS1OuterIdx = -1;
    }

    for (uint32_t cid = 0; cid < splitRes.usedCoreNum; cid++)
    {
        splitRes.s1out[cid] = totalSize;
    }

    return true;
}

bool FlashAttnMetadataCpuKernel::GenMetaData(SplitResult &splitRes)
{
    if (splitCoreMode == SplitCoreMode::SPLIT_S1OUT_CUBE) {
        optiling::detail::S1outFaMetaData* metaDataPtr = (optiling::detail::S1outFaMetaData*)(metaData_->GetData());
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
            // KERNEL_LOG_DEBUG("success into GenMetaData : %d : %d %d %d %d\n",cid, metaDataPtr->s1outMetadata[cid][S1OUT_CORE_ENABLE_INDEX],
            //                                     metaDataPtr->s1outMetadata[cid][S1OUT_M_STRIDE_INDEX],
            //                                     metaDataPtr->s1outMetadata[cid][S1OUT_M_NUM_INDEX],
            //                                     metaDataPtr->s1outMetadata[cid][S1OUT_FIRST_FULL_LOAD_S1_OUTET_IDX_INDEX]);
        }
    }
    
    return true;
}

namespace {
    static const char *kernelType = "FlashAttnMetadata";
    REGISTER_CPU_KERNEL(kernelType, FlashAttnMetadataCpuKernel);
}

} // namespace aicpu