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
    coreNum_ = aicCoreNum_;
    actualSeqLengths.resize(batchSize_, maxSeqlenQ_);
    actualSeqLengthsKV.resize(batchSize_, maxSeqlenKv_);
    if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
        auto cuSeqlensQ = (int32_t *)cuSeqlensQ_->GetData();
        for (uint32_t i = 0; i < batchSize_; ++i) {
            actualSeqLengths[i] = cuSeqlensQ[i+1] - cuSeqlensQ[i];
        }
    }
    if (cuSeqlensKv_ != nullptr && cuSeqlensKv_->GetData() != nullptr) {
        auto cuSeqlensKv = (int32_t *)cuSeqlensKv_->GetData();
        for (uint32_t i = 0; i < batchSize_; ++i) {
            actualSeqLengthsKV[i] = cuSeqlensKv[i+1] - cuSeqlensKv[i];
        }
    }

    if (winLeft_ == -1) {
        preToken_ = std::numeric_limits<int>::max();
    } else {
        preToken_ = winLeft_;
    }
    if (winRight_ == -1) {
        nextToken_ == std::numeric_limits<int>::max();
    } else {
        nextToken_ = winRight_;
    }

    // splitParam
    groupSize_ = numHeadsQ_ / numHeadsKv_;
    if (layoutQ_ == "TND" || layoutQ_ == "NTD") {
        mBaseSize_ = NUM_512;
    } else {
        uint32_t sInnerSizeAlign_ = AlignUp(s2BaseSize_, 16U);
        if (maxSeqlenQ_ <= 16) {
            if (sInnerSizeAlign_ <= NUM_512) {
                mBaseSize_ = NUM_512;
            } else if (sInnerSizeAlign_ <= NUM_1024) {
                mBaseSize_ = NUM_256;
            } else if (sInnerSizeAlign_ <= NUM_2048) {
                mBaseSize_ = NUM_128;
            } else if (sInnerSizeAlign_ <= NUM_4096) {
                mBaseSize_ = NUM_64;
            } else { // sInnerSizeAlign_最大值为8192
                mBaseSize_ = NUM_32;
            }
        } else {
            mBaseSize_ = NUM_512;
        }
    }
    if (layoutQ_ == "TND" || layoutQ_ == "NTD") {
        s2BaseSize_ = NUM_512;
    } else {
        if (maxSeqlenQ_ <= 16) {
            s2BaseSize_ = NUM_8192; // 8192
            uint32_t sInnerSize[3U] = {8192U, 4096U, 2048U};
            uint32_t idx = std::min(groupSize_ / 5U, 2U);
            s2BaseSize_ = sInnerSize[idx];
        } else {
            s2BaseSize_ = NUM_512;
        }
    }
    if (maskMode_ == 2 || maskMode_ == 3 || maskMode_ == 4) {
        s2BaseSize_ = std::min(static_cast<int64_t>(s2BaseSize_), NUM_1024); // attention mask压缩场景，基本块最大支持1024*1024
    }
    // if (fiaInfo_->pageAttentionFlag && fiaInfo_->blockSize != 0) {
    //     uint32_t blockSize = static_cast<uint32_t>(fiaInfo_->blockSize);
    //     if ((s2BaseSize_ > blockSize) && (s2BaseSize_ % blockSize != 0)) {
    //         s2BaseSize_ = (s2BaseSize_ / blockSize) * blockSize;
    //     }
    // }
    if (s2BaseSize_ > maxSeqlenKv_) {
        s2BaseSize_ = maxSeqlenKv_;
    }
    gS1BaseSizeOfFd_ = 8;
    return true;
}

uint32_t FlashAttnMetadataCpuKernel::GetS1SeqSize(uint32_t bIdx)
{
    return actualSeqLengths[bIdx];
}

uint32_t FlashAttnMetadataCpuKernel::GetS2SeqSize(uint32_t bIdx)
{
    return actualSeqLengthsKV[bIdx];
}

uint32_t FlashAttnMetadataCpuKernel::GetSparseSeqSize(uint32_t bIdx)
{
    return 0U;
}

void FlashAttnMetadataCpuKernel::CalcSplitInfo(SplitContext &splitContext)
{
    // 计算每个batch的切分，统计是否为空batch，记录最后有效batch（每个batch的每个N2切分是一样的）
    SplitInfo &splitInfo = splitContext.splitInfo;
    for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
        uint32_t s1Size = GetS1SeqSize(bIdx);
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

int64_t FlashAttnMetadataCpuKernel::CalcPreTokenLeftUp(
    uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<MaskMode>(maskMode_);
    if (mode == MaskMode::BAND) {
        return static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + preToken_;
    }
    return preToken_;
}

int64_t FlashAttnMetadataCpuKernel::CalcNextTokenLeftUp(
    uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<MaskMode>(maskMode_);
    switch (mode) {
        case MaskMode::DEFAULT_MASK:
        case MaskMode::ALL_MASK:
        case MaskMode::LEFT_UP_CAUSAL:
            return nextToken_;
        case MaskMode::RIGHT_DOWN_CAUSAL:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        case MaskMode::BAND:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + nextToken_;
        default:
            return nextToken_;
    }
}

int64_t FlashAttnMetadataCpuKernel::CalcCost(
    uint32_t basicM, uint32_t basicS2)
{
    float a = 25.04f;
    float b = 2.41f;
    float c = 54.99f;
    return static_cast<int64_t>(a * basicM + b * basicS2 + c);
}

BlockCost<int64_t> FlashAttnMetadataCpuKernel::CalcCostTable(uint32_t s1NormalSize, 
    uint32_t s2NormalSize, uint32_t s1GTailSize, uint32_t s2TailSize)
{
    BlockCost<int64_t> typeCost {};
    typeCost[NORMAL_BLOCK][NORMAL_BLOCK] = CalcCost(s1NormalSize, s2NormalSize);
    typeCost[TAIL_BLOCK][NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2NormalSize);
    typeCost[NORMAL_BLOCK][TAIL_BLOCK] = (s2TailSize == 0U) ? 0U : CalcCost(s1NormalSize, s2TailSize);
    typeCost[TAIL_BLOCK][TAIL_BLOCK] = (s1GTailSize == 0U || s2TailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2TailSize);
    return typeCost;
}

Range<uint32_t> FlashAttnMetadataCpuKernel::CalcS2Range(
    uint32_t s1GIdx, const BatchCache &batchCache)
{
    uint32_t s2Start = 0U;
    uint32_t s2End = 0U;

    // actual seq == 0
    if (batchCache.s1Size == 0U || batchCache.s2Size == 0U) {
        return std::make_pair(s2Start, s2End);
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

    // 2. trans index of token to index of block
    // no valid token
    if (s2FirstToken >= static_cast<int64_t>(batchCache.s2Size) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
        s2Start = 0U;
        s2End = 0U;
        return std::make_pair(s2Start, s2End);
    }
    // get valid range
    s2FirstToken = Clip(s2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
    s2LastToken = Clip(s2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));

    s2Start = static_cast<uint32_t>(s2FirstToken) / s2BaseSize_;
    s2End = static_cast<uint32_t>(s2LastToken) / s2BaseSize_ + 1U; // end of block index, Right-open interval

    // printf("s2Start: %u, s2End: %u, s2FirstToken: %lld, s2LastToken: %lld \n", s2Start, s2End, s2FirstToken, s2LastToken);
    return std::make_pair(s2Start, s2End);
}

void FlashAttnMetadataCpuKernel::CalcBatchCache(
    uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1SeqSize(bIdx);
    batchCache.s2Size = GetS2SeqSize(bIdx);
    batchCache.preTokenLeftUp = CalcPreTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
    batchCache.nextTokenLeftUp = CalcNextTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
    batchCache.typeCost = CalcCostTable(mBaseSize_, s2BaseSize_, splitInfo.s1GTailSize[bIdx],
        splitInfo.s2TailSize[bIdx]);
}

void FlashAttnMetadataCpuKernel::CalcS1GCache(uint32_t s1GIdx, 
    const SplitContext &splitContext, const BatchCache &batchCache, S1GCache &s1GCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    s1GCache.bIdx = batchCache.bIdx;
    s1GCache.s1GIdx = s1GIdx;

    auto s2Range = CalcS2Range(s1GIdx, batchCache);
    s1GCache.s2Start = s2Range.first;
    s1GCache.s2End = s2Range.second;

    if (s1GCache.s2Start >= s1GCache.s2End || splitInfo.s1GBaseNum[batchCache.bIdx] == 0) {
        s1GCache.s1GBlock = 0;
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
        return;
    }

    // 计算S2方向满块、尾块数量
    s1GCache.s1GBlock = s1GCache.s2End - s1GCache.s2Start;
    uint32_t curTailS2Num = (splitInfo.s2TailSize[batchCache.bIdx] != 0U &&
        s1GCache.s2End == splitInfo.s2BaseNum[batchCache.bIdx]) ? 1U : 0U;
    uint32_t curNormalS2Num = s1GCache.s1GBlock - curTailS2Num;
    if (s1GIdx == (splitInfo.s1GBaseNum[batchCache.bIdx] - 1U) && splitInfo.s1GTailSize[batchCache.bIdx] != 0U) {
        s1GCache.s1GCost = batchCache.typeCost[TAIL_BLOCK][NORMAL_BLOCK] * curNormalS2Num +
            batchCache.typeCost[TAIL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GLastBlockCost = curTailS2Num > 0U ? batchCache.typeCost[TAIL_BLOCK][TAIL_BLOCK] :
                                 batchCache.typeCost[TAIL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GNormalBlockCost = batchCache.typeCost[TAIL_BLOCK][NORMAL_BLOCK];
    } else {
        s1GCache.s1GCost = batchCache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK] * curNormalS2Num +
            batchCache.typeCost[NORMAL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GLastBlockCost = curTailS2Num > 0U ? batchCache.typeCost[NORMAL_BLOCK][TAIL_BLOCK] :
                                 batchCache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GNormalBlockCost = batchCache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
    }

    if (s1GCache.s1GBlock == 1) {
        s1GCache.s1GLastBlockCost += V0_COST;
    }
    s1GCache.s1GCost += V0_COST;
}

void FlashAttnMetadataCpuKernel::CalcBatchCost(
    uint32_t bIdx, const SplitContext &splitContext, CostInfo &costInfo)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    costInfo.bN2CostOfEachBatch[bIdx] = 0;
    costInfo.bN2BlockOfEachBatch[bIdx] = 0U;
    costInfo.bN2LastBlockCostOfEachBatch[bIdx] = 0U;

    if (GetS1SeqSize(bIdx) == 0U || GetS2SeqSize(bIdx) == 0U) {
        return;
    }

    BatchCache bCache;
    S1GCache s1GCache;
    CalcBatchCache(bIdx, splitContext, bCache);
    for (uint32_t s1GIdx = 0; s1GIdx < splitInfo.s1GBaseNum[bIdx]; s1GIdx++) {
        CalcS1GCache(s1GIdx, splitContext, bCache, s1GCache);
        costInfo.bN2CostOfEachBatch[bIdx] += s1GCache.s1GCost;
        costInfo.bN2BlockOfEachBatch[bIdx] += s1GCache.s1GBlock;
        if(s1GCache.s1GBlock > 0){
            costInfo.bN2LastBlockCostOfEachBatch[bIdx] = s1GCache.s1GLastBlockCost;
        }
    }
}

void FlashAttnMetadataCpuKernel::CalcCostInfo(SplitContext &splitContext)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    CostInfo &costInfo = splitContext.costInfo;

    if (splitInfo.isKvSeqAllZero) {
        costInfo.totalCost = 0;
        costInfo.totalBlockNum = 0U;
        return;
    }

    // 计算batch的负载并记录，用于按batch分配，需要按行计算起止点，统计块数、负载
    for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
        CalcBatchCost(bIdx, splitContext, costInfo);
        costInfo.totalCost += costInfo.bN2CostOfEachBatch[bIdx] * numHeadsKv_;
        costInfo.totalBlockNum += costInfo.bN2BlockOfEachBatch[bIdx] * numHeadsKv_;
    }
}

void FlashAttnMetadataCpuKernel::UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    const CostInfo &costInfo = splitContext.costInfo;

    bool UpdateS1G = false;
    bool UpdateBatch = false;

    // Update S2
    if (assignContext.curS2Idx >= assignContext.s1GCache.s2End) {    // 边界assignInfo.s2End是取不到的开区间
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
    if (assignContext.curBN2Idx == batchSize_ * numHeadsKv_) {  // 所有负载全部分配完，设置最后一个核的右开区间，返回
        assignContext.curS1GIdx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.isFinished = true;
        return;
    }

    if (assignContext.curBN2Idx / numHeadsKv_ != assignContext.curBIdx) {
        assignContext.curBIdx = assignContext.curBN2Idx / numHeadsKv_;
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
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void FlashAttnMetadataCpuKernel::AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }
    const CostInfo &costInfo = splitContext.costInfo;
    while (assignContext.bN2Cost == 0 || IsWithinTolerance(assignContext.coreCache.costLimit,
        costInfo.bN2LastBlockCostOfEachBatch[assignContext.curBIdx] / FA_TOLERANCE_RATIO,
        assignContext.coreCache.cost + assignContext.bN2Cost)) {
        assignContext.coreCache.cost += assignContext.bN2Cost;
        assignContext.coreCache.block += assignContext.bN2Block;
        assignContext.curBN2Idx++;

        // to the end
        if (assignContext.curBN2Idx == batchSize_ * numHeadsKv_) {
            assignContext.curS1GIdx = 0U;
            assignContext.curS2Idx = 0U;
            assignContext.isFinished = true;
            return;
        }

        // next batch
        if (assignContext.curBN2Idx / numHeadsKv_ != assignContext.curBIdx) {
            assignContext.curBIdx = assignContext.curBN2Idx / numHeadsKv_;
            CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        }

        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
        assignContext.curS1GIdx = 0U;
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void FlashAttnMetadataCpuKernel::AssignByRow(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit,
        assignContext.s1GCache.s1GLastBlockCost / FA_TOLERANCE_RATIO,
        assignContext.coreCache.cost + assignContext.s1GCache.s1GCost)) {
        assignContext.coreCache.cost += assignContext.s1GCache.s1GCost;
        assignContext.coreCache.block += assignContext.s1GCache.s1GBlock;

        // 当前batch被分配一行出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost > assignContext.s1GCache.s1GCost ?
                                assignContext.bN2Cost - assignContext.s1GCache.s1GCost : 0;
        assignContext.bN2Block = assignContext.bN2Block > assignContext.s1GCache.s1GBlock ?
                                 assignContext.bN2Block - assignContext.s1GCache.s1GBlock : 0U;
        // 计算新一行的信息
        do {
            assignContext.curS1GIdx++;
            CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        } while(assignContext.s1GCache.s1GBlock == 0);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void FlashAttnMetadataCpuKernel::AssignByBlock(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }
    int64_t realCost = curCost;
    if (assignContext.curS2Idx == 0 && assignContext.s1GCache.s1GBlock != 1) {
        realCost += V0_COST;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit, realCost / FA_TOLERANCE_RATIO, 
            assignContext.coreCache.cost + realCost)) { // (costLimit - curCostOnCore) * FA_TOLERANCE_RATIO > curCost；至少分配1块
        assignContext.coreCache.cost += realCost;
        assignContext.coreCache.block++;
        assignContext.curS2Idx++;
        // 当前batch被分配一块出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost - realCost;
        // 当前行被分配一块出去，更新剩余负载
        assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - realCost;
        assignContext.bN2Block--;
        assignContext.s1GCache.s1GBlock--;

        realCost = curCost;
    }
}

void FlashAttnMetadataCpuKernel::ForceAssign(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }
    int64_t realCost = curCost;
    if (assignContext.curS2Idx == 0 && assignContext.s1GCache.s1GBlock != 1) {
        realCost += V0_COST;
    }

    assignContext.coreCache.cost += realCost;
    assignContext.coreCache.block++;
    assignContext.curS2Idx++;
    // 当前batch被分配一块出去，更新剩余负载
    assignContext.bN2Cost = assignContext.bN2Cost - realCost;
    assignContext.bN2Block--;
    // 当前行被分配一块出去，更新剩余负载
    assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - realCost;
    assignContext.s1GCache.s1GBlock--;
    UpdateCursor(splitContext, assignContext); 
}

bool FlashAttnMetadataCpuKernel::IsNeedRecordFDInfo(const AssignContext &assignContext, const SplitResult &splitRes)
{
    // 切分点大概率不会刚好在行尾，因此滞后处理归约信息的统计，到下一个切分点再判断是否需要归约
    // 核0无需处理
    if (assignContext.curCoreIdx == 0U) {
        return false;
    }
    // 无跨核行，无需处理
    if (assignContext.curKvSplitPart <= 1U) {
        return false;
    }
    // 需要归约的行还未处理完
    if (assignContext.curBN2Idx == splitRes.bN2End[assignContext.curCoreIdx - 1U] &&
        assignContext.curS1GIdx == splitRes.gS1End[assignContext.curCoreIdx - 1U]) {
        return false;
    }
    return true;
}

void FlashAttnMetadataCpuKernel::RecordFDInfo(const SplitContext &splitContext, const AssignContext &assignContext, SplitResult &result)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    // 需要规约的行是上一个核的切分点所在位置
    uint32_t splitBIdx = result.bN2End[assignContext.curCoreIdx - 1U] / numHeadsKv_;
    uint32_t splitS1GIdx = result.gS1End[assignContext.curCoreIdx - 1U];
    uint32_t s1Size = GetS1SeqSize(splitBIdx);

    // 计算归约数据的FD均衡划分信息
    uint32_t curFdS1gSize = (splitS1GIdx == splitInfo.s1GBaseNum[splitBIdx] - 1U) ?
                            (s1Size * groupSize_ - splitS1GIdx * mBaseSize_) : mBaseSize_;
    // 记录
    result.maxS2SplitNum = std::max(result.maxS2SplitNum, assignContext.curKvSplitPart);
    // 若存在头归约，则切分点一定为上一个核结束的位置
    result.fdRes.bN2Idx[result.numOfFdHead] = result.bN2End[assignContext.curCoreIdx - 1U];
    result.fdRes.mIdx[result.numOfFdHead] = result.gS1End[assignContext.curCoreIdx - 1U];
    result.fdRes.worksapceIdx[result.numOfFdHead] = assignContext.preFdDataNum;
    result.fdRes.s2SplitNum[result.numOfFdHead] = assignContext.curKvSplitPart;
    result.fdRes.mSize[result.numOfFdHead] = curFdS1gSize;
    result.numOfFdHead++;
}

void FlashAttnMetadataCpuKernel::CalcSplitPlan(uint32_t coreNum,
    int64_t costLimit, const SplitContext &splitContext, SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;

    if (coreNum == 0U) {
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

    for (uint32_t i = 0; i < coreNum; ++i) {
        if (result.maxCost > costLimit) {
            return;
        }
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
            break;
        }

        assignContext.curCoreIdx = i;
        result.firstFdDataWorkspaceIdx[assignContext.curCoreIdx] = assignContext.preFdDataNum +
            assignContext.curKvSplitPart - 1U;

        assignContext.coreCache = {};
        assignContext.coreCache.costLimit = assignContext.unassignedCost / (coreNum - assignContext.curCoreIdx);

        // 1、按整batch分配
        AssignByBatch(splitContext, assignContext);
        // 2、按行分配
        AssignByRow(splitContext, assignContext);
        // 3、按块分配
        AssignByBlock(splitContext, assignContext);
        // 4、强制分配
        if (assignContext.coreCache.block == 0) {
            ForceAssign(splitContext, assignContext);
        }
        result.bN2End[i] = assignContext.curBN2Idx;
        result.gS1End[i] = assignContext.curS1GIdx;
        result.s2End[i] = assignContext.curS2Idx;
        result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);

        assignContext.unassignedCost -= assignContext.coreCache.cost;
        // printf("assignContext.unassignedCost: %u \n", assignContext.unassignedCost);

        // 对之前的归约信息进行记录并清理
        if (IsNeedRecordFDInfo(assignContext, result)) {
            RecordFDInfo(splitContext, assignContext, result);
            assignContext.preFdDataNum += assignContext.curKvSplitPart;
            assignContext.curKvSplitPart = 1U;
        }

        // 更新S2切分信息
        if (assignContext.curS2Idx > assignContext.s1GCache.s2Start &&
            assignContext.curS2Idx <= assignContext.s1GCache.s2End) {
            assignContext.curKvSplitPart++;
        }
    }

    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

void FlashAttnMetadataCpuKernel::CopyTmpResult(SplitResult &tmpRes, SplitResult &splitRes)
{
    splitRes.usedCoreNum = tmpRes.usedCoreNum;
    splitRes.maxCost = tmpRes.maxCost;
    splitRes.numOfFdHead = tmpRes.numOfFdHead;
    splitRes.maxS2SplitNum = tmpRes.maxS2SplitNum;

    for (size_t i = 0; i < aicCoreNum_; ++i) {
        splitRes.bN2End[i] = tmpRes.bN2End[i];
        splitRes.gS1End[i] = tmpRes.gS1End[i];
        splitRes.s2End[i] = tmpRes.s2End[i];
        splitRes.firstFdDataWorkspaceIdx[i] = tmpRes.firstFdDataWorkspaceIdx[i];

        splitRes.fdRes.bN2Idx[i] = tmpRes.fdRes.bN2Idx[i];
        splitRes.fdRes.mIdx[i] = tmpRes.fdRes.mIdx[i];
        splitRes.fdRes.worksapceIdx[i] = tmpRes.fdRes.worksapceIdx[i];
        splitRes.fdRes.s2SplitNum[i] = tmpRes.fdRes.s2SplitNum[i];
        splitRes.fdRes.mSize[i] = tmpRes.fdRes.mSize[i];
    }
}

void FlashAttnMetadataCpuKernel::ClearTmpResult(SplitResult &tmpResult)
{
    tmpResult.usedCoreNum = 0U;
    tmpResult.maxCost = 0;
    tmpResult.numOfFdHead = 0U;
    tmpResult.maxS2SplitNum = 0U;
    tmpResult.fdRes.usedVecNum = 0U;

    for (size_t i = 0; i < aicCoreNum_; ++i) {
        tmpResult.bN2End[i] = 0U;
        tmpResult.gS1End[i] = 0U;
        tmpResult.s2End[i] = 0U;
        tmpResult.firstFdDataWorkspaceIdx[i] = 0U;

        tmpResult.fdRes.bN2Idx[i] = 0U;
        tmpResult.fdRes.mIdx[i] = 0U;
        tmpResult.fdRes.worksapceIdx[i] = 0U;
        tmpResult.fdRes.s2SplitNum[i] = 0U;
        tmpResult.fdRes.mSize[i] = 0U;
    }
}

void FlashAttnMetadataCpuKernel::RollBackCursor(const SplitContext &splitContext,
    const CostInfo &costInfo, SplitResult &splitRes)
{
    for (size_t i = 0; i < splitRes.usedCoreNum; ++i) {
        // x, y, z
        if (splitRes.s2End[i] > 0U) {
            splitRes.s2End[i] = splitRes.s2End[i] - 1U;
            continue;
        }
        uint32_t bIdx = splitRes.bN2End[i] / numHeadsKv_;
        // x, y, 0
        if (splitRes.gS1End[i] > 0U) {
            splitRes.gS1End[i] = splitRes.gS1End[i] - 1U;
            splitRes.s2End[i] = splitContext.splitInfo.s2BaseNum[bIdx] - 1U;
            continue;
        }

        // x, 0, 0
        uint32_t bN2Idx = splitRes.bN2End[i] > 0U ? splitRes.bN2End[i] - 1U : 0U;
        bIdx = bN2Idx / numHeadsKv_;

        // last end point
        if (i == splitRes.usedCoreNum - 1U && costInfo.bN2BlockOfEachBatch[bIdx] == 0U) {
            splitRes.bN2End[i] = batchSize_ * numHeadsKv_ - 1;
            splitRes.gS1End[i] = splitContext.splitInfo.s1GBaseNum[bIdx] > 0 ?
                                 splitContext.splitInfo.s1GBaseNum[bIdx] - 1U : 0U;
            splitRes.s2End[i] = splitContext.splitInfo.s2BaseNum[bIdx] > 0 ?
                                splitContext.splitInfo.s2BaseNum[bIdx] - 1U : 0U;
            continue;
        }

        while (bN2Idx > 0U && costInfo.bN2BlockOfEachBatch[bIdx] == 0U) {
            bN2Idx -= 1U;
            bIdx = bN2Idx / numHeadsKv_;
        }

        if (costInfo.bN2BlockOfEachBatch[bIdx] != 0U) {
            splitRes.bN2End[i] = bN2Idx;
            splitRes.gS1End[i] = splitContext.splitInfo.s1GBaseNum[bIdx] - 1U;
            splitRes.s2End[i] = splitContext.splitInfo.s2BaseNum[bIdx] - 1U;
        } else {
            splitRes.bN2End[i] = 0U;
            splitRes.gS1End[i] = 0U;
            splitRes.s2End[i] = 0U;
        }
    }
}

void FlashAttnMetadataCpuKernel::SplitFD(SplitResult &result)
{
    FlashDecodeResult &fdRes = result.fdRes;

    // 计算FD的总数据量
    uint64_t totalFDLoad = 0;
    for (uint32_t i = 0; i < result.numOfFdHead; ++i) {
        totalFDLoad += fdRes.s2SplitNum[i] * fdRes.mSize[i];
    }

    // 计算每个核处理的load
    uint64_t avgLoad = (totalFDLoad + aivCoreNum_ - 1U) / aivCoreNum_;
    uint32_t curCoreIndex = 0;
    for (uint32_t i = 0; i < result.numOfFdHead; ++i) {
        uint32_t curFDVectorNum = fdRes.s2SplitNum[i] * fdRes.mSize[i] / avgLoad;
        uint32_t curAvgMSize = (fdRes.mSize[i] + curFDVectorNum - 1U) / curFDVectorNum;
        curFDVectorNum = (fdRes.mSize[i] + curAvgMSize - 1U) / curAvgMSize;

        for (uint32_t vid = 0; vid < curFDVectorNum; ++vid) {
            fdRes.taskIdx[curCoreIndex] = i;
            fdRes.mStart[curCoreIndex] = vid * curAvgMSize;
            fdRes.mLen[curCoreIndex] = (vid < curFDVectorNum - 1) ? curAvgMSize : (fdRes.mSize[i] - vid * curAvgMSize);
            curCoreIndex++;
        }
    }
    fdRes.usedVecNum = curCoreIndex;
}

bool FlashAttnMetadataCpuKernel::BalanceSchedule(SplitResult &splitRes)
{
    SplitContext splitContext(batchSize_);
    // 1、划分基本块，统计信息
    CalcSplitInfo(splitContext);
    // 全空case
    if (splitContext.splitInfo.isKvSeqAllZero) {
        splitRes.usedCoreNum = 1U;
        splitRes.bN2End[0] = batchSize_ * numHeadsKv_ - 1;
        splitRes.gS1End[0] = 0U;
        splitRes.s2End[0] = 0U;
        return true;
    }
    CalcCostInfo(splitContext);
    // 2、获取每个核的分配方案
    uint32_t maxCore = std::min(coreNum_, splitContext.costInfo.totalBlockNum);
    uint32_t minCore = static_cast<uint32_t>(
        std::sqrt(static_cast<float>(splitContext.costInfo.totalBlockNum) + 0.25f) + 0.5f);
    minCore = std::min(minCore, maxCore);

    splitRes.maxCost = INT64_MAX;
    splitRes.usedCoreNum = 1U;
    SplitResult tmpResult {aicCoreNum_, aivCoreNum_};
    for (uint32_t i = minCore; i <= maxCore; ++i) {
        CalcSplitPlan(i, splitRes.maxCost, splitContext, tmpResult);
        if (tmpResult.maxCost < splitRes.maxCost) {
            CopyTmpResult(tmpResult, splitRes);
        }
        ClearTmpResult(tmpResult);
    }
    // 3、存在FD任务，对FD进行负载均衡分配
    if (splitRes.numOfFdHead > 0U) {
        SplitFD(splitRes);
    }
    splitRes.usedCoreNum = std::max(splitRes.usedCoreNum, 1U);  // 至少使用1个core
    RollBackCursor(splitContext, splitContext.costInfo, splitRes);
    return true;
}

bool FlashAttnMetadataCpuKernel::GenMetaData(SplitResult &splitRes)
{
    uint32_t sectionNum = 1;
    detail::FaMetaData faMetadata(metaData_->GetData(), sectionNum);
    for (uint32_t sectionId = 0; sectionId < sectionNum; ++sectionId) {
        for (size_t i = 0; i < aicCoreNum_; ++i) {
            // faMetadata.setFaMetadata(sectionId, i, optiling::FA_CORE_ENABLE_INDEX, 0U);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_BN2_START_INDEX, 0U);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_M_START_INDEX, 0U);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_S2_START_INDEX, 0U);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_BN2_END_INDEX, 0U);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_M_END_INDEX, 0U);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_S2_END_INDEX, 0U);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, 0U);
        }
        for (size_t i = 0; i < aivCoreNum_; ++i) {
            // faMetadata.setFdMetadata(sectionId, i, optiling::FD_CORE_ENABLE_INDEX, 0U);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_BN2_IDX_INDEX, 0U);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_M_IDX_INDEX, 0U);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_WORKSPACE_IDX_INDEX, 0U);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_WORKSPACE_NUM_INDEX, 0U);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_M_START_INDEX, 0U);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_M_NUM_INDEX, 0U);
        }
        // FA Metadata Generate
        for (size_t i = 0; i < splitRes.usedCoreNum; ++i) {
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_CORE_ENABLE_INDEX, 1U);
            // FA start
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_BN2_START_INDEX, (i == 0) ? 0U : splitRes.bN2End[i - 1]);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_M_START_INDEX, (i == 0) ? 0U : splitRes.gS1End[i - 1]);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_S2_START_INDEX, (i == 0) ? 0U : splitRes.s2End[i - 1]);
            // FA end
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_BN2_END_INDEX, splitRes.bN2End[i]);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_M_END_INDEX, splitRes.gS1End[i]);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_S2_END_INDEX, splitRes.s2End[i]);
            // FA idx
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, splitRes.firstFdDataWorkspaceIdx[i]);
        }

        // FD Metadata Generate
        const FlashDecodeResult &fdRes = splitRes.fdRes;
        for (size_t i = 0; i < fdRes.usedVecNum; ++i) {
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_CORE_ENABLE_INDEX, 1U);
            uint32_t curTaskIdx = fdRes.taskIdx[i];
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_BN2_IDX_INDEX, fdRes.bN2Idx[curTaskIdx]);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_M_IDX_INDEX, fdRes.mIdx[curTaskIdx]);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_WORKSPACE_IDX_INDEX, fdRes.worksapceIdx[curTaskIdx]);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_WORKSPACE_NUM_INDEX, fdRes.s2SplitNum[curTaskIdx]);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_M_START_INDEX, fdRes.mStart[i]);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_M_NUM_INDEX, fdRes.mLen[i]);
        }
    }
    return true;
}

namespace {
static const char *kernelType = "FlashAttnMetadata";
REGISTER_CPU_KERNEL(kernelType, FlashAttnMetadataCpuKernel);
} // namespace

} // namespace aicpu