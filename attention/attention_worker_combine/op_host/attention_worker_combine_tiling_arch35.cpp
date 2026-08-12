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
 * \file attention_worker_combine_tiling_arch35.cpp
 * \brief
 */
#include <cstring>
#include "attention_worker_combine_tiling_base.h"
#include "../op_kernel/arch35/attention_worker_combine_tiling_struct.h"

namespace optiling {

constexpr int64_t EXPERT_SCALES_INDEX = 1;
constexpr int64_t SHAPE_IDX_BS = 0;
constexpr int64_t SHAPE_IDX_K = 1;
constexpr int64_t HIDDEN_SIZE_INDEX = 0;
constexpr int64_t TOKEN_DTYPE_INDEX = 1;
constexpr int64_t NEED_SCHEDULE_INDEX = 2;
constexpr int64_t DIM_SIZE = 2;
constexpr int64_t H_ALIGN_SIZE = 512;
constexpr int64_t K_UPPER_BOUND = 64;

constexpr int64_t TILING_KEY_DIVIDE_BS_FP16 = 10000;
constexpr int64_t TILING_KEY_DIVIDE_BS_BF16 = 10001;
constexpr int64_t TILING_KEY_DIVIDE_H_FP16 = 10010;
constexpr int64_t TILING_KEY_DIVIDE_H_BF16 = 10011;
constexpr int64_t TILING_KEY_DIVIDE_K_FP16 = 10020;
constexpr int64_t TILING_KEY_DIVIDE_K_BF16 = 10021;

constexpr int64_t B32_DTYPE_BYTES = 4;
constexpr int64_t B16_DTYPE_BYTES = 2;

inline int64_t CeilDiv(int64_t N, int64_t n)
{
    if (unlikely(n == 0)) {
        return N;
    }
    return ((N + n - 1) / n);
}

inline int64_t AlignUp(int64_t N, int64_t n)
{
    if (unlikely(n == 0)) {
        return N;
    }
    return (((N + n - 1) / n) * n);
}

inline int64_t AlignDown(int64_t N, int64_t n)
{
    if (unlikely(n == 0)) {
        return N;
    }
    return N / n * n;
}

static std::tuple<int64_t, int64_t> GetShapeTuple(const gert::TilingContext *context, const int64_t index = 0)
{
    const gert::StorageShape *shapePtr = context->GetInputShape(index);
    OP_CHECK_IF(shapePtr == nullptr, OP_LOGE(context->GetNodeName(), "Shape is nullptr."),
                return std::make_tuple(0, 0));
    OP_CHECK_IF(shapePtr->GetStorageShape().GetDimNum() != DIM_SIZE,
                OP_LOGE(context->GetNodeName(), "Shape must be (BS, K)."), return std::make_tuple(0, 0));
    return std::make_tuple(shapePtr->GetStorageShape().GetDim(SHAPE_IDX_BS),
                           shapePtr->GetStorageShape().GetDim(SHAPE_IDX_K));
}

class AttentionWorkerCombineTilingArch35 : public AttentionWorkerCombineTilingBase {
public:
    explicit AttentionWorkerCombineTilingArch35(gert::TilingContext *context)
        : AttentionWorkerCombineTilingBase(context)
    {}
    ~AttentionWorkerCombineTilingArch35() override = default;

protected:
    bool IsCapable() override
    {
        return Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_);
    }

    ge::graphStatus DoGetPlatformInfo() override;
    ge::graphStatus DoGetShapeAttrsInfo() override;
    ge::graphStatus CalcOpTiling() override;
    ge::graphStatus CalcTilingKey() override;
    void DoPostTiling() override;

private:
    void DoOpTilingFullK(int64_t batchSize, int64_t bsCoreNum, int64_t hAlign, int64_t kIn);
    void SelectBestHCore(int64_t hAlign, int64_t hInSplitK, int64_t &bsCoreNum, int64_t &lastCoreNum,
                         int64_t &lastBestHCore);

    AttentionWorkerCombineRegbaseTilingData *tilingData{nullptr};
    int64_t batchSize_{0};
    int64_t k_{0};
    int64_t hiddenSize_{0};
    int64_t needSchedule_{0};
};

ge::graphStatus AttentionWorkerCombineTilingArch35::DoGetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_->GetNodeName(), "platformInfo is nullptr."),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ubSize_ = ubSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttentionWorkerCombineTilingArch35::DoGetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE(context_->GetNodeName(), "context_ can not be nullptr."),
                return ge::GRAPH_FAILED);

    auto expertScalesShapeTuple = GetShapeTuple(context_, EXPERT_SCALES_INDEX);
    int64_t batchSize = std::get<SHAPE_IDX_BS>(expertScalesShapeTuple);
    int64_t k = std::get<SHAPE_IDX_K>(expertScalesShapeTuple);

    OP_CHECK_IF(k > K_UPPER_BOUND, OP_LOGE(context_->GetNodeName(), "k should <= 64."), return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto hiddenSize = attrs->GetInt(HIDDEN_SIZE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, hiddenSize);
    auto tokenDtype = attrs->GetInt(TOKEN_DTYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tokenDtype);
    tokenDtype_ = *tokenDtype;
    auto needSchedule = attrs->GetInt(NEED_SCHEDULE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, needSchedule);

    batchSize_ = batchSize;
    k_ = k;
    hiddenSize_ = *hiddenSize;
    needSchedule_ = *needSchedule;
    return ge::GRAPH_SUCCESS;
}

void AttentionWorkerCombineTilingArch35::DoOpTilingFullK(int64_t batchSize, int64_t bsCoreNum, int64_t hAlign,
                                                         int64_t kIn)
{
    int64_t bsCoreFactor = 0;
    int64_t mainCoreNum = 0;
    int64_t tailCoreNum = 0;
    int64_t tailCoreBsLoopNum = 0;

    tilingData->usedCoreNum = bsCoreNum;
    tilingData->BsSplitFactor = 1;
    tilingData->BsSplitCoreNum = bsCoreNum;

    bsCoreFactor = CeilDiv(batchSize, bsCoreNum);
    tilingData->mainCoreBsLoopNum = bsCoreFactor;
    if (bsCoreFactor == 0) {
        bsCoreFactor = 1;
    }
    mainCoreNum = batchSize / bsCoreFactor;
    tailCoreNum = bsCoreNum - mainCoreNum;
    if (tailCoreNum == 0) {
        tilingData->tailCoreBsLoopNum = bsCoreFactor;
    } else {
        tailCoreBsLoopNum = (batchSize - mainCoreNum * bsCoreFactor) / tailCoreNum;
        tilingData->tailCoreBsLoopNum = tailCoreBsLoopNum;
    }

    tilingData->HSplitFactor = hAlign;
    tilingData->HSplitCoreNum = 1;
    tilingData->mainCoreHLoopNum = 1;
    tilingData->tailCoreHLoopNum = 1;

    tilingData->KSplitFactor = kIn;
    tilingData->KSplitTailFactor = kIn;
    tilingData->KSplitLoopNum = 1;

    tilingKey_ = TILING_KEY_DIVIDE_BS_FP16;
    if (tokenDtype_ == 1) {
        tilingKey_ += 1;
    }
    context_->SetBlockDim(bsCoreNum);
}

void AttentionWorkerCombineTilingArch35::SelectBestHCore(int64_t hAlign, int64_t hInSplitK, int64_t &bsCoreNum,
                                                         int64_t &lastCoreNum, int64_t &lastBestHCore)
{
    int64_t bsMaxCore = 0;
    int64_t batchSize = tilingData->BS;
    int64_t maxCoreNum = static_cast<int64_t>(coreNum_ * 0.8);

    int64_t hOut = CeilDiv(hAlign, hInSplitK);
    int64_t hMaxCore = std::min(hOut, maxCoreNum);
    for (int x = 2; x < hMaxCore + 1; x++) {
        bsMaxCore = (static_cast<int64_t>(coreNum_) / static_cast<int64_t>(x));
        bsCoreNum = CeilDiv(batchSize, CeilDiv(batchSize, bsMaxCore));
        if (x * bsCoreNum > lastCoreNum) {
            lastCoreNum = x * bsCoreNum;
            lastBestHCore = x;
        }
        if (x * bsCoreNum >= maxCoreNum) {
            break;
        }
    }
}

ge::graphStatus AttentionWorkerCombineTilingArch35::CalcOpTiling()
{
    tilingData = context_->GetTilingData<AttentionWorkerCombineRegbaseTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);

    tilingData->BS = batchSize_;
    tilingData->K = k_;
    tilingData->H = hiddenSize_;
    tilingData->needSchedule = needSchedule_;

    int64_t k = tilingData->K;
    int64_t hiddenSize = tilingData->H;
    int64_t batchSize = tilingData->BS;
    int64_t needSchedule = tilingData->needSchedule;

    int64_t maxCoreNum = static_cast<int64_t>(coreNum_ * 0.8);
    int64_t kAddOne = k + 1;
    int64_t hInFullK = AlignDown(static_cast<int64_t>(ubSize_) /
                                     (2 * kAddOne * B16_DTYPE_BYTES + 2 * B16_DTYPE_BYTES + 2 * B32_DTYPE_BYTES),
                                 H_ALIGN_SIZE);
    int64_t bsCoreFactor = CeilDiv(batchSize, static_cast<int64_t>(coreNum_));
    int64_t bsCoreNum = CeilDiv(batchSize, bsCoreFactor);
    int64_t hInSplitK =
        AlignDown(static_cast<int64_t>(ubSize_) / (2 * B16_DTYPE_BYTES + 2 * B16_DTYPE_BYTES + 2 * B32_DTYPE_BYTES),
                  H_ALIGN_SIZE);
    int64_t kIn = 1;
    int64_t hAlign = AlignUp(hiddenSize, H_ALIGN_SIZE);
    int64_t lastCoreNum = 1;
    int64_t lastBestHCore = 1;
    int64_t blockDim = bsCoreNum;
    int64_t mainCoreNum = 0;
    int64_t tailCoreNum = 0;
    int64_t hTotalBlockNum = 0;
    int64_t tailCoreBsLoopNum = 1;
    int64_t hSplitFactor = hInSplitK;
    int64_t hSplitCoreNum = 1;
    int64_t hSplitTailFactor = 0;
    int64_t mainCoreHLoopNum = 0;
    int64_t tailCoreHLoopNum = 1;

    if (hAlign <= hInFullK) {
        DoOpTilingFullK(batchSize, bsCoreNum, hAlign, k);
        return ge::GRAPH_SUCCESS;
    }

    if (bsCoreNum <= maxCoreNum) {
        if (hAlign > hInSplitK) {
            if (needSchedule) {
                hSplitFactor = hInSplitK;
                tailCoreHLoopNum = CeilDiv(hAlign, hInSplitK);
            } else {
                SelectBestHCore(hAlign, hInSplitK, bsCoreNum, lastCoreNum, lastBestHCore);
                blockDim = lastCoreNum;
                hSplitFactor = hInSplitK;
                hSplitTailFactor = hInSplitK == 0 ? hInSplitK : hAlign % hInSplitK;

                hTotalBlockNum = CeilDiv(hAlign, hSplitFactor);
                mainCoreHLoopNum = CeilDiv(hTotalBlockNum, lastBestHCore);
                hSplitCoreNum = CeilDiv(hTotalBlockNum, mainCoreHLoopNum);
                tailCoreHLoopNum = mainCoreHLoopNum == 0 ? mainCoreHLoopNum : hTotalBlockNum % mainCoreHLoopNum;
            }
            tilingKey_ = TILING_KEY_DIVIDE_H_FP16;
        } else {
            hSplitFactor = hAlign;
            if (hAlign == 0) {
                hAlign = 1;
            }
            kIn = std::min(k, hInSplitK / hAlign);
            tilingKey_ = TILING_KEY_DIVIDE_K_FP16;
        }
    } else {
        hSplitCoreNum = 1;
        if (hAlign > hInSplitK) {
            hSplitFactor = hInSplitK;
            tailCoreHLoopNum = CeilDiv(hAlign, hInSplitK);
            tilingKey_ = TILING_KEY_DIVIDE_H_FP16;
        } else {
            hSplitFactor = hAlign;
            if (hAlign == 0) {
                hAlign = 1;
            }
            kIn = std::min(k, hInSplitK / hAlign);
            tilingKey_ = TILING_KEY_DIVIDE_K_FP16;
        }
    }
    tilingData->usedCoreNum = blockDim;
    tilingData->BsSplitFactor = 1;
    tilingData->BsSplitCoreNum = bsCoreNum;

    if (hSplitTailFactor == 0) {
        hSplitTailFactor = hInSplitK;
    }
    tilingData->HSplitFactor = hSplitFactor;
    tilingData->HSplitTailFactor = hAlign % hSplitFactor;
    tilingData->HSplitCoreNum = hSplitCoreNum;

    tilingData->KSplitFactor = kIn;
    if (kIn == 0) {
        kIn = 1;
    }
    tilingData->KSplitTailFactor = k % kIn;
    tilingData->KSplitLoopNum = k / kIn;

    bsCoreFactor = CeilDiv(batchSize, bsCoreNum);
    tilingData->mainCoreBsLoopNum = bsCoreFactor;
    if (bsCoreFactor == 0) {
        bsCoreFactor = 1;
    }
    mainCoreNum = batchSize / bsCoreFactor;
    tailCoreNum = bsCoreNum - mainCoreNum;
    if (tailCoreNum == 0) {
        tilingData->tailCoreBsLoopNum = bsCoreFactor;
    } else {
        tailCoreBsLoopNum = (batchSize - mainCoreNum * bsCoreFactor) / tailCoreNum;
        tilingData->tailCoreBsLoopNum = tailCoreBsLoopNum;
    }

    if (tailCoreHLoopNum == 0) {
        tailCoreHLoopNum = mainCoreHLoopNum;
    }
    tilingData->mainCoreHLoopNum = mainCoreHLoopNum;
    tilingData->tailCoreHLoopNum = tailCoreHLoopNum;

    context_->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttentionWorkerCombineTilingArch35::CalcTilingKey()
{
    if (tokenDtype_ == 1) {
        tilingKey_ += 1;
    }
    return ge::GRAPH_SUCCESS;
}

void AttentionWorkerCombineTilingArch35::DoPostTiling()
{
    return;
}

REGISTER_OPS_TILING_TEMPLATE(AttentionWorkerCombine, AttentionWorkerCombineTilingArch35, 1000);

} // namespace optiling
