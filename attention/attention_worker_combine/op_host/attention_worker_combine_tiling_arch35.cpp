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
#include <string>
#include "attention_worker_combine_tiling_base.h"
#include "../op_kernel/arch35/attention_worker_combine_tiling_struct.h"

namespace optiling {

constexpr int64_t EXPERT_SCALES_INDEX = 1;
constexpr int64_t HIDDEN_SIZE_INDEX = 0;
constexpr int64_t TOKEN_DTYPE_INDEX = 1;
constexpr int64_t NEED_SCHEDULE_INDEX = 2;
constexpr int64_t H_ALIGN_SIZE = 512;
constexpr int64_t K_UPPER_BOUND = 64;

constexpr int64_t TILING_KEY_DIVIDE_BS_FP16 = 10000;
constexpr int64_t TILING_KEY_DIVIDE_H_FP16 = 10010;
constexpr int64_t TILING_KEY_DIVIDE_K_FP16 = 10020;

// Add token_dtype (2/3/4) to select E5M2/E4M3FN/E2M1 within each MXFP strategy.
constexpr int64_t TILING_KEY_DIVIDE_BS_MXFP_BASE = 11000;
constexpr int64_t TILING_KEY_DIVIDE_H_MXFP_BASE = 11010;
constexpr int64_t TILING_KEY_DIVIDE_K_MXFP_BASE = 11020;

constexpr int64_t B32_DTYPE_BYTES = 4;
constexpr int64_t B16_DTYPE_BYTES = 2;

constexpr int64_t TOKEN_DTYPE_MXFP4 = 4;
constexpr int64_t MXFP_SCALE_GROUP_SIZE = 32;
constexpr int64_t UB_BLOCK_BYTES = 32;
constexpr int64_t FP4_ELEMENTS_PER_BYTE = 2;
constexpr int64_t FP8_ELEMENTS_PER_BYTE = 1;
// One FP32 decoded value, one FP32 accumulator and one BF16 output per element.
constexpr int64_t MXFP_WORK_BYTES_PER_ELEMENT = B32_DTYPE_BYTES + B32_DTYPE_BYTES + B16_DTYPE_BYTES;
// A packed FP4 block also contains an integral number of scale groups.
constexpr int64_t MXFP_H_TILE_ALIGN = UB_BLOCK_BYTES * FP4_ELEMENTS_PER_BYTE;
static_assert(MXFP_H_TILE_ALIGN % MXFP_SCALE_GROUP_SIZE == 0);

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
    ge::graphStatus DoOpTilingMxfp();
    ge::graphStatus DoOpTilingNonquant();
    void SetBsTiling(int64_t bsCoreNum);
    void SetHTiling(int64_t hAlign, int64_t hCapacity, int64_t &bsCoreNum);
    void DoOpTilingSplitH(int64_t hAlign, int64_t hCapacity, int64_t &bsCoreNum);
    void DoOpTilingFullK(int64_t bsCoreNum, int64_t hAlign, int64_t kIn);
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
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ubSize_ = ubSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttentionWorkerCombineTilingArch35::DoGetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("AttentionWorkerCombine", "context", "nullptr", "Must not be null."),
                return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto hiddenSize = attrs->GetInt(HIDDEN_SIZE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, hiddenSize);
    auto tokenDtype = attrs->GetInt(TOKEN_DTYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tokenDtype);
    tokenDtype_ = *tokenDtype;
    auto needSchedule = attrs->GetInt(NEED_SCHEDULE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, needSchedule);

    OP_CHECK_IF(tokenDtype_ < 0 || tokenDtype_ > 4,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "token_dtype",
                    std::to_string(tokenDtype_).c_str(), "Must be 0, 1, 2, 3 or 4."), return ge::GRAPH_FAILED);
    const bool mxfp = tokenDtype_ >= 2;
    const auto *shapePtr = context_->GetInputShape(EXPERT_SCALES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapePtr);
    const auto &shape = shapePtr->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != (mxfp ? 3U : 2U),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "expert_scales",
                    std::to_string(shape.GetDimNum()).c_str(), "Rank must be 3 for MXFP, otherwise 2."),
                return ge::GRAPH_FAILED);
    const int64_t batchSize = shape.GetDim(0);
    // MXFP includes the shared expert in the scale/token row count.
    const int64_t k = shape.GetDim(1);
    const int64_t routedK = mxfp ? k - 1 : k;
    OP_CHECK_IF(batchSize <= 0 || routedK <= 0 || routedK > K_UPPER_BOUND || *hiddenSize <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "R,K,H",
                    (std::to_string(batchSize) + "," + std::to_string(routedK) + "," +
                     std::to_string(*hiddenSize)).c_str(), "R/H must be positive and K must be in [1,64]."),
                return ge::GRAPH_FAILED);
    const auto *scaleDesc = context_->GetInputDesc(EXPERT_SCALES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleDesc);
    OP_CHECK_IF(scaleDesc->GetDataType() != (mxfp ? ge::DT_FLOAT8_E8M0 : ge::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "expert_scales",
                    Ops::Base::ToString(scaleDesc->GetDataType()).c_str(),
                    mxfp ? "FLOAT8_E8M0" : "FLOAT"),
                return ge::GRAPH_FAILED);
    if (mxfp) {
        OP_CHECK_IF(shape.GetDim(2) != AlignUp(CeilDiv(*hiddenSize, 32), 2),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "expert_scales dim 2",
                        std::to_string(shape.GetDim(2)).c_str(), "Must equal align_up(ceil(H/32),2)."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(*needSchedule != 0 && *needSchedule != 1,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "need_schedule",
                        std::to_string(*needSchedule).c_str(), "MXFP need_schedule must be 0 or 1."),
                    return ge::GRAPH_FAILED);
        const auto *outDesc = context_->GetOutputDesc(0);
        OP_CHECK_NULL_WITH_CONTEXT(context_, outDesc);
        OP_CHECK_IF(outDesc->GetDataType() != ge::DT_BF16,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "y",
                        Ops::Base::ToString(outDesc->GetDataType()).c_str(), "BF16"), return ge::GRAPH_FAILED);
    }

    batchSize_ = batchSize;
    k_ = k;
    hiddenSize_ = *hiddenSize;
    needSchedule_ = *needSchedule;
    return ge::GRAPH_SUCCESS;
}

void AttentionWorkerCombineTilingArch35::SetBsTiling(int64_t bsCoreNum)
{
    tilingData->BsSplitFactor = 1;
    tilingData->BsSplitCoreNum = bsCoreNum;
    const int64_t factor = CeilDiv(batchSize_, bsCoreNum);
    tilingData->mainCoreBsLoopNum = factor;
    const int64_t mainCores = batchSize_ / factor;
    const int64_t tailCores = bsCoreNum - mainCores;
    tilingData->tailCoreBsLoopNum = tailCores == 0 ? factor : (batchSize_ - mainCores * factor) / tailCores;
}

void AttentionWorkerCombineTilingArch35::DoOpTilingFullK(int64_t bsCoreNum, int64_t hAlign, int64_t kIn)
{
    tilingData->usedCoreNum = bsCoreNum;
    SetBsTiling(bsCoreNum);

    tilingData->HSplitFactor = hAlign;
    tilingData->HSplitCoreNum = 1;
    tilingData->mainCoreHLoopNum = 1;
    tilingData->tailCoreHLoopNum = 1;

    tilingData->KSplitFactor = kIn;
    tilingData->KSplitTailFactor = kIn;
    tilingData->KSplitLoopNum = 1;

    tilingKey_ = TILING_KEY_DIVIDE_BS_FP16;
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

ge::graphStatus AttentionWorkerCombineTilingArch35::DoOpTilingMxfp()
{
    // E = k_ includes the shared expert. Keep the nonquant BS -> K -> H decision order.
    // A tile holds N packed input rows, one decoded FP32 row, one FP32 sum and BF16 output.
    // Readiness flags reuse the input/output queues before/after token computation.
    const int64_t available = static_cast<int64_t>(ubSize_);
    const int64_t elementsPerByte = tokenDtype_ == TOKEN_DTYPE_MXFP4 ?
                                    FP4_ELEMENTS_PER_BYTE : FP8_ELEMENTS_PER_BYTE;
    const int64_t hAlign = AlignUp(hiddenSize_, MXFP_SCALE_GROUP_SIZE);
    const int64_t inputRow = AlignUp(CeilDiv(hAlign, elementsPerByte), UB_BLOCK_BYTES);
    int64_t kTile = 1;
    int64_t bsCores = CeilDiv(batchSize_, CeilDiv(batchSize_, static_cast<int64_t>(coreNum_)));

    // BS/K process a complete H row on one core; the H branch overrides these fields.
    tilingData->HSplitFactor = hAlign;
    tilingData->HSplitTailFactor = hiddenSize_ % hAlign;
    tilingData->HSplitCoreNum = 1;
    tilingData->mainCoreHLoopNum = 1;
    tilingData->tailCoreHLoopNum = 1;

    // Divide before comparing to avoid multiplying E by a potentially large H.
    const int64_t fullRowCapacity = hAlign <= available / MXFP_WORK_BYTES_PER_ELEMENT ?
        (available - MXFP_WORK_BYTES_PER_ELEMENT * hAlign) / inputRow : 0;
    if (fullRowCapacity >= k_) {
        kTile = k_;
        tilingKey_ = TILING_KEY_DIVIDE_BS_MXFP_BASE + tokenDtype_;
    } else if (fullRowCapacity >= 1) {
        kTile = fullRowCapacity;
        tilingKey_ = TILING_KEY_DIVIDE_K_MXFP_BASE + tokenDtype_;
    } else {
        // Budget one aligned input block and its decoded, accumulation and output buffers.
        const int64_t inputBlockBytes = MXFP_H_TILE_ALIGN / elementsPerByte;
        const int64_t bytesPerHBlock = inputBlockBytes + MXFP_H_TILE_ALIGN * MXFP_WORK_BYTES_PER_ELEMENT;
        const int64_t hTile = available / bytesPerHBlock * MXFP_H_TILE_ALIGN;
        tilingKey_ = TILING_KEY_DIVIDE_H_MXFP_BASE + tokenDtype_;
        SetHTiling(hiddenSize_, hTile, bsCores);
    }

    SetBsTiling(bsCores);
    tilingData->KSplitFactor = kTile;
    tilingData->KSplitLoopNum = k_ / kTile;
    tilingData->KSplitTailFactor = k_ % kTile;
    tilingData->usedCoreNum = bsCores * tilingData->HSplitCoreNum;
    context_->SetBlockDim(tilingData->usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttentionWorkerCombineTilingArch35::CalcOpTiling()
{
    tilingData = context_->GetTilingData<AttentionWorkerCombineRegbaseTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);

    tilingData->BS = batchSize_;
    tilingData->K = k_;
    tilingData->H = hiddenSize_;
    tilingData->needSchedule = needSchedule_;

    if (tokenDtype_ >= 2) {
        return DoOpTilingMxfp();
    }

    return DoOpTilingNonquant();
}

void AttentionWorkerCombineTilingArch35::SetHTiling(
    int64_t hAlign, int64_t hCapacity, int64_t &bsCoreNum)
{
    int64_t hCores = 1;
    int64_t mainLoops = CeilDiv(hAlign, hCapacity);
    int64_t tailLoops = mainLoops;
    const int64_t maxCoreNum = static_cast<int64_t>(coreNum_ * 0.8);
    if (needSchedule_ == 0 && bsCoreNum <= maxCoreNum) {
        int64_t lastCoreNum = 1;
        int64_t bestHCores = 1;
        SelectBestHCore(hAlign, hCapacity, bsCoreNum, lastCoreNum, bestHCores);
        const int64_t hBlocks = CeilDiv(hAlign, hCapacity);
        mainLoops = CeilDiv(hBlocks, bestHCores);
        hCores = CeilDiv(hBlocks, mainLoops);
        tailLoops = hBlocks % mainLoops;
        if (tailLoops == 0) {
            tailLoops = mainLoops;
        }
    }
    tilingData->HSplitFactor = hCapacity;
    tilingData->HSplitTailFactor = hAlign % hCapacity;
    tilingData->HSplitCoreNum = hCores;
    tilingData->mainCoreHLoopNum = mainLoops;
    tilingData->tailCoreHLoopNum = tailLoops;
}

void AttentionWorkerCombineTilingArch35::DoOpTilingSplitH(
    int64_t hAlign, int64_t hCapacity, int64_t &bsCoreNum)
{
    SetHTiling(hAlign, hCapacity, bsCoreNum);
    tilingData->KSplitFactor = 1;
    tilingData->KSplitTailFactor = 0;
    tilingData->KSplitLoopNum = k_;
    tilingKey_ = TILING_KEY_DIVIDE_H_FP16;
}

ge::graphStatus AttentionWorkerCombineTilingArch35::DoOpTilingNonquant()
{
    const int64_t availableUb = static_cast<int64_t>(ubSize_);
    // k+1行专家全载ub
    const int64_t hFullK = AlignDown(availableUb /
        (2 * (k_ + 1) * B16_DTYPE_BYTES + 2 * B16_DTYPE_BYTES + 2 * B32_DTYPE_BYTES), H_ALIGN_SIZE);
    // 单行专家搬ub
    const int64_t hSplitK = AlignDown(availableUb /
        (2 * B16_DTYPE_BYTES + 2 * B16_DTYPE_BYTES + 2 * B32_DTYPE_BYTES), H_ALIGN_SIZE);
    // hAlign 按512对齐
    const int64_t hAlign = AlignUp(hiddenSize_, H_ALIGN_SIZE);
    // bsCoreNum 实际用的核
    int64_t bsCoreNum = CeilDiv(batchSize_, CeilDiv(batchSize_, static_cast<int64_t>(coreNum_)));
    if (hAlign <= hFullK) {
        DoOpTilingFullK(bsCoreNum, hAlign, k_);
        return ge::GRAPH_SUCCESS;
    }
    if (hAlign > hSplitK) {
        DoOpTilingSplitH(hAlign, hSplitK, bsCoreNum);
    } else {
        const int64_t kIn = std::min(k_, hSplitK / hAlign);
        tilingData->HSplitFactor = hAlign;
        tilingData->HSplitTailFactor = 0;
        tilingData->HSplitCoreNum = 1;
        tilingData->mainCoreHLoopNum = 1;
        tilingData->tailCoreHLoopNum = 1;
        tilingData->KSplitFactor = kIn;
        tilingData->KSplitTailFactor = k_ % kIn;
        tilingData->KSplitLoopNum = k_ / kIn;
        tilingKey_ = TILING_KEY_DIVIDE_K_FP16;
    }
    SetBsTiling(bsCoreNum);
    tilingData->usedCoreNum = bsCoreNum * tilingData->HSplitCoreNum;
    context_->SetBlockDim(tilingData->usedCoreNum);
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
