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
 * \file fused_causal_conv1d_cut_bsh_tiling_arch35.cpp
 * \brief
 */

#include "fused_causal_conv1d_cut_bsh_tiling_arch35.h"

namespace optiling {
constexpr uint64_t DIM_0 = 0;
constexpr uint64_t DIM_1 = 1;
constexpr uint64_t DIM_2 = 2;

constexpr uint64_t INPUT_X_INDEX = 0;
constexpr uint64_t INPUT_WEIGHT_INDEX = 1;
constexpr uint64_t INPUT_CACHE_STATES_INDEX = 2;
constexpr uint64_t INPUT_QUERY_START_LOC_INDEX = 3;
constexpr uint64_t INPUT_CACHE_INDICES_INDEX = 4;
constexpr uint64_t INPUT_INITIAL_STATE_MODE_INDEX = 5;

constexpr int32_t ATTR_ACTIVATION_MODE_INDEX = 0;
constexpr int32_t ATTR_PAD_SLOT_ID_INDEX = 1;
constexpr int32_t ATTR_RUN_MODE_INDEX = 2;
constexpr int32_t ATTR_RESIDUAL_CONNECTION_INDEX = 3;

constexpr uint64_t OUTPUT_Y_INDEX = 0;
constexpr uint64_t OUTPUT_CACHE_STATES_INDEX = 1;

constexpr uint64_t X_DIM_NUM = 2;
constexpr uint64_t WEIGHT_DIM_NUM = 2;
constexpr uint64_t CACHE_STATES_DIM_NUM = 3;
constexpr uint64_t SEQ_START_INDEX_DIM_NUM = 1;

constexpr uint64_t DIM_MIN = 128;
constexpr uint64_t DIM_MAX = 16384;
constexpr uint64_t DIM_ALIGN = 16;
constexpr uint64_t CU_SEQ_LEN_MIN = 1;
constexpr uint64_t CU_SEQ_LEN_MAX = 65536;
constexpr uint64_t BATCH_MIN = 1;
constexpr uint64_t BATCH_MAX = 256;
constexpr uint64_t KERNEL_WIDTH_MAX = 6;

constexpr uint64_t DIM_ALIGN_ELEMENTS = 128;  // 256 bytes / 2 bytes per element (fp16/bf16)
constexpr uint64_t SYSTEM_RESERVED_UB_SIZE = 8 * 1024;  // 8 KB system reserved UB space
constexpr uint64_t DOUBLE_BUFFER_NUM = 2;
constexpr uint64_t TILING_KEY_BSH_BF16 = 10000UL;
constexpr uint64_t TILING_KEY_BSH_FP16 = 10001UL;
constexpr uint64_t SYS_WORKSPACE_SIZE = static_cast<uint64_t>(16 * 1024 * 1024);

bool FusedCausalConv1dCutBSHTiling::IsCapable()
{
    return true;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::GetPlatformInfo()
{
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platform info is null");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    totalCoreNum_ = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
    if (totalCoreNum_ == 0UL) {
        OP_LOGE(context_->GetNodeName(), "coreNum is 0");
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == static_cast<uint64_t>(0)) {
        OP_LOGE(context_->GetNodeName(), "ubSize is 0");
        return ge::GRAPH_FAILED;
    }

    // 核内 UB 有固定 8KB 的系统保留空间，需要扣除
    if (ubSize <= SYSTEM_RESERVED_UB_SIZE) {
        OP_LOGE(context_->GetNodeName(), "ubSize %lu is too small, must be > %lu", ubSize, SYSTEM_RESERVED_UB_SIZE);
        return ge::GRAPH_FAILED;
    }
    ubSize_ = ubSize - SYSTEM_RESERVED_UB_SIZE;

    return ge::GRAPH_SUCCESS;
}

// 检查输入数据类型
ge::graphStatus FusedCausalConv1dCutBSHTiling::CheckInputDtype()
{
    if (xType_ != ge::DataType::DT_FLOAT16 && xType_ != ge::DataType::DT_BF16) {
        OP_LOGE(context_->GetNodeName(), "X dtype must be fp16 or bf16, but got: %s",
                Ops::Base::ToString(xType_).c_str());
        return ge::GRAPH_FAILED;
    }

    if (weightType_ != xType_) {
        OP_LOGE(context_->GetNodeName(), "Weight dtype must equal to X dtype. X dtype: %s, Weight dtype: %s",
                Ops::Base::ToString(xType_).c_str(), Ops::Base::ToString(weightType_).c_str());
        return ge::GRAPH_FAILED;
    }

    auto cacheStatesType = context_->GetInputDesc(INPUT_CACHE_STATES_INDEX)->GetDataType();
    if (cacheStatesType != xType_) {
        OP_LOGE(context_->GetNodeName(), "CacheStates dtype must equal to X dtype. X dtype: %s, CacheStates dtype: %s",
                Ops::Base::ToString(xType_).c_str(), Ops::Base::ToString(cacheStatesType).c_str());
        return ge::GRAPH_FAILED;
    }

    // 检查 cacheIndices (OPTIONAL)
    auto cacheIndicesDesc = context_->GetOptionalInputDesc(INPUT_CACHE_INDICES_INDEX);
    if (cacheIndicesDesc != nullptr) {
        auto cacheIndicesType = cacheIndicesDesc->GetDataType();
        if (cacheIndicesType != ge::DataType::DT_INT32) {
            OP_LOGE(context_->GetNodeName(), "CacheIndices dtype must be INT32, but got: %s",
                    Ops::Base::ToString(cacheIndicesType).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    // 检查 queryStartLoc (OPTIONAL)
    auto seqStartIndexDesc = context_->GetOptionalInputDesc(INPUT_QUERY_START_LOC_INDEX);
    if (seqStartIndexDesc != nullptr) {
        auto seqStartIndexType = seqStartIndexDesc->GetDataType();
        if (seqStartIndexType != ge::DataType::DT_INT32) {
            OP_LOGE(context_->GetNodeName(), "SeqStartIndex dtype must be INT32, but got: %s",
                    Ops::Base::ToString(seqStartIndexType).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

// 检查输入数据维度
ge::graphStatus FusedCausalConv1dCutBSHTiling::CheckXDim()
{
    uint64_t xDimNum = xShape_.GetDimNum();
    OP_CHECK_IF(xDimNum != X_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "X dim must be 2, but got: %lu", xDimNum),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(cuSeqLen_ >= CU_SEQ_LEN_MIN && cuSeqLen_ <= CU_SEQ_LEN_MAX),
                OP_LOGE(context_->GetNodeName(), "cu_seq_len must in [%lu, %lu], but got: %lu",
                        CU_SEQ_LEN_MIN, CU_SEQ_LEN_MAX, cuSeqLen_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(dim_ >= DIM_MIN && dim_ <= DIM_MAX && dim_ % DIM_ALIGN_ELEMENTS == 0),
                OP_LOGE(context_->GetNodeName(), "dim must be > %lu, <= %lu and be multiple of %lu, but got: %lu",
                        DIM_ALIGN_ELEMENTS, DIM_MAX, DIM_ALIGN_ELEMENTS, dim_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::CheckWeightDim()
{
    uint64_t weightDimNum = weightShape_.GetDimNum();
    OP_CHECK_IF(weightDimNum != WEIGHT_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "Weight dim must be 2, but got: %lu", weightDimNum),
                return ge::GRAPH_FAILED);
    uint64_t weightDim = weightShape_.GetDim(DIM_1);
    OP_CHECK_IF(weightDim != dim_,
                OP_LOGE(context_->GetNodeName(), "Weight dim[1] must equal to X dim[1], X dim: %lu, Weight dim: %lu",
                        dim_, weightDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kernelWidth_ != 3,
                OP_LOGE(context_->GetNodeName(), "Kernel width must be 3, but got: %lu", kernelWidth_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::CheckCacheStatesDim()
{
    uint64_t cacheStatesDimNum = cacheStatesShape_.GetDimNum();
    OP_CHECK_IF(cacheStatesDimNum != CACHE_STATES_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "CacheStates dim must be 3, but got: %lu", cacheStatesDimNum),
                return ge::GRAPH_FAILED);
    uint64_t cacheStatesDim1 = cacheStatesShape_.GetDim(DIM_1);
    OP_CHECK_IF(cacheStatesDim1 != (kernelWidth_ - 1),
                OP_LOGE(context_->GetNodeName(), "CacheStates dim[1] must equal to K-1=%lu, but got: %lu",
                        kernelWidth_ - 1, cacheStatesDim1),
                return ge::GRAPH_FAILED);
    uint64_t cacheStatesDim2 = cacheStatesShape_.GetDim(DIM_2);
    OP_CHECK_IF(cacheStatesDim2 != dim_,
                OP_LOGE(context_->GetNodeName(), "CacheStates dim[2] must equal to dim=%lu, but got: %lu",
                        dim_, cacheStatesDim2),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::CheckIndexDims()
{
    auto seqStartIndexStorageShape = context_->GetOptionalInputShape(INPUT_QUERY_START_LOC_INDEX);
    OP_CHECK_IF(seqStartIndexStorageShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "QueryStartLoc must be provided"),
                return ge::GRAPH_FAILED);
    auto cacheIndicesShape = context_->GetOptionalInputShape(INPUT_CACHE_INDICES_INDEX);
    OP_CHECK_IF(cacheIndicesShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "CacheIndices must be provided"),
                return ge::GRAPH_FAILED);
    auto seqStartIndexShape = seqStartIndexStorageShape->GetOriginShape();
    uint64_t seqStartIndexDimNum = seqStartIndexShape.GetDimNum();
    OP_CHECK_IF(seqStartIndexDimNum != SEQ_START_INDEX_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "SeqStartIndex dim must be 1, but got: %lu", seqStartIndexDimNum),
                return ge::GRAPH_FAILED);
    uint64_t seqStartIndexDim0 = seqStartIndexShape.GetDim(DIM_0);
    OP_CHECK_IF(seqStartIndexDim0 != (batch_ + 1),
                OP_LOGE(context_->GetNodeName(), "SeqStartIndex dim[0] must equal to batch+1=%lu, but got: %lu",
                        batch_ + 1, seqStartIndexDim0),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::CheckInputDim()
{
    if (CheckXDim() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (CheckWeightDim() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (CheckCacheStatesDim() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (CheckIndexDims() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    OP_CHECK_IF(!(batch_ >= BATCH_MIN && batch_ <= BATCH_MAX),
                OP_LOGE(context_->GetNodeName(), "batch must in [%lu, %lu], but got: %lu",
                        BATCH_MIN, BATCH_MAX, batch_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 检查输入参数
ge::graphStatus FusedCausalConv1dCutBSHTiling::CheckInputParams()
{
    if (CheckInputDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckInputDim() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::CheckOutputParams()
{
    auto outputYDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYDesc);
    auto outputYType = outputYDesc->GetDataType();
    OP_CHECK_IF(xType_ != outputYType,
                OP_LOGE(context_->GetNodeName(), "Output Y dtype must equal to X dtype. X dtype: %s, Y dtype: %s",
                        Ops::Base::ToString(xType_).c_str(), Ops::Base::ToString(outputYType).c_str()),
                return ge::GRAPH_FAILED);

    auto outputCacheStatesDesc = context_->GetOutputDesc(OUTPUT_CACHE_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputCacheStatesDesc);
    auto outputCacheStatesType = outputCacheStatesDesc->GetDataType();
    OP_CHECK_IF(xType_ != outputCacheStatesType,
                OP_LOGE(context_->GetNodeName(),
                        "Output CacheStates dtype must equal to X dtype. X dtype: %s, CacheStates dtype: %s",
                        Ops::Base::ToString(xType_).c_str(), Ops::Base::ToString(outputCacheStatesType).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::GetInputShapes()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_X_INDEX));
    xShape_ = context_->GetInputShape(INPUT_X_INDEX)->GetOriginShape();
    cuSeqLen_ = xShape_.GetDim(DIM_0);
    dim_ = xShape_.GetDim(DIM_1);

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_WEIGHT_INDEX));
    weightShape_ = context_->GetInputShape(INPUT_WEIGHT_INDEX)->GetOriginShape();
    kernelWidth_ = weightShape_.GetDim(DIM_0);

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_CACHE_STATES_INDEX));
    cacheStatesShape_ = context_->GetInputShape(INPUT_CACHE_STATES_INDEX)->GetOriginShape();

    auto seqStartIndexStorageShape = context_->GetOptionalInputShape(INPUT_QUERY_START_LOC_INDEX);
    if (seqStartIndexStorageShape != nullptr) {
        seqStartIndexShape_ = seqStartIndexStorageShape->GetOriginShape();
        batch_ = seqStartIndexShape_.GetDim(DIM_0) - 1;
    } else {
        batch_ = 1;
        seqStartIndexShape_ = gert::Shape({0, static_cast<int64_t>(cuSeqLen_)});
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::GetInputDtypes()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INPUT_X_INDEX));
    xType_ = context_->GetInputDesc(INPUT_X_INDEX)->GetDataType();

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INPUT_WEIGHT_INDEX));
    weightType_ = context_->GetInputDesc(INPUT_WEIGHT_INDEX)->GetDataType();

    xDtypeSize_ = GetSizeByDataType(xType_);
    OP_CHECK_IF(xDtypeSize_ == 0,
                OP_LOGE(context_->GetNodeName(), "FusedCausalConv1dCutBSH get X dtype[%s] size is 0.",
                        Ops::Base::ToString(xType_).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::GetInputStrides()
{
    if (context_->InputIsView(INPUT_X_INDEX)) {
        auto* xStride = context_->GetInputStride(INPUT_X_INDEX);
        OP_CHECK_IF(xStride == nullptr || xStride->GetDimNum() == 0,
                    OP_LOGE(context_->GetNodeName(), "x stride is invalid."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(xStride->GetDimNum() != xShape_.GetDimNum(),
                    OP_LOGE(context_->GetNodeName(), "The number of dimensions in x stride must match that of x shape."),
                    return ge::GRAPH_FAILED);
        xStride_ = xStride->GetStride(DIM_0);
    } else {
        xStride_ = dim_;
    }

    if (context_->InputIsView(INPUT_CACHE_STATES_INDEX)) {
        auto* cacheStride = context_->GetInputStride(INPUT_CACHE_STATES_INDEX);
        OP_CHECK_IF(cacheStride == nullptr || cacheStride->GetDimNum() == 0,
                    OP_LOGE(context_->GetNodeName(), "cache_states stride is invalid."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(cacheStride->GetDimNum() != cacheStatesShape_.GetDimNum(),
                    OP_LOGE(context_->GetNodeName(), "The number of dimensions in cache_states stride must match that of cache_states shape."),
                    return ge::GRAPH_FAILED);
        cacheStride0_ = cacheStride->GetStride(DIM_0);
        cacheStride1_ = cacheStride->GetStride(DIM_1);
    } else {
        cacheStride0_ = (kernelWidth_ - 1) * dim_;
        cacheStride1_ = dim_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("FusedCausalConv1dCutBSH", "context is null"), return ge::GRAPH_FAILED);

    if (GetInputShapes() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (GetInputDtypes() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;

    padSlotId_ = -1;
    if (context_->GetAttrs() != nullptr && context_->GetAttrs()->GetInt(ATTR_PAD_SLOT_ID_INDEX) != nullptr) {
        padSlotId_ = *(context_->GetAttrs()->GetInt(ATTR_PAD_SLOT_ID_INDEX));
    }
    residualConnection_ = 0;
    if (context_->GetAttrs() != nullptr && context_->GetAttrs()->GetInt(ATTR_RESIDUAL_CONNECTION_INDEX) != nullptr) {
        residualConnection_ = *(context_->GetAttrs()->GetInt(ATTR_RESIDUAL_CONNECTION_INDEX));
    }

    if (GetInputStrides() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;

    OP_CHECK_IF(CheckInputParams() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "FusedCausalConv1dCutBSH CheckInputParams FAILED."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOutputParams() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "FusedCausalConv1dCutBSH CheckOutputParams FAILED."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 辅助函数：计算切cu_seq_len时的核间切分信息（均分多尾核策略）
FusedCausalConv1dCutBSHTiling::CuSeqLenSplitInfo FusedCausalConv1dCutBSHTiling::CalculateCuSeqLenSplitInfo(
    uint64_t cuSeqLen, uint64_t bsOverlap, uint64_t coreNum) const
{
    CuSeqLenSplitInfo info;

    if (coreNum == 0) {
        return info;
    }

    // 均分策略：将带重叠的总长度均分到所有核
    // effectiveTotal = cuSeqLen + (coreNum - 1) * overlap
    info.effectiveTotal = cuSeqLen + (coreNum - 1) * bsOverlap;

    // 向下取整的基础长度
    info.baseLen = info.effectiveTotal / coreNum;

    // 余数（需要多分配的块数）
    info.remainder = info.effectiveTotal % coreNum;

    // 前 remainder 个核是大核，后面是小核
    if (info.remainder > 0) {
        info.blockFactor = info.baseLen + 1;      // 大核载入长度
        info.blockTailFactor = info.baseLen;      // 小核载入长度
    } else {
        // 所有核均匀分配
        info.blockFactor = info.baseLen;
        info.blockTailFactor = info.baseLen;
    }
    info.realCoreNum = coreNum;               // 所有核都使用

    return info;
}

// 计算二维切分时的tiling（支持不均匀切分 + dim循环）
ge::graphStatus FusedCausalConv1dCutBSHTiling::CalcCoreUbTiling(
    uint64_t coreDim, uint64_t coreBS, uint64_t bsBlockFactor,
    int64_t availableUbSize, uint64_t weightCacheCoeffPerDim, uint64_t bsOverlap,
    uint64_t& ubFactorBS, uint64_t& ubFactorDim,
    uint64_t& loopNumBS, uint64_t& ubTailFactorBS,
    uint64_t& loopNumDim, uint64_t& ubTailFactorDim)
{
    int64_t maxUbDim = availableUbSize /
        (weightCacheCoeffPerDim + coreBS * xDtypeSize_ * DOUBLE_BUFFER_NUM);
    maxUbDim = (maxUbDim / DIM_ALIGN_ELEMENTS) * DIM_ALIGN_ELEMENTS;

    if (maxUbDim >= static_cast<int64_t>(DIM_ALIGN_ELEMENTS)) {
        ubFactorBS = coreBS;
        ubFactorDim = (std::min(static_cast<uint64_t>(maxUbDim), coreDim) /
                       DIM_ALIGN_ELEMENTS) * DIM_ALIGN_ELEMENTS;
        if (ubFactorDim == 0) ubFactorDim = DIM_ALIGN_ELEMENTS;
    } else {
        ubFactorDim = DIM_ALIGN_ELEMENTS;
        int64_t availableForX = availableUbSize - weightCacheCoeffPerDim * ubFactorDim;
        int64_t maxBS = availableForX / (ubFactorDim * xDtypeSize_ * DOUBLE_BUFFER_NUM);
        ubFactorBS = static_cast<uint64_t>(std::min(std::max(maxBS, static_cast<int64_t>(1)),
                                                    static_cast<int64_t>(coreBS)));
        if (ubFactorBS == 0) {
            OP_LOGE(context_->GetNodeName(), "UB size is not enough for tiling");
            return ge::GRAPH_FAILED;
        }
    }

    loopNumDim = (coreDim <= ubFactorDim) ? 1 : (coreDim + ubFactorDim - 1) / ubFactorDim;
    ubTailFactorDim = (coreDim <= ubFactorDim) ? ubFactorDim : coreDim - (loopNumDim - 1) * ubFactorDim;

    if (bsBlockFactor <= ubFactorBS) {
        loopNumBS = 1;
    } else {
        uint64_t remaining = bsBlockFactor - ubFactorBS;
        loopNumBS = 1 + Ops::Base::CeilDiv(remaining, ubFactorBS - bsOverlap);
    }
    uint64_t lastLoopInput = bsBlockFactor - (loopNumBS - 1) * (ubFactorBS - bsOverlap);
    ubTailFactorBS = std::min(lastLoopInput, ubFactorBS);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::Calculate2DTiling()
{
    uint64_t bsOverlap = kernelWidth_ - 1;
    uint64_t fixedUbSize = (2 * batch_ + 1 + batch_) * sizeof(int32_t);
    uint64_t weightCacheCoeffPerDim = (kernelWidth_ + kernelWidth_ - 1) * xDtypeSize_;
    int64_t availableUbSize = static_cast<int64_t>(ubSize_) - fixedUbSize;

    if (CalcCoreUbTiling(dimBlockFactor_, bsBlockFactor_, bsBlockFactor_,
                         availableUbSize, weightCacheCoeffPerDim, bsOverlap,
                         ubFactorBS_, ubFactorDim_, loopNumBS_, ubTailFactorBS_,
                         loopNumDim_, ubTailFactorDim_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint64_t tailCoreDim = (dimRemainderCores_ > 0) ? dimBlockTailFactor_ : dimBlockFactor_;
    if (CalcCoreUbTiling(tailCoreDim, bsBlockTailFactor_, bsBlockTailFactor_,
                         availableUbSize, weightCacheCoeffPerDim, bsOverlap,
                         tailBlockubFactorBS_, tailBlockubFactorDim_,
                         tailBlockloopNumBS_, tailBlockubTailFactorBS_,
                         tailBlockloopNumDim_, tailBlockubTailFactorDim_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::SearchBestCoreSplit(
    uint64_t N, uint64_t bsOverlap, uint64_t& bestDimCores, CuSeqLenSplitInfo& bestBSSplitInfo)
{
    uint64_t bestUsed = 0;
    for (uint64_t dc = N; dc >= 1; --dc) {
        uint64_t base = N / dc;
        if (base == 0) continue;

        uint64_t maxAllowedBSByCore = totalCoreNum_ / dc;
        if (maxAllowedBSByCore == 0) continue;

        uint64_t maxAllowedBSBySeqLen = (cuSeqLen_ > bsOverlap) ? (cuSeqLen_ - bsOverlap) : 1;
        uint64_t maxAllowedBS = std::min(maxAllowedBSByCore, maxAllowedBSBySeqLen);

        auto splitInfo = CalculateCuSeqLenSplitInfo(cuSeqLen_, bsOverlap, maxAllowedBS);
        uint64_t usedCores = dc * splitInfo.realCoreNum;

        if (usedCores > bestUsed || (usedCores == bestUsed && dc > bestDimCores)) {
            bestDimCores = dc;
            bestUsed = usedCores;
            bestBSSplitInfo = splitInfo;
        }
        if (bestUsed == totalCoreNum_) break;
    }

    if (bestUsed == 0) {
        OP_LOGE(context_->GetNodeName(), "Failed to find valid tiling strategy");
        return ge::GRAPH_FAILED;
    }
    realCoreNum_ = bestUsed;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::ApplyDimSplit(uint64_t N, uint64_t bestDimCores)
{
    if (bestDimCores == 0) {
        OP_LOGE(context_->GetNodeName(), "ApplyDimSplit: bestDimCores is 0");
        return ge::GRAPH_FAILED;
    }
    constexpr uint64_t DIM_GRANULARITY = DIM_ALIGN_ELEMENTS;
    uint64_t base = N / bestDimCores;
    uint64_t remainder = N % bestDimCores;

    dimCoreNum_ = bestDimCores;
    dimRemainderCores_ = remainder;
    dimBlockFactor_ = (remainder > 0 ? base + 1 : base) * DIM_GRANULARITY;
    dimBlockTailFactor_ = base * DIM_GRANULARITY;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::DoOpTiling()
{
    uint64_t bsOverlap = kernelWidth_ - 1;
    constexpr uint64_t DIM_GRANULARITY = DIM_ALIGN_ELEMENTS;

    uint64_t N = dim_ / DIM_GRANULARITY;
    if (N == 0) {
        OP_LOGE(context_->GetNodeName(), "dim %lu is smaller than DIM_GRANULARITY %lu",
                dim_, DIM_GRANULARITY);
        return ge::GRAPH_FAILED;
    }

    uint64_t bestDimCores = 0;
    CuSeqLenSplitInfo bestBSSplitInfo = {};
    if (SearchBestCoreSplit(N, bsOverlap, bestDimCores, bestBSSplitInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (ApplyDimSplit(N, bestDimCores) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;

    bsCoreNum_ = bestBSSplitInfo.realCoreNum;
    bsRemainderCores_ = bestBSSplitInfo.remainder;
    bsBlockFactor_ = bestBSSplitInfo.blockFactor;
    bsBlockTailFactor_ = bestBSSplitInfo.blockTailFactor;

    return Calculate2DTiling();
}

uint64_t FusedCausalConv1dCutBSHTiling::GetTilingKey() const
{
    // 根据数据类型返回不同的 tiling key
    if (xType_ == ge::DataType::DT_BF16) {
        return TILING_KEY_BSH_BF16;
    } else if (xType_ == ge::DataType::DT_FLOAT16) {
        return TILING_KEY_BSH_FP16;
    }
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::GetWorkspaceSize()
{
    workspaceSize_ = SYS_WORKSPACE_SIZE;

    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::PostTiling()
{
    // Set block dimension (number of cores to use)
    context_->SetBlockDim(realCoreNum_);

    // Populate tiling data
    tilingData_.loopNumBS = loopNumBS_;
    tilingData_.loopNumDim = loopNumDim_;
    tilingData_.ubFactorBS = ubFactorBS_;
    tilingData_.ubTailFactorBS = ubTailFactorBS_;
    tilingData_.ubFactorDim = ubFactorDim_;
    tilingData_.ubTailFactorDim = ubTailFactorDim_;
    tilingData_.tailBlockloopNumBS = tailBlockloopNumBS_;
    tilingData_.tailBlockloopNumDim = tailBlockloopNumDim_;
    tilingData_.tailBlockubFactorBS = tailBlockubFactorBS_;
    tilingData_.tailBlockubTailFactorBS = tailBlockubTailFactorBS_;
    tilingData_.tailBlockubFactorDim = tailBlockubFactorDim_;
    tilingData_.tailBlockubTailFactorDim = tailBlockubTailFactorDim_;

    // dim方向核间切分信息
    tilingData_.dimCoreNum = dimCoreNum_;
    tilingData_.dimRemainderCores = dimRemainderCores_;
    tilingData_.dimBlockFactor = dimBlockFactor_;
    tilingData_.dimBlockTailFactor = dimBlockTailFactor_;

    // BS方向核间切分信息
    tilingData_.bsCoreNum = bsCoreNum_;
    tilingData_.bsRemainderCores = bsRemainderCores_;
    tilingData_.bsBlockFactor = bsBlockFactor_;
    tilingData_.bsBlockTailFactor = bsBlockTailFactor_;

    // 核数信息
    tilingData_.realCoreNum = realCoreNum_;

    // 其他参数
    tilingData_.kernelWidth = kernelWidth_;
    tilingData_.cuSeqLen = cuSeqLen_;
    tilingData_.dim = dim_;
    tilingData_.batch = batch_;
    tilingData_.padSlotId = padSlotId_;
    tilingData_.xStride = xStride_;
    tilingData_.cacheStride0 = cacheStride0_;
    tilingData_.cacheStride1 = cacheStride1_;
    tilingData_.residualConnection = residualConnection_;

    // Save tiling data to buffer
    auto tilingDataSize = sizeof(FusedCausalConv1dCutBSHTilingData);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(),
                    context_->GetRawTilingData()->GetCapacity(),
                    reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    return ge::GRAPH_SUCCESS;
}

void FusedCausalConv1dCutBSHTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "cuSeqLen: " << cuSeqLen_ << std::endl;
    info << "dim: " << dim_ << std::endl;
    info << "kernelWidth: " << kernelWidth_ << std::endl;
    info << "batch: " << batch_ << std::endl;
    info << "padSlotId: " << padSlotId_ << std::endl;

    // dim方向核间切分信息
    info << "dimCoreNum: " << dimCoreNum_ << std::endl;
    info << "dimRemainderCores: " << dimRemainderCores_ << std::endl;
    info << "dimBlockFactor: " << dimBlockFactor_ << std::endl;
    info << "dimBlockTailFactor: " << dimBlockTailFactor_ << std::endl;

    // BS方向核间切分信息
    info << "bsCoreNum: " << bsCoreNum_ << std::endl;
    info << "bsRemainderCores: " << bsRemainderCores_ << std::endl;
    info << "bsBlockFactor: " << bsBlockFactor_ << std::endl;
    info << "bsBlockTailFactor: " << bsBlockTailFactor_ << std::endl;

    // 核数信息
    info << "realCoreNum: " << realCoreNum_ << std::endl;

    // 核内切分参数
    info << "loopNumBS: " << loopNumBS_ << std::endl;
    info << "loopNumDim: " << loopNumDim_ << std::endl;
    info << "ubFactorBS: " << ubFactorBS_ << std::endl;
    info << "ubTailFactorBS: " << ubTailFactorBS_ << std::endl;
    info << "ubFactorDim: " << ubFactorDim_ << std::endl;
    info << "ubTailFactorDim: " << ubTailFactorDim_ << std::endl;
    info << "tailBlockloopNumBS: " << tailBlockloopNumBS_ << std::endl;
    info << "tailBlockloopNumDim: " << tailBlockloopNumDim_ << std::endl;
    info << "tailBlockubFactorBS: " << tailBlockubFactorBS_ << std::endl;
    info << "tailBlockubTailFactorBS: " << tailBlockubTailFactorBS_ << std::endl;
    info << "tailBlockubFactorDim: " << tailBlockubFactorDim_ << std::endl;
    info << "tailBlockubTailFactorDim: " << tailBlockubTailFactorDim_ << std::endl;
    info << "residualConnection: " << residualConnection_ << std::endl;
    info << "xStride: " << xStride_ << std::endl;
    info << "cacheStride0: " << cacheStride0_ << std::endl;
    info << "cacheStride1: " << cacheStride1_ << std::endl;

    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

} // namespace optiling
