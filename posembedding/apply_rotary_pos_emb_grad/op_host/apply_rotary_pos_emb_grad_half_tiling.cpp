/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file apply_rotary_pos_emb_grad_half_tiling.cpp
 * \brief
 */
#include "apply_rotary_pos_emb_grad_half_tiling.h"
#include "apply_rotary_pos_emb_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "graph/types.h"

namespace {
static constexpr uint64_t INDEX_GRAD_QUERY_EMBED = 0;
static constexpr uint64_t INDEX_GRAD_KEY_EMBED = 1;
static constexpr uint64_t INDEX_COS = 2;
static constexpr uint64_t INDEX_SIN = 3;
static constexpr uint64_t INDEX_QUERY = 4;
static constexpr uint64_t INDEX_KEY = 5;
static constexpr uint64_t ATTR_INDEX_ROTARY_MODE = 0;
static constexpr uint64_t ATTR_INDEX_LAYOUT = 1;
static constexpr uint64_t INDEX_GRAD_QUERY = 0;
static constexpr uint64_t INDEX_GRAD_KEY = 1;
static constexpr uint64_t INDEX_GRAD_COS = 2;
static constexpr uint64_t INDEX_GRAD_SIN = 3;

static constexpr uint64_t DIM_ZERO = 0;
static constexpr uint64_t DIM_ONE = 1;
static constexpr uint64_t DIM_TWO = 2;
static constexpr uint64_t DIM_THREE = 3;
static constexpr uint64_t DIM_SIZE_THREE = 3;
static constexpr uint64_t DIM_SIZE_FOUR = 4;
static constexpr int64_t BLOCK_SIZE = 32;
static constexpr int64_t DTYPE_SIZE_FP16 = 2;
static constexpr int64_t DTYPE_SIZE_FP32 = 4;
static constexpr int64_t RESERVE_UB_SIZE = 20 * 1024;
static constexpr int64_t IN_OUT_TENSOR_NUM = 7;
static constexpr int64_t TENSOR_NUM = 5;
static constexpr int64_t TENSOR_CONV_NUM = 9;
static constexpr int64_t IN_OUT_TENSOR_SINGLE_NUM = 4;
static constexpr int64_t TENSOR_SINGLE_NUM = 3;
static constexpr int64_t TENSOR_CONV_SINGLE_NUM = 7;
static constexpr uint64_t MASK_NUM_FP32 = 64;
static constexpr uint64_t CHUNK_SIZE = 2;
static constexpr uint64_t DIM_D_MAX = 128;
static constexpr uint64_t DIM_D_MULTIPLE = 16;

static constexpr uint64_t SHAPE_KEY_ALIGN = 0;
static constexpr uint64_t SHAPE_KEY_PAD = 100;
static constexpr uint64_t NEED_BACKWARD_KEY_TRUE = 0;
static constexpr uint64_t NEED_BACKWARD_KEY_FALSE = 1000;

static constexpr uint64_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;

static int64_t GetCeilInt(const int64_t value1, const int64_t value2)
{
    OP_CHECK_IF(value2 == 0, OP_LOGE("ApplyRotaryPosEmbGradHalf", "Divisor is 0 when calling GetCeilInt"),
                return value1);
    return static_cast<int64_t>((value1 + value2 - 1) / value2);
}

static int64_t GetGcdForRotary(const int64_t value1, const int64_t value2)
{
    OP_CHECK_IF(value2 == 0, OP_LOGE("ApplyRotaryPosEmbGradHalf", "Divisor is 0 when calling GetGcdForRotary"),
                return value1);
    int64_t remainder = value1 % value2;
    if (remainder == 0) {
        return value2;
    } else {
        return GetGcdForRotary(value2, remainder);
    }
}

static int64_t GetLcmForRotary(const int64_t value1, const int64_t value2)
{
    int64_t temp = value1 * value2;
    OP_CHECK_IF(value2 == 0, OP_LOGE("ApplyRotaryPosEmbGradHalf", "Divisor is 0 when calling GetLcmForRotary"),
                return temp);
    temp = temp / GetGcdForRotary(value1, value2);
    return temp;
}
} // namespace

namespace optiling {
void ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradAssignParam() // 赋值tiling参数
{
    tiling.applyRopeHalfGradParams.set_hasOptionalInputs(hasOptionalInputs_);
    tiling.applyRopeHalfGradParams.set_layout(layout_);
    tiling.applyRopeHalfGradParams.set_dQShapeSize(dQShapeSize_);
    tiling.applyRopeHalfGradParams.set_dKShapeSize(dKShapeSize_);
    tiling.applyRopeHalfGradParams.set_cosSinShapeSize(cosSinShapeSize_);
    tiling.applyRopeHalfGradParams.set_dQKDimB(dQKDimB_);
    tiling.applyRopeHalfGradParams.set_dimS(dimS_);
    tiling.applyRopeHalfGradParams.set_dQDimN(dQDimN_);
    tiling.applyRopeHalfGradParams.set_dKDimN(dKDimN_);
    tiling.applyRopeHalfGradParams.set_dimD(dimD_);
    tiling.applyRopeHalfGradParams.set_cosSinDimB(cosSinDimB_);
    tiling.applyRopeHalfGradParams.set_cosSinDimN(cosSinDimN_);
    tiling.applyRopeHalfGradParams.set_halfDimDAlignNum(halfDimDAlignNum_);
    tiling.applyRopeHalfGradParams.set_elementsPerCore(elementsPerCore_);
    tiling.applyRopeHalfGradParams.set_lastCoreElements(lastCoreElements_);
    tiling.applyRopeHalfGradParams.set_fullTileLoopsPerCore(fullTileLoopsPerCore_);
    tiling.applyRopeHalfGradParams.set_tailElementsPerCore(tailElementsPerCore_);
    tiling.applyRopeHalfGradParams.set_lastCoreFullTileLoops(lastCoreFullTileLoops_);
    tiling.applyRopeHalfGradParams.set_lastCoreTailElements(lastCoreTailElements_);
    tiling.applyRopeHalfGradParams.set_alignUbSize(alignUbSize_);
    tiling.applyRopeHalfGradParams.set_calcUbSize(calcUbSize_);
    tiling.applyRopeHalfGradParams.set_coreUsed(coreUsed_);
    tiling.applyRopeHalfGradParams.set_cosSinOuterLoops(cosSinOuterLoops_);
    tiling.applyRopeHalfGradParams.set_dQHeadInnerLoops(dQHeadInnerLoops_);
    tiling.applyRopeHalfGradParams.set_dKHeadInnerLoops(dKHeadInnerLoops_);
    tiling.applyRopeHalfGradParams.set_dQOuterLoopGmStride(dQOuterLoopGmStride_);
    tiling.applyRopeHalfGradParams.set_dKOuterLoopGmStride(dKOuterLoopGmStride_);
    tiling.applyRopeHalfGradParams.set_innerPaddedBlockLen(innerPaddedBlockLen_);
    tiling.applyRopeHalfGradParams.set_dQInnerPaddedSrcStride(dQInnerPaddedSrcStride_);
    tiling.applyRopeHalfGradParams.set_dKInnerPaddedSrcStride(dKInnerPaddedSrcStride_);
}

void ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradPrintParam() // 打印结构体参数
{
    OP_LOGD(context_->GetNodeName(), "hasOptionalInputs = %ld", tiling.applyRopeHalfGradParams.get_hasOptionalInputs());
    OP_LOGD(context_->GetNodeName(), "layout = %ld", tiling.applyRopeHalfGradParams.get_layout());
    OP_LOGD(context_->GetNodeName(), "dQShapeSize = %ld", tiling.applyRopeHalfGradParams.get_dQShapeSize());
    OP_LOGD(context_->GetNodeName(), "dKShapeSize = %ld", tiling.applyRopeHalfGradParams.get_dKShapeSize());
    OP_LOGD(context_->GetNodeName(), "cosSinShapeSize = %ld", tiling.applyRopeHalfGradParams.get_cosSinShapeSize());
    OP_LOGD(context_->GetNodeName(), "dQKDimB = %ld", tiling.applyRopeHalfGradParams.get_dQKDimB());
    OP_LOGD(context_->GetNodeName(), "dimS = %ld", tiling.applyRopeHalfGradParams.get_dimS());
    OP_LOGD(context_->GetNodeName(), "dQDimN = %ld", tiling.applyRopeHalfGradParams.get_dQDimN());
    OP_LOGD(context_->GetNodeName(), "dKDimN = %ld", tiling.applyRopeHalfGradParams.get_dKDimN());
    OP_LOGD(context_->GetNodeName(), "dimD = %ld", tiling.applyRopeHalfGradParams.get_dimD());
    OP_LOGD(context_->GetNodeName(), "cosSinDimB = %ld", tiling.applyRopeHalfGradParams.get_cosSinDimB());
    OP_LOGD(context_->GetNodeName(), "cosSinDimN = %ld", tiling.applyRopeHalfGradParams.get_cosSinDimN());
    OP_LOGD(context_->GetNodeName(), "halfDimDAlignNum = %ld", tiling.applyRopeHalfGradParams.get_halfDimDAlignNum());
    OP_LOGD(context_->GetNodeName(), "elementsPerCore = %ld", tiling.applyRopeHalfGradParams.get_elementsPerCore());
    OP_LOGD(context_->GetNodeName(), "lastCoreElements = %ld", tiling.applyRopeHalfGradParams.get_lastCoreElements());
    OP_LOGD(context_->GetNodeName(), "fullTileLoopsPerCore = %ld",
            tiling.applyRopeHalfGradParams.get_fullTileLoopsPerCore());
    OP_LOGD(context_->GetNodeName(), "tailElementsPerCore = %ld",
            tiling.applyRopeHalfGradParams.get_tailElementsPerCore());
    OP_LOGD(context_->GetNodeName(), "lastCoreFullTileLoops = %ld",
            tiling.applyRopeHalfGradParams.get_lastCoreFullTileLoops());
    OP_LOGD(context_->GetNodeName(), "lastCoreTailElements = %ld",
            tiling.applyRopeHalfGradParams.get_lastCoreTailElements());
    OP_LOGD(context_->GetNodeName(), "alignUbSize = %ld", tiling.applyRopeHalfGradParams.get_alignUbSize());
    OP_LOGD(context_->GetNodeName(), "calcUbSize = %ld", tiling.applyRopeHalfGradParams.get_calcUbSize());
    OP_LOGD(context_->GetNodeName(), "coreUsed = %ld", tiling.applyRopeHalfGradParams.get_coreUsed());
    OP_LOGD(context_->GetNodeName(), "cosSinOuterLoops = %ld", tiling.applyRopeHalfGradParams.get_cosSinOuterLoops());
    OP_LOGD(context_->GetNodeName(), "dQHeadInnerLoops = %ld", tiling.applyRopeHalfGradParams.get_dQHeadInnerLoops());
    OP_LOGD(context_->GetNodeName(), "dKHeadInnerLoops = %ld", tiling.applyRopeHalfGradParams.get_dKHeadInnerLoops());
    OP_LOGD(context_->GetNodeName(), "dQOuterLoopGmStride = %ld",
            tiling.applyRopeHalfGradParams.get_dQOuterLoopGmStride());
    OP_LOGD(context_->GetNodeName(), "dKOuterLoopGmStride = %ld",
            tiling.applyRopeHalfGradParams.get_dKOuterLoopGmStride());
    OP_LOGD(context_->GetNodeName(), "innerPaddedBlockLen = %ld",
            tiling.applyRopeHalfGradParams.get_innerPaddedBlockLen());
    OP_LOGD(context_->GetNodeName(), "dQInnerPaddedSrcStride = %ld",
            tiling.applyRopeHalfGradParams.get_dQInnerPaddedSrcStride());
    OP_LOGD(context_->GetNodeName(), "dKInnerPaddedSrcStride = %ld",
            tiling.applyRopeHalfGradParams.get_dKInnerPaddedSrcStride());
}

// 输入Tensor校验并获取shape相关参数
ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradCheckInput()
{
    OP_CHECK_IF(ApplyRopeHalfGradCheckAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Check attr failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ApplyRopeHalfGradCheckGradQueryEmbed() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "Check gradQueryEmbed failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ApplyRopeHalfGradCheckGradKeyEmbed() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "Check gradKeyEmbed failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ApplyRopeHalfGradCheckCos() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Check cos failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ApplyRopeHalfGradCheckSin() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Check sin failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ApplyRopeHalfGradCheckQuery() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Check query failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ApplyRopeHalfGradCheckKey() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Check key failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ApplyRopeHalfGradCheckInputDatatype() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Check input datatype failed."), return ge::GRAPH_FAILED);
    if (hasOptionalInputs_ == 1) {
        OP_CHECK_IF(ApplyRopeHalfGradCheckOptionalInputDatatype() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Check optional input datatype failed."), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradCheckAttr()
{
    OP_CHECK_IF(rotaryMode_ != "half",
                OP_LOGE(context_, "Attr rotaryMode: expected half, but got %s", rotaryMode_.data()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(layout_ < static_cast<int64_t>(ApplyRotaryPosEmbGradLayout::BSND) ||
                    layout_ > static_cast<int64_t>(ApplyRotaryPosEmbGradLayout::TND),
                OP_LOGE(context_, "Attr layout: got invalid value %ld", layout_), return ge::GRAPH_FAILED);

    OP_CHECK_IF(layout_ == static_cast<int64_t>(ApplyRotaryPosEmbGradLayout::BNSD),
                OP_LOGE(context_, "Attr layout: BNSD is currently unsupported"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradCheckGradQueryEmbed()
{
    auto gradQueryEmbedShape = context_->GetInputShape(INDEX_GRAD_QUERY_EMBED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradQueryEmbedShape);
    gradQueryEmbedDimSize_ = gradQueryEmbedShape->GetOriginShape().GetDimNum();
    OP_CHECK_IF(gradQueryEmbedDimSize_ != DIM_SIZE_THREE && gradQueryEmbedDimSize_ != DIM_SIZE_FOUR,
                OP_LOGE(context_, "gradQueryEmbed: Rank must be 3 or 4, but got %lu.", gradQueryEmbedDimSize_),
                return ge::GRAPH_FAILED);

    if (gradQueryEmbedDimSize_ == DIM_SIZE_THREE) {
        dQKDimB_ = 1;
        dimS_ = gradQueryEmbedShape->GetOriginShape().GetDim(DIM_ZERO);
        dQDimN_ = gradQueryEmbedShape->GetOriginShape().GetDim(DIM_ONE);
        dimD_ = gradQueryEmbedShape->GetOriginShape().GetDim(DIM_TWO);
    } else if (layout_ == static_cast<int64_t>(ApplyRotaryPosEmbGradLayout::BSND)) {
        dQKDimB_ = gradQueryEmbedShape->GetOriginShape().GetDim(DIM_ZERO);
        dimS_ = gradQueryEmbedShape->GetOriginShape().GetDim(DIM_ONE);
        dQDimN_ = gradQueryEmbedShape->GetOriginShape().GetDim(DIM_TWO);
        dimD_ = gradQueryEmbedShape->GetOriginShape().GetDim(DIM_THREE);
    } else { // SBND
        dimS_ = gradQueryEmbedShape->GetOriginShape().GetDim(DIM_ZERO);
        dQKDimB_ = gradQueryEmbedShape->GetOriginShape().GetDim(DIM_ONE);
        dQDimN_ = gradQueryEmbedShape->GetOriginShape().GetDim(DIM_TWO);
        dimD_ = gradQueryEmbedShape->GetOriginShape().GetDim(DIM_THREE);
    }

    OP_CHECK_IF(
        dimD_ > DIM_D_MAX || dimD_ <= 0,
        OP_LOGE(context_,
                "gradQueryEmbed: Dimension %ld (D) must be less than or equal to 128 and greater than 0, but got %ld.",
                gradQueryEmbedDimSize_ - 1, dimD_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(dimD_ % DIM_D_MULTIPLE != 0,
                OP_LOGE(context_, "gradQueryEmbed: Dimension %ld (D) must be a multiple of 16, but got %ld.",
                        gradQueryEmbedDimSize_ - 1, dimD_),
                return ge::GRAPH_FAILED);

    dQShapeSize_ = gradQueryEmbedShape->GetStorageShape().GetShapeSize();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradCheckGradKeyEmbed()
{
    auto gradKeyEmbedShape = context_->GetInputShape(INDEX_GRAD_KEY_EMBED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradKeyEmbedShape);
    auto gradKeyEmbedDimSize = gradKeyEmbedShape->GetOriginShape().GetDimNum();
    OP_CHECK_IF(
        gradKeyEmbedDimSize != gradQueryEmbedDimSize_,
        OP_LOGE(context_, "gradKeyEmbed: Rank must be %lu, but got %lu.", gradQueryEmbedDimSize_, gradKeyEmbedDimSize),
        return ge::GRAPH_FAILED);

    if (gradKeyEmbedDimSize == DIM_SIZE_THREE) {
        OP_CHECK_IF(gradKeyEmbedShape->GetOriginShape().GetDim(DIM_ZERO) != dimS_,
                    OP_LOGE(context_, "gradKeyEmbed: Must match gradQueryEmbed on dimension 0, but got %ld vs %ld.",
                            gradKeyEmbedShape->GetOriginShape().GetDim(DIM_ZERO), dimS_), return ge::GRAPH_FAILED);
        dKDimN_ = gradKeyEmbedShape->GetOriginShape().GetDim(DIM_ONE);
        OP_CHECK_IF(gradKeyEmbedShape->GetOriginShape().GetDim(DIM_TWO) != dimD_,
                    OP_LOGE(context_, "gradKeyEmbed: Must match gradQueryEmbed on dimension 2, but got %ld vs %ld.",
                            gradKeyEmbedShape->GetOriginShape().GetDim(DIM_TWO), dimD_), return ge::GRAPH_FAILED);
    } else if (layout_ == static_cast<int64_t>(ApplyRotaryPosEmbGradLayout::BSND)) {
        OP_CHECK_IF(gradKeyEmbedShape->GetOriginShape().GetDim(DIM_ZERO) != dQKDimB_,
                    OP_LOGE(context_, "gradKeyEmbed: Must match gradQueryEmbed on dimension 0, but got %ld vs %ld.",
                            gradKeyEmbedShape->GetOriginShape().GetDim(DIM_ZERO), dQKDimB_), return ge::GRAPH_FAILED);
        OP_CHECK_IF(gradKeyEmbedShape->GetOriginShape().GetDim(DIM_ONE) != dimS_,
                    OP_LOGE(context_, "gradKeyEmbed: Must match gradQueryEmbed on dimension 1, but got %ld vs %ld.",
                            gradKeyEmbedShape->GetOriginShape().GetDim(DIM_ONE), dimS_), return ge::GRAPH_FAILED);
        dKDimN_ = gradKeyEmbedShape->GetOriginShape().GetDim(DIM_TWO);
        OP_CHECK_IF(gradKeyEmbedShape->GetOriginShape().GetDim(DIM_THREE) != dimD_,
                    OP_LOGE(context_, "gradKeyEmbed: Must match gradQueryEmbed on dimension 3, but got %ld vs %ld.",
                            gradKeyEmbedShape->GetOriginShape().GetDim(DIM_THREE), dimD_), return ge::GRAPH_FAILED);
    } else { // SBND
        OP_CHECK_IF(gradKeyEmbedShape->GetOriginShape().GetDim(DIM_ZERO) != dimS_,
                    OP_LOGE(context_, "gradKeyEmbed: Must match gradQueryEmbed on dimension 0, but got %ld vs %ld.",
                            gradKeyEmbedShape->GetOriginShape().GetDim(DIM_ZERO), dimS_), return ge::GRAPH_FAILED);
        OP_CHECK_IF(gradKeyEmbedShape->GetOriginShape().GetDim(DIM_ONE) != dQKDimB_,
                    OP_LOGE(context_, "gradKeyEmbed: Must match gradQueryEmbed on dimension 1, but got %ld vs %ld.",
                            gradKeyEmbedShape->GetOriginShape().GetDim(DIM_ONE), dQKDimB_), return ge::GRAPH_FAILED);
        dKDimN_ = gradKeyEmbedShape->GetOriginShape().GetDim(DIM_TWO);
        OP_CHECK_IF(gradKeyEmbedShape->GetOriginShape().GetDim(DIM_THREE) != dimD_,
                    OP_LOGE(context_, "gradKeyEmbed: Must match gradQueryEmbed on dimension 3, but got %ld vs %ld.",
                            gradKeyEmbedShape->GetOriginShape().GetDim(DIM_THREE), dimD_), return ge::GRAPH_FAILED);
    }

    dKShapeSize_ = gradKeyEmbedShape->GetStorageShape().GetShapeSize();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradCheckCos()
{
    auto cosShape = context_->GetInputShape(INDEX_COS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cosShape);
    auto cosDimSize = cosShape->GetOriginShape().GetDimNum();
    OP_CHECK_IF(cosDimSize != gradQueryEmbedDimSize_, OP_LOGE(context_,
        "cos: Rank must be %lu, but got %lu.", gradQueryEmbedDimSize_, cosDimSize), return ge::GRAPH_FAILED);
    if (cosDimSize == DIM_SIZE_THREE) {
        cosSinDimB_ = 1;
        OP_CHECK_IF(cosShape->GetOriginShape().GetDim(DIM_ZERO) != dimS_,
                    OP_LOGE(context_, "cos: Must match gradQueryEmbed on dimension 0, but got %ld vs %ld.",
                            cosShape->GetOriginShape().GetDim(DIM_ZERO), dimS_), return ge::GRAPH_FAILED);
        cosSinDimN_ = cosShape->GetOriginShape().GetDim(DIM_ONE);
        OP_CHECK_IF(cosSinDimN_ != 1, OP_LOGE(context_, "cos: Dimension 1 (N) must be 1, but got %ld.", cosSinDimN_),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(cosShape->GetOriginShape().GetDim(DIM_TWO) != dimD_,
                    OP_LOGE(context_, "cos: Must match gradQueryEmbed on dimension 2, but got %ld vs %ld.",
                            cosShape->GetOriginShape().GetDim(DIM_TWO), dimD_), return ge::GRAPH_FAILED);
    } else if (layout_ == static_cast<int64_t>(ApplyRotaryPosEmbGradLayout::BSND)) {
        cosSinDimB_ = cosShape->GetOriginShape().GetDim(DIM_ZERO);
        OP_CHECK_IF(cosSinDimB_ != dQKDimB_, OP_LOGE(context_, "cos: Must match gradQueryEmbed on dimension 0, "
            "but got %ld vs %ld.", cosSinDimB_, dQKDimB_), return ge::GRAPH_FAILED);
        OP_CHECK_IF(cosShape->GetOriginShape().GetDim(DIM_ONE) != dimS_,
                    OP_LOGE(context_, "cos: Must match gradQueryEmbed on dimension 1, but got %ld vs %ld.",
                            cosShape->GetOriginShape().GetDim(DIM_ONE), dimS_), return ge::GRAPH_FAILED);
        cosSinDimN_ = cosShape->GetOriginShape().GetDim(DIM_TWO);
        OP_CHECK_IF(cosSinDimN_ != 1, OP_LOGE(context_, "cos: Dimension 2 (N) must be 1, but got %ld.", cosSinDimN_),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(cosShape->GetOriginShape().GetDim(DIM_THREE) != dimD_,
                    OP_LOGE(context_, "cos: Must match gradQueryEmbed on dimension 3, but got %ld vs %ld.",
                            cosShape->GetOriginShape().GetDim(DIM_THREE), dimD_), return ge::GRAPH_FAILED);
    } else {  // SBND
        OP_CHECK_IF(cosShape->GetOriginShape().GetDim(DIM_ZERO) != dimS_,
                    OP_LOGE(context_, "cos: Must match gradQueryEmbed on dimension 0, but got %ld vs %ld.",
                            cosShape->GetOriginShape().GetDim(DIM_ZERO), dimS_), return ge::GRAPH_FAILED);
        cosSinDimB_ = cosShape->GetOriginShape().GetDim(DIM_ONE);
        OP_CHECK_IF(cosSinDimB_ != dQKDimB_,
                    OP_LOGE(context_, "cos: Must match gradQueryEmbed on dimension 1, but got %ld vs %ld.", cosSinDimB_,
                            dQKDimB_), return ge::GRAPH_FAILED);
        cosSinDimN_ = cosShape->GetOriginShape().GetDim(DIM_TWO);
        OP_CHECK_IF(cosSinDimN_ != 1, OP_LOGE(context_, "cos: Dimension 2 (N) must be 1, but got %ld.", cosSinDimN_),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(cosShape->GetOriginShape().GetDim(DIM_THREE) != dimD_,
                    OP_LOGE(context_, "cos: Must match gradQueryEmbed on dimension 3, but got %ld vs %ld.",
                            cosShape->GetOriginShape().GetDim(DIM_THREE), dimD_), return ge::GRAPH_FAILED);
    }
    cosSinShapeSize_ = cosShape->GetStorageShape().GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradCheckSin()
{
    auto sinShape = context_->GetInputShape(INDEX_SIN);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sinShape);
    auto sinDimSize = sinShape->GetOriginShape().GetDimNum();
    OP_CHECK_IF(sinDimSize != gradQueryEmbedDimSize_,
                OP_LOGE(context_, "sin: Rank must be %lu, but got %lu.", gradQueryEmbedDimSize_, sinDimSize),
                return ge::GRAPH_FAILED);

    if (sinDimSize == DIM_SIZE_THREE) {
        OP_CHECK_IF(sinShape->GetOriginShape().GetDim(DIM_ZERO) != dimS_,
                    OP_LOGE(context_, "sin: Must match gradQueryEmbed on dimension 0, but got %ld vs %ld.",
                            sinShape->GetOriginShape().GetDim(DIM_ZERO), dimS_), return ge::GRAPH_FAILED);
        OP_CHECK_IF(sinShape->GetOriginShape().GetDim(DIM_ONE) != 1,
                    OP_LOGE(context_, "sin: Dimension 1 (N) must be 1, but got %ld.",
                            sinShape->GetOriginShape().GetDim(DIM_ONE)), return ge::GRAPH_FAILED);
        OP_CHECK_IF(sinShape->GetOriginShape().GetDim(DIM_TWO) != dimD_,
                    OP_LOGE(context_, "sin: Must match gradQueryEmbed on dimension 2, but got %ld vs %ld.",
                            sinShape->GetOriginShape().GetDim(DIM_TWO), dimD_), return ge::GRAPH_FAILED);
    } else if (layout_ == static_cast<int64_t>(ApplyRotaryPosEmbGradLayout::BSND)) {
        OP_CHECK_IF(sinShape->GetOriginShape().GetDim(DIM_ZERO) != dQKDimB_,
                    OP_LOGE(context_, "sin: Must match gradQueryEmbed on dimension 0, but got %ld vs %ld.",
                            sinShape->GetOriginShape().GetDim(DIM_ZERO), dQKDimB_), return ge::GRAPH_FAILED);
        OP_CHECK_IF(sinShape->GetOriginShape().GetDim(DIM_ONE) != dimS_,
                    OP_LOGE(context_, "sin: Must match gradQueryEmbed on dimension 1, but got %ld vs %ld.",
                            sinShape->GetOriginShape().GetDim(DIM_ONE), dimS_), return ge::GRAPH_FAILED);
        OP_CHECK_IF(sinShape->GetOriginShape().GetDim(DIM_TWO) != 1,
                    OP_LOGE(context_, "sin: Dimension 2 (N) must be 1, but got %ld.",
                            sinShape->GetOriginShape().GetDim(DIM_TWO)), return ge::GRAPH_FAILED);
        OP_CHECK_IF(sinShape->GetOriginShape().GetDim(DIM_THREE) != dimD_,
                    OP_LOGE(context_, "sin: Must match gradQueryEmbed on dimension 3, but got %ld vs %ld.",
                            sinShape->GetOriginShape().GetDim(DIM_THREE), dimD_), return ge::GRAPH_FAILED);
    } else { // SBND
        OP_CHECK_IF(sinShape->GetOriginShape().GetDim(DIM_ZERO) != dimS_,
                    OP_LOGE(context_, "sin: Must match gradQueryEmbed on dimension 0, but got %ld vs %ld.",
                            sinShape->GetOriginShape().GetDim(DIM_ZERO), dimS_), return ge::GRAPH_FAILED);
        OP_CHECK_IF(sinShape->GetOriginShape().GetDim(DIM_ONE) != dQKDimB_,
                    OP_LOGE(context_, "sin: Must match gradQueryEmbed on dimension 1, but got %ld vs %ld.",
                            sinShape->GetOriginShape().GetDim(DIM_ONE), dQKDimB_), return ge::GRAPH_FAILED);
        OP_CHECK_IF(sinShape->GetOriginShape().GetDim(DIM_TWO) != 1,
                    OP_LOGE(context_, "sin: Dimension 2 (N) must be 1, but got %ld.",
                            sinShape->GetOriginShape().GetDim(DIM_TWO)), return ge::GRAPH_FAILED);
        OP_CHECK_IF(sinShape->GetOriginShape().GetDim(DIM_THREE) != dimD_,
                    OP_LOGE(context_, "sin: Must match gradQueryEmbed on dimension 3, but got %ld vs %ld.",
                            sinShape->GetOriginShape().GetDim(DIM_THREE), dimD_), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradCheckQuery()
{
    auto queryShape = context_->GetOptionalInputShape(INDEX_QUERY);
    if (queryShape != nullptr) {
        OP_CHECK_NULL_WITH_CONTEXT(context_, queryShape);
        auto queryDimSize = queryShape->GetOriginShape().GetDimNum();
        OP_CHECK_IF(queryDimSize != gradQueryEmbedDimSize_,
                    OP_LOGE(context_, "query: Rank must be %lu, but got %lu.", gradQueryEmbedDimSize_, queryDimSize),
                    return ge::GRAPH_FAILED);

        if (queryDimSize == DIM_SIZE_THREE) {
            const int64_t expVals[3] = {dimS_, dQDimN_, dimD_};
            if (CheckShapeDimMatch3D(queryShape, expVals, "query") != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        } else {
            auto gradQueryEmbedShape = context_->GetInputShape(INDEX_GRAD_QUERY_EMBED);
            if (CheckShapeDimMatch(queryShape, gradQueryEmbedShape, "query") != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        }
        hasOptionalInputs_ = 1;
    } else {
        hasOptionalInputs_ = 0;
    }
    return ge::GRAPH_SUCCESS;
}

// 4D 逐维比对: actualShape 每个维度与 expectShape 逐维比对
ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::CheckShapeDimMatch(const gert::StorageShape *actualShape,
                                                                           const gert::StorageShape *expectShape,
                                                                           const std::string &label)
{
    for (uint64_t d = 0; d < DIM_SIZE_FOUR; d++) {
        OP_CHECK_IF(actualShape->GetOriginShape().GetDim(d) != expectShape->GetOriginShape().GetDim(d),
                    OP_LOGE(context_, "%s: Must match on dimension %lu, but got %ld vs %ld.", label, d,
                            actualShape->GetOriginShape().GetDim(d), expectShape->GetOriginShape().GetDim(d)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// 3D 逐维比对: actualShape 每个维度与 expVals[3] 比对
ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::CheckShapeDimMatch3D(const gert::StorageShape *actualShape,
                                                                             const int64_t expVals[3],
                                                                             const std::string &label)
{
    for (uint64_t d = 0; d < DIM_SIZE_THREE; d++) {
        OP_CHECK_IF(actualShape->GetOriginShape().GetDim(d) != expVals[d],
                    OP_LOGE(context_, "%s: Must match on dimension %lu, but got %ld vs %ld.", label, d,
                            actualShape->GetOriginShape().GetDim(d), expVals[d]),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradCheckKey()
{
    auto keyShape = context_->GetOptionalInputShape(INDEX_KEY);
    if (hasOptionalInputs_ && keyShape != nullptr) {
        OP_CHECK_NULL_WITH_CONTEXT(context_, keyShape);
        auto keyDimSize = keyShape->GetOriginShape().GetDimNum();
        OP_CHECK_IF(keyDimSize != gradQueryEmbedDimSize_,
                    OP_LOGE(context_, "key: Rank must be %lu, but got %lu.", gradQueryEmbedDimSize_, keyDimSize),
                    return ge::GRAPH_FAILED);
        if (keyDimSize == DIM_SIZE_THREE) {
            const int64_t expVals[3] = {dimS_, dKDimN_, dimD_};
            if (CheckShapeDimMatch3D(keyShape, expVals, "key") != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        } else {
            auto gradKeyEmbedShape = context_->GetInputShape(INDEX_GRAD_KEY_EMBED);
            if (CheckShapeDimMatch(keyShape, gradKeyEmbedShape, "key") != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        }
    } else if (!hasOptionalInputs_ && keyShape == nullptr) { // skip
    } else {
        OP_LOGE(context_, "query & key: Must either both be provided or both be absent.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradCheckInputDatatype()
{
    auto gradQueryEmbedDesc = context_->GetInputDesc(INDEX_GRAD_QUERY_EMBED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradQueryEmbedDesc);
    gradQueryEmbedDtype_ = gradQueryEmbedDesc->GetDataType();
    OP_CHECK_IF(gradQueryEmbedDtype_ != ge::DataType::DT_FLOAT && gradQueryEmbedDtype_ != ge::DataType::DT_FLOAT16 &&
                    gradQueryEmbedDtype_ != ge::DataType::DT_BF16,
                OP_LOGE(context_, "gradQueryEmbedDtype: Unsupported data type. Expected %s, %s or %s, but got %s.",
                        ge::TypeUtils::DataTypeToSerialString(ge::DataType::DT_FLOAT).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(ge::DataType::DT_FLOAT16).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(ge::DataType::DT_BF16).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(gradQueryEmbedDtype_).c_str()),
                return ge::GRAPH_FAILED);

    inputDtypeSize_ = GetSizeByDataType(gradQueryEmbedDtype_);

    auto gradKeyEmbedDesc = context_->GetInputDesc(INDEX_GRAD_KEY_EMBED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradKeyEmbedDesc);
    auto gradKeyEmbedDtype = gradKeyEmbedDesc->GetDataType();
    OP_CHECK_IF(gradKeyEmbedDtype != gradQueryEmbedDtype_,
                OP_LOGE(context_, "gradKeyEmbedDtype: Mismatched data type. Expected %s, but got %s.",
                        ge::TypeUtils::DataTypeToSerialString(gradQueryEmbedDtype_).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(gradKeyEmbedDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto cosDesc = context_->GetInputDesc(INDEX_COS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cosDesc);
    auto cosDtype = cosDesc->GetDataType();
    OP_CHECK_IF(cosDtype != gradQueryEmbedDtype_,
                OP_LOGE(context_, "cosDtype: Mismatched data type. Expected %s, but got %s.",
                        ge::TypeUtils::DataTypeToSerialString(gradQueryEmbedDtype_).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(cosDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto sinDesc = context_->GetInputDesc(INDEX_SIN);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sinDesc);
    auto sinDtype = sinDesc->GetDataType();
    OP_CHECK_IF(sinDtype != gradQueryEmbedDtype_,
                OP_LOGE(context_, "sinDtype: Mismatched data type. Expected %s, but got %s.",
                        ge::TypeUtils::DataTypeToSerialString(gradQueryEmbedDtype_).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(sinDtype).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradCheckOptionalInputDatatype()
{
    auto queryDesc = context_->GetOptionalInputDesc(INDEX_QUERY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryDesc);
    auto queryDtype = queryDesc->GetDataType();
    OP_CHECK_IF(queryDtype != gradQueryEmbedDtype_,
                OP_LOGE(context_, "queryDtype: Mismatched data type. Expected %s, but got %s.",
                        ge::TypeUtils::DataTypeToSerialString(gradQueryEmbedDtype_).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(queryDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto keyDesc = context_->GetOptionalInputDesc(INDEX_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyDesc);
    auto keyDtype = keyDesc->GetDataType();
    OP_CHECK_IF(keyDtype != gradQueryEmbedDtype_,
                OP_LOGE(context_, "keyDtype: Mismatched data type. Expected %s, but got %s.",
                        ge::TypeUtils::DataTypeToSerialString(gradQueryEmbedDtype_).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(keyDtype).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradCheckOutput()
{
    OP_CHECK_IF(ApplyRopeHalfGradCheckGradQuery() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Check gradQuery failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ApplyRopeHalfGradCheckGradKey() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Check gradKey failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ApplyRopeHalfGradCheckOutputDatatype() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "Check output datatype failed"), return ge::GRAPH_FAILED);

    auto gradCosShape = context_->GetOutputShape(INDEX_GRAD_COS);
    auto gradCosDesc = context_->GetOutputDesc(INDEX_GRAD_COS);
    auto gradSinShape = context_->GetOutputShape(INDEX_GRAD_SIN);
    auto gradSinDesc = context_->GetOutputDesc(INDEX_GRAD_SIN);
    if (hasOptionalInputs_) {
        if (gradCosShape != nullptr && gradCosDesc != nullptr && gradSinShape != nullptr && gradSinDesc != nullptr) {
            OP_CHECK_IF(ApplyRopeHalfGradCheckGradCos() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Check gradCos failed"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(ApplyRopeHalfGradCheckGradSin() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Check gradSin failed"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(ApplyRopeHalfGradCheckOptionalOutputDatatype() != ge::GRAPH_SUCCESS,
                        OP_LOGE(context_, "Check optional output datatype failed"), return ge::GRAPH_FAILED);
        } else {
            OP_LOGE(context_, "gradQuery & gradKey: Must both be provided.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradCheckGradQuery()
{
    auto gradQueryShape = context_->GetOutputShape(INDEX_GRAD_QUERY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradQueryShape);

    auto gradQueryDimSize = gradQueryShape->GetOriginShape().GetDimNum();
    OP_CHECK_IF(
        gradQueryDimSize != gradQueryEmbedDimSize_,
        OP_LOGE(context_, "gradQuery: Rank must be %lu, but got %lu.", gradQueryEmbedDimSize_, gradQueryDimSize),
        return ge::GRAPH_FAILED);

    if (gradQueryDimSize == DIM_SIZE_THREE) {
        const int64_t expVals[3] = {dimS_, dQDimN_, dimD_};
        if (CheckShapeDimMatch3D(gradQueryShape, expVals, "gradQuery") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        auto gradQueryEmbedShape = context_->GetInputShape(INDEX_GRAD_QUERY_EMBED);
        if (CheckShapeDimMatch(gradQueryShape, gradQueryEmbedShape, "gradQuery") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradCheckGradKey()
{
    auto gradKeyShape = context_->GetOutputShape(INDEX_GRAD_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradKeyShape);

    auto gradKeyDimSize = gradKeyShape->GetOriginShape().GetDimNum();
    OP_CHECK_IF(gradKeyDimSize != gradQueryEmbedDimSize_,
                OP_LOGE(context_, "gradKey: Rank must be %lu, but got %lu.", gradQueryEmbedDimSize_, gradKeyDimSize),
                return ge::GRAPH_FAILED);

    if (gradKeyDimSize == DIM_SIZE_THREE) {
        const int64_t expVals[3] = {dimS_, dKDimN_, dimD_};
        if (CheckShapeDimMatch3D(gradKeyShape, expVals, "gradKey") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        auto gradKeyEmbedShape = context_->GetInputShape(INDEX_GRAD_KEY_EMBED);
        if (CheckShapeDimMatch(gradKeyShape, gradKeyEmbedShape, "gradKey") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradCheckGradCos()
{
    auto gradCosShape = context_->GetOutputShape(INDEX_GRAD_COS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradCosShape);

    auto gradCosDimSize = gradCosShape->GetOriginShape().GetDimNum();
    OP_CHECK_IF(gradCosDimSize != gradQueryEmbedDimSize_,
                OP_LOGE(context_, "gradCos: Rank must be %lu, but got %lu.", gradQueryEmbedDimSize_, gradCosDimSize),
                return ge::GRAPH_FAILED);

    if (gradCosDimSize == DIM_SIZE_THREE) {
        const int64_t expVals[3] = {dimS_, cosSinDimN_, dimD_};
        if (CheckShapeDimMatch3D(gradCosShape, expVals, "gradCos") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        auto cosShape = context_->GetInputShape(INDEX_COS);
        if (CheckShapeDimMatch(gradCosShape, cosShape, "gradCos") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradCheckGradSin()
{
    auto gradSinShape = context_->GetOutputShape(INDEX_GRAD_SIN);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradSinShape);

    auto gradSinDimSize = gradSinShape->GetOriginShape().GetDimNum();
    OP_CHECK_IF(gradSinDimSize != gradQueryEmbedDimSize_,
                OP_LOGE(context_, "gradSin: Rank must be %lu, but got %lu.", gradQueryEmbedDimSize_, gradSinDimSize),
                return ge::GRAPH_FAILED);

    if (gradSinDimSize == DIM_SIZE_THREE) {
        const int64_t expVals[3] = {dimS_, cosSinDimN_, dimD_};
        if (CheckShapeDimMatch3D(gradSinShape, expVals, "gradSin") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        auto sinShape = context_->GetInputShape(INDEX_SIN);
        if (CheckShapeDimMatch(gradSinShape, sinShape, "gradSin") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradCheckOutputDatatype()
{
    auto gradQueryDesc = context_->GetOutputDesc(INDEX_GRAD_QUERY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradQueryDesc);
    auto gradQueryDtype = gradQueryDesc->GetDataType();
    OP_CHECK_IF(gradQueryDtype != gradQueryEmbedDtype_,
                OP_LOGE(context_, "gradQueryDtype: Mismatched data type. Expected %s, but got %s.",
                        ge::TypeUtils::DataTypeToSerialString(gradQueryEmbedDtype_).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(gradQueryDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto gradKeyDesc = context_->GetOutputDesc(INDEX_GRAD_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradKeyDesc);
    auto gradKeyDtype = gradKeyDesc->GetDataType();
    OP_CHECK_IF(gradKeyDtype != gradQueryEmbedDtype_,
                OP_LOGE(context_, "gradQueryDtype: Mismatched data type. Expected %s, but got %s.",
                        ge::TypeUtils::DataTypeToSerialString(gradQueryEmbedDtype_).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(gradKeyDtype).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradCheckOptionalOutputDatatype()
{
    auto gradCosDesc = context_->GetOutputDesc(INDEX_GRAD_COS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradCosDesc);
    auto gradCosDtype = gradCosDesc->GetDataType();
    OP_CHECK_IF(gradCosDtype != gradQueryEmbedDtype_,
                OP_LOGE(context_, "gradCos: Mismatched data type. Expected %s, but got %s.",
                        ge::TypeUtils::DataTypeToSerialString(gradQueryEmbedDtype_).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(gradCosDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto gradSinDesc = context_->GetOutputDesc(INDEX_GRAD_SIN);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradSinDesc);
    auto gradSinDtype = gradSinDesc->GetDataType();
    OP_CHECK_IF(gradSinDtype != gradQueryEmbedDtype_,
                OP_LOGE(context_, "gradCos: Mismatched data type. Expected %s, but got %s.",
                        ge::TypeUtils::DataTypeToSerialString(gradQueryEmbedDtype_).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(gradSinDtype).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradSetTiling(const int64_t availableUbSize,
                                                                                   const uint64_t maxCoreNum)
{
    int64_t reserveAlignNum = dimD_;
    alignUbSize_ = availableUbSize;
    calcUbSize_ = availableUbSize;
    int64_t ubPerReserveNum = 1;

    OP_CHECK_IF(halfDimDAlignNum_ == 0, OP_LOGE("ApplyRopeHalfGrad", "halfDimDAlignNum_ is 0"),
                return ge::GRAPH_FAILED);
    if (dimD_ % MASK_NUM_FP32 != 0) {
        reserveAlignNum = GetLcmForRotary(halfDimDAlignNum_ * CHUNK_SIZE, MASK_NUM_FP32);
        ubPerReserveNum = reserveAlignNum / (halfDimDAlignNum_ * CHUNK_SIZE);
    }
    OP_CHECK_IF(reserveAlignNum == 0, OP_LOGE("ApplyRopeHalfGrad", "reserveAlignNum is 0"), return ge::GRAPH_FAILED);
    alignUbSize_ = availableUbSize / reserveAlignNum * reserveAlignNum;
    calcUbSize_ = availableUbSize / reserveAlignNum * ubPerReserveNum * dimD_;
    OP_CHECK_IF(
        halfDimDAlignNum_ * 2 > alignUbSize_ || alignUbSize_ == 0 || calcUbSize_ == 0,
        OP_LOGE("ApplyRopeHalfGrad", "reserveAlignNum = %lu too large, aicore do not support.", halfDimDAlignNum_ * 2),
        return ge::GRAPH_FAILED);

    int64_t taskNum = cosSinShapeSize_; // BSND/SBND/TND：切分cos的BS合轴
    elementsPerCore_ = GetCeilInt(taskNum, maxCoreNum);
    elementsPerCore_ = GetCeilInt(elementsPerCore_, dimD_) * dimD_;
    OP_CHECK_IF(elementsPerCore_ == 0, OP_LOGE("ApplyRopeHalfGrad", "elementsPerCore_ is 0"), return ge::GRAPH_FAILED);
    coreUsed_ = GetCeilInt(taskNum, elementsPerCore_);
    lastCoreElements_ = taskNum % elementsPerCore_ != 0 ? taskNum % elementsPerCore_ : elementsPerCore_;
    fullTileLoopsPerCore_ = elementsPerCore_ / calcUbSize_;
    tailElementsPerCore_ = elementsPerCore_ % calcUbSize_;
    lastCoreFullTileLoops_ = lastCoreElements_ / calcUbSize_;
    lastCoreTailElements_ = lastCoreElements_ % calcUbSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradInitSplitInfo() // 计算切分相关参数
{
    int64_t availableUbSize = static_cast<int64_t>(aicoreParams_.ubSize) - RESERVE_UB_SIZE;
    if (hasOptionalInputs_) {
        if (inputDtypeSize_ == DIM_SIZE_FOUR) {
            availableUbSize = availableUbSize / (DTYPE_SIZE_FP32 * IN_OUT_TENSOR_NUM + DTYPE_SIZE_FP32 * TENSOR_NUM);
        } else {
            availableUbSize =
                availableUbSize / (DTYPE_SIZE_FP16 * IN_OUT_TENSOR_NUM + DTYPE_SIZE_FP32 * TENSOR_CONV_NUM);
        }
    } else {
        if (inputDtypeSize_ == DIM_SIZE_FOUR) {
            availableUbSize =
                availableUbSize / (DTYPE_SIZE_FP32 * IN_OUT_TENSOR_SINGLE_NUM + DTYPE_SIZE_FP32 * TENSOR_SINGLE_NUM);
        } else {
            availableUbSize = availableUbSize /
                              (DTYPE_SIZE_FP16 * IN_OUT_TENSOR_SINGLE_NUM + DTYPE_SIZE_FP32 * TENSOR_CONV_SINGLE_NUM);
        }
    }
    OP_CHECK_IF(ApplyRopeHalfGradSetTiling(availableUbSize, aicoreParams_.blockDim) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "ApplyRopeHalfGradInitSplitInfo failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradInitParamInfo() // 计算循环和搬运相关参数
{
    innerPaddedBlockLen_ = dimD_ / CHUNK_SIZE * inputDtypeSize_;

    if (layout_ == static_cast<int64_t>(ApplyRotaryPosEmbGradLayout::BSND) ||
        layout_ == static_cast<int64_t>(ApplyRotaryPosEmbGradLayout::TND)) {
        cosSinOuterLoops_ = dQKDimB_ / cosSinDimB_;
        dQHeadInnerLoops_ = dQDimN_ / cosSinDimN_;
        dKHeadInnerLoops_ = dKDimN_ / cosSinDimN_;
        dQOuterLoopGmStride_ = cosSinDimB_ * dimS_ * dQDimN_ * dimD_;
        dKOuterLoopGmStride_ = cosSinDimB_ * dimS_ * dKDimN_ * dimD_;
        dQInnerPaddedSrcStride_ = (dQDimN_ / cosSinDimN_ * dimD_ - dimD_ / CHUNK_SIZE) * inputDtypeSize_;
        dKInnerPaddedSrcStride_ = (dKDimN_ / cosSinDimN_ * dimD_ - dimD_ / CHUNK_SIZE) * inputDtypeSize_;
    } else { // SBND
        cosSinOuterLoops_ = 1;
        dQHeadInnerLoops_ = dQKDimB_ / cosSinDimB_ * dQDimN_ / cosSinDimN_;
        dKHeadInnerLoops_ = dQKDimB_ / cosSinDimB_ * dKDimN_ / cosSinDimN_;
        dQOuterLoopGmStride_ = 1;
        dKOuterLoopGmStride_ = 1;
        dQInnerPaddedSrcStride_ =
            (dQKDimB_ / cosSinDimB_ * dQDimN_ / cosSinDimN_ * dimD_ - dimD_ / CHUNK_SIZE) * inputDtypeSize_;
        dKInnerPaddedSrcStride_ =
            (dQKDimB_ / cosSinDimB_ * dKDimN_ / cosSinDimN_ * dimD_ - dimD_ / CHUNK_SIZE) * inputDtypeSize_;
    }

    return ge::GRAPH_SUCCESS;
}

uint32_t ApplyRotaryPosEmbGradHalfTiling::ApplyRopeHalfGradSelectTilingKey() // 计算tilingkey
{
    uint32_t shapeAlignKey = 0;
    uint32_t needBackwardKey = 0;
    if (dimD_ % MASK_NUM_FP32 == 0) {
        shapeAlignKey = SHAPE_KEY_ALIGN;
    } else {
        shapeAlignKey = SHAPE_KEY_PAD;
    }
    if (hasOptionalInputs_) {
        needBackwardKey = NEED_BACKWARD_KEY_TRUE;
    } else {
        needBackwardKey = NEED_BACKWARD_KEY_FALSE;
    }
    uint32_t tilingKey = shapeAlignKey + needBackwardKey;
    return tilingKey;
}

ge::graphStatus ApplyRotaryPosEmbGradHalfTiling::DoOpTiling()
{
    OP_LOGD("ApplyRotaryPosEmbGradHalf tiling start");

    OP_CHECK_IF( // 输入的维度校验并获取shape相关参数，可以增加子函数
        ApplyRopeHalfGradCheckInput() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "ApplyRopeHalfGradCheckInput not support."), return ge::GRAPH_FAILED);
    OP_LOGD("ApplyRopeHalfGradTiling ApplyRopeHalfGradCheckInput");

    uint64_t dataEachBlock = BLOCK_SIZE / inputDtypeSize_;
    halfDimDAlignNum_ = GetCeilInt(dimD_ / CHUNK_SIZE, dataEachBlock) * dataEachBlock;

    OP_CHECK_IF(ApplyRopeHalfGradCheckOutput() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "ApplyRopeHalfGradCheckOutput not support."), return ge::GRAPH_FAILED);
    OP_LOGD("ApplyRopeHalfGradTiling ApplyRopeHalfGradCheckOutput");

    OP_CHECK_IF(ApplyRopeHalfGradInitSplitInfo() != ge::GRAPH_SUCCESS, // 计算切分相关参数
                OP_LOGE(context_->GetNodeName(), "ApplyRopeHalfGradInitSplitInfo not support."),
                return ge::GRAPH_FAILED);
    OP_LOGD("ApplyRopeHalfGradTiling ApplyRopeHalfGradInitSplitInfo");

    OP_CHECK_IF( // 计算循环和搬运相关参数
        ApplyRopeHalfGradInitParamInfo() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "ApplyRopeHalfGradInitParamInfo not support."), return ge::GRAPH_FAILED);
    OP_LOGD("ApplyRopeHalfGradTiling ApplyRopeHalfGradInitParamInfo");

    ApplyRopeHalfGradAssignParam(); // 赋值tiling参数
    ApplyRopeHalfGradPrintParam();  // 打印结构体参数

    context_->SetBlockDim(tiling.applyRopeHalfGradParams.get_coreUsed());

    uint32_t tilingKey = ApplyRopeHalfGradSelectTilingKey(); // 计算tilingkey
    context_->SetTilingKey(tilingKey);

    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t sysWorkspaceSize = static_cast<size_t>(SYS_WORKSPACE_SIZE);
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;

    OP_LOGD("ApplyRopeHalfGradTiling tiling end");
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling