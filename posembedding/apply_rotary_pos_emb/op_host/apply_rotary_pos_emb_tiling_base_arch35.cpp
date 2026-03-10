/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_rotary_pos_emb_regbase_tiling_base.cpp
 * \brief
 */
#include "apply_rotary_pos_emb_tiling.h"
#include "log/log.h"

namespace {
constexpr int64_t Q_INDEX = 0;
constexpr int64_t K_INDEX = 1;
constexpr int64_t COS_INDEX = 2;
constexpr int64_t SIN_INDEX = 3;
constexpr int64_t QOUT_INDEX = 0;
constexpr int64_t KOUT_INDEX = 1;
constexpr int64_t DIM_NUM = 4;
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t HALF_INTERLEAVE_MODE_COEF = 2;
constexpr int64_t QUARTER_MODE_COEF = 4;
constexpr int64_t D_LIMIT = 1024;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t V_REG_SIZE = 256;
const std::vector<ge::DataType> SUPPORT_DTYPE = {ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16};
} // namespace

namespace optiling {
ge::graphStatus ApplyRotaryPosEmbRegbaseTilingBaseClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.blockDim = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
        socVersion_ = ascendcPlatform.GetSocVersion();
    } else {
        auto compileInfoPtr = reinterpret_cast<const ApplyRotaryPosEmbCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr);
        aicoreParams_.blockDim = compileInfoPtr->blockDim;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        socVersion_ = compileInfoPtr->socVersion;
    }
    blockSize_ = BLOCK_SIZE;
    vLength_ = V_REG_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbRegbaseTilingBaseClass::CheckNullptr()
{
    for (int64_t i = 0; i <= SIN_INDEX; i++) {
        auto desc = context_->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, desc);
        auto shape = context_->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, shape);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbRegbaseTilingBaseClass::CheckShapeAllPositive(int64_t idx)
{
    auto shape = context_->GetInputShape(idx)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(
            shape.GetDim(i) <= 0,
            OP_LOGE(context_, "input %ld has non positive shape, dim %lu actual %ld .", idx, i, shape.GetDim(i)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbRegbaseTilingBaseClass::CheckShapeAllPositive()
{
    ge::graphStatus qRes = CheckShapeAllPositive(Q_INDEX);
    ge::graphStatus kRes = CheckShapeAllPositive(K_INDEX);
    // q, k有一个为空Tensor时，提示用 rotaryPositionEmbedding 算子
    OP_CHECK_IF(qRes != kRes,
                OP_LOGE(context_, "q or k has non positive shape, please use rotaryPositionEmbedding operator"),
                return ge::GRAPH_FAILED);
    // q, k 都为空Tensor，处理空tensor
    OP_CHECK_IF(qRes != ge::GRAPH_SUCCESS, OP_LOGE(context_, "query and key has non positive shape."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeAllPositive(COS_INDEX) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "cos has non positive shape."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeAllPositive(SIN_INDEX) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "sin has non positive shape."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbRegbaseTilingBaseClass::CheckRotaryModeShapeRelation(int64_t d)
{
    OP_CHECK_IF(d > D_LIMIT, OP_LOGE(context_, "D must be small than %ld, actual %ld.", D_LIMIT, d),
                return ge::GRAPH_FAILED);
    if (rotaryModeStr_ == "half" || rotaryModeStr_ == "interleave") {
        OP_CHECK_IF(d % HALF_INTERLEAVE_MODE_COEF != 0,
                    OP_LOGE(context_, "D must be multiples of 2 in half and interleave mode, actual %ld.", d),
                    return ge::GRAPH_FAILED);
    } else if (rotaryModeStr_ == "quarter") {
        OP_CHECK_IF(d % QUARTER_MODE_COEF != 0,
                    OP_LOGE(context_, "D must be multiples of 4 in quarter mode, actual %ld.", d),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbRegbaseTilingBaseClass::CheckShapeRelation(const gert::Shape &qShape,
                                                                            const gert::Shape &kShape,
                                                                            const gert::Shape &cosShape,
                                                                            const gert::Shape &sinShape)
{
    int64_t bIdx = DIM_0;
    int64_t sIdx = DIM_1;
    int64_t nIdx = DIM_2;
    int64_t dIdx = DIM_3;
    if (layout_ == ApplyRotaryPosEmbLayout::BNSD) {
        sIdx = DIM_2;
        nIdx = DIM_1;
    } else if (layout_ == ApplyRotaryPosEmbLayout::SBND) {
        sIdx = DIM_0;
        bIdx = DIM_1;
        nIdx = DIM_2;
    }
    auto &qOutShape = context_->GetOutputShape(QOUT_INDEX)->GetStorageShape();
    auto &kOutShape = context_->GetOutputShape(KOUT_INDEX)->GetStorageShape();
    OP_CHECK_IF(cosShape != sinShape, OP_LOGE(context_, "shape of cos and sin should be same."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qShape != qOutShape, OP_LOGE(context_, "shape of query in and out should be same."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kShape != kOutShape, OP_LOGE(context_, "shape of key in and out should be same."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cosShape.GetDim(nIdx) != 1,
                OP_LOGE(context_, "N of cos, sin should be 1, actual %ld.", cosShape.GetDim(nIdx)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(cosShape.GetDim(sIdx) == qShape.GetDim(sIdx) && kShape.GetDim(sIdx) == qShape.GetDim(sIdx)),
                OP_LOGE(context_, "S of query, key, cos, sin should be same, actual %ld %ld %ld %ld.",
                        qShape.GetDim(sIdx), kShape.GetDim(sIdx), cosShape.GetDim(sIdx), cosShape.GetDim(sIdx)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(cosShape.GetDim(dIdx) == qShape.GetDim(dIdx) && kShape.GetDim(dIdx) == qShape.GetDim(dIdx)),
                OP_LOGE(context_, "D of query, key, cos, sin should be same, actual %ld %ld %ld %ld.",
                        qShape.GetDim(dIdx), kShape.GetDim(dIdx), cosShape.GetDim(dIdx), cosShape.GetDim(dIdx)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        kShape.GetDim(bIdx) != qShape.GetDim(bIdx),
        OP_LOGE(context_, "B of query, key should be same, actual %ld %ld.", qShape.GetDim(bIdx), kShape.GetDim(bIdx)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(cosShape.GetDim(bIdx) == qShape.GetDim(bIdx) || cosShape.GetDim(bIdx) == 1),
                OP_LOGE(context_, "B of cos, sin should be able to broadcast to query, key, actual %ld %ld %ld %ld.",
                        cosShape.GetDim(bIdx), cosShape.GetDim(bIdx), qShape.GetDim(bIdx), kShape.GetDim(bIdx)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckRotaryModeShapeRelation(qShape.GetDim(dIdx)) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "D is invalid for rotary mode."), return ge::GRAPH_FAILED);
    return CheckShapeAllPositive();
}

ge::graphStatus ApplyRotaryPosEmbRegbaseTilingBaseClass::CheckShape()
{
    auto &qShape = context_->GetInputShape(Q_INDEX)->GetStorageShape();
    auto &kShape = context_->GetInputShape(K_INDEX)->GetStorageShape();
    auto &cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto &sinShape = context_->GetInputShape(SIN_INDEX)->GetStorageShape();
    auto &qoutShape = context_->GetOutputShape(QOUT_INDEX)->GetStorageShape();
    auto &koutShape = context_->GetOutputShape(KOUT_INDEX)->GetStorageShape();
    OP_CHECK_IF(qShape.GetDimNum() != DIM_NUM,
                OP_LOGE(context_, "dim of query expect 4, actual %zu.", qShape.GetDimNum()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(kShape.GetDimNum() != DIM_NUM,
                OP_LOGE(context_, "dim of key expect 4, actual %zu.", kShape.GetDimNum()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(cosShape.GetDimNum() != DIM_NUM,
                OP_LOGE(context_, "dim of cos expect 4, actual %zu.", cosShape.GetDimNum()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(sinShape.GetDimNum() != DIM_NUM,
                OP_LOGE(context_, "dim of sin expect 4, actual %zu.", sinShape.GetDimNum()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qoutShape.GetDimNum() != DIM_NUM,
                OP_LOGE(context_, "dim of query out expect 4, actual %zu.", qoutShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(koutShape.GetDimNum() != DIM_NUM,
                OP_LOGE(context_, "dim of key out expect 4, actual %zu.", koutShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    return CheckShapeRelation(qShape, kShape, cosShape, sinShape);
}

ge::graphStatus ApplyRotaryPosEmbRegbaseTilingBaseClass::CheckDtypeAndAttr()
{
    OP_CHECK_IF(
        (rotaryModeStr_ != "half" && rotaryModeStr_ != "interleave" && rotaryModeStr_ != "quarter"),
        OP_LOGE(context_, "rotary mode only support half, interleave, quarter, actual %s.", rotaryModeStr_.c_str()),
        return ge::GRAPH_FAILED);
    auto targetType = context_->GetInputDesc(Q_INDEX)->GetDataType();
    OP_CHECK_IF(std::find(SUPPORT_DTYPE.begin(), SUPPORT_DTYPE.end(), targetType) == SUPPORT_DTYPE.end(),
                OP_LOGE(context_->GetNodeName(), "Only support F32, BF16, F16 datetype, actual %d.", targetType),
                return ge::GRAPH_FAILED);
    for (int64_t i = Q_INDEX; i <= SIN_INDEX; i++) {
        auto type = context_->GetInputDesc(i)->GetDataType();
        OP_CHECK_IF(type != targetType,
                    OP_LOGE(context_, "input %ld datatype expect %d, actual %d.", i, targetType, type),
                    return ge::GRAPH_FAILED);
    }
    for (int64_t i = QOUT_INDEX; i <= KOUT_INDEX; i++) {
        auto type = context_->GetInputDesc(i)->GetDataType();
        OP_CHECK_IF(type != targetType,
                    OP_LOGE(context_, "output %ld datatype expect %d, actual %d.", i, targetType, type),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbRegbaseTilingBaseClass::CheckParam()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (ascendcPlatform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND910_95) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(CheckNullptr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check nullptr fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDtypeAndAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check dtype and attr fail."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check shape fail."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void ApplyRotaryPosEmbRegbaseTilingBaseClass::ConvertRotaryMode()
{
    if (rotaryModeStr_ == "half") {
        rotaryMode_ = ApplyRotaryPosEmbRotaryMode::HALF;
        dSplitCoef_ = HALF_INTERLEAVE_MODE_COEF;
    } else if (rotaryModeStr_ == "interleave") {
        rotaryMode_ = ApplyRotaryPosEmbRotaryMode::INTERLEAVE;
        dSplitCoef_ = 1;
    } else if (rotaryModeStr_ == "quarter") {
        rotaryMode_ = ApplyRotaryPosEmbRotaryMode::QUARTER;
        dSplitCoef_ = QUARTER_MODE_COEF;
    }
}

ge::graphStatus ApplyRotaryPosEmbRegbaseTilingBaseClass::GetShapeAttrsInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (ascendcPlatform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND910_95) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t *layout = attrs->GetAttrPointer<int64_t>(0);
    int64_t layoutValue = (layout == nullptr) ? static_cast<int64_t>(ApplyRotaryPosEmbLayout::BSND) : (*layout);
    OP_CHECK_IF((static_cast<int64_t>(layoutValue) < static_cast<int64_t>(ApplyRotaryPosEmbLayout::BSND) ||
                 static_cast<int64_t>(layoutValue) > static_cast<int64_t>(ApplyRotaryPosEmbLayout::BNSD)),
                OP_LOGE(context_, "layout should in [1, 3], actual %ld.", static_cast<int64_t>(layoutValue)),
                return ge::GRAPH_FAILED);
    layout_ = static_cast<ApplyRotaryPosEmbLayout>(layoutValue);
    const char *rotaryMode = attrs->GetAttrPointer<char>(1);
    rotaryModeStr_ = (rotaryMode == nullptr) ? "half" : rotaryMode;

    OP_CHECK_IF(CheckParam() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check param fail."), return ge::GRAPH_FAILED);

    dtype_ = context_->GetInputDesc(Q_INDEX)->GetDataType();
    ConvertRotaryMode();

    auto qShape = context_->GetInputShape(Q_INDEX)->GetStorageShape();
    auto kShape = context_->GetInputShape(K_INDEX)->GetStorageShape();
    auto cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    if (layout_ == ApplyRotaryPosEmbLayout::BSND) {
        b_ = qShape.GetDim(DIM_0);
        cosb_ = cosShape.GetDim(DIM_0);
        s_ = qShape.GetDim(DIM_1);
        qn_ = qShape.GetDim(DIM_2);
        kn_ = kShape.GetDim(DIM_2);
        d_ = qShape.GetDim(DIM_3);
    } else if (layout_ == ApplyRotaryPosEmbLayout::BNSD) {
        b_ = qShape.GetDim(DIM_0);
        cosb_ = cosShape.GetDim(DIM_0);
        qn_ = qShape.GetDim(DIM_1);
        kn_ = kShape.GetDim(DIM_1);
        s_ = qShape.GetDim(DIM_2);
        d_ = qShape.GetDim(DIM_3);
    } else if (layout_ == ApplyRotaryPosEmbLayout::SBND) {
        s_ = qShape.GetDim(DIM_0);
        b_ = qShape.GetDim(DIM_1);
        cosb_ = cosShape.GetDim(DIM_1);
        qn_ = qShape.GetDim(DIM_2);
        kn_ = kShape.GetDim(DIM_2);
        d_ = qShape.GetDim(DIM_3);
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling