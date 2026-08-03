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
 * \file apply_rotary_pos_emb_grad_tiling.cpp
 * \brief apply_rotary_pos_emb_grad host 侧 tiling 母板类实现
 * 双路反向：形状检查、layout 判断、dtype 检查
 */
#include <cmath>
#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "posembedding/apply_rotary_pos_emb_grad/op_kernel/arch35/apply_rotary_pos_emb_grad_dag.h"
#include "posembedding/apply_rotary_pos_emb_grad/op_kernel/arch35/apply_rotary_pos_emb_grad_tiling_key.h"
#include "apply_rotary_pos_emb_grad_tiling.h"

namespace {
constexpr int64_t DYQ_INDEX = 0;
constexpr int64_t DYK_INDEX = 1;
constexpr int64_t COS_INDEX = 2;
constexpr int64_t SIN_INDEX = 3;
constexpr int64_t XQ_INDEX = 4;
constexpr int64_t XK_INDEX = 5;
constexpr int64_t DXQ_INDEX = 0;
constexpr int64_t DXK_INDEX = 1;
constexpr int64_t DCOS_INDEX = 2;
constexpr int64_t DSIN_INDEX = 3;
constexpr int64_t DIM_NUM = 4;
constexpr int64_t DIM_NUM_TND = 3;
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t HALF_MODE_COEF = 2;
constexpr int64_t D_LIMIT = 1024;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t WORKSPACE_TENSOR_NUM = 2; // xQ_rot + xK_rot

const std::vector<ge::DataType> SUPPORT_DTYPE = {ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16};
static const std::vector<std::string> inputNames = {"dy_q", "dy_k", "cos", "sin", "x_q", "x_k"};
static const std::vector<std::string> outputNames = {"dx_q", "dx_k", "dcos", "dsin"};
} // namespace

namespace optiling {
using namespace Ops::Base;

ge::graphStatus ApplyRopeGradRegbaseTilingClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
        socVersion_ = ascendcPlatform.GetSocVersion();
    }
    blockSize_ = Ops::Base::GetUbBlockSize(context_);
    vLength_ = Ops::Base::GetVRegSize(context_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckNullptr()
{
    int64_t checkInputIdx = SIN_INDEX; // cos, sin always required
    if (context_->GetInputShape(XQ_INDEX) != nullptr && context_->GetInputShape(XK_INDEX) != nullptr) {
        checkInputIdx = XK_INDEX;
    }

    for (int64_t i = DYQ_INDEX; i <= checkInputIdx; i++) {
        auto desc = context_->GetInputDesc(i);
        OP_CHECK_IF(desc == nullptr, OP_LOGE(context_, "input %ld desc is nullptr.", i), return ge::GRAPH_FAILED);
        auto shape = context_->GetInputShape(i);
        OP_CHECK_IF(shape == nullptr, OP_LOGE(context_, "input %ld shape is nullptr.", i), return ge::GRAPH_FAILED);
    }

    for (int64_t i = DXQ_INDEX; i <= DSIN_INDEX; i++) {
        auto desc = context_->GetOutputDesc(i);
        OP_CHECK_IF(desc == nullptr, OP_LOGE(context_, "output %ld desc is nullptr.", i), return ge::GRAPH_FAILED);
        auto shape = context_->GetOutputShape(i);
        OP_CHECK_IF(shape == nullptr, OP_LOGE(context_, "output %ld shape is nullptr.", i), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckInPutShapeAllPositive(const int64_t idx) const
{
    auto shape = context_->GetInputShape(idx)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) <= 0) {
            std::string shapeMsg = ToString(shape);
            std::string reasonMsg = "The shape of input " + inputNames[idx] +
                " can not be an empty tensor or an invalid tensor with a negative dimension";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), inputNames[idx].c_str(),
                shapeMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckOutPutShapeAllPositive(const int64_t idx) const
{
    auto shape = context_->GetOutputShape(idx)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) <= 0) {
            std::string shapeMsg = ToString(shape);
            std::string reasonMsg = "The shape of output " + outputNames[idx] +
                " can not be an empty tensor or an invalid tensor with a negative dimension";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), outputNames[idx].c_str(),
                shapeMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckShapeAllPositive() const
{
    OP_CHECK_IF(CheckInPutShapeAllPositive(DYQ_INDEX) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "dy_q has non positive shape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInPutShapeAllPositive(DYK_INDEX) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "dy_k has non positive shape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInPutShapeAllPositive(COS_INDEX) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "cos has non positive shape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInPutShapeAllPositive(SIN_INDEX) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "sin has non positive shape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOutPutShapeAllPositive(DXQ_INDEX) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "dx_q has non positive shape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOutPutShapeAllPositive(DXK_INDEX) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "dx_k has non positive shape."), return ge::GRAPH_FAILED);
    if (context_->GetInputShape(XQ_INDEX) != nullptr && context_->GetInputShape(XK_INDEX) != nullptr) {
        OP_CHECK_IF(CheckInPutShapeAllPositive(XQ_INDEX) != ge::GRAPH_SUCCESS,
            OP_LOGE(context_, "x_q has non positive shape."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckInPutShapeAllPositive(XK_INDEX) != ge::GRAPH_SUCCESS,
            OP_LOGE(context_, "x_k has non positive shape."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckOutPutShapeAllPositive(DCOS_INDEX) != ge::GRAPH_SUCCESS,
            OP_LOGE(context_, "dcos has non positive shape."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckOutPutShapeAllPositive(DSIN_INDEX) != ge::GRAPH_SUCCESS,
            OP_LOGE(context_, "dsin has non positive shape."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckShapeDim() const
{
    auto& dyQShape = context_->GetInputShape(DYQ_INDEX)->GetStorageShape();
    auto& dyKShape = context_->GetInputShape(DYK_INDEX)->GetStorageShape();
    auto& cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto& sinShape = context_->GetInputShape(SIN_INDEX)->GetStorageShape();
    auto& dxQShape = context_->GetOutputShape(DXQ_INDEX)->GetStorageShape();
    auto& dxKShape = context_->GetOutputShape(DXK_INDEX)->GetStorageShape();

    for (auto& sh : {&dyQShape, &dyKShape, &cosShape, &sinShape, &dxQShape, &dxKShape}) {
        if (sh->GetDimNum() != DIM_NUM && sh->GetDimNum() != DIM_NUM_TND) {
            std::string dimNumStr = std::to_string(sh->GetDimNum());
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "dy_q/dy_k/cos/sin/dx_q/dx_k", dimNumStr.c_str(), "3D or 4D");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::JudgeLayoutByShape(
    const gert::Shape& dyQShape, const gert::Shape& cosShape)
{
    // TND (3D) 不是独立的 layout，而是维度映射方式
    // TND 映射后: B=1, S=T, N=N, 然后走正常的 5 种 layout 判断
    isTndLayout_ = (dyQShape.GetDimNum() == DIM_NUM_TND);
    uint64_t xShape0 = isTndLayout_ ? 1 : dyQShape.GetDim(DIM_0);
    uint64_t xShape1 = isTndLayout_ ? dyQShape.GetDim(DIM_0) : dyQShape.GetDim(DIM_1);
    uint64_t xShape2 = isTndLayout_ ? dyQShape.GetDim(DIM_1) : dyQShape.GetDim(DIM_2);
    uint64_t cosShape0 = isTndLayout_ ? 1 : cosShape.GetDim(DIM_0);
    uint64_t cosShape1 = isTndLayout_ ? cosShape.GetDim(DIM_0) : cosShape.GetDim(DIM_1);
    uint64_t cosShape2 = isTndLayout_ ? cosShape.GetDim(DIM_1) : cosShape.GetDim(DIM_2);

    // 原始 5 种 layout 判断 → 映射到 4 个 Layout 值
    if (xShape0 == cosShape0 && xShape1 == cosShape1 && xShape2 == cosShape2) {
        // dy shape == cos shape — NO_BROADCAST → Layout=3 (KEY_A=205)
        layout_ = ApplyRopeLayout::NO_BROADCAST_AND_BROADCAST_BSN;
        cosb_ = xShape0;
    } else if (cosShape0 == 1 && cosShape1 == 1 && cosShape2 == 1) {
        // 111D 全广播 — BROADCAST_BSN → Layout=3 (KEY_B=206)
        layout_ = ApplyRopeLayout::NO_BROADCAST_AND_BROADCAST_BSN;
        cosb_ = 1;
    } else if (cosShape2 == 1 && cosShape0 == 1 && xShape1 == cosShape1) {
        // BSND (1S1D) → Layout=0 (BAB=203)
        layout_ = ApplyRopeLayout::BSND;
        cosb_ = cosShape0;
    } else if (cosShape2 == 1 && xShape0 == cosShape0 && (cosShape1 == 1 || cosShape1 == xShape1)) {
        // SBND (S11D, SB1D, BS1D) → Layout=1 (AB=204)
        layout_ = ApplyRopeLayout::SBND;
        cosb_ = cosShape1;
    } else if (cosShape1 == 1 && xShape2 == cosShape2 && (cosShape0 == 1 || cosShape0 == xShape0)) {
        // BNSD (11SD, B1SD) → Layout=2 (ABA/BA=201/202) — 预留
        layout_ = ApplyRopeLayout::BNSD;
        cosb_ = cosShape0;
    } else if (cosShape0 == 1 && xShape1 == cosShape1 && xShape2 == cosShape2) {
        // 1SND → reshape 为 BNSD → Layout=2 (预留)
        layout_ = ApplyRopeLayout::BNSD;
        cosb_ = cosShape0;
        is1snd_ = true;
    } else {
        std::string shapeMsg = ToString(cosShape) + " and " + ToString(dyQShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cos and dy_q", shapeMsg.c_str(),
            "Each axis of input cos except the last must be 1 or equal to the same axis of input dy_q");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckRotaryModeShapeRelation(const int64_t d)
{
    if (d > D_LIMIT) {
        std::string reasonMsg = "The D axis of input dy_q can not be greater than " + std::to_string(D_LIMIT) +
            ", where D refers to the last dim";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dy_q",
            ToString(context_->GetInputShape(DYQ_INDEX)->GetStorageShape()).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (d % HALF_MODE_COEF != 0) {
        std::string reasonMsg = "The D axis of input dy_q should be divisible by " + std::to_string(HALF_MODE_COEF) +
            " for half mode";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dy_q",
            ToString(context_->GetInputShape(DYQ_INDEX)->GetStorageShape()).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckShapeLimit()
{
    auto& dyQShape = context_->GetInputShape(DYQ_INDEX)->GetStorageShape();
    auto& dyKShape = context_->GetInputShape(DYK_INDEX)->GetStorageShape();
    auto& cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto& sinShape = context_->GetInputShape(SIN_INDEX)->GetStorageShape();
    auto& dxQShape = context_->GetOutputShape(DXQ_INDEX)->GetStorageShape();
    auto& dxKShape = context_->GetOutputShape(DXK_INDEX)->GetStorageShape();

    if (cosShape != sinShape) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "cos and sin",
            (ToString(cosShape) + " and " + ToString(sinShape)).c_str(),
            "The shapes of input cos and sin should be the same");
        return ge::GRAPH_FAILED;
    }
    if (dyQShape != dxQShape) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "dy_q and dx_q",
            (ToString(dyQShape) + " and " + ToString(dxQShape)).c_str(),
            "The shapes of input dy_q and output dx_q should be the same");
        return ge::GRAPH_FAILED;
    }
    if (dyKShape != dxKShape) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "dy_k and dx_k",
            (ToString(dyKShape) + " and " + ToString(dxKShape)).c_str(),
            "The shapes of input dy_k and output dx_k should be the same");
        return ge::GRAPH_FAILED;
    }

    // 检查可选的 x_q/x_k
    // dcos/dsin 需要两个 x 都存在
    bool hasXQ = (context_->GetInputShape(XQ_INDEX) != nullptr);
    bool hasXK = (context_->GetInputShape(XK_INDEX) != nullptr);
    if (hasXQ && hasXK) {
        dCosFlag_ = 1;
        return CheckOptionalInput();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckOptionalInput() const
{
    auto& cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto& sinShape = context_->GetInputShape(SIN_INDEX)->GetStorageShape();
    auto& dyQShape = context_->GetInputShape(DYQ_INDEX)->GetStorageShape();
    auto& dyKShape = context_->GetInputShape(DYK_INDEX)->GetStorageShape();
    auto& xQShape = context_->GetInputShape(XQ_INDEX)->GetStorageShape();
    auto& xKShape = context_->GetInputShape(XK_INDEX)->GetStorageShape();
    auto& dcosShape = context_->GetOutputShape(DCOS_INDEX)->GetStorageShape();
    auto& dsinShape = context_->GetOutputShape(DSIN_INDEX)->GetStorageShape();

    if (xQShape != dyQShape) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x_q and dy_q",
            (ToString(xQShape) + " and " + ToString(dyQShape)).c_str(),
            "The shapes of input x_q and dy_q should be the same");
        return ge::GRAPH_FAILED;
    }
    if (xKShape != dyKShape) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x_k and dy_k",
            (ToString(xKShape) + " and " + ToString(dyKShape)).c_str(),
            "The shapes of input x_k and dy_k should be the same");
        return ge::GRAPH_FAILED;
    }
    if (cosShape != dcosShape) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "cos and dcos",
            (ToString(cosShape) + " and " + ToString(dcosShape)).c_str(),
            "The shapes of input cos and output dcos should be the same");
        return ge::GRAPH_FAILED;
    }
    if (sinShape != dsinShape) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "sin and dsin",
            (ToString(sinShape) + " and " + ToString(dsinShape)).c_str(),
            "The shapes of input sin and output dsin should be the same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckShape()
{
    auto& cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto& dyQShape = context_->GetInputShape(DYQ_INDEX)->GetStorageShape();
    OP_CHECK_IF(CheckShapeDim() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "check shape dim fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeLimit() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "CheckShapeLimit fail."), return ge::GRAPH_FAILED);
    isTndLayout_ = (dyQShape.GetDimNum() == DIM_NUM_TND);
    int64_t dyLastDim = isTndLayout_ ? dyQShape.GetDim(DIM_2) : dyQShape.GetDim(DIM_3);
    int64_t cosLastDim = isTndLayout_ ? cosShape.GetDim(DIM_2) : cosShape.GetDim(DIM_3);
    if (cosLastDim != dyLastDim) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "dy_q and cos",
            (ToString(dyQShape) + " and " + ToString(cosShape)).c_str(),
            "The D axis of input dy_q and input cos should be the same");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(CheckRotaryModeShapeRelation(dyLastDim) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "D is invalid for half mode."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckDtypeAndAttr()
{
    dtype_ = context_->GetInputDesc(DYQ_INDEX)->GetDataType();
    if (std::find(SUPPORT_DTYPE.begin(), SUPPORT_DTYPE.end(), dtype_) == SUPPORT_DTYPE.end()) {
        std::string dtypeStr = ToString(dtype_);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dy_q", dtypeStr.c_str(), "FLOAT, BF16 or FLOAT16");
        return ge::GRAPH_FAILED;
    }
    for (int64_t i = DYQ_INDEX; i <= SIN_INDEX; i++) {
        auto type = context_->GetInputDesc(i)->GetDataType();
        if (type != dtype_) {
            std::string paramMsg = inputNames[i] + " and dy_q";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(),
                (ToString(type) + " and " + ToString(dtype_)).c_str(),
                ("The dtypes of input " + inputNames[i] + " and input dy_q should be the same").c_str());
            return ge::GRAPH_FAILED;
        }
    }
    if (context_->GetInputShape(XQ_INDEX) != nullptr && context_->GetInputShape(XK_INDEX) != nullptr) {
        for (int64_t i = XQ_INDEX; i <= XK_INDEX; i++) {
            auto type = context_->GetInputDesc(i)->GetDataType();
            if (type != dtype_) {
                std::string paramMsg = inputNames[i] + " and dy_q";
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(),
                    (ToString(type) + " and " + ToString(dtype_)).c_str(),
                    ("The dtypes of input " + inputNames[i] + " and input dy_q should be the same").c_str());
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckParam()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
        OP_LOGE(context_, "platform info is nullptr."), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        return ge::GRAPH_SUCCESS;  // 非 regbase 芯片直接返回成功（不会调用到此分支，因为只有 ascend950）
    }
    OP_CHECK_IF(CheckNullptr() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "check nullptr fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDtypeAndAttr() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "check dtype and attr fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "check shape fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeAllPositive() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "check shape positive fail."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::GetShapeAttrsInfo()
{
    const gert::RuntimeAttrs* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int32_t* mode = attrs->GetAttrPointer<int32_t>(0);
    int32_t modeValue = (mode == nullptr) ? 0 : static_cast<int32_t>(*mode);
    // 仅支持 half (0)
    if (modeValue != 0) {
        std::string modeStr = std::to_string(modeValue);
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "mode", modeStr.c_str(), "0 (half only)");
        return ge::GRAPH_FAILED;
    }
    rotaryMode_ = modeValue;

    OP_CHECK_IF(CheckParam() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "check param fail."), return ge::GRAPH_FAILED);

    dtype_ = context_->GetInputDesc(DYQ_INDEX)->GetDataType();
    auto dyQShape = context_->GetInputShape(DYQ_INDEX)->GetStorageShape();
    auto cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto dyKShape = context_->GetInputShape(DYK_INDEX)->GetStorageShape();
    OP_CHECK_IF(JudgeLayoutByShape(dyQShape, cosShape) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "JudgeLayoutByShape fail."), return ge::GRAPH_FAILED);

    d_ = isTndLayout_ ? dyQShape.GetDim(DIM_2) : dyQShape.GetDim(DIM_3);
    if (isTndLayout_) {
        // TND (3D): B=1, S=T, N=N — 维度提取后走正常的 5 种 layout 判断
        // TND+cos=T1D → BSND (BAB), TND+cos=1ND → NO_BROADCAST_AND_BROADCAST_BSN (A/B)
        b_ = 1;
        s_ = dyQShape.GetDim(DIM_0);
        nQ_ = dyQShape.GetDim(DIM_1);
        nK_ = dyKShape.GetDim(DIM_1);
        // cosb_ 已在 JudgeLayoutByShape 中由 layout 判定设置
    } else if (layout_ == ApplyRopeLayout::BSND) {
        b_ = dyQShape.GetDim(DIM_0);
        cosb_ = cosShape.GetDim(DIM_0);
        s_ = dyQShape.GetDim(DIM_1);
        nQ_ = dyQShape.GetDim(DIM_2);
        nK_ = dyKShape.GetDim(DIM_2);
    } else if (layout_ == ApplyRopeLayout::BNSD) {
        b_ = dyQShape.GetDim(DIM_0);
        cosb_ = cosShape.GetDim(DIM_0);
        nQ_ = dyQShape.GetDim(DIM_1);
        nK_ = dyKShape.GetDim(DIM_1);
        s_ = dyQShape.GetDim(DIM_2);
        if (is1snd_) {
            s_ = s_ * nQ_;
            nQ_ = 1;
            nK_ = 1;
        }
    } else if (layout_ == ApplyRopeLayout::SBND) {
        s_ = dyQShape.GetDim(DIM_0);
        b_ = dyQShape.GetDim(DIM_1);
        cosb_ = cosShape.GetDim(DIM_1);
        nQ_ = dyQShape.GetDim(DIM_2);
        nK_ = dyKShape.GetDim(DIM_2);
    } else {
        // NO_BROADCAST_AND_BROADCAST_BSN (4D): 全同或全广播，按 BNSD 方式提取维度即可
        // A/B tiling 会通过 MergeDim 将所有维合并到 b_
        b_ = dyQShape.GetDim(DIM_0);
        cosb_ = cosShape.GetDim(DIM_0);
        nQ_ = dyQShape.GetDim(DIM_1);
        nK_ = dyKShape.GetDim(DIM_1);
        s_ = dyQShape.GetDim(DIM_2);
    }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::InitTilingData()
{
    if (tilingData_ == nullptr) {
        tilingData_ = context_->GetTilingData<ApplyRopeGradTilingData>();
        OP_CHECK_IF(tilingData_ == nullptr,
            OP_LOGE(context_->GetNodeName(), "get tilingdata ptr failed"), return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((memset_s(tilingData_, sizeof(ApplyRopeGradTilingData), 0, sizeof(ApplyRopeGradTilingData)) != EOK),
        OP_LOGE(context_->GetNodeName(), "memset tilingdata failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::GetInputParam(
    Ops::Base::ReduceOpInputParam& opInput, uint32_t inputIdx, uint32_t axesIdx)
{
    auto inputDesc = context_->GetInputDesc(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    opInput.inputDtype = inputDesc->GetDataType();
    auto dyInput = context_->GetInputTensor(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyInput);
    auto dyInputShape = Ops::Transformer::OpTiling::EnsureNotScalar(dyInput->GetStorageShape());
    size_t shapeSize = dyInputShape.GetDimNum();
    opInput.shape.resize(shapeSize);
    for (size_t i = 0; i < shapeSize; i++) {
        opInput.shape[i] = dyInputShape.GetDim(i);
    }
    inputDesc = context_->GetInputDesc(axesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    auto cosInput = context_->GetInputTensor(axesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cosInput);
    auto cosShapeRef = Ops::Transformer::OpTiling::EnsureNotScalar(cosInput->GetStorageShape());
    for (size_t i = 0; i < shapeSize; i++) {
        if (cosShapeRef.GetDim(i) == 1 && opInput.shape[i] != 1) {
            opInput.axes.push_back(i);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::TilingReduce()
{
    Ops::Base::ReduceOpInputParam opInput;
    OP_CHECK_IF((GetInputParam(opInput, DYQ_INDEX, COS_INDEX) == ge::GRAPH_FAILED),
        OP_LOGE(context_->GetNodeName(), "ReduceOp get input param failed"), return ge::GRAPH_FAILED);
    ge::graphStatus status;
    if (ge::GetSizeByDataType(opInput.inputDtype) == sizeof(float)) {
        status = Ops::Base::Tiling4ReduceOp<ApplyRotaryPosEmbGrad::ApplyRopeGradDag<float, float>::OpDag>(
            context_, opInput, key_, &tilingData_->reduceTiling);
    } else {
        status = Ops::Base::Tiling4ReduceOp<ApplyRotaryPosEmbGrad::ApplyRopeGradDag<Ops::Base::half, float>::OpDag>(
            context_, opInput, key_, &tilingData_->reduceTiling);
    }
    OP_CHECK_IF(status == ge::GRAPH_FAILED,
        OP_LOGE(context_->GetNodeName(), "ReduceOp Tiling failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::SetTilingKeyBlockDim(uint32_t dxTilingKey, uint32_t layout)
{
    GEN_REDUCE_TILING_KEY(tilingKey_, key_, dxTilingKey, dCosFlag_);
    // 将 layout 编码进 tiling key 高位
    tilingKey_ |= (static_cast<uint64_t>(layout) << (key_.totalBitCount + 1));
    OP_LOGI(context_->GetNodeName(),
        "patternID:%u, dxTilingKey:%u, dCosFlag:%u, layout:%u, tilingKey:%lu.",
        key_.patternID, dxTilingKey, dCosFlag_, layout, tilingKey_);
    int64_t reduceBlockNum = context_->GetBlockDim();
    context_->SetBlockDim(std::max(usedCoreNum_, reduceBlockNum));
    context_->SetTilingKey(tilingKey_);
    auto workspaces = context_->GetWorkspaceSizes(1);
    // workspace = xQ_rot + xK_rot + reduce workspace
    int64_t totalN = nQ_ + nK_;
    auto usrWorkSpaceSize = b_ * s_ * totalN * d_ * ge::GetSizeByDataType(dtype_);
    workspaces[0] = workspaces[0] + usrWorkSpaceSize;
    if (dCosFlag_ == 1) {
        context_->SetScheduleMode(1);
    }
    OP_LOGD(context_->GetNodeName(), "workspaces[0]:%ld, usrWorkSpaceSize:%ld.", workspaces[0], usrWorkSpaceSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::SetRotaryXTilingData()
{
    int64_t ubFactorN = 1;
    int64_t totalN = nQ_ + nK_;
    int64_t dSumNum = b_ * s_ * totalN;
    int64_t blockFactorN = Ops::Base::CeilDiv(dSumNum, usedCoreNum_);
    int64_t rotaryUsedCoreNum = Ops::Base::CeilDiv(dSumNum, blockFactorN);
    int64_t blockTailFactorN = (dSumNum % blockFactorN == 0) ? blockFactorN : (dSumNum % blockFactorN);
    int64_t dSize = Ops::Base::CeilAlign(d_ * GetSizeByDataType(dtype_) / dSplitCoef_, blockSize_) * dSplitCoef_;
    int64_t numOfDAvailable = Ops::Base::FloorDiv(static_cast<int64_t>(aicoreParams_.ubSize), DOUBLE_BUFFER * dSize);
    ubFactorN = numOfDAvailable / 3; // 1 输入 + 2 输出 (x_rot workspace for both cos/sin)
    ubFactorN = std::min(blockFactorN, ubFactorN);

    tilingData_->rotaryXParams.b = b_;
    tilingData_->rotaryXParams.s = s_;
    tilingData_->rotaryXParams.nQ = nQ_;
    tilingData_->rotaryXParams.nK = nK_;
    tilingData_->rotaryXParams.d = d_;
    tilingData_->rotaryXParams.usedCoreNum = rotaryUsedCoreNum;
    tilingData_->rotaryXParams.blockFactorN = blockFactorN;
    tilingData_->rotaryXParams.blockTailFactorN = blockTailFactorN;
    tilingData_->rotaryXParams.ubFactorN = ubFactorN;
    tilingData_->rotaryXParams.rotaryMode = rotaryMode_;

    OP_LOGI(context_->GetNodeName(),
        "RotaryX tilingData: usedCoreNum=%ld, blockFactorN=%ld, blockTailFactorN=%ld, ubFactorN=%ld",
        tilingData_->rotaryXParams.usedCoreNum, tilingData_->rotaryXParams.blockFactorN,
        tilingData_->rotaryXParams.blockTailFactorN, tilingData_->rotaryXParams.ubFactorN);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::GetReduceOpCompileInfo(Ops::Base::ReduceOpCompileInfo* compileInfo)
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(compileInfo->vectorCoreNum == 0,
        OP_LOGE(context_->GetNodeName(), "ReduceOp GetHardwareInfo Failed, vectorCoreNum:%lu", compileInfo->vectorCoreNum),
        return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    compileInfo->cacheLineSize = Ops::Base::GetCacheLineSize(context_);
    compileInfo->ubBlockSize = Ops::Base::GetUbBlockSize(context_);
    compileInfo->vRegSize = Ops::Base::GetVRegSize(context_);
    return ge::GRAPH_SUCCESS;
}

uint64_t ApplyRopeGradRegbaseTilingClass::GetTilingKey() const
{
    return tilingKey_;
}

} // namespace optiling
