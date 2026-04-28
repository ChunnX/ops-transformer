/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qbmm_reduce_scatter_add_rms_norm_cast_tiling_check.cpp
 * \brief
 */
#include "qbmm_reduce_scatter_add_rms_norm_cast_tiling_check.h"
namespace MC2Tiling {

bool QbmmReduceScatterAddRmsNormCastCheckTiling::CheckAttrs(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return false);
    // group空字符串校验
    const char *groupPtr = attrs->GetAttrPointer<char>(GROUP_INDEX);
    OP_TILING_CHECK(groupPtr == nullptr, OP_LOGE(nodeName, "groupPtr is null."), return false);
    OP_TILING_CHECK(std::string(groupPtr).empty(),
        OP_LOGE(nodeName, "group should not be empty."), return false);
    // transpose校验
    const bool *transposeX2Ptr = attrs->GetAttrPointer<bool>(TRANSPOSE_X2_INDEX);
    OP_TILING_CHECK(transposeX2Ptr == nullptr, OP_LOGE(nodeName, "transposeX2Ptr is nullptr."), return false);
    OP_TILING_CHECK(*transposeX2Ptr,
        OP_LOGE(nodeName, "transposeX2 should be false."), return false);
    // 输出type校验（猜测属性的dtype和最终输出的y1的类型应该保持一致）
    const int64_t *outputTypePtr = attrs->GetAttrPointer<int64_t>(OUT_PUT_DTYPE_INDEX);
    OP_TILING_CHECK(outputTypePtr == nullptr, OP_LOGE(nodeName, "outputTypePtr is nullptr."), return false);
    ge::DataType outputType = static_cast<ge::DataType>(*outputTypePtr);
    OP_TILING_CHECK(outputType != ge::DT_FLOAT,
                    OP_LOGE(nodeName, "outPutType should be float, but actual value is %s.",
                            Ops::Base::ToString(outputType).c_str()),
                    return false);
    // epsilon校验
    const float *epsilonPtr = attrs->GetAttrPointer<float>(TRANSPOSE_X2_INDEX);
    OP_TILING_CHECK(epsilonPtr == nullptr, OP_LOGE(nodeName, "epsilonPtr is nullptr."), return false);
    // OP_TILING_CHECK(*epsilonPtr != 1e-6f,
    //     OP_LOGE(nodeName, "epsilon should be 1e-6f."), return false);
    return true;
}

bool QbmmReduceScatterAddRmsNormCastCheckTiling::CheckTensorFormat(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    ge::Format x1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(X1_INDEX)->GetStorageFormat()));
    ge::Format x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(X2_INDEX)->GetStorageFormat()));
    ge::Format yFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(Y_INDEX)->GetStorageFormat()));
    ge::Format gammaFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(GAMMA_INDEX)->GetStorageFormat()));
    ge::Format scaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(SCALE_INDEX)->GetStorageFormat()));
    ge::Format biasFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(BIAS_INDEX)->GetStorageFormat()));
    ge::Format pertokenScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(PER_TOKEN_SCALE_INDEX)->GetStorageFormat()));
    ge::Format y1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(Y1_INDEX)->GetStorageFormat()));
    ge::Format y2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(Y2_INDEX)->GetStorageFormat()));
    ge::Format xFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(X_INDEX)->GetStorageFormat()));
    OP_TILING_CHECK((x1Format != ge::FORMAT_ND) && (yFormat != ge::FORMAT_ND) && (gammaFormat != ge::FORMAT_ND) && (scaleFormat != ge::FORMAT_ND) && 
        (biasFormat != ge::FORMAT_ND) && (pertokenScaleFormat != ge::FORMAT_ND),
        OP_LOGE(nodeName, "The input format of x1/y/gamma/scale/bias/pertokenScale should be ND, but current format of x1/y/gamma/scale/bias/pertokenScale"
        "are %s/%s/%s/%s/%s/%s.", Ops::Base::ToString(x1Format).c_str(), Ops::Base::ToString(yFormat).c_str(), Ops::Base::ToString(gammaFormat).c_str(),
        Ops::Base::ToString(scaleFormat).c_str(), Ops::Base::ToString(biasFormat).c_str(), Ops::Base::ToString(pertokenScaleFormat).c_str()), return false);

    OP_TILING_CHECK((y1Format != ge::FORMAT_ND) && (y2Format != ge::FORMAT_ND) && (xFormat != ge::FORMAT_ND),
        OP_LOGE(nodeName, "The output format of y1/y2/x should be ND, but current format of y1/y2/x"
        "are %s/%s/%s.", Ops::Base::ToString(y1Format).c_str(), Ops::Base::ToString(y2Format).c_str(), Ops::Base::ToString(xFormat).c_str()), return false);
    
    OP_TILING_CHECK((x2Format != ge::FORMAT_FRACTAL_NZ) && (x2Format != ge::FORMAT_ND),
        OP_LOGE(nodeName, "x2 format should be ND or NZ, but current x2 format is %s.",
        Ops::Base::ToString(x2Format).c_str()), return false);
    return true;
}

bool QbmmReduceScatterAddRmsNormCastCheckTiling::CheckTensorDim(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    const gert::StorageShape *x1Shape = context->GetInputShape(X1_INDEX);
    ge::Format x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(X2_INDEX)->GetStorageFormat()));
    // 输入
    OP_TILING_CHECK(x1Shape == nullptr, OP_LOGE(nodeName, "x1Shape is null."), return false);
    const gert::StorageShape *x2Shape = context->GetInputShape(X2_INDEX);
    OP_TILING_CHECK(x2Shape == nullptr, OP_LOGE(nodeName, "x2Shape is null."), return false);
    const gert::StorageShape *yShape = context->GetInputShape(Y_INDEX);
    OP_TILING_CHECK(yShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return false);
    const gert::StorageShape *gammaShape = context->GetInputShape(GAMMA_INDEX);
    OP_TILING_CHECK(gammaShape == nullptr, OP_LOGE(nodeName, "gammaShape is null."), return false);
    const gert::StorageShape *scaleShape = context->GetInputShape(SCALE_INDEX);
    OP_TILING_CHECK(scaleShape == nullptr, OP_LOGE(nodeName, "scaleShape is null."), return false);
    const gert::StorageShape *biasShape = context->GetInputShape(BIAS_INDEX);
    OP_TILING_CHECK(biasShape == nullptr, OP_LOGE(nodeName, "biasShape is null."), return false);
    const gert::StorageShape *perTokenScaleShape = context->GetInputShape(PER_TOKEN_SCALE_INDEX);
    OP_TILING_CHECK(perTokenScaleShape == nullptr, OP_LOGE(nodeName, "perTokenScaleShape is null."), return false);
    // 输出
    const gert::StorageShape *y1Shape = context->GetOutputShape(Y1_INDEX);
    OP_TILING_CHECK(y1Shape == nullptr, OP_LOGE(nodeName, "y1Shape is null."), return false);
    const gert::StorageShape *y2Shape = context->GetOutputShape(Y2_INDEX);
    OP_TILING_CHECK(y2Shape == nullptr, OP_LOGE(nodeName, "y2Shape is null."), return false);
    const gert::StorageShape *xShape = context->GetOutputShape(X_INDEX);
    OP_TILING_CHECK(xShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return false);
    // 获取维度
    size_t x1Dim = x1Shape->GetStorageShape().GetDimNum();
    size_t x2Dim = x2Shape->GetStorageShape().GetDimNum();
    size_t yDim = yShape->GetStorageShape().GetDimNum();
    size_t gammaDim = gammaShape->GetStorageShape().GetDimNum();
    size_t scaleDim = scaleShape->GetStorageShape().GetDimNum();
    size_t biasDim = biasShape->GetStorageShape().GetDimNum();
    size_t perTokenScaleDim = perTokenScaleShape->GetStorageShape().GetDimNum();
    size_t y1Dim = y1Shape->GetStorageShape().GetDimNum();
    size_t y2Dim = y2Shape->GetStorageShape().GetDimNum();
    size_t xDim = xShape->GetStorageShape().GetDimNum();
    // 获取维度值
    uint64_t x1ValueOne = x1Shape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t x1ValueTwo = x1Shape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t x2ValueOne = x2Shape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t x2ValueTwo = x2Shape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t x2ValueThree = 0;
    uint64_t x2ValueFour = 0;
    if (x2Format == ge::FORMAT_FRACTAL_NZ) {
        x2ValueThree = x2Shape->GetStorageShape().GetDim(DIM_TWO);
        x2ValueFour = x2Shape->GetStorageShape().GetDim(DIM_THREE);
    }
    uint64_t yValueOne = yShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t yValueTwo = yShape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t gammaValue = gammaShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t scaleValue = scaleShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t biasValue = biasShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t pertokenScaleValue = perTokenScaleShape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t y1ValueOne = y1Shape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t y1ValueTwo = y1Shape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t y2ValueOne = y2Shape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t y2ValueTwo = y2Shape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t xValueOne = xShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t xValueTwo = xShape->GetStorageShape().GetDim(DIM_ONE);
    // TODO:抽成函数
    //校验维度
    OP_TILING_CHECK((x1Dim != TWO_DIMS) || (yDim != TWO_DIMS) || (y1Dim != TWO_DIMS) || (y2Dim != TWO_DIMS) || (xDim != TWO_DIMS),
        OP_LOGE(nodeName, "The dim of x1,y,y1,y2,x should be 2, but current x1Dim=%lu, yDim=%lu, y1Dim=%lu, y2Dim=%lu, xDim=%lu.",
                x1Dim, yDim, y1Dim, y2Dim, xDim), return false);
    OP_TILING_CHECK((x2Dim != FOUR_DIMS) && (x2Dim != TWO_DIMS),
        OP_LOGE(nodeName, "x2Dim should be 4 or 2, but current x2Dim is %lu.", x2Dim), return false);
    OP_TILING_CHECK((gammaDim != ONE_DIM) || (scaleDim != ONE_DIM) || (biasDim != ONE_DIM) || (perTokenScaleDim != ONE_DIM),
        OP_LOGE(nodeName, "The dim of gamma,scale,bias,perTokenScale should be 1, but current gammaDim=%lu, scaleDim=%lu, biasDim=%lu, perTokenScaleDim=%lu",
                gammaDim, scaleDim, biasDim, perTokenScaleDim), return false);
    //校验维度值关系
    OP_TILING_CHECK((yValueOne != y1ValueOne) && (yValueOne != y2ValueOne) && (yValueOne != xValueOne) && 
                    (yValueTwo != y1ValueTwo) && (yValueTwo != y2ValueTwo) && (yValueTwo != xValueTwo),
        OP_LOGE(nodeName, "The dims of y,y1,y2,x should be match, but current yShape=(%lu,%lu), y1Shape=(%lu,%lu), y2Shape=(%lu,%lu), xShape=(%lu,%lu)",
                yValueOne, yValueTwo, y1ValueOne, y1ValueTwo, y2ValueOne, y2ValueTwo, xValueOne, xValueTwo), return false);
    
    OP_TILING_CHECK((yValueTwo != gammaValue) && (yValueTwo != scaleValue) && (yValueTwo != biasValue) && (yValueTwo != pertokenScaleValue),
        OP_LOGE(nodeName, "The dims of gamma,scale,bias,pertokenScale should be match N, but current gammaDim=%lu, scaleDim=%lu, biasDim=%lu, pertokenScaleDim=%lu, N=%lu)",
                gammaValue, scaleValue, biasValue, pertokenScaleValue, yValueTwo), return false);
    if (x2Format == ge::FORMAT_FRACTAL_NZ) {
        OP_TILING_CHECK((x2ValueTwo * x2ValueThree != x1ValueTwo) || (x2ValueOne * x2ValueFour != yValueTwo),
            OP_LOGE(nodeName, "When x2 is NZ format, x2Dim1 * x2Dim2 should be equal to K(x1Dim1), yDim1 should be equal to N(x2Dim0 * x2Dim3),"
            "but current x2Dim1/x2Dim2/x1Dim1/yDim1/x2Dim0/x2Dim3 are %lu/%lu/%lu/%lu/%lu/%lu)",
            x2ValueTwo, x2ValueThree, x1ValueTwo, yValueTwo, x2ValueOne, x2ValueFour), return false);
    } else {
        OP_TILING_CHECK((x2ValueOne != x1ValueTwo) && (x2ValueTwo != yValueTwo),
            OP_LOGE(nodeName, "When x2 is ND format, x2Dim0 should be equal to x1Dim1, x2Dim1 should be equal to yDim1, but current x2Dim0/x1Dim1/x2Dim1/yDim1 are %lu/%lu/%lu/%lu)",
            x2ValueOne, x1ValueTwo, x2ValueTwo, yValueTwo), return false);
    }
    OP_TILING_CHECK((yValueOne != (x1ValueOne/TP_NUMBER)),
        OP_LOGE(nodeName, "yDim0 should be equal to x1Dim0/tp_num, but current yDim0/x1Dim0/tp_num are %lu/%lu/%lu)",
                yValueOne, x1ValueOne, TP_NUMBER), return false);
    return true;
}


bool QbmmReduceScatterAddRmsNormCastCheckTiling::CheckTensorDataType(const gert::TilingContext *context)
{
    // 获取输入输出的dtype
    const char *nodeName = context->GetNodeName();
    auto x1Desc = context->GetInputDesc(X1_INDEX);
    auto x2Desc = context->GetInputDesc(X2_INDEX);
    auto yDesc = context->GetInputDesc(Y_INDEX);
    auto gammaDesc = context->GetInputDesc(GAMMA_INDEX);
    auto scaleDesc = context->GetInputDesc(SCALE_INDEX);
    auto biasDesc = context->GetInputDesc(BIAS_INDEX);
    auto perTokenScaleDesc = context->GetInputDesc(PER_TOKEN_SCALE_INDEX);
    auto y1ScaleDesc = context->GetOutputDesc(Y1_INDEX);
    auto y2ScaleDesc = context->GetOutputDesc(Y2_INDEX);
    auto xScaleDesc = context->GetOutputDesc(X_INDEX);
    OP_TILING_CHECK(x1Desc == nullptr, OP_LOGE(nodeName, "x1Desc is null."), return false);
    OP_TILING_CHECK(x2Desc == nullptr, OP_LOGE(nodeName, "x2Desc is null."), return false);
    OP_TILING_CHECK(yDesc == nullptr, OP_LOGE(nodeName, "yDesc is null."), return false);
    OP_TILING_CHECK(gammaDesc == nullptr, OP_LOGE(nodeName, "gammaDesc is null."), return false);
    OP_TILING_CHECK(scaleDesc == nullptr, OP_LOGE(nodeName, "scaleDesc is null."), return false);
    OP_TILING_CHECK(biasDesc == nullptr, OP_LOGE(nodeName, "biasDesc is null."), return false);
    OP_TILING_CHECK(perTokenScaleDesc == nullptr, OP_LOGE(nodeName, "perTokenScaleDesc is null."), return false);
    OP_TILING_CHECK(y1ScaleDesc == nullptr, OP_LOGE(nodeName, "y1ScaleDesc is null."), return false);
    OP_TILING_CHECK(y2ScaleDesc == nullptr, OP_LOGE(nodeName, "y2ScaleDesc is null."), return false);
    OP_TILING_CHECK(xScaleDesc == nullptr, OP_LOGE(nodeName, "xScaleDesc is null."), return false);
    ge::DataType x1Dtype = x1Desc->GetDataType();
    ge::DataType x2Dtype = x2Desc->GetDataType();
    ge::DataType yDtype = yDesc->GetDataType();
    ge::DataType gammaDtype = gammaDesc->GetDataType();
    ge::DataType scaleDtype = scaleDesc->GetDataType();
    ge::DataType biasDtype = biasDesc->GetDataType();
    ge::DataType perTokenScaleDtype = perTokenScaleDesc->GetDataType();
    ge::DataType y1Dtype =y1ScaleDesc->GetDataType();
    ge::DataType y2Dtype = y2ScaleDesc->GetDataType();
    ge::DataType xDtype = xScaleDesc->GetDataType();
    // 校验输入输出的dtype TODO:抽成函数
    OP_TILING_CHECK((yDtype != y2Dtype) && (yDtype != xDtype) && ((yDtype != ge::DT_BF16) || (yDtype != ge::DT_FLOAT16)),
        OP_LOGE(nodeName, "The dataType of y/x/y2 should be the same, and the dtype should be float16 or bfloat16,"
        "but current y/x/y2 dtype are %s/%s/%s",
        Ops::Base::ToString(yDtype).c_str(), Ops::Base::ToString(xDtype).c_str(), Ops::Base::ToString(y2Dtype).c_str()), return false);
    OP_TILING_CHECK((x1Dtype != ge::DT_INT8) && (x2Dtype != ge::DT_INT8),
        OP_LOGE(nodeName, "The dataType of x1/x2 should be the same, and the dtype should be int8, but current x1/x2 dtype are %s/%s.",
        Ops::Base::ToString(x1Dtype).c_str(), Ops::Base::ToString(x2Dtype).c_str()), return false);
    OP_TILING_CHECK(((scaleDtype != ge::DT_BF16) && (scaleDtype != ge::DT_FLOAT)),
        OP_LOGE(nodeName, "The dataType of scale should be bfloat16 or float, but current dtype is %s.",
        Ops::Base::ToString(scaleDtype).c_str()), return false);
    OP_TILING_CHECK(perTokenScaleDtype != ge::DT_FLOAT, OP_LOGE(nodeName, "The dataType of pertokenScale should be float, but current dtype is %s.",
        Ops::Base::ToString(perTokenScaleDtype).c_str()), return false);
    OP_TILING_CHECK((gammaDtype != ge::DT_FLOAT) && (gammaDtype != ge::DT_BF16), OP_LOGE(nodeName, "The dataType of gamma should be the float, but current gamma dtype is %s.",
        Ops::Base::ToString(gammaDtype).c_str()), return false);
    OP_TILING_CHECK(y1Dtype != ge::DT_FLOAT, OP_LOGE(nodeName, "The dataType of y1 should be the float, but current y1 dtype is %s.",
        Ops::Base::ToString(y1Dtype).c_str()), return false);
    // OP_TILING_CHECK((biasDtype != ge::DT_FLOAT && biasDtype != ge::DT_BF16 && biasDtype != ge::DT_FLOAT16 && biasDtype != ge::DT_INT32), OP_LOGE(nodeName, "The dataType of bias should be the bfloat16 or int32 or float16 or float, but current bias dtype is %s.",
    //     Ops::Base::ToString(biasDtype).c_str()), return false);
    return true;
}

ge::graphStatus QbmmReduceScatterAddRmsNormCastCheckTiling::TilingCheckQbmmReduceScatterAddRmsNormCast(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(!CheckAttrs(context),
        OP_LOGE(nodeName, "attr is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorFormat(context),
        OP_LOGE(nodeName, "params format is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDataType(context),
        OP_LOGE(nodeName, "params dtype is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDim(context),
        OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace MC2Tiling
