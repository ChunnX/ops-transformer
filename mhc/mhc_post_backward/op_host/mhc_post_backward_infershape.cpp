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
 * \file mhc_post_backward_infershape.cpp
 * \brief MhcPostBackward infershape implementation
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_info.h"
#include "runtime/rt_external_base.h"
#include "platform/soc_spec.h"

using namespace ge;

namespace ops {

static constexpr size_t INDEX_GRADY = 0;
static constexpr size_t INDEX_X = 1;
static constexpr size_t INDEX_HRES = 2;
static constexpr size_t INDEX_HOUT = 3;
static constexpr size_t INDEX_HPOST = 4;
static constexpr size_t INDEX_GRADX = 0;
static constexpr size_t INDEX_GRADHRES = 1;
static constexpr size_t INDEX_GRADHOUT = 2;
static constexpr size_t INDEX_GRADHPOST = 3;
static constexpr size_t DIMS_THREE = 3;
static constexpr size_t DIMS_FOUR = 4;
static constexpr int64_t UNKNOWN_RANK_DIM_VALUE = -2LL;
static constexpr int64_t UNKNOWN_DIM_VALUE = -1LL;

void SetUnknownRank(gert::Shape &shape)
{
    shape.SetDimNum(0);
    shape.AppendDim(UNKNOWN_RANK_DIM_VALUE);
}

bool IsUnknownRank(const gert::Shape &shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == UNKNOWN_RANK_DIM_VALUE;
}

bool IsUnknownShape(const gert::Shape &shape)
{
    size_t dimNum = shape.GetDimNum();
    for (size_t i = 0; i < dimNum; i++) {
        if (shape.GetDim(i) == UNKNOWN_DIM_VALUE) {
            return true;
        }
    }
    return false;
}

void CopyShape(const gert::Shape &src, const gert::Shape &dst){
    size_t dimNum = src.GetDimNum();
    dst.SetDim(dimNum);
     for (size_t i = 0; i < dimNum; i++) {
        dst.SetDim(i, src.GetDim(i));
    }

}

static void ShowInputShapeInfo(gert::InferShapeContext *context, const gert::Shape *gradYShape, const gert::Shape *xShape,
                               const gert::Shape *hResShape, const gert::Shape *hOutShape,
                               const gert::Shape *hPostShape)
{   
    OP_LOGD(context, "gradYShape is: %s.", Ops::Base::ToString(*gradYShape).c_str());
    OP_LOGD(context, "xShape is: %s.", Ops::Base::ToString(*xShape).c_str());
    OP_LOGD(context, "hResShape is: %s.", Ops::Base::ToString(*hResShape).c_str());
    OP_LOGD(context, "hOutShape is: %s.", Ops::Base::ToString(*hOutShape).c_str());
    OP_LOGD(context, "hPostShape is: %s.", Ops::Base::ToString(*hPostShape).c_str());
}

// ShowOutputShapeInfo(context, gradXShape, gradHresShape, gradHoutShape, gradHpostShape)
static void ShowOutputShapeInfo(gert::InferShapeContext *context, const gert::Shape *gradXShape, const gert::Shape *gradHresShape, const gert::Shape *gradHoutShape, const gert::Shape *gradHpostShape)
{
    OP_LOGD(context, "gradXShape is: %s.", Ops::Base::ToString(*gradXShape).c_str());
    OP_LOGD(context, "gradHresShape is: %s.", Ops::Base::ToString(*gradHresShape).c_str());
    OP_LOGD(context, "gradHoutShape is: %s.", Ops::Base::ToString(*gradHoutShape).c_str());
    OP_LOGD(context, "gradHpostShape is: %s.", Ops::Base::ToString(*gradHpostShape).c_str());
}

static ge::graphStatus InferShapeForMhcPostBackward(gert::InferShapeContext *context)
{
    OP_LOGD(context, "Begin to do MhcPostInfershape.");
    const gert::Shape *gradYShape = context->GetInputShape(INDEX_GRADY);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradYShape);
    const gert::Shape *xShape = context->GetInputShape(INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *hResShape = context->GetInputShape(INDEX_HRES);
    OP_CHECK_NULL_WITH_CONTEXT(context, hResShape);
    const gert::Shape *hOutShape = context->GetInputShape(INDEX_HOUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, hOutShape);
    const gert::Shape *hPostShape = context->GetInputShape(INDEX_HPOST);
    OP_CHECK_NULL_WITH_CONTEXT(context, hPostShape);
    gert::Shape *gradXShape = context->GetOutputShape(INDEX_GRADX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradXShape);
    gert::Shape *gradHresShape = context->GetOutputShape(INDEX_GRADHRES);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradHresShape);
    gert::Shape *gradHoutShape = context->GetOutputShape(INDEX_GRADHOUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradHoutShape);
    gert::Shape *gradHpostShape = context->GetOutputShape(INDEX_GRADHPOST);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradHpostShape);
  
    if (IsUnknownRank(*xShape)) {
        SetUnknownRank(*gradXShape);
        OP_LOGD(context->GetNodeName(), "MhcPostBackward infershape handles unknown x rank.");
        return ge::GRAPH_SUCCESS;
    }
    if (IsUnknownRank(*hResShape)) {
        SetUnknownRank(*gradHresShape);
        OP_LOGD(context->GetNodeName(), "MhcPostBackward infershape handles unknown hRes rank.");
        return ge::GRAPH_SUCCESS;
    }
    if (IsUnknownRank(*hOutShape)) {
        SetUnknownRank(*gradHoutShape);
        OP_LOGD(context->GetNodeName(), "MhcPostBackward infershape handles unknown hOut rank.");
        return ge::GRAPH_SUCCESS;
    }
    if (IsUnknownRank(*hPostShape)) {
        SetUnknownRank(*gradHpostShape);
        OP_LOGD(context->GetNodeName(), "MhcPostBackward infershape handles unknown hPost rank.");
        return ge::GRAPH_SUCCESS;
    }

    if (IsUnknownShape(*xShape)) {
        CopyShape(*xShape, *gradXShape);
        OP_LOGD(context->GetNodeName(), "MhcPostBackward infershape handles unknown x shape.");
        return ge::GRAPH_SUCCESS;
    }

    if(IsUnknownShape(*hResShape)) {
        CopyShape(*hResShape, *gradHresShape);
        OP_LOGD(context->GetNodeName(), "MhcPostBackward infershape handles unknown hRes shape.");
        return ge::GRAPH_SUCCESS;
    }
    if(IsUnknownShape(*hOutShape)) {
        CopyShape(*hOutShape, *gradHoutShape);
        OP_LOGD(context->GetNodeName(), "MhcPostBackward infershape handles unknown hOut shape.");
        return ge::GRAPH_SUCCESS;
    }
    if(IsUnknownShape(*hPostShape)) {
        CopyShape(*hPostShape, *gradHpostShape);
        OP_LOGD(context->GetNodeName(), "MhcPostBackward infershape handles unknown hPost shape.");
        return ge::GRAPH_SUCCESS;
    }
    size_t xDims = xShape->GetDimNum();
    OP_CHECK_IF((xDims != DIMS_THREE) && (xDims != DIMS_FOUR),
                OP_LOGE(context->GetNodeName(), "The dim of x should be 3 or 4, but got %lu", xDims),
                return ge::GRAPH_FAILED);

    // Output shape is same as input x
    gradXShape->SetDimNum(xDims);
    for (size_t i = 0; i < xDims; ++i) {
        gradXShape->SetDim(i, xShape->GetDim(i));
    }
    size_t hResDims = hResShape->GetDimNum();
    gradHresShape->SetDimNum(hResDims);
    for (size_t i = 0; i < hResDims; ++i) {
       gradHresShape->SetDim(i, hResShape->GetDim(i));
    }

    size_t hOutDims = hOutShape->GetDimNum();
    gradHoutShape->SetDimNum(hOutDims);
    for (size_t i = 0; i < hOutDims; ++i) {
        gradHoutShape->SetDim(i, hOutShape->GetDim(i));
    }

    size_t hPostDims = hPostShape->GetDimNum();
    gradHpostShape->SetDimNum(hPostDims);
    for (size_t i = 0; i < hPostDims; ++i) {
        gradHpostShape->SetDim(i, hPostShape->GetDim(i));
    }

    ShowInputShapeInfo(context, gradYShape, xShape, hResShape, hOutShape, hPostShape);
    ShowOutputShapeInfo(context, gradXShape, gradHresShape, gradHoutShape, gradHpostShape);

    OP_LOGD(context, "End to do MhcPostBackwardInfershape.");

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMhcPostBackward(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MhcPostInferDataType.");
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    // Output dtype is same as x
    const ge::DataType xDtype = context->GetInputDataType(INDEX_X);
    const ge::DataType hResDtype = context->GetInputDataType(INDEX_HRES);
    const ge::DataType hOutDtype = context->GetInputDataType(INDEX_HOUT);
    const ge::DataType hPostDtype = context->GetInputDataType(INDEX_HPOST);
    context->SetOutputDataType(INDEX_GRADX, xDtype);
    context->SetOutputDataType(INDEX_GRADHRES, hResDtype);
    context->SetOutputDataType(INDEX_GRADHOUT, hOutDtype);
    context->SetOutputDataType(INDEX_GRADHPOST, hPostDtype);
    OP_LOGD(context->GetNodeName(), "End to do MhcPostInferDataType.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MhcPostBackward)
    .InferShape(InferShapeForMhcPostBackward)
    .InferDataType(InferDataTypeForMhcPostBackward);

}  // namespace ops
