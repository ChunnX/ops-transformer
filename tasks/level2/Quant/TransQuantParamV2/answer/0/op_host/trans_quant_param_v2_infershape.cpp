/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file trans_quant_param_v2_infershape.cpp
 * \brief Bench：去除对 Ops::Base / util/shape_util 的依赖，保留与 infer 语义一致的分支。
 */
#include "common/log.h"
#include "register/op_impl_registry.h"

static constexpr int INPUT_SCALE_INDEX = 0;
static constexpr int INPUT_OFFSET_INDEX = 1;
static constexpr int OUTPUT_Y_INDEX = 0;

using namespace ge;
namespace ops {

static bool IsUnknownLike(const gert::Shape& s)
{
    for (size_t i = 0; i < s.GetDimNum(); ++i) {
        const int64_t d = s.GetDim(i);
        if (d < 0) {
            return true;
        }
    }
    return false;
}

static ge::graphStatus InferShape4TransQuantParamV2(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("TransQuantParamV2", "InferShapeContext is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4TransQuantParamV2.");

    const gert::Shape* scaleShape = context->GetInputShape(INPUT_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);

    const gert::Shape* offsetShape = context->GetOptionalInputShape(INPUT_OFFSET_INDEX);

    gert::Shape* outYShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outYShape);

    *outYShape = *scaleShape;

    if (IsUnknownLike(*scaleShape) || (offsetShape != nullptr && IsUnknownLike(*offsetShape))) {
        OP_LOGI(context->GetNodeName(), "End to do InferShape4TransQuantParamV2, input has unknown dims.");
        return GRAPH_SUCCESS;
    }

    if (offsetShape != nullptr && (offsetShape->GetDimNum() > scaleShape->GetDimNum())) {
        *outYShape = *offsetShape;
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape4TransQuantParamV2.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4TransQuantParamV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4TransQuantParamV2");
    context->SetOutputDataType(OUTPUT_Y_INDEX, ge::DT_UINT64);
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4TransQuantParamV2");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TransQuantParamV2)
    .InferShape(InferShape4TransQuantParamV2)
    .InferDataType(InferDataType4TransQuantParamV2);
} // namespace ops
