/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file roi_extractor_infershape.cpp
 * \brief y = [R, C, pooled_height, pooled_width] — pooled_height and
 *        pooled_width are used independently (canndev dynamic-impl
 *        pool_w=pooled_h typo is NOT carried over; SPEC decision 2).
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr uint32_t FEATURES_0_INDEX = 0;
constexpr uint32_t ROIS_INDEX = 8;
constexpr uint32_t ATTR_POOLED_HEIGHT_INDEX = 3;
constexpr uint32_t ATTR_POOLED_WIDTH_INDEX = 4;
constexpr uint32_t Y_INDEX = 0;

static ge::graphStatus InferShapeRoiExtractor(gert::InferShapeContext *context)
{
    const gert::Shape *featuresShape = context->GetInputShape(FEATURES_0_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, featuresShape);
    const gert::Shape *roisShape = context->GetInputShape(ROIS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, roisShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto pooledHPtr = attrs->GetAttrPointer<int64_t>(ATTR_POOLED_HEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, pooledHPtr);
    auto pooledWPtr = attrs->GetAttrPointer<int64_t>(ATTR_POOLED_WIDTH_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, pooledWPtr);

    gert::Shape *yShape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    yShape->SetDimNum(4);
    yShape->SetDim(0, roisShape->GetDim(0));     // R
    yShape->SetDim(1, featuresShape->GetDim(1)); // C
    yShape->SetDim(2, *pooledHPtr);              // pooled_height (independent)
    yShape->SetDim(3, *pooledWPtr);              // pooled_width  (independent)
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeRoiExtractor(gert::InferDataTypeContext *context)
{
    // y dtype = features dtype (fp16/fp32 coupled with rois/y per SPEC §5)
    context->SetOutputDataType(Y_INDEX, context->GetInputDataType(FEATURES_0_INDEX));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RoiExtractor).InferShape(InferShapeRoiExtractor).InferDataType(InferDataTypeRoiExtractor);
} // namespace ops
