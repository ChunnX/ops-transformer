/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the LICENSE file in the root of the software repository for the full text of the License.
 */

/*!
 * \file roi_extractor_infershape.cpp
 * \brief y = [K, C, pooled_height, pooled_width]; K from rois, C from feats0.
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr uint32_t ROIS_INDEX = 5;
constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t ATTR_POOLED_HEIGHT_INDEX = 3;
constexpr uint32_t ATTR_POOLED_WIDTH_INDEX = 4;
constexpr int64_t POOLED_LENGTH_DEFAULT = 7;

static ge::graphStatus InferShapeRoiExtractor(gert::InferShapeContext *context)
{
    const gert::Shape *roisShape = context->GetInputShape(ROIS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, roisShape);

    const gert::Shape *feats0Shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, feats0Shape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *pooledHeightPtr = attrs->GetAttrPointer<int64_t>(ATTR_POOLED_HEIGHT_INDEX);
    const int64_t *pooledWidthPtr = attrs->GetAttrPointer<int64_t>(ATTR_POOLED_WIDTH_INDEX);
    const int64_t pooledHeight = pooledHeightPtr == nullptr ? POOLED_LENGTH_DEFAULT : *pooledHeightPtr;
    const int64_t pooledWidth = pooledWidthPtr == nullptr ? POOLED_LENGTH_DEFAULT : *pooledWidthPtr;

    gert::Shape *yShape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(4);
    yShape->SetDim(0, roisShape->GetDim(0));
    yShape->SetDim(1, feats0Shape->GetDim(1));
    yShape->SetDim(2, pooledHeight);
    yShape->SetDim(3, pooledWidth);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeRoiExtractor(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(Y_INDEX, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RoiExtractor)
    .InferShape(InferShapeRoiExtractor)
    .InferDataType(InferDataTypeRoiExtractor);
} // namespace ops
