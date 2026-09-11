/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.cann.huawei.com/CANN_OPEN_SOURCE_LICENSE_ACK
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

/*!
 * \file pdist_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t Y_INDEX = 0;

static ge::graphStatus InferShapePdist(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape *yShape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    int64_t nSize = xShape->GetDim(0);
    yShape->SetDimNum(1);
    yShape->SetDim(0, nSize * (nSize - 1) / 2);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePdist(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(Y_INDEX, context->GetInputDataType(X_INDEX));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Pdist)
    .InferShape(InferShapePdist)
    .InferDataType(InferDataTypePdist);
} // namespace ops
