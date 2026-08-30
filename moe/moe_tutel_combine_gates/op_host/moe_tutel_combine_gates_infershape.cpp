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
 * \file moe_tutel_combine_gates_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_common/op_host/util/shape_util.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t INPUT_INDICES_INDEX = 2;
constexpr size_t OUTPUT_GATES_GRAD_INDEX = 0;

const char *GetSafeNodeName(gert::InferShapeContext *context)
{
    const char *nodeName = context->GetNodeName();
    return nodeName == nullptr ? "MoeTutelCombineGates" : nodeName;
}
} // namespace

// gates_grad follows the routing tensor shape [K, S]. Shape constraints that
// need all inputs are checked by concrete-shape tiling.
static ge::graphStatus InferShape4MoeTutelCombineGates(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    OP_LOGD(GetSafeNodeName(context), "Begin to do InferShape4MoeTutelCombineGates");

    const gert::Shape *indicesShape = context->GetInputShape(INPUT_INDICES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
    gert::Shape *gatesGradShape = context->GetOutputShape(OUTPUT_GATES_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gatesGradShape);

    if (Ops::Base::IsUnknownRank(*indicesShape)) {
        Ops::Base::SetUnknownRank(*gatesGradShape);
        OP_LOGD(GetSafeNodeName(context), "indices has an unknown rank; gates_grad is set to an unknown rank");
        return GRAPH_SUCCESS;
    }

    *gatesGradShape = *indicesShape;

    OP_LOGD(GetSafeNodeName(context), "Finished InferShape4MoeTutelCombineGates");
    return GRAPH_SUCCESS;
}

// Keep InferDataType on the host registration: the vendor package path links
// this object, while its graph-plugin object is not linked. Host UT and the
// GEIR smoke test cover this packaging-specific registration.
static ge::graphStatus InferDataType4MoeTutelCombineGates(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    return context->SetOutputDataType(OUTPUT_GATES_GRAD_INDEX, context->GetInputDataType(INPUT_X_INDEX));
}

IMPL_OP_INFERSHAPE(MoeTutelCombineGates)
    .InferShape(InferShape4MoeTutelCombineGates)
    .InferDataType(InferDataType4MoeTutelCombineGates);
} // namespace ops
