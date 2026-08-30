/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_tutel_dispatch_infershape.cpp
 * \brief Ascend 950 graph inference. Output shape is supplied by the caller.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t OUTPUT_Y_INDEX = 0;
} // namespace

static ge::graphStatus InferShape4MoeTutelDispatch(gert::InferShapeContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_LOGD(context, "Begin InferShape4MoeTutelDispatch");
    // E is absent from the historical input ABI. GetOutputShape points to the
    // caller-provided y descriptor, so preserving it is the only valid shape
    // inference behavior; deriving E from gates would incorrectly force E=K.
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputShape(OUTPUT_Y_INDEX));
    OP_LOGD(context, "Finished InferShape4MoeTutelDispatch without changing caller output shape");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeTutelDispatch(gert::InferDataTypeContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_LOGD(context, "Begin InferDataType4MoeTutelDispatch");
    const ge::graphStatus ret = context->SetOutputDataType(OUTPUT_Y_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    OP_LOGD(context, "Finished InferDataType4MoeTutelDispatch");
    return ret;
}

// Shape and dtype inference stay in op_host rather than op_graph because the
// custom vendor package links this object into libcust_opsproto_rt2.0.so;
// graph-plugin objects are not part of that library.
IMPL_OP_INFERSHAPE(MoeTutelDispatch)
    .InferShape(InferShape4MoeTutelDispatch)
    .InferDataType(InferDataType4MoeTutelDispatch);
} // namespace ops
