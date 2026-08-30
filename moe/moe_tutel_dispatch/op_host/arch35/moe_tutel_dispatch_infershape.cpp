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

#include "graph/operator_reg.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ge {
// The runtime-v2 InferShapeContext owns a fresh output-shape slot and therefore
// cannot observe a shape prefilled through Operator::UpdateOutputDesc.  This
// compatibility operator has no E input/attribute, so use the Operator-based GE
// callback to retain the caller descriptor verbatim instead of fabricating E=K.
IMPLEMT_COMMON_INFERFUNC(InferShape4MoeTutelDispatchV1)
{
    const TensorDesc outputDesc = op.GetOutputDesc("y");
    return op.UpdateOutputDesc("y", outputDesc);
}

COMMON_INFER_FUNC_REG(MoeTutelDispatch, InferShape4MoeTutelDispatchV1);
} // namespace ge

namespace ops {
namespace {
constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t OUTPUT_Y_INDEX = 0;
} // namespace

static ge::graphStatus InferDataType4MoeTutelDispatch(gert::InferDataTypeContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_LOGD(context, "Begin InferDataType4MoeTutelDispatch");
    const ge::graphStatus ret = context->SetOutputDataType(OUTPUT_Y_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    OP_LOGD(context, "Finished InferDataType4MoeTutelDispatch");
    return ret;
}

// Dtype inference stays in the runtime-v2 registry.  Shape inference uses the
// Operator-based compatibility registration above so the caller-owned E dim is
// not discarded before graph compilation.
IMPL_OP_INFERSHAPE(MoeTutelDispatch).InferDataType(InferDataType4MoeTutelDispatch);
} // namespace ops
