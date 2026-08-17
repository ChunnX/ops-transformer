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
 * \file moe_tutel_dispatch_graph_plugin.cpp
 * \brief moe_tutel_dispatch operator graph infer resource
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
namespace {
constexpr size_t MOE_TUTEL_DISPATCH_IN_X_IDX = 0;
constexpr size_t MOE_TUTEL_DISPATCH_OUT_Y_IDX = 0;
} // namespace

// y 与 x 同 dtype（same-as-input）
static ge::graphStatus InferDataType4MoeTutelDispatch(gert::InferDataTypeContext *context)
{
    OP_LOGD(context, "Begin to do InferDataType4MoeTutelDispatch");
    auto ret = context->SetOutputDataType(MOE_TUTEL_DISPATCH_OUT_Y_IDX,
                                          context->GetInputDataType(MOE_TUTEL_DISPATCH_IN_X_IDX));
    OP_LOGD(context, "End to do InferDataType4MoeTutelDispatch");
    return ret;
}

IMPL_OP_INFERSHAPE(MoeTutelDispatch).InferDataType(InferDataType4MoeTutelDispatch);
} // namespace ops
