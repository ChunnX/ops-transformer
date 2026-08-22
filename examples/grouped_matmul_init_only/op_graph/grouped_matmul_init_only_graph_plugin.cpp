/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_init_only_graph_plugin.cpp
 * \brief grouped_matmul_init_only operator graph plugin resource
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
using namespace ge;

static constexpr int64_t IDX_X = 0;
static constexpr int64_t IDX_Y = 0;

static ge::graphStatus InferDataTypeGroupedMatmulInitOnly(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeGroupedMatmulInitOnly");

    // 输出 y 的 dtype 跟随输入 x（fp16）
    ge::DataType xDtype = context->GetInputDataType(IDX_X);
    context->SetOutputDataType(IDX_Y, xDtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeGroupedMatmulInitOnly");
    return GRAPH_SUCCESS;
}

IMPL_OP(GroupedMatmulInitOnly).InferDataType(InferDataTypeGroupedMatmulInitOnly);

}; // namespace ops
