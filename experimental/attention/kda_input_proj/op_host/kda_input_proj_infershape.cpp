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
 * \file kda_input_proj_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

#include "err/ops_err.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeKdaInputProj(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("KdaInputProj", "InferShapeContext is nullptr!"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeKdaInputProj(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("KdaInputProj", "InferDataTypeContext is nullptr!"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KdaInputProj).InferShape(InferShapeKdaInputProj).InferDataType(InferDataTypeKdaInputProj);
} // namespace ops
