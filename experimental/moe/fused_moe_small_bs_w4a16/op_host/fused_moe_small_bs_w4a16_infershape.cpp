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
 * \file fused_moe_small_bs_w4a16_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr size_t IDX_0 = 0;
static constexpr size_t IDX_1 = 1;

static ge::graphStatus InferShapeFusedMoeSmallBsW4a16(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeFusedMoeSmallBsW4a16");

    // 输入x: [BatchSize, HiddenSize]
    const gert::Shape *xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // 输出y: [BatchSize, HiddenSize]，与x同形状
    gert::Shape *yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    yShape->SetDimNum(IDX_1 + 1);
    yShape->SetDim(IDX_0, xShape->GetDim(IDX_0));
    yShape->SetDim(IDX_1, xShape->GetDim(IDX_1));

    OP_LOGD(context->GetNodeName(), "End to do InferShapeFusedMoeSmallBsW4a16");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedMoeSmallBsW4a16).InferShape(InferShapeFusedMoeSmallBsW4a16);
} // namespace ops
