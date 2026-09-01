/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static bool CheckShapeBlockAttentionResidualsGrad(gert::InferShapeContext *context, int64_t &B, int64_t &N, int64_t &H)
{
    const gert::Shape *psShape = context->GetInputShape(0);
    const gert::Shape *brShape = context->GetInputShape(1);
    const gert::Shape *pwShape = context->GetInputShape(2);
    const gert::Shape *nwShape = context->GetInputShape(3);
    const gert::Shape *gradHiddenStateShape = context->GetInputShape(4);
    const gert::Shape *irShape = context->GetInputShape(5);
    const gert::Shape *pbShape = context->GetInputShape(6);

    if (psShape->GetDimNum() != 2 || brShape->GetDimNum() != 3 || pwShape->GetDimNum() != 2 ||
        nwShape->GetDimNum() != 1 || gradHiddenStateShape->GetDimNum() != 2 || irShape->GetDimNum() != 2 ||
        pbShape->GetDimNum() != 2) {
        return false;
    }

    B = psShape->GetDim(0);
    H = psShape->GetDim(1);
    N = brShape->GetDim(1);

    if (brShape->GetDim(0) != B || gradHiddenStateShape->GetDim(0) != B || irShape->GetDim(0) != B ||
        pbShape->GetDim(0) != B) {
        return false;
    }
    if (brShape->GetDim(2) != H || pwShape->GetDim(0) != 1 || pwShape->GetDim(1) != H || nwShape->GetDim(0) != H ||
        gradHiddenStateShape->GetDim(1) != H) {
        return false;
    }
    if (irShape->GetDim(1) != N + 1 || pbShape->GetDim(1) != N + 1) {
        return false;
    }

    return true;
}

static ge::graphStatus InferShapeBlockAttentionResidualsGrad(gert::InferShapeContext *context)
{
    int64_t B = 0, N = 0, H = 0;
    if (!CheckShapeBlockAttentionResidualsGrad(context, B, N, H)) {
        return GRAPH_FAILED;
    }

    // grad_partial_block [B, H] = partial_block
    context->GetOutputShape(0)->SetDimNum(2);
    context->GetOutputShape(0)->SetDim(0, B);
    context->GetOutputShape(0)->SetDim(1, H);

    // grad_block_res [B, N, H] = block_res
    context->GetOutputShape(1)->SetDimNum(3);
    context->GetOutputShape(1)->SetDim(0, B);
    context->GetOutputShape(1)->SetDim(1, N);
    context->GetOutputShape(1)->SetDim(2, H);

    // grad_proj_weight [H] = proj_weight
    context->GetOutputShape(2)->SetDimNum(2);
    context->GetOutputShape(2)->SetDim(0, 1);
    context->GetOutputShape(2)->SetDim(1, H);

    // grad_norm_weight [H] = norm_weight
    context->GetOutputShape(3)->SetDimNum(1);
    context->GetOutputShape(3)->SetDim(0, H);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeBlockAttentionResidualsGrad(gert::InferDataTypeContext *context)
{
    const auto invNormDtype = context->GetInputDataType(5);
    const auto probsDtype = context->GetInputDataType(6);
    if (invNormDtype != ge::DT_FLOAT || probsDtype != ge::DT_FLOAT) {
        OP_LOGE(context->GetNodeName(), "inv_norm and probs only support fp32, got %d and %d",
                static_cast<int>(invNormDtype), static_cast<int>(probsDtype));
        return GRAPH_FAILED;
    }

    auto dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
    context->SetOutputDataType(1, dtype);
    context->SetOutputDataType(2, dtype);
    context->SetOutputDataType(3, dtype);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BlockAttentionResidualsGrad)
    .InferShape(InferShapeBlockAttentionResidualsGrad)
    .InferDataType(InferDataTypeBlockAttentionResidualsGrad);
} // namespace ops
