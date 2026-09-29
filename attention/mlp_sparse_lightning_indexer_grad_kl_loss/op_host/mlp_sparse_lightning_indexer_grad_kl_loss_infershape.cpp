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
#include "slig_host_checks.h"

namespace ops {
enum InputIndex : size_t {
    QUERY_INDEX_INPUT = 2,
    KEY_INDEX_INPUT = 3,
    WEIGHT_INPUT = 4
};

enum OutputIndex : size_t {
    D_QUERY_INDEX_OUTPUT = 0,
    D_KEY_INDEX_OUTPUT = 1,
    D_WEIGHT_OUTPUT = 2,
    LOSS_OUTPUT = 3
};

ge::graphStatus InferShapeMlpSparseLightningIndexerGradKLLoss(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MlpSparseLightningIndexerGradKLLoss", "context is null"),
                return ge::GRAPH_FAILED);
    const gert::Shape *queryIndexShape = context->GetInputShape(QUERY_INDEX_INPUT);
    const gert::Shape *keyIndexShape = context->GetInputShape(KEY_INDEX_INPUT);
    const gert::Shape *weightShape = context->GetInputShape(WEIGHT_INPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryIndexShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyIndexShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightShape);

    gert::Shape *dQueryIndexShape = context->GetOutputShape(D_QUERY_INDEX_OUTPUT);
    gert::Shape *dKeyIndexShape = context->GetOutputShape(D_KEY_INDEX_OUTPUT);
    gert::Shape *dWeightShape = context->GetOutputShape(D_WEIGHT_OUTPUT);
    gert::Shape *lossShape = context->GetOutputShape(LOSS_OUTPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, dQueryIndexShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, dKeyIndexShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, dWeightShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, lossShape);

    *dQueryIndexShape = *queryIndexShape;
    *dKeyIndexShape = *keyIndexShape;
    *dWeightShape = *weightShape;
    lossShape->SetDimNum(1);
    lossShape->SetDim(0, 1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeMlpSparseLightningIndexerGradKLLoss(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MlpSparseLightningIndexerGradKLLoss", "context is null"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(D_QUERY_INDEX_OUTPUT, context->GetInputDataType(QUERY_INDEX_INPUT));
    context->SetOutputDataType(D_KEY_INDEX_OUTPUT, context->GetInputDataType(KEY_INDEX_INPUT));
    context->SetOutputDataType(D_WEIGHT_OUTPUT, ge::DT_FLOAT);
    context->SetOutputDataType(LOSS_OUTPUT, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MlpSparseLightningIndexerGradKLLoss)
    .InferShape(InferShapeMlpSparseLightningIndexerGradKLLoss)
    .InferDataType(InferDataTypeMlpSparseLightningIndexerGradKLLoss);
} // namespace ops
