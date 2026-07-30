/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "../utils/inc/error/ops_error.h"

using namespace ge;

namespace ops {
enum InputIdx {
    queryEnum = 0,
    keyEnum,
    queryIndexEnum,
    keyIndexEnum,
    weightEnum,
    sparseIndexEnum,
    maxEnum,
    sumEnum,
    queryRopeEnum,
    keyRopeEnum,
    actualSeqLengthsQueryEnum,
    actualSeqLengthsKvEnum
};

enum OutputIdx {
    dQueryIndexEnum = 0,
    dKeyIndexEnum,
    dWeightEnum,
    lossEnum
};

ge::graphStatus InferShapeSparseLightningIndexerGradKLLoss(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("SparseLightningIndexerGradKLLoss", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(queryEnum);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED)
    const gert::Shape *keyShape = context->GetInputShape(keyEnum);
    OPS_LOG_E_IF_NULL(context, keyShape, return ge::GRAPH_FAILED)
    const gert::Shape *weightShape = context->GetInputShape(weightEnum);
    OPS_LOG_E_IF_NULL(context, weightShape, return ge::GRAPH_FAILED)
    const gert::Shape *queryIndexShape = context->GetInputShape(queryIndexEnum);
    OPS_LOG_E_IF_NULL(context, queryIndexShape, return ge::GRAPH_FAILED)
    const gert::Shape *keyIndexShape = context->GetInputShape(keyIndexEnum);
    OPS_LOG_E_IF_NULL(context, keyIndexShape, return ge::GRAPH_FAILED)

    gert::Shape *lossShape = context->GetOutputShape(lossEnum);
    OPS_LOG_E_IF_NULL(context, lossShape, return ge::GRAPH_FAILED)
    gert::Shape *dWeightShape = context->GetOutputShape(dWeightEnum);
    OPS_LOG_E_IF_NULL(context, dWeightShape, return ge::GRAPH_FAILED)
    gert::Shape *dQIndexShape = context->GetOutputShape(dQueryIndexEnum);
    OPS_LOG_E_IF_NULL(context, dQIndexShape, return ge::GRAPH_FAILED)
    gert::Shape *dKIndexShape = context->GetOutputShape(dKeyIndexEnum);
    OPS_LOG_E_IF_NULL(context, dKIndexShape, return ge::GRAPH_FAILED)

    *dWeightShape = *weightShape;
    *dQIndexShape = *queryIndexShape;
    *dKIndexShape = *keyIndexShape;
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeSparseLightningIndexerGradKLLoss(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("SparseLightningIndexerGradKLLoss", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(queryEnum);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(SparseLightningIndexerGradKLLoss).InferShape(InferShapeSparseLightningIndexerGradKLLoss).InferDataType(InferDataTypeSparseLightningIndexerGradKLLoss);
} // namespace ops
  
