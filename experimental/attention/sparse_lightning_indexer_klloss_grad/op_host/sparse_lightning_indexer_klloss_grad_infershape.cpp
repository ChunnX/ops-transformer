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
 * \file sparse_lightning_indexer_klloss_grad_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
static const uint64_t INDEX_LAYOUT = 0;
static const uint64_t DIM_NUM_0 = 0;
static const uint64_t SIZE_1 = 1;

enum InputIdx {
    queryEnum = 0,
    keyEnum,
    weightEnum,
    sparseIndicesEnum,
    attnSoftmaxL1Enum,
    cuSeqlensQEnum,
    cuSeqlensKEnum,
    sequsedQEnum,
    sequsedKEnum,
    metadataEnum
};

enum OutputIdx {
    dQueryEnum = 0,
    dKeyEnum,
    dWeightEnum,
    softmaxOutEnum
};

ge::graphStatus InferShapeSparseLightningIndexerKllossGrad(gert::InferShapeContext *context)
{
    OP_LOGI(context, "Start enter ShapeSparseLightningIndexerKllossGrad runtime infershape impl.");
    if (context == nullptr){
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *queryShape = context->GetInputShape(queryEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(keyEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    const gert::Shape *weightsShape = context->GetInputShape(weightEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightsShape);
    const gert::Shape *attnSoftmaxL1Shape = context->GetInputShape(attnSoftmaxL1Enum);
    OP_CHECK_NULL_WITH_CONTEXT(context, attnSoftmaxL1Shape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const char *inputLayout = attrs->GetAttrPointer<char>(INDEX_LAYOUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayout);

    std::string inputLayoutSlig = std::string(inputLayout);
    for (auto &c : inputLayoutSlig) {
        c = toupper(c);
    }
    if (inputLayoutSlig != "BSND" && inputLayoutSlig != "TND") {
        OP_LOGE(context, "The SparseLightningIndexerKllossGrad inputLayout should be BSND/TND, but got %s.",
                  inputLayoutSlig.c_str());
        return GRAPH_FAILED;
    }

    gert::Shape *softmaxOutShape = context->GetOutputShape(softmaxOutEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxOutShape);
    gert::Shape *dWShape = context->GetOutputShape(dWeightEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, dWShape);
    gert::Shape *dQShape = context->GetOutputShape(dQueryEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, dQShape);
    gert::Shape *dKShape = context->GetOutputShape(dKeyEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, dKShape);

    *dWShape = *weightsShape;
    *dQShape = *queryShape;
    *dKShape = *keyShape;
    *softmaxOutShape = *attnSoftmaxL1Shape;

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeSparseLightningIndexerKllossGrad(gert::InferDataTypeContext *context)
{
    OP_LOGI(context, "Start enter ShapeSparseLightningIndexerKllossGrad runtime infershape impl.");
    if (context == nullptr){
        return ge::GRAPH_FAILED;
    } 
    const auto inputDataType = context->GetInputDataType(queryEnum);
    const auto inputWeightDataType = context->GetInputDataType(weightEnum);
    const auto inputAttnSoftmaxL1DataType = context->GetInputDataType(attnSoftmaxL1Enum);
    context->SetOutputDataType(dQueryEnum, inputDataType);
    context->SetOutputDataType(dKeyEnum, inputDataType);
    context->SetOutputDataType(dWeightEnum, inputWeightDataType);
    context->SetOutputDataType(softmaxOutEnum, inputAttnSoftmaxL1DataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseLightningIndexerKllossGrad).InferShape(InferShapeSparseLightningIndexerKllossGrad).InferDataType(InferDataTypeSparseLightningIndexerKllossGrad);
} // namespace ops