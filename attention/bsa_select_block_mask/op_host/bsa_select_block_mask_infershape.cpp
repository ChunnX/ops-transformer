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
 * \file bsa_select_block_mask_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "err/ops_err.h"

using namespace ge;

namespace ops {
enum InputIdx {
    queryEnum = 0,
    keyEnum,
    blockShapeEnum,
    postBlockShapeEnum,
    actualSeqLensQEnum,
    actualSeqLensKvEnum,
    actualBlockLenQEnum,
    actualBlockLenKvEnum
};

enum OutputIdx {
    maskOutEnum = 0
};

ge::graphStatus InferShapeBSASelectBlockMask(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("BSASelectBlockMask", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeBSASelectBlockMask");

    const gert::Shape *queryShape = context->GetInputShape(queryEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(keyEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);

    gert::Shape *maskShape = context->GetOutputShape(maskOutEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskShape);

    auto attrs = context->GetAttrs();
    size_t idx = 0;
    auto qLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    auto kvLayoutPtr = attrs->GetAttrPointer<char>(idx++);

    int64_t sqLen = 0;
    int64_t skvLen = 0;
    int64_t batchSize = 0;
    int64_t numHeads = 0;

    size_t qDimNum = queryShape->GetDimNum();
    if (qLayoutPtr != nullptr && qLayoutPtr[0] == 'B' && qDimNum == 4) {
        batchSize = queryShape->GetDim(0);
        numHeads = queryShape->GetDim(1);
        sqLen = queryShape->GetDim(2);
        skvLen = keyShape->GetDim(2);
    } else if (qDimNum == 3) {
        numHeads = queryShape->GetDim(1);
        sqLen = queryShape->GetDim(0);
        skvLen = keyShape->GetDim(0);
        batchSize = 1;
    }

    maskShape->SetDimNum(4);
    maskShape->SetDim(0, batchSize);
    maskShape->SetDim(1, numHeads);
    maskShape->SetDim(2, sqLen);
    maskShape->SetDim(3, skvLen);

    OP_LOGD(context->GetNodeName(), "End to do InferShapeBSASelectBlockMask");
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeBSASelectBlockMask(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("BSASelectBlockMask", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeBSASelectBlockMask");
    context->SetOutputDataType(maskOutEnum, DT_INT8);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeBSASelectBlockMask");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(BSASelectBlockMask).InferShape(InferShapeBSASelectBlockMask).InferDataType(InferDataTypeBSASelectBlockMask);
} // namespace ops
