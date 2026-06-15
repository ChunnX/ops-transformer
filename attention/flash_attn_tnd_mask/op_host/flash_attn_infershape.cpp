/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {

static constexpr size_t ATTR_IDX_SOFTMAX_SCALE      = 0;
static constexpr size_t ATTR_IDX_MASK_MODE          = 1;
static constexpr size_t ATTR_IDX_WIN_LEFT           = 2;
static constexpr size_t ATTR_IDX_WIN_RIGHT          = 3;
static constexpr size_t ATTR_IDX_MAX_SEQLEN_Q       = 4;
static constexpr size_t ATTR_IDX_MAX_SEQLEN_KV      = 5;
static constexpr size_t ATTR_IDX_LAYOUT_Q           = 6;
static constexpr size_t ATTR_IDX_LAYOUT_KV          = 7;
static constexpr size_t ATTR_IDX_LAYOUT_OUT         = 8;
static constexpr size_t ATTR_IDX_RETURN_SOFTMAX_LSE = 9;

static constexpr size_t INPUT_IDX_Q = 0;
static constexpr size_t INPUT_IDX_K = 1;
static constexpr size_t INPUT_IDX_V = 2;
static constexpr size_t OUTPUT_IDX_ATTN_OUT = 0;
static constexpr size_t OUTPUT_IDX_SOFTMAX_LSE = 1;

ge::graphStatus InferShapeFlashAttnTndMask(gert::InferShapeContext *context)
{
    OP_LOGI(context, "FlashAttnTndMask InferShape start.");
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *qShape = context->GetInputShape(INPUT_IDX_Q);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    const gert::Shape *vShape = context->GetInputShape(INPUT_IDX_V);
    OP_CHECK_NULL_WITH_CONTEXT(context, vShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto returnSoftmaxLsePtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_RETURN_SOFTMAX_LSE);
    OP_CHECK_NULL_WITH_CONTEXT(context, returnSoftmaxLsePtr);
    int64_t returnSoftmaxLse = *returnSoftmaxLsePtr;

    if (qShape->GetDimNum() < 3) {
        OP_LOGE(context, "TND layout requires 3-dim, got %zu.", qShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }
    int64_t seqLenQ   = qShape->GetDim(0);
    int64_t numHeadsQ = qShape->GetDim(1);
    int64_t headDim   = qShape->GetDim(2);

    int64_t headDimV = headDim;
    if (vShape->GetDimNum() >= 3) {
        headDimV = vShape->GetDim(2);
    }

    gert::Shape *attnOutShape = context->GetOutputShape(OUTPUT_IDX_ATTN_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, attnOutShape);
    attnOutShape->SetDimNum(3);
    attnOutShape->SetDim(0, seqLenQ);
    attnOutShape->SetDim(1, numHeadsQ);
    attnOutShape->SetDim(2, headDimV);

    gert::Shape *lseShape = context->GetOutputShape(OUTPUT_IDX_SOFTMAX_LSE);
    if (lseShape != nullptr) {
        if (returnSoftmaxLse != 0) {
            lseShape->SetDimNum(2);
            lseShape->SetDim(0, seqLenQ);
            lseShape->SetDim(1, numHeadsQ);
        } else {
            lseShape->SetDimNum(1);
            lseShape->SetDim(0, 0);
        }
    }

    OP_LOGI(context, "FlashAttnTndMask InferShape done. attnOut dims=%zu.", attnOutShape->GetDimNum());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeFlashAttnTndMask(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto qDtype = context->GetInputDataType(INPUT_IDX_Q);
    context->SetOutputDataType(OUTPUT_IDX_ATTN_OUT, qDtype);
    context->SetOutputDataType(OUTPUT_IDX_SOFTMAX_LSE, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FlashAttnTndMask)
    .InferShape(InferShapeFlashAttnTndMask)
    .InferDataType(InferDataTypeFlashAttnTndMask);

}  // namespace ops
