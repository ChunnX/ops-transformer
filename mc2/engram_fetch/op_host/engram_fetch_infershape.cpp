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
 * \file engram_fetch_infershape.cpp
 * \brief EngramFetch shape inference
 */
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static constexpr size_t FETCH_INPUT_HANDLE_INDEX = 0;
static constexpr size_t FETCH_INPUT_INDICES_INDEX = 1;
static constexpr size_t FETCH_OUTPUT_INDEX = 0;

static ge::graphStatus InferShapeEngramFetch(gert::InferShapeContext *context)
{
    // Get indices shape
    const gert::Shape *indicesShape = context->GetInputShape(FETCH_INPUT_INDICES_INDEX);
    if (indicesShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Get output shape
    gert::Shape *outputShape = context->GetOutputShape(FETCH_OUTPUT_INDEX);
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Output shape: [num_tokens, hidden_size]
    // num_tokens from indices shape[0]
    // hidden_size needs to be obtained from context or attributes

    *outputShape = *indicesShape; // Simplified: copy input shape

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeEngramFetch(gert::InferDataTypeContext *context)
{
    // Output dtype: BF16
    context->SetOutputDataType(FETCH_OUTPUT_INDEX, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EngramFetch).InferShape(InferShapeEngramFetch).InferDataType(InferDataTypeEngramFetch);

} // namespace ops