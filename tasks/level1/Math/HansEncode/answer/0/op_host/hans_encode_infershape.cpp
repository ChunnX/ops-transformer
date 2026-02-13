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
 * \file hans_encode_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
// #include "log/log.h"

using namespace ge;

namespace ops {

#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }

static constexpr int IDX_0 = 0;
static constexpr int IDX_1 = 1;
static constexpr int IDX_2 = 2;
static constexpr int IDX_3 = 3;

static ge::graphStatus InferShape4HansEncode(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4HansEncode");
    // get input shapes
    const gert::Shape* inputShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_LOGD(context->GetNodeName(), "End to do InferShape4HansEncode");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HansEncode).InferShape(InferShape4HansEncode);
} // namespace ops
