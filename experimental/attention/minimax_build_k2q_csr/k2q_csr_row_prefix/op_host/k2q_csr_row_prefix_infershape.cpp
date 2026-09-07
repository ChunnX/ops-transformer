/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * K2qCsrRowPrefix infer shape.
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferShapeK2qCsrRowPrefix(gert::InferShapeContext *context)
{
    gert::Shape *rowPtrShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, rowPtrShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *totalRowsPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *headsPtr = attrs->GetAttrPointer<int64_t>(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, totalRowsPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, headsPtr);
    int64_t totalRows = *totalRowsPtr;
    int64_t H = *headsPtr;
    OP_CHECK_IF(totalRows < 0, OP_LOGE(context->GetNodeName(), "total_rows must be >= 0"), return GRAPH_FAILED);

    rowPtrShape->SetDimNum(2);
    rowPtrShape->SetDim(0, H);
    rowPtrShape->SetDim(1, totalRows + 1);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(K2qCsrRowPrefix).InferShape(InferShapeK2qCsrRowPrefix);
} // namespace ops
