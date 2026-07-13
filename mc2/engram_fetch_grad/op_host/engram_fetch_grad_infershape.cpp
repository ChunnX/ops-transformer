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
 * \file engram_fetch_grad_infershape.cpp
 * \brief EngramFetchGrad 算子 infershape 实现
 *        gradUniqueOut: [-1, hiddenSize] fp32（动态 shape，dim0=numUnique 运行时决定）
 *        uniqueLocalEntryOut: [-1] int32（1D sparse index，与 gradUniqueOut dim0 一致）
 *        numUniqueOut: [1] int32（固定标量）
 */

#include "mc2_log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
using namespace ge;
namespace ops {

static constexpr size_t OUT_GRAD_UNIQUE = 0UL;
static constexpr size_t OUT_UNIQUE_LOCAL_ENTRY = 1UL;
static constexpr size_t OUT_NUM_UNIQUE = 2UL;
static constexpr size_t ATTR_HIDDEN_SIZE = 0UL;
static constexpr int64_t FIRST_ELE_SIZE = -1;

static ge::graphStatus InferShapeEngramFetchGrad(gert::InferShapeContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do InferShapeEngramFetchGrad.");

    auto *attrs = context->GetAttrs();
    auto hiddenSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_HIDDEN_SIZE);
    OPS_CHECK_NULL_WITH_CONTEXT(context, hiddenSizePtr);

    // gradUniqueOut = [-1, hiddenSize] fp32（前 numUnique 有效）
    auto *gradUniqueShape = context->GetOutputShape(OUT_GRAD_UNIQUE);
    gradUniqueShape->SetDimNum(2);
    gradUniqueShape->SetDim(0, FIRST_ELE_SIZE);
    gradUniqueShape->SetDim(1, *hiddenSizePtr);

    // uniqueLocalEntryOut = [-1] int32
    auto *uniqueLocalEntryShape = context->GetOutputShape(OUT_UNIQUE_LOCAL_ENTRY);
    uniqueLocalEntryShape->SetDimNum(1);
    uniqueLocalEntryShape->SetDim(0, FIRST_ELE_SIZE);

    // numUniqueOut = [1] int32
    auto *numUniqueShape = context->GetOutputShape(OUT_NUM_UNIQUE);
    numUniqueShape->SetDimNum(1);
    numUniqueShape->SetDim(0, 1);

    OPS_LOG_D(context->GetNodeName(), "End to do InferShapeEngramFetchGrad.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeEngramFetchGrad(gert::InferDataTypeContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do InferDataTypeEngramFetchGrad.");
    // gradUnique: fp32（反向 FP32 聚合）
    context->SetOutputDataType(OUT_GRAD_UNIQUE, ge::DT_FLOAT);
    // uniqueLocalEntry: int32
    context->SetOutputDataType(OUT_UNIQUE_LOCAL_ENTRY, ge::DT_INT32);
    // numUnique: int32
    context->SetOutputDataType(OUT_NUM_UNIQUE, ge::DT_INT32);
    OPS_LOG_D(context->GetNodeName(), "End to do InferDataTypeEngramFetchGrad.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EngramFetchGrad)
    .InferShape(InferShapeEngramFetchGrad)
    .InferDataType(InferDataTypeEngramFetchGrad);
} // namespace ops
