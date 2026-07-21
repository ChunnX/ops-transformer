/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/*!
 * \file engram_fetch_train_infershape.cpp
 * \brief EngramFetchTrain 算子 infershape 实现
 *        fetched: [T, hiddenSize] bf16/fp16/fp32
 *        permOut: [T] int32
 *        sendCountsOut: [W] int32（W=rankSize，由调用方在 C++ 侧 at::empty 指定）
 *        recvCountsOut: [W] int32
 *        recvLocalEntryOut: [R_max] int32（R_max=numTokens*rankSize）
 *        numRecvOut: [1] int32
 */

#include "mc2_log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
using namespace ge;
namespace ops {

static constexpr size_t IN_INDICES = 1UL;
static constexpr size_t OUT_FETCHED = 0UL;
static constexpr size_t OUT_PERM = 1UL;
static constexpr size_t OUT_SEND_COUNTS = 2UL;
static constexpr size_t OUT_RECV_COUNTS = 3UL;
static constexpr size_t OUT_RECV_LOCAL_ENTRY = 4UL;
static constexpr size_t OUT_NUM_RECV = 5UL;
static constexpr size_t ATTR_HIDDEN_SIZE = 0UL;

static ge::graphStatus InferShapeEngramFetchTrain(gert::InferShapeContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do InferShapeEngramFetchTrain.");

    auto *attrs = context->GetAttrs();
    auto hiddenSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_HIDDEN_SIZE);
    OPS_CHECK_NULL_WITH_CONTEXT(context, hiddenSizePtr);

    const gert::Shape *indicesShape = context->GetInputShape(IN_INDICES);
    OPS_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
    int64_t numTokens = indicesShape->GetDim(0);

    // fetched: [T, hiddenSize]
    auto *fetchedShape = context->GetOutputShape(OUT_FETCHED);
    fetchedShape->SetDimNum(2);
    fetchedShape->SetDim(0, numTokens);
    fetchedShape->SetDim(1, *hiddenSizePtr);

    // permOut: [T]
    auto *permShape = context->GetOutputShape(OUT_PERM);
    permShape->SetDimNum(1);
    permShape->SetDim(0, numTokens);

    // sendCountsOut / recvCountsOut: [W] — rankSize 由 C++ at::empty 指定，infershape 设 -1
    auto *sendCountsShape = context->GetOutputShape(OUT_SEND_COUNTS);
    sendCountsShape->SetDimNum(1);
    sendCountsShape->SetDim(0, -1);

    auto *recvCountsShape = context->GetOutputShape(OUT_RECV_COUNTS);
    recvCountsShape->SetDimNum(1);
    recvCountsShape->SetDim(0, -1);

    // recvLocalEntryOut: [R_max] — R_max 由 C++ 指定，infershape 设 -1
    auto *recvLocalEntryShape = context->GetOutputShape(OUT_RECV_LOCAL_ENTRY);
    recvLocalEntryShape->SetDimNum(1);
    recvLocalEntryShape->SetDim(0, -1);

    // numRecvOut: [1]
    auto *numRecvShape = context->GetOutputShape(OUT_NUM_RECV);
    numRecvShape->SetDimNum(1);
    numRecvShape->SetDim(0, 1);

    OPS_LOG_D(context->GetNodeName(), "End to do InferShapeEngramFetchTrain.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeEngramFetchTrain(gert::InferDataTypeContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do InferDataTypeEngramFetchTrain.");
    // fetched: 与 indices device 一致但 dtype 跟随 fetched（bf16/fp16/fp32），由框架推断
    context->SetOutputDataType(OUT_FETCHED, ge::DT_BF16);
    // permOut / sendCountsOut / recvCountsOut / recvLocalEntryOut / numRecvOut: int32
    context->SetOutputDataType(OUT_PERM, ge::DT_INT32);
    context->SetOutputDataType(OUT_SEND_COUNTS, ge::DT_INT32);
    context->SetOutputDataType(OUT_RECV_COUNTS, ge::DT_INT32);
    context->SetOutputDataType(OUT_RECV_LOCAL_ENTRY, ge::DT_INT32);
    context->SetOutputDataType(OUT_NUM_RECV, ge::DT_INT32);
    OPS_LOG_D(context->GetNodeName(), "End to do InferDataTypeEngramFetchTrain.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EngramFetchTrain)
    .InferShape(InferShapeEngramFetchTrain)
    .InferDataType(InferDataTypeEngramFetchTrain);
} // namespace ops
