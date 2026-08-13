/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <limits>
#include "mc2_log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t X_INDEX = 0;
constexpr size_t WEIGHT_INDEX = 1;
constexpr size_t Y_INDEX = 0;
constexpr size_t EP_ATTR_INDEX = 0;
constexpr size_t SEND_ATTR_INDEX = 1;
constexpr size_t RECV_ATTR_INDEX = 2;
constexpr size_t X_DIMS = 2;
constexpr size_t WEIGHT_DIMS = 3;
constexpr int64_t MAX_WORLD_SIZE = 8;
constexpr int64_t MAX_LOCAL_EXPERTS = 64;

ge::graphStatus InferShapeAlltoAllvGroupedMatMulShmem(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(X_INDEX);
    const gert::Shape *weightShape = context->GetInputShape(WEIGHT_INDEX);
    gert::Shape *yShape = context->GetOutputShape(Y_INDEX);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, weightShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t *epWorldSize = attrs->GetAttrPointer<int64_t>(EP_ATTR_INDEX);
    const gert::ContinuousVector *sendCounts = attrs->GetAttrPointer<gert::ContinuousVector>(SEND_ATTR_INDEX);
    const gert::ContinuousVector *recvCounts = attrs->GetAttrPointer<gert::ContinuousVector>(RECV_ATTR_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epWorldSize);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sendCounts);
    OPS_CHECK_NULL_WITH_CONTEXT(context, recvCounts);

    OPS_ERR_IF(xShape->GetDimNum() != X_DIMS || weightShape->GetDimNum() != WEIGHT_DIMS,
               VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "gmm_x/gmm_weight must be 2D/3D."),
               return ge::GRAPH_FAILED);
    const int64_t m = xShape->GetDim(0);
    const int64_t k = xShape->GetDim(1);
    const int64_t experts = weightShape->GetDim(0);
    const int64_t weightK = weightShape->GetDim(1);
    const int64_t n = weightShape->GetDim(2);
    OPS_ERR_IF(*epWorldSize < 1 || *epWorldSize > MAX_WORLD_SIZE || experts < 1 || experts > MAX_LOCAL_EXPERTS ||
                   k <= 0 || n <= 0 || k != weightK,
               VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Invalid EP/E/K/N contract."),
               return ge::GRAPH_FAILED);

    const size_t expected = static_cast<size_t>(*epWorldSize * experts);
    OPS_ERR_IF(sendCounts->GetSize() != expected || recvCounts->GetSize() != expected,
               VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
                   context->GetNodeName(), "send_counts/recv_counts length must equal ep_world_size * experts."),
               return ge::GRAPH_FAILED);
    const int64_t *sendData = static_cast<const int64_t *>(sendCounts->GetData());
    const int64_t *recvData = static_cast<const int64_t *>(recvCounts->GetData());
    OPS_CHECK_NULL_WITH_CONTEXT(context, sendData);
    OPS_CHECK_NULL_WITH_CONTEXT(context, recvData);
    uint64_t sendTotal = 0;
    uint64_t recvTotal = 0;
    for (int64_t peer = 0; peer < *epWorldSize; ++peer) {
        uint64_t peerSendTotal = 0;
        uint64_t peerRecvTotal = 0;
        for (int64_t expert = 0; expert < experts; ++expert) {
            const size_t i = static_cast<size_t>(peer * experts + expert);
            OPS_ERR_IF(sendData[i] < 0 || recvData[i] < 0 ||
                           sendData[i] > std::numeric_limits<uint32_t>::max() ||
                           recvData[i] > std::numeric_limits<uint32_t>::max(),
                       VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
                           context->GetNodeName(), "Counts must be non-negative uint32 values."),
                       return ge::GRAPH_FAILED);
            peerSendTotal += static_cast<uint64_t>(sendData[i]);
            peerRecvTotal += static_cast<uint64_t>(recvData[i]);
        }
        OPS_ERR_IF(peerSendTotal > std::numeric_limits<uint32_t>::max() ||
                       peerRecvTotal > std::numeric_limits<uint32_t>::max(),
                   VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
                       context->GetNodeName(), "Each peer route total must fit in uint32."),
                   return ge::GRAPH_FAILED);
        sendTotal += peerSendTotal;
        recvTotal += peerRecvTotal;
    }
    OPS_ERR_IF(m >= 0 && sendTotal != static_cast<uint64_t>(m),
               VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "sum(send_counts) must equal gmm_x M."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(recvTotal > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
               VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "sum(recv_counts) exceeds int64."),
               return ge::GRAPH_FAILED);
    yShape->SetDimNum(X_DIMS);
    yShape->SetDim(0, static_cast<int64_t>(recvTotal));
    yShape->SetDim(1, n);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeAlltoAllvGroupedMatMulShmem(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(Y_INDEX, context->GetInputDataType(X_INDEX));
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(AlltoAllvGroupedMatMulShmem)
    .InferShape(InferShapeAlltoAllvGroupedMatMulShmem)
    .InferDataType(InferDataTypeAlltoAllvGroupedMatMulShmem);
} // namespace ops
