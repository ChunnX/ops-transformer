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
 * \file moe_tutel_combine_x_infershape.cpp
 * \brief InferShape / InferDataType of the MoeTutelCombineX operator (graph mode).
 *
 * Shape contract: x_grad.shape = gates.shape + (y_grad.shape[-1],), with
 * rank(y_grad) = rank(gates) + 1, rank(y_grad) in {2, 3} and gates/indices/
 * locations sharing the same shape. All tensors must be FORMAT_ND.
 */
#include "register/op_impl_registry.h"              // IMPL_OP_INFERSHAPE macro
#include "exe_graph/runtime/infer_shape_context.h"  // InferShapeContext / InferDataTypeContext
#include "op_common/log/log.h"                      // OP_CHECK_NULL_WITH_CONTEXT / OP_LOGE macro

using namespace ge;

namespace ops {

// ---------------------------------------------------------------------------
// InferShape4MoeTutelCombineX — output shape = gates.shape + (H,)
// ---------------------------------------------------------------------------
static ge::graphStatus InferShape4MoeTutelCombineX(gert::InferShapeContext *context)
{
    const gert::Shape *ygShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, ygShape);
    const gert::Shape *gatesShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gatesShape);
    const gert::Shape *indicesShape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
    const gert::Shape *locationsShape = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, locationsShape);
    gert::Shape *outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    // Only FORMAT_ND is supported.
    for (size_t i = 0; i < 4; ++i) {
        const gert::CompileTimeTensorDesc *desc = context->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, desc);
        if (desc->GetFormat().GetOriginFormat() != ge::FORMAT_ND ||
            desc->GetFormat().GetStorageFormat() != ge::FORMAT_ND) {
            OP_LOGE(context->GetNodeName(), "input %zu format must be ND, got origin=%d storage=%d",
                    i, static_cast<int>(desc->GetFormat().GetOriginFormat()),
                    static_cast<int>(desc->GetFormat().GetStorageFormat()));
            return ge::GRAPH_FAILED;
        }
    }

    // A dim is either >= 0 (0 allowed for empty tensors) or -1 (unknown).
    auto ValidateDims = [&](const gert::Shape &s, const char *name) -> bool {
        for (size_t d = 0; d < s.GetDimNum(); ++d) {
            const int64_t v = s.GetDim(d);
            if (v < -1) {
                OP_LOGE(context->GetNodeName(), "%s dim %zu has invalid value %ld", name, d, v);
                return false;
            }
        }
        return true;
    };
    if (!ValidateDims(*ygShape, "y_grad") || !ValidateDims(*gatesShape, "gates") ||
        !ValidateDims(*indicesShape, "indices") || !ValidateDims(*locationsShape, "locations")) {
        return ge::GRAPH_FAILED;
    }

    const size_t ygRank = ygShape->GetDimNum();
    const size_t gatesRank = gatesShape->GetDimNum();
    if (ygRank != 2 && ygRank != 3) {
        OP_LOGE(context->GetNodeName(), "y_grad rank must be 2 or 3, got %zu", ygRank);
        return ge::GRAPH_FAILED;
    }
    if (gatesRank != 1 && gatesRank != 2) {
        OP_LOGE(context->GetNodeName(), "gates rank must be 1 or 2, got %zu", gatesRank);
        return ge::GRAPH_FAILED;
    }
    if (ygRank != gatesRank + 1) {
        OP_LOGE(context->GetNodeName(), "rank(y_grad) must equal rank(gates)+1, got y_grad=%zu gates=%zu",
                ygRank, gatesRank);
        return ge::GRAPH_FAILED;
    }

    // gates / indices / locations share the same shape (unknown dims tolerated).
    auto DimsMatch = [](const gert::Shape &a, const gert::Shape &b) -> bool {
        if (a.GetDimNum() != b.GetDimNum()) {
            return false;
        }
        for (size_t d = 0; d < a.GetDimNum(); ++d) {
            const int64_t av = a.GetDim(d);
            const int64_t bv = b.GetDim(d);
            if (av >= 0 && bv >= 0 && av != bv) {
                return false;
            }
        }
        return true;
    };
    if (!DimsMatch(*gatesShape, *indicesShape) || !DimsMatch(*gatesShape, *locationsShape)) {
        OP_LOGE(context->GetNodeName(), "gates/indices/locations shapes must match");
        return ge::GRAPH_FAILED;
    }

    *outputShape = *gatesShape;
    outputShape->AppendDim(ygShape->GetDim(ygRank - 1));
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// InferDataType4MoeTutelCombineX — output x_grad shares the dtype of y_grad,
// y_grad/gates must be the same float dtype and indices/locations must be INT32.
// ---------------------------------------------------------------------------
static ge::graphStatus InferDataType4MoeTutelCombineX(gert::InferDataTypeContext *context)
{
    const ge::DataType ygDtype = context->GetInputDataType(0);
    const ge::DataType gatesDtype = context->GetInputDataType(1);
    const ge::DataType indicesDtype = context->GetInputDataType(2);
    const ge::DataType locationsDtype = context->GetInputDataType(3);

    if (ygDtype != ge::DT_FLOAT16 && ygDtype != ge::DT_FLOAT && ygDtype != ge::DT_BF16) {
        OP_LOGE(context->GetNodeName(), "y_grad dtype must be FP16/FP32/BF16, got %d",
                static_cast<int>(ygDtype));
        return ge::GRAPH_FAILED;
    }
    if (gatesDtype != ygDtype) {
        OP_LOGE(context->GetNodeName(), "gates dtype must match y_grad dtype, got y_grad=%d gates=%d",
                static_cast<int>(ygDtype), static_cast<int>(gatesDtype));
        return ge::GRAPH_FAILED;
    }
    if (indicesDtype != ge::DT_INT32 || locationsDtype != ge::DT_INT32) {
        OP_LOGE(context->GetNodeName(), "indices/locations dtype must be INT32, got indices=%d locations=%d",
                static_cast<int>(indicesDtype), static_cast<int>(locationsDtype));
        return ge::GRAPH_FAILED;
    }

    return context->SetOutputDataType(0, ygDtype);
}

IMPL_OP_INFERSHAPE(MoeTutelCombineX)
    .InferShape(InferShape4MoeTutelCombineX)
    .InferDataType(InferDataType4MoeTutelCombineX);
} // namespace ops
