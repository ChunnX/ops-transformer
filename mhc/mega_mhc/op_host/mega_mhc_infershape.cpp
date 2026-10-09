/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr int64_t kX = 0;
constexpr int64_t kResidual = 1;
constexpr int64_t kHInPrime = 0;
constexpr int64_t kHPostNew = 1;
constexpr int64_t kHResNew = 2;
constexpr int64_t kX2Out = 3;
constexpr int64_t kHPre = 4;

/** Copy a shape and append the fixed four-head dimension used by MHC. */
void CopyWithHeadAxis(const gert::Shape &source, gert::Shape *destination)
{
    destination->SetDimNum(source.GetDimNum());
    for (int64_t axis = 0; axis < source.GetDimNum() - 1; ++axis) {
        destination->SetDim(axis, source.GetDim(axis));
    }
    destination->SetDim(source.GetDimNum() - 1, 4);
}

/** Copy a shape and append the two four-head axes used by the residual matrix. */
void CopyWithResidualAxes(const gert::Shape &source, gert::Shape *destination)
{
    destination->SetDimNum(source.GetDimNum() + 1);
    for (int64_t axis = 0; axis < source.GetDimNum() - 1; ++axis) {
        destination->SetDim(axis, source.GetDim(axis));
    }
    destination->SetDim(source.GetDimNum() - 1, 4);
    destination->SetDim(source.GetDimNum(), 4);
}

/** Infer the five outputs from h_out/residual leading dimensions; N is fixed to four. */
ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x = context->GetDynamicInputShape(kX, 0);
    const gert::Shape *residual = context->GetDynamicInputShape(kResidual, 0);
    if (x == nullptr || residual == nullptr || (x->GetDimNum() != 2 && x->GetDimNum() != 3) ||
        residual->GetDimNum() != x->GetDimNum() + 1) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape *hInPrime = context->GetOutputShape(kHInPrime);
    gert::Shape *hPostNew = context->GetOutputShape(kHPostNew);
    gert::Shape *hResNew = context->GetOutputShape(kHResNew);
    gert::Shape *x2Out = context->GetOutputShape(kX2Out);
    gert::Shape *hPre = context->GetOutputShape(kHPre);
    if (hInPrime == nullptr || hPostNew == nullptr || hResNew == nullptr || x2Out == nullptr) {
        return ge::GRAPH_FAILED;
    }

    hInPrime->SetDimNum(x->GetDimNum());
    for (int64_t axis = 0; axis < x->GetDimNum(); ++axis) {
        hInPrime->SetDim(axis, x->GetDim(axis));
    }
    CopyWithHeadAxis(*x, hPostNew);
    if (hPre != nullptr) {
        // Reserved output: retain a zero-element launcher slot without producing pre coefficients.
        hPre->SetDimNum(1);
        hPre->SetDim(0, 0);
    }
    CopyWithResidualAxes(*x, hResNew);
    x2Out->SetDimNum(residual->GetDimNum());
    for (int64_t axis = 0; axis < residual->GetDimNum(); ++axis) {
        x2Out->SetDim(axis, residual->GetDim(axis));
    }
    return ge::GRAPH_SUCCESS;
}

/** Preserve low-precision activation types and expose projection coefficients as FP32. */
ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType inputType = context->GetInputDataType(kX);
    context->SetOutputDataType(kHInPrime, inputType);
    context->SetOutputDataType(kHPostNew, ge::DT_FLOAT);
    context->SetOutputDataType(kHResNew, ge::DT_FLOAT);
    context->SetOutputDataType(kX2Out, inputType);
    context->SetOutputDataType(kHPre, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(MegaMhc).InferShape(InferShape).InferDataType(InferDataType);
} // namespace ops
