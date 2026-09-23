/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file mhc_pre_split_post_res_infershape.cpp
 * \brief MhcPreSplitPostRes 的 InferShape/InferDataType：h_post/h_res 形状由 x（[B,S,N,D]/[T,N,D]）
 *        与 phi 行数（N^2+N 完整形态 / N 仅 post 形态）推导，输出 dtype 恒 FP32
 */
#include <string>
#include <sstream>

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "err/ops_err.h"

using namespace gert;
using namespace ge;

namespace ops {

// Input tensor indices.
const constexpr int64_t X_INDEX = 0;
const constexpr int64_t PHI_INDEX = 1;
const constexpr int64_t ALPHA_INDEX = 2;
const constexpr int64_t BIAS_INDEX = 3;

// Output tensor indices.
const constexpr int64_t OUT_H_POST_INDEX = 0;
const constexpr int64_t OUT_H_RES_INDEX = 1;

// Supported input ranks.
const constexpr int64_t BSND_DIM_NUM = 4;
const constexpr int64_t TND_DIM_NUM = 3;
const constexpr int64_t PHI_DIM_NUM = 2;
const constexpr int64_t ALPHA_DIM_NUM = 1;
const constexpr int64_t BIAS_DIM_NUM = 1;
const constexpr int64_t ALPHA_SIZE = 2;

static void SetShape1D(gert::Shape *shape, int64_t d0)
{
    shape->SetDimNum(1);
    shape->SetDim(0, d0);
}

static void SetShape2D(gert::Shape *shape, int64_t d0, int64_t d1)
{
    shape->SetDimNum(2);
    shape->SetDim(0, d0);
    shape->SetDim(1, d1);
}

static void SetShape3D(gert::Shape *shape, int64_t d0, int64_t d1, int64_t d2)
{
    shape->SetDimNum(3);
    shape->SetDim(0, d0);
    shape->SetDim(1, d1);
    shape->SetDim(2, d2);
}

static void SetShape4D(gert::Shape *shape, int64_t d0, int64_t d1, int64_t d2, int64_t d3)
{
    shape->SetDimNum(4);
    shape->SetDim(0, d0);
    shape->SetDim(1, d1);
    shape->SetDim(2, d2);
    shape->SetDim(3, d3);
}

static ge::graphStatus InferShape4MhcPreSplitPostRes(InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape MhcPreSplitPostRes");
    const gert::Shape *xShape = context->GetInputShape(X_INDEX);
    const gert::Shape *phiShape = context->GetInputShape(PHI_INDEX);
    const gert::Shape *alphaShape = context->GetInputShape(ALPHA_INDEX);
    const gert::Shape *biasShape = context->GetInputShape(BIAS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, phiShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, alphaShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, biasShape);

    const int64_t xDim = xShape->GetDimNum();
    const int64_t phiDim = phiShape->GetDimNum();
    OP_CHECK_IF(xDim != BSND_DIM_NUM && xDim != TND_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(xDim).c_str(), "3 or 4"),
                return GRAPH_FAILED);
    OP_CHECK_IF(phiDim != PHI_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "phi", std::to_string(phiDim).c_str(), "2"),
                return GRAPH_FAILED);
    OP_CHECK_IF(alphaShape->GetDimNum() != ALPHA_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "alpha",
                                             std::to_string(alphaShape->GetDimNum()).c_str(), "1"),
                return GRAPH_FAILED);
    OP_CHECK_IF(alphaShape->GetDim(0) != ALPHA_SIZE,
                OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "alpha",
                                          Ops::Base::ToString(*alphaShape).c_str(), "[2]"),
                return GRAPH_FAILED);
    OP_CHECK_IF(biasShape->GetDimNum() != BIAS_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "bias",
                                             std::to_string(biasShape->GetDimNum()).c_str(), "1"),
                return GRAPH_FAILED);

    const int64_t n = (xDim == BSND_DIM_NUM) ? xShape->GetDim(2) : xShape->GetDim(1);
    const int64_t phiDim0 = phiShape->GetDim(0);
    // phi 双形态：N^2+N 行（完整形态，输出 h_res=[.,N,N]）或 N 行（仅 post 形态，h_res 为 0 元素）
    const bool fullForm = (phiDim0 == n * n + n);
    OP_CHECK_IF(!fullForm && phiDim0 != n,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "phi",
                                                      Ops::Base::ToString(*phiShape).c_str(),
                                                      "phi dim0 must be n^2+n or n"),
                return GRAPH_FAILED);
    OP_CHECK_IF(biasShape->GetDim(0) != phiDim0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "bias",
                                                      Ops::Base::ToString(*biasShape).c_str(),
                                                      "bias dim0 must equal phi dim0"),
                return GRAPH_FAILED);

    gert::Shape *hPostShape = context->GetOutputShape(OUT_H_POST_INDEX);
    gert::Shape *hResShape = context->GetOutputShape(OUT_H_RES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, hPostShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, hResShape);

    if (xDim == BSND_DIM_NUM) {
        const int64_t b = xShape->GetDim(0);
        const int64_t s = xShape->GetDim(1);
        SetShape3D(hPostShape, b, s, n);
        if (fullForm) {
            SetShape4D(hResShape, b, s, n, n);
        } else {
            SetShape1D(hResShape, 0);
        }
    } else {
        const int64_t t = xShape->GetDim(0);
        SetShape2D(hPostShape, t, n);
        if (fullForm) {
            SetShape3D(hResShape, t, n, n);
        } else {
            SetShape1D(hResShape, 0);
        }
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape MhcPreSplitPostRes");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MhcPreSplitPostRes(gert::InferDataTypeContext *context)
{
    // h_post/h_res 恒 FP32（kernel 全 FP32 计算后写出，与 x dtype 无关）
    context->SetOutputDataType(OUT_H_POST_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(OUT_H_RES_INDEX, ge::DT_FLOAT);
    return ge::graphStatus::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MhcPreSplitPostRes).InferShape(InferShape4MhcPreSplitPostRes).InferDataType(
    InferDataType4MhcPreSplitPostRes);
}  // namespace ops
