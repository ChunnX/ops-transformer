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
 * \file mhc_sinkhorn_infershape.cpp
 * \brief mhc_sinkhorn_infershape
 */

#include <vector>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
using namespace ge;
using namespace std;

namespace {
constexpr size_t X_INDEX = 0;
constexpr size_t Y_INDEX = 0;

constexpr size_t INDEX_EPS = 0;
constexpr size_t INDEX_NUM_ITERS = 1;
constexpr size_t INDEX_OUT_FLAG = 2;

constexpr size_t BSNN_DIMS = 4;
constexpr size_t TNN_DIMS = 3;
} // namespace

namespace ops {
static ge::graphStatus InferShape4MhcSinkhorn(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    gert::Shape* y_shape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    auto attr_ptr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr_ptr);
    auto eps_ptr = attr_ptr->GetAttrPointer<gert::ContinuousVector>(INDEX_EPS);
    OP_CHECK_NULL_WITH_CONTEXT(context, eps_ptr);
    auto num_iters_ptr = attr_ptr->GetAttrPointer<gert::ContinuousVector>(INDEX_NUM_ITERS);
    OP_CHECK_NULL_WITH_CONTEXT(context, num_iters_ptr);
    auto out_flag_ptr = attr_ptr->GetAttrPointer<gert::ContinuousVector>(INDEX_OUT_FLAG);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_flag_ptr);
    
    size_t x_dim_num = x_shape->GetDimNum();
    OP_CHECK_IF(
        x_dim_num != BSNN_DIMS && x_dim_num != TNN_DIMS,
        OP_LOGE(context, "The dims of x not equal 4 or 3."), return GRAPH_FAILED);
    size_t y_dim_num = y_shape->GetDimNum();
    OP_CHECK_IF(
        y_dim_num != x_dim_num,
        OP_LOGE(context, "The dims of y not equal x."), return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

static graphStatus InferDtype4MhcSinkhorn(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "MhcSinkhornInferDtype enter");
    const ge::DataType x = context->GetInputDataType(0);
    context->SetOutputDataType(0, x);
    OP_LOGD(context->GetNodeName(), "MhcSinkhornInferDtype end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MhcSinkhorn)
    .InferShape(InferShape4MhcSinkhorn)
    .InferDataType(InferDtype4MhcSinkhorn);
} // namespace ops