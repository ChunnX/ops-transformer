/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file rotate_dynamic_quant_infershape.cpp
 * \brief 算子形状推导实现
 */
#include <map>
#include <string>
#include <sstream>
#include <initializer_list>

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "../../../common/include/err/ops_err.h"

using namespace gert;
using namespace ge;

namespace ops {

const constexpr int64_t X_INDEX = 0;
const constexpr int64_t ROT_INDEX = 1;

const constexpr int64_t YOUT_INDEX = 0;
const constexpr int64_t SCALEOUT_INDEX = 1;

static ge::graphStatus InferShapeRotateDynamicQuant(InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape RotateDynamicQuant");

    // 获取输入x的形状
    auto xShape = context->GetInputShape(X_INDEX);
    if (xShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "x shape is null");
        return GRAPH_FAILED;
    }

    // 获取输入rot的形状
    auto rotShape = context->GetInputShape(ROT_INDEX);
    if (rotShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "rot shape is null");
        return GRAPH_FAILED;
    }

    // yOut的形状与输入x一致
    auto yOutShape = context->GetOutputShape(YOUT_INDEX);
    if (yOutShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "yOut shape is null");
        return GRAPH_FAILED;
    }
    *yOutShape = *xShape;

    // scaleOut的形状为[M]，即x的第一维
    auto scaleOutShape = context->GetOutputShape(SCALEOUT_INDEX);
    if (scaleOutShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "scaleOut shape is null");
        return GRAPH_FAILED;
    }
    scaleOutShape->SetDimNum(1);
    scaleOutShape->SetDim(0, xShape->GetDim(0));

    OP_LOGD(context->GetNodeName(), "End to do InferShape RotateDynamicQuant");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataTypeRotateDynamicQuant(gert::InferDataTypeContext *context)
{
    // 获取dst_type属性
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "attrs is null");
        return GRAPH_FAILED;
    }

    int64_t dstType = 0;
    auto dstTypePtr = attrs->GetAttrPointer<int64_t>(0);
    if (dstTypePtr != nullptr) {
        dstType = *dstTypePtr;
    }

    // 根据dst_type设置yOut的数据类型
    // dst_type: 0 -> DT_INT8, 1 -> DT_INT32
    ge::DataType yOutDataType = (dstType == 1) ? ge::DT_INT32 : ge::DT_INT8;
    context->SetOutputDataType(YOUT_INDEX, yOutDataType);

    // scaleOut固定为DT_FLOAT
    context->SetOutputDataType(SCALEOUT_INDEX, ge::DT_FLOAT);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RotateDynamicQuant)
    .InferShape(InferShapeRotateDynamicQuant)
    .InferDataType(InferDataTypeRotateDynamicQuant);
} // namespace ops
