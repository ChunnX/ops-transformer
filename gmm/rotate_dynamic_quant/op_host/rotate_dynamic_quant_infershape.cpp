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
 * \brief RotateDynamicQuant算子形状推导实现
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

    // 获取输入x的形状 [M, N]
    auto xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape, return GRAPH_FAILED);

    auto xStorageShape = context->GetInputStorageShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape, return GRAPH_FAILED);

    // 获取维度信息
    int64_t dimNum = xShape->GetDimNum();
    int64_t M = (dimNum >= 2) ? xShape->GetDim(0) : 1;
    int64_t N = (dimNum >= 2) ? xShape->GetDim(1) : xShape->GetDim(0);

    // 设置输出yOut的形状 [M, N]，与输入x一致
    auto yOutShape = context->GetOutputShape(YOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yOutShape, return GRAPH_FAILED);
    yOutShape->SetDimNum(dimNum);
    for (int64_t i = 0; i < dimNum; i++) {
        yOutShape->SetDim(i, xShape->GetDim(i));
    }

    // 设置输出yOut的StorageShape
    auto yOutStorageShape = context->GetOutputStorageShape(YOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yOutStorageShape, return GRAPH_FAILED);
    yOutStorageShape->SetDimNum(dimNum);
    for (int64_t i = 0; i < dimNum; i++) {
        yOutStorageShape->SetDim(i, xStorageShape->GetDim(i));
    }

    // 设置输出scaleOut的形状 [M]
    auto scaleOutShape = context->GetOutputShape(SCALEOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleOutShape, return GRAPH_FAILED);
    scaleOutShape->SetDimNum(1);
    scaleOutShape->SetDim(0, M);

    // 设置输出scaleOut的StorageShape
    auto scaleOutStorageShape = context->GetOutputStorageShape(SCALEOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleOutStorageShape, return GRAPH_FAILED);
    scaleOutStorageShape->SetDimNum(1);
    scaleOutStorageShape->SetDim(0, M);

    OP_LOGD(context->GetNodeName(), "End to do InferShape RotateDynamicQuant");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataTypeRotateDynamicQuant(gert::InferDataTypeContext *context)
{
    // 获取dst_type属性
    auto attrs = context->GetAttrs();
    int32_t dstType = 0;  // 默认为INT8
    if (attrs != nullptr) {
        auto dstTypePtr = attrs->GetAttrPointer<int32_t>(0);
        if (dstTypePtr != nullptr) {
            dstType = *dstTypePtr;
        }
    }

    // 根据dst_type设置yOut的数据类型
    // dst_type: 0 -> DT_INT8, 1 -> DT_INT32
    DataType yOutDtype = (dstType == 1) ? DataType::DT_INT32 : DataType::DT_INT8;
    context->SetOutputDataType(YOUT_INDEX, yOutDtype);

    // scaleOut固定为DT_FLOAT
    context->SetOutputDataType(SCALEOUT_INDEX, DataType::DT_FLOAT);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RotateDynamicQuant)
    .InferShape(InferShapeRotateDynamicQuant)
    .InferDataType(InferDataTypeRotateDynamicQuant);
} // namespace ops
