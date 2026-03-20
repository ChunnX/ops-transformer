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

// 实现形状推导函数
static ge::graphStatus InferShapeRotateDynamicQuant(InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape RotateDynamicQuant");

    // 获取输入 x 的形状 [M, N]
    auto xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // yOut 形状与输入 x 一致 [M, N]
    auto yOutShape = context->GetOutputShape(YOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yOutShape);
    *yOutShape = *xShape;

    // scaleOut 形状为 [M]，取 x 的第一维
    auto scaleOutShape = context->GetOutputShape(SCALEOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleOutShape);
    int64_t dimNum = xShape->GetDimNum();
    scaleOutShape->SetDimNum(1);
    if (dimNum > 0) {
        scaleOutShape->SetDim(0, xShape->GetDim(0));
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape RotateDynamicQuant");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataTypeRotateDynamicQuant(gert::InferDataTypeContext *context)
{
    // 获取属性 dst_type 来决定 yOut 的数据类型
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* dstTypePtr = attrs->GetAttrPointer<int64_t>(0);
    int64_t dstType = 0;  // 默认值 0 -> DT_INT8
    if (dstTypePtr != nullptr) {
        dstType = *dstTypePtr;
    }

    // 根据 dst_type 设置 yOut 数据类型
    // 0 -> DT_INT8, 1 -> DT_INT32
    DataType yOutDataType = (dstType == 1) ? DataType::DT_INT32 : DataType::DT_INT8;
    context->SetOutputDataType(YOUT_INDEX, yOutDataType);

    // scaleOut 数据类型固定为 DT_FLOAT
    context->SetOutputDataType(SCALEOUT_INDEX, DataType::DT_FLOAT);

    return GRAPH_SUCCESS;
}

// 注册形状推导实现
IMPL_OP_INFERSHAPE(RotateDynamicQuant)
    .InferShape(InferShapeRotateDynamicQuant)
    .InferDataType(InferDataTypeRotateDynamicQuant);

} // namespace ops
