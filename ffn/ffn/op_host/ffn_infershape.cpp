/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ffn_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferShapeFFN(gert::InferShapeContext *context)
{
    // 布局识别需要激活类型（swiglu 的隐藏宽是 w1 边长的一半）
    bool isSwiglu = false;
    const auto *attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const char *act = attrs->GetAttrPointer<char>(0);
        if (act != nullptr) {
            isSwiglu = (act[0] == 's' || act[0] == 'S') && (act[1] == 'w' || act[1] == 'W') &&
                       (act[2] == 'i' || act[2] == 'I') && (act[3] == 'g' || act[3] == 'G') &&
                       (act[4] == 'l' || act[4] == 'L') && (act[5] == 'u' || act[5] == 'U') && act[6] == '\0';
        }
    }
    auto in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    *out_shape = *in_shape;
    // 输出最后一维 = weight2 的 N：
    //  - canonical [K, N]：N 在最后一维（arch22 原语义）
    //  - linear [N, K]（out, in）：N 在倒数第二维（arch35 与 PyTorch Linear 对齐）
    // 按 x 的 K 与 weight2 的维度关系自动识别，避免 N != K 的 shape 被误判为非法。
    auto weight2_shape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, weight2_shape);
    auto weight1_shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, weight1_shape);
    size_t xDimNum = in_shape->GetDimNum();
    size_t w1DimNum = weight1_shape->GetDimNum();
    size_t w2DimNum = weight2_shape->GetDimNum();
    if (xDimNum > 0 && w1DimNum >= 2 && w2DimNum >= 2) {
        int64_t xK = in_shape->GetDim(xDimNum - 1);
        int64_t w1d0 = weight1_shape->GetDim(w1DimNum - 2);
        int64_t w1d1 = weight1_shape->GetDim(w1DimNum - 1);
        int64_t w2d0 = weight2_shape->GetDim(w2DimNum - 2);
        int64_t w2d1 = weight2_shape->GetDim(w2DimNum - 1);
        // 布局自动识别（arch35 fused 同时支持 linear/canonical）：
        //  1) w1 非方阵：w1d1==xK → linear；w1d0==xK → canonical；
        //  2) w1 方阵（w1d0==w1d1==xK，形状不可区分）：用 w2 的布局一致性消歧——
        //     swiglu 的隐藏宽是 w1 边长的一半，其余激活隐藏宽=w1 边长；
        //     w2 第二维==隐藏宽 → linear；w2 第一维==隐藏宽 → canonical；
        //  3) w1/w2 都方阵（H==K==N，全等）或均不匹配：真歧义，按 dtype 约定
        //     （BF16=arch35 linear 优先，其余=canonical）兜底。
        const int64_t hiddenW = isSwiglu ? w1d0 / 2 : w1d0;
        const int64_t hiddenW2 = isSwiglu ? w1d1 / 2 : w1d1;
        bool isLinear = false;
        if (w1d1 == xK && w1d0 != xK) {
            isLinear = true;
        } else if (w1d0 == xK && w1d1 != xK) {
            isLinear = false;
        } else if (w1d0 == xK && w1d1 == xK) {
            if (w2d1 == hiddenW && w2d0 != hiddenW2) {
                isLinear = true;
            } else if (w2d0 == hiddenW2 && w2d1 != hiddenW) {
                isLinear = false;
            } else {
                const auto *xDesc = context->GetInputDesc(0);
                isLinear = (xDesc != nullptr && xDesc->GetDataType() == ge::DT_BF16);
            }
        }
        int64_t n = isLinear ? w2d0 : w2d1;
        out_shape->SetDim(xDimNum - 1, n);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFFN(gert::InferDataTypeContext *context)
{
    auto input_x_dtype = context->GetInputDataType(0);
    if (input_x_dtype == ge::DT_INT8) {
        auto attrs = context->GetAttrs();
        const int64_t *output_dtype = attrs->GetInt(2);
        if (output_dtype != nullptr && *output_dtype == 1) {
            context->SetOutputDataType(0, ge::DT_BF16);
        } else {
            context->SetOutputDataType(0, ge::DT_FLOAT16);
        }
    } else {
        context->SetOutputDataType(0, input_x_dtype);
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FFN).InferShape(InferShapeFFN).InferDataType(InferDataTypeFFN);
} // namespace ops
