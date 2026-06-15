/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
namespace {
constexpr int64_t X_INDEX = 0;
constexpr int64_t WEIGHT_SCALE_INDEX = 4;
constexpr int64_t M_DIM_INDEX = 0;
constexpr int64_t WEIGHT_SCALE_TRANS_N_DIM_INDEX = 1;
constexpr int64_t WEIGHT_SCALE_N_DIM_INDEX = 2;
constexpr int64_t OUT_Y_INDEX = 0;
constexpr int64_t OUT_Y_SCALE_INDEX = 1;
constexpr int64_t OUT_M_DIM_INDEX = 0;
constexpr int64_t OUT_N_DIM_INDEX = 1;
constexpr int64_t OUT_SCALE_M_DIM_INDEX = 0;
constexpr int64_t OUT_SCALE_N_BLOCK_DIM_INDEX = 1;
constexpr int64_t OUT_SCALE_PAIR_DIM_INDEX = 2;
constexpr int64_t OUT_DIM_NUM = 2;
constexpr int64_t OUT_SCALE_DIM_NUM = 3;
constexpr int64_t MX_GROUP_SIZE = 64;
constexpr int64_t MX_SCALE_PAIR = 2;
constexpr size_t ATTR_INDEX_TRANSPOSE_WEIGHT = 2;
constexpr size_t ATTR_INDEX_QUANT_DTYPE = 5;
constexpr int64_t DYNAMIC_DIM = -1;
} // namespace

static ge::graphStatus InferShape4GroupedMatmulActivationQuant(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *weightScaleShape = context->GetDynamicInputShape(WEIGHT_SCALE_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScaleShape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool *transposeWeightPtr = attrs->GetBool(ATTR_INDEX_TRANSPOSE_WEIGHT);
    bool transposeWeight = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;

    int64_t m = xShape->GetDim(M_DIM_INDEX);
    int64_t n =
        transposeWeight ? weightScaleShape->GetDim(WEIGHT_SCALE_TRANS_N_DIM_INDEX) :
                          weightScaleShape->GetDim(WEIGHT_SCALE_N_DIM_INDEX);

    auto outShape = context->GetOutputShape(OUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    outShape->SetDimNum(OUT_DIM_NUM);
    outShape->SetDim(OUT_M_DIM_INDEX, m);
    outShape->SetDim(OUT_N_DIM_INDEX, n);

    auto outScaleShape = context->GetOutputShape(OUT_Y_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outScaleShape);
    outScaleShape->SetDimNum(OUT_SCALE_DIM_NUM);
    outScaleShape->SetDim(OUT_SCALE_M_DIM_INDEX, m);
    if (n == DYNAMIC_DIM) {
        outScaleShape->SetDim(OUT_SCALE_N_BLOCK_DIM_INDEX, DYNAMIC_DIM);
    } else {
        OP_CHECK_IF(n % MX_GROUP_SIZE != 0,
                    OP_LOGE(context->GetNodeName(), "N must be a multiple of %ld, actual is %ld.", MX_GROUP_SIZE, n),
                    return GRAPH_FAILED);
        outScaleShape->SetDim(OUT_SCALE_N_BLOCK_DIM_INDEX, (n + MX_GROUP_SIZE - 1) / MX_GROUP_SIZE);
    }
    outScaleShape->SetDim(OUT_SCALE_PAIR_DIM_INDEX, MX_SCALE_PAIR);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4GroupedMatmulActivationQuant(gert::InferDataTypeContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *quantDtype = attrs->GetInt(ATTR_INDEX_QUANT_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, quantDtype);
    context->SetOutputDataType(OUT_Y_INDEX, static_cast<ge::DataType>(*quantDtype));
    context->SetOutputDataType(OUT_Y_SCALE_INDEX, ge::DT_FLOAT8_E8M0);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedMatmulActivationQuant)
    .InferShape(InferShape4GroupedMatmulActivationQuant)
    .InferDataType(InferDataType4GroupedMatmulActivationQuant);
} // namespace ops
