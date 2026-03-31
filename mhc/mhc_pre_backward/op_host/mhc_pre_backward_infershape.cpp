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
 * \file mhc_pre_backward_infershape.cpp
 * \brief
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
#include "err/ops_err.h"

using namespace gert;
using namespace ge;

namespace ops {

const constexpr int64_t H_IN_GRAD_INDEX = 3;
const constexpr int64_t H_POST_GRAD_INDEX = 4;
const constexpr int64_t H_COMB_BEFORE_GRAD_INDEX = 5;

const constexpr int64_t OUT_X_GRAD_INDEX = 0;
const constexpr int64_t OUT_HC_WEIGHT_GRAD_INDEX = 1;
const constexpr int64_t OUT_ALPHA_GRAD_INDEX = 2;
const constexpr int64_t OUT_BIAS_POST_GRAD_INDEX = 3;
const constexpr int64_t OUT_GAMMA_GRAD_INDEX = 4;

const constexpr int64_t BSD_DIM_NUM = 3;
const constexpr int64_t BSNN_DIM_NUM = 4;
const constexpr int64_t TD_DIM_NUM = 2;
const constexpr int64_t TN_DIM_NUM = 2;
const constexpr int64_t TNN_DIM_NUM = 3;
const constexpr int64_t TND_DIM_NUM = 3;

const constexpr int64_t INDEX_B = 0;
const constexpr int64_t INDEX_S = 1;
const constexpr int64_t INDEX_D = 2;

const constexpr int64_t INDEX_T = 0;
const constexpr int64_t INDEX_N = 1;
const constexpr int64_t INDEX_D_TND = 1;

static ge::graphStatus InferShape4mHCPreGrad(InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape MhcPreBackward");
    const gert::Shape *hInGradShape = context->GetDynamicInputShape(H_IN_GRAD_INDEX, 0);
    const gert::Shape *hPostGradShape = context->GetDynamicInputShape(H_POST_GRAD_INDEX, 0);
    const gert::Shape *hCombBeforeGradShape = context->GetDynamicInputShape(H_COMB_BEFORE_GRAD_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, hInGradShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, hPostGradShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, hCombBeforeGradShape);

    auto xGradShape = context->GetOutputShape(OUT_X_GRAD_INDEX);
    auto hcWeightGradShape = context->GetOutputShape(OUT_HC_WEIGHT_GRAD_INDEX);
    auto alphaGradShape = context->GetOutputShape(OUT_ALPHA_GRAD_INDEX);
    auto biasPostGradShape = context->GetOutputShape(OUT_BIAS_POST_GRAD_INDEX);
    auto gammaGradShape = context->GetOutputShape(OUT_GAMMA_GRAD_INDEX);

    auto hInGradDimNum = hInGradShape->GetDimNum();
    auto hPostGradDimNum = hPostGradShape->GetDimNum();
    auto hCombBeforeGradDimNum = hCombBeforeGradShape->GetDimNum();

    if ((hInGradDimNum != BSD_DIM_NUM && hInGradDimNum != TD_DIM_NUM) ||
        (hPostGradDimNum != BSD_DIM_NUM && hPostGradDimNum != TN_DIM_NUM) ||
        (hCombBeforeGradDimNum != BSNN_DIM_NUM && hCombBeforeGradDimNum != TNN_DIM_NUM)) {
        OP_LOGE(context->GetNodeName(), "input dims invalid for MhcPreBackward");
        return GRAPH_FAILED;
    }

    // 检查输入维度是否一致
    if (hInGradDimNum != hPostGradDimNum) {
        OP_LOGE(context->GetNodeName(), "h_in_grad and h_post_grad must have the same dim num");
        return GRAPH_FAILED;
    }

    uint64_t numsResidual = 0;
    uint64_t dimen = 0;

    if (hInGradDimNum == BSD_DIM_NUM) {
        // BSND格式: h_in_grad [B,S,D], h_post_grad [B,S,N], h_comb_before_grad [B,S,N,N]
        uint64_t batch = hInGradShape->GetDim(INDEX_B);
        uint64_t sequence = hInGradShape->GetDim(INDEX_S);
        dimen = hInGradShape->GetDim(INDEX_D);
        numsResidual = hPostGradShape->GetDim(INDEX_D);

        xGradShape->SetDimNum(BSNN_DIM_NUM);
        xGradShape->SetDim(0, batch);
        xGradShape->SetDim(1, sequence);
        xGradShape->SetDim(2, numsResidual);
        xGradShape->SetDim(3, dimen);
    } else if (hInGradDimNum == TD_DIM_NUM) {
        // TND格式: h_in_grad [T,D], h_post_grad [T,N], h_comb_before_grad [T,N,N]
        uint64_t t = hInGradShape->GetDim(INDEX_T);
        dimen = hInGradShape->GetDim(INDEX_D_TND);
        numsResidual = hPostGradShape->GetDim(INDEX_N);

        xGradShape->SetDimNum(TND_DIM_NUM);
        xGradShape->SetDim(0, t);
        xGradShape->SetDim(1, numsResidual);
        xGradShape->SetDim(2, dimen);
    }

    hcWeightGradShape->SetDimNum(2);
    hcWeightGradShape->SetDim(0, (2 * numsResidual) + (numsResidual * numsResidual));
    hcWeightGradShape->SetDim(1, numsResidual * dimen);

    alphaGradShape->SetDimNum(1);
    alphaGradShape->SetDim(0, 3);

    biasPostGradShape->SetDimNum(1);
    biasPostGradShape->SetDim(0, (2 * numsResidual) + (numsResidual * numsResidual));

    gammaGradShape->SetDimNum(2);
    gammaGradShape->SetDim(0, numsResidual);
    gammaGradShape->SetDim(1, dimen);

    OP_LOGD(context->GetNodeName(), "End to do InferShape MhcPreBackward");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4mHCPreGrad(gert::InferDataTypeContext *context)
{
    auto xGradType = context->GetInputDataType(H_IN_GRAD_INDEX);
    context->SetOutputDataType(OUT_X_GRAD_INDEX, xGradType);
    context->SetOutputDataType(OUT_HC_WEIGHT_GRAD_INDEX, DataType::DT_FLOAT);
    context->SetOutputDataType(OUT_ALPHA_GRAD_INDEX, DataType::DT_FLOAT);
    context->SetOutputDataType(OUT_BIAS_POST_GRAD_INDEX, DataType::DT_FLOAT);
    context->SetOutputDataType(OUT_GAMMA_GRAD_INDEX, DataType::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MhcPreBackward)
    .InferShape(InferShape4mHCPreGrad)
    .InferDataType(InferDataType4mHCPreGrad);
} // namespace ops
