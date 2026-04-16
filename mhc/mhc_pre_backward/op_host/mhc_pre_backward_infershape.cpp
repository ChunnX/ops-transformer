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

const constexpr int64_t GRAD_H_IN_INDEX = 3;
const constexpr int64_t GRAD_H_POST_INDEX = 4;
const constexpr int64_t GRAD_H_RES_INDEX = 5;

const constexpr int64_t OUT_GRAD_X_INDEX = 0;
const constexpr int64_t OUT_GRAD_PHI_INDEX = 1;
const constexpr int64_t OUT_GRAD_ALPHA_INDEX = 2;
const constexpr int64_t OUT_GRAD_BIAS_INDEX = 3;
const constexpr int64_t OUT_GRAD_GAMMA_INDEX = 4;

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

ge::graphStatus GetInputShapes(InferShapeContext *context, const gert::Shape *&hInGradShape,
                               const gert::Shape *&hPostGradShape, const gert::Shape *&hCombBeforeGradShape)
{
    hInGradShape = context->GetDynamicInputShape(GRAD_H_IN_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, hInGradShape);
    hPostGradShape = context->GetDynamicInputShape(GRAD_H_POST_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, hPostGradShape);
    hCombBeforeGradShape = context->GetDynamicInputShape(GRAD_H_RES_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, hCombBeforeGradShape);
    return GRAPH_SUCCESS;
}

ge::graphStatus ValidateInputDims(int64_t hInGradDimNum, int64_t hPostGradDimNum, int64_t hCombBeforeGradDimNum)
{
    if ((hInGradDimNum != BSD_DIM_NUM && hInGradDimNum != TD_DIM_NUM) ||
        (hPostGradDimNum != BSD_DIM_NUM && hPostGradDimNum != TN_DIM_NUM) ||
        (hCombBeforeGradDimNum != BSNN_DIM_NUM && hCombBeforeGradDimNum != TNN_DIM_NUM)) {
        return GRAPH_FAILED;
    }
    if (hInGradDimNum != hPostGradDimNum) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferBSDFormat(const gert::Shape *hInGradShape, const gert::Shape *hPostGradShape,
                               gert::Shape *xGradShape, uint64_t &numsResidual, uint64_t &dimen)
{
    uint64_t batch = hInGradShape->GetDim(INDEX_B);
    uint64_t sequence = hInGradShape->GetDim(INDEX_S);
    dimen = hInGradShape->GetDim(INDEX_D);
    numsResidual = hPostGradShape->GetDim(INDEX_D);

    xGradShape->SetDimNum(BSNN_DIM_NUM);
    xGradShape->SetDim(0, batch);
    xGradShape->SetDim(1, sequence);
    xGradShape->SetDim(2, numsResidual);
    xGradShape->SetDim(3, dimen);
    return GRAPH_SUCCESS;
}

ge::graphStatus InferTNDFormat(const gert::Shape *hInGradShape, const gert::Shape *hPostGradShape,
                               gert::Shape *xGradShape, uint64_t &numsResidual, uint64_t &dimen)
{
    uint64_t t = hInGradShape->GetDim(INDEX_T);
    dimen = hInGradShape->GetDim(INDEX_D_TND);
    numsResidual = hPostGradShape->GetDim(INDEX_N);

    xGradShape->SetDimNum(TND_DIM_NUM);
    xGradShape->SetDim(0, t);
    xGradShape->SetDim(1, numsResidual);
    xGradShape->SetDim(2, dimen);
    return GRAPH_SUCCESS;
}

ge::graphStatus InferOutputShapes(gert::Shape *xGradShape, gert::Shape *hcWeightGradShape, gert::Shape *alphaGradShape,
                                  gert::Shape *biasPostGradShape, gert::Shape *gammaGradShape, uint64_t numsResidual,
                                  uint64_t dimen)
{
    uint64_t n2Plus2n = (2 * numsResidual) + (numsResidual * numsResidual);

    hcWeightGradShape->SetDimNum(2);
    hcWeightGradShape->SetDim(0, n2Plus2n);
    hcWeightGradShape->SetDim(1, numsResidual * dimen);

    alphaGradShape->SetDimNum(1);
    alphaGradShape->SetDim(0, 3);

    biasPostGradShape->SetDimNum(1);
    biasPostGradShape->SetDim(0, n2Plus2n);

    if (gammaGradShape != nullptr) {
        gammaGradShape->SetDimNum(2);
        gammaGradShape->SetDim(0, numsResidual);
        gammaGradShape->SetDim(1, dimen);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4mHCPreGrad(InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape MhcPreBackward");
    const gert::Shape *hInGradShape = nullptr;
    const gert::Shape *hPostGradShape = nullptr;
    const gert::Shape *hCombBeforeGradShape = nullptr;

    auto ret = GetInputShapes(context, hInGradShape, hPostGradShape, hCombBeforeGradShape);
    if (ret != GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "input shapes are invalid");
        return ret;
    }

    auto xGradShape = context->GetOutputShape(OUT_GRAD_X_INDEX);
    auto hcWeightGradShape = context->GetOutputShape(OUT_GRAD_PHI_INDEX);
    auto alphaGradShape = context->GetOutputShape(OUT_GRAD_ALPHA_INDEX);
    auto biasPostGradShape = context->GetOutputShape(OUT_GRAD_BIAS_INDEX);
    auto gammaGradShape = context->GetOutputShape(OUT_GRAD_GAMMA_INDEX);

    auto hInGradDimNum = hInGradShape->GetDimNum();
    auto hPostGradDimNum = hPostGradShape->GetDimNum();
    auto hCombBeforeGradDimNum = hCombBeforeGradShape->GetDimNum();

    ret = ValidateInputDims(hInGradDimNum, hPostGradDimNum, hCombBeforeGradDimNum);
    if (ret != GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "input dims invalid for MhcPreBackward");
        return ret;
    }

    uint64_t numsResidual = 0;
    uint64_t dimen = 0;

    if (hInGradDimNum == BSD_DIM_NUM) {
        ret = InferBSDFormat(hInGradShape, hPostGradShape, xGradShape, numsResidual, dimen);
    } else if (hInGradDimNum == TD_DIM_NUM) {
        ret = InferTNDFormat(hInGradShape, hPostGradShape, xGradShape, numsResidual, dimen);
    }
    if (ret != GRAPH_SUCCESS) {
        return ret;
    }

    ret = InferOutputShapes(xGradShape, hcWeightGradShape, alphaGradShape, biasPostGradShape, gammaGradShape,
                            numsResidual, dimen);
    if (ret != GRAPH_SUCCESS) {
        return ret;
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape MhcPreBackward");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4mHCPreGrad(gert::InferDataTypeContext *context)
{
    auto xGradType = context->GetInputDataType(GRAD_H_IN_INDEX);
    context->SetOutputDataType(OUT_GRAD_X_INDEX, xGradType);
    context->SetOutputDataType(OUT_GRAD_PHI_INDEX, DataType::DT_FLOAT);
    context->SetOutputDataType(OUT_GRAD_ALPHA_INDEX, DataType::DT_FLOAT);
    context->SetOutputDataType(OUT_GRAD_BIAS_INDEX, DataType::DT_FLOAT);
    context->SetOutputDataType(OUT_GRAD_GAMMA_INDEX, DataType::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MhcPreBackward).InferShape(InferShape4mHCPreGrad).InferDataType(InferDataType4mHCPreGrad);
} // namespace ops
