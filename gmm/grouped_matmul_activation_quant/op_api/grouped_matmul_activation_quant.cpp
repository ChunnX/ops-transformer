/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul_activation_quant.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "util/math_util.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GroupedMatmulGeluTanhQuant);

namespace {
constexpr int64_t X_M_DIM_INDEX = 0L;
constexpr int64_t WEIGHT_SCALE_TRANS_N_DIM_INDEX = 1L;
constexpr int64_t WEIGHT_SCALE_N_DIM_INDEX = 2L;
constexpr int64_t MX_GROUP_SIZE = 64L;
constexpr int64_t MX_SCALE_PAIR = 2L;
constexpr size_t FIRST_TENSOR_INDEX = 0UL;
} // namespace

const std::tuple<aclTensor *, aclTensor *> GroupedMatmulActivationQuant(const aclTensor *x,
    const aclTensor *xScale, const aclTensor *groupList, const aclTensorList *weight,
    const aclTensorList *weightScale, int64_t dequantDtype, int64_t quantDtype,
    const char *roundMode, int64_t scaleAlg, float dstTypeMax, bool transposeWeight,
    int64_t groupListType, const aclIntArray *tuningConfigOptional, aclOpExecutor *executor)
{
    L0_DFX(GroupedMatmulActivationQuant, x, xScale, groupList, weight, weightScale, dequantDtype, quantDtype,
           roundMode, scaleAlg, dstTypeMax, transposeWeight, groupListType, tuningConfigOptional);
    if (x == nullptr || weightScale == nullptr || weightScale->Size() == 0 || (*weightScale)[FIRST_TENSOR_INDEX] == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x and weightScale must not be nullptr.");
        return std::tuple(nullptr, nullptr);
    }

    int64_t m = x->GetViewShape().GetDim(X_M_DIM_INDEX);
    const auto &weightScaleShape = (*weightScale)[FIRST_TENSOR_INDEX]->GetViewShape();
    int64_t n = transposeWeight ? weightScaleShape.GetDim(WEIGHT_SCALE_TRANS_N_DIM_INDEX) :
                                  weightScaleShape.GetDim(WEIGHT_SCALE_N_DIM_INDEX);
    gert::Shape outShape({m, n});
    gert::Shape scaleOutShape({m, static_cast<int64_t>(Ops::Base::CeilDiv(n, MX_GROUP_SIZE)), MX_SCALE_PAIR});
    auto out = executor->AllocTensor(outShape, static_cast<ge::DataType>(quantDtype), ge::FORMAT_ND);
    auto scaleOut = executor->AllocTensor(scaleOutShape, DataType::DT_FLOAT8_E8M0, ge::FORMAT_ND);

    auto ret = INFER_SHAPE(GroupedMatmulGeluTanhQuant,
        OP_INPUT(x, xScale, groupList, weight, weightScale), OP_OUTPUT(out, scaleOut),
        OP_ATTR(dequantDtype, quantDtype, roundMode, scaleAlg, dstTypeMax, transposeWeight, groupListType,
                tuningConfigOptional));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return std::tuple(nullptr, nullptr);
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(GroupedMatmulGeluTanhQuant,
        OP_INPUT(x, xScale, groupList, weight, weightScale), OP_OUTPUT(out, scaleOut),
        OP_ATTR(dequantDtype, quantDtype, roundMode, scaleAlg, dstTypeMax, transposeWeight, groupListType,
                tuningConfigOptional));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple(nullptr, nullptr);
    }

    return std::tie(out, scaleOut);
}
} // namespace l0op
