/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/op_log.h"
#include "log/log.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "util/math_util.h"
#include "grouped_matmul_swiglu_quant_v3.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GroupedMatmulSwigluQuantV3);

constexpr int64_t SWIGLU_SPLIT_SIZE = 64L;
constexpr size_t MX_WEIGHT_SCALE_N_DIM = 2UL;

const std::tuple<aclTensor *, aclTensor *> GroupedMatmulSwigluQuantV3(
    const aclTensor *x, const aclTensorList *weight, const aclTensorList *weightScale, const aclTensor *xScale,
    const aclTensorList *weightAssistanceMatrix, const aclTensor *bias, const aclTensor *smoothScale,
    const aclTensor *groupList, int64_t dequantMode, int64_t dequantDtype, int64_t quantMode, int64_t quantDtype,
    bool transposeWeight, int64_t groupListType, const aclIntArray *tuningConfigOptional, int64_t swigluMode,
    float clampLimit, float gluAlpha, float gluBias, const char *roundMode, int64_t scaleAlg, float dstTypeMax,
    aclOpExecutor *executor)
{
    L0_DFX(GroupedMatmulSwigluQuantV3, x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList,
           dequantMode, dequantDtype, quantMode, quantDtype, transposeWeight, groupListType, tuningConfigOptional,
           swigluMode, clampLimit, gluAlpha, gluBias, roundMode, scaleAlg, dstTypeMax);
    if (x == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("GroupedMatmulSwigluQuantV3", "x", "does not support nullptr");
        return std::tuple(nullptr, nullptr);
    }
    if (xScale == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("GroupedMatmulSwigluQuantV3", "xScale", "does not support nullptr");
        return std::tuple(nullptr, nullptr);
    }
    if (weightScale == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("GroupedMatmulSwigluQuantV3", "weightScale",
                                                 "does not support nullptr");
        return std::tuple(nullptr, nullptr);
    }
    if (weightScale->Size() == 0) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("GroupedMatmulSwigluQuantV3", "weightScale",
                                                 "does not support empty tensor list");
        return std::tuple(nullptr, nullptr);
    }
    if ((*weightScale)[0] == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("GroupedMatmulSwigluQuantV3", "weightScale[0]",
                                                 "does not support nullptr");
        return std::tuple(nullptr, nullptr);
    }
    int64_t m = x->GetViewShape().GetDim(0);
    auto weightScaleShape = (*weightScale)[0]->GetViewShape();
    int64_t n = transposeWeight ? weightScaleShape.GetDim(1) : weightScaleShape.GetDim(MX_WEIGHT_SCALE_N_DIM);
    int64_t nAfterHalve = n / 2;
    int64_t nAfterSplit = static_cast<int64_t>(Ops::Base::CeilDiv(nAfterHalve, SWIGLU_SPLIT_SIZE));
    gert::Shape outShape({m, nAfterHalve});
    gert::Shape scaleOutShape({m, nAfterSplit, 2});
    auto out = executor->AllocTensor(outShape, static_cast<ge::DataType>(quantDtype), ge::FORMAT_ND);
    auto scaleOut = executor->AllocTensor(scaleOutShape, DataType::DT_FLOAT8_E8M0, ge::FORMAT_ND);
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AllocTensor for GroupedMatmulSwigluQuantV3 out failed.");
        return std::tuple(nullptr, nullptr);
    }
    if (scaleOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AllocTensor for GroupedMatmulSwigluQuantV3 scaleOut failed.");
        return std::tuple(nullptr, nullptr);
    }
    auto ret =
        INFER_SHAPE(GroupedMatmulSwigluQuantV3,
                    OP_INPUT(x, xScale, groupList, weight, weightScale, weightAssistanceMatrix, bias, smoothScale),
                    OP_OUTPUT(out, scaleOut),
                    OP_ATTR(dequantMode, dequantDtype, quantMode, quantDtype, transposeWeight, groupListType,
                             tuningConfigOptional, swigluMode, clampLimit, gluAlpha, gluBias, roundMode, scaleAlg,
                             dstTypeMax));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return std::tuple(nullptr, nullptr);
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        GroupedMatmulSwigluQuantV3,
        OP_INPUT(x, xScale, groupList, weight, weightScale, weightAssistanceMatrix, bias, smoothScale),
        OP_OUTPUT(out, scaleOut),
        OP_ATTR(dequantMode, dequantDtype, quantMode, quantDtype, transposeWeight, groupListType,
                tuningConfigOptional, swigluMode, clampLimit, gluAlpha, gluBias, roundMode, scaleAlg, dstTypeMax));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple(nullptr, nullptr);
    }

    return std::tie(out, scaleOut);
}

} // namespace l0op
