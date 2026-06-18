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
OP_TYPE_REGISTER(GroupedMatmulActivationQuant);

namespace {
constexpr int64_t X_M_DIM_INDEX = 0L;
constexpr int64_t WEIGHT_SCALE_TRANS_N_DIM_INDEX = 1L;
constexpr int64_t WEIGHT_SCALE_N_DIM_INDEX = 2L;
constexpr int64_t MX_GROUP_SIZE = 64L;
constexpr int64_t MX_SCALE_PAIR = 2L;
constexpr size_t FIRST_TENSOR_INDEX = 0UL;
} // namespace

const std::tuple<aclTensor *, aclTensor *> GroupedMatmulActivationQuant(const aclTensor *x,
    const aclTensor *groupList, const aclTensorList *weight, const aclTensorList *weightScale,
    const aclTensorList *bias, const aclTensor *xScale, const char *activationType, bool transposeWeight,
    int64_t groupListType, const aclIntArray *tuningConfig, const char *quantMode, int64_t yDtype,
    const char *roundMode, int64_t scaleAlg, float dstTypeMax, aclOpExecutor *executor)
{
    L0_DFX(GroupedMatmulActivationQuant, x, groupList, weight, weightScale, bias, xScale, activationType,
           transposeWeight, groupListType, tuningConfig, quantMode, yDtype, roundMode, scaleAlg, dstTypeMax);
    OP_LOGD("GroupedMatmulActivationQuant L0 start. activationType=%s, quantMode=%s, roundMode=%s, "
            "transposeWeight=%d, groupListType=%ld, yDtype=%ld, scaleAlg=%ld, dstTypeMax=%f.",
            activationType == nullptr ? "nullptr" : activationType, quantMode == nullptr ? "nullptr" : quantMode,
            roundMode == nullptr ? "nullptr" : roundMode, static_cast<int>(transposeWeight), groupListType,
            yDtype, scaleAlg, dstTypeMax);
    if (x == nullptr || weightScale == nullptr || weightScale->Size() == 0 || (*weightScale)[FIRST_TENSOR_INDEX] == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x and weightScale must not be nullptr.");
        return std::tuple(nullptr, nullptr);
    }

    int64_t m = x->GetViewShape().GetDim(X_M_DIM_INDEX);
    const auto &weightScaleShape = (*weightScale)[FIRST_TENSOR_INDEX]->GetViewShape();
    int64_t n = transposeWeight ? weightScaleShape.GetDim(WEIGHT_SCALE_TRANS_N_DIM_INDEX) :
                                  weightScaleShape.GetDim(WEIGHT_SCALE_N_DIM_INDEX);
    gert::Shape outShape({m, n});
    int64_t scaleN = static_cast<int64_t>(Ops::Base::CeilDiv(n, MX_GROUP_SIZE));
    gert::Shape scaleOutShape({m, scaleN, MX_SCALE_PAIR});
    OP_LOGD("GroupedMatmulActivationQuant L0 infer output shapes. M=%ld, N=%ld, scaleN=%ld.", m, n, scaleN);
    auto out = executor->AllocTensor(outShape, static_cast<ge::DataType>(yDtype), ge::FORMAT_ND);
    auto scaleOut = executor->AllocTensor(scaleOutShape, DataType::DT_FLOAT8_E8M0, ge::FORMAT_ND);
    OP_LOGD("GroupedMatmulActivationQuant L0 allocated outputs. out=%p, scaleOut=%p.", out, scaleOut);

    OP_LOGD("GroupedMatmulActivationQuant L0 call INFER_SHAPE.");
    auto ret = INFER_SHAPE(GroupedMatmulActivationQuant,
        OP_INPUT(x, groupList, weight, weightScale, bias, xScale), OP_OUTPUT(out, scaleOut),
        OP_ATTR(activationType, transposeWeight, groupListType, tuningConfig, quantMode, yDtype, roundMode, scaleAlg,
                dstTypeMax));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return std::tuple(nullptr, nullptr);
    }

    OP_LOGD("GroupedMatmulActivationQuant L0 INFER_SHAPE success, call ADD_TO_LAUNCHER_LIST_AICORE.");
    ret = ADD_TO_LAUNCHER_LIST_AICORE(GroupedMatmulActivationQuant,
        OP_INPUT(x, groupList, weight, weightScale, bias, xScale), OP_OUTPUT(out, scaleOut),
        OP_ATTR(activationType, transposeWeight, groupListType, tuningConfig, quantMode, yDtype, roundMode, scaleAlg,
                dstTypeMax));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple(nullptr, nullptr);
    }

    OP_LOGD("GroupedMatmulActivationQuant L0 success.");
    return std::tie(out, scaleOut);
}
} // namespace l0op
