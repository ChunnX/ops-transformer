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

#include <cstdio>

#include "opdev/make_op_executor.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
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

template <typename... Args>
void GmmaqPrint(const char *format, Args... args)
{
    std::printf("[GMMAQ L0] ");
    std::printf(format, args...);
    std::printf("\n");
    std::fflush(stdout);
}

void PrintTensorInfo(const char *name, const aclTensor *tensor)
{
    if (tensor == nullptr) {
        GmmaqPrint("launch tensor %s: nullptr.", name);
        return;
    }
    GmmaqPrint("launch tensor %s: dtype=%s, storageFormat=%s(primary=%s), viewShape=%s, storageShape=%s, "
               "viewDim=%zu, storageDim=%zu, isEmpty=%d.",
               name, op::ToString(tensor->GetDataType()).GetString(),
               op::ToString(tensor->GetStorageFormat()).GetString(),
               op::ToString(static_cast<op::Format>(ge::GetPrimaryFormat(tensor->GetStorageFormat()))).GetString(),
               op::ToString(tensor->GetViewShape()).GetString(),
               op::ToString(tensor->GetStorageShape()).GetString(),
               tensor->GetViewShape().GetDimNum(), tensor->GetStorageShape().GetDimNum(),
               static_cast<int>(tensor->IsEmpty()));
}

void PrintTensorListInfo(const char *name, const aclTensorList *tensorList)
{
    if (tensorList == nullptr) {
        GmmaqPrint("launch tensorList %s: nullptr.", name);
        return;
    }
    GmmaqPrint("launch tensorList %s: size=%zu.", name, tensorList->Size());
    for (size_t i = 0; i < tensorList->Size(); ++i) {
        char tensorName[128] = {};
        std::snprintf(tensorName, sizeof(tensorName), "%s[%zu]", name, i);
        PrintTensorInfo(tensorName, (*tensorList)[i]);
    }
}

void PrintLaunchInfo(const char *stage, const aclTensor *x, const aclTensor *groupList, const aclTensorList *weight,
    const aclTensorList *weightScale, const aclTensorList *bias, const aclTensor *xScale, const aclTensor *out,
    const aclTensor *scaleOut)
{
    GmmaqPrint("GroupedMatmulActivationQuant L0 launch tensor info %s.", stage);
    PrintTensorInfo("x", x);
    PrintTensorInfo("groupList", groupList);
    PrintTensorListInfo("weight", weight);
    PrintTensorListInfo("weightScale", weightScale);
    PrintTensorListInfo("bias", bias);
    PrintTensorInfo("xScale", xScale);
    PrintTensorInfo("out", out);
    PrintTensorInfo("scaleOut", scaleOut);
}
} // namespace

const std::tuple<aclTensor *, aclTensor *> GroupedMatmulActivationQuant(const aclTensor *x,
    const aclTensor *groupList, const aclTensorList *weight, const aclTensorList *weightScale,
    const aclTensorList *bias, const aclTensor *xScale, const char *activationType, bool transposeWeight,
    int64_t groupListType, const aclIntArray *tuningConfig, const char *quantMode, int64_t yDtype,
    const char *roundMode, int64_t scaleAlg, float dstTypeMax, aclOpExecutor *executor)
{
    L0_DFX(GroupedMatmulActivationQuant, x, groupList, weight, weightScale, bias, xScale, activationType,
           transposeWeight, groupListType, tuningConfig, quantMode, yDtype, roundMode, scaleAlg, dstTypeMax);
    GmmaqPrint("GroupedMatmulActivationQuant L0 start. activationType=%s, quantMode=%s, roundMode=%s, "
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
    GmmaqPrint("GroupedMatmulActivationQuant L0 infer output shapes. M=%ld, N=%ld, scaleN=%ld.", m, n, scaleN);
    auto out = executor->AllocTensor(outShape, static_cast<ge::DataType>(yDtype), ge::FORMAT_ND);
    auto scaleOut = executor->AllocTensor(scaleOutShape, DataType::DT_FLOAT8_E8M0, ge::FORMAT_ND);
    GmmaqPrint("GroupedMatmulActivationQuant L0 allocated outputs. out=%p, scaleOut=%p.", out, scaleOut);

    PrintLaunchInfo("before INFER_SHAPE", x, groupList, weight, weightScale, bias, xScale, out, scaleOut);
    GmmaqPrint("GroupedMatmulActivationQuant L0 call INFER_SHAPE.");
    auto ret = INFER_SHAPE(GroupedMatmulActivationQuant,
        OP_INPUT(x, groupList, weight, weightScale, bias, xScale), OP_OUTPUT(out, scaleOut),
        OP_ATTR(activationType, transposeWeight, groupListType, tuningConfig, quantMode, yDtype, roundMode, scaleAlg,
                dstTypeMax));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return std::tuple(nullptr, nullptr);
    }

    GmmaqPrint("GroupedMatmulActivationQuant L0 INFER_SHAPE success, call ADD_TO_LAUNCHER_LIST_AICORE.");
    PrintLaunchInfo("after INFER_SHAPE before ADD_TO_LAUNCHER_LIST_AICORE", x, groupList, weight, weightScale, bias,
        xScale, out, scaleOut);
    ret = ADD_TO_LAUNCHER_LIST_AICORE(GroupedMatmulActivationQuant,
        OP_INPUT(x, groupList, weight, weightScale, bias, xScale), OP_OUTPUT(out, scaleOut),
        OP_ATTR(activationType, transposeWeight, groupListType, tuningConfig, quantMode, yDtype, roundMode, scaleAlg,
                dstTypeMax));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple(nullptr, nullptr);
    }

    GmmaqPrint("GroupedMatmulActivationQuant L0 success.");
    return std::tie(out, scaleOut);
}
} // namespace l0op
