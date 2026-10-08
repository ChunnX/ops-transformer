/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_grouped_rotate_quant.h"

#include <vector>
#include "grouped_rotate_quant_checker.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"

using namespace op;

namespace {
aclnnStatus MakeContiguous(const aclTensorList *&list, aclOpExecutor *executor)
{
    std::vector<const aclTensor *> tensors;
    tensors.reserve(list->Size());
    for (size_t i = 0; i < list->Size(); ++i) {
        const auto tensor = l0op::Contiguous((*list)[i], executor);
        CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        tensors.emplace_back(tensor);
    }
    list = executor->AllocTensorList(tensors.data(), tensors.size());
    CHECK_RET(list != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus PrepareInputs(grq::Params &p, aclOpExecutor *executor)
{
    CHECK_RET(MakeContiguous(p.x, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguous(p.rotation, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    if (p.alpha != nullptr) {
        p.alpha = l0op::Contiguous(p.alpha, executor);
        CHECK_RET(p.alpha != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (p.groupList != nullptr) {
        p.groupList = l0op::Contiguous(p.groupList, executor);
        CHECK_RET(p.groupList != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

bool AllOutputsEmpty(const aclTensorList *outputs)
{
    for (size_t i = 0; i < outputs->Size(); ++i) {
        if (!(*outputs)[i]->IsEmpty()) {
            return false;
        }
    }
    return true;
}

aclnnStatus CopyOutputs(const aclTensorList *results, const aclTensorList *outputs, aclOpExecutor *executor)
{
    CHECK_RET(results != nullptr && results->Size() == outputs->Size(), ACLNN_ERR_INNER_NULLPTR);
    for (size_t i = 0; i < outputs->Size(); ++i) {
        CHECK_RET((*results)[i] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (!(*outputs)[i]->IsEmpty()) {
            CHECK_RET(l0op::ViewCopy((*results)[i], (*outputs)[i], executor) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnGroupedRotateQuantGetWorkspaceSize(
    const aclTensorList *x, const aclTensorList *rotation, const aclTensor *alpha,
    const aclTensor *groupListOptional, int64_t splitItem, int64_t groupType, int64_t groupListType,
    int64_t axis, const char *roundMode, int64_t scaleAlg, double dstTypeMax, bool trans,
    const aclIntArray *tuningConfigOptional, aclTensorList *yOut, aclTensorList *scaleOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    grq::Params params{x, rotation, alpha, groupListOptional, splitItem, groupType, groupListType,
                       axis, roundMode, scaleAlg, dstTypeMax, trans, tuningConfigOptional, yOut, scaleOut};
    // Validate lists/elements before DFX or metadata access, including the empty-output path.
    const auto ret = grq::CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_COND(op::GetCurrentPlatformInfo().GetCurNpuArch() == op::NpuArch::DAV_3510,
               ACLNN_ERR_PARAM_INVALID, "GroupedRotateQuant currently supports Ascend 950 only.");
    L2_DFX_PHASE_1(aclnnGroupedRotateQuant,
                   DFX_IN(x, rotation, alpha, groupListOptional, splitItem, groupType, groupListType,
                          axis, roundMode, scaleAlg, dstTypeMax, trans, tuningConfigOptional),
                   DFX_OUT(yOut, scaleOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    if (AllOutputsEmpty(params.y)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    CHECK_RET(PrepareInputs(params, uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    auto result = l0op::GroupedRotateQuant(params, uniqueExecutor.get());
    CHECK_RET(CopyOutputs(std::get<0>(result), yOut, uniqueExecutor.get()) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CopyOutputs(std::get<1>(result), scaleOut, uniqueExecutor.get()) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedRotateQuant(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    L2_DFX_PHASE_2(aclnnGroupedRotateQuant);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
