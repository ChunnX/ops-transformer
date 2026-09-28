/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "aclnn_pointer_grouped_matmul_add.h"
#include "pointer_grouped_matmul_add.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

using namespace op;

namespace {
constexpr size_t MAX_GROUPS = 128;

bool TensorRange(const aclTensor *tensor, size_t elementBytes, std::pair<uintptr_t, uintptr_t> &range)
{
    if (tensor == nullptr || tensor->Numel() < 0) {
        return false;
    }
    const auto elements = static_cast<uint64_t>(tensor->Numel());
    if (elements == 0) {
        range = {0, 0};
        return true;
    }
    if (tensor->GetData() == nullptr) {
        return false;
    }
    if (elements > std::numeric_limits<uintptr_t>::max() / elementBytes) {
        return false;
    }
    const auto begin = reinterpret_cast<uintptr_t>(tensor->GetData());
    const auto bytes = static_cast<uintptr_t>(elements * elementBytes);
    if (begin > std::numeric_limits<uintptr_t>::max() - bytes) {
        return false;
    }
    range = {begin, begin + bytes};
    return true;
}

bool Overlaps(const std::pair<uintptr_t, uintptr_t> &lhs,
              const std::pair<uintptr_t, uintptr_t> &rhs)
{
    return lhs.first < rhs.second && rhs.first < lhs.second;
}

aclnnStatus CheckTensor(const aclTensor *tensor, DataType dtype, const char *name)
{
    CHECK_COND(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "%s must not be nullptr.", name);
    CHECK_COND(tensor->GetDataType() == dtype, ACLNN_ERR_PARAM_INVALID, "%s has an unsupported dtype.", name);
    CHECK_COND(tensor->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
               "%s must use ND format.", name);
    CHECK_COND(IsContiguous(tensor), ACLNN_ERR_PARAM_INVALID, "%s must be contiguous; no copy fallback is allowed.",
               name);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const aclTensor *x, const aclTensor *weight, const aclTensor *groupList,
                        aclTensorList *yRefs, bool transposeX, bool transposeWeight,
                        int64_t groupType, int64_t groupListType)
{
    CHECK_COND(yRefs != nullptr, ACLNN_ERR_PARAM_NULLPTR, "yRefs must not be nullptr.");
    CHECK_COND(yRefs->Size() > 0 && yRefs->Size() <= MAX_GROUPS, ACLNN_ERR_PARAM_INVALID,
               "yRefs length must be in [1, %zu].", MAX_GROUPS);
    CHECK_COND(CheckTensor(x, DataType::DT_BF16, "x") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "x validation failed.");
    CHECK_COND(CheckTensor(weight, DataType::DT_BF16, "weight") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "weight validation failed.");
    CHECK_COND(CheckTensor(groupList, DataType::DT_INT64, "groupList") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "groupList validation failed.");
    CHECK_COND(transposeX && !transposeWeight, ACLNN_ERR_PARAM_INVALID,
               "transposeX must be true and transposeWeight must be false.");
    CHECK_COND(groupType == 2 && groupListType == 1, ACLNN_ERR_PARAM_INVALID,
               "only groupType=2 and groupListType=1 are supported.");

    const auto xShape = x->GetViewShape();
    const auto weightShape = weight->GetViewShape();
    const auto groupShape = groupList->GetViewShape();
    CHECK_COND(xShape.GetDimNum() == 2 && weightShape.GetDimNum() == 2, ACLNN_ERR_PARAM_INVALID,
               "x and weight must be two-dimensional.");
    CHECK_COND(groupShape.GetDimNum() == 1 && static_cast<size_t>(groupShape.GetDim(0)) == yRefs->Size(),
               ACLNN_ERR_PARAM_INVALID, "groupList length must equal yRefs length.");
    CHECK_COND(xShape.GetDim(0) == weightShape.GetDim(0), ACLNN_ERR_PARAM_INVALID,
               "x and weight packed K dimensions must match.");

    std::pair<uintptr_t, uintptr_t> xRange;
    std::pair<uintptr_t, uintptr_t> weightRange;
    std::pair<uintptr_t, uintptr_t> groupRange;
    CHECK_COND(TensorRange(x, sizeof(uint16_t), xRange) &&
                   TensorRange(weight, sizeof(uint16_t), weightRange) &&
                   TensorRange(groupList, sizeof(int64_t), groupRange),
               ACLNN_ERR_PARAM_INVALID, "input tensor address range is invalid.");
    std::vector<std::pair<uintptr_t, uintptr_t>> targetRanges;
    targetRanges.reserve(yRefs->Size());

    for (size_t index = 0; index < yRefs->Size(); ++index) {
        auto *target = (*yRefs)[index];
        CHECK_COND(CheckTensor(target, DataType::DT_FLOAT, "yRefs element") == ACLNN_SUCCESS,
                   ACLNN_ERR_PARAM_INVALID, "yRefs[%zu] validation failed.", index);
        const auto targetShape = target->GetViewShape();
        CHECK_COND(targetShape.GetDimNum() == 2 && targetShape.GetDim(0) == xShape.GetDim(1) &&
                       targetShape.GetDim(1) == weightShape.GetDim(1),
                   ACLNN_ERR_PARAM_INVALID, "yRefs[%zu] must have shape [x.size(1), weight.size(1)].", index);
        std::pair<uintptr_t, uintptr_t> targetRange;
        CHECK_COND(TensorRange(target, sizeof(float), targetRange), ACLNN_ERR_PARAM_INVALID,
                   "yRefs[%zu] address range is invalid.", index);
        CHECK_COND(!Overlaps(targetRange, xRange) && !Overlaps(targetRange, weightRange) &&
                       !Overlaps(targetRange, groupRange),
                   ACLNN_ERR_PARAM_INVALID, "yRefs[%zu] must not overlap an input tensor.", index);
        for (size_t prior = 0; prior < targetRanges.size(); ++prior) {
            CHECK_COND(!Overlaps(targetRange, targetRanges[prior]), ACLNN_ERR_PARAM_INVALID,
                       "yRefs indices %zu and %zu overlap.", prior, index);
        }
        targetRanges.push_back(targetRange);
    }
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" {
aclnnStatus aclnnPointerGroupedMatmulAddGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight, const aclTensor *groupList, aclTensorList *yRefs,
    bool transposeX, bool transposeWeight, int64_t groupType, int64_t groupListType,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_COND(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be nullptr.");
    CHECK_COND(GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510, ACLNN_ERR_PARAM_INVALID,
               "PointerGroupedMatmulAdd is supported only on Ascend 950-class devices.");
    CHECK_COND(CheckParams(x, weight, groupList, yRefs, transposeX, transposeWeight, groupType, groupListType) ==
                   ACLNN_SUCCESS,
               ACLNN_ERR_PARAM_INVALID, "PointerGroupedMatmulAdd parameter validation failed.");

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    if (x->IsEmpty() || weight->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // PyTorch may expose a two-dimensional view backed by a flattened
    // storage descriptor.  CreateView only fixes the executor metadata; it
    // aliases the same device address and performs no allocation or copy.
    auto *xView = uniqueExecutor->CreateView(x, x->GetViewShape(), x->GetViewOffset());
    auto *weightView = uniqueExecutor->CreateView(weight, weight->GetViewShape(), weight->GetViewOffset());
    auto *groupListView =
        uniqueExecutor->CreateView(groupList, groupList->GetViewShape(), groupList->GetViewOffset());
    CHECK_RET(xView != nullptr && weightView != nullptr && groupListView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto *result = l0op::PointerGroupedMatmulAdd(xView, weightView, groupListView, yRefs,
                                                 transposeX, transposeWeight,
                                                 groupType, groupListType, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnPointerGroupedMatmulAdd(void *workspace, uint64_t workspaceSize,
                                         aclOpExecutor *executor, aclrtStream stream)
{
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "PointerGroupedMatmulAdd kernel execution failed.");
    return ACLNN_SUCCESS;
}
} // extern "C"
