/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_allto_allv_grouped_mat_mul_v2.h"

#include <cstdint>
#include <initializer_list>

#include "aclnnInner_allto_allv_grouped_mat_mul_shmem.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "common/op_api/mc2_aclnn_util.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

namespace {
constexpr int64_t MIN_EP_WORLD_SIZE = 1;
constexpr int64_t MAX_EP_WORLD_SIZE = 8;
constexpr uint64_t CONTEXT_TENSOR_ELEMENTS = 1;
constexpr uint64_t DIM_X = 2;
constexpr uint64_t DIM_WEIGHT = 3;

aclnnStatus CheckRequired(const aclTensor *gmmX, const aclTensor *gmmWeight, void *shmemContext, aclTensor *gmmY,
                          uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_NULL(gmmX, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(gmmWeight, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(shmemContext, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(gmmY, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(workspaceSize, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(executor, return ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckTensorContract(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmY)
{
    const std::initializer_list<op::DataType> supportedDtypes = {op::DataType::DT_BF16};
    OP_CHECK_DTYPE_NOT_SUPPORT(gmmX, supportedDtypes, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(gmmWeight, supportedDtypes, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(gmmY, supportedDtypes, return ACLNN_ERR_PARAM_INVALID);
    if (gmmX->GetStorageFormat() != op::Format::FORMAT_ND ||
        gmmWeight->GetStorageFormat() != op::Format::FORMAT_ND ||
        gmmY->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmX/gmmWeight/gmmY must use ND storage format.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!MC2Aclnn::IsTensorContiguous(gmmX) || !MC2Aclnn::IsTensorContiguous(gmmWeight) ||
        !MC2Aclnn::IsTensorContiguous(gmmY)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "gmmX/gmmWeight/gmmY must be contiguous; SHMEM gmmX cannot be replaced by a temporary copy.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (gmmX->GetViewShape().GetDimNum() != DIM_X || gmmWeight->GetViewShape().GetDimNum() != DIM_WEIGHT ||
        gmmY->GetViewShape().GetDimNum() != DIM_X) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmX/gmmWeight/gmmY must be 2D/3D/2D tensors.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t m = gmmX->GetViewShape().GetDim(0);
    const int64_t k = gmmX->GetViewShape().GetDim(1);
    const int64_t experts = gmmWeight->GetViewShape().GetDim(0);
    const int64_t weightK = gmmWeight->GetViewShape().GetDim(1);
    const int64_t n = gmmWeight->GetViewShape().GetDim(2);
    if (m < 0 || k <= 0 || experts <= 0 || experts > 64 || n <= 0 || k != weightK ||
        gmmY->GetViewShape().GetDim(1) != n) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid source-compatible M/K/N/E dimensions.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckCounts(const aclIntArray *sendCounts, const aclIntArray *recvCounts, int64_t epWorldSize,
                        int64_t localExperts)
{
    if (sendCounts == nullptr || recvCounts == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sendCounts and recvCounts must not be null.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    uint64_t sendSize = 0;
    uint64_t recvSize = 0;
    aclGetIntArraySize(sendCounts, &sendSize);
    aclGetIntArraySize(recvCounts, &recvSize);
    const uint64_t expected = static_cast<uint64_t>(epWorldSize * localExperts);
    if (sendSize != expected || recvSize != expected) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "sendCounts/recvCounts length must equal epWorldSize * localExperts, expected %llu, got %llu/%llu.",
                static_cast<unsigned long long>(expected), static_cast<unsigned long long>(sendSize),
                static_cast<unsigned long long>(recvSize));
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclTensor *CreateShmemContextTensor(void *shmemContext)
{
    const int64_t shape[1] = {static_cast<int64_t>(CONTEXT_TENSOR_ELEMENTS)};
    const int64_t strides[1] = {1};
    // Only the data address is consumed by the A5 kernel; the context memory is owned by the SHMEM runtime.
    return aclCreateTensor(shape, 1, ACL_UINT8, strides, 0, ACL_FORMAT_ND, shape, 1, shmemContext);
}
} // namespace

extern "C" aclnnStatus aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, void *shmemContext, int64_t epWorldSize,
    const aclIntArray *sendCounts, const aclIntArray *recvCounts, aclTensor *gmmY, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    auto ret = CheckRequired(gmmX, gmmWeight, shmemContext, gmmY, workspaceSize, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The SHMEM V2 interface only supports Ascend 950 (DAV_3510).");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (epWorldSize < MIN_EP_WORLD_SIZE || epWorldSize > MAX_EP_WORLD_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "epWorldSize must be in [1, 8], got %ld.", epWorldSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    ret = CheckTensorContract(gmmX, gmmWeight, gmmY);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckCounts(sendCounts, recvCounts, epWorldSize, gmmWeight->GetViewShape().GetDim(0));
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    aclTensor *contextTensor = CreateShmemContextTensor(shmemContext);
    if (contextTensor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Failed to wrap shmemContext as a borrowed device tensor.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    ret = aclnnInnerAlltoAllvGroupedMatMulShmemGetWorkspaceSize(
        gmmX, gmmWeight, contextTensor, epWorldSize, sendCounts, recvCounts, gmmY, workspaceSize, executor);
    OP_LOGD("AlltoAllvGroupedMatMulV2 SHMEM GetWorkspaceSize returned %d.", ret);
    return ret;
}

extern "C" aclnnStatus aclnnAlltoAllvGroupedMatMulV2(void *workspace, uint64_t workspaceSize,
                                                      aclOpExecutor *executor, aclrtStream stream)
{
    const aclnnStatus ret = aclnnInnerAlltoAllvGroupedMatMulShmem(workspace, workspaceSize, executor, stream);
    OP_LOGD("AlltoAllvGroupedMatMulV2 SHMEM execute returned %d.", ret);
    return ret;
}
