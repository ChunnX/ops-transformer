/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * EngramFetch ACLNN interface (极简骨架版)
 */

#include "aclnn_engram_fetch.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnEngramFetchGetWorkspaceSize(aclTensor *handle, aclTensor *indices, aclTensor *fetched,
                                             uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGE(ACLNN_SUCCESS, "=== aclnnEngramFetchGetWorkspaceSize ENTER ===");
    OP_LOGI("=== aclnnEngramFetchGetWorkspaceSize INFO ===");
    OP_LOGD("aclnnEngramFetchGetWorkspaceSize DEBUG");

    if (handle == nullptr || indices == nullptr || fetched == nullptr || workspaceSize == nullptr ||
        executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Invalid parameters");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    *workspaceSize = 0;
    *executor = nullptr;

    OP_LOGE(ACLNN_SUCCESS, "=== aclnnEngramFetchGetWorkspaceSize EXIT ===");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnEngramFetch(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    OP_LOGE(ACLNN_SUCCESS, "=== aclnnEngramFetch ENTER22 ===");
    OP_LOGI("EngramFetch(issue) skeleton called");
    OP_LOGD("aclnnEngramFetch DEBUG");
    OP_LOGE(ACLNN_SUCCESS, "=== aclnnEngramFetch EXIT ===");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif