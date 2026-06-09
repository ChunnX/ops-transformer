/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * EngramFetchWait ACLNN interface (极简骨架版)
 */

#include "aclnn_engram_fetch_wait.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnEngramFetchWaitGetWorkspaceSize(aclTensor *handle, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnnEngramFetchWaitGetWorkspaceSize enter");

    if (handle == nullptr || workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Invalid parameters");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    *workspaceSize = 0;
    *executor = nullptr;

    OP_LOGD("aclnnEngramFetchWaitGetWorkspaceSize exit");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnEngramFetchWait(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    OP_LOGD("aclnnEngramFetchWait enter");
    OP_LOGI("EngramFetchWait skeleton called");
    OP_LOGD("aclnnEngramFetchWait exit");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif