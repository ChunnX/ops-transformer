/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 *
 *  aclnn_dispatch_teardown.cpp —— 对外 aclnn 接口的薄壳实现。
 *  注册框架（op_host/CMakeLists 的 add_modules_sources ... ACLNNTYPE aclnn aclnn_inner）
 *  会从 op 定义生成内层 aclnnInnerDispatchTeardown*；本壳直接转发。
 *  与 all_gather 不同：本算子是纯 AIV、不走通信引擎，故【不】调用 NnopbaseSetHcclServerType。
 */
#include "aclnn_dispatch_teardown.h"

#ifdef __cplusplus
extern "C" {
#endif

// 框架生成的内层接口（签名 = op 定义的 输入... 输出... + workspaceSize + executor）。
extern aclnnStatus aclnnInnerDispatchTeardownGetWorkspaceSize(const aclTensor *status, aclTensor *done,
                                                              uint64_t *workspaceSize, aclOpExecutor **executor);
extern aclnnStatus aclnnInnerDispatchTeardown(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                              aclrtStream stream);

aclnnStatus aclnnDispatchTeardownGetWorkspaceSize(const aclTensor *status, aclTensor *done,
                                                  uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerDispatchTeardownGetWorkspaceSize(status, done, workspaceSize, executor);
}

aclnnStatus aclnnDispatchTeardown(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                  aclrtStream stream)
{
    return aclnnInnerDispatchTeardown(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
