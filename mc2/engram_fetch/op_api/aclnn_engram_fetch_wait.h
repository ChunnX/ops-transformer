/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ACLNN_ENGRAM_FETCH_WAIT_
#define OP_API_INC_ACLNN_ENGRAM_FETCH_WAIT_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * EngramFetchWait算子（极简骨架版） - wait阶段
 * @brief 获取workspace大小
 * @domain aclnn_ops_infer
 * @param [in] handle: 占位符Tensor
 * @param [out] workspaceSize: workspace大小
 * @param [out] executor: op执行器
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnEngramFetchWaitGetWorkspaceSize(aclTensor *handle,
                                                                                        uint64_t *workspaceSize,
                                                                                        aclOpExecutor **executor);

/**
 * EngramFetchWait算子 - wait阶段执行
 * @param [in] workspace: workspace内存
 * @param [in] workspaceSize: workspace大小
 * @param [in] executor: op执行器
 * @param [in] stream: acl stream
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnEngramFetchWait(void *workspace, uint64_t workspaceSize,
                                                                        aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_ENGRAM_FETCH_WAIT_