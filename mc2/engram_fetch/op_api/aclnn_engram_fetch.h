/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ACLNN_ENGRAM_FETCH_
#define OP_API_INC_ACLNN_ENGRAM_FETCH_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * EngramFetch算子（极简骨架版） - issue阶段
 * @brief 获取workspace大小
 * @domain aclnn_ops_infer
 * @param [in] handle: 占位符Tensor，极简版使用空Tensor
 * @param [in] indices: int32 Tensor，全局索引 [num_tokens]
 * @param [out] fetched: bf16 Tensor，输出buffer [num_tokens, hidden]
 * @param [out] workspaceSize: 出参，返回需要在NPU device侧申请的workspace大小
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnEngramFetchGetWorkspaceSize(
    aclTensor *handle, aclTensor *indices, aclTensor *fetched, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * EngramFetch算子 - issue阶段执行
 * @param [in] workspace: 在NPU device侧申请的workspace内存起址
 * @param [in] workspaceSize: workspace大小
 * @param [in] executor: op执行器
 * @param [in] stream: acl stream流
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnEngramFetch(void *workspace, uint64_t workspaceSize,
                                                                    aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_ENGRAM_FETCH_