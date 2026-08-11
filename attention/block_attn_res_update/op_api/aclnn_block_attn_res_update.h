/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_BLOCK_ATTN_RES_UPDATE_H_
#define ACLNN_BLOCK_ATTN_RES_UPDATE_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算执行 aclnnBlockAttnResUpdate 所需的 workspace 大小并创建执行器。
 * @domain aclnn_ops_infer
 * @param partialBlock [in/out] 必选连续输入输出，数据类型为 FLOAT，计算结果原地写回。
 * @param delta [in] 必选连续输入，数据类型为 BFLOAT16。
 * @param interNumerator [in] 必选连续输入，数据类型为 FLOAT。
 * @param interMax [in] 必选连续输入，数据类型为 FLOAT。
 * @param interExpSum [in] 必选连续输入，数据类型为 FLOAT。
 * @param effectiveQuery [in] 必选连续输入，数据类型为 FLOAT。
 * @param eps [in] 必选 FP32 浮点属性，必须为有限正数。
 * @param h [out] 必选连续输出，数据类型为 BFLOAT16，shape 与 delta 一致。
 * @param workspaceSize [out] 返回所需 workspace 大小。
 * @param executor [out] 返回执行器。
 * @return aclnnStatus 状态码。
 */
ACLNN_API aclnnStatus aclnnBlockAttnResUpdateGetWorkspaceSize(
    aclTensor *partialBlock, const aclTensor *delta, const aclTensor *interNumerator, const aclTensor *interMax,
    const aclTensor *interExpSum, const aclTensor *effectiveQuery, float eps, aclTensor *h, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief 执行 BlockAttnResUpdate 算子计算。
 * @param workspace [in] workspace 内存地址。
 * @param workspaceSize [in] workspace 大小。
 * @param executor [in] 执行器。
 * @param stream [in] ACL 流。
 * @return aclnnStatus 状态码。
 */
ACLNN_API aclnnStatus aclnnBlockAttnResUpdate(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_BLOCK_ATTN_RES_UPDATE_H_
