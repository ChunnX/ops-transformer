/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LAST_TRANSPOSE_NDDMA_BENCH_H_
#define OP_API_INC_LAST_TRANSPOSE_NDDMA_BENCH_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnLastTransposeNddmaBench 第一段接口：计算 workspace 大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] x: npu device 侧 aclTensor，4D shape (dim0,dim1,dim2,dim3)，
 *   数据类型支持 FLOAT16/FLOAT/INT8，数据格式支持 ND。
 * @param [in] isMte3: host 侧 int64，1=转置结果写回 GM（精度对比，默认），0=去掉 MTE3。
 * @param [in] y: npu device 侧 aclTensor，4D shape (dim0,dim1,dim3,dim2)，
 *   数据类型与 x 一致，数据格式支持 ND。
 * @param [out] workspaceSize: 返回需要在 npu device 侧申请的 workspace 大小。
 * @param [out] executor: 返回 op 执行器。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLastTransposeNddmaBenchGetWorkspaceSize(
    const aclTensor* x, int64_t isMte3, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnLastTransposeNddmaBench 第二段接口：执行计算。
 * @param [in] workspace: npu device 侧 workspace 内存起址。
 * @param [in] workspaceSize: workspace 大小，由第一段接口获取。
 * @param [in] executor: op 执行器。
 * @param [in] stream: acl stream 流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLastTransposeNddmaBench(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LAST_TRANSPOSE_NDDMA_BENCH_H_
