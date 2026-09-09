/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.cann.huawei.com/CANN_OPEN_SOURCE_LICENSE_ACK
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef ACLNN_PDIST_H_
#define ACLNN_PDIST_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算执行 aclnnPdist 所需的 workspace 大小并创建执行器。
 * @domain aclnn_ops_infer
 * @param x [in] 必选连续输入，数据类型为 FLOAT16/FLOAT32，shape 为 (N, M)，2 <= N <= 65535，1 <= M <= 65535。
 * @param p [in] 必选 FP64 范数阶，p >= 0 或 +inf；p = 0 统计不相等维度数，p = +inf 取最大绝对差，
 *               0 < p < inf 为幂和的 1/p 次方；p < 0 或 NaN 报参数错误。
 * @param out [out] 必选连续输出，数据类型与 x 一致，shape 为 (N*(N-1)/2,)，上三角行优先排列。
 * @param workspaceSize [out] 返回所需 workspace 大小。
 * @param executor [out] 返回执行器。
 * @return aclnnStatus 状态码。
 */
ACLNN_API aclnnStatus aclnnPdistGetWorkspaceSize(const aclTensor *x, double p, const aclTensor *out,
                                                 uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief 执行 pdist 算子计算。
 * @param workspace [in] workspace 内存地址。
 * @param workspaceSize [in] workspace 大小。
 * @param executor [in] 执行器。
 * @param stream [in] ACL 流。
 * @return aclnnStatus 状态码。
 */
ACLNN_API aclnnStatus aclnnPdist(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                 aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_PDIST_H_
