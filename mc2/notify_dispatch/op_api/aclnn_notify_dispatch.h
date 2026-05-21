/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_NOTIFY_DISPATCH_
#define OP_API_INC_NOTIFY_DISPATCH_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现Notify Dispatch通信操作，用于All-to-All通信的通知阶段。
 * @brief aclnnNotifyDispatch的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] sendData: 计算输入，Tensor，数据类型float16、float32、int32，数据格式支持ND。
 *                       发送的数据。
 * @param [in] tokenPerExpertData: 计算输入，Tensor，数据类型float16、float32、int32，数据格式支持ND。
 *                                 Token per expert数据。
 * @param [in] sendCount: 计算输入，int。发送数量。
 * @param [in] numTokens: 计算输入，int。Token总数。
 * @param [in] commGroup: 计算输入，str。通信域名称。
 * @param [in] rankSize: 计算输入，int。Rank总数。
 * @param [in] rankId: 计算输入，int。当前Rank ID。
 * @param [in] localRankSize: 计算输入，int。Local rank size。
 * @param [in] localRankId: 计算输入，int。Local rank ID。
 * @param [out] sendDataOffset: 计算输出，Tensor，数据类型int64，必须为1维，数据格式支持ND。
 *                              发送数据偏移，shape为[rankSize]。
 * @param [out] recvData: 计算输出，Tensor，数据类型与sendData相同，数据格式支持ND。
 *                        接收的数据。
 * @param [out] totalRecvTokens: 计算输出，Tensor，数据类型int64，必须为1维，数据格式支持ND。
 *                               总接收token数，shape为[1]。
 * @param [out] recvCount: 计算输出，Tensor，数据类型int64，必须为1维，数据格式支持ND。
 *                         接收计数，shape为[rankSize]。
 * @param [out] recvOffset: 计算输出，Tensor，数据类型int64，必须为1维，数据格式支持ND。
 *                          接收偏移，shape为[rankSize]。
 * @param [out] maxBs: 计算输出，Tensor，数据类型int64，必须为1维，数据格式支持ND。
 *                     最大batch size，shape为[1]。
 * @param [out] recvTokensPerExpert: 计算输出，Tensor，数据类型int64，必须为1维，数据格式支持ND。
 *                                   每个 expert 接收的 token 数，shape为[rankSize]。
 * @param [out] workspaceSize: 出参，返回需要在NPU device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码
 *
 */
ACLNN_API aclnnStatus aclnnNotifyDispatchGetWorkspaceSize(
    const aclTensor *sendData, const aclTensor *tokenPerExpertData, int64_t sendCount, int64_t numTokens,
    const char *commGroup, int64_t rankSize, int64_t rankId, int64_t localRankSize, int64_t localRankId,
    aclTensor *sendDataOffset, aclTensor *recvData, aclTensor *totalRecvTokens, aclTensor *recvCount,
    aclTensor *recvOffset, aclTensor *maxBs, aclTensor *recvTokensPerExpert, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnNotifyDispatch的第二段接口，用于执行计算。
 * @param [in] workspace: 在NPU device侧申请的workspace内存起始地址。
 * @param [in] workspaceSize: 在NPU device侧申请的workspace大小，由第一段接口aclnnNotifyDispatchGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnNotifyDispatch(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_NOTIFY_DISPATCH_