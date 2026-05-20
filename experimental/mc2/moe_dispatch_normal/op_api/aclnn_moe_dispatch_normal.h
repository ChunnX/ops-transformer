/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MOE_DISPATCH_NORMAL_
#define OP_API_INC_MOE_DISPATCH_NORMAL_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现MoE Dispatch阶段计算，将token数据分发到对应的专家rank。
 * @brief aclnnMoeDispatchNormal的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] x: 计算输入，Tensor，数据类型float16、bfloat16，必须为2维，数据格式支持ND。
 *                输入token特征，shape为[BS, H]。
 * @param [in] topkIdx: 计算输入，Tensor，数据类型int32，必须为2维，数据格式支持ND。
 *                      TopK专家索引，shape为[BS, K]。
 * @param [in] sendOffset: 计算输入，Tensor，数据类型int32，必须为1维，数据格式支持ND。
 *                         发送偏移量，shape为[epWorldSize]。
 * @param [in] sendTokenIdx: 计算输入，Tensor，数据类型int32，必须为1维，数据格式支持ND。
 *                           发送token索引。
 * @param [in] recvOffset: 计算输入，Tensor，数据类型int32，必须为1维，数据格式支持ND。
 *                         接收偏移量，shape为[epWorldSize]。
 * @param [in] recvCount: 计算输入，Tensor，数据类型int32，必须为1维，数据格式支持ND。
 *                        接收计数，shape为[epWorldSize]。
 * @param [in] groupEp: 计算输入，str。EP通信域名称，专家并行的通信域。
 * @param [in] epWorldSize: 计算输入，int。EP通信域大小。
 * @param [in] epRankId: 计算输入，int。EP本卡ID。同一个EP通信域中各卡的epRankId不重复。
 * @param [in] groupTpOptional: 计算可选输入，str。TP通信域名称，若有TP域通信需要传参，否则传空字符串。
 * @param [in] tpWorldSize: 计算可选输入，int。TP通信域大小。
 * @param [in] tpRankId: 计算可选输入，int。TP本卡ID。同一个TP通信域中各卡的tpRankId不能重复。
 * @param [in] moeExpertNum: 计算输入，int。MoE专家数量。
 * @param [in] quantMode: 计算可选输入，int。量化模式。当前仅支持传0。
 * @param [in] globalBs: 计算可选输入，int。全局batch size。
 * @param [out] recvX: 计算输出，Tensor，必选输出，数据类型支持float16、bfloat16，仅支持2维，数据格式支持ND。
 *                     接收到的token特征，shape为[total_recv_tokens, H]。
 * @param [out] recvXScales: 计算输出，Tensor，必选输出，数据类型支持float32，仅支持1维，数据格式支持ND。
 *                           接收到的scales，shape为[total_recv_tokens]。
 * @param [out] assistInfoForCombine: 计算输出，Tensor，必选输出，数据类型支持int32，仅支持2维，数据格式支持ND。
 *                                    辅助信息用于combine，shape为[total_recv_tokens, 3]。
 * @param [out] waitRecvCostStatsOutOptional: 计算可选输出，Tensor，数据类型int32，必须为1维，数据格式支持ND。
 *                                             接收耗时统计信息，shape为[epWorldSize]。
 * @param [out] workspaceSize: 出参，返回需要在NPU device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码
 *
 */
ACLNN_API aclnnStatus aclnnMoeDispatchNormalGetWorkspaceSize(
    const aclTensor *x, const aclTensor *topkIdx, const aclTensor *sendOffset, const aclTensor *sendTokenIdx,
    const aclTensor *recvOffset, const aclTensor *recvCount, const char *groupEp, int64_t epWorldSize, int64_t epRankId,
    const char *groupTpOptional, int64_t tpWorldSize, int64_t tpRankId, int64_t moeExpertNum, int64_t quantMode,
    int64_t globalBs, aclTensor *recvX, aclTensor *recvXScales, aclTensor *assistInfoForCombine,
    const aclTensor *waitRecvCostStatsOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnMoeDispatchNormal的第二段接口，用于执行计算。
 * @param [in] workspace: 在NPU device侧申请的workspace内存起始地址。
 * @param [in] workspaceSize: 在NPU
 * device侧申请的workspace大小，由第一段接口aclnnMoeDispatchNormalGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMoeDispatchNormal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MOE_DISPATCH_NORMAL_