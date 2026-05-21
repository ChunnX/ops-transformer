/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MOE_COMBINE_NORMAL_
#define OP_API_INC_MOE_COMBINE_NORMAL_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现MoE Combine阶段计算，将专家计算结果按路由权重加权求和。
 * @brief aclnnMoeCombineNormal的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] recvX: 计算输入，Tensor，数据类型float16、bfloat16，必须为2维，数据格式支持ND。
 *                    接收到的专家计算结果，shape为[A, H]，其中A为总接收token数。
 * @param [in] tokenSrcInfo: 计算输入，Tensor，数据类型int32，必须为2维，数据格式支持ND。
 *                           Token来源信息三元组，shape为[A, 3]，每行包含[epRankId, tokenIndex, topk_id]。
 * @param [in] epRecvCount: 计算输入，Tensor，数据类型int32，必须为1维，数据格式支持ND。
 *                          EP域接收计数（累积和），shape为[numExperts]。
 * @param [in] topkWeights: 计算输入，Tensor，数据类型float32，必须为2维，数据格式支持ND。
 *                          TopK路由权重，shape为[BS, K]。
 * @param [in] tpRecvCountOptional: 计算可选输入，Tensor，数据类型int32，必须为1维，数据格式支持ND。
 *                                   TP域接收计数，若有TP域通信需要传参，否则传空指针。
 * @param [in] groupEp: 计算输入，str。EP通信域名称，专家并行的通信域。
 * @param [in] epWorldSize: 计算输入，int。EP通信域大小。
 * @param [in] epRankId: 计算输入，int。EP本卡ID。同一个EP通信域中各卡的epRankId不重复。
 * @param [in] tpWorldSize: 计算可选输入，int。TP通信域大小。
 * @param [in] tpRankId: 计算可选输入，int。TP本卡ID。同一个TP通信域中各卡的tpRankId不能重复。
 * @param [in] moeExpertNum: 计算输入，int。MoE专家数量。
 * @param [in] moeExpertPerRankNum: 计算输入，int。每个rank上的专家数量。
 * @param [in] globalBs: 计算可选输入，int。全局batch size。
 * @param [in] expertShardType: 计算可选输入，int。专家切分类型。当前仅支持传0。
 * @param [in] isEnableDiagnose: 计算可选输入，bool。是否启用诊断模式。
 * @param [out] xOut: 计算输出，Tensor，必选输出，数据类型支持float16、bfloat16，仅支持2维，数据格式支持ND。
 *                    加权求和后的输出，shape为[BS, H]。
 * @param [out] sendCostStatsOutOptional: 计算可选输出，Tensor，数据类型int32，必须为1维，数据格式支持ND。
 *                                         发送耗时统计信息，shape为[epWorldSize]。
 * @param [out] workspaceSize: 出参，返回需要在NPU device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码
 *
 */
ACLNN_API aclnnStatus aclnnMoeCombineNormalGetWorkspaceSize(
    const aclTensor* recvX,
    const aclTensor* tokenSrcInfo,
    const aclTensor* epRecvCount,
    const aclTensor* topkWeights,
    const aclTensor* tpRecvCountOptional,
    const char* groupEp,
    int64_t epWorldSize,
    int64_t epRankId,
    int64_t tpWorldSize,
    int64_t tpRankId,
    int64_t moeExpertNum,
    int64_t moeExpertPerRankNum,
    int64_t globalBs,
    int64_t expertShardType,
    bool isEnableDiagnose,
    aclTensor* xOut,
    aclTensor* sendCostStatsOutOptional,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnMoeCombineNormal的第二段接口，用于执行计算。
 * @param [in] workspace: 在NPU device侧申请的workspace内存起始地址。
 * @param [in] workspaceSize: 在NPU device侧申请的workspace大小，由第一段接口aclnnMoeCombineNormalGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMoeCombineNormal(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MOE_COMBINE_NORMAL_