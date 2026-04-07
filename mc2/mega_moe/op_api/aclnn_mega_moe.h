/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MEGA_MOE_
#define OP_API_INC_MEGA_MOE_

#include <string>

#include "aclnn/aclnn_base.h"
#define ACLNN_API __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现MegaMoe功能。
 * @brief aclnnMegaMoe的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] x: 计算输入，Tensor，数据类型float16，bfloat16，必须为2维，shape为(M, H)，数据格式支持ND。输入的token数据。
 * @param [in] expertIds: 计算输入，Tensor，数据类型int32，必须为2维，数据格式支持ND。每个token的topK个专家索引。
 * @param [in] scalesOptional: 计算可选输入，Tensor，数据类型float32，必须为2维，数据格式支持ND。每个专家的smooth权重。
 * @param [in] xActiveMaskOptional: 计算输入，Tensor，数据类型Bool，必须为1维，数据格式支持ND。
 * @param [in] expertScalesOptional: 计算输入，Tensor，必须为2维，数据格式支持ND。
 * @param [in] weight1: 计算可选输入，Tensor，数据类型mxfp8，必须为3维，shape为(expertPerRank, H, N)，数据格式支持ND。
 * @param [in] weightscales1: 计算输入，Tensor，数据类型fp8_e8m0，必须为3维，shape为(expertPerRank, N, H/64)，数据格式支持ND。
 * @param [in] weight2: 计算可选输入，Tensor，数据类型mxfp8，必须为3维，shape为(expertPerRank, N/2, H)，数据格式支持ND。
 * @param [in] weightscales2: 计算输入，Tensor，数据类型fp8_e8m0，必须为3维，shape为(expertPerRank, H, N/2/64)，数据格式支持ND。
 * @param [in] groupEp: 计算输入，str。ep通信域名称，专家并行的通信域。不能和groupTp相同。
 * @param [in] epWorldSize: 计算输入，int。ep通信域size。
 * @param [in] epRankId: 计算输入，int。ep本卡Id。同一个EP通信域中各卡的epRankId不能重复
 * @param [in] moeExpertNum: 计算输入，int。MOE专家数量。
 * @param [in] maxRecvTokenNum: 计算输入，int。Gmm算子输出shape的M轴最大值，用于申请workspace，超出则截断。
 * @param [in] dispatchQuantMode: 计算输入，int。dispatch阶段的量化模式。
 * @param [in] combineQuantMode: 计算输入，int。combine阶段的量化模式。
 * @param [in] commAlg: 计算输入，str。通信使用的算法名称。
 * @param [in] globalBs: 计算输入，int64_t。全局batch size大小
 * @param [out] y: 计算输出，Tensor，必选输出，数据类型支持float16, bfloat16，仅支持2维，shape为(M, H)，数据格式支持ND。
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码
 *
 */
ACLNN_API aclnnStatus aclnnMegaMoeGetWorkspaceSize(
                const aclTensor* x, const aclTensor* expertIds, const aclTensor* scalesOptional,
                const aclTensor* xActiveMaskOptional, const aclTensor* expertScalesOptional, const aclTensor* weight1,
                const aclTensor* weightscales1, const aclTensor* weight2,  const aclTensor* weightscales2,
                const char* groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
                int64_t maxRecvTokenNum, int64_t shareExpertNum, int64_t shareExpertRankNum, int64_t dispatchQuantMode,
                int64_t dispatchQuantOutType, int64_t combineQuantMode, const char* commAlg,
                int64_t globalBs, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnMegaMoe的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnMegaMoeGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMegaMoe(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                 aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MEGA_MOE_