/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_URMA_P2P_BENCHMARK_
#define OP_API_INC_URMA_P2P_BENCHMARK_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现UrmaP2pBenchmark功能，测试URMA点对点通信性能的基准测试算子。
 * @brief aclnnUrmaP2pBenchmark的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] x: 计算输入，Tensor，数据类型float16，必须为2维[bs, data_size]，数据格式支持ND。发送的token数据。
 * @param [in] group: 计算输入，str。通信域名称。
 * @param [in] worldSize: 计算输入，int64_t。通信域中的rank总数。
 * @param [in] maxBs: 计算输入，int64_t。最大batch size大小。
 * @param [in] mode: 计算输入，int64_t。测试模式：0=overhead(纯计算开销), 1=latency(时延), 2=bandwidth(带宽)。
 * @param [in] dstRankIdAttr: 计算输入，int64_t。点对点通信目标rank ID，-1表示全部rank。
 * @param [in] aivNum: 计算输入，int64_t。AIV核数量，-1表示使用平台默认值。
 * @param [out] y: 计算输出，Tensor，必选输出，数据类型float16，仅支持2维[max_bs*world_size, data_size]，数据格式支持ND。接收的token数据。
 * @param [out] receiveCnt: 计算输出，Tensor，必选输出，数据类型int32，仅支持1维[world_size]，数据格式支持ND。从各rank接收的token数。
 * @param [out] performanceInfo: 计算输出，Tensor，必选输出，数据类型int64，仅支持1维[6]，数据格式支持ND。Kernel内部各阶段时间戳(cycle计数)。
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码
 *
 */
ACLNN_API aclnnStatus aclnnUrmaP2pBenchmarkGetWorkspaceSize(
    const aclTensor* x,
    char* group, int64_t worldSize, int64_t maxBs, int64_t mode,
    int64_t cclBufferSize, int64_t dstRankIdAttr, int64_t aivNum,
    aclTensor* y, aclTensor* receiveCnt, aclTensor* performanceInfo,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnUrmaP2pBenchmark的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnUrmaP2pBenchmarkGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnUrmaP2pBenchmark(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                              aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif  // OP_API_INC_URMA_P2P_BENCHMARK_