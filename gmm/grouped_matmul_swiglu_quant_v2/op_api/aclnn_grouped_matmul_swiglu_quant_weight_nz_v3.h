/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, either express or implied,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_OP_API_ACLNN_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_V3_H
#define OP_HOST_OP_API_ACLNN_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_V3_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupedMatmulSwigluQuantWeightNzV3 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 此接口当前仅支持 Ascend 950（arch35）的单 Tensor MXFP8 weightNZ 场景。
 * x 和 weight 必须为 FLOAT8_E4M3FN，weight 必须为 FRACTAL_NZ 格式，
 * weight 的逻辑 view 为三维、storage 为五维，weight 和 weightScale 的 TensorList 长度必须均为 1。
 * xScale、weightScale 和 outputScale 必须为 FLOAT8_E8M0，dequantMode 和 quantMode 必须为 2，
 * dequantDtype 必须为 DT_FLOAT。weightAssistMatrix、bias 和 smoothScale 当前不支持。
 *
 * @param [in] x: 激活矩阵。
 * @param [in] weight: FLOAT8_E4M3FN、FRACTAL_NZ 格式的权重矩阵，TensorList 长度为 1。
 * @param [in] weightScale: FLOAT8_E8M0 权重量化因子，TensorList 长度为 1。
 * @param [in] weightAssistMatrix: 权重辅助矩阵，当前必须为空。
 * @param [in] bias: 偏置，当前必须为 nullptr。
 * @param [in] xScale: FLOAT8_E8M0 激活矩阵量化因子。
 * @param [in] smoothScale: 平滑量化因子，当前必须为 nullptr。
 * @param [in] groupList: 分组信息。
 * @param [in] dequantMode: 反量化模式，当前仅支持 2。
 * @param [in] dequantDtype: GroupedMatmul 中间结果的数据类型，当前仅支持 DT_FLOAT。
 * @param [in] quantMode: 输出量化模式，当前仅支持 2。
 * @param [in] groupListType: groupList 的解释方式。
 * @param [in] tuningConfigOptional: 调优参数。
 * @param [in] swigluMode: SwiGLU 模式，当前仅支持 2。
 * @param [in] clampLimit: SwiGLU clamp 上界，必须大于 0。
 * @param [in] gluAlpha: SwiGLU 的 alpha 参数。
 * @param [in] gluBias: SwiGLU 的 bias 参数。
 * @param [in] roundMode: MX 量化舍入模式，当前仅支持 "rint"；nullptr 或空字符串按 "rint" 处理。
 * @param [in] scaleAlg: MX 量化 scale 算法，MXFP8 输出支持 0（OCP）和 1（cuBLAS）；2 仅支持 FLOAT4_E2M1，因此本接口传入 2 会报错。
 * @param [in] dstTypeMax: MX 量化目标类型最大值，MXFP8 输出当前仅支持 0.0。
 * @param [out] output: FLOAT8_E4M3FN 量化输出。
 * @param [out] outputScale: FLOAT8_E8M0 输出量化因子。
 * @param [out] workspaceSize: 返回用户需要在 NPU device 侧申请的 workspace 大小。
 * @param [out] executor: 返回 op 执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
    const aclTensor *x, const aclTensorList *weight, const aclTensorList *weightScale,
    const aclTensorList *weightAssistMatrix, const aclTensor *bias, const aclTensor *xScale,
    const aclTensor *smoothScale, const aclTensor *groupList, int64_t dequantMode, int64_t dequantDtype,
    int64_t quantMode, int64_t groupListType, const aclIntArray *tuningConfigOptional, int64_t swigluMode,
    double clampLimit, double gluAlpha, double gluBias, const char *roundMode, int64_t scaleAlg, double dstTypeMax,
    aclTensor *output, aclTensor *outputScale, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnGroupedMatmulSwigluQuantWeightNzV3 的第二段接口，用于执行计算。
 * @param [in] workspace: 在 NPU device 侧申请的 workspace 内存起址。
 * @param [in] workspaceSize: 在 NPU device 侧申请的 workspace 大小，由第一段接口获取。
 * @param [in] executor: op 执行器，包含算子计算流程。
 * @param [in] stream: acl stream 流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV3(void *workspace, uint64_t workspaceSize,
                                                              aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
