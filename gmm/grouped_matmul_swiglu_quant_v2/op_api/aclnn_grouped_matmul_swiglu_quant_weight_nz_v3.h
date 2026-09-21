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
 * x 和 weight 必须为 FLOAT8_E4M3FN，weight 必须为 FRACTAL_NZ 格式。weight 和 weightScale 的
 * TensorList 长度必须均为 1。weight 的 API 入口 view 固定为 (E, K, N)：非转置场景使用连续 stride
 * (K * N, N, 1)，storage shape 为 (E, ceil(N / 32), ceil(K / 16), 16, 32)；转置场景先构造
 * (E, N, K) 的 source，再交换末两轴形成 stride (K * N, 1, K) 的 (E, K, N) 入口 view，
 * storage shape 为 (E, ceil(K / 32), ceil(N / 16), 16, 32)。weightScale 的 API 入口 view 固定为
 * (E, ceil(K / 64), N, 2)。非转置 stride 为 (ceil(K / 64) * N * 2, N * 2, 2, 1)；
 * 转置场景先构造 storage shape 为 (E, N, ceil(K / 64), 2) 的 source，再交换中间两轴形成 stride
 * (N * ceil(K / 64) * 2, 2, ceil(K / 64) * 2, 1) 的入口 view。
 * xScale、weightScale 和 outputScale 必须为 FLOAT8_E8M0，dequantMode 和 quantMode 必须为 2，
 * dequantDtype 必须为 DT_FLOAT。weightAssistMatrix、bias 和 smoothScale 当前不支持。
 *
 * @param [in] x: 必选，非空。FLOAT8_E4M3FN、ND，shape 为 (M, K)，M 和 K 均须大于 0。
 * @param [in] weight: 必选，非空。长度为 1 的 TensorList；唯一 Tensor 为 FLOAT8_E4M3FN、
 * FRACTAL_NZ，入口 view shape 为 (E, K, N)，stride 和 storage shape 须满足上述 NZ/ZN 规则。
 * @param [in] weightScale: 必选，非空。长度为 1 的 TensorList；唯一 Tensor 为 FLOAT8_E8M0、ND，
 * 入口 view shape 为 (E, ceil(K / 64), N, 2)，转置属性必须与 weight 一致。
 * @param [in] weightAssistMatrix: 可选。当前仅支持 nullptr、空 TensorList，或长度为 1 且唯一元素为
 * nullptr/shape (0) 的 TensorList。
 * @param [in] bias: 可选。当前必须为 nullptr。
 * @param [in] xScale: 必选，非空。FLOAT8_E8M0、ND，shape 为 (M, ceil(K / 64), 2)。
 * @param [in] smoothScale: 可选。当前必须为 nullptr。
 * @param [in] groupList: 必选，非空。INT64、ND，shape 为 (E,)。
 * @param [in] dequantMode: 反量化模式，当前仅支持 2。
 * @param [in] dequantDtype: GroupedMatmul 中间结果的数据类型，当前仅支持 DT_FLOAT。
 * @param [in] quantMode: 输出量化模式，当前仅支持 2。
 * @param [in] groupListType: groupList 的解释方式；0 表示 cumsum，1 表示 count。
 * @param [in] tuningConfigOptional: 可选。当前仅支持 nullptr 或空数组。
 * @param [in] swigluMode: SwiGLU 模式，当前仅支持 2。
 * @param [in] clampLimit: SwiGLU clamp 上界，必须有限、可由 FLOAT 表示且大于 0。
 * @param [in] gluAlpha: SwiGLU 的 alpha 参数，必须有限且可由 FLOAT 表示。
 * @param [in] gluBias: SwiGLU 的 bias 参数，必须有限且可由 FLOAT 表示。
 * @param [in] roundMode: MX 量化舍入模式，当前仅支持 "rint"；nullptr 或空字符串按 "rint" 处理。
 * @param [in] scaleAlg: MX 量化 scale 算法，MXFP8 输出支持 0（OCP）和 1（cuBLAS）；2 仅支持
 * FLOAT4_E2M1，因此本接口传入 2 会报错。
 * @param [in] dstTypeMax: MX 量化目标类型最大值，MXFP8 输出当前仅支持 0.0。
 * @param [out] output: 必选，非空。FLOAT8_E4M3FN、ND，shape 为 (M, N / 2)。
 * @param [out] outputScale: 必选，非空。FLOAT8_E8M0、ND，shape 为
 * (M, ceil((N / 2) / 64), 2)。
 * @param [out] workspaceSize: 必选，非空。返回用户需要在 NPU device 侧申请的 workspace 大小。
 * @param [out] executor: 必选，非空。返回 op 执行器，包含算子计算流程。
 * @retval ACLNN_SUCCESS 执行成功。
 * @retval ACLNN_ERR_PARAM_NULLPTR（161001）必选指针为空，或 weight/weightScale 中包含空 Tensor。
 * @retval ACLNN_ERR_PARAM_INVALID（161002）参数的 dtype、format、shape、取值或可选参数内容不满足约束。
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
 * @param [in] workspace: 在 NPU device 侧申请的 workspace 内存起址。workspaceSize 大于 0 时必选、非空；
 * workspaceSize 等于 0 时可传 nullptr。
 * @param [in] workspaceSize: 在 NPU device 侧申请的 workspace 大小，由第一段接口获取。
 * @param [in] executor: 必选、非空。op 执行器，包含算子计算流程。
 * @param [in] stream: acl stream 流。
 * @retval ACLNN_SUCCESS 执行成功。
 * @retval ACLNN_ERR_PARAM_NULLPTR（161001）executor 为空，或 workspaceSize 大于 0 时 workspace 为空。
 * @retval ACLNN_ERR_INNER 执行器运行失败。
 */
ACLNN_API aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV3(void *workspace, uint64_t workspaceSize,
                                                              aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
