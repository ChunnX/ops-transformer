/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_DISPATCH_FFN_COMBINE_
#define OP_API_INC_DISPATCH_FFN_COMBINE_

#include <string>

#include "aclnn/aclnn_base.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Computes the workspace size required by the unified DispatchFFNCombine operator.
 *
 * The API follows the unified op def (mc2/dispatch_ffn_combine_unified/op_host/dispatch_ffn_combine_def.cpp).
 *
 * @param context         [REQUIRED]  Communication context (INT32 ND).
 * @param x               [REQUIRED]  Input activations (FP16/BF16 ND).
 * @param topkIds         [REQUIRED]  Top-K expert indices (INT32 ND).
 * @param topkWeights     [REQUIRED]  Top-K expert weights (FP32 ND).
 * @param weight1         [DYNAMIC]   First weight tensor list (INT4/INT32/FP16/BF16/INT8).
 * @param weight2         [DYNAMIC]   Second weight tensor list.
 * @param weightScales1   [DYNAMIC]   First weight per-channel scales (INT64 ND).
 * @param weightScales2   [DYNAMIC]   Second weight per-channel scales.
 * @param xActiveMask     [OPTIONAL]  Mask for inactive input tokens (INT8 ND).
 * @param scales          [OPTIONAL]  Input scales (FP32 ND).
 * @param moeExpertNum    [REQUIRED]  Total number of MoE experts.
 * @param epWorldSize     [REQUIRED]  Expert parallelism world size.
 * @param cclBufferSize   [REQUIRED]  CCL buffer size.
 * @param maxRecvTokenNum [OPTIONAL]  Maximum number of tokens received per rank (default 0).
 * @param dispatchQuantMode    [OPTIONAL] Dispatch quantization mode (default 0).
 * @param dispatchQuantOutType [OPTIONAL] Dispatch quantization output type (default DT_UNDEFINED).
 * @param combineQuantMode     [OPTIONAL] Combine quantization mode (default 0).
 * @param commAlg         [OPTIONAL]  Communication algorithm (default "").
 * @param numMaxTokenPerRank    [OPTIONAL] Maximum tokens per rank (default 0).
 * @param activation      [OPTIONAL]  Activation type (default "").
 * @param activationClamp [OPTIONAL]  Activation clamp (default INFINITY, double).
 * @param activation_out_dtype      [OPTIONAL]  GMM2 X dtype (default 0).
 * @param transposeWeight1 [OPTIONAL] Transpose weight1 (default false).
 * @param transposeWeight2 [OPTIONAL] Transpose weight2 (default false).
 * @param weight1Interleave [OPTIONAL] Weight1 interleave (default 0).
 * @param yOut            [REQUIRED]  Output tensor (FP16/BF16 ND).
 * @param expertTokenNumsOut [REQUIRED] Per-expert token counts output (INT32 ND).
 * @param workspaceSize   [OUT]       Required workspace size in bytes.
 * @param executor        [OUT]       Op executor handle.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDispatchFFNCombineGetWorkspaceSize(
    const aclTensor *context, const aclTensor *x, const aclTensor *topkIds, const aclTensor *topkWeights,
    const aclTensorList *weight1, const aclTensorList *weight2,
    const aclTensorList *weightScales1, const aclTensorList *weightScales2,
    const aclTensor *bias1, const aclTensor *bias2,
    const aclTensor *xActiveMask, const aclTensor *scales,
    int64_t moeExpertNum, int64_t epWorldSize, int64_t cclBufferSize,
    int64_t maxRecvTokenNum, int64_t dispatchQuantMode, int64_t dispatchQuantOutType,
    int64_t combineQuantMode, const char *commAlg, int64_t numMaxTokenPerRank,
    const char *activation, float activationClamp, int64_t activation_out_dtype, bool transposeWeight1, bool transposeWeight2,
    int64_t weight1Interleave,
    aclTensor *yOut, aclTensor *expertTokenNumsOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief Launches the unified DispatchFFNCombine operator.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDispatchFFNCombine(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_DISPATCH_FFN_COMBINE_
