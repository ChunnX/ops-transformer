/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_MLA_PREPROCESS_V2_H
#define ACLNN_MLA_PREPROCESS_V2_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculates the workspace size and creates an executor for MlaPreprocessV2.
 *
 * @param input [IN] Hidden-state tensor used to calculate Query and Key.
 * @param gamma0 [IN] Gamma parameter of the first RMSNorm operation.
 * @param beta0 [IN] Beta parameter of the first RMSNorm operation.
 * @param quantScale0 [IN] Quantization scale of the first RMSNormQuant operation.
 * @param quantOffset0 [IN] Quantization offset of the first RMSNormQuant operation.
 * @param wdqkv [IN] Down-projection weight applied to input.
 * @param descale0 [IN] Dequantization scale for the matrix multiplication with wdqkv.
 * @param bias0 [IN] Bias for the matrix multiplication with wdqkv.
 * @param gamma1 [IN] Gamma parameter of the second RMSNorm operation.
 * @param beta1 [IN] Beta parameter of the second RMSNorm operation.
 * @param quantScale1 [IN] Quantization scale of the second RMSNormQuant operation.
 * @param quantOffset1 [IN] Quantization offset of the second RMSNormQuant operation.
 * @param wuq [IN] Query up-projection weight.
 * @param descale1 [IN] Dequantization scale for the matrix multiplication with wuq.
 * @param bias1 [IN] Bias for the matrix multiplication with wuq.
 * @param gamma2 [IN] Gamma parameter used to normalize the compressed Key/Value tensor.
 * @param cos [IN] Cosine tensor for RoPE. Pass nullptr together with sin to disable RoPE; passing only one
 * RoPE tensor as nullptr is invalid.
 * @param sin [IN] Sine tensor for RoPE. Pass nullptr together with cos to disable RoPE; passing only one
 * RoPE tensor as nullptr is invalid.
 * @param wuk [IN] Key up-projection weight used to calculate the Q-nope output.
 * @param kvCache [IN] Key/Value cache tensor updated according to slotMapping.
 * @param kvCacheRope [IN] RoPE cache tensor. Its presence and layout depend on cacheMode.
 * @param slotMapping [IN] Cache slot index for each input token.
 * @param ctkvScale [IN] Quantization scale for the compressed Key/Value cache when cacheMode is 2.
 * @param qNopeScale [IN] Quantization scale for the Q-nope output when cacheMode is 2.
 * @param wdqDim [IN] Query LoRA dimension split from the wdqkv matrix multiplication result.
 * @param qRopeDim [IN] Dimension of the Query component processed by RoPE.
 * @param kRopeDim [IN] Dimension of the Key component processed by RoPE.
 * @param epsilon [IN] Epsilon added to the RMSNorm denominator for numerical stability.
 * @param qRotaryCoeff [IN] Rotary coefficient for Query.
 * @param kRotaryCoeff [IN] Rotary coefficient for Key.
 * @param transeposeWdq [IN] Whether wdqkv is transposed for matrix multiplication.
 * @param transeposeWuq [IN] Whether wuq is transposed for matrix multiplication.
 * @param transeposeWuk [IN] Whether wuk is transposed for matrix multiplication.
 * @param cacheMode [IN] Cache layout and output mode. Supported values are 0, 1, 2, and 3.
 * @param quantMode [IN] RMSNorm quantization mode.
 * @param doRmsNorm [IN] Whether to apply RMSNorm before the first quantization operation.
 * @param wdkvSplitCount [IN] Number of split segments used for the wdqkv down-projection.
 * @param qDownOutFlag [IN] Whether to produce the Query down-projection output.
 * @param qOutOut [OUT] Query output tensor.
 * @param kvCacheOutOut [OUT] Updated Key/Value cache output tensor.
 * @param qRopeOutOut [OUT] Query RoPE component output tensor for split-cache modes.
 * @param krCacheOutOut [OUT] Updated Key RoPE cache output tensor for split-cache modes.
 * @param qDownOutOut [OUT] Query down-projection output tensor.
 * @param workspaceSize [OUT] Required device workspace size in bytes.
 * @param executor [OUT] Executor containing the prepared operator execution flow.
 * @return aclnnStatus ACLNN_SUCCESS on success; otherwise, an error code is returned.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMlaPreprocessV2GetWorkspaceSize(
    const aclTensor *input, const aclTensor *gamma0, const aclTensor *beta0, const aclTensor *quantScale0,
    const aclTensor *quantOffset0, const aclTensor *wdqkv, const aclTensor *descale0, const aclTensor *bias0,
    const aclTensor *gamma1, const aclTensor *beta1, const aclTensor *quantScale1, const aclTensor *quantOffset1,
    const aclTensor *wuq, const aclTensor *descale1, const aclTensor *bias1, const aclTensor *gamma2,
    const aclTensor *cos, const aclTensor *sin, const aclTensor *wuk, const aclTensor *kvCache,
    const aclTensor *kvCacheRope, const aclTensor *slotMapping, const aclTensor *ctkvScale,
    const aclTensor *qNopeScale, int64_t wdqDim, int64_t qRopeDim, int64_t kRopeDim, double epsilon,
    int64_t qRotaryCoeff, int64_t kRotaryCoeff, bool transeposeWdq, bool transeposeWuq, bool transeposeWuk,
    int64_t cacheMode, int64_t quantMode, bool doRmsNorm, int64_t wdkvSplitCount, bool qDownOutFlag,
    const aclTensor *qOutOut, const aclTensor *kvCacheOutOut, const aclTensor *qRopeOutOut,
    const aclTensor *krCacheOutOut, const aclTensor *qDownOutOut, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief Executes MlaPreprocessV2 asynchronously using the executor created by
 * aclnnMlaPreprocessV2GetWorkspaceSize.
 *
 * @param workspace [IN] Device workspace address. It can be nullptr when workspaceSize is 0.
 * @param workspaceSize [IN] Device workspace size in bytes returned by the first-phase interface.
 * @param executor [IN] Executor returned by aclnnMlaPreprocessV2GetWorkspaceSize.
 * @param stream [IN] ACL runtime stream used to execute the operator.
 * @return aclnnStatus ACLNN_SUCCESS on successful launch; otherwise, an error code is returned.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMlaPreprocessV2(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_MLA_PREPROCESS_V2_H
