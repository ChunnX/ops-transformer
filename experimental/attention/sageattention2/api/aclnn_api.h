/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_API_H
#define ACLNN_API_H

#include <stdint.h>
#include <stdbool.h>
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Computes workspace size for SageAttention2 on Ascend 950PR.
 *
 * @param[in]  q               Query tensor [B, Hq, Sq, D] (HND) or [B, Sq, Hq, D] (NHD), FP16 or BF16.
 * @param[in]  k               Key tensor [B, Hk, Sk, D] (HND) or [B, Sk, Hk, D] (NHD), matching dtype of q.
 * @param[in]  v               Value tensor [B, Hk, Sk, D] (HND) or [B, Sk, Hk, D] (NHD), matching dtype of q.
 * @param[in]  tensorLayout    Tensor memory layout string: "HND" or "NHD".
 * @param[in]  isCausal        Whether causal triangular masking is enforced (requires Sq == Sk).
 * @param[in]  qkQuantGran     Quantization granularity: "per_thread" (default) or "per_warp".
 * @param[in]  smScale         Softmax scaling factor; must be finite and non-zero.
 * @param[in]  pvAccumDtype    Accumulator mode: "fp32+fp16" (default), "fp32+fp32", or "fp32".
 * @param[in]  smoothK         Enables K-channel sequence mean subtraction for outlier suppression.
 * @param[in]  smoothV         Enables V-channel sequence mean subtraction (active only when pvAccumDtype is "fp32").
 * @param[in]  returnLse       Whether to compute and export the logsumexp normalization tensor.
 * @param[out] out             Output attention tensor, matching shape and dtype of q.
 * @param[out] lse             Logsumexp tensor [B, Hq, Sq], required if returnLse is true, float32.
 * @param[out] workspaceSize   Allocated byte size of the device temporary workspace.
 * @param[out] executor        Opaque pointer to the constructed CANN operation executor.
 * @return aclnnStatus         ACLNN_SUCCESS on success, appropriate error code otherwise.
 */
ACL_FUNC_VISIBILITY aclnnStatus aclnnSageAttention2GetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const char *tensorLayout,
    bool isCausal,
    const char *qkQuantGran,
    double smScale,
    const char *pvAccumDtype,
    bool smoothK,
    bool smoothV,
    bool returnLse,
    const aclTensor *out,
    const aclTensor *lse,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief Launches the pre-built SageAttention2 executor on the specified Ascend runtime stream.
 *
 * @param[in] workspace        Device memory pointer to the workspace allocated by the caller.
 * @param[in] workspaceSize    Byte length of the allocated workspace.
 * @param[in] executor         CANN operation executor created during workspace calculation.
 * @param[in] stream           Ascend runtime stream for asynchronous kernel dispatch.
 * @return aclnnStatus         ACLNN_SUCCESS on success, appropriate error code otherwise.
 */
ACL_FUNC_VISIBILITY aclnnStatus aclnnSageAttention2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_API_H
