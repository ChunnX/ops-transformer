/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_SAGE_ATTENTION2_OP_API_ACLNN_SAGE_ATTENTION2_H
#define OPS_SAGE_ATTENTION2_OP_API_ACLNN_SAGE_ATTENTION2_H

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "op_api/level2_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * SageAttention2 两段式 aclnn 接口（INT8 QK^T + FP8 PV，前向）。
 * 与 op_host/sage_attention2_def.cpp 的输入/输出/属性注册一一对应：
 * inputs: query, key, value, key_mean(可选), lse_correction(可选)
 * outputs: output, lse(return_lse=false 时可传 nullptr 语义的占位)
 * attrs: tensor_layout, is_causal, qk_quant_gran, sm_scale, pv_accum_dtype,
 *        smooth_k, smooth_v（顺序与 def 注册一致）
 */
aclnnStatus aclnnSageAttention2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *keyMeanOptional, const aclTensor *lseCorrectionOptional,
    const char *tensorLayout, bool isCausal, const char *qkQuantGran, double smScale,
    const char *pvAccumDtype, bool smoothK, bool smoothV,
    const aclTensor *attentionOut, const aclTensor *softmaxLse,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnSageAttention2(void *workspace, uint64_t workspaceSize,
                                aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OPS_SAGE_ATTENTION2_OP_API_ACLNN_SAGE_ATTENTION2_H
