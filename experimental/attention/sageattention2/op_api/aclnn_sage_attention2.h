/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "aclnn/acl_meta.h"
#ifdef __cplusplus
extern "C" {
#endif
// smoothV with fp32+fp16 or fp32+fp32 logs a warning and is ignored.
// qkQuantGran supports per_thread (default) and per_warp.
// pvAccumDtype supports fp32+fp16 (default), fp32+fp32, and fp32.
// PV always uses MXFP8.
ACL_FUNC_VISIBILITY aclnnStatus aclnnSageAttention2GetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const char *tensorLayout, bool isCausal,
    const char *qkQuantGran, double smScale, const char *pvAccumDtype, bool smoothK, bool smoothV, bool returnLse,
    const aclTensor *out, const aclTensor *lse, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSageAttention2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                    aclrtStream stream);
#ifdef __cplusplus
}
#endif
