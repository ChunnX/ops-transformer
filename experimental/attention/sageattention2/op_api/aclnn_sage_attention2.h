/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software and is licensed under the CANN Open Software License Agreement
 * Version 2.0. THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND.
 */

#ifndef ACLNN_SAGE_ATTENTION2_H_
#define ACLNN_SAGE_ATTENTION2_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnSageAttention2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, char *tensorLayout, bool isCausal,
    char *qkQuantGran, double smScale, char *pvAccumDtype, bool smoothK, bool smoothV, bool returnLse,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnSageAttention2(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SAGE_ATTENTION2_H_

