/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_compressor_grad.h
 * \brief
 */

#ifndef ACLNN_COMPRESSOR_GRAD_H
#define ACLNN_COMPRESSOR_GRAD_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnCompressorGradGetWorkspaceSize(
    const aclTensor *x, const aclTensor *wkv, const aclTensor *wgate,
    const aclTensor *dCmpKv, const aclTensor *softmaxScore, const aclTensor *kv,
    const aclTensor *cuSeqLensOptional, const aclTensor *sequsedOptional,
    const aclTensor *startPosOptional, const char *layoutX, const char *layoutDCmpKv,
    const char *layoutSoftmaxScore, const char *layoutKv,
    const char *layoutDX, int64_t coff, int64_t cmpRatio,
    const aclTensor *dX, const aclTensor *dWkv, const aclTensor *dWgate,
    const aclTensor *dApe, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus
aclnnCompressorGrad(void *workspace,
                    uint64_t workspaceSize,
                    aclOpExecutor *executor,
                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_COMPRESSOR_GRAD_H
