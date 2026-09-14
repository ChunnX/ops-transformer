/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef OP_API_INC_ACLNN_MHC_PRE_SPLIT_POST_RES_H_
#define OP_API_INC_ACLNN_MHC_PRE_SPLIT_POST_RES_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 参数顺序：输入(x, phi, alpha, bias) + 标量属性(norm_eps) 在前，输出(h_post, h_res) 在后，
 * 与同族 aclnnMhcSandwichNormPostPreonly 及 torch_extension 调用顺序保持一致。
 */
aclnnStatus aclnnMhcPreSplitPostResGetWorkspaceSize(
    const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
    float normEps, const aclTensor *hPostOut, const aclTensor *hResOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnMhcPreSplitPostRes(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_MHC_PRE_SPLIT_POST_RES_H_
