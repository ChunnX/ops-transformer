/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "common/op_host/op_api/matmul_util.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace Ops::Transformer;
using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

extern aclnnStatus aclnnInnerDispatchFFNCombineW4A8GetWorkspaceSize(const aclTensor* x, const aclTensorList* weight1, const aclTensorList* weight2,
                                                         const aclTensor* expertId, const aclTensorList* scale1, const aclTensorList* scale2,
                                                         const aclTensorList* bias1, const aclTensorList * bias2,
                                                         const aclTensor* probs,
                                                         const char* group, int64_t maxOutputSize,
                                                         bool transB, bool weightNz,
                                                         const aclTensor* out, const aclTensor* expertTokenNums,
                                                         uint64_t* workspaceSize, aclOpExecutor** executor);


extern aclnnStatus aclnnInnerDispatchFFNCombineW4A8(void *workspace, uint64_t workspaceSize,
                                            aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnDispatchFFNCombineW4A8GetWorkspaceSize(const aclTensor* x, const aclTensorList* weight1, const aclTensorList* weight2,
                                                    const aclTensor* expertId, const aclTensorList* scale1, const aclTensorList* scale2,
                                                    const aclTensorList* bias1, const aclTensorList * bias2,
                                                    const aclTensor* probs, const char* group, int64_t maxOutputSize,
                                                    const aclTensor* out, const aclTensor* expertTokenNums,
                                                    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    bool transB = false;
    bool weightNz = true;

    aclnnStatus ret = aclnnInnerDispatchFFNCombineW4A8GetWorkspaceSize(x, weight1, weight2, expertId, 
                                                                    scale1, scale2, bias1, bias2, 
                                                                    probs, group, maxOutputSize, 
                                                                    transB, weightNz,
                                                                    out, expertTokenNums, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnDispatchFFNCombineW4A8(void* workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerDispatchFFNCombineW4A8(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif