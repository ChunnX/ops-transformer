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

// enum class NnopbaseHcclServerType : uint32_t {
//     NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
//     NNOPBASE_HCCL_SERVER_TYPE_MTE,
//     NNOPBASE_HCCL_SERVER_TYPE_CCU,
//     NNOPBASE_HCCL_SERVER_TYPE_END
// };

// extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

extern aclnnStatus aclnnInnerDispatchFFNCombineGetWorkspaceSize(
    const aclTensor *context, const aclTensor *x, const aclTensor *topkIds, const aclTensor *topkWeights,
    const aclTensorList *weight1, const aclTensorList *weight2,
    const aclTensorList *weightScales1, const aclTensorList *weightScales2,
    const aclTensor *bias1, const aclTensor *bias2,
    const aclTensor *xActiveMask, const aclTensor *scales,
    aclTensor *yOut, aclTensor *expertTokenNumsOut,
    int64_t moeExpertNum, int64_t epWorldSize, int64_t cclBufferSize,
    int64_t maxRecvTokenNum, int64_t dispatchQuantMode, int64_t dispatchQuantOutType,
    int64_t combineQuantMode, const char *commAlg, int64_t numMaxTokenPerRank,
    const char *activation, float activationClamp, bool transposeWeight1, bool transposeWeight2,
    int64_t gmm2XDtype, int64_t weight1Interleave,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerDispatchFFNCombine(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnDispatchFFNCombineGetWorkspaceSize(
    const aclTensor *context, const aclTensor *x, const aclTensor *topkIds, const aclTensor *topkWeights,
    const aclTensorList *weight1, const aclTensorList *weight2,
    const aclTensorList *weightScales1, const aclTensorList *weightScales2,
    const aclTensor *bias1, const aclTensor *bias2,
    const aclTensor *xActiveMask, const aclTensor *scales,
    int64_t moeExpertNum, int64_t epWorldSize, int64_t cclBufferSize,
    int64_t maxRecvTokenNum, int64_t dispatchQuantMode, int64_t dispatchQuantOutType,
    int64_t combineQuantMode, const char *commAlg, int64_t numMaxTokenPerRank,
    const char *activation, float activationClamp, bool transposeWeight1, bool transposeWeight2,
    int64_t gmm2XDtype, int64_t weight1Interleave,
    aclTensor *yOut, aclTensor *expertTokenNumsOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnn_dispatch_ffn_combine WorkspaceSize start");

    OP_CHECK_NULL(context, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(topkIds, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(topkWeights, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(weight1, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(weight2, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(yOut, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(expertTokenNumsOut, return ACLNN_ERR_PARAM_NULLPTR);

    aclnnStatus getWorkspaceSizesRes = aclnnInnerDispatchFFNCombineGetWorkspaceSize(
        context, x, topkIds, topkWeights, weight1, weight2,
        weightScales1, weightScales2, bias1, bias2, xActiveMask, scales,
        yOut, expertTokenNumsOut,
        moeExpertNum, epWorldSize, cclBufferSize, maxRecvTokenNum,
        dispatchQuantMode, dispatchQuantOutType, combineQuantMode,
        commAlg, numMaxTokenPerRank, activation, activationClamp,
        transposeWeight1, transposeWeight2,
        gmm2XDtype, weight1Interleave,
        workspaceSize, executor);

    return getWorkspaceSizesRes;
}

aclnnStatus aclnnDispatchFFNCombine(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    OP_LOGD("aclnn_dispatch_ffn_combine start");
    // if (NnopbaseSetHcclServerType) {
    //     NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
    // }
    return aclnnInnerDispatchFFNCombine(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
