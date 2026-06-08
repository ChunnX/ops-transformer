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
#include <vector>
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "common/op_host/op_api/mc2_3rd_matmul_util.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnnInner_mega_moe.h"

using namespace Ops::Transformer;
using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static aclTensorList *CreateEmptyTensorList(aclDataType dataType, aclOpExecutor *executor)
{
    std::vector<aclTensor *> emptyTensors;
    aclTensor *emptyTensor = executor->AllocTensor({0}, static_cast<op::DataType>(dataType));
    emptyTensors.push_back(emptyTensor);
    aclTensorList *emptyList = executor->AllocTensorList(emptyTensors.data(), emptyTensors.size());
    return emptyList;
}

aclnnStatus aclnnMegaMoeGetWorkspaceSize(
    const aclTensor *context, const aclTensor *x, const aclTensor *topkIds, const aclTensor *topkWeights,
    const aclTensorList *weight1, const aclTensorList *weight2, const aclTensorList *weightScales1Optional,
    const aclTensorList *weightScales2Optional, const aclTensorList *bias1Optional, const aclTensorList *bias2Optional,
    const aclTensor *xActiveMaskOptional, int64_t moeExpertNum, int64_t epWorldSize, int64_t cclBufferSize,
    int64_t maxRecvTokenNum, int64_t dispatchQuantMode, int64_t dispatchQuantOutDtype, int64_t combineQuantMode,
    const char *commAlg, int64_t numMaxTokensPerRank, const char *activation, float activationClamp, aclTensor *yOut,
    aclTensor *expertTokenNumsOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnn_mega_moe WorkspaceSize start");

    OP_CHECK_NULL(context, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(topkIds, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(topkWeights, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(weight1, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(weight2, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(yOut, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(expertTokenNumsOut, return ACLNN_ERR_PARAM_NULLPTR);

    // 确保 executor 已创建，以便调用 CreateEmptyTensor
    if (*executor == nullptr) {
        auto uniqueExec = CREATE_EXECUTOR();
        uniqueExec.ReleaseTo(executor);
    }

    // DYNAMIC 参数为 nullptr 时创建带 dtype 的 dummy tensor list 满足支持列表校验
    const aclTensorList *bias1Param =
        (bias1Optional != nullptr) ? bias1Optional : CreateEmptyTensorList(ACL_FLOAT, *executor);
    const aclTensorList *bias2Param =
        (bias2Optional != nullptr) ? bias2Optional : CreateEmptyTensorList(ACL_FLOAT, *executor);
    const aclTensorList *weightScale1Param =
        (weightScales1Optional != nullptr) ? weightScales1Optional : CreateEmptyTensorList(ACL_UINT64, *executor);
    const aclTensorList *weightScale2Param =
        (weightScales2Optional != nullptr) ? weightScales2Optional : CreateEmptyTensorList(ACL_UINT64, *executor);

    aclnnStatus getWorkspaceSizesRes = aclnnInnerMegaMoeGetWorkspaceSize(
        context, x, topkIds, topkWeights, weight1, weight2, weightScale1Param, weightScale2Param, bias1Param,
        bias2Param, xActiveMaskOptional, nullptr, moeExpertNum, epWorldSize, cclBufferSize, maxRecvTokenNum,
        dispatchQuantMode, dispatchQuantOutDtype, combineQuantMode, const_cast<char *>(commAlg), 0,
        const_cast<char *>(activation), activationClamp, ge::DT_UNDEFINED, false, false, 0, yOut, expertTokenNumsOut, workspaceSize,
        executor);

    return getWorkspaceSizesRes;
}

aclnnStatus aclnnMegaMoe(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    OP_LOGD("aclnn_mega_moe start");
    return aclnnInnerMegaMoe(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif