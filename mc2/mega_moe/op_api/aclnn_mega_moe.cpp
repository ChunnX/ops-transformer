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

static void CreateEmptyTensor(aclDataType dataType, const aclTensorList *&ioList,
                              aclTensorList *&outList, aclOpExecutor *executor)
{
    if (ioList == nullptr) {
        std::vector<aclTensor*> emptyTensors;
        aclTensor *emptyTensor = executor->AllocTensor({0}, static_cast<op::DataType>(dataType));
        emptyTensors.emplace_back(emptyTensor);
        outList = executor->AllocTensorList(emptyTensors.data(), emptyTensors.size());
        ioList = outList;
    }
}

aclnnStatus aclnnMegaMoeGetWorkspaceSize(
    const aclTensor* context, const aclTensor* x, const aclTensor* topkIds, const aclTensor* topkWeights,
    const aclTensorList* weight1, const aclTensorList* weight2, const aclTensorList* weightScales1Optional,
    const aclTensorList* weightScales2Optional, const aclTensor* xActiveMaskOptional,
    const aclTensor* scalesOptional, int64_t moeExpertNum, int64_t epWorldSize, int64_t cclBufferSize,
    int64_t maxRecvTokenNum, int64_t dispatchQuantMode, int64_t dispatchQuantOutType, int64_t combineQuantMode,
    const char* commAlg, int64_t globalBs, aclTensor* yOut, aclTensor* expertTokenNumsOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
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

    const aclTensorList* bias1Optional = nullptr;
    const aclTensorList* bias2Optional = nullptr;
    // 可选 DYNAMIC 参数为 nullptr 时创建带 dtype 的 dummy tensor list 满足支持列表校验
    aclTensorList *tmpBiasList = nullptr;
    aclTensorList *tmpScaleList = nullptr;
    CreateEmptyTensor(ACL_FLOAT, bias1Optional, tmpBiasList, *executor);
    CreateEmptyTensor(ACL_FLOAT, bias2Optional, tmpBiasList, *executor);
    CreateEmptyTensor(ACL_UINT64, weightScales1Optional, tmpScaleList, *executor);
    CreateEmptyTensor(ACL_UINT64, weightScales2Optional, tmpScaleList, *executor);

    aclnnStatus getWorkspaceSizesRes = aclnnInnerMegaMoeGetWorkspaceSize(
        context, x, topkIds, topkWeights, weight1, weight2,
        weightScales1Optional, weightScales2Optional, bias1Optional, bias2Optional,
        xActiveMaskOptional, scalesOptional,
        moeExpertNum, epWorldSize, cclBufferSize, maxRecvTokenNum, dispatchQuantMode, dispatchQuantOutType,
        combineQuantMode, const_cast<char*>(commAlg), 0, "swiglu",
        std::numeric_limits<float>::max(), ge::DT_UNDEFINED, false, false, 0,
        yOut, expertTokenNumsOut, workspaceSize, executor);

    return getWorkspaceSizesRes;
}

aclnnStatus aclnnMegaMoe(void* workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    OP_LOGD("aclnn_mega_moe start");
    return aclnnInnerMegaMoe(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif