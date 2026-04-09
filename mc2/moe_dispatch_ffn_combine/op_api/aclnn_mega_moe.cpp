/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_mega_moe.h"
#include <algorithm>
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "common/op_host/op_api/matmul_util.h"
#include "mc2_moe_context_tensor.h"

using namespace Ops::Transformer;
using namespace op;
using namespace Mc2MoeDistributeContext;

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};


extern aclnnStatus aclnnInnerMegaMoeGetWorkspaceSize(
                const aclTensor* context, const aclTensor* x, const aclTensor* expertIds, const aclTensor* scalesOptional,
                const aclTensor* xActiveMaskOptional, const aclTensor* expertScalesOptional, const aclTensor* weight1,
                const aclTensor* weightscales1, const aclTensor* weight2,  const aclTensor* weightscales2,
                const char* groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t cclBufferSize,
                int64_t maxRecvTokenNum, int64_t shareExpertNum, int64_t shareExpertRankNum, int64_t dispatchQuantMode,
                int64_t dispatchQuantOutType ,int64_t combineQuantMode, const char* commAlg,
                int64_t globalBs, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor);


extern aclnnStatus aclnnInnerMegaMoe(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor, aclrtStream stream);

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnMegaMoeGetWorkspaceSize(
                const aclTensor* x, const aclTensor* expertIds, const aclTensor* scalesOptional,
                const aclTensor* xActiveMaskOptional, const aclTensor* expertScalesOptional, const aclTensor* weight1,
                const aclTensor* weightscales1, const aclTensor* weight2,  const aclTensor* weightscales2,
                const char* groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
                int64_t maxRecvTokenNum, int64_t shareExpertNum, int64_t shareExpertRankNum, int64_t dispatchQuantMode,
                int64_t dispatchQuantOutType, int64_t combineQuantMode, const char* commAlg,
                int64_t globalBs, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("aclnn_mega_moe WorkspaceSize start");

    aclnnStatus ret;
    aclTensor* mc2Context = nullptr;
    uint64_t hcclBuffSize = 0;
    const char* opName = "mega_moe";
    ret = GetMc2ContextTensor(groupEp, opName, hcclBuffSize, mc2Context);

    OP_LOGD("GetMc2ContextTensor success.");

    aclnnStatus getWorkspaceSizesRes = aclnnInnerMegaMoeGetWorkspaceSize(
                                        mc2Context, x, expertIds, scalesOptional, xActiveMaskOptional, expertScalesOptional,
                                        weight1, weightscales1, weight2, weightscales2, groupEp, epWorldSize, epRankId,
                                        moeExpertNum, hcclBuffSize, maxRecvTokenNum, shareExpertNum, shareExpertRankNum,
                                        dispatchQuantMode, dispatchQuantOutType, combineQuantMode,
                                        commAlg, globalBs, y, workspaceSize, executor);

    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }

    OP_LOGD("aclnnInnerMegaMoeGetWorkspaceSize get workspaceSize");

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