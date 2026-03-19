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

using namespace Ops::Transformer;
using namespace op;


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
                const aclTensor* x, const aclTensor* expertIds, const aclTensor* weight1,
                const aclTensor* weightscales1, const aclTensor* weight2,  const aclTensor* weightscales2,
                const char* groupEp, int64_t epWorldSize, int64_t maxoutputsize, aclTensor* y,
                uint64_t* workspaceSize, aclOpExecutor** executor);


extern aclnnStatus aclnnInnerMegaMoe(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor, aclrtStream stream);

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnMegaMoeGetWorkspaceSize(
                const aclTensor* x, const aclTensor* expertIds, const aclTensor* weight1,
                const aclTensor* weightscales1, const aclTensor* weight2,  const aclTensor* weightscales2,
                const char* groupEp, int64_t epWorldSize, int64_t maxoutputsize, aclTensor* y,
                uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("aclnn_mega_moe WorkspaceSize start");
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus getWorkspaceSizesRes = aclnnInnerMegaMoeGetWorkspaceSize(
                                                    x, expertIds, weight1, weightscales1, weight2,
                                                    weightscales2, groupEp, epWorldSize, maxoutputsize, y,
                                                    workspaceSize, executor);

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