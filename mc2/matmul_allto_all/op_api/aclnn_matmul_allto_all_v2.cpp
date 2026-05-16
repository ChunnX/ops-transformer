/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_matmul_allto_all_v2.h"
#include "matmul_allto_all_base.h"
#include "securec.h"
// #include "matmul_allto_all_util.h"
// #include "common/utils/op_mc2.h"
// #include "acl/acl.h"
// #include "common/utils/op_mc2_def.h"
// #include "common/op_api/mc2_aclnn_util.h"
// #include "opdev/common_types.h"
// #include "opdev/make_op_executor.h"
// #include "aclnn_kernels/common/op_error_check.h"
// #include "opdev/op_dfx.h"
// #include "opdev/platform.h"
// #include "opdev/op_executor.h"
#include "opdev/op_log.h"
// #include "opdev/format_utils.h"
// #include "aclnn_kernels/transdata.h"
// #include "common/utils/hccl_util.h"
// #include "mc2_comm_utils.h"

// 两段式接口
extern "C" aclnnStatus aclnnMatmulAlltoAllV2GetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                           const aclTensor *biasOptional,
                                                           const aclIntArray *alltoAllAxesOptional, const char *group,
                                                           const char *commMode, bool transposeX1, bool transposeX2, const aclTensor *output,
                                                           uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnnMatmulAlltoAllV2GetWorkspaceSize start");
    return aclnnMatmulAlltoAllBaseGetWorkspaceSize(
        x1, x2, biasOptional, x1Scale, x2Scale, commScaleOptional, x1OffsetOptional, x2OffsetOptional,
        alltoAllAxesOptional, group, commMode, x1QuantMode, x2QuantMode, commQuantMode, commQuantDtype, groupSize,
        transposeX1, transposeX2, output, workspaceSize, executor);
}

extern "C" aclnnStatus aclnnMatmulAlltoAllV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             aclrtStream stream)
{
    OP_LOGD("aclnnMatmulAlltoAllV2 start");
    return aclnnMatmulAlltoAllBase(workspace, workspaceSize, executor, stream);
}
