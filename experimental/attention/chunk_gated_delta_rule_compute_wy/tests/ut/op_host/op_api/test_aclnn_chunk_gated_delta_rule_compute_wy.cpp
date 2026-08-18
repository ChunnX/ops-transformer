/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "../../../../op_host/op_api/aclnn_chunk_gated_delta_rule_compute_wy.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

namespace {
struct ComputeWyTensors {
    TensorDesc q{{1, 64, 2, 64}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc k{{1, 64, 2, 64}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc v{{1, 64, 4, 64}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc g{{1, 64, 4}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc beta{{1, 64, 4}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc qKernel{{1, 2, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc kKernel{{1, 2, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc wKernel{{1, 4, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc uKernel{{1, 4, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc gKernel{{1, 4, 64}, ACL_FLOAT, ACL_FORMAT_ND};
};

aclnnStatus GetWorkspaceSize(ComputeWyTensors &tensors, int64_t chunkSize, uint64_t *workspaceSize)
{
    return OP_API_UT(aclnnChunkGatedDeltaRuleComputeWy,
                     INPUT(tensors.q, tensors.k, tensors.v, tensors.g, tensors.beta, chunkSize),
                     OUTPUT(tensors.qKernel, tensors.kKernel, tensors.wKernel, tensors.uKernel, tensors.gKernel))
        .TestGetWorkspaceSize(workspaceSize);
}
} // namespace

TEST(AclnnChunkGatedDeltaRuleComputeWy, AcceptsSupportedChunkSize)
{
    ComputeWyTensors tensors;
    uint64_t workspaceSize = 0;
    EXPECT_EQ(GetWorkspaceSize(tensors, 64, &workspaceSize), ACLNN_SUCCESS);
}

TEST(AclnnChunkGatedDeltaRuleComputeWy, RejectsUnsupportedChunkSize)
{
    ComputeWyTensors tensors;
    uint64_t workspaceSize = 0;
    EXPECT_EQ(GetWorkspaceSize(tensors, 32, &workspaceSize), ACLNN_ERR_PARAM_INVALID);
}
