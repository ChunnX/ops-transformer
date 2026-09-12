/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, either express or implied,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "acl/acl.h"

#include "../../../op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.h"

TEST(GroupedMatmulSwigluQuantWeightNzV3OpApi, DirectLaunchRejectsNullExecutor)
{
    EXPECT_NE(aclnnGroupedMatmulSwigluQuantWeightNzV3(nullptr, 0, nullptr, nullptr), ACL_SUCCESS);
}

TEST(GroupedMatmulSwigluQuantWeightNzV3OpApi, WorkspaceRejectsNullRequiredInputs)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    EXPECT_NE(aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, nullptr, -1,
                  true, 0, 7.0, 1.702, 1.0, -1, "rint", 0, 0.0, nullptr, nullptr, &workspaceSize, &executor),
              ACL_SUCCESS);
}
