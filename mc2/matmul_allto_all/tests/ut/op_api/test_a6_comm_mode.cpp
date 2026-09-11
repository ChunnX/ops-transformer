/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include "opdev/platform.h"
#include "../../../op_api/matmul_allto_all_util.h"
#include "mc2_a6_utils.h"
#include "mc2_comm_utils.h"

namespace {
class AlltoAllA6CommModeTest : public testing::Test {
protected:
    void SetUp() override
    {
        previousArch_ = op::GetCurrentPlatformInfo().GetCurNpuArch();
        op::SetPlatformNpuArch(NpuArch::DAV_9201);
    }
    void TearDown() override
    {
        op::SetPlatformNpuArch(previousArch_);
    }
    NpuArch previousArch_;
};

TEST_F(AlltoAllA6CommModeTest, AcceptsOnlyAicpuForBothOperators)
{
    for (const char *name : {"allto_all_matmul", "matmul_allto_all"}) {
        uint8_t mode = Mc2Comm::COMM_MODE_CCU;
        EXPECT_EQ(matmul_allto_all_check::CheckAndHandleCommMode(name, "group", "ai_cpu", mode), ACLNN_SUCCESS);
        EXPECT_EQ(mode, Mc2Comm::COMM_MODE_AICPU);
        for (const char *invalid : {"ccu", "aiv", "", "ai_cpu_extra"}) {
            EXPECT_EQ(matmul_allto_all_check::CheckAndHandleCommMode(name, "group", invalid, mode),
                      ACLNN_ERR_PARAM_INVALID);
        }
        EXPECT_EQ(matmul_allto_all_check::CheckAndHandleCommMode(name, "group", nullptr, mode),
                  ACLNN_ERR_PARAM_NULLPTR);
    }
}

TEST_F(AlltoAllA6CommModeTest, PreservesA5CommunicationModes)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    uint8_t mode = Mc2Comm::COMM_MODE_AICPU;
    EXPECT_EQ(matmul_allto_all_check::CheckAndHandleCommMode("matmul_allto_all", "group", "ccu", mode), ACLNN_SUCCESS);
    EXPECT_EQ(mode, Mc2Comm::COMM_MODE_CCU);
    EXPECT_EQ(matmul_allto_all_check::CheckAndHandleCommMode("matmul_allto_all", "group", "ai_cpu", mode),
              ACLNN_SUCCESS);
    EXPECT_EQ(mode, Mc2Comm::COMM_MODE_AICPU);
}

static_assert(mc2tiling::GetAlltoAllCvRatio(NpuArch::DAV_9201) == 1);
static_assert(mc2tiling::GetAlltoAllCvRatio(NpuArch::DAV_3510) == 2);
static_assert(mc2tiling::IsA5OrA6(NpuArch::DAV_9201));
static_assert(!mc2tiling::IsA5OrA6(NpuArch::DAV_2201));
} // namespace
