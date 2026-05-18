/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "../../../op_api/aclnn_all_gather_matmul_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "platform/platform_info.h"

using namespace op;
using namespace std;

namespace {

class AllGatherMatmulV2AclnnTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(NpuArch::DAV_3510);
        cout << "AllGatherMatmulV2AclnnTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "AllGatherMatmulV2AclnnTest TearDown" << endl;
    }
};

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherFirstApi1)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmulV2,
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                        OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherFirstApi2)
{
    TensorDesc x1 = TensorDesc({0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmulV2,
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                        OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherGatherOutFalse)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmulV2,
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                        OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherFourthApi)
{
    TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmulV2,
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                        OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherFifthApi)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmulV2,
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                        OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherSixthApi)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmulV2,
                        INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                        OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherCheckHif8BaisInvaild)
{
    TensorDesc x1 = TensorDesc({8, 1}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({1, 1}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmulV2,
                        INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                        OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherCheckScale)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc quant_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, quant_scale, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherCheckAmaxout)
{
    TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOutput = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmulV2,
                        INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                        OUTPUT(output, gatherOut, amaxOutput));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherCheckX1scaleNotNullptr)
{
    TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOutput = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmulV2,
                        INPUT(x1, x2, bias, nullptr, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                        OUTPUT(output, gatherOut, amaxOutput));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherCheckX2scaleNotNullptr)
{
    TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOutput = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmulV2,
                        INPUT(x1, x2, bias, x1Scale, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                        OUTPUT(output, gatherOut, amaxOutput));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherCheckX2scaleNotScalar)
{
    TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOutput = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmulV2,
                        INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                        OUTPUT(output, gatherOut, amaxOutput));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherCheckQuantscaleNotScalar)
{
    TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc quant_quant_scale = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOutput = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, bias, x1Scale, x2Scale, quant_quant_scale, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
        OUTPUT(output, gatherOut, amaxOutput));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherMatmulTransPerblock)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, {1, 256});
    TensorDesc x1Scale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({256, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({256, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 549764202624, "aicpu"),
        OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherV2X2NonContiguousWithoutTranspose)
{
    // 测试场景：非转置的连续x2输入，应返回校验失败
    TensorDesc x1 = TensorDesc({32, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    // 设置x2非转置非连续步长，假设每行之间间隔10个元素[128 + 10, 1]，实际内存存储storageShape为[256, 128 + 10]
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, {128 + 10, 1}, 0, {256, 128 + 10});
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({32, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({32, 256}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));

    // 预期：由于 x2 是非连续的（非转置），应该返回 ACLNN_ERR_PARAM_INVALID
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestStreamModeInvalid)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmulV2,
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 0, 0, "aicpu"),
                        OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestDtypeMismatchX1X2)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestDtypeMismatchX1Output)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestBiasDtypeMismatch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({512}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmulV2,
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                        OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestKAxisTooSmall)
{
    TensorDesc x1 = TensorDesc({8, 100}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({100, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 100}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestKAxisTooLarge)
{
    TensorDesc x1 = TensorDesc({8, 70000}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({70000, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 70000}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestKAxisMismatch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({512, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestNAxisMismatch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestFP16EmptyX1WithValidK)
{
    TensorDesc x1 = TensorDesc({0, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({0, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestFP16KZeroEmpty)
{
    TensorDesc x1 = TensorDesc({8, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAmaxOutWithFP16)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOut = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                  OUTPUT(output, gatherOut, amaxOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAmaxOutWrongDtype)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOut = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                  OUTPUT(output, gatherOut, amaxOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAmaxOutNotScalar)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOut = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                  OUTPUT(output, gatherOut, amaxOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestQuantScaleWrongDtype)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc quantScale = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, quantScale, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestHif8DtypeMismatch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestUnsupportedOutputDtype)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestUnsupportedBiasDtype)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({512}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmulV2,
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                        OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAmaxOutNot1D)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOut = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
                  OUTPUT(output, gatherOut, amaxOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

class AllGatherMatmulV2AclnnAIVTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(NpuArch::DAV_2201);
        cout << "AllGatherMatmulV2AclnnAIVTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "AllGatherMatmulV2AclnnAIVTest TearDown" << endl;
    }
};

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVFP16Basic)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVBF16Basic)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_BF16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVINT8WithScale)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVINT4WithScale)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVInvalidDtypeFP8)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVStreamModeInvalid)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 0, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVKAxisMismatch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({512, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVGatherOutKAxisMismatch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVNAxisMismatch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

} // namespace
