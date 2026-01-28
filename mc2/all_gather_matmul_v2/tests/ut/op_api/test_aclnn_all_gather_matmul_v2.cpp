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

TEST_F(AllGatherMatmulV2AclnnTest, test_all_gather_first_api_1)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
        OUTPUT(output, gatherOut, nullptr)
    );
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, test_all_gather_first_api_2)
{
	TensorDesc x1 = TensorDesc({0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc x2 = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc output = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc gatherOut = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
		OUTPUT(output, gatherOut, nullptr)
    );
	uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
	aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, test_all_gather_gather_out_false)
{
	TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc gatherOut = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
		OUTPUT(output, gatherOut, nullptr)
    );
	uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
	aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, test_all_gather_fourth_api)
{
	TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc output = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc gatherOut = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
        OUTPUT(output, gatherOut, nullptr)
    );
	uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
	aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, test_all_gather_fifth_api)
{
	TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc output = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc gatherOut = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
		OUTPUT(output, gatherOut, nullptr)
    );
	uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
	aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, test_all_gather_sixth_api)
{
	TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc x1_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc x2_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc output = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc gatherOut = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, bias, x1_scale, x2_scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
		OUTPUT(output, gatherOut, nullptr)
    );
	uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
	aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, test_all_gather_check_hif8_bais_invaild)
{
	TensorDesc x1 = TensorDesc({8, 1}, ACL_HIFLOAT8, ACL_FORMAT_ND);
	TensorDesc x2 = TensorDesc({1, 1}, ACL_HIFLOAT8, ACL_FORMAT_ND);
	TensorDesc x1_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc x2_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc bias = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc output = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc gatherOut = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, bias, x1_scale, x2_scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
		OUTPUT(output, gatherOut, nullptr)
    );
	uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
	aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, test_all_gather_check_scale)
{
	TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
	TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
	TensorDesc x1_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc x2_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc bias = TensorDesc({256}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc quant_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc output = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc gatherOut = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, bias, x1_scale, x2_scale, quant_scale, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
		OUTPUT(output, gatherOut, nullptr)
    );
	uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
	aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, test_all_gather_check_amaxout)
{
	TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
	TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
	TensorDesc x1_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc x2_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc output = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc amax_output = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, bias, x1_scale, x2_scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
		OUTPUT(output, gatherOut, amax_output)
    );
	uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
	aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, test_all_gather_check_x1scale_not_nullptr)
{
	TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
	TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
	TensorDesc x2_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc output = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc amax_output = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, bias, nullptr, x2_scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
		OUTPUT(output, gatherOut, amax_output)
    );
	uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
	aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, test_all_gather_check_x2scale_not_nullptr)
{
	TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
	TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
	TensorDesc x1_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc output = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc amax_output = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, bias, x1_scale, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
		OUTPUT(output, gatherOut, amax_output)
    );
	uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
	aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, test_all_gather_check_x2scale_not_scalar)
{
	TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
	TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
	TensorDesc x1_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc x2_scale = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc output = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc amax_output = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, bias, x1_scale, x2_scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
		OUTPUT(output, gatherOut, amax_output)
    );
	uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
	aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, test_all_gather_check_quantscale_not_scalar)
{
	TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
	TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
	TensorDesc x1_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc x2_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc quant_quant_scale = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc output = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc amax_output = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, bias, x1_scale, x2_scale, quant_quant_scale, 0, "test_all_gather_group", 0, 8, 1, 0, "aicpu"),
        OUTPUT(output, gatherOut, amax_output)
    );
	uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
	aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, test_all_gather_matmul_trans_perblock)
{
	TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
	TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, {1, 256});
	TensorDesc x1_scale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc x2_scale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc output = TensorDesc({256, 256}, ACL_FLOAT, ACL_FORMAT_ND);
	TensorDesc gatherOut = TensorDesc({256, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, nullptr, x1_scale, x2_scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 549764202624, "aicpu"),
		OUTPUT(output, gatherOut, nullptr)
    );
	uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
	aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

} // namespace