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
#include "../../../op_api/aclnn_matmul_reduce_scatter_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "platform/platform_info.h"

using namespace op;
using namespace std;

namespace {

class MatmulReduceScatterV2AclnnTest : public testing::Test {
protected:
    static void SetUpTestCase()	{
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "MatmulReduceScatterV2AclnnTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "MatmulReduceScatterV2AclnnTest TearDown" << endl;
    }
};

TEST_F(MatmulReduceScatterV2AclnnTest, basic)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x1_scale = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_scale = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc quant_scale = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnMatmulReduceScatterV2,
        INPUT(x1, x2, bias, x1_scale, x2_scale, quant_scale, 0, "test_group", "sum", 8, 1, 0, "aicpu"),
        OUTPUT(output, nullptr)
    );
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(MatmulReduceScatterV2AclnnTest, basic2)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x1_scale = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_scale = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc quant_scale = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnMatmulReduceScatterV2,
        INPUT(x1, x2, bias, x1_scale, x2_scale, quant_scale, 0, "test_group", "sum", 8, 1, 0, "aicpu"),
        OUTPUT(output, nullptr)
    );
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(MatmulReduceScatterV2AclnnTest, 3scale)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x1_scale = TensorDesc({16, 32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_scale = TensorDesc({32, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc quant_scale = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnMatmulReduceScatterV2,
        INPUT(x1, x2, bias, x1_scale, x2_scale, quant_scale, 0, "test_group", "sum", 8, 1, 0, "aicpu"),
        OUTPUT(output, nullptr)
    );
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(MatmulReduceScatterV2AclnnTest, empty_tensor)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOut = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnMatmulReduceScatterV2,
        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "aicpu"),
        OUTPUT(output, amaxOut)
    );
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(MatmulReduceScatterV2AclnnTest, fp16)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMatmulReduceScatterV2,
        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "aicpu"),
        OUTPUT(output, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(MatmulReduceScatterV2AclnnTest, 16bit)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMatmulReduceScatterV2,
        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "aicpu"),
        OUTPUT(output, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(MatmulReduceScatterV2AclnnTest, 16bit_trans)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_BF16, ACL_FORMAT_ND, {1, 256});
    TensorDesc bias = TensorDesc({256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({256, 256}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMatmulReduceScatterV2,
        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "aicpu"),
        OUTPUT(output, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(MatmulReduceScatterV2AclnnTest, e4m3fn_not_support)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x1_scale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2_scale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc quant_scale = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOut = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMatmulReduceScatterV2,
        INPUT(x1, x2, nullptr, x1_scale, x2_scale, quant_scale, 0, "test_group", "sum", 8, 1, 549764202624, "aicpu"),
        OUTPUT(output, amaxOut)
    );
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

} // namespace