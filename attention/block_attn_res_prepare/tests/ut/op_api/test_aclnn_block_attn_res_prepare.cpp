/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \\file test_aclnn_block_attn_res_prepare.cpp
 * \\brief BlockAttnResPrepare ACLNN unit tests.
 */

#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_block_attn_res_prepare.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;

namespace {

constexpr int64_t TOTAL_T = 2;
constexpr int64_t TOTAL_N = 8;
constexpr int64_t TOTAL_S = 3;
constexpr int64_t TOTAL_D = 128;
constexpr float DEFAULT_EPS = 1.0e-6F;

class BlockAttnResPrepareAclnnTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        SetPlatformSocVersion(SocVersion::ASCEND950);
    }
};

// Shape, value-range, and invalid-parameter literals below intentionally define the input and boundary test vectors.

TEST_F(BlockAttnResPrepareAclnnTest, Basic)
{
    auto blockRes = TensorDesc({TOTAL_T, TOTAL_N, TOTAL_D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-0.1, 0.1);
    auto validBlocks = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).Value(std::vector<int64_t>{TOTAL_N});
    auto pseudoQuery = TensorDesc({TOTAL_S, TOTAL_D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-0.1, 0.1);
    auto numerator = TensorDesc({TOTAL_S, TOTAL_T, TOTAL_D}, ACL_FLOAT, ACL_FORMAT_ND);
    auto logitMax = TensorDesc({TOTAL_S, TOTAL_T}, ACL_FLOAT, ACL_FORMAT_ND);
    auto expSum = TensorDesc({TOTAL_S, TOTAL_T}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBlockAttnResPrepare, INPUT(blockRes, validBlocks, pseudoQuery),
                        OUTPUT(numerator, logitMax, expSum), DEFAULT_EPS);
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(BlockAttnResPrepareAclnnTest, RejectsInvalidBlockResDtype)
{
    auto blockRes = TensorDesc({TOTAL_T, TOTAL_N, TOTAL_D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto validBlocks = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto pseudoQuery = TensorDesc({TOTAL_S, TOTAL_D}, ACL_FLOAT, ACL_FORMAT_ND);
    auto numerator = TensorDesc({TOTAL_S, TOTAL_T, TOTAL_D}, ACL_FLOAT, ACL_FORMAT_ND);
    auto logitMax = TensorDesc({TOTAL_S, TOTAL_T}, ACL_FLOAT, ACL_FORMAT_ND);
    auto expSum = TensorDesc({TOTAL_S, TOTAL_T}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBlockAttnResPrepare, INPUT(blockRes, validBlocks, pseudoQuery),
                        OUTPUT(numerator, logitMax, expSum), DEFAULT_EPS);
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(BlockAttnResPrepareAclnnTest, RejectsMismatchedOutputShape)
{
    auto blockRes = TensorDesc({TOTAL_T, TOTAL_N, TOTAL_D}, ACL_FLOAT, ACL_FORMAT_ND);
    auto validBlocks = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto pseudoQuery = TensorDesc({TOTAL_S, TOTAL_D}, ACL_FLOAT, ACL_FORMAT_ND);
    auto numerator = TensorDesc({TOTAL_S, TOTAL_T, TOTAL_D + 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto logitMax = TensorDesc({TOTAL_S, TOTAL_T}, ACL_FLOAT, ACL_FORMAT_ND);
    auto expSum = TensorDesc({TOTAL_S, TOTAL_T}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBlockAttnResPrepare, INPUT(blockRes, validBlocks, pseudoQuery),
                        OUTPUT(numerator, logitMax, expSum), DEFAULT_EPS);
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(BlockAttnResPrepareAclnnTest, RejectsNonPositiveEps)
{
    auto blockRes = TensorDesc({TOTAL_T, TOTAL_N, TOTAL_D}, ACL_FLOAT, ACL_FORMAT_ND);
    auto validBlocks = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto pseudoQuery = TensorDesc({TOTAL_S, TOTAL_D}, ACL_FLOAT, ACL_FORMAT_ND);
    auto numerator = TensorDesc({TOTAL_S, TOTAL_T, TOTAL_D}, ACL_FLOAT, ACL_FORMAT_ND);
    auto logitMax = TensorDesc({TOTAL_S, TOTAL_T}, ACL_FLOAT, ACL_FORMAT_ND);
    auto expSum = TensorDesc({TOTAL_S, TOTAL_T}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBlockAttnResPrepare, INPUT(blockRes, validBlocks, pseudoQuery),
                        OUTPUT(numerator, logitMax, expSum), 0.0F);
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(BlockAttnResPrepareAclnnTest, RejectsNonFiniteEps)
{
    auto blockRes = TensorDesc({TOTAL_T, TOTAL_N, TOTAL_D}, ACL_FLOAT, ACL_FORMAT_ND);
    auto validBlocks = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto pseudoQuery = TensorDesc({TOTAL_S, TOTAL_D}, ACL_FLOAT, ACL_FORMAT_ND);
    auto numerator = TensorDesc({TOTAL_S, TOTAL_T, TOTAL_D}, ACL_FLOAT, ACL_FORMAT_ND);
    auto logitMax = TensorDesc({TOTAL_S, TOTAL_T}, ACL_FLOAT, ACL_FORMAT_ND);
    auto expSum = TensorDesc({TOTAL_S, TOTAL_T}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBlockAttnResPrepare, INPUT(blockRes, validBlocks, pseudoQuery),
                        OUTPUT(numerator, logitMax, expSum), std::numeric_limits<float>::infinity());
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

} // namespace
