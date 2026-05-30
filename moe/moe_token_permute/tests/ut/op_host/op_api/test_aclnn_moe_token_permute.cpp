/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_moe_token_permute.h"
#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_moe_token_permute_regbase_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_moe_token_permute_regbase_test SetUp" << endl;
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }

    static void TearDownTestCase()
    {
        cout << "l2_moe_token_permute_regbase_test TearDown" << endl;
    }
};

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_fp16_int32)
{
    auto tokens = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_fp32_int32)
{
    auto tokens = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_bf16_int32)
{
    auto tokens = TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_BF16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_fp16_int64)
{
    auto tokens = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_tokens_nullptr)
{
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(nullptr, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_indices_nullptr)
{
    auto tokens = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, nullptr, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_permute_tokens_out_nullptr)
{
    auto tokens = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(nullptr, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_sorted_indices_out_nullptr)
{
    auto tokens = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_padded_mode_invalid)
{
    auto tokens = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = true;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}