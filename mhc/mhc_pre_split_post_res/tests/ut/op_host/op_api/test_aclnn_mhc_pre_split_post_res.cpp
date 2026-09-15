/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_mhc_pre_split_post_res.h"
#include "opdev/platform.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_mhc_pre_split_post_res_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_mhc_pre_split_post_res_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_mhc_pre_split_post_res_test TearDown" << endl;
    }
};

// BSND 布局 [B,S,n,D]，bf16，D=2560（SINGLE_TILE 路径）
TEST_F(l2_mhc_pre_split_post_res_test, mhc_pre_split_post_res_BSND_bf16_2560)
{
    int64_t B = 1;
    int64_t S = 4;
    int64_t D = 2560;
    int64_t N = 4;
    int64_t phiRows = N * N + N;

    auto x = TensorDesc({B, S, N, D}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto phi = TensorDesc({phiRows, N * D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto alpha = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 2);
    auto bias = TensorDesc({phiRows}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-0.5, 0.5);
    auto norm_eps = 1e-6;

    auto h_post_out = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto h_res_out = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMhcPreSplitPostRes,
        INPUT(x, phi, alpha, bias, norm_eps),
        OUTPUT(h_post_out, h_res_out));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// BSND 布局，fp16，D=5120（MULTI_TILE 路径）
TEST_F(l2_mhc_pre_split_post_res_test, mhc_pre_split_post_res_BSND_fp16_5120)
{
    int64_t B = 1;
    int64_t S = 4;
    int64_t D = 5120;
    int64_t N = 4;
    int64_t phiRows = N * N + N;

    auto x = TensorDesc({B, S, N, D}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto phi = TensorDesc({phiRows, N * D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto alpha = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 2);
    auto bias = TensorDesc({phiRows}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-0.5, 0.5);
    auto norm_eps = 1e-6;

    auto h_post_out = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto h_res_out = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMhcPreSplitPostRes,
        INPUT(x, phi, alpha, bias, norm_eps),
        OUTPUT(h_post_out, h_res_out));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// TND 布局 [T,n,D]，bf16，D=2560
TEST_F(l2_mhc_pre_split_post_res_test, mhc_pre_split_post_res_TND_bf16_2560)
{
    int64_t T = 128;
    int64_t D = 2560;
    int64_t N = 4;
    int64_t phiRows = N * N + N;

    auto x = TensorDesc({T, N, D}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto phi = TensorDesc({phiRows, N * D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto alpha = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 2);
    auto bias = TensorDesc({phiRows}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-0.5, 0.5);
    auto norm_eps = 1e-6;

    auto h_post_out = TensorDesc({T, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto h_res_out = TensorDesc({T, N, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMhcPreSplitPostRes,
        INPUT(x, phi, alpha, bias, norm_eps),
        OUTPUT(h_post_out, h_res_out));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// TND 布局，fp16，D=5120，较大 T（走单核/token 分组）
TEST_F(l2_mhc_pre_split_post_res_test, mhc_pre_split_post_res_TND_fp16_5120)
{
    int64_t T = 512;
    int64_t D = 5120;
    int64_t N = 4;
    int64_t phiRows = N * N + N;

    auto x = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto phi = TensorDesc({phiRows, N * D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto alpha = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 2);
    auto bias = TensorDesc({phiRows}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-0.5, 0.5);
    auto norm_eps = 1e-6;

    auto h_post_out = TensorDesc({T, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto h_res_out = TensorDesc({T, N, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMhcPreSplitPostRes,
        INPUT(x, phi, alpha, bias, norm_eps),
        OUTPUT(h_post_out, h_res_out));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}
