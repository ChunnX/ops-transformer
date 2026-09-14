/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
#include "../../../../op_host/op_api/aclnn_mhc_sandwich_norm_post_preonly.h"
#include "opdev/platform.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_mhc_sandwich_norm_post_preonly_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_mhc_sandwich_norm_post_preonly_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_mhc_sandwich_norm_post_preonly_test TearDown" << endl;
    }
};

TEST_F(l2_mhc_sandwich_norm_post_preonly_test, mhc_sandwich_norm_post_preonly_T16)
{
    int64_t T = 16;
    int64_t D = 2560;
    int64_t N = 4;

    auto h_out = TensorDesc({T, D}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto residual = TensorDesc({T, N, D}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto h_post = TensorDesc({T, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto h_res = TensorDesc({T, N, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto phi = TensorDesc({N, N * D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto alpha_0 = TensorDesc({N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto bias = TensorDesc({N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto gamma_0 = TensorDesc({D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto gamma_1 = TensorDesc({D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto gamma_2 = TensorDesc({N * D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto norm_eps = 1e-6;
    auto hc_eps = 1e-6;

    auto h_in_prime = TensorDesc({T, D}, ACL_BF16, ACL_FORMAT_ND);
    auto x_2_out = TensorDesc({T, N, D}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMhcSandwichNormPostPreonly,
        INPUT(
            h_out,
            residual,
            h_post,
            h_res,
            phi,
            alpha_0,
            bias,
            gamma_0,
            gamma_1,
            gamma_2,
            norm_eps,
            hc_eps),
        OUTPUT(h_in_prime, x_2_out));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_mhc_sandwich_norm_post_preonly_test, mhc_sandwich_norm_post_preonly_BS16)
{
    int64_t B = 2;
    int64_t S = 8; 
    int64_t D = 2560;
    int64_t N = 4;

    auto h_out = TensorDesc({B, S, D}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto residual = TensorDesc({B, S, N, D}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto h_post = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto h_res = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto phi = TensorDesc({N, N * D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto alpha_0 = TensorDesc({N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto bias = TensorDesc({N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto gamma_0 = TensorDesc({D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto gamma_1 = TensorDesc({D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto gamma_2 = TensorDesc({N * D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto norm_eps = 1e-6;
    auto hc_eps = 1e-6;

    auto h_in_prime = TensorDesc({B, S, D}, ACL_BF16, ACL_FORMAT_ND);
    auto x_2_out = TensorDesc({B, S, N, D}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMhcSandwichNormPostPreonly,
        INPUT(
            h_out,
            residual,
            h_post,
            h_res,
            phi,
            alpha_0,
            bias,
            gamma_0,
            gamma_1,
            gamma_2,
            norm_eps,
            hc_eps),
        OUTPUT(h_in_prime, x_2_out));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_mhc_sandwich_norm_post_preonly_test, mhc_sandwich_norm_post_preonly_v2_T16)
{
    int64_t T = 16;
    int64_t D = 2560;
    int64_t N = 4;

    auto h_out = TensorDesc({T, D}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto residual = TensorDesc({T, N, D}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto h_post = TensorDesc({T, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto h_res = TensorDesc({T, N, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto phi = TensorDesc({N, N * D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto alpha_0 = TensorDesc({N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto bias = TensorDesc({N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto gamma_0 = TensorDesc({D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto gamma_1 = TensorDesc({D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto gamma_2 = TensorDesc({N * D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto norm_eps = 1e-6;
    auto hc_eps = 1e-6;
    auto return_h_in_f32 = false;

    auto h_in_prime = TensorDesc({T, D}, ACL_BF16, ACL_FORMAT_ND);
    auto x_2_out = TensorDesc({T, N, D}, ACL_BF16, ACL_FORMAT_ND);
    auto h_in_f32 = TensorDesc({T, D}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMhcSandwichNormPostPreonlyV2,
        INPUT(
            h_out,
            residual,
            h_post,
            h_res,
            phi,
            alpha_0,
            bias,
            gamma_0,
            gamma_1,
            gamma_2,
            norm_eps,
            hc_eps,
            return_h_in_f32),
        OUTPUT(h_in_prime, x_2_out, h_in_f32));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_mhc_sandwich_norm_post_preonly_test, mhc_sandwich_norm_post_preonly_v2_BS16)
{
    int64_t B = 2;
    int64_t S = 8; 
    int64_t D = 5120;
    int64_t N = 4;

    auto h_out = TensorDesc({B, S, D}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto residual = TensorDesc({B, S, N, D}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto h_post = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto h_res = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto phi = TensorDesc({N, N * D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto alpha_0 = TensorDesc({N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto bias = TensorDesc({N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto gamma_0 = TensorDesc({D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto gamma_1 = TensorDesc({D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto gamma_2 = TensorDesc({N * D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto norm_eps = 1e-6;
    auto hc_eps = 1e-6;
    auto return_h_in_f32 = false;

    auto h_in_prime = TensorDesc({B, S, D}, ACL_BF16, ACL_FORMAT_ND);
    auto x_2_out = TensorDesc({B, S, N, D}, ACL_BF16, ACL_FORMAT_ND);
    auto h_in_f32 = TensorDesc({B, S, D}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMhcSandwichNormPostPreonlyV2,
        INPUT(
            h_out,
            residual,
            h_post,
            h_res,
            phi,
            alpha_0,
            bias,
            gamma_0,
            gamma_1,
            gamma_2,
            norm_eps,
            hc_eps,
            return_h_in_f32),
        OUTPUT(h_in_prime, x_2_out, h_in_f32));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}