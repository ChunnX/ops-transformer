/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "../../../op_api/aclnn_mhc_pre_backward.h"
#include "opdev/platform.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_ai_mhc_pre_backward_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_ai_mhc_pre_backward_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_ai_mhc_pre_backward_test TearDown" << endl;
    }
};

// ==================== 正常测试用例（成功路径） ====================

// TND格式 BF16测试
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_tnd_bf16_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// BSND格式 FP16测试
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_bsnd_fp16_0)
{
    int64_t b = 2;
    int64_t s = 2;
    int64_t n = 4;
    int64_t d = 8;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({b, s, n, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({b, s, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({b, s, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({b, s}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({b, s, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({b, s, n, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// TND格式 无gamma测试
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_tnd_no_gamma_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, nullptr, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// BSND格式 无gamma测试
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_bsnd_no_gamma_0)
{
    int64_t b = 2;
    int64_t s = 2;
    int64_t n = 4;
    int64_t d = 8;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({b, s, n, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({b, s, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({b, s, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({b, s}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({b, s, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({b, s, n, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, nullptr, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 空指针检查用例（输入参数10个） ====================

// 异常用例: x为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_x_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT((const aclTensor*)nullptr, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: phi为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_phi_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, (const aclTensor*)nullptr, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: alpha为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_alpha_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, (const aclTensor*)nullptr, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: h_in_grad为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_h_in_grad_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, (const aclTensor*)nullptr, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: h_post_grad为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_h_post_grad_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, (const aclTensor*)nullptr, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: h_res_grad为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_h_res_grad_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, (const aclTensor*)nullptr, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: inv_rms为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_inv_rms_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, (const aclTensor*)nullptr, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: mm_res为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_mm_res_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, (const aclTensor*)nullptr, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: h_pre为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_h_pre_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, (const aclTensor*)nullptr, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: h_post为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_h_post_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, (const aclTensor*)nullptr, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 空指针检查用例（输出参数5个） ====================

// 异常用例: x_grad为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_x_grad_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT((const aclTensor*)nullptr, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: hc_weight_grad为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_hc_weight_grad_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, (const aclTensor*)nullptr, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: alpha_grad为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_alpha_grad_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, (const aclTensor*)nullptr, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: bias_post_grad为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_bias_post_grad_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, (const aclTensor*)nullptr, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gamma_grad为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_gamma_grad_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, (const aclTensor*)nullptr));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 空tensor检查用例（输入参数10个） ====================

// 异常用例: x为空tensor
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_x_empty_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({0}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: phi为空tensor
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_phi_empty_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: alpha为空tensor
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_alpha_empty_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: h_in_grad为空tensor
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_h_in_grad_empty_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({0}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: h_post_grad为空tensor
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_h_post_grad_empty_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: h_res_grad为空tensor
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_h_res_grad_empty_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: inv_rms为空tensor
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_inv_rms_empty_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: mm_res为空tensor
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_mm_res_empty_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: h_pre为空tensor
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_h_pre_empty_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: h_post为空tensor
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_h_post_empty_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 维度检查用例 ====================

// 异常用例: x维度错误
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_x_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: x_grad维度错误-TND格式下
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_x_grad_dim_invalid_tnd_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: x_grad维度错误-BSND格式下
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_x_grad_dim_invalid_bsnd_0)
{
    int64_t b = 2;
    int64_t s = 2;
    int64_t n = 4;
    int64_t d = 8;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({b, s, n, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({b, s, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({b, s, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({b, s}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({b, s, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({b, s, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== Dtype检查用例 ====================

// 异常用例: x dtype错误（FP32不支持）
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_x_dtype_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: x_grad dtype不匹配
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_x_grad_dtype_mismatch_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== Format检查用例 ====================

// 异常用例: x格式错误（使用FRACTAL_NZ格式）
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_x_format_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== Shape检查用例 ====================

// 异常用例: alpha形状错误
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_alpha_shape_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: hc_weight_grad形状错误
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_hc_weight_grad_shape_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n + 1, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== Dtype检查用例（补充） ====================

// 异常用例: phi dtype错误（BF16不支持）
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_phi_dtype_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: h_in_grad dtype错误（FP32不支持）
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_h_in_grad_dtype_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: hc_weight_grad dtype错误（BF16不支持）
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_hc_weight_grad_dtype_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: alpha_grad dtype错误（BF16不支持）
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_alpha_grad_dtype_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: bias_post_grad dtype错误（BF16不支持）
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_bias_post_grad_dtype_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gamma_grad dtype错误（BF16不支持）
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_gamma_grad_dtype_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 维度检查用例（补充） ====================

// 异常用例: phi维度错误
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_phi_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({4*4+8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, 4*4+8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({4*4+8, n*d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({4*4+8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gamma维度错误
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_gamma_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: h_post_grad维度错误
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_h_post_grad_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: h_res_grad维度错误
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_h_res_grad_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: mm_res维度错误
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_mm_res_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: h_pre维度错误
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_h_pre_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: h_post维度错误
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_h_post_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: hc_weight_grad维度错误
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_hc_weight_grad_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: alpha_grad维度错误
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_alpha_grad_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: bias_post_grad维度错误
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_bias_post_grad_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gamma_grad维度错误
TEST_F(l2_ai_mhc_pre_backward_test, Ascend910B_mhc_pre_backward_gamma_grad_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto h_in_grad = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post_grad = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_res_grad = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inv_rms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto mm_res = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_pre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto h_post = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto x_grad = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hc_weight_grad = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha_grad = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto bias_post_grad = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma_grad = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma, hcEps),
        OUTPUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}
