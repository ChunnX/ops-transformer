/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include "../../../../op_host/op_api/aclnn_mhc_pre.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class MhcPreOpapiUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        cout << "MhcPreOpapiUt SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "MhcPreOpapiUt TearDown" << endl;
    }
};

namespace {
constexpr int64_t B = 2;
constexpr int64_t S = 4;
constexpr int64_t N = 4;
constexpr int64_t D = 128;
constexpr int64_t PHI_ROWS = N * N + 2 * N;
constexpr int64_t ND = N * D;
constexpr double NORM_EPS = 1e-6;
constexpr double HC_EPS = 1e-6;

aclnnStatus RunMhcPre(TensorDesc &x, TensorDesc &phi, TensorDesc &alpha, TensorDesc &bias, TensorDesc &gamma,
                      TensorDesc &hIn, TensorDesc &hPost, TensorDesc &hRes, TensorDesc &invRms, TensorDesc &hMix,
                      TensorDesc &hPre)
{
    auto ut = OP_API_UT(aclnnMhcPre, INPUT(x, phi, alpha, bias, gamma, NORM_EPS, HC_EPS),
                        OUTPUT(hIn, hPost, hRes, invRms, hMix, hPre));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    return ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
}

aclnnStatus RunMhcPreNoOptional(TensorDesc &x, TensorDesc &phi, TensorDesc &alpha, TensorDesc &bias, TensorDesc &gamma,
                                TensorDesc &hIn, TensorDesc &hPost, TensorDesc &hRes)
{
    auto ut = OP_API_UT(aclnnMhcPre, INPUT(x, phi, alpha, bias, gamma, NORM_EPS, HC_EPS),
                        OUTPUT(hIn, hPost, hRes, nullptr, nullptr, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    return ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
}

aclnnStatus RunMhcPrePartialOptional(TensorDesc &x, TensorDesc &phi, TensorDesc &alpha, TensorDesc &bias,
                                     TensorDesc &gamma, TensorDesc &hIn, TensorDesc &hPost, TensorDesc &hRes,
                                     TensorDesc &invRms)
{
    auto xAcl = x.ToAclType();
    auto phiAcl = phi.ToAclType();
    auto alphaAcl = alpha.ToAclType();
    auto biasAcl = bias.ToAclType();
    auto gammaAcl = gamma.ToAclType();
    auto hInAcl = hIn.ToAclType();
    auto hPostAcl = hPost.ToAclType();
    auto hResAcl = hRes.ToAclType();
    auto invRmsAcl = invRms.ToAclType();
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    return aclnnMhcPreGetWorkspaceSize(xAcl.get(), phiAcl.get(), alphaAcl.get(), biasAcl.get(), gammaAcl.get(),
                                       NORM_EPS, HC_EPS, hInAcl.get(), hPostAcl.get(), hResAcl.get(), invRmsAcl.get(),
                                       nullptr, nullptr, &workspaceSize, &executor);
}
} // namespace

TEST_F(MhcPreOpapiUt, aclnn_mhc_pre_valid_4d_fp16_with_optional_outputs)
{
    auto x = TensorDesc({B, S, N, D}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({PHI_ROWS, ND}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto bias = TensorDesc({PHI_ROWS}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({N, D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hIn = TensorDesc({B, S, D}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hPost = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hRes = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto invRms = TensorDesc({B, S}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hMix = TensorDesc({B, S, PHI_ROWS}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hPre = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    EXPECT_NE(RunMhcPre(x, phi, alpha, bias, gamma, hIn, hPost, hRes, invRms, hMix, hPre), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPreOpapiUt, aclnn_mhc_pre_invalid_input_dtype_x)
{
    auto x = TensorDesc({B, S, N, D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({PHI_ROWS, ND}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto bias = TensorDesc({PHI_ROWS}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({N, D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hIn = TensorDesc({B, S, D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hPost = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hRes = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    EXPECT_EQ(RunMhcPreNoOptional(x, phi, alpha, bias, gamma, hIn, hPost, hRes), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPreOpapiUt, aclnn_mhc_pre_invalid_input_shape_phi)
{
    auto x = TensorDesc({B, S, N, D}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({PHI_ROWS - 1, ND}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto bias = TensorDesc({PHI_ROWS}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({N, D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hIn = TensorDesc({B, S, D}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hPost = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hRes = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    EXPECT_EQ(RunMhcPreNoOptional(x, phi, alpha, bias, gamma, hIn, hPost, hRes), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPreOpapiUt, aclnn_mhc_pre_invalid_output_dtype_hin)
{
    auto x = TensorDesc({B, S, N, D}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({PHI_ROWS, ND}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto bias = TensorDesc({PHI_ROWS}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({N, D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hIn = TensorDesc({B, S, D}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hPost = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hRes = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    EXPECT_EQ(RunMhcPreNoOptional(x, phi, alpha, bias, gamma, hIn, hPost, hRes), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPreOpapiUt, aclnn_mhc_pre_invalid_output_shape_hpost)
{
    auto x = TensorDesc({B, S, N, D}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({PHI_ROWS, ND}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto bias = TensorDesc({PHI_ROWS}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({N, D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hIn = TensorDesc({B, S, D}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hPost = TensorDesc({B, S, N + 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hRes = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    EXPECT_EQ(RunMhcPreNoOptional(x, phi, alpha, bias, gamma, hIn, hPost, hRes), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPreOpapiUt, aclnn_mhc_pre_invalid_optional_output_shape_hmix)
{
    auto x = TensorDesc({B, S, N, D}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({PHI_ROWS, ND}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto bias = TensorDesc({PHI_ROWS}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({N, D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hIn = TensorDesc({B, S, D}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hPost = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hRes = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto invRms = TensorDesc({B, S}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hMix = TensorDesc({B, S, PHI_ROWS + 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hPre = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    EXPECT_EQ(RunMhcPre(x, phi, alpha, bias, gamma, hIn, hPost, hRes, invRms, hMix, hPre), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPreOpapiUt, aclnn_mhc_pre_invalid_partial_optional_outputs)
{
    auto x = TensorDesc({B, S, N, D}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({PHI_ROWS, ND}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto bias = TensorDesc({PHI_ROWS}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({N, D}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hIn = TensorDesc({B, S, D}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hPost = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hRes = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto invRms = TensorDesc({B, S}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    EXPECT_EQ(RunMhcPrePartialOptional(x, phi, alpha, bias, gamma, hIn, hPost, hRes, invRms), ACLNN_ERR_PARAM_INVALID);
}
