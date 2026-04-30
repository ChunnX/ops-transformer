
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
#include "../../../op_api/aclnn_ai_infra_kv_rms_norm_rope_cache.h"
#include "opdev/platform.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_ai_infra_kv_rms_norm_rope_cache_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_ai_infra_kv_rms_norm_rope_cache_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_ai_infra_kv_rms_norm_rope_cache_test TearDown" << endl;
    }
};

TEST_F(l2_ai_infra_kv_rms_norm_rope_cache_test, ai_infra_kv_rms_norm_rope_cache_opapi_000)
{
    auto kv = TensorDesc({1, 1, 8, 196}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto cos = TensorDesc({1, 1, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sin = TensorDesc({1, 1, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto index = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND);

    auto kCacheRef = TensorDesc({4, 16, 1, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ckvCacheRef = TensorDesc({4, 16, 1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);

    // 可选输入
    auto kRopeScaleOptional = nullptr;
    auto ckvScaleOptional = nullptr;
    auto kRopeOffsetOptional = nullptr;
    auto ckvOffsetOptional = nullptr;
    auto vOptional = nullptr;

    // 属性
    double epsilon = 1e-6;
    char *cacheModeOptional = "PA";
    char *rotaryModeOptional = "half";
    char *quantModeOptional = "static";
    bool isOutputKv = false;

    // 输出
    auto kRopeOut = TensorDesc({1, 1, 8, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto ckvOut = TensorDesc({1, 1, 8, 128}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAiInfraKvRmsNormRopeCache,
        INPUT(
            kv,
            gamma,
            cos,
            sin,
            index,
            kCacheRef,
            ckvCacheRef,
            kRopeScaleOptional,
            ckvScaleOptional,
            kRopeOffsetOptional,
            ckvOffsetOptional,
            vOptional,
            epsilon,
            cacheModeOptional,
            rotaryModeOptional,
            quantModeOptional,
            isOutputKv),
        OUTPUT(kRopeOut, ckvOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_ai_infra_kv_rms_norm_rope_cache_test, ai_infra_kv_rms_norm_rope_cache_opapi_001)
{
    auto kv = TensorDesc({1, 1, 8, 196}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto cos = TensorDesc({1, 1, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sin = TensorDesc({1, 1, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto index = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND);

    auto kCacheRef = nullptr;
    auto ckvCacheRef = TensorDesc({4, 16, 1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);

    // 可选输入
    auto kRopeScaleOptional = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.001, 1.0);
    auto ckvScaleOptional = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.001, 1.0);
    auto kRopeOffsetOptional = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ckvOffsetOptional = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto vOptional = nullptr;

    // 属性
    double epsilon = 1e-6;
    char *cacheModeOptional = "PA";
    char *rotaryModeOptional = "half";
    char *quantModeOptional = "static";
    bool isOutputKv = false;

    // 输出
    auto kRopeOut = TensorDesc({1, 1, 8, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto ckvOut = TensorDesc({1, 1, 8, 128}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAiInfraKvRmsNormRopeCache,
        INPUT(
            kv,
            gamma,
            cos,
            sin,
            index,
            kCacheRef,
            ckvCacheRef,
            kRopeScaleOptional,
            ckvScaleOptional,
            kRopeOffsetOptional,
            ckvOffsetOptional,
            vOptional,
            epsilon,
            cacheModeOptional,
            rotaryModeOptional,
            quantModeOptional,
            isOutputKv),
        OUTPUT(kRopeOut, ckvOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_ai_infra_kv_rms_norm_rope_cache_test, ai_infra_kv_rms_norm_rope_cache_opapi_002)
{
    auto kv = TensorDesc({1, 1, 8, 196}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto cos = TensorDesc({1, 1, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sin = TensorDesc({1, 1, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto index = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND);

    auto kCacheRef = TensorDesc({4, 16, 1, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ckvCacheRef = TensorDesc({4, 16, 1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);

    // 可选输入
    auto kRopeScaleOptional = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.001, 1.0);
    auto ckvScaleOptional = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.001, 1.0);
    auto kRopeOffsetOptional = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ckvOffsetOptional = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto vOptional = TensorDesc({4, 16, 1, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);

    // 属性
    double epsilon = 1e-6;
    char *cacheModeOptional = "Norm";
    char *rotaryModeOptional = "half";
    char *quantModeOptional = "static";
    bool isOutputKv = false;

    // 输出
    auto kRopeOut = TensorDesc({1, 1, 8, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto ckvOut = TensorDesc({1, 1, 8, 128}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAiInfraKvRmsNormRopeCache,
        INPUT(
            kv,
            gamma,
            cos,
            sin,
            index,
            kCacheRef,
            ckvCacheRef,
            kRopeScaleOptional,
            ckvScaleOptional,
            kRopeOffsetOptional,
            ckvOffsetOptional,
            vOptional,
            epsilon,
            cacheModeOptional,
            rotaryModeOptional,
            quantModeOptional,
            isOutputKv),
        OUTPUT(kRopeOut, ckvOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}