/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>

#include "gtest/gtest.h"

#include "../../../op_api/aclnn_grouped_matmul_activation_weight_nz.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

namespace {
using ops::ut::BuildAclTensorDescFromSpec;

constexpr size_t SINGLE_TENSOR_SIZE = 1UL;
constexpr int64_t MX_DEQUANT_MODE = 2L;
constexpr int64_t FLOAT_DEQUANT_DTYPE = 0L;
constexpr int64_t SWIGLU_ACT_TYPE = 0L;
constexpr int64_t SWIGLU_MODE_NO_CLAMP = 0L;
constexpr int64_t SWIGLU_MODE_FRONT_BACK_HALF = 2L;
constexpr double CLAMP_LIMIT = 7.0;
constexpr double GLU_ALPHA = 1.702;
constexpr double GLU_BIAS = 1.0;

TEST(grouped_matmul_activation_weight_nz_opapi, a4w4_bf16_output_uses_direct_worker_path)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto x = BuildAclTensorDescFromSpec("32:128", "FLOAT4_E2M1", "ND").ToAclType();
    auto xScale = BuildAclTensorDescFromSpec("32:2:2", "FLOAT8_E8M0", "ND").ToAclType();
    auto groupList = BuildAclTensorDescFromSpec("1", "INT64", "ND").ToAclType();
    auto weightTensor =
        BuildAclTensorDescFromSpec("1:128:128#16384:128:1#1:2:8:16:64", "FLOAT4_E2M1", "FRACTAL_NZ").ToAclType();
    auto weightScaleTensor = BuildAclTensorDescFromSpec("1:2:128:2", "FLOAT8_E8M0", "ND").ToAclType();
    auto y = BuildAclTensorDescFromSpec("32:64", "BF16", "ND").ToAclType();

    const aclTensor* weightElements[] = {weightTensor.get()};
    std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> weight(
        aclCreateTensorList(weightElements, SINGLE_TENSOR_SIZE), aclDestroyTensorList);
    ASSERT_NE(weight.get(), nullptr);
    (void)weightTensor.release();
    const aclTensor* weightScaleElements[] = {weightScaleTensor.get()};
    std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> weightScale(
        aclCreateTensorList(weightScaleElements, SINGLE_TENSOR_SIZE), aclDestroyTensorList);
    ASSERT_NE(weightScale.get(), nullptr);
    (void)weightScaleTensor.release();

    uint64_t workspaceSize = 0UL;
    aclOpExecutor* executor = nullptr;
    const aclnnStatus ret = aclnnGroupedMatmulActivationWeightNzGetWorkspaceSize(
        x.get(), xScale.get(), groupList.get(), weight.get(), weightScale.get(), nullptr, MX_DEQUANT_MODE,
        FLOAT_DEQUANT_DTYPE, SWIGLU_ACT_TYPE, false, 0L, nullptr, SWIGLU_MODE_FRONT_BACK_HALF, CLAMP_LIMIT, GLU_ALPHA,
        GLU_BIAS, y.get(), &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    if (executor != nullptr) {
        (void)aclDestroyAclOpExecutor(executor);
    }
}

TEST(grouped_matmul_activation_weight_nz_opapi, a4w4_bf16_output_supports_no_clamp_swiglu)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto x = BuildAclTensorDescFromSpec("32:128", "FLOAT4_E2M1", "ND").ToAclType();
    auto xScale = BuildAclTensorDescFromSpec("32:2:2", "FLOAT8_E8M0", "ND").ToAclType();
    auto groupList = BuildAclTensorDescFromSpec("1", "INT64", "ND").ToAclType();
    auto weightTensor =
        BuildAclTensorDescFromSpec("1:128:128#16384:128:1#1:2:8:16:64", "FLOAT4_E2M1", "FRACTAL_NZ").ToAclType();
    auto weightScaleTensor = BuildAclTensorDescFromSpec("1:2:128:2", "FLOAT8_E8M0", "ND").ToAclType();
    auto y = BuildAclTensorDescFromSpec("32:64", "BF16", "ND").ToAclType();

    const aclTensor* weightElements[] = {weightTensor.get()};
    std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> weight(
        aclCreateTensorList(weightElements, SINGLE_TENSOR_SIZE), aclDestroyTensorList);
    ASSERT_NE(weight.get(), nullptr);
    (void)weightTensor.release();
    const aclTensor* weightScaleElements[] = {weightScaleTensor.get()};
    std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> weightScale(
        aclCreateTensorList(weightScaleElements, SINGLE_TENSOR_SIZE), aclDestroyTensorList);
    ASSERT_NE(weightScale.get(), nullptr);
    (void)weightScaleTensor.release();

    uint64_t workspaceSize = 0UL;
    aclOpExecutor* executor = nullptr;
    const aclnnStatus ret = aclnnGroupedMatmulActivationWeightNzGetWorkspaceSize(
        x.get(), xScale.get(), groupList.get(), weight.get(), weightScale.get(), nullptr, MX_DEQUANT_MODE,
        FLOAT_DEQUANT_DTYPE, SWIGLU_ACT_TYPE, false, 0L, nullptr, SWIGLU_MODE_NO_CLAMP, 0.0, 0.0, 0.0, y.get(),
        &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    if (executor != nullptr) {
        (void)aclDestroyAclOpExecutor(executor);
    }
}
} // namespace
