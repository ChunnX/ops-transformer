/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <limits>
#include <memory>

#include "gtest/gtest.h"
#include "../../../op_api/aclnn_grouped_matmul_swiglu_quant_v2.h"
#include "../../../op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"
#include "../../../op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.h"
#include "../../../../common/op_api/gmm_tensor_storage_check.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

namespace {
using TensorPtr = std::unique_ptr<aclTensor, void (*)(aclTensor *)>;
using TensorListPtr = std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)>;
enum class StorageApi {
    ND_V2,
    NZ_V2,
    NZ_V3
};

struct StorageInputs {
    TensorPtr x = TensorDesc({32, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ToAclType();
    TensorPtr weight;
    TensorPtr weightScale = TensorDesc({1, 2, 128, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ToAclType();
    TensorPtr xScale = TensorDesc({32, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ToAclType();
    TensorPtr groupList = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).ToAclType();
    TensorPtr output = TensorDesc({32, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ToAclType();
    TensorPtr outputScale = TensorDesc({32, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ToAclType();

    explicit StorageInputs(StorageApi api)
        : weight(api == StorageApi::ND_V2 ? TensorDesc({1, 128, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ToAclType() :
                                            TensorDesc({1, 128, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_FRACTAL_NZ,
                                                       {16384, 128, 1}, 0, {1, 4, 8, 16, 32})
                                                .ToAclType())
    {}

    std::array<aclTensor *, 7> RequiredTensors()
    {
        return {x.get(),         weight.get(), weightScale.get(), xScale.get(),
                groupList.get(), output.get(), outputScale.get()};
    }
};

void ExpectWorkspaceStatus(StorageApi api, StorageInputs &inputs, aclnnStatus expected,
                           const aclTensorList *assistance = nullptr, const aclTensor *bias = nullptr,
                           const aclTensor *smoothScale = nullptr, bool nullX = false)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    const aclTensor *weightElements[] = {inputs.weight.get()};
    const aclTensor *scaleElements[] = {inputs.weightScale.get()};
    TensorListPtr weights(aclCreateTensorList(weightElements, 1), aclDestroyTensorList);
    ASSERT_NE(weights.get(), nullptr);
    (void)inputs.weight.release();
    TensorListPtr scales(aclCreateTensorList(scaleElements, 1), aclDestroyTensorList);
    ASSERT_NE(scales.get(), nullptr);
    (void)inputs.weightScale.release();
    constexpr uint64_t WORKSPACE_SENTINEL = 12345;
    uint64_t workspaceSize = WORKSPACE_SENTINEL;
    aclOpExecutor *executor = nullptr;
    const auto *x = nullX ? nullptr : inputs.x.get();
    aclnnStatus status;
    if (api == StorageApi::NZ_V3) {
        status = aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
            x, weights.get(), scales.get(), assistance, bias, inputs.xScale.get(), smoothScale, inputs.groupList.get(),
            2, 0, 2, 0, nullptr, 2, 7.0, 1.702, 1.0, "rint", 0, 0.0, inputs.output.get(), inputs.outputScale.get(),
            &workspaceSize, &executor);
    } else {
        auto function = api == StorageApi::ND_V2 ? aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize :
                                                   aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize;
        status = function(x, weights.get(), scales.get(), assistance, bias, inputs.xScale.get(), smoothScale,
                          inputs.groupList.get(), 2, 0, 2, 0, nullptr, inputs.output.get(), inputs.outputScale.get(),
                          &workspaceSize, &executor);
    }
    EXPECT_EQ(status, expected);
    if (expected != ACLNN_SUCCESS) {
        EXPECT_EQ(workspaceSize, WORKSPACE_SENTINEL);
        EXPECT_EQ(executor, nullptr);
    }
    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }
}

class GmmsqTensorStorageTest : public testing::TestWithParam<StorageApi> {};

TEST_P(GmmsqTensorStorageTest, RejectsShiftedFullStorageForEveryRequiredTensor)
{
    constexpr size_t REQUIRED_TENSOR_COUNT = 7;
    for (size_t index = 0; index < REQUIRED_TENSOR_COUNT; ++index) {
        SCOPED_TRACE(index);
        StorageInputs inputs(GetParam());
        inputs.RequiredTensors()[index]->SetViewOffset(1);
        ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_P(GmmsqTensorStorageTest, RejectsNegativeOffset)
{
    StorageInputs inputs(GetParam());
    inputs.x->SetViewOffset(-1);
    ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_INVALID);
}

TEST_P(GmmsqTensorStorageTest, RejectsOffsetAtStorageEnd)
{
    StorageInputs inputs(GetParam());
    inputs.x->SetViewOffset(inputs.x->GetStorageShape().GetShapeSize());
    ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_INVALID);
}

TEST_P(GmmsqTensorStorageTest, RejectsStrideExtentOverflow)
{
    StorageInputs inputs(GetParam());
    inputs.x->SetViewStrides(op::Strides({std::numeric_limits<int64_t>::max(), 1}));
    ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_INVALID);
}

TEST_P(GmmsqTensorStorageTest, RejectsStorageProductOverflow)
{
    StorageInputs inputs(GetParam());
    inputs.x->SetStorageShape(op::Shape({std::numeric_limits<int64_t>::max(), 2}));
    ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_INVALID);
}

TEST_P(GmmsqTensorStorageTest, RejectsNegativeStrideAndRankMismatch)
{
    {
        StorageInputs inputs(GetParam());
        inputs.x->SetViewStrides(op::Strides({128, -1}));
        ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_INVALID);
    }
    {
        StorageInputs inputs(GetParam());
        inputs.x->SetViewStrides(op::Strides({1}));
        ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_P(GmmsqTensorStorageTest, ChecksOptionalTensorStorage)
{
    for (size_t index = 0; index < 3; ++index) {
        SCOPED_TRACE(index);
        StorageInputs inputs(GetParam());
        auto optional = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
        optional->SetViewOffset(1);
        if (index == 0) {
            const aclTensor *elements[] = {optional.get()};
            TensorListPtr assistance(aclCreateTensorList(elements, 1), aclDestroyTensorList);
            ASSERT_NE(assistance.get(), nullptr);
            (void)optional.release();
            ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_INVALID, assistance.get());
        } else {
            ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_INVALID, nullptr,
                                  index == 1 ? optional.get() : nullptr, index == 2 ? optional.get() : nullptr);
        }
    }
}

TEST_P(GmmsqTensorStorageTest, RequiredNullPointerHasPriorityOverInvalidStorage)
{
    StorageInputs inputs(GetParam());
    inputs.output->SetViewOffset(1);
    ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_NULLPTR, nullptr, nullptr, nullptr, true);
}

TEST_P(GmmsqTensorStorageTest, AcceptsNonzeroOffsetWithinDeclaredStorage)
{
    StorageInputs inputs(GetParam());
    inputs.x->SetStorageShape(op::Shape({33, 128}));
    inputs.x->SetViewOffset(1);
    ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_SUCCESS);
}

TEST_P(GmmsqTensorStorageTest, RejectsInvalidAuxiliaryStorageBeforeEmptyFastPath)
{
    StorageInputs inputs(GetParam());
    inputs.x->SetViewShape(op::Shape({0, 128}));
    inputs.groupList->SetViewOffset(1);
    ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_INVALID);
}

INSTANTIATE_TEST_SUITE_P(AllApis, GmmsqTensorStorageTest,
                         testing::Values(StorageApi::ND_V2, StorageApi::NZ_V2, StorageApi::NZ_V3));

TEST(GmmsqStorageBoundsHelper, EmptyViewDoesNotInventStorageAccess)
{
    auto tensor = TensorDesc({0, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ToAclType();
    EXPECT_TRUE(gmm::CheckTensorStorageBounds(tensor.get(), "GroupedMatmulSwigluQuant", "empty"));
    tensor->SetViewOffset(-1);
    EXPECT_FALSE(gmm::CheckTensorStorageBounds(tensor.get(), "GroupedMatmulSwigluQuant", "empty"));
}

TEST(GmmsqStorageBoundsHelper, BoundaryLastElementIsAccepted)
{
    auto tensor = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND, {2}, 1, {4}).ToAclType();
    EXPECT_TRUE(gmm::CheckTensorStorageBounds(tensor.get(), "GroupedMatmulSwigluQuant", "boundary"));
    tensor->SetViewOffset(2);
    EXPECT_FALSE(gmm::CheckTensorStorageBounds(tensor.get(), "GroupedMatmulSwigluQuant", "boundary"));
}
} // namespace
