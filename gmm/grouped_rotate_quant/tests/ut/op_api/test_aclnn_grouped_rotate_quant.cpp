/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <limits>
#include <memory>
#include "gtest/gtest.h"
#include "../../../op_api/grouped_rotate_quant_checker.h"
#include "../../../op_api/aclnn_grouped_rotate_quant.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

namespace {
using TensorListPtr = std::unique_ptr<aclTensorList, void (*)(aclTensorList *)>;
using TensorPtr = std::unique_ptr<aclTensor, void (*)(aclTensor *)>;

struct Inputs {
    TensorListPtr x = TensorListDesc({TensorDesc({5, 128}, ACL_FLOAT16, ACL_FORMAT_ND)}).ToAclType();
    TensorListPtr rotation = TensorListDesc({TensorDesc({64, 64}, ACL_FLOAT16, ACL_FORMAT_ND)}).ToAclType();
    TensorListPtr y = TensorListDesc({TensorDesc({5, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND)}).ToAclType();
    TensorListPtr scale = TensorListDesc({TensorDesc({5, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)}).ToAclType();
    TensorPtr groups = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).ToAclType();

    grq::Params Params() const
    {
        return {x.get(), rotation.get(), nullptr, groups.get(), 3, 0, 0, -1,
                nullptr, 0, 0.0, false, nullptr, y.get(), scale.get()};
    }

    void SeparateX()
    {
        x = TensorListDesc({TensorDesc({2, 128}, ACL_FLOAT16, ACL_FORMAT_ND),
                           TensorDesc({3, 128}, ACL_FLOAT16, ACL_FORMAT_ND)}).ToAclType();
    }

    void SeparateY()
    {
        y = TensorListDesc({TensorDesc({2, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND),
                           TensorDesc({3, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND)}).ToAclType();
        scale = TensorListDesc({TensorDesc({2, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND),
                               TensorDesc({3, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)}).ToAclType();
    }
};

TEST(GroupedRotateQuantApi, NormalizesAllFourMLayouts)
{
    for (int layout = 0; layout < 4; ++layout) {
        Inputs inputs;
        if (layout == 0 || layout == 2) {
            inputs.SeparateX();
        }
        if (layout < 2) {
            inputs.SeparateY();
        }
        auto p = inputs.Params();
        p.splitItem = layout < 2 ? 0 : 2;
        p.groupListType = layout % 2;
        EXPECT_EQ(grq::CheckParams(p), ACLNN_SUCCESS);
        EXPECT_EQ(p.splitItem, layout);
        EXPECT_STREQ(p.roundMode, "rint");
    }
}

TEST(GroupedRotateQuantApi, IndependentTensorsAllowDifferentRanksAndWidths)
{
    Inputs inputs;
    inputs.x = TensorListDesc({TensorDesc({2, 128}, ACL_FLOAT16, ACL_FORMAT_ND),
                              TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND)}).ToAclType();
    inputs.y = TensorListDesc({TensorDesc({2, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND),
                              TensorDesc({64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND)}).ToAclType();
    inputs.scale = TensorListDesc({TensorDesc({2, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND),
                                  TensorDesc({1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)}).ToAclType();
    auto p = inputs.Params();
    p.groupType = -1;
    p.groupList = nullptr;
    p.splitItem = 1;
    EXPECT_EQ(grq::CheckParams(p), ACLNN_SUCCESS);
    EXPECT_EQ(p.splitItem, 0);
    p.axis = 1;
    EXPECT_EQ(grq::CheckParams(p), ACLNN_ERR_PARAM_INVALID);
}

TEST(GroupedRotateQuantApi, SeparateInputsCanOmitGroupList)
{
    Inputs inputs;
    inputs.SeparateX();
    auto p = inputs.Params();
    p.groupList = nullptr;
    EXPECT_EQ(grq::CheckParams(p), ACLNN_SUCCESS);
    p.x = nullptr;
    EXPECT_EQ(grq::CheckParams(p), ACLNN_ERR_PARAM_NULLPTR);
}

TEST(GroupedRotateQuantApi, SingleGroupedInputRequiresGroupList)
{
    Inputs inputs;
    auto p = inputs.Params();
    p.groupList = nullptr;
    EXPECT_EQ(grq::CheckParams(p), ACLNN_ERR_PARAM_NULLPTR);
}

TEST(GroupedRotateQuantApi, MixedEmptyAndNonemptyGroupsRemainValid)
{
    Inputs inputs;
    inputs.x = TensorListDesc({TensorDesc({0, 128}, ACL_FLOAT16, ACL_FORMAT_ND),
                              TensorDesc({5, 128}, ACL_FLOAT16, ACL_FORMAT_ND)}).ToAclType();
    inputs.y = TensorListDesc({TensorDesc({0, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND),
                              TensorDesc({5, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND)}).ToAclType();
    inputs.scale = TensorListDesc({TensorDesc({0, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND),
                                  TensorDesc({5, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)}).ToAclType();
    auto p = inputs.Params();
    p.splitItem = 0;
    EXPECT_EQ(grq::CheckParams(p), ACLNN_SUCCESS);
}

TEST(GroupedRotateQuantApi, RejectsInvalidGroupListMetadata)
{
    Inputs inputs;
    inputs.groups = TensorDesc({2, 1}, ACL_INT64, ACL_FORMAT_ND).ToAclType();
    auto p = inputs.Params();
    EXPECT_EQ(grq::CheckParams(p), ACLNN_ERR_PARAM_INVALID);
    inputs.groups = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    p = inputs.Params();
    EXPECT_EQ(grq::CheckParams(p), ACLNN_ERR_PARAM_INVALID);
}

TEST(GroupedRotateQuantApi, RejectsUnsupportedGroupingEvenForEmptyOutputs)
{
    Inputs inputs;
    inputs.x = TensorListDesc({TensorDesc({0, 128}, ACL_FLOAT16, ACL_FORMAT_ND)}).ToAclType();
    inputs.y = TensorListDesc({TensorDesc({0, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND)}).ToAclType();
    inputs.scale = TensorListDesc({TensorDesc({0, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)}).ToAclType();
    auto p = inputs.Params();
    EXPECT_EQ(grq::CheckParams(p), ACLNN_SUCCESS);
    p.groupType = 2;
    EXPECT_EQ(grq::CheckParams(p), ACLNN_ERR_PARAM_INVALID);
    p.groupType = 0;
    p.groupListType = 2;
    EXPECT_EQ(grq::CheckParams(p), ACLNN_ERR_PARAM_INVALID);
}

TEST(GroupedRotateQuantApi, ChecksGroupCountAndOutputListLengths)
{
    Inputs inputs;
    inputs.SeparateY();
    inputs.groups = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND).ToAclType();
    auto p = inputs.Params();
    p.splitItem = 1;
    EXPECT_EQ(grq::CheckParams(p), ACLNN_ERR_PARAM_INVALID);
    inputs.groups = TensorDesc({1025}, ACL_INT64, ACL_FORMAT_ND).ToAclType();
    p = inputs.Params();
    EXPECT_EQ(grq::CheckParams(p), ACLNN_ERR_PARAM_INVALID);
}

TEST(GroupedRotateQuantApi, ChecksTotalAndPerGroupOutputRows)
{
    Inputs inputs;
    inputs.SeparateX();
    inputs.SeparateY();
    inputs.y = TensorListDesc({TensorDesc({3, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND),
                              TensorDesc({2, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND)}).ToAclType();
    auto p = inputs.Params();
    p.splitItem = 0;
    EXPECT_EQ(grq::CheckParams(p), ACLNN_ERR_PARAM_INVALID);
    inputs.y = TensorListDesc({TensorDesc({6, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND)}).ToAclType();
    inputs.scale = TensorListDesc({TensorDesc({6, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)}).ToAclType();
    p = inputs.Params();
    EXPECT_EQ(grq::CheckParams(p), ACLNN_ERR_PARAM_INVALID);
}

TEST(GroupedRotateQuantApi, RotationBatchIsColumnBlocksNotExperts)
{
    Inputs inputs;
    inputs.rotation = TensorListDesc({TensorDesc({4, 32, 32}, ACL_FLOAT16, ACL_FORMAT_ND)}).ToAclType();
    auto p = inputs.Params();
    EXPECT_EQ(grq::CheckParams(p), ACLNN_SUCCESS);
    inputs.rotation = TensorListDesc({TensorDesc({2, 32, 32}, ACL_FLOAT16, ACL_FORMAT_ND)}).ToAclType();
    p = inputs.Params();
    EXPECT_EQ(grq::CheckParams(p), ACLNN_ERR_PARAM_INVALID);
}

TEST(GroupedRotateQuantApi, ChecksRotationDtypeAndSquareShape)
{
    Inputs inputs;
    inputs.rotation = TensorListDesc({TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND)}).ToAclType();
    auto p = inputs.Params();
    EXPECT_EQ(grq::CheckParams(p), ACLNN_ERR_PARAM_INVALID);
    inputs.rotation = TensorListDesc({TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND)}).ToAclType();
    p = inputs.Params();
    EXPECT_EQ(grq::CheckParams(p), ACLNN_ERR_PARAM_INVALID);
}

TEST(GroupedRotateQuantApi, ChecksMxScalePaddingForWidth32)
{
    Inputs inputs;
    inputs.x = TensorListDesc({TensorDesc({5, 32}, ACL_FLOAT16, ACL_FORMAT_ND)}).ToAclType();
    inputs.rotation = TensorListDesc({TensorDesc({32, 32}, ACL_FLOAT16, ACL_FORMAT_ND)}).ToAclType();
    inputs.y = TensorListDesc({TensorDesc({5, 32}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND)}).ToAclType();
    inputs.scale = TensorListDesc({TensorDesc({5, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)}).ToAclType();
    auto p = inputs.Params();
    EXPECT_EQ(grq::CheckParams(p), ACLNN_SUCCESS);
    inputs.scale = TensorListDesc({TensorDesc({5, 1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)}).ToAclType();
    p = inputs.Params();
    EXPECT_EQ(grq::CheckParams(p), ACLNN_ERR_PARAM_INVALID);
}

TEST(GroupedRotateQuantApi, RejectsInvalidFp8QuantizationAttributes)
{
    Inputs inputs;
    auto p = inputs.Params();
    p.roundMode = "round";
    EXPECT_EQ(grq::CheckParams(p), ACLNN_ERR_PARAM_INVALID);
    p.roundMode = "rint";
    p.scaleAlg = 2;
    EXPECT_EQ(grq::CheckParams(p), ACLNN_ERR_PARAM_INVALID);
    p.scaleAlg = 1;
    p.dstTypeMax = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(grq::CheckParams(p), ACLNN_ERR_PARAM_INVALID);
    p.dstTypeMax = 0.0;
    EXPECT_EQ(grq::CheckParams(p), ACLNN_SUCCESS);
}

TEST(GroupedRotateQuantApi, ChecksFp4DynamicRangeAndAlpha)
{
    Inputs inputs;
    inputs.y = TensorListDesc({TensorDesc({5, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND)}).ToAclType();
    auto alpha = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).ToAclType();
    auto p = inputs.Params();
    p.alpha = alpha.get();
    p.scaleAlg = 2;
    p.dstTypeMax = 6.0;
    p.roundMode = "floor";
    EXPECT_EQ(grq::CheckParams(p), ACLNN_SUCCESS);
    p.dstTypeMax = 12.1;
    EXPECT_EQ(grq::CheckParams(p), ACLNN_ERR_PARAM_INVALID);
    p.dstTypeMax = 12.0;
    alpha = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND).ToAclType();
    p.alpha = alpha.get();
    EXPECT_EQ(grq::CheckParams(p), ACLNN_ERR_PARAM_INVALID);
}

TEST(GroupedRotateQuantApi, AcceptsValidStridedViewsAndRejectsOutOfBoundsStorage)
{
    Inputs inputs;
    inputs.y = TensorListDesc({TensorDesc({5, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND,
                                         {256, 2}, 0, {5, 256})}).ToAclType();
    auto p = inputs.Params();
    EXPECT_EQ(grq::CheckParams(p), ACLNN_SUCCESS);
    inputs.y = TensorListDesc({TensorDesc({5, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND,
                                         {128, 1}, 1, {5, 128})}).ToAclType();
    p = inputs.Params();
    EXPECT_EQ(grq::CheckParams(p), ACLNN_ERR_PARAM_INVALID);
}

TEST(GroupedRotateQuantApi, EmptyWorkspacePathPreservesParameterValidation)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    Inputs inputs;
    inputs.x = TensorListDesc({TensorDesc({0, 128}, ACL_FLOAT16, ACL_FORMAT_ND)}).ToAclType();
    inputs.y = TensorListDesc({TensorDesc({0, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND)}).ToAclType();
    inputs.scale = TensorListDesc({TensorDesc({0, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)}).ToAclType();
    uint64_t workspaceSize = 123;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnGroupedRotateQuantGetWorkspaceSize(
        inputs.x.get(), inputs.rotation.get(), nullptr, inputs.groups.get(), 3, 0, 0, -1,
        nullptr, 0, 0.0, false, nullptr, inputs.y.get(), inputs.scale.get(), &workspaceSize, &executor);
    EXPECT_EQ(status, ACLNN_SUCCESS);
    EXPECT_EQ(workspaceSize, 0);
    EXPECT_NE(executor, nullptr);
    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }
    executor = nullptr;
    status = aclnnGroupedRotateQuantGetWorkspaceSize(
        inputs.x.get(), inputs.rotation.get(), nullptr, inputs.groups.get(), 3, 2, 0, -1,
        nullptr, 0, 0.0, false, nullptr, inputs.y.get(), inputs.scale.get(), &workspaceSize, &executor);
    EXPECT_EQ(status, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}
} // namespace
