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
#include <vector>

#include "gtest/gtest.h"

#include "../../../../common/op_api/gmm_tensor_storage_check.h"
#include "../../../op_api/aclnn_grouped_matmul_v5.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

namespace {

constexpr int64_t MAX_DIM = std::numeric_limits<int64_t>::max();

bool CheckBounds(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides, int64_t offset,
                 const std::vector<int64_t>& storageShape, aclDataType dtype = ACL_FLOAT,
                 aclFormat format = ACL_FORMAT_ND)
{
    auto tensor = TensorDesc(shape, dtype, format, strides, offset, storageShape).ToAclType();
    return gmm::CheckTensorStorageBounds(tensor.get(), "testTensor");
}

TEST(GmmTensorStorageBounds, TransposeAndOffsetAtStorageBoundary)
{
    EXPECT_TRUE(CheckBounds({3, 2}, {1, 3}, 0, {2, 3}, ACL_HIFLOAT8));
    // T92 and T94: the last accessed element is 6 or 8, outside the six-element allocation.
    EXPECT_FALSE(CheckBounds({3, 2}, {1, 3}, 1, {2, 3}, ACL_HIFLOAT8));
    EXPECT_FALSE(CheckBounds({3, 2}, {1, 3}, 3, {2, 3}, ACL_HIFLOAT8));
    EXPECT_TRUE(CheckBounds({3, 2}, {1, 3}, 3, {3, 3}, ACL_HIFLOAT8));
}

TEST(GmmTensorStorageBounds, AcceptsSlicesBroadcastAndSingletonStrides)
{
    EXPECT_TRUE(CheckBounds({2, 3}, {8, 2}, 3, {16}));
    EXPECT_FALSE(CheckBounds({2, 3}, {8, 2}, 4, {16}));
    EXPECT_TRUE(CheckBounds({100, 3}, {0, 1}, 0, {3}));
    EXPECT_TRUE(CheckBounds({1, 3}, {MAX_DIM, 1}, 0, {3}));
}

TEST(GmmTensorStorageBounds, ScalarOccupiesOneElement)
{
    EXPECT_TRUE(CheckBounds({}, {}, 0, {}));
    EXPECT_TRUE(CheckBounds({}, {}, 2, {3}));
    EXPECT_FALSE(CheckBounds({}, {}, 3, {3}));
    EXPECT_FALSE(CheckBounds({}, {}, 0, {0}));
}

TEST(GmmTensorStorageBounds, EmptyViewsDoNotAccessStorage)
{
    EXPECT_TRUE(CheckBounds({0, 3}, {3, 1}, 0, {0}));
    EXPECT_TRUE(CheckBounds({2, 0}, {MAX_DIM, 1}, MAX_DIM, {0}));
    // A zero storage dimension must be found before multiplying the nonzero dimensions.
    EXPECT_TRUE(CheckBounds({0}, {1}, 0, {MAX_DIM, MAX_DIM, 0}));
    EXPECT_FALSE(CheckBounds({1}, {1}, 0, {MAX_DIM, MAX_DIM, 0}));
}

TEST(GmmTensorStorageBounds, RejectsNegativeMetadataEvenForEmptyViews)
{
    EXPECT_FALSE(CheckBounds({2, 3}, {3, 1}, -1, {6}));
    EXPECT_FALSE(CheckBounds({2, 3}, {-3, 1}, 3, {6}));
    EXPECT_FALSE(CheckBounds({0, -1}, {1, 1}, 0, {0}));
    EXPECT_FALSE(CheckBounds({0, 3}, {3, -1}, 0, {0}));
    EXPECT_FALSE(CheckBounds({0}, {1}, 0, {0, -1}));
    EXPECT_FALSE(CheckBounds({0}, {1}, -1, {0}));
}

TEST(GmmTensorStorageBounds, RejectsShapeStrideRankMismatch)
{
    auto tensor = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    tensor->SetViewStrides(op::Strides{1});
    EXPECT_FALSE(gmm::CheckTensorStorageBounds(tensor.get(), "rankMismatch"));
    tensor->SetViewStrides(op::Strides{3, 1, 1});
    EXPECT_FALSE(gmm::CheckTensorStorageBounds(tensor.get(), "rankMismatch"));
}

TEST(GmmTensorStorageBounds, RejectsStorageProductAndAddressOverflow)
{
    EXPECT_FALSE(CheckBounds({1}, {1}, 0, {MAX_DIM, 2}));
    EXPECT_FALSE(CheckBounds({3}, {MAX_DIM}, 0, {MAX_DIM}));
    EXPECT_FALSE(CheckBounds({2, 2}, {MAX_DIM - 1, 2}, 0, {MAX_DIM}));
    EXPECT_FALSE(CheckBounds({2}, {1}, MAX_DIM, {MAX_DIM}));
    EXPECT_TRUE(CheckBounds({2}, {1}, MAX_DIM - 2, {MAX_DIM}));
    EXPECT_FALSE(CheckBounds({2}, {1}, MAX_DIM - 1, {MAX_DIM}));
}

TEST(GmmTensorStorageBounds, FractalNzUsesPhysicalStorageCapacity)
{
    EXPECT_TRUE(CheckBounds({17, 33}, {33, 1}, 0, {3, 2, 16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ));
    EXPECT_FALSE(CheckBounds({17, 33}, {33, 1}, 0, {1, 1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ));
}

TEST(GmmTensorStorageBounds, Int4AndPackedInt32UseDeclaredElementUnits)
{
    EXPECT_TRUE(CheckBounds({8}, {1}, 0, {8}, ACL_INT4));
    EXPECT_TRUE(CheckBounds({7}, {1}, 1, {8}, ACL_INT4));
    EXPECT_FALSE(CheckBounds({8}, {1}, 1, {8}, ACL_INT4));
    // Each INT32 is a carrier for eight INT4 values, but the incoming metadata counts INT32 elements.
    EXPECT_TRUE(CheckBounds({2, 4}, {4, 1}, 0, {2, 4}, ACL_INT32));
    EXPECT_TRUE(CheckBounds({2, 3}, {4, 1}, 1, {2, 4}, ACL_INT32));
    EXPECT_FALSE(CheckBounds({2, 4}, {4, 1}, 1, {2, 4}, ACL_INT32));
}

TEST(GmmTensorStorageBounds, ChecksEveryTensorListMember)
{
    auto tensors = TensorListDesc({TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND),
                                   TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND, {1, 3}, 1, {2, 3})})
                       .ToAclType();
    EXPECT_FALSE(gmm::CheckTensorListStorageBounds(tensors.get(), "tensorList"));
}

TEST(GmmTensorStorageBounds, LeavesNullAndEmptyHandlingToOperatorChecks)
{
    EXPECT_TRUE(gmm::CheckTensorStorageBounds(nullptr, "optionalTensor"));
    EXPECT_TRUE(gmm::CheckTensorListStorageBounds(nullptr, "optionalList"));
    const aclTensor* nullTensor = nullptr;
    std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> nullMember(aclCreateTensorList(&nullTensor, 1),
                                                                               aclDestroyTensorList);
    ASSERT_NE(nullMember.get(), nullptr);
    EXPECT_TRUE(gmm::CheckTensorListStorageBounds(nullMember.get(), "nullMember"));
    std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> empty(aclCreateTensorList(nullptr, 0),
                                                                          aclDestroyTensorList);
    ASSERT_NE(empty.get(), nullptr);
    EXPECT_TRUE(gmm::CheckTensorListStorageBounds(empty.get(), "emptyList"));
}

enum class ListArgument {
    X,
    WEIGHT,
    BIAS,
    SCALE,
    OFFSET,
    ANTIQUANT_SCALE,
    ANTIQUANT_OFFSET,
    PER_TOKEN_SCALE,
    ACTIVATION_INPUT,
    ACTIVATION_QUANT_SCALE,
    ACTIVATION_QUANT_OFFSET,
    OUT,
    ACTIVATION_FEATURE_OUT,
    DYN_QUANT_SCALE_OUT
};

TEST(GroupedMatmulStorageBounds, RejectsEveryOutOfBoundsListBeforeWorkspaceConstruction)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    for (int argument = static_cast<int>(ListArgument::X);
         argument <= static_cast<int>(ListArgument::DYN_QUANT_SCALE_OUT); ++argument) {
        SCOPED_TRACE(argument);
        auto x = TensorListDesc(1, TensorDesc({16, 4}, ACL_INT32, ACL_FORMAT_ND)).ToAclType();
        auto weight = TensorListDesc(1, TensorDesc({32, 2}, ACL_INT32, ACL_FORMAT_ND)).ToAclType();
        auto out = TensorListDesc(1, TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND)).ToAclType();
        auto bad = TensorListDesc({TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND),
                                   TensorDesc({3, 2}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 3}, 1, {2, 3})})
                       .ToAclType();
        const auto choose = [&](ListArgument slot, const aclTensorList* normal = nullptr) {
            return argument == static_cast<int>(slot) ? bad.get() : normal;
        };
        uint64_t workspaceSize = 12345;
        aclOpExecutor* executor = nullptr;
        const auto ret = aclnnGroupedMatmulV5GetWorkspaceSize(
            choose(ListArgument::X, x.get()), choose(ListArgument::WEIGHT, weight.get()), choose(ListArgument::BIAS),
            choose(ListArgument::SCALE), choose(ListArgument::OFFSET), choose(ListArgument::ANTIQUANT_SCALE),
            choose(ListArgument::ANTIQUANT_OFFSET), choose(ListArgument::PER_TOKEN_SCALE), nullptr,
            choose(ListArgument::ACTIVATION_INPUT), choose(ListArgument::ACTIVATION_QUANT_SCALE),
            choose(ListArgument::ACTIVATION_QUANT_OFFSET), 0, -1, 0, 0, nullptr,
            const_cast<aclTensorList*>(choose(ListArgument::OUT, out.get())),
            const_cast<aclTensorList*>(choose(ListArgument::ACTIVATION_FEATURE_OUT)),
            const_cast<aclTensorList*>(choose(ListArgument::DYN_QUANT_SCALE_OUT)), &workspaceSize, &executor);
        EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
        EXPECT_EQ(workspaceSize, 12345U);
        EXPECT_EQ(executor, nullptr);
        // V5 unpacks these descriptors before other semantic checks. Keeping them intact
        // proves that failure came from the entry guard, even for reserved auxiliary arguments.
        EXPECT_EQ((*x)[0]->GetDataType(), op::DataType::DT_INT32);
        EXPECT_EQ((*x)[0]->GetViewShape().GetDim(1), 4);
        EXPECT_EQ((*weight)[0]->GetDataType(), op::DataType::DT_INT32);
        EXPECT_EQ((*weight)[0]->GetViewShape().GetDim(1), 2);
        if (executor != nullptr) {
            aclDestroyAclOpExecutor(executor);
        }
    }
}

TEST(GroupedMatmulStorageBounds, RejectsOutOfBoundsGroupList)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto x = TensorListDesc(1, TensorDesc({16, 4}, ACL_INT32, ACL_FORMAT_ND)).ToAclType();
    auto weight = TensorListDesc(1, TensorDesc({32, 2}, ACL_INT32, ACL_FORMAT_ND)).ToAclType();
    auto out = TensorListDesc(1, TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND)).ToAclType();
    auto groupList = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND, {1}, 1, {1}).ToAclType();
    uint64_t workspaceSize = 12345;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(aclnnGroupedMatmulV5GetWorkspaceSize(x.get(), weight.get(), nullptr, nullptr, nullptr, nullptr, nullptr,
                                                   nullptr, groupList.get(), nullptr, nullptr, nullptr, 0, -1, 0, 0,
                                                   nullptr, out.get(), nullptr, nullptr, &workspaceSize, &executor),
              ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspaceSize, 12345U);
    EXPECT_EQ(executor, nullptr);
    EXPECT_EQ((*x)[0]->GetDataType(), op::DataType::DT_INT32);
    EXPECT_EQ((*x)[0]->GetViewShape().GetDim(1), 4);
    EXPECT_EQ((*weight)[0]->GetDataType(), op::DataType::DT_INT32);
    EXPECT_EQ((*weight)[0]->GetViewShape().GetDim(1), 2);
    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }
}

TEST(GroupedMatmulStorageBounds, RequiredListNullptrReturnCodesArePreserved)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    for (int index = 0; index < 3; ++index) {
        SCOPED_TRACE(index);
        auto x = TensorListDesc(1, TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND)).ToAclType();
        auto weight = TensorListDesc(1, TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND)).ToAclType();
        auto out = TensorListDesc(1, TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND)).ToAclType();
        uint64_t workspaceSize = 12345;
        aclOpExecutor* executor = nullptr;
        EXPECT_EQ(aclnnGroupedMatmulV5GetWorkspaceSize(
                      index == 0 ? nullptr : x.get(), index == 1 ? nullptr : weight.get(), nullptr, nullptr, nullptr,
                      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, -1, 0, 0, nullptr,
                      index == 2 ? nullptr : out.get(), nullptr, nullptr, &workspaceSize, &executor),
                  ACLNN_ERR_PARAM_NULLPTR);
        EXPECT_EQ(workspaceSize, 12345U);
        EXPECT_EQ(executor, nullptr);
        if (executor != nullptr) {
            aclDestroyAclOpExecutor(executor);
        }
    }
}

} // namespace
