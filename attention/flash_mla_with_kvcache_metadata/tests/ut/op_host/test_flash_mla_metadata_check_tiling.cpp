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
#include <memory>
#include <vector>
#include "aclnn/acl_meta.h"
#include "../../../op_host/flash_mla_with_kvcache_metadata_check.h"

namespace FlashMlaWithKvcacheMetadataUT {
using Checker = FlashMlaWithKvcacheMetadataCheck;

class FlashMlaMetadataHeadTest : public testing::TestWithParam<int> {};
TEST_P(FlashMlaMetadataHeadTest, QueryHeadBounds)
{
    const auto heads = GetParam();
    EXPECT_EQ(Checker::CheckBaseAttr(-1, -1, heads, 1, 576, 512, "TND") == ACLNN_SUCCESS, heads >= 1 && heads <= 128);
}
INSTANTIATE_TEST_SUITE_P(Bounds, FlashMlaMetadataHeadTest, testing::Range(-1, 130));

TEST(FlashMlaMetadataCheckerTest, QueryLayouts)
{
    for (const auto *layout : {"BSND", "BNSD", "TND", "NTD", "BAD", ""}) {
        SCOPED_TRACE(layout);
        const bool valid =
            std::string(layout) == "BSND" || std::string(layout) == "BNSD" || std::string(layout) == "TND";
        EXPECT_EQ(Checker::CheckBaseAttr(-1, -1, 64, 1, 576, 512, layout) == ACLNN_SUCCESS, valid);
    }
}

TEST(FlashMlaMetadataCheckerTest, NullLayoutReturnsError)
{
    EXPECT_EQ(
        Checker::ParamsCheck(nullptr, nullptr, nullptr, -1, -1, 64, 1, 576, 512, 0, -1, -1, nullptr, nullptr, 32, 64),
        ACLNN_ERR_PARAM_NULLPTR);
}

TEST(FlashMlaMetadataCheckerTest, FixedMlaGeometry)
{
    for (int heads : {-1, 0, 2}) {
        EXPECT_NE(Checker::CheckBaseAttr(-1, -1, 64, heads, 576, 512, "TND"), ACLNN_SUCCESS);
    }
    for (int width : {0, 512, 575, 577}) {
        EXPECT_NE(Checker::CheckBaseAttr(-1, -1, 64, 1, width, 512, "TND"), ACLNN_SUCCESS);
    }
    for (int width : {0, 511, 513, 576}) {
        EXPECT_NE(Checker::CheckBaseAttr(-1, -1, 64, 1, 576, width, "TND"), ACLNN_SUCCESS);
    }
}

TEST(FlashMlaMetadataCheckerTest, MaskModes)
{
    for (int mode : {-1, 0, 1, 2, 3, 4, 5}) {
        SCOPED_TRACE(mode);
        EXPECT_EQ(Checker::CheckMask(mode, -1, -1) == ACLNN_SUCCESS, mode == 0 || mode == 3 || mode == 4);
    }
}

TEST(FlashMlaMetadataCheckerTest, BandWindowBounds)
{
    for (int64_t value : {-2L, -1L, 0L, int64_t(INT32_MAX), int64_t(INT32_MAX) + 1}) {
        SCOPED_TRACE(value);
        const bool valid = value >= -1 && value <= INT32_MAX;
        EXPECT_EQ(Checker::CheckMask(4, value, -1) == ACLNN_SUCCESS, valid);
        EXPECT_EQ(Checker::CheckMask(4, -1, value) == ACLNN_SUCCESS, valid);
    }
    for (int mode : {0, 3}) {
        EXPECT_NE(Checker::CheckMask(mode, 0, -1), ACLNN_SUCCESS);
        EXPECT_NE(Checker::CheckMask(mode, -1, 0), ACLNN_SUCCESS);
    }
}

TEST(FlashMlaMetadataCheckerTest, MissingRequiredInputs)
{
    EXPECT_NE(
        Checker::ParamsCheck(nullptr, nullptr, nullptr, -1, -1, 64, 1, 576, 512, 0, -1, -1, "TND", nullptr, 32, 64),
        ACLNN_SUCCESS);
}

TEST(FlashMlaMetadataCheckerTest, ZeroCoreCount)
{
    EXPECT_EQ(
        Checker::ParamsCheck(nullptr, nullptr, nullptr, -1, -1, 64, 1, 576, 512, 0, -1, -1, "TND", nullptr, 0, 64),
        ACLNN_ERR_RUNTIME_ERROR);
    EXPECT_EQ(
        Checker::ParamsCheck(nullptr, nullptr, nullptr, -1, -1, 64, 1, 576, 512, 0, -1, -1, "TND", nullptr, 32, 0),
        ACLNN_ERR_RUNTIME_ERROR);
}

// Descriptor-only ACL tensors backed by host memory; the checker never reads device values.
class FlashMlaMetadataInputsTest : public testing::Test {
protected:
    using TensorPtr = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;
    int32_t storage[4096]{};
    TensorPtr cu{nullptr, aclDestroyTensor};
    TensorPtr cache{nullptr, aclDestroyTensor};
    TensorPtr metadata{nullptr, aclDestroyTensor};

    TensorPtr MakeTensor(std::vector<int64_t> dims, aclDataType dtype = ACL_INT32)
    {
        std::vector<int64_t> strides(dims.size(), 1);
        for (size_t i = dims.size(); i > 1; --i) {
            strides[i - 2] = strides[i - 1] * dims[i - 1];
        }
        return TensorPtr(aclCreateTensor(dims.data(), dims.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, dims.data(),
                                         dims.size(), storage),
                         aclDestroyTensor);
    }

    void SetUp() override
    {
        cu = MakeTensor({3});
        cache = MakeTensor({2});
        metadata = MakeTensor({4096});
        ASSERT_NE(cu.get(), nullptr);
        ASSERT_NE(cache.get(), nullptr);
        ASSERT_NE(metadata.get(), nullptr);
    }

    aclnnStatus Check(const char *layout = "TND", const aclTensor *used = nullptr)
    {
        return Checker::ParamsCheck(cu.get(), cache.get(), used, -1, -1, 64, 1, 576, 512, 4, 0, 0, layout,
                                    metadata.get(), 32, 64);
    }
};

TEST_F(FlashMlaMetadataInputsTest, CompleteValidInputs)
{
    EXPECT_EQ(Check(), ACLNN_SUCCESS);
    auto used = MakeTensor({2});
    ASSERT_NE(used.get(), nullptr);
    EXPECT_EQ(Check("TND", used.get()), ACLNN_SUCCESS);
}

TEST_F(FlashMlaMetadataInputsTest, CuSeqlensPresenceByLayout)
{
    EXPECT_NE(Check("BSND"), ACLNN_SUCCESS);
    EXPECT_NE(Check("BNSD"), ACLNN_SUCCESS);
    cu.reset();
    EXPECT_NE(Check("TND"), ACLNN_SUCCESS);
    EXPECT_EQ(Check("BSND"), ACLNN_SUCCESS);
    EXPECT_EQ(Check("BNSD"), ACLNN_SUCCESS);
}

TEST_F(FlashMlaMetadataInputsTest, MetadataCapacityBoundary)
{
    // (32 cube + 64 vector) cores * 2 batches * 16 words + 16-word header.
    metadata = MakeTensor({3088});
    ASSERT_NE(metadata.get(), nullptr);
    EXPECT_EQ(Check(), ACLNN_SUCCESS);
    metadata = MakeTensor({3087});
    ASSERT_NE(metadata.get(), nullptr);
    EXPECT_NE(Check(), ACLNN_SUCCESS);
}

TEST_F(FlashMlaMetadataInputsTest, MetadataDtypeRankAndEmpty)
{
    metadata = MakeTensor({4096}, ACL_FLOAT);
    ASSERT_NE(metadata.get(), nullptr);
    EXPECT_NE(Check(), ACLNN_SUCCESS);
    for (auto dims : {std::vector<int64_t>{2, 2048}, std::vector<int64_t>{0}}) {
        metadata = MakeTensor(dims);
        ASSERT_NE(metadata.get(), nullptr);
        EXPECT_NE(Check(), ACLNN_SUCCESS);
    }
    metadata.reset();
    EXPECT_NE(Check(), ACLNN_SUCCESS);
}

TEST_F(FlashMlaMetadataInputsTest, SequenceDtypeRankAndLength)
{
    for (auto *tensor : {&cu, &cache}) {
        const int64_t count = tensor == &cu ? 3 : 2;
        *tensor = MakeTensor({count}, ACL_INT64);
        ASSERT_NE(tensor->get(), nullptr);
        EXPECT_NE(Check(), ACLNN_SUCCESS);
        *tensor = MakeTensor({1, count});
        ASSERT_NE(tensor->get(), nullptr);
        EXPECT_NE(Check(), ACLNN_SUCCESS);
        *tensor = MakeTensor({count + 1});
        ASSERT_NE(tensor->get(), nullptr);
        EXPECT_NE(Check(), ACLNN_SUCCESS);
        *tensor = MakeTensor({0});
        ASSERT_NE(tensor->get(), nullptr);
        EXPECT_NE(Check(), ACLNN_SUCCESS);
        *tensor = MakeTensor({count});
        ASSERT_NE(tensor->get(), nullptr);
    }
    auto used = MakeTensor({3});
    ASSERT_NE(used.get(), nullptr);
    EXPECT_NE(Check("TND", used.get()), ACLNN_SUCCESS);
    cache.reset();
    EXPECT_NE(Check(), ACLNN_SUCCESS);
}

} // namespace FlashMlaWithKvcacheMetadataUT
