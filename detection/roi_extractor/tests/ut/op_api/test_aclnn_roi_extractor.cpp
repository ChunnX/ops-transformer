/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the LICENSE file in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_roi_extractor.cpp
 * \brief aclnnRoiExtractor L2 parameter-validation UT.
 */
#include <gtest/gtest.h>
#include <vector>
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "../../../op_host/op_api/aclnn_roi_extractor.h"

using namespace op;
using namespace std;

class AclnnRoiExtractorTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        std::cout << "RoiExtractor AclnnRoiExtractorTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "RoiExtractor AclnnRoiExtractorTest TearDown" << std::endl;
    }
};

TEST_F(AclnnRoiExtractorTest, FourLevelsFp32AvgSuccess)
{
    auto feats0 = TensorDesc({2, 64, 200, 336}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto feats1 = TensorDesc({2, 64, 100, 168}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto feats2 = TensorDesc({2, 64, 50, 84}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto feats3 = TensorDesc({2, 64, 25, 42}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto feats = TensorListDesc({feats0, feats1, feats2, feats3});
    auto rois = TensorDesc({8, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 200);
    auto y = TensorDesc({8, 64, 7, 7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scales = FloatArrayDesc(vector<float>{0.25f, 0.125f, 0.0625f, 0.03125f});

    auto ut = OP_API_UT(aclnnRoiExtractor,
                        INPUT(feats, rois, nullptr, 56, 0.0, scales, 7, 7, 0, const_cast<char *>("avg"), true),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AclnnRoiExtractorTest, SingleLevelFp16WithIndexSuccess)
{
    auto feats = TensorListDesc({TensorDesc({1, 32, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1)});
    auto rois = TensorDesc({5, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 60);
    auto index = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto y = TensorDesc({5, 32, 7, 14}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto scales = FloatArrayDesc(vector<float>{0.25f});

    auto ut = OP_API_UT(aclnnRoiExtractor,
                        INPUT(feats, rois, index, 56, 0.0, scales, 7, 14, 2, const_cast<char *>("max"), true),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AclnnRoiExtractorTest, TooManyLevelsRejected)
{
    vector<TensorDesc> levels;
    for (int i = 0; i < 6; i++) {
        levels.push_back(TensorDesc({1, 8, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1));
    }
    auto feats = TensorListDesc(levels);
    auto rois = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 30);
    auto y = TensorDesc({2, 8, 7, 7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scales = FloatArrayDesc(vector<float>(6, 0.25f));

    auto ut = OP_API_UT(aclnnRoiExtractor,
                        INPUT(feats, rois, nullptr, 56, 0.0, scales, 7, 7, 0, const_cast<char *>("avg"), true),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnRoiExtractorTest, BadRoisLastDimRejected)
{
    auto feats = TensorListDesc({TensorDesc({1, 8, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1)});
    auto rois = TensorDesc({2, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 30);
    auto y = TensorDesc({2, 8, 7, 7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scales = FloatArrayDesc(vector<float>{0.25f});

    auto ut = OP_API_UT(aclnnRoiExtractor,
                        INPUT(feats, rois, nullptr, 56, 0.0, scales, 7, 7, 0, const_cast<char *>("avg"), true),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnRoiExtractorTest, Roi4WithoutIndexRejected)
{
    auto feats = TensorListDesc({TensorDesc({1, 8, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1)});
    auto rois = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 30);
    auto y = TensorDesc({2, 8, 7, 7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scales = FloatArrayDesc(vector<float>{0.25f});

    auto ut = OP_API_UT(aclnnRoiExtractor,
                        INPUT(feats, rois, nullptr, 56, 0.0, scales, 7, 7, 0, const_cast<char *>("avg"), true),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnRoiExtractorTest, SpatialScaleLengthMismatchRejected)
{
    auto feats = TensorListDesc({
        TensorDesc({1, 8, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1),
        TensorDesc({1, 8, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1),
    });
    auto rois = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 30);
    auto y = TensorDesc({2, 8, 7, 7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scales = FloatArrayDesc(vector<float>{0.25f, 0.125f, 0.0625f});

    auto ut = OP_API_UT(aclnnRoiExtractor,
                        INPUT(feats, rois, nullptr, 56, 0.0, scales, 7, 7, 0, const_cast<char *>("avg"), true),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnRoiExtractorTest, BadPoolModeRejected)
{
    auto feats = TensorListDesc({TensorDesc({1, 8, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1)});
    auto rois = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 30);
    auto y = TensorDesc({2, 8, 7, 7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scales = FloatArrayDesc(vector<float>{0.25f});

    auto ut = OP_API_UT(aclnnRoiExtractor,
                        INPUT(feats, rois, nullptr, 56, 0.0, scales, 7, 7, 0, const_cast<char *>("median"), true),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnRoiExtractorTest, BadPooledSizeRejected)
{
    auto feats = TensorListDesc({TensorDesc({1, 8, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1)});
    auto rois = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 30);
    auto y = TensorDesc({2, 8, 64, 7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scales = FloatArrayDesc(vector<float>{0.25f});

    auto ut = OP_API_UT(aclnnRoiExtractor,
                        INPUT(feats, rois, nullptr, 56, 0.0, scales, 64, 7, 0, const_cast<char *>("avg"), true),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnRoiExtractorTest, DtypeMismatchRejected)
{
    auto feats = TensorListDesc({TensorDesc({1, 8, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1)});
    auto rois = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 30);
    auto y = TensorDesc({2, 8, 7, 7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scales = FloatArrayDesc(vector<float>{0.25f});

    auto ut = OP_API_UT(aclnnRoiExtractor,
                        INPUT(feats, rois, nullptr, 56, 0.0, scales, 7, 7, 0, const_cast<char *>("avg"), true),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnRoiExtractorTest, EmptyRoisSuccess)
{
    auto feats = TensorListDesc({TensorDesc({2, 8, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1)});
    auto rois = TensorDesc({0, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y = TensorDesc({0, 8, 7, 7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scales = FloatArrayDesc(vector<float>{0.25f});

    auto ut = OP_API_UT(aclnnRoiExtractor,
                        INPUT(feats, rois, nullptr, 56, 0.0, scales, 7, 7, 0, const_cast<char *>("avg"), true),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
