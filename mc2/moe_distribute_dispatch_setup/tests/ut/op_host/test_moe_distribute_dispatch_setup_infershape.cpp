/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moe_distribute_dispatch_setup_infershape.cpp
 * \brief op_host infer shape ut
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "aclnn/aclnn_base.h"
#include "aclnn_moe_distribute_dispatch_setup.h"
#include "ops/ops_parser/infer_sequence_format.h"
#include "toolchain/common/acl_mdl.h"

namespace {
class MoeDistributeDispatchSetupInferShapeTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

aclTensor* CreateAclTensor(const std::vector<int64_t>& shape, aclDataType dataType)
{
    auto tensorDesc = aclCreateTensorDesc(dataType, shape.size(), shape.data(), ACL_FORMAT_ND, nullptr);
    if (tensorDesc == nullptr) {
        return nullptr;
    }
    void* deviceData = nullptr;
    auto tensor = aclCreateTensor(tensorDesc, nullptr, nullptr, deviceData);
    if (tensor == nullptr) {
        aclDestroyTensorDesc(tensorDesc);
        return nullptr;
    }
    return tensor;
}

TEST_F(MoeDistributeDispatchSetupInferShapeTest, TeardownCalcOutputSize_QuantMode0_Success)
{
    constexpr int64_t bs = 4;
    constexpr int64_t h = 128;
    constexpr int64_t k = 2;
    constexpr int64_t epWorldSize = 2;
    constexpr int64_t epRankId = 0;
    constexpr int64_t moeExpertNum = 8;
    constexpr int64_t sharedExpertNum = 1;
    constexpr int64_t sharedExpertRankNum = 1;
    constexpr int64_t quantMode = 0;
    constexpr int64_t globalBs = 0;

    auto x = CreateAclTensor({bs, h}, ACL_DATA_TYPE_HALF);
    auto expertIds = CreateAclTensor({bs, k}, ACL_DATA_TYPE_INT32);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(expertIds, nullptr);

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    auto ret = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, nullptr, nullptr, "group_ep", epWorldSize, epRankId, moeExpertNum, 0,
        sharedExpertNum, sharedExpertRankNum, quantMode, globalBs, 0, 0, nullptr,
        tokenMsgSize, expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_EQ(tokenMsgSize, 128);
    EXPECT_EQ(expandIdxOutSize, bs * k);
    EXPECT_GT(assistInfoForCombineOutSize, 0);
    EXPECT_GT(commCmdInfoOutSize, 0);

    aclDestroyTensor(x);
    aclDestroyTensor(expertIds);
}

TEST_F(MoeDistributeDispatchSetupInferShapeTest, TeardownCalcOutputSize_QuantMode1_Success)
{
    constexpr int64_t bs = 4;
    constexpr int64_t h = 128;
    constexpr int64_t k = 2;
    constexpr int64_t epWorldSize = 2;
    constexpr int64_t epRankId = 0;
    constexpr int64_t moeExpertNum = 8;
    constexpr int64_t sharedExpertNum = 1;
    constexpr int64_t sharedExpertRankNum = 1;
    constexpr int64_t quantMode = 1;
    constexpr int64_t globalBs = 0;

    auto x = CreateAclTensor({bs, h}, ACL_DATA_TYPE_HALF);
    auto expertIds = CreateAclTensor({bs, k}, ACL_DATA_TYPE_INT32);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(expertIds, nullptr);

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    auto ret = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, nullptr, nullptr, "group_ep", epWorldSize, epRankId, moeExpertNum, 0,
        sharedExpertNum, sharedExpertRankNum, quantMode, globalBs, 0, 0, nullptr,
        tokenMsgSize, expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_EQ(expandIdxOutSize, bs * k);
    EXPECT_GT(assistInfoForCombineOutSize, 0);
    EXPECT_GT(commCmdInfoOutSize, 0);

    aclDestroyTensor(x);
    aclDestroyTensor(expertIds);
}

TEST_F(MoeDistributeDispatchSetupInferShapeTest, TeardownCalcOutputSize_SharedExpertRankIdLessThanShared_Success)
{
    constexpr int64_t bs = 4;
    constexpr int64_t h = 128;
    constexpr int64_t k = 2;
    constexpr int64_t epWorldSize = 4;
    constexpr int64_t epRankId = 0;
    constexpr int64_t moeExpertNum = 8;
    constexpr int64_t sharedExpertNum = 2;
    constexpr int64_t sharedExpertRankNum = 2;
    constexpr int64_t quantMode = 0;
    constexpr int64_t globalBs = 0;

    auto x = CreateAclTensor({bs, h}, ACL_DATA_TYPE_HALF);
    auto expertIds = CreateAclTensor({bs, k}, ACL_DATA_TYPE_INT32);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(expertIds, nullptr);

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    auto ret = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, nullptr, nullptr, "group_ep", epWorldSize, epRankId, moeExpertNum, 0,
        sharedExpertNum, sharedExpertRankNum, quantMode, globalBs, 0, 0, nullptr,
        tokenMsgSize, expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_GT(assistInfoForCombineOutSize, 0);

    aclDestroyTensor(x);
    aclDestroyTensor(expertIds);
}

TEST_F(MoeDistributeDispatchSetupInferShapeTest, TeardownCalcOutputSize_EpRankIdNegative_Fail)
{
    constexpr int64_t bs = 4;
    constexpr int64_t h = 128;
    constexpr int64_t k = 2;
    constexpr int64_t epWorldSize = 2;
    constexpr int64_t epRankId = -1;
    constexpr int64_t moeExpertNum = 8;
    constexpr int64_t sharedExpertNum = 1;
    constexpr int64_t sharedExpertRankNum = 1;
    constexpr int64_t quantMode = 0;
    constexpr int64_t globalBs = 0;

    auto x = CreateAclTensor({bs, h}, ACL_DATA_TYPE_HALF);
    auto expertIds = CreateAclTensor({bs, k}, ACL_DATA_TYPE_INT32);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(expertIds, nullptr);

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    auto ret = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, nullptr, nullptr, "group_ep", epWorldSize, epRankId, moeExpertNum, 0,
        sharedExpertNum, sharedExpertRankNum, quantMode, globalBs, 0, 0, nullptr,
        tokenMsgSize, expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);

    aclDestroyTensor(x);
    aclDestroyTensor(expertIds);
}

TEST_F(MoeDistributeDispatchSetupInferShapeTest, TeardownCalcOutputSize_SharedExpertRankNumExceedWorldSize_Fail)
{
    constexpr int64_t bs = 4;
    constexpr int64_t h = 128;
    constexpr int64_t k = 2;
    constexpr int64_t epWorldSize = 2;
    constexpr int64_t epRankId = 0;
    constexpr int64_t moeExpertNum = 8;
    constexpr int64_t sharedExpertNum = 1;
    constexpr int64_t sharedExpertRankNum = 3;
    constexpr int64_t quantMode = 0;
    constexpr int64_t globalBs = 0;

    auto x = CreateAclTensor({bs, h}, ACL_DATA_TYPE_HALF);
    auto expertIds = CreateAclTensor({bs, k}, ACL_DATA_TYPE_INT32);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(expertIds, nullptr);

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    auto ret = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, nullptr, nullptr, "group_ep", epWorldSize, epRankId, moeExpertNum, 0,
        sharedExpertNum, sharedExpertRankNum, quantMode, globalBs, 0, 0, nullptr,
        tokenMsgSize, expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);

    aclDestroyTensor(x);
    aclDestroyTensor(expertIds);
}

TEST_F(MoeDistributeDispatchSetupInferShapeTest, TeardownCalcOutputSize_XDimNumNot2_Fail)
{
    constexpr int64_t h = 128;
    constexpr int64_t k = 2;
    constexpr int64_t epWorldSize = 2;
    constexpr int64_t epRankId = 0;
    constexpr int64_t moeExpertNum = 8;
    constexpr int64_t sharedExpertNum = 1;
    constexpr int64_t sharedExpertRankNum = 1;
    constexpr int64_t quantMode = 0;
    constexpr int64_t globalBs = 0;

    auto x = CreateAclTensor({h}, ACL_DATA_TYPE_HALF);
    auto expertIds = CreateAclTensor({k}, ACL_DATA_TYPE_INT32);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(expertIds, nullptr);

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    auto ret = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, nullptr, nullptr, "group_ep", epWorldSize, epRankId, moeExpertNum, 0,
        sharedExpertNum, sharedExpertRankNum, quantMode, globalBs, 0, 0, nullptr,
        tokenMsgSize, expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);

    aclDestroyTensor(x);
    aclDestroyTensor(expertIds);
}

TEST_F(MoeDistributeDispatchSetupInferShapeTest, TeardownCalcOutputSize_WithGlobalBs_Success)
{
    constexpr int64_t bs = 4;
    constexpr int64_t h = 128;
    constexpr int64_t k = 2;
    constexpr int64_t epWorldSize = 2;
    constexpr int64_t epRankId = 1;
    constexpr int64_t moeExpertNum = 8;
    constexpr int64_t sharedExpertNum = 1;
    constexpr int64_t sharedExpertRankNum = 1;
    constexpr int64_t quantMode = 0;
    constexpr int64_t globalBs = 8;

    auto x = CreateAclTensor({bs, h}, ACL_DATA_TYPE_HALF);
    auto expertIds = CreateAclTensor({bs, k}, ACL_DATA_TYPE_INT32);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(expertIds, nullptr);

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    auto ret = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, nullptr, nullptr, "group_ep", epWorldSize, epRankId, moeExpertNum, 0,
        sharedExpertNum, sharedExpertRankNum, quantMode, globalBs, 0, 0, nullptr,
        tokenMsgSize, expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_GT(assistInfoForCombineOutSize, 0);

    aclDestroyTensor(x);
    aclDestroyTensor(expertIds);
}
}