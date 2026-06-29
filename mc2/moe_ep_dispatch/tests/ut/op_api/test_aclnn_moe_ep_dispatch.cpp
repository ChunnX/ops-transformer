/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_moe_ep_dispatch.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class L2AclnnMoeEpDispatchTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        cout << "L2AclnnMoeEpDispatchTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "L2AclnnMoeEpDispatchTest TearDown" << endl;
    }
};

TEST_F(L2AclnnMoeEpDispatchTest, TestMoeEpDispatchBasic)
{
    int64_t numTokens = 2048;
    int64_t hidden = 7168;
    int64_t topK = 8;
    int64_t epWorldSize = 8;
    int64_t epRankId = 0;
    int64_t numExperts = 64;
    int64_t numMaxTokensPerRank = 2048;
    int64_t cclBufferSize = 200 * 1024 * 1024;
    int64_t expertAlignment = 1;
    int64_t numLocalExperts = numExperts / epWorldSize;
    int64_t upperA = epWorldSize * numMaxTokensPerRank;

    TensorDesc contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xDesc = TensorDesc({numTokens, hidden}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc topkIdxDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc topkWeightsDesc = TensorDesc({numTokens, topK}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc numRecvPerRankDesc = TensorDesc({epWorldSize}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerExpertDesc = TensorDesc({numLocalExperts}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc dstSlotIdxDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeEpDispatch,
                        INPUT(contextDesc, xDesc, topkIdxDesc, topkWeightsDesc, nullptr, nullptr, epWorldSize,
                              epRankId, numExperts, numMaxTokensPerRank, cclBufferSize, expertAlignment, true, 0),
                        OUTPUT(numRecvPerRankDesc, numRecvPerExpertDesc, dstSlotIdxDesc));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(L2AclnnMoeEpDispatchTest, TestMoeEpDispatchNullTopk)
{
    int64_t numTokens = 2048;
    int64_t hidden = 7168;
    int64_t topK = 8;
    int64_t epWorldSize = 8;
    int64_t numExperts = 64;
    int64_t numMaxTokensPerRank = 2048;
    int64_t cclBufferSize = 200 * 1024 * 1024;
    int64_t numLocalExperts = numExperts / epWorldSize;

    TensorDesc contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xDesc = TensorDesc({numTokens, hidden}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc topkWeightsDesc = TensorDesc({numTokens, topK}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeEpDispatch,
                        INPUT(contextDesc, xDesc, nullptr, topkWeightsDesc, nullptr, nullptr,
                              epWorldSize, 0, numExperts, numMaxTokensPerRank, cclBufferSize,
                              1, true, 0),
                        OUTPUT(TensorDesc({epWorldSize}, ACL_INT32, ACL_FORMAT_ND),
                               TensorDesc({numLocalExperts}, ACL_INT64, ACL_FORMAT_ND),
                               TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND)));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2AclnnMoeEpDispatchTest, TestMoeEpDispatchInvalidEpWorldSize)
{
    int64_t numTokens = 2048;
    int64_t hidden = 7168;
    int64_t topK = 8;
    int64_t cclBufferSize = 200 * 1024 * 1024;

    TensorDesc contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xDesc = TensorDesc({numTokens, hidden}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc topkIdxDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc topkWeightsDesc = TensorDesc({numTokens, topK}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeEpDispatch,
                        INPUT(contextDesc, xDesc, topkIdxDesc, topkWeightsDesc, nullptr, nullptr,
                              1, 0, 8, 2048, cclBufferSize, 1, true, 0),
                        OUTPUT(TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND),
                               TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND),
                               TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND)));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnMoeEpDispatchTest, TestMoeEpDispatchCachedMode)
{
    int64_t numTokens = 1024;
    int64_t hidden = 4096;
    int64_t topK = 8;
    int64_t epWorldSize = 4;
    int64_t epRankId = 0;
    int64_t numExperts = 32;
    int64_t numMaxTokensPerRank = 1024;
    int64_t cclBufferSize = 200 * 1024 * 1024;
    int64_t numLocalExperts = numExperts / epWorldSize;

    TensorDesc contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xDesc = TensorDesc({numTokens, hidden}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc topkIdxDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc cachedSlotDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerRankDesc = TensorDesc({epWorldSize}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerExpertDesc = TensorDesc({numLocalExperts}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc dstSlotIdxDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeEpDispatch,
                        INPUT(contextDesc, xDesc, topkIdxDesc, nullptr, nullptr, cachedSlotDesc, epWorldSize,
                              epRankId, numExperts, numMaxTokensPerRank, cclBufferSize, 1, false, 0),
                        OUTPUT(numRecvPerRankDesc, numRecvPerExpertDesc, dstSlotIdxDesc));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(L2AclnnMoeEpDispatchTest, TestMoeEpDispatchFp16Dtype)
{
    int64_t numTokens = 1024;
    int64_t hidden = 4096;
    int64_t topK = 4;
    int64_t epWorldSize = 4;
    int64_t epRankId = 0;
    int64_t numExperts = 16;
    int64_t numMaxTokensPerRank = 1024;
    int64_t cclBufferSize = 200 * 1024 * 1024;
    int64_t numLocalExperts = numExperts / epWorldSize;

    TensorDesc contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xDesc = TensorDesc({numTokens, hidden}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc topkIdxDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc topkWeightsDesc = TensorDesc({numTokens, topK}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc numRecvPerRankDesc = TensorDesc({epWorldSize}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerExpertDesc = TensorDesc({numLocalExperts}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc dstSlotIdxDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeEpDispatch,
                        INPUT(contextDesc, xDesc, topkIdxDesc, topkWeightsDesc, nullptr, nullptr, epWorldSize,
                              epRankId, numExperts, numMaxTokensPerRank, cclBufferSize, 1, true, 0),
                        OUTPUT(numRecvPerRankDesc, numRecvPerExpertDesc, dstSlotIdxDesc));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(L2AclnnMoeEpDispatchTest, TestMoeEpDispatchInvalidNumExperts)
{
    int64_t numTokens = 512;
    int64_t hidden = 4096;
    int64_t topK = 4;
    int64_t epWorldSize = 8;
    int64_t cclBufferSize = 200 * 1024 * 1024;

    TensorDesc contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xDesc = TensorDesc({numTokens, hidden}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc topkIdxDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc topkWeightsDesc = TensorDesc({numTokens, topK}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeEpDispatch,
                        INPUT(contextDesc, xDesc, topkIdxDesc, topkWeightsDesc, nullptr, nullptr, epWorldSize,
                              0, 63, 2048, cclBufferSize, 1, true, 0),
                        OUTPUT(TensorDesc({epWorldSize}, ACL_INT32, ACL_FORMAT_ND),
                               TensorDesc({63 / epWorldSize}, ACL_INT64, ACL_FORMAT_ND),
                               TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND)));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnMoeEpDispatchTest, TestMoeEpDispatchNullX)
{
    int64_t topK = 8;
    int64_t epWorldSize = 8;
    int64_t cclBufferSize = 200 * 1024 * 1024;
    int64_t numExperts = 64;

    TensorDesc contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc topkIdxDesc = TensorDesc({2048, topK}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc topkWeightsDesc = TensorDesc({2048, topK}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeEpDispatch,
                        INPUT(contextDesc, TensorDesc(), topkIdxDesc, topkWeightsDesc, nullptr, nullptr,
                              epWorldSize, 0, numExperts, 2048, cclBufferSize, 1, true, 0),
                        OUTPUT(TensorDesc({epWorldSize}, ACL_INT32, ACL_FORMAT_ND),
                               TensorDesc({numExperts / epWorldSize}, ACL_INT64, ACL_FORMAT_ND),
                               TensorDesc({2048, topK}, ACL_INT32, ACL_FORMAT_ND)));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}