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
#include "../../../op_api/aclnn_moe_ep_combine.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class L2AclnnMoeEpCombineTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        cout << "L2AclnnMoeEpCombineTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "L2AclnnMoeEpCombineTest TearDown" << endl;
    }
};

TEST_F(L2AclnnMoeEpCombineTest, TestCombineBasicPureSum)
{
    int64_t numTokens = 2048;
    int64_t hidden = 7168;
    int64_t topK = 8;
    int64_t epWorldSize = 8;
    int64_t numExperts = 64;
    int64_t cclBufferSize = 200 * 1024 * 1024;
    int64_t numLocalExperts = numExperts / epWorldSize;
    int64_t A = epWorldSize * 2048;

    TensorDesc contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xDesc = TensorDesc({A, hidden}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc dstSlotDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc recvSrcMetaDesc = TensorDesc({A, 3}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerRankDesc = TensorDesc({epWorldSize}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc nullDesc;
    TensorDesc combinedXDesc = TensorDesc({numTokens, hidden}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc combinedTopkWeightsDesc;

    auto ut = OP_API_UT(aclnnMoeEpCombine,
                        INPUT(contextDesc, xDesc, dstSlotDesc, recvSrcMetaDesc, numRecvPerRankDesc, nullDesc, nullptr,
                              nullptr, epWorldSize, 0, numExperts, 2048, cclBufferSize),
                        OUTPUT(combinedXDesc, combinedTopkWeightsDesc));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(L2AclnnMoeEpCombineTest, TestCombineWithTopkWeights)
{
    int64_t numTokens = 2048;
    int64_t hidden = 7168;
    int64_t topK = 8;
    int64_t epWorldSize = 8;
    int64_t numExperts = 64;
    int64_t cclBufferSize = 200 * 1024 * 1024;
    int64_t A = epWorldSize * 2048;

    TensorDesc contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xDesc = TensorDesc({A, hidden}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc dstSlotDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc recvSrcMetaDesc = TensorDesc({A, 3}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerRankDesc = TensorDesc({epWorldSize}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc topkWeightsDesc = TensorDesc({numTokens, topK}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc combinedXDesc = TensorDesc({numTokens, hidden}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc combinedTopkWeightsDesc = TensorDesc({numTokens, topK}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeEpCombine,
                        INPUT(contextDesc, xDesc, dstSlotDesc, recvSrcMetaDesc, numRecvPerRankDesc, topkWeightsDesc,
                              nullptr, nullptr, epWorldSize, 0, numExperts, 2048, cclBufferSize),
                        OUTPUT(combinedXDesc, combinedTopkWeightsDesc));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(L2AclnnMoeEpCombineTest, TestCombineBiasRejected)
{
    int64_t numTokens = 2048;
    int64_t hidden = 7168;
    int64_t topK = 8;
    int64_t epWorldSize = 8;
    int64_t numExperts = 64;
    int64_t cclBufferSize = 200 * 1024 * 1024;
    int64_t A = epWorldSize * 2048;

    TensorDesc contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xDesc = TensorDesc({A, hidden}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc dstSlotDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc recvSrcMetaDesc = TensorDesc({A, 3}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerRankDesc = TensorDesc({epWorldSize}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc biasDesc = TensorDesc({numTokens, hidden}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeEpCombine,
                        INPUT(contextDesc, xDesc, dstSlotDesc, recvSrcMetaDesc, numRecvPerRankDesc, TensorDesc(),
                              biasDesc, TensorDesc(), epWorldSize, 0, numExperts, 2048, cclBufferSize),
                        OUTPUT(TensorDesc({numTokens, hidden}, ACL_BF16, ACL_FORMAT_ND), TensorDesc()));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnMoeEpCombineTest, TestCombineInvalidEpWorldSize)
{
    int64_t numTokens = 512;
    int64_t hidden = 4096;
    int64_t topK = 4;
    int64_t cclBufferSize = 200 * 1024 * 1024;

    TensorDesc contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xDesc = TensorDesc({1024, hidden}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc dstSlotDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc recvSrcMetaDesc = TensorDesc({1024, 3}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerRankDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeEpCombine,
                        INPUT(contextDesc, xDesc, dstSlotDesc, recvSrcMetaDesc, numRecvPerRankDesc, TensorDesc(),
                              nullptr, nullptr, 1, 0, 8, 1024, cclBufferSize),
                        OUTPUT(TensorDesc({numTokens, hidden}, ACL_BF16, ACL_FORMAT_ND), TensorDesc()));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnMoeEpCombineTest, TestCombineInvalidNumExperts)
{
    int64_t numTokens = 512;
    int64_t hidden = 4096;
    int64_t topK = 4;
    int64_t epWorldSize = 8;
    int64_t cclBufferSize = 200 * 1024 * 1024;

    TensorDesc contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xDesc = TensorDesc({4096, hidden}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc dstSlotDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc recvSrcMetaDesc = TensorDesc({4096, 3}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerRankDesc = TensorDesc({epWorldSize}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeEpCombine,
                        INPUT(contextDesc, xDesc, dstSlotDesc, recvSrcMetaDesc, numRecvPerRankDesc, TensorDesc(),
                              nullptr, nullptr, epWorldSize, 0, 63, 1024, cclBufferSize),
                        OUTPUT(TensorDesc({numTokens, hidden}, ACL_BF16, ACL_FORMAT_ND), TensorDesc()));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnMoeEpCombineTest, TestCombineNullX)
{
    int64_t numTokens = 512;
    int64_t hidden = 4096;
    int64_t topK = 4;
    int64_t epWorldSize = 8;
    int64_t numExperts = 64;
    int64_t cclBufferSize = 200 * 1024 * 1024;

    TensorDesc contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc dstSlotDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc recvSrcMetaDesc = TensorDesc({4096, 3}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerRankDesc = TensorDesc({epWorldSize}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeEpCombine,
                        INPUT(contextDesc, TensorDesc(), dstSlotDesc, recvSrcMetaDesc, numRecvPerRankDesc, TensorDesc(),
                              nullptr, nullptr, epWorldSize, 0, numExperts, 1024, cclBufferSize),
                        OUTPUT(TensorDesc({numTokens, hidden}, ACL_BF16, ACL_FORMAT_ND), TensorDesc()));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}