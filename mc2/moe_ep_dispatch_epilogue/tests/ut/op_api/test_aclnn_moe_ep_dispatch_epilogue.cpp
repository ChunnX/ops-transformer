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
#include "../../../op_api/aclnn_moe_ep_dispatch_epilogue.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class L2AclnnMoeEpDispatchEpilogueTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        cout << "L2AclnnMoeEpDispatchEpilogueTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "L2AclnnMoeEpDispatchEpilogueTest TearDown" << endl;
    }
};

TEST_F(L2AclnnMoeEpDispatchEpilogueTest, TestEpilogueBasic)
{
    int64_t numTokens = 2048;
    int64_t topK = 8;
    int64_t epWorldSize = 8;
    int64_t numExperts = 64;
    int64_t numMaxTokensPerRank = 2048;
    int64_t cclBufferSize = 200 * 1024 * 1024;
    int64_t numLocalExperts = numExperts / epWorldSize;
    int64_t hidden = 7168;
    int64_t upperA = epWorldSize * numMaxTokensPerRank;

    TensorDesc contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc dstSlotDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerRankDesc = TensorDesc({epWorldSize}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerExpertDesc = TensorDesc({numLocalExperts}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc cachedRecvSrcMetadataDesc = TensorDesc({upperA, 3}, ACL_INT32, ACL_FORMAT_ND);

    TensorDesc recvXDesc = TensorDesc({upperA, hidden}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc recvSrcMetadataDesc = TensorDesc({upperA, 3}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc recvTopkWeightsDesc = TensorDesc({upperA}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMoeEpDispatchEpilogue,
                  INPUT(contextDesc, dstSlotDesc, numRecvPerRankDesc, numRecvPerExpertDesc, cachedRecvSrcMetadataDesc,
                        epWorldSize, 0, numExperts, numMaxTokensPerRank, cclBufferSize, 1),
                  OUTPUT(recvXDesc, recvSrcMetadataDesc, recvTopkWeightsDesc, nullptr));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(L2AclnnMoeEpDispatchEpilogueTest, TestEpilogueNoTopkWeights)
{
    int64_t numTokens = 2048;
    int64_t topK = 8;
    int64_t epWorldSize = 8;
    int64_t numExperts = 64;
    int64_t numMaxTokensPerRank = 2048;
    int64_t cclBufferSize = 200 * 1024 * 1024;
    int64_t numLocalExperts = numExperts / epWorldSize;
    int64_t hidden = 7168;
    int64_t upperA = epWorldSize * numMaxTokensPerRank;

    TensorDesc contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc dstSlotDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerRankDesc = TensorDesc({epWorldSize}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerExpertDesc = TensorDesc({numLocalExperts}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc cachedRecvSrcMetadataDesc = TensorDesc({upperA, 3}, ACL_INT32, ACL_FORMAT_ND);

    TensorDesc recvXDesc = TensorDesc({upperA, hidden}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc recvSrcMetadataDesc = TensorDesc({upperA, 3}, ACL_INT32, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMoeEpDispatchEpilogue,
                  INPUT(contextDesc, dstSlotDesc, numRecvPerRankDesc, numRecvPerExpertDesc, cachedRecvSrcMetadataDesc,
                        epWorldSize, 0, numExperts, numMaxTokensPerRank, cclBufferSize, 1),
                  OUTPUT(recvXDesc, recvSrcMetadataDesc, nullptr, nullptr));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(L2AclnnMoeEpDispatchEpilogueTest, TestEpilogueInvalidEpWorldSize)
{
    int64_t numTokens = 512;
    int64_t topK = 4;
    int64_t cclBufferSize = 200 * 1024 * 1024;

    TensorDesc contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc dstSlotDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerRankDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerExpertDesc = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeEpDispatchEpilogue,
                        INPUT(contextDesc, dstSlotDesc, numRecvPerRankDesc, numRecvPerExpertDesc, nullptr,
                              1, 0, 8, 1024, cclBufferSize, 1),
                        OUTPUT(TensorDesc({2048, 4096}, ACL_BF16, ACL_FORMAT_ND),
                               TensorDesc({2048, 3}, ACL_INT32, ACL_FORMAT_ND),
                               TensorDesc({2048}, ACL_FLOAT, ACL_FORMAT_ND), nullptr));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnMoeEpDispatchEpilogueTest, TestEpilogueInvalidNumExperts)
{
    int64_t numTokens = 512;
    int64_t topK = 4;
    int64_t epWorldSize = 4;
    int64_t cclBufferSize = 200 * 1024 * 1024;

    TensorDesc contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc dstSlotDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerRankDesc = TensorDesc({epWorldSize}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerExpertDesc = TensorDesc({33 / epWorldSize}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeEpDispatchEpilogue,
                        INPUT(contextDesc, dstSlotDesc, numRecvPerRankDesc, numRecvPerExpertDesc, nullptr,
                              epWorldSize, 0, 33, 1024, cclBufferSize, 1),
                        OUTPUT(TensorDesc({4096, 4096}, ACL_BF16, ACL_FORMAT_ND),
                               TensorDesc({4096, 3}, ACL_INT32, ACL_FORMAT_ND),
                               TensorDesc({4096}, ACL_FLOAT, ACL_FORMAT_ND), nullptr));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnMoeEpDispatchEpilogueTest, TestEpilogueCachedMode)
{
    int64_t numTokens = 512;
    int64_t topK = 4;
    int64_t epWorldSize = 4;
    int64_t numExperts = 32;
    int64_t numMaxTokensPerRank = 512;
    int64_t cclBufferSize = 200 * 1024 * 1024;
    int64_t numLocalExperts = numExperts / epWorldSize;
    int64_t hidden = 4096;
    int64_t upperA = epWorldSize * numMaxTokensPerRank;

    TensorDesc contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc dstSlotDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerRankDesc = TensorDesc({epWorldSize}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerExpertDesc = TensorDesc({numLocalExperts}, ACL_INT64, ACL_FORMAT_ND);

    TensorDesc recvXDesc = TensorDesc({upperA, hidden}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc recvSrcMetadataDesc = TensorDesc({upperA, 3}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc recvTopkWeightsDesc = TensorDesc({upperA}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeEpDispatchEpilogue,
                        INPUT(contextDesc, dstSlotDesc, numRecvPerRankDesc, numRecvPerExpertDesc, nullptr,
                              epWorldSize, 0, numExperts, numMaxTokensPerRank, cclBufferSize, 1),
                        OUTPUT(recvXDesc, recvSrcMetadataDesc, recvTopkWeightsDesc, nullptr));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(L2AclnnMoeEpDispatchEpilogueTest, TestEpilogueSmallShape)
{
    int64_t numTokens = 128;
    int64_t topK = 4;
    int64_t epWorldSize = 2;
    int64_t numExperts = 16;
    int64_t numMaxTokensPerRank = 128;
    int64_t cclBufferSize = 200 * 1024 * 1024;
    int64_t numLocalExperts = numExperts / epWorldSize;
    int64_t hidden = 2048;
    int64_t upperA = epWorldSize * numMaxTokensPerRank;

    TensorDesc contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc dstSlotDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerRankDesc = TensorDesc({epWorldSize}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerExpertDesc = TensorDesc({numLocalExperts}, ACL_INT64, ACL_FORMAT_ND);

    TensorDesc recvXDesc = TensorDesc({upperA, hidden}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc recvSrcMetadataDesc = TensorDesc({upperA, 3}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc recvTopkWeightsDesc = TensorDesc({upperA}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeEpDispatchEpilogue,
                        INPUT(contextDesc, dstSlotDesc, numRecvPerRankDesc, numRecvPerExpertDesc, nullptr,
                              epWorldSize, 0, numExperts, numMaxTokensPerRank, cclBufferSize, 1),
                        OUTPUT(recvXDesc, recvSrcMetadataDesc, recvTopkWeightsDesc, nullptr));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
