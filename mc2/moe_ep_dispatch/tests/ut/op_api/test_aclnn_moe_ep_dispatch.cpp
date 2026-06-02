#include <float.h>
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "aclnn_moe_ep_dispatch.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class L2AclnnMoeEpDispatchTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        cout << "L2AclnnMoeEpDispatchTest SetUp" << endl;
    }
    static void TearDownTestCase() {
        cout << "L2AclnnMoeEpDispatchTest TearDown" << endl;
    }
};

TEST_F(L2AclnnMoeEpDispatchTest, TestMoeEpDispatchBasic) {
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
    TensorDesc numRecvPerRankHpDesc = TensorDesc({epWorldSize}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerExpertHpDesc = TensorDesc({numLocalExperts}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc dstSlotIdxDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc recvSrcMetaInternalDesc = TensorDesc({upperA, 3}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeEpDispatch,
        INPUT(contextDesc, xDesc, topkIdxDesc, topkWeightsDesc,
              TensorDesc()),
        ATTR(epWorldSize, epRankId, numExperts,
             numMaxTokensPerRank, cclBufferSize, expertAlignment, true),
        OUTPUT(numRecvPerRankHpDesc, numRecvPerExpertHpDesc,
               dstSlotIdxDesc, recvSrcMetaInternalDesc));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.GetWorkspaceSize(&workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(L2AclnnMoeEpDispatchTest, TestMoeEpDispatchNullTopk) {
    int64_t numTokens = 2048;
    int64_t hidden = 7168;
    int64_t epWorldSize = 8;
    int64_t numExperts = 64;
    int64_t numMaxTokensPerRank = 2048;
    int64_t cclBufferSize = 200 * 1024 * 1024;
    int64_t numLocalExperts = numExperts / epWorldSize;

    TensorDesc contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xDesc = TensorDesc({numTokens, hidden}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc nullDesc;

    auto ut = OP_API_UT(aclnnMoeEpDispatch,
        INPUT(contextDesc, xDesc, nullDesc, nullDesc, nullDesc),
        ATTR(epWorldSize, 0, numExperts,
             numMaxTokensPerRank, cclBufferSize, 1, true),
        OUTPUT(TensorDesc({epWorldSize}, ACL_INT32, ACL_FORMAT_ND),
               TensorDesc({numLocalExperts}, ACL_INT64, ACL_FORMAT_ND),
               TensorDesc({numTokens, 8}, ACL_INT32, ACL_FORMAT_ND),
               TensorDesc({epWorldSize * numMaxTokensPerRank, 3}, ACL_INT32, ACL_FORMAT_ND)));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.GetWorkspaceSize(&workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2AclnnMoeEpDispatchTest, TestMoeEpDispatchInvalidEpWorldSize) {
    int64_t numTokens = 2048;
    int64_t hidden = 7168;
    int64_t topK = 8;
    int64_t cclBufferSize = 200 * 1024 * 1024;

    TensorDesc contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xDesc = TensorDesc({numTokens, hidden}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc topkIdxDesc = TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeEpDispatch,
        INPUT(contextDesc, xDesc, topkIdxDesc, TensorDesc({numTokens, topK}, ACL_FLOAT, ACL_FORMAT_ND),
              TensorDesc()),
        ATTR(1, 0, 8, 2048, cclBufferSize, 1, true),
        OUTPUT(TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND),
               TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND),
               TensorDesc({numTokens, topK}, ACL_INT32, ACL_FORMAT_ND),
               TensorDesc({2048, 3}, ACL_INT32, ACL_FORMAT_ND)));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.GetWorkspaceSize(&workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}