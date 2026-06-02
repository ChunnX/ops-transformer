#include <float.h>
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "aclnn_moe_ep_combine.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class L2AclnnMoeEpCombineTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        cout << "L2AclnnMoeEpCombineTest SetUp" << endl;
    }
    static void TearDownTestCase() {
        cout << "L2AclnnMoeEpCombineTest TearDown" << endl;
    }
};

TEST_F(L2AclnnMoeEpCombineTest, TestCombineBasicPureSum) {
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
        INPUT(contextDesc, xDesc, dstSlotDesc, recvSrcMetaDesc, numRecvPerRankDesc, nullDesc, nullDesc, nullDesc),
        ATTR(epWorldSize, 0, numExperts, 2048, cclBufferSize),
        OUTPUT(combinedXDesc, combinedTopkWeightsDesc));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.GetWorkspaceSize(&workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(L2AclnnMoeEpCombineTest, TestCombineWithTopkWeights) {
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
    TensorDesc nullDesc;

    TensorDesc combinedXDesc = TensorDesc({numTokens, hidden}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc combinedTopkWeightsDesc = TensorDesc({numTokens, topK}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeEpCombine,
        INPUT(contextDesc, xDesc, dstSlotDesc, recvSrcMetaDesc, numRecvPerRankDesc, topkWeightsDesc, nullDesc, nullDesc),
        ATTR(epWorldSize, 0, numExperts, 2048, cclBufferSize),
        OUTPUT(combinedXDesc, combinedTopkWeightsDesc));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.GetWorkspaceSize(&workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(L2AclnnMoeEpCombineTest, TestCombineBiasRejected) {
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
        INPUT(contextDesc, xDesc, dstSlotDesc, recvSrcMetaDesc, numRecvPerRankDesc, TensorDesc(), biasDesc, TensorDesc()),
        ATTR(epWorldSize, 0, numExperts, 2048, cclBufferSize),
        OUTPUT(TensorDesc({numTokens, hidden}, ACL_BF16, ACL_FORMAT_ND), TensorDesc()));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.GetWorkspaceSize(&workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}