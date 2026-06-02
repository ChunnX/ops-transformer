#include <float.h>
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "aclnn_moe_ep_dispatch_epilogue.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class L2AclnnMoeEpDispatchEpilogueTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        cout << "L2AclnnMoeEpDispatchEpilogueTest SetUp" << endl;
    }
    static void TearDownTestCase() {
        cout << "L2AclnnMoeEpDispatchEpilogueTest TearDown" << endl;
    }
};

TEST_F(L2AclnnMoeEpDispatchEpilogueTest, TestEpilogueBasic) {
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
    TensorDesc recvSrcMetaInternalDesc = TensorDesc({upperA, 3}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerRankHpDesc = TensorDesc({epWorldSize}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerExpertHpDesc = TensorDesc({numLocalExperts}, ACL_INT64, ACL_FORMAT_ND);

    TensorDesc recvXDesc = TensorDesc({upperA, hidden}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc recvTopkIdxDesc = TensorDesc({upperA, topK}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc recvTopkWeightsDesc = TensorDesc({upperA, topK}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc recvSrcMetadataDesc = TensorDesc({upperA, 3}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerRankDesc = TensorDesc({epWorldSize}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc numRecvPerExpertDesc = TensorDesc({numLocalExperts}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeEpDispatchEpilogue,
        INPUT(contextDesc, dstSlotDesc, recvSrcMetaInternalDesc,
              numRecvPerRankHpDesc, numRecvPerExpertHpDesc),
        ATTR(epWorldSize, 0, numExperts, numMaxTokensPerRank, cclBufferSize, 1),
        OUTPUT(recvXDesc, recvTopkIdxDesc, recvTopkWeightsDesc,
               recvSrcMetadataDesc, numRecvPerRankDesc, numRecvPerExpertDesc));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.GetWorkspaceSize(&workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}