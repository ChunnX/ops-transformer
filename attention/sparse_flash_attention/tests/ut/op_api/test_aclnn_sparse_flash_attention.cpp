/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array> 
#include <cstdint>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_sparse_flash_attention.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
using namespace std;
using namespace op;

class sparse_flash_attention_opapi_ut : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "sparse_flash_attention_opapi_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "sparse_flash_attention_opapi_ut TearDown" << endl;
    }
};

// BSND layout, MHA/GQA (attentionMode=0), returnSoftmaxLse=false
// query:  (B=2, S_q=1,   N_q=32, D=64)
// key:    (B=2, S_kv=128, N_kv=1, D=64)
// value:  (B=2, S_kv=128, N_kv=1, D=64)
// sparseIndices: (B=2, N_kv=1, S_q_blocks=1, sparseBlockCount=S_kv/sparseBlockSize=128/64=2)
//              = (2, 1, 1, 2)   dtype=INT32
// attentionOut:  (B=2, S_q=1, N_q=32, D=64)
TEST_F(sparse_flash_attention_opapi_ut, sparse_flash_attention_aclnn_0) {
    const double scaleValue = 0.0416666666666667;
    const int64_t sparseBlockSize = 64;
    char layoutQ[] = "BSND";
    char layoutKv[] = "BSND";
    const int64_t sparseMode = 3;
    const int64_t preTokens = INT64_MAX;
    const int64_t nextTokens = INT64_MAX;
    const int64_t attentionMode = 0;
    const bool returnSoftmaxLse = false;

    auto ut = OP_API_UT(
        aclnnSparseFlashAttention,
        INPUT(
            TensorDesc({2, 1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1),   // query
            TensorDesc({2, 128, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1),  // key
            TensorDesc({2, 128, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1),  // value
            TensorDesc({2, 1, 1, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1),        // sparseIndices: B=2
            nullptr,   // blockTableOptional
            nullptr,   // actualSeqLengthsQueryOptional
            nullptr,   // actualSeqLengthsKvOptional
            nullptr,   // queryRopeOptional
            nullptr,   // keyRopeOptional
            scaleValue,
            sparseBlockSize,
            layoutQ,
            layoutKv,
            sparseMode,
            preTokens,
            nextTokens,
            attentionMode,
            returnSoftmaxLse
        ),
        OUTPUT(
            TensorDesc({2, 1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND),  // attentionOut
            TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND),                 // softmaxMax (unused when returnSoftmaxLse=false)
            TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND)                  // softmaxSum (unused when returnSoftmaxLse=false)
        )
    );

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}