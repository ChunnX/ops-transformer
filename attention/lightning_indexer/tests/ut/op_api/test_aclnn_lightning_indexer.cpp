/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <cstdint>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_lightning_indexer.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
using namespace std;
using namespace op;

class lightning_indexer_opapi_ut : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "lightning_indexer_opapi_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "lightning_indexer_opapi_ut TearDown" << endl;
    }
};

// BSND layout, returnValues=true — both sparseIndicesOut and sparseValuesOut are returned.
// query:   (B=1, S_q=8,  N_q=8,   D=128)
// key:     (B=1, S_kv=64, N_kv=1, D=128)
// weights: (B=1, S_q=8,  N_q=8)
// sparseIndicesOut: (B=1, S_q=8, N_kv=1, sparseCount=4)   dtype=INT32
// sparseValuesOut:  (B=1, S_q=8, N_kv=1, sparseCount=4)   dtype=FP16 (matches query)
TEST_F(lightning_indexer_opapi_ut, lightning_indexer_aclnn_0)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "BSND";
    const int64_t sparseCount = 4;
    const int64_t sparseMode = 3;
    const int64_t preTokens = INT64_MAX;
    const int64_t nextTokens = INT64_MAX;
    const bool returnValues = true;

    auto ut = OP_API_UT(
        aclnnLightningIndexer,
        INPUT(
            TensorDesc({1, 8, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1),  // query
            TensorDesc({1, 64, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1), // key
            TensorDesc({1, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1),         // weights
            nullptr,       // actualSeqLengthsQueryOptional
            nullptr,       // actualSeqLengthsKeyOptional
            nullptr,       // blockTableOptional
            layoutQuery,
            layoutKey,
            sparseCount,
            sparseMode,
            preTokens,
            nextTokens,
            returnValues
        ),
        OUTPUT(
            TensorDesc({1, 8, 1, 4}, ACL_INT32,   ACL_FORMAT_ND),  // sparseIndicesOut
            TensorDesc({1, 8, 1, 4}, ACL_FLOAT16, ACL_FORMAT_ND)   // sparseValuesOut (same dtype as query)
        )
    );

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// BSND layout, returnValues=false — only sparseIndicesOut is returned; sparseValuesOut is unused.
// Identical input shapes; sparseValuesOut output is empty (shape {0}) since it is not produced.
TEST_F(lightning_indexer_opapi_ut, lightning_indexer_aclnn_1)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "BSND";
    const int64_t sparseCount = 4;
    const int64_t sparseMode = 3;
    const int64_t preTokens = INT64_MAX;
    const int64_t nextTokens = INT64_MAX;
    const bool returnValues = false;

    auto ut = OP_API_UT(
        aclnnLightningIndexer,
        INPUT(
            TensorDesc({1, 8, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1),  // query
            TensorDesc({1, 64, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1), // key
            TensorDesc({1, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1),         // weights
            nullptr,       // actualSeqLengthsQueryOptional
            nullptr,       // actualSeqLengthsKeyOptional
            nullptr,       // blockTableOptional
            layoutQuery,
            layoutKey,
            sparseCount,
            sparseMode,
            preTokens,
            nextTokens,
            returnValues
        ),
        OUTPUT(
            TensorDesc({1, 8, 1, 4}, ACL_INT32,   ACL_FORMAT_ND),  // sparseIndicesOut
            TensorDesc({0},          ACL_FLOAT16, ACL_FORMAT_ND)   // sparseValuesOut: empty, not returned
        )
    );

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
