/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_turbo_quant_attention_score.cpp
 * \brief ACLNN unit test for TurboQuantAttentionScore (aclnn_exclude).
 */

#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "aclnnop/aclnn_turbo_quant_attention_score.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class aclnnTurboQuantAttentionScoreTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "aclnnTurboQuantAttentionScoreTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "aclnnTurboQuantAttentionScoreTest TearDown" << endl;
    }
};

namespace {
constexpr int64_t NUM_TOKENS = 2;
constexpr int64_t NUM_Q_HEADS = 32;
constexpr int64_t NUM_KV_HEADS = 8;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t PAGED_BLOCK_SIZE = 128;
constexpr int64_t NUM_BLOCKS = 8;
constexpr int64_t MAX_BLOCKS_PER_SEQ = 4;
constexpr int64_t MSE_BITS = 3;
constexpr int64_t IDX_PACKED_BYTES = (HEAD_DIM * MSE_BITS + 7) / 8;
constexpr int64_t QJL_PACKED_BYTES = (HEAD_DIM + 7) / 8;
constexpr int64_t MAX_KV_LEN = MAX_BLOCKS_PER_SEQ * PAGED_BLOCK_SIZE;
} // namespace

TEST_F(aclnnTurboQuantAttentionScoreTest, GetWorkspaceSizeNormal)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    auto queryRotated = TensorDesc({NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto queryQjl = TensorDesc({NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto keyCacheIdx =
        TensorDesc({NUM_BLOCKS, PAGED_BLOCK_SIZE, NUM_KV_HEADS, IDX_PACKED_BYTES}, ACL_UINT8, ACL_FORMAT_ND);
    auto keyCacheQjl =
        TensorDesc({NUM_BLOCKS, PAGED_BLOCK_SIZE, NUM_KV_HEADS, QJL_PACKED_BYTES}, ACL_UINT8, ACL_FORMAT_ND);
    auto keyCacheNorm =
        TensorDesc({NUM_BLOCKS, PAGED_BLOCK_SIZE, NUM_KV_HEADS}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto keyCacheGamma =
        TensorDesc({NUM_BLOCKS, PAGED_BLOCK_SIZE, NUM_KV_HEADS}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto blockTable = TensorDesc({NUM_TOKENS, MAX_BLOCKS_PER_SEQ}, ACL_INT32, ACL_FORMAT_ND);
    auto seqLens = TensorDesc({NUM_TOKENS}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{128, 256});
    auto attnScores = TensorDesc({NUM_TOKENS, NUM_Q_HEADS, MAX_KV_LEN}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTurboQuantAttentionScore,
                        INPUT(queryRotated, queryQjl, keyCacheIdx, keyCacheQjl, keyCacheNorm, keyCacheGamma, blockTable,
                              seqLens, MSE_BITS),
                        OUTPUT(attnScores));
    uint64_t workspaceSize = 0U;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}
