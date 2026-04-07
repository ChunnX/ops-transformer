/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_flash_attention_score_grad.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class flash_attention_score_grad_v4_opapi_ut : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "flash_attention_score_grad_v4_opapi_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "flash_attention_score_grad_v4_opapi_ut TearDown" << endl;
    }
};

TEST_F(flash_attention_score_grad_v4_opapi_ut, flash_attention_score_grad_aclnn_0) {
    auto tensorQ = TensorDesc({256, 1, 192}, ACL_FLOAT, ACL_FORMAT_ND)
        .ValueRange(-1, 1);
    auto tensorK = TensorDesc({256, 1, 192}, ACL_FLOAT, ACL_FORMAT_ND)
        .ValueRange(-1, 1);
    auto tensorV = TensorDesc({256, 1, 192}, ACL_FLOAT, ACL_FORMAT_ND)
        .ValueRange(-1, 1);
    auto tensorDx = TensorDesc({256, 1, 192}, ACL_FLOAT, ACL_FORMAT_ND)
        .ValueRange(-1, 1);
    auto tensorAttenMask = TensorDesc({256, 256}, ACL_UINT8, ACL_FORMAT_ND)
        .Value(vector<uint8_t>{0});

    auto tensorSoftmaxMax =
        TensorDesc({1, 1, 256, 8}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 1 * 256 * 8, 3.0f));
    auto tensorSoftmaxSum =
        TensorDesc({1, 1, 256, 8}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 1 * 256 * 8, 3.0f));
    auto tensorAttentionIn = TensorDesc({256, 1, 192}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    const double scaleValue = 1.0 / sqrt(128);
    const double keepProb = 1.0;
    const int64_t preTokens = 65536;
    const int64_t nextTokens = 65536;
    const int64_t headNum = 1;
    char inputlayout[] = "SBH";
    const int64_t innerPrecise = 0;
    const int64_t sparseMod = 0;
    const int64_t outDtype = 1;
    const int64_t pseType = 1;
    const int64_t seed = 0;
    const int64_t offset = 0;

    auto tensorDQDesc = TensorDesc({256, 1, 192}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDKDesc = TensorDesc({256, 1, 192}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDVDesc = TensorDesc({256, 1, 192}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDPseDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDQROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDKROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    // auto tensorDSinkOutDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFlashAttentionScoreGradV4,
        INPUT(
            tensorQ,
            tensorK,
            tensorV,
            tensorDx,
            nullptr,                  // pse
            nullptr,                  // dropmask
            nullptr,                  // psepaddingmask
            tensorAttenMask,
            tensorSoftmaxMax,
            tensorSoftmaxSum,
            nullptr,                  // softmaxin
            tensorAttentionIn,
            nullptr,                  // sink
            nullptr,                  // queryrope
            nullptr,                  // keyrope
            nullptr,                  // dscaleq
            nullptr,                  // dscalek
            nullptr,                  // dscalev
            nullptr,                  // dscaledy
            nullptr,                  // dscaleo
            nullptr,                  // Prefix
            nullptr,                  // ActualSeqQLen
            nullptr,                  // ActualSeqKVLen
            nullptr,                  // QStartIdx
            nullptr,                  // KVStartIdx
            scaleValue,
            keepProb,
            preTokens,
            nextTokens,
            headNum,
            inputlayout,
            nullptr,                  // softmaxInLayout
            innerPrecise,
            sparseMod,
            pseType,
            seed,
            offset,
            outDtype
        ),
        OUTPUT(
            tensorDQDesc,
            tensorDKDesc,
            tensorDVDesc,
            tensorDQROPEDesc,
            tensorDKROPEDesc,
            tensorDPseDesc,
            nullptr
        )
    );
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(flash_attention_score_grad_v4_opapi_ut, flash_attention_score_grad_aclnn_1_bnsd_layout) {
    auto tensorQ = TensorDesc({1, 8, 256, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({1, 8, 256, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({1, 8, 256, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorDx = TensorDesc({1, 8, 256, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorAttenMask = TensorDesc({1, 8, 256, 256}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});
    auto tensorSoftmaxMax = TensorDesc({1, 8, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 8 * 256, 3.0f));
    auto tensorSoftmaxSum = TensorDesc({1, 8, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 8 * 256, 3.0f));
    auto tensorAttentionIn = TensorDesc({1, 8, 256, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    const double scaleValue = 1.0 / sqrt(128);
    const double keepProb = 1.0;
    const int64_t preTokens = 65536;
    const int64_t nextTokens = 65536;
    const int64_t headNum = 8;
    char inputlayout[] = "BNSD";
    const int64_t innerPrecise = 0;
    const int64_t sparseMod = 0;
    const int64_t outDtype = 1;
    const int64_t pseType = 1;
    const int64_t seed = 0;
    const int64_t offset = 0;

    auto tensorDQDesc = TensorDesc({1, 8, 256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDKDesc = TensorDesc({1, 8, 256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDVDesc = TensorDesc({1, 8, 256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDPseDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDQROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDKROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFlashAttentionScoreGradV4,
        INPUT(
            tensorQ,
            tensorK,
            tensorV,
            tensorDx,
            nullptr,
            nullptr,
            nullptr,
            tensorAttenMask,
            tensorSoftmaxMax,
            tensorSoftmaxSum,
            nullptr,
            tensorAttentionIn,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            scaleValue,
            keepProb,
            preTokens,
            nextTokens,
            headNum,
            inputlayout,
            nullptr,
            innerPrecise,
            sparseMod,
            pseType,
            seed,
            offset,
            outDtype
        ),
        OUTPUT(
            tensorDQDesc,
            tensorDKDesc,
            tensorDVDesc,
            tensorDQROPEDesc,
            tensorDKROPEDesc,
            tensorDPseDesc,
            nullptr
        )
    );
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(flash_attention_score_grad_v4_opapi_ut, flash_attention_score_grad_aclnn_2_bsnd_layout) {
    auto tensorQ = TensorDesc({1, 256, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({1, 256, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({1, 256, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorDx = TensorDesc({1, 256, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorAttenMask = TensorDesc({1, 8, 256, 256}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});
    auto tensorSoftmaxMax = TensorDesc({1, 8, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 8 * 256, 3.0f));
    auto tensorSoftmaxSum = TensorDesc({1, 8, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 8 * 256, 3.0f));
    auto tensorAttentionIn = TensorDesc({1, 256, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    const double scaleValue = 1.0 / sqrt(128);
    const double keepProb = 1.0;
    const int64_t preTokens = 65536;
    const int64_t nextTokens = 65536;
    const int64_t headNum = 8;
    char inputlayout[] = "BSND";
    const int64_t innerPrecise = 0;
    const int64_t sparseMod = 0;
    const int64_t outDtype = 1;
    const int64_t pseType = 1;
    const int64_t seed = 0;
    const int64_t offset = 0;

    auto tensorDQDesc = TensorDesc({1, 256, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDKDesc = TensorDesc({1, 256, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDVDesc = TensorDesc({1, 256, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDPseDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDQROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDKROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFlashAttentionScoreGradV4,
        INPUT(
            tensorQ,
            tensorK,
            tensorV,
            tensorDx,
            nullptr,
            nullptr,
            nullptr,
            tensorAttenMask,
            tensorSoftmaxMax,
            tensorSoftmaxSum,
            nullptr,
            tensorAttentionIn,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            scaleValue,
            keepProb,
            preTokens,
            nextTokens,
            headNum,
            inputlayout,
            nullptr,
            innerPrecise,
            sparseMod,
            pseType,
            seed,
            offset,
            outDtype
        ),
        OUTPUT(
            tensorDQDesc,
            tensorDKDesc,
            tensorDVDesc,
            tensorDQROPEDesc,
            tensorDKROPEDesc,
            tensorDPseDesc,
            nullptr
        )
    );
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(flash_attention_score_grad_v4_opapi_ut, flash_attention_score_grad_aclnn_3_bsh_layout) {
    auto tensorQ = TensorDesc({1, 256, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({1, 256, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({1, 256, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorDx = TensorDesc({1, 256, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorAttenMask = TensorDesc({1, 256, 256}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});
    auto tensorSoftmaxMax = TensorDesc({1, 16, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 16 * 256, 3.0f));
    auto tensorSoftmaxSum = TensorDesc({1, 16, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 16 * 256, 3.0f));
    auto tensorAttentionIn = TensorDesc({1, 256, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    const double scaleValue = 1.0 / sqrt(64);
    const double keepProb = 1.0;
    const int64_t preTokens = 65536;
    const int64_t nextTokens = 65536;
    const int64_t headNum = 16;
    char inputlayout[] = "BSH";
    const int64_t innerPrecise = 0;
    const int64_t sparseMod = 0;
    const int64_t outDtype = 1;
    const int64_t pseType = 1;
    const int64_t seed = 0;
    const int64_t offset = 0;

    auto tensorDQDesc = TensorDesc({1, 256, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDKDesc = TensorDesc({1, 256, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDVDesc = TensorDesc({1, 256, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDPseDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDQROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDKROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFlashAttentionScoreGradV4,
        INPUT(
            tensorQ,
            tensorK,
            tensorV,
            tensorDx,
            nullptr,
            nullptr,
            nullptr,
            tensorAttenMask,
            tensorSoftmaxMax,
            tensorSoftmaxSum,
            nullptr,
            tensorAttentionIn,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            scaleValue,
            keepProb,
            preTokens,
            nextTokens,
            headNum,
            inputlayout,
            nullptr,
            innerPrecise,
            sparseMod,
            pseType,
            seed,
            offset,
            outDtype
        ),
        OUTPUT(
            tensorDQDesc,
            tensorDKDesc,
            tensorDVDesc,
            tensorDQROPEDesc,
            tensorDKROPEDesc,
            tensorDPseDesc,
            nullptr
        )
    );
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(flash_attention_score_grad_v4_opapi_ut, flash_attention_score_grad_aclnn_4_tnd_layout) {
    auto tensorQ = TensorDesc({256, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({256, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({256, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorDx = TensorDesc({256, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorAttenMask = TensorDesc({256, 256}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});
    auto tensorSoftmaxMax = TensorDesc({8, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(8 * 256, 3.0f));
    auto tensorSoftmaxSum = TensorDesc({8, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(8 * 256, 3.0f));
    auto tensorAttentionIn = TensorDesc({256, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    const double scaleValue = 1.0 / sqrt(128);
    const double keepProb = 1.0;
    const int64_t preTokens = 65536;
    const int64_t nextTokens = 65536;
    const int64_t headNum = 8;
    char inputlayout[] = "TND";
    const int64_t innerPrecise = 0;
    const int64_t sparseMod = 0;
    const int64_t outDtype = 1;
    const int64_t pseType = 1;
    const int64_t seed = 0;
    const int64_t offset = 0;

    auto tensorDQDesc = TensorDesc({256, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDKDesc = TensorDesc({256, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDVDesc = TensorDesc({256, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDPseDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDQROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDKROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFlashAttentionScoreGradV4,
        INPUT(
            tensorQ,
            tensorK,
            tensorV,
            tensorDx,
            nullptr,
            nullptr,
            nullptr,
            tensorAttenMask,
            tensorSoftmaxMax,
            tensorSoftmaxSum,
            nullptr,
            tensorAttentionIn,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            scaleValue,
            keepProb,
            preTokens,
            nextTokens,
            headNum,
            inputlayout,
            nullptr,
            innerPrecise,
            sparseMod,
            pseType,
            seed,
            offset,
            outDtype
        ),
        OUTPUT(
            tensorDQDesc,
            tensorDKDesc,
            tensorDVDesc,
            tensorDQROPEDesc,
            tensorDKROPEDesc,
            tensorDPseDesc,
            nullptr
        )
    );
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(flash_attention_score_grad_v4_opapi_ut, flash_attention_score_grad_aclnn_5_bf16_dtype) {
    auto tensorQ = TensorDesc({256, 1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({256, 1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({256, 1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorDx = TensorDesc({256, 1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorAttenMask = TensorDesc({256, 256}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});
    auto tensorSoftmaxMax = TensorDesc({1, 1, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 1 * 256, 3.0f));
    auto tensorSoftmaxSum = TensorDesc({1, 1, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 1 * 256, 3.0f));
    auto tensorAttentionIn = TensorDesc({256, 1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);

    const double scaleValue = 1.0 / sqrt(128);
    const double keepProb = 1.0;
    const int64_t preTokens = 65536;
    const int64_t nextTokens = 65536;
    const int64_t headNum = 1;
    char inputlayout[] = "SBH";
    const int64_t innerPrecise = 0;
    const int64_t sparseMod = 0;
    const int64_t outDtype = 1;
    const int64_t pseType = 1;
    const int64_t seed = 0;
    const int64_t offset = 0;

    auto tensorDQDesc = TensorDesc({256, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto tensorDKDesc = TensorDesc({256, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto tensorDVDesc = TensorDesc({256, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto tensorDPseDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDQROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDKROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFlashAttentionScoreGradV4,
        INPUT(
            tensorQ,
            tensorK,
            tensorV,
            tensorDx,
            nullptr,
            nullptr,
            nullptr,
            tensorAttenMask,
            tensorSoftmaxMax,
            tensorSoftmaxSum,
            nullptr,
            tensorAttentionIn,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            scaleValue,
            keepProb,
            preTokens,
            nextTokens,
            headNum,
            inputlayout,
            nullptr,
            innerPrecise,
            sparseMod,
            pseType,
            seed,
            offset,
            outDtype
        ),
        OUTPUT(
            tensorDQDesc,
            tensorDKDesc,
            tensorDVDesc,
            tensorDQROPEDesc,
            tensorDKROPEDesc,
            tensorDPseDesc,
            nullptr
        )
    );
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(flash_attention_score_grad_v4_opapi_ut, flash_attention_score_grad_aclnn_6_head_dim_72) {
    auto tensorQ = TensorDesc({256, 1, 72}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({256, 1, 72}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({256, 1, 72}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorDx = TensorDesc({256, 1, 72}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorAttenMask = TensorDesc({256, 256}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});
    auto tensorSoftmaxMax = TensorDesc({1, 1, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 1 * 256, 3.0f));
    auto tensorSoftmaxSum = TensorDesc({1, 1, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 1 * 256, 3.0f));
    auto tensorAttentionIn = TensorDesc({256, 1, 72}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    const double scaleValue = 1.0 / sqrt(72);
    const double keepProb = 1.0;
    const int64_t preTokens = 65536;
    const int64_t nextTokens = 65536;
    const int64_t headNum = 1;
    char inputlayout[] = "SBH";
    const int64_t innerPrecise = 0;
    const int64_t sparseMod = 0;
    const int64_t outDtype = 1;
    const int64_t pseType = 1;
    const int64_t seed = 0;
    const int64_t offset = 0;

    auto tensorDQDesc = TensorDesc({256, 1, 72}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDKDesc = TensorDesc({256, 1, 72}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDVDesc = TensorDesc({256, 1, 72}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDPseDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDQROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDKROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFlashAttentionScoreGradV4,
        INPUT(
            tensorQ,
            tensorK,
            tensorV,
            tensorDx,
            nullptr,
            nullptr,
            nullptr,
            tensorAttenMask,
            tensorSoftmaxMax,
            tensorSoftmaxSum,
            nullptr,
            tensorAttentionIn,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            scaleValue,
            keepProb,
            preTokens,
            nextTokens,
            headNum,
            inputlayout,
            nullptr,
            innerPrecise,
            sparseMod,
            pseType,
            seed,
            offset,
            outDtype
        ),
        OUTPUT(
            tensorDQDesc,
            tensorDKDesc,
            tensorDVDesc,
            tensorDQROPEDesc,
            tensorDKROPEDesc,
            tensorDPseDesc,
            nullptr
        )
    );
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(flash_attention_score_grad_v4_opapi_ut, flash_attention_score_grad_aclnn_7_head_dim_88) {
    auto tensorQ = TensorDesc({256, 1, 88}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({256, 1, 88}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({256, 1, 88}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorDx = TensorDesc({256, 1, 88}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorAttenMask = TensorDesc({256, 256}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});
    auto tensorSoftmaxMax = TensorDesc({1, 1, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 1 * 256, 3.0f));
    auto tensorSoftmaxSum = TensorDesc({1, 1, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 1 * 256, 3.0f));
    auto tensorAttentionIn = TensorDesc({256, 1, 88}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    const double scaleValue = 1.0 / sqrt(88);
    const double keepProb = 1.0;
    const int64_t preTokens = 65536;
    const int64_t nextTokens = 65536;
    const int64_t headNum = 1;
    char inputlayout[] = "SBH";
    const int64_t innerPrecise = 0;
    const int64_t sparseMod = 0;
    const int64_t outDtype = 1;
    const int64_t pseType = 1;
    const int64_t seed = 0;
    const int64_t offset = 0;

    auto tensorDQDesc = TensorDesc({256, 1, 88}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDKDesc = TensorDesc({256, 1, 88}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDVDesc = TensorDesc({256, 1, 88}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDPseDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDQROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDKROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFlashAttentionScoreGradV4,
        INPUT(
            tensorQ,
            tensorK,
            tensorV,
            tensorDx,
            nullptr,
            nullptr,
            nullptr,
            tensorAttenMask,
            tensorSoftmaxMax,
            tensorSoftmaxSum,
            nullptr,
            tensorAttentionIn,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            scaleValue,
            keepProb,
            preTokens,
            nextTokens,
            headNum,
            inputlayout,
            nullptr,
            innerPrecise,
            sparseMod,
            pseType,
            seed,
            offset,
            outDtype
        ),
        OUTPUT(
            tensorDQDesc,
            tensorDKDesc,
            tensorDVDesc,
            tensorDQROPEDesc,
            tensorDKROPEDesc,
            tensorDPseDesc,
            nullptr
        )
    );
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(flash_attention_score_grad_v4_opapi_ut, flash_attention_score_grad_aclnn_8_with_dropmask) {
    auto tensorQ = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorDx = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorAttenMask = TensorDesc({256, 256}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});
    auto tensorDropMask = TensorDesc({256, 256}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});
    auto tensorSoftmaxMax = TensorDesc({1, 1, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 1 * 256, 3.0f));
    auto tensorSoftmaxSum = TensorDesc({1, 1, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 1 * 256, 3.0f));
    auto tensorAttentionIn = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    const double scaleValue = 1.0 / sqrt(128);
    const double keepProb = 0.9;
    const int64_t preTokens = 65536;
    const int64_t nextTokens = 6555536;
    const int64_t headNum = 1;
    char inputlayout[] = "SBH";
    const int64_t innerPrecise = 0;
    const int64_t sparseMod = 0;
    const int64_t outDtype = 1;
    const int64_t pseType = 1;
    const int64_t seed = 0;
    const int64_t offset = 0;

    auto tensorDQDesc = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDKDesc = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDVDesc = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDPseDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDQROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDKROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFlashAttentionScoreGradV4,
        INPUT(
            tensorQ,
            tensorK,
            tensorV,
            tensorDx,
            nullptr,
            tensorDropMask,
            nullptr,
            tensorAttenMask,
            tensorSoftmaxMax,
            tensorSoftmaxSum,
            nullptr,
            tensorAttentionIn,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            scaleValue,
            keepProb,
            preTokens,
            nextTokens,
            headNum,
            inputlayout,
            nullptr,
            innerPrecise,
            sparseMod,
            pseType,
            seed,
            offset,
            outDtype
        ),
        OUTPUT(
            tensorDQDesc,
            tensorDKDesc,
            tensorDVDesc,
            tensorDQROPEDesc,
            tensorDKROPEDesc,
            tensorDPseDesc,
            nullptr
        )
    );
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(flash_attention_score_grad_v4_opapi_ut, flash_attention_score_grad_aclnn_9_with_sinkin) {
    auto tensorQ = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorDx = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorAttenMask = TensorDesc({256, 256}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});
    auto tensorSoftmaxMax = TensorDesc({1, 1, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 1 * 256, 3.0f));
    auto tensorSoftmaxSum = TensorDesc({1, 1, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 1 * 256, 3.0f));
    auto tensorAttentionIn = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorSinkIn = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{0.0f});

    const double scaleValue = 1.0 / sqrt(128);
    const double keepProb = 1.0;
    const int64_t preTokens = 65536;
    const int64_t nextTokens = 65536;
    const int64_t headNum = 1;
    char inputlayout[] = "SBH";
    const int64_t innerPrecise = 0;
    const int64_t sparseMod = 0;
    const int64_t outDtype = 1;
    const int64_t pseType = 1;
    const int64_t seed = 0;
    const int64_t offset = 0;

    auto tensorDQDesc = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDKDesc = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDVDesc = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDPseDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDQROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDKROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDSinkOutDesc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFlashAttentionScoreGradV4,
        INPUT(
            tensorQ,
            tensorK,
            tensorV,
            tensorDx,
            nullptr,
            nullptr,
            nullptr,
            tensorAttenMask,
            tensorSoftmaxMax,
            tensorSoftmaxSum,
            nullptr,
            tensorAttentionIn,
            tensorSinkIn,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            scaleValue,
            keepProb,
            preTokens,
            nextTokens,
            headNum,
            inputlayout,
            nullptr,
            innerPrecise,
            sparseMod,
            pseType,
            seed,
            offset,
            outDtype
        ),
        OUTPUT(
            tensorDQDesc,
            tensorDKDesc,
            tensorDVDesc,
            tensorDQROPEDesc,
            tensorDKROPEDesc,
            tensorDPseDesc,
            tensorDSinkOutDesc
        )
    );
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(flash_attention_score_grad_v4_opapi_ut, flash_attention_score_grad_aclnn_10_with_queryrope) {
    auto tensorQ = TensorDesc({256, 8, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({256, 8, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({256, 8, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorDx = TensorDesc({256, 8, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorQROPE = TensorDesc({256, 8, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorKROPE = TensorDesc({256, 8, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorAttenMask = TensorDesc({256, 256}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});
    auto tensorSoftmaxMax = TensorDesc({8, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(8 * 256, 3.0f));
    auto tensorSoftmaxSum = TensorDesc({8, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(8 * 256, 3.0f));
    auto tensorAttentionIn = TensorDesc({256, 8, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);

    const double scaleValue = 1.0 / sqrt(128);
    const double keepProb = 1.0;
    const int64_t preTokens = 65536;
    const int64_t nextTokens = 65536;
    const int64_t headNum = 8;
    char inputlayout[] = "TND";
    const int64_t innerPrecise = 0;
    const int64_t sparseMod = 0;
    const int64_t outDtype = 1;
    const int64_t pseType = 1;
    const int64_t seed = 0;
    const int64_t offset = 0;

    auto tensorDQDesc = TensorDesc({256, 8, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto tensorDKDesc = TensorDesc({256, 8, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto tensorDVDesc = TensorDesc({256, 8, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto tensorDQROPEDesc = TensorDesc({256, 8, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto tensorDKROPEDesc = TensorDesc({256, 8, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto tensorDPseDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFlashAttentionScoreGradV4,
        INPUT(
            tensorQ,
            tensorK,
            tensorV,
            tensorDx,
            nullptr,
            nullptr,
            nullptr,
            tensorAttenMask,
            tensorSoftmaxMax,
            tensorSoftmaxSum,
            nullptr,
            tensorAttentionIn,
            nullptr,
            tensorQROPE,
            tensorKROPE,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            scaleValue,
            keepProb,
            preTokens,
            nextTokens,
            headNum,
            inputlayout,
            nullptr,
            innerPrecise,
            sparseMod,
            pseType,
            seed,
            offset,
            outDtype
        ),
        OUTPUT(
            tensorDQDesc,
            tensorDKDesc,
            tensorDVDesc,
            tensorDQROPEDesc,
            tensorDKROPEDesc,
            tensorDPseDesc,
            nullptr
        )
    );
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(flash_attention_score_grad_v4_opapi_ut, flash_attention_score_grad_aclnn_11_nullptr_query) {
    auto tensorK = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorDx = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorAttenMask = TensorDesc({256, 256}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});
    auto tensorSoftmaxMax = TensorDesc({1, 1, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 1 * 256, 3.0f));
    auto tensorSoftmaxSum = TensorDesc({1, 1, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 1 * 256, 3.0f));
    auto tensorAttentionIn = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    const double scaleValue = 1.0 / sqrt(128);
    const double keepProb = 1.0;
    const int64_t preTokens = 65536;
    const int64_t nextTokens = 65536;
    const int64_t headNum = 1;
    char inputlayout[] = "SBH";
    const int64_t innerPrecise = 0;
    const int64_t sparseMod = 0;
    const int64_t outDtype = 1;
    const int64_t pseType = 1;
    const int64_t seed = 0;
    const int64_t offset = 0;

    auto tensorDQDesc = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDKDesc = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDVDesc = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDPseDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDQROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDKROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFlashAttentionScoreGradV4,
        INPUT(
            nullptr,
            tensorK,
            tensorV,
            tensorDx,
            nullptr,
            nullptr,
            nullptr,
            tensorAttenMask,
            tensorSoftmaxMax,
            tensorSoftmaxSum,
            nullptr,
            tensorAttentionIn,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            scaleValue,
            keepProb,
            preTokens,
            nextTokens,
            headNum,
            inputlayout,
            nullptr,
            innerPrecise,
            sparseMod,
            pseType,
            seed,
            offset,
            outDtype
        ),
        OUTPUT(
            tensorDQDesc,
            tensorDKDesc,
            tensorDVDesc,
            tensorDQROPEDesc,
            tensorDKROPEDesc,
            tensorDPseDesc,
            nullptr
        )
    );
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(flash_attention_score_grad_v4_opapi_ut, flash_attention_score_grad_aclnn_12_invalid_headnum) {
    auto tensorQ = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorDx = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorAttenMask = TensorDesc({256, 256}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});
    auto tensorSoftmaxMax = TensorDesc({1, 1, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 1 * 256, 3.0f));
    auto tensorSoftmaxSum = TensorDesc({1, 1, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 1 * 256, 3.0f));
    auto tensorAttentionIn = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    const double scaleValue = 1.0 / sqrt(128);
    const double keepProb = 1.0;
    const int64_t preTokens = 65536;
    const int64_t nextTokens = 65536;
    const int64_t headNum = 0;
    char inputlayout[] = "SBH";
    const int64_t innerPrecise = 0;
    const int64_t sparseMod = 0;
    const int64_t outDtype = 1;
    const int64_t pseType = 1;
    const int64_t seed = 0;
    const int64_t offset = 0;

    auto tensorDQDesc = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDKDesc = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDVDesc = TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDPseDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDQROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDKROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFlashAttentionScoreGradV4,
        INPUT(
            tensorQ,
            tensorK,
            tensorV,
            tensorDx,
            nullptr,
            nullptr,
            nullptr,
            tensorAttenMask,
            tensorSoftmaxMax,
            tensorSoftmaxSum,
            nullptr,
            tensorAttentionIn,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            scaleValue,
            keepProb,
            preTokens,
            nextTokens,
            headNum,
            inputlayout,
            nullptr,
            innerPrecise,
            sparseMod,
            pseType,
            seed,
            offset,
            outDtype
        ),
        OUTPUT(
            tensorDQDesc,
            tensorDKDesc,
            tensorDVDesc,
            tensorDQROPEDesc,
            tensorDKROPEDesc,
            tensorDPseDesc,
            nullptr
        )
    );
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(flash_attention_score_grad_v4_opapi_ut, flash_attention_score_grad_aclnn_13_head_dim_192) {
    auto tensorQ = TensorDesc({256, 1, 192}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({256, 1, 192}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({256, 1, 192}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorDx = TensorDesc({256, 1, 192}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorAttenMask = TensorDesc({256, 256}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});
    auto tensorSoftmaxMax = TensorDesc({1, 1, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 1 * 256, 3.0f));
    auto tensorSoftmaxSum = TensorDesc({1, 1, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 1 * 256, 3.0f));
    auto tensorAttentionIn = TensorDesc({256, 1, 192}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    const double scaleValue = 1.0 / sqrt(192);
    const double keepProb = 1.0;
    const int64_t preTokens = 65536;
    const int64_t nextTokens = 65536;
    const int64_t headNum = 1;
    char inputlayout[] = "SBH";
    const int64_t innerPrecise = 0;
    const int64_t sparseMod = 0;
    const int64_t outDtype = 1;
    const int64_t pseType = 1;
    const int64_t seed = 0;
    const int64_t offset = 0;

    auto tensorDQDesc = TensorDesc({256, 1, 192}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDKDesc = TensorDesc({256, 1, 192}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDVDesc = TensorDesc({256, 1, 192}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDPseDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDQROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDKROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFlashAttentionScoreGradV4,
        INPUT(
            tensorQ,
            tensorK,
            tensorV,
            tensorDx,
            nullptr,
            nullptr,
            nullptr,
            tensorAttenMask,
            tensorSoftmaxMax,
            tensorSoftmaxSum,
            nullptr,
            tensorAttentionIn,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            scaleValue,
            keepProb,
            preTokens,
            nextTokens,
            headNum,
            inputlayout,
            nullptr,
            innerPrecise,
            sparseMod,
            pseType,
            seed,
            offset,
            outDtype
        ),
        OUTPUT(
            tensorDQDesc,
            tensorDKDesc,
            tensorDVDesc,
            tensorDQROPEDesc,
            tensorDKROPEDesc,
            tensorDPseDesc,
            nullptr
        )
    );
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(flash_attention_score_grad_v4_opapi_ut, flash_attention_score_grad_aclnn_14_head_dim_8) {
    auto tensorQ = TensorDesc({256, 1, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({256, 1, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({256, 1, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorDx = TensorDesc({256, 1, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorAttenMask = TensorDesc({256, 256}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});
    auto tensorSoftmaxMax = TensorDesc({1, 1, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 1 * 256, 3.0f));
    auto tensorSoftmaxSum = TensorDesc({1, 1, 256}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(1 * 1 * 256, 3.0f));
    auto tensorAttentionIn = TensorDesc({256, 1, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    const double scaleValue = 1.0 / sqrt(8);
    const double keepProb = 1.0;
    const int64_t preTokens = 65536;
    const int64_t nextTokens = 65536;
    const int64_t headNum = 1;
    char inputlayout[] = "SBH";
    const int64_t innerPrecise = 0;
    const int64_t sparseMod = 0;
    const int64_t outDtype = 1;
    const int64_t pseType = 1;
    const int64_t seed = 0;
    const int64_t offset = 0;

    auto tensorDQDesc = TensorDesc({256, 1, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDKDesc = TensorDesc({256, 1, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDVDesc = TensorDesc({256, 1, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorDPseDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDQROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorDKROPEDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFlashAttentionScoreGradV4,
        INPUT(
            tensorQ,
            tensorK,
            tensorV,
            tensorDx,
            nullptr,
            nullptr,
            nullptr,
            tensorAttenMask,
            tensorSoftmaxMax,
            tensorSoftmaxSum,
            nullptr,
            tensorAttentionIn,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            scaleValue,
            keepProb,
            preTokens,
            nextTokens,
            headNum,
            inputlayout,
            nullptr,
            innerPrecise,
            sparseMod,
            pseType,
            seed,
            offset,
            outDtype
        ),
        OUTPUT(
            tensorDQDesc,
            tensorDKDesc,
            tensorDVDesc,
            tensorDQROPEDesc,
            tensorDKROPEDesc,
            tensorDPseDesc,
            nullptr
        )
    );
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}