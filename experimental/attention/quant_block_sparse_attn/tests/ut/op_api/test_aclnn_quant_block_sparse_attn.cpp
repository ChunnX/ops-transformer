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
#include <array>
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_quant_block_sparse_attn.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class quant_block_sparse_attn_opapi_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        SetPlatformSocVersion(SocVersion::ASCEND950);
        cout << "quant_block_sparse_attn_opapi_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "quant_block_sparse_attn_opapi_ut TearDown" << endl;
    }
};

TEST_F(quant_block_sparse_attn_opapi_ut, quant_block_sparse_attn_nullptr_query)
{
    auto tensorQuery = TensorDesc({128, 1, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorKey = TensorDesc({1, 1, 128, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorValue = TensorDesc({1, 1, 128, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorQDescale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.001, 0.002);
    auto tensorKDescale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.001, 0.002);
    auto tensorVDescale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.001, 0.002);
    auto tensorPScale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(100, 200);
    auto tensorSparseIndices = TensorDesc({1, 1, 1, 1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto tensorSparseSeqLen = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{128});
    auto tensorAttenMask = TensorDesc({128, 128}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});
    auto tensorAttentionOut = TensorDesc({128, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto tensorSoftmaxLse = TensorDesc({1, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t maxSeqlenQ = 128;
    int64_t maxSeqlenKv = 128;
    double softmaxScale = 0.08838834764;
    int64_t sparseQBlockSize = 128;
    int64_t sparseKvBlockSize = 128;
    int64_t paBlockStride = 0;
    char *layoutKv = (char *)"PA_BNSD";
    char *layoutQ = (char *)"TND";
    char *layoutSparseIndices = (char *)"B_N_Qb_Kb";
    char *layoutOut = (char *)"TND";
    int64_t quantMode = 1;
    int64_t maskMode = 3;
    bool returnSoftmaxLse = false;

    auto ut = OP_API_UT(aclnnQuantBlockSparseAttn,
                        INPUT(nullptr, tensorKey, tensorValue, tensorQDescale, tensorKDescale, tensorVDescale,
                              tensorPScale, nullptr, nullptr, nullptr, nullptr, tensorSparseIndices, tensorSparseSeqLen,
                              nullptr, tensorAttenMask, nullptr, maxSeqlenQ, maxSeqlenKv, softmaxScale,
                              sparseQBlockSize, sparseKvBlockSize, paBlockStride, layoutKv, layoutQ,
                              layoutSparseIndices, layoutOut, quantMode, maskMode, returnSoftmaxLse),
                        OUTPUT(tensorAttentionOut, tensorSoftmaxLse));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(quant_block_sparse_attn_opapi_ut, quant_block_sparse_attn_nullptr_key)
{
    auto tensorQuery = TensorDesc({128, 1, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorValue = TensorDesc({1, 1, 128, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorQDescale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.001, 0.002);
    auto tensorKDescale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.001, 0.002);
    auto tensorVDescale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.001, 0.002);
    auto tensorPScale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(100, 200);
    auto tensorSparseIndices = TensorDesc({1, 1, 1, 1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto tensorSparseSeqLen = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{128});
    auto tensorAttenMask = TensorDesc({128, 128}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});
    auto tensorAttentionOut = TensorDesc({128, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto tensorSoftmaxLse = TensorDesc({1, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t maxSeqlenQ = 128;
    int64_t maxSeqlenKv = 128;
    double softmaxScale = 0.08838834764;
    int64_t sparseQBlockSize = 128;
    int64_t sparseKvBlockSize = 128;
    int64_t paBlockStride = 0;
    char *layoutKv = (char *)"PA_BNSD";
    char *layoutQ = (char *)"TND";
    char *layoutSparseIndices = (char *)"B_N_Qb_Kb";
    char *layoutOut = (char *)"TND";
    int64_t quantMode = 1;
    int64_t maskMode = 3;
    bool returnSoftmaxLse = false;

    auto ut = OP_API_UT(aclnnQuantBlockSparseAttn,
                        INPUT(tensorQuery, nullptr, tensorValue, tensorQDescale, tensorKDescale, tensorVDescale,
                              tensorPScale, nullptr, nullptr, nullptr, nullptr, tensorSparseIndices, tensorSparseSeqLen,
                              nullptr, tensorAttenMask, nullptr, maxSeqlenQ, maxSeqlenKv, softmaxScale,
                              sparseQBlockSize, sparseKvBlockSize, paBlockStride, layoutKv, layoutQ,
                              layoutSparseIndices, layoutOut, quantMode, maskMode, returnSoftmaxLse),
                        OUTPUT(tensorAttentionOut, tensorSoftmaxLse));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(quant_block_sparse_attn_opapi_ut, quant_block_sparse_attn_nullptr_output)
{
    auto tensorQuery = TensorDesc({128, 1, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorKey = TensorDesc({1, 1, 128, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorValue = TensorDesc({1, 1, 128, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorQDescale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.001, 0.002);
    auto tensorKDescale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.001, 0.002);
    auto tensorVDescale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.001, 0.002);
    auto tensorPScale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(100, 200);
    auto tensorSparseIndices = TensorDesc({1, 1, 1, 1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto tensorSparseSeqLen = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{128});
    auto tensorAttenMask = TensorDesc({128, 128}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});
    auto tensorSoftmaxLse = TensorDesc({1, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t maxSeqlenQ = 128;
    int64_t maxSeqlenKv = 128;
    double softmaxScale = 0.08838834764;
    int64_t sparseQBlockSize = 128;
    int64_t sparseKvBlockSize = 128;
    int64_t paBlockStride = 0;
    char *layoutKv = (char *)"PA_BNSD";
    char *layoutQ = (char *)"TND";
    char *layoutSparseIndices = (char *)"B_N_Qb_Kb";
    char *layoutOut = (char *)"TND";
    int64_t quantMode = 1;
    int64_t maskMode = 3;
    bool returnSoftmaxLse = false;

    auto ut = OP_API_UT(aclnnQuantBlockSparseAttn,
                        INPUT(tensorQuery, tensorKey, tensorValue, tensorQDescale, tensorKDescale, tensorVDescale,
                              tensorPScale, nullptr, nullptr, nullptr, nullptr, tensorSparseIndices, tensorSparseSeqLen,
                              nullptr, tensorAttenMask, nullptr, maxSeqlenQ, maxSeqlenKv, softmaxScale,
                              sparseQBlockSize, sparseKvBlockSize, paBlockStride, layoutKv, layoutQ,
                              layoutSparseIndices, layoutOut, quantMode, maskMode, returnSoftmaxLse),
                        OUTPUT(nullptr, tensorSoftmaxLse));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(quant_block_sparse_attn_opapi_ut, quant_block_sparse_attn_normal_case)
{
    auto tensorQuery = TensorDesc({128, 1, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorKey = TensorDesc({1, 1, 128, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorValue = TensorDesc({1, 1, 128, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorQDescale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.001, 0.002);
    auto tensorKDescale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.001, 0.002);
    auto tensorVDescale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.001, 0.002);
    auto tensorPScale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(100, 200);
    auto tensorSparseIndices = TensorDesc({1, 1, 1, 1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto tensorSparseSeqLen = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{128});
    auto tensorAttenMask = TensorDesc({128, 128}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});
    auto tensorAttentionOut = TensorDesc({128, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto tensorSoftmaxLse = TensorDesc({1, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t maxSeqlenQ = 128;
    int64_t maxSeqlenKv = 128;
    double softmaxScale = 0.08838834764;
    int64_t sparseQBlockSize = 128;
    int64_t sparseKvBlockSize = 128;
    int64_t paBlockStride = 0;
    char *layoutKv = (char *)"PA_BNSD";
    char *layoutQ = (char *)"TND";
    char *layoutSparseIndices = (char *)"B_N_Qb_Kb";
    char *layoutOut = (char *)"TND";
    int64_t quantMode = 1;
    int64_t maskMode = 3;
    bool returnSoftmaxLse = false;

    auto ut = OP_API_UT(aclnnQuantBlockSparseAttn,
                        INPUT(tensorQuery, tensorKey, tensorValue, tensorQDescale, tensorKDescale, tensorVDescale,
                              tensorPScale, nullptr, nullptr, nullptr, nullptr, tensorSparseIndices, tensorSparseSeqLen,
                              nullptr, tensorAttenMask, nullptr, maxSeqlenQ, maxSeqlenKv, softmaxScale,
                              sparseQBlockSize, sparseKvBlockSize, paBlockStride, layoutKv, layoutQ,
                              layoutSparseIndices, layoutOut, quantMode, maskMode, returnSoftmaxLse),
                        OUTPUT(tensorAttentionOut, tensorSoftmaxLse));

    GTEST_SKIP() << "Normal case requires NPU hardware for executor creation";
}
