/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <vector>
#include <cstdint>
#include "gtest/gtest.h"
#include "aclnnop/aclnn_dense_lightning_indexer_softmax_lse.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
using namespace std;
using namespace op;

namespace {
void DestroyAclTensor(aclTensor *tensor)
{
    Release(tensor);
}
} // namespace

class dense_lightning_indexer_softmax_lse_opapi_ut : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        cout << "dense_lightning_indexer_softmax_lse_opapi_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "dense_lightning_indexer_softmax_lse_opapi_ut TearDown" << endl;
    }
};

// Null queryIndex - should fail
TEST_F(dense_lightning_indexer_softmax_lse_opapi_ut, dense_lightning_indexer_softmax_lse_aclnn_0)
{
    char layout[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        layout,
        3,
        INT64_MAX,
        INT64_MAX,
        nullptr,
        nullptr,
        &workspaceSize,
        &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// Missing keyIndex - should fail
TEST_F(dense_lightning_indexer_softmax_lse_opapi_ut, dense_lightning_indexer_softmax_lse_aclnn_1)
{
    char layout[] = "BSND";
    auto queryIndex = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weight = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto softmaxMaxOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto softmaxSumOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(
        queryIndex.get(),
        nullptr,
        weight.get(),
        nullptr,
        nullptr,
        layout,
        3,
        INT64_MAX,
        INT64_MAX,
        softmaxMaxOut.get(),
        softmaxSumOut.get(),
        &workspaceSize,
        &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// Missing weight - should fail
TEST_F(dense_lightning_indexer_softmax_lse_opapi_ut, dense_lightning_indexer_softmax_lse_aclnn_2)
{
    char layout[] = "BSND";
    auto queryIndex = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto keyIndex = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 64, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto softmaxMaxOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto softmaxSumOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(
        queryIndex.get(),
        keyIndex.get(),
        nullptr,
        nullptr,
        nullptr,
        layout,
        3,
        INT64_MAX,
        INT64_MAX,
        softmaxMaxOut.get(),
        softmaxSumOut.get(),
        &workspaceSize,
        &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// Missing softmaxMaxOut - should fail
TEST_F(dense_lightning_indexer_softmax_lse_opapi_ut, dense_lightning_indexer_softmax_lse_aclnn_3)
{
    char layout[] = "BSND";
    auto queryIndex = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto keyIndex = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 64, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weight = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto softmaxSumOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(
        queryIndex.get(),
        keyIndex.get(),
        weight.get(),
        nullptr,
        nullptr,
        layout,
        3,
        INT64_MAX,
        INT64_MAX,
        nullptr,
        softmaxSumOut.get(),
        &workspaceSize,
        &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// Missing softmaxSumOut - should fail
TEST_F(dense_lightning_indexer_softmax_lse_opapi_ut, dense_lightning_indexer_softmax_lse_aclnn_4)
{
    char layout[] = "BSND";
    auto queryIndex = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto keyIndex = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 64, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weight = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto softmaxMaxOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(
        queryIndex.get(),
        keyIndex.get(),
        weight.get(),
        nullptr,
        nullptr,
        layout,
        3,
        INT64_MAX,
        INT64_MAX,
        softmaxMaxOut.get(),
        nullptr,
        &workspaceSize,
        &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// TND layout valid input - basic smoke test
TEST_F(dense_lightning_indexer_softmax_lse_opapi_ut, dense_lightning_indexer_softmax_lse_aclnn_5)
{
    char layout[] = "TND";
    auto queryIndex = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({128, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto keyIndex = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weight = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({128, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto softmaxMaxOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 128}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto softmaxSumOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 128}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(
        queryIndex.get(),
        keyIndex.get(),
        weight.get(),
        nullptr,
        nullptr,
        layout,
        3,
        INT64_MAX,
        INT64_MAX,
        softmaxMaxOut.get(),
        softmaxSumOut.get(),
        &workspaceSize,
        &executor);

    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

// BSND layout valid input - basic smoke test
TEST_F(dense_lightning_indexer_softmax_lse_opapi_ut, dense_lightning_indexer_softmax_lse_aclnn_6)
{
    char layout[] = "BSND";
    auto queryIndex = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 128, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto keyIndex = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 256, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weight = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 128, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto softmaxMaxOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 1, 128}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto softmaxSumOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 1, 128}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(
        queryIndex.get(),
        keyIndex.get(),
        weight.get(),
        nullptr,
        nullptr,
        layout,
        3,
        INT64_MAX,
        INT64_MAX,
        softmaxMaxOut.get(),
        softmaxSumOut.get(),
        &workspaceSize,
        &executor);

    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

// BF16 data type test
TEST_F(dense_lightning_indexer_softmax_lse_opapi_ut, dense_lightning_indexer_softmax_lse_aclnn_7)
{
    char layout[] = "BSND";
    auto queryIndex = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 64, 16, 128}, ACL_BF16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto keyIndex = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 128, 1, 128}, ACL_BF16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weight = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 64, 16}, ACL_BF16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto softmaxMaxOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto softmaxSumOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(
        queryIndex.get(),
        keyIndex.get(),
        weight.get(),
        nullptr,
        nullptr,
        layout,
        3,
        INT64_MAX,
        INT64_MAX,
        softmaxMaxOut.get(),
        softmaxSumOut.get(),
        &workspaceSize,
        &executor);

    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

// Invalid sparseMode - should fail
TEST_F(dense_lightning_indexer_softmax_lse_opapi_ut, dense_lightning_indexer_softmax_lse_aclnn_8)
{
    char layout[] = "BSND";
    auto queryIndex = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto keyIndex = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 64, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weight = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto softmaxMaxOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto softmaxSumOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(
        queryIndex.get(),
        keyIndex.get(),
        weight.get(),
        nullptr,
        nullptr,
        layout,
        0,
        INT64_MAX,
        INT64_MAX,
        softmaxMaxOut.get(),
        softmaxSumOut.get(),
        &workspaceSize,
        &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// Large shape test for A5
TEST_F(dense_lightning_indexer_softmax_lse_opapi_ut, dense_lightning_indexer_softmax_lse_aclnn_9)
{
    char layout[] = "BSND";
    auto queryIndex = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 4096, 64, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto keyIndex = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8192, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weight = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 4096, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto softmaxMaxOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 1, 4096}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto softmaxSumOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 1, 4096}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(
        queryIndex.get(),
        keyIndex.get(),
        weight.get(),
        nullptr,
        nullptr,
        layout,
        3,
        INT64_MAX,
        INT64_MAX,
        softmaxMaxOut.get(),
        softmaxSumOut.get(),
        &workspaceSize,
        &executor);

    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_NE(executor, nullptr);
}
