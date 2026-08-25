/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_sparse_flash_attention_grad.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

namespace {
void DestroyAclTensor(aclTensor *tensor)
{
    Release(tensor);
}

using AclTensorPtr = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>;

AclTensorPtr MakeTensor(const vector<int64_t> &shape, aclDataType dtype)
{
    return AclTensorPtr(TensorDesc(shape, dtype, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
}
} // namespace

class SparseFlashAttentionGradOpapiUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "SparseFlashAttentionGradOpapiUt SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "SparseFlashAttentionGradOpapiUt TearDown" << endl;
    }
};

TEST_F(SparseFlashAttentionGradOpapiUt, A1_bsnd_fp16_value_present)
{
    auto query = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto key = MakeTensor({1, 128, 1, 128}, ACL_FLOAT16);
    auto value = MakeTensor({1, 128, 1, 128}, ACL_FLOAT16);
    auto sparseIndices = MakeTensor({1, 64, 1, 4}, ACL_INT32);
    auto dOut = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto out = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto softmaxMax = MakeTensor({1, 8, 64}, ACL_FLOAT);
    auto softmaxSum = MakeTensor({1, 8, 64}, ACL_FLOAT);
    auto dQueryOut = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto dKeyOut = MakeTensor({1, 128, 1, 128}, ACL_FLOAT16);
    auto dValueOut = MakeTensor({1, 128, 1, 128}, ACL_FLOAT16);

    char layout[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnSparseFlashAttentionGradGetWorkspaceSize(
        query.get(), key.get(), value.get(), sparseIndices.get(), dOut.get(), out.get(), softmaxMax.get(), softmaxSum.get(),
        nullptr, nullptr, nullptr, nullptr, 0.0883883476, 64, layout, 3, INT64_MAX, INT64_MAX, false,
        dQueryOut.get(), dKeyOut.get(), dValueOut.get(), nullptr, nullptr, &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

TEST_F(SparseFlashAttentionGradOpapiUt, A2_bsnd_fp16_value_optional_null)
{
    auto query = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto key = MakeTensor({1, 128, 1, 128}, ACL_FLOAT16);
    auto sparseIndices = MakeTensor({1, 64, 1, 4}, ACL_INT32);
    auto dOut = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto out = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto softmaxMax = MakeTensor({1, 8, 64}, ACL_FLOAT);
    auto softmaxSum = MakeTensor({1, 8, 64}, ACL_FLOAT);
    auto dQueryOut = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto dKeyOut = MakeTensor({1, 128, 1, 128}, ACL_FLOAT16);

    char layout[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnSparseFlashAttentionGradGetWorkspaceSize(
        query.get(), key.get(), nullptr, sparseIndices.get(), dOut.get(), out.get(), softmaxMax.get(), softmaxSum.get(),
        nullptr, nullptr, nullptr, nullptr, 0.0883883476, 64, layout, 3, INT64_MAX, INT64_MAX, false,
        dQueryOut.get(), dKeyOut.get(), nullptr, nullptr, nullptr, &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

TEST_F(SparseFlashAttentionGradOpapiUt, E1_null_query)
{
    auto key = MakeTensor({1, 128, 1, 128}, ACL_FLOAT16);
    auto value = MakeTensor({1, 128, 1, 128}, ACL_FLOAT16);
    auto sparseIndices = MakeTensor({1, 64, 1, 4}, ACL_INT32);
    auto dOut = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto out = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto softmaxMax = MakeTensor({1, 8, 64}, ACL_FLOAT);
    auto softmaxSum = MakeTensor({1, 8, 64}, ACL_FLOAT);
    auto dQueryOut = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto dKeyOut = MakeTensor({1, 128, 1, 128}, ACL_FLOAT16);
    auto dValueOut = MakeTensor({1, 128, 1, 128}, ACL_FLOAT16);

    char layout[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnSparseFlashAttentionGradGetWorkspaceSize(
        nullptr, key.get(), value.get(), sparseIndices.get(), dOut.get(), out.get(), softmaxMax.get(), softmaxSum.get(),
        nullptr, nullptr, nullptr, nullptr, 0.0883883476, 64, layout, 3, INT64_MAX, INT64_MAX, false,
        dQueryOut.get(), dKeyOut.get(), dValueOut.get(), nullptr, nullptr, &workspaceSize, &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

class SparseFlashAttentionGradArch35OpapiUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }
};

TEST_F(SparseFlashAttentionGradArch35OpapiUt, A1_bsnd_bf16_kvmerge)
{
    auto query = MakeTensor({1, 16, 8, 512}, ACL_BF16);
    auto key = MakeTensor({1, 128, 1, 512}, ACL_BF16);
    auto sparseIndices = MakeTensor({1, 16, 1, 1024}, ACL_INT32);
    auto dOut = MakeTensor({1, 16, 8, 512}, ACL_BF16);
    auto out = MakeTensor({1, 16, 8, 512}, ACL_BF16);
    auto softmaxMax = MakeTensor({1, 8, 16}, ACL_FLOAT);
    auto softmaxSum = MakeTensor({1, 8, 16}, ACL_FLOAT);
    auto dQueryOut = MakeTensor({1, 16, 8, 512}, ACL_BF16);
    auto dKeyOut = MakeTensor({1, 128, 1, 512}, ACL_BF16);

    char layout[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = aclnnSparseFlashAttentionGradGetWorkspaceSize(
        query.get(), key.get(), nullptr, sparseIndices.get(), dOut.get(), out.get(), softmaxMax.get(), softmaxSum.get(),
        nullptr, nullptr, nullptr, nullptr, 0.0441941738, 1, layout, 3, INT64_MAX, INT64_MAX, false,
        dQueryOut.get(), dKeyOut.get(), nullptr, nullptr, nullptr, &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

TEST_F(SparseFlashAttentionGradArch35OpapiUt, A2_bsnd_bf16_kvmerge_rope)
{
    auto query = MakeTensor({1, 16, 8, 512}, ACL_BF16);
    auto key = MakeTensor({1, 128, 1, 512}, ACL_BF16);
    auto sparseIndices = MakeTensor({1, 16, 1, 1024}, ACL_INT32);
    auto dOut = MakeTensor({1, 16, 8, 512}, ACL_BF16);
    auto out = MakeTensor({1, 16, 8, 512}, ACL_BF16);
    auto softmaxMax = MakeTensor({1, 8, 16}, ACL_FLOAT);
    auto softmaxSum = MakeTensor({1, 8, 16}, ACL_FLOAT);
    auto queryRope = MakeTensor({1, 16, 8, 64}, ACL_BF16);
    auto keyRope = MakeTensor({1, 128, 1, 64}, ACL_BF16);
    auto dQueryOut = MakeTensor({1, 16, 8, 512}, ACL_BF16);
    auto dKeyOut = MakeTensor({1, 128, 1, 512}, ACL_BF16);
    auto dQueryRopeOut = MakeTensor({1, 16, 8, 64}, ACL_BF16);
    auto dKeyRopeOut = MakeTensor({1, 128, 1, 64}, ACL_BF16);

    char layout[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = aclnnSparseFlashAttentionGradGetWorkspaceSize(
        query.get(), key.get(), nullptr, sparseIndices.get(), dOut.get(), out.get(), softmaxMax.get(), softmaxSum.get(),
        nullptr, nullptr, queryRope.get(), keyRope.get(), 0.0441941738, 1, layout, 3, INT64_MAX, INT64_MAX, false,
        dQueryOut.get(), dKeyOut.get(), nullptr, dQueryRopeOut.get(), dKeyRopeOut.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

TEST_F(SparseFlashAttentionGradArch35OpapiUt, E1_kvmerge_value_null_dvalue_present)
{
    auto query = MakeTensor({1, 16, 8, 512}, ACL_BF16);
    auto key = MakeTensor({1, 128, 1, 512}, ACL_BF16);
    auto sparseIndices = MakeTensor({1, 16, 1, 1024}, ACL_INT32);
    auto dOut = MakeTensor({1, 16, 8, 512}, ACL_BF16);
    auto out = MakeTensor({1, 16, 8, 512}, ACL_BF16);
    auto softmaxMax = MakeTensor({1, 8, 16}, ACL_FLOAT);
    auto softmaxSum = MakeTensor({1, 8, 16}, ACL_FLOAT);
    auto dQueryOut = MakeTensor({1, 16, 8, 512}, ACL_BF16);
    auto dKeyOut = MakeTensor({1, 128, 1, 512}, ACL_BF16);
    auto dValueOut = MakeTensor({1, 128, 1, 512}, ACL_BF16);

    char layout[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = aclnnSparseFlashAttentionGradGetWorkspaceSize(
        query.get(), key.get(), nullptr, sparseIndices.get(), dOut.get(), out.get(), softmaxMax.get(), softmaxSum.get(),
        nullptr, nullptr, nullptr, nullptr, 0.0441941738, 1, layout, 3, INT64_MAX, INT64_MAX, false,
        dQueryOut.get(), dKeyOut.get(), dValueOut.get(), nullptr, nullptr, &workspaceSize, &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

TEST_F(SparseFlashAttentionGradArch35OpapiUt, E2_value_present_dvalue_null)
{
    auto query = MakeTensor({1, 16, 8, 512}, ACL_BF16);
    auto key = MakeTensor({1, 128, 1, 512}, ACL_BF16);
    auto value = MakeTensor({1, 128, 1, 512}, ACL_BF16);
    auto sparseIndices = MakeTensor({1, 16, 1, 1024}, ACL_INT32);
    auto dOut = MakeTensor({1, 16, 8, 512}, ACL_BF16);
    auto out = MakeTensor({1, 16, 8, 512}, ACL_BF16);
    auto softmaxMax = MakeTensor({1, 8, 16}, ACL_FLOAT);
    auto softmaxSum = MakeTensor({1, 8, 16}, ACL_FLOAT);
    auto dQueryOut = MakeTensor({1, 16, 8, 512}, ACL_BF16);
    auto dKeyOut = MakeTensor({1, 128, 1, 512}, ACL_BF16);

    char layout[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = aclnnSparseFlashAttentionGradGetWorkspaceSize(
        query.get(), key.get(), value.get(), sparseIndices.get(), dOut.get(), out.get(), softmaxMax.get(), softmaxSum.get(),
        nullptr, nullptr, nullptr, nullptr, 0.0441941738, 1, layout, 3, INT64_MAX, INT64_MAX, false,
        dQueryOut.get(), dKeyOut.get(), nullptr, nullptr, nullptr, &workspaceSize, &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}
