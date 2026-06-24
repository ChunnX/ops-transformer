/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "../../../op_api/aclnn_engram_fetch.h"

using namespace op;
using namespace std;

class AclnnEngramFetchTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        std::cout << "EngramFetch AclnnEngramFetchTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        std::cout << "EngramFetch AclnnEngramFetchTest TearDown" << std::endl;
    }
};

TEST_F(AclnnEngramFetchTest, ascend950_success) {
    auto commContext_desc = TensorDesc({2048}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);
    auto fetched_desc = TensorDesc({8, 512}, ACL_BF16, ACL_FORMAT_ND);

    int32_t hiddenSize = 512;
    int64_t numEntriesPerRank = 4;
    int64_t outputDtype = 1;

    auto ut = OP_API_UT(aclnnEngramFetch,
                        INPUT(commContext_desc, indices_desc, fetched_desc, hiddenSize, numEntriesPerRank, outputDtype),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AclnnEngramFetchTest, ascend950_nullptr_commContext) {
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);
    auto fetched_desc = TensorDesc({8, 512}, ACL_BF16, ACL_FORMAT_ND);

    int32_t hiddenSize = 512;
    int64_t numEntriesPerRank = 4;
    int64_t outputDtype = 1;

    auto ut = OP_API_UT(aclnnEngramFetch,
                        INPUT(nullptr, indices_desc, fetched_desc, hiddenSize, numEntriesPerRank, outputDtype),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnEngramFetchTest, ascend950_nullptr_indices) {
    auto commContext_desc = TensorDesc({2048}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto fetched_desc = TensorDesc({8, 512}, ACL_BF16, ACL_FORMAT_ND);

    int32_t hiddenSize = 512;
    int64_t numEntriesPerRank = 4;
    int64_t outputDtype = 1;

    auto ut = OP_API_UT(aclnnEngramFetch,
                        INPUT(commContext_desc, nullptr, fetched_desc, hiddenSize, numEntriesPerRank, outputDtype),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnEngramFetchTest, ascend950_nullptr_fetched) {
    auto commContext_desc = TensorDesc({2048}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);

    int32_t hiddenSize = 512;
    int64_t numEntriesPerRank = 4;
    int64_t outputDtype = 1;

    auto ut = OP_API_UT(aclnnEngramFetch,
                        INPUT(commContext_desc, indices_desc, nullptr, hiddenSize, numEntriesPerRank, outputDtype),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
