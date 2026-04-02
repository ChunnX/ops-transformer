/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_distribute_barrier_extend.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class L2AclnnDistributeBarrierExtendTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        cout << "L2AclnnDistributeBarrierExtendTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2AclnnDistributeBarrierExtendTest TearDown" << endl;
    }
};

TEST_F(L2AclnnDistributeBarrierExtendTest, TestAclnnDistributeBarrierExtendFirstApi)
{
    TensorDesc context = TensorDesc({1, 2052}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierExtend,
                        INPUT(context, xRef, nullptr, nullptr, "test_distribute_barrier_extend", worldSize),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnDistributeBarrierExtendTest, TestAclnnDistributeBarrierExtendFirstApiNullptrGroup)
{
    TensorDesc context = TensorDesc({1, 2052}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierExtend,
                        INPUT(context, xRef, nullptr, nullptr, nullptr, worldSize),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2AclnnDistributeBarrierExtendTest, TestAclnnDistributeBarrierExtendFirstApiGroupMin)
{
    TensorDesc context = TensorDesc({1, 2052}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierExtend,
                        INPUT(context, xRef, nullptr, nullptr, "", worldSize),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnDistributeBarrierExtendTest, TestAclnnDistributeBarrierExtendFirstApiTimeOut)
{
    TensorDesc context = TensorDesc({1, 2052}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc timeOut = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierExtend,
                        INPUT(context, xRef, timeOut, nullptr, "test_distribute_barrier_extend", worldSize),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnDistributeBarrierExtendTest, TestAclnnDistributeBarrierExtendFirstApiElasticInfo)
{
    TensorDesc context = TensorDesc({1, 2052}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc elasticInfo = TensorDesc({36}, ACL_INT32, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierExtend,
                        INPUT(context, xRef, nullptr, elasticInfo, "test_distribute_barrier_extend", worldSize),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnDistributeBarrierExtendTest, TestAclnnDistributeBarrierExtendFirstApiTimeOutElasticInfo)
{
    TensorDesc context = TensorDesc({1, 2052}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc timeOut = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc elasticInfo = TensorDesc({36}, ACL_INT32, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierExtend,
                        INPUT(context, xRef, timeOut, elasticInfo, "test_distribute_barrier_extend", worldSize),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}