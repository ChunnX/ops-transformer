/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_moe_distribute_combine_setup.cpp
 * \brief aclnn ut
 */

#include <float.h>
#include <array>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_moe_distribute_combine_setup.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;

namespace MoeDistributeCombineSetupUT {
class TestAclnnMoeDistributeCombineSetup : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(NpuArch::DAV_3510);
        std::cout << "TestAclnnMoeDistributeCombineSetup SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        std::cout << "TestAclnnMoeDistributeCombineSetup TearDown" << std::endl;
    }
};

struct MoeDistributeCombineSetupAclnnTestParam {
    string caseName;

    // shape
    std::vector<int64_t> expandXShape;
    std::vector<int64_t> expertIdsShape;
    std::vector<int64_t> assistInfoForCombineShape;
    std::vector<int64_t> quantExpandXOutShape;
    std::vector<int64_t> commCmdInfoOutShape;

    // 通信域标识
    char *groupEp;

    // dtype
    aclDataType expandXDtype;
    aclDataType expertIdsDtype;
    aclDataType assistInfoForCombineDtype;
    aclDataType quantExpandXOutDtype;
    aclDataType commCmdInfoOutDtype;

    // format
    aclFormat expandXFormat;
    aclFormat expertIdsFormat;
    aclFormat assistInfoForCombineFormat;
    aclFormat quantExpandXOutFormat;
    aclFormat commCmdInfoOutFormat;

    // 返回状态
    aclnnStatus aclnnStatusUt;
};

// 用例列表集
static MoeDistributeCombineSetupAclnnTestParam test_cases[] = {
    // 正常用例
    {"test_aclnn_moe_distribute_combine_setup_1",
     {96, 4096}, {8, 6}, {12288}, {96, 6144}, {1568},
     "MoeDistributeCombineSetup_test_groupEp",
     ACL_FLOAT16, ACL_INT32, ACL_INT32, ACL_INT8, ACL_INT32,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, 
     ACLNN_SUCCESS},
};

static void TestOneParamCase(const MoeDistributeCombineSetupAclnnTestParam &param)
{
    std::cout << "run case " << param.caseName << std::endl;
    if (param.groupEp == nullptr) {
        std::cerr << "[ERROR]: group is null" << std::endl;
        return;
    }
    std::vector<int64_t> expandXShape = param.expandXShape;
    std::vector<int64_t> expertIdsShape = param.expertIdsShape;
    std::vector<int64_t> assistInfoForCombineShape = param.assistInfoForCombineShape;
    std::vector<int64_t> quantExpandXOutShape = param.quantExpandXOutShape;
    std::vector<int64_t> commCmdInfoOutShape = param.commCmdInfoOutShape;
    char *groupEp = param.groupEp;
    aclDataType expandXDtype = param.expandXDtype;
    aclDataType expertIdsDtype = param.expertIdsDtype;
    aclDataType assistInfoForCombineDtype = param.assistInfoForCombineDtype;
    aclDataType quantExpandXOutDtype = param.quantExpandXOutDtype;
    aclDataType commCmdInfoOutDtype = param.commCmdInfoOutDtype;
    aclFormat expandXFormat = param.expandXFormat;
    aclFormat expertIdsFormat = param.expertIdsFormat;
    aclFormat assistInfoForCombineFormat = param.assistInfoForCombineFormat;
    aclFormat quantExpandXOutFormat = param.quantExpandXOutFormat;
    aclFormat commCmdInfoOutFormat = param.commCmdInfoOutFormat;
    aclnnStatus retStatus = param.aclnnStatusUt;
    // 封装
    TensorDesc expandX = TensorDesc(expandXShape, expandXDtype, expandXFormat);
    TensorDesc expertIds = TensorDesc(expertIdsShape, expertIdsDtype, expertIdsFormat);
    TensorDesc assistInfoForCombine = TensorDesc(assistInfoForCombineShape, assistInfoForCombineDtype, assistInfoForCombineFormat);
    TensorDesc quantExpandXOut = TensorDesc(quantExpandXOutShape, quantExpandXOutDtype, quantExpandXOutFormat);
    TensorDesc commCmdInfoOut = TensorDesc(commCmdInfoOutShape, commCmdInfoOutDtype, commCmdInfoOutFormat);
    auto ut = OP_API_UT(aclnnMoeDistributeCombineSetup,
                        INPUT(expandX, expertIds, assistInfoForCombine, groupEp, 2, 0, 32, 0, 0, 0, 0, 0, 2, ""),
                        OUTPUT(quantExpandXOut, commCmdInfoOut));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    if (retStatus == ACLNN_SUCCESS) {
        EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, retStatus);
    }
}

TEST_F(TestAclnnMoeDistributeCombineSetup, test_cases)
{
    if (std::size(test_cases) != 0) {
        uint64_t numCases = sizeof(test_cases) / sizeof(test_cases[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestOneParamCase(test_cases[idx]);
        }
    }
}

} // namespace MoeDistributeCombineSetupUT
