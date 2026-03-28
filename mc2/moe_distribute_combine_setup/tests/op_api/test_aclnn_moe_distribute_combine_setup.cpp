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
 
#include <iostream>
#include <thread>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_moe_distribute_combine_setup.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

namespace {

using namespace op;

class test_aclnn_moe_distribute_combine_setup : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(NpuArch::DAV_3510);
        std::cout << "test_aclnn_moe_distribute_combine_setup SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        std::cout << "test_aclnn_moe_distribute_combine_setup TearDown" << std::endl;
    }
};

// 定义用例信息结构体
struct MoeDistributeCombineSetupAclnnTestParam {
    string case_name;

    // 输入信息shape
    std::vector<int64_t> expandXShape;
    std::vector<int64_t> quantExpandXShape;
    std::vector<int64_t> expertIdsShape;
    std::vector<int64_t> expandIdxShape;
    std::vector<int64_t> expertScalesShape;
    std::vector<int64_t> commCmdInfoShape;
    std::vector<int64_t> xActiveMaskShape;
    std::vector<int64_t> sharedExpertXShape;

    // 输入信息dtype
    aclDataType expandXDtype;
    aclDataType quantExpandXDtype;
    aclDataType expertIdsDtype;
    aclDataType expandIdxDtype;
    aclDataType expertScalesDtype;
    aclDataType commCmdInfoDtype;
    aclDataType xActiveMaskDtype;
    aclDataType sharedExpertXDtype;

    // 输入信息format
    aclFormat expandXFormat;
    aclFormat quantExpandXFormat;
    aclFormat expertIdsFormat;
    aclFormat expandIdxFormat;
    aclFormat expertScalesFormat;
    aclFormat commCmdInfoFormat;
    aclFormat xActiveMaskFormat;
    aclFormat sharedExpertXFormat;

    // 输入Attrs
    int64_t epWorldSize;
    int64_t epRankId;
    int64_t moeExpertNum;
    int64_t expertShardType;
    int64_t sharedExpertNum;
    int64_t sharedExpertRankNum;
    int64_t globalBs;
    int64_t commQuantMode;
    int64_t commType;

    // 输出信息shape
    std::vector<int64_t> outputXShape;

    // 输出信息dtype
    aclDataType outputXDtype;

    // 输出信息format
    aclFormat outputXFormat;

    // 预期结果
    aclnnStatus expectStatus;
};

// 用例列表集
static MoeDistributeCombineSetupAclnnTestParam test_cases[] = {
    //===============================================Ascend910C===================================================
    {"test_aclnn_moe_distribute_combine_setup",
     {1536, 4096}, {1536, 6144}, {16, 6}, {96}, {16, 6}, {24832}, {}, {},
     ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     16, 0, 256, 0, 0, 0, 0, 0, 2,
     {16, 4096}, ACL_FLOAT16, ACL_FORMAT_ND, ACLNN_SUCCESS},
};

static void TestOneParamCase(const MoeDistributeCombineSetupAclnnTestParam &param)
{
    std::cout << "run case " << param.case_name << std::endl;

    // 输入信息
    TensorDesc expandX = TensorDesc(param.expandXShape, param.expandXDtype, param.expandXFormat);
    TensorDesc quantExpandX = TensorDesc(param.quantExpandXShape, param.quantExpandXDtype, param.quantExpandXFormat);
    TensorDesc expertIds = TensorDesc(param.expertIdsShape, param.expertIdsDtype, param.expertIdsFormat);
    TensorDesc expandIdx = TensorDesc(param.expandIdxShape, param.expandIdxDtype, param.expandIdxFormat);
    TensorDesc expertScales = TensorDesc(param.expertScalesShape, param.expertScalesDtype, param.expertScalesFormat);
    TensorDesc commCmdInfo = TensorDesc(param.commCmdInfoShape, param.commCmdInfoDtype, param.commCmdInfoFormat);
    TensorDesc xActiveMask = TensorDesc(param.xActiveMaskShape, param.xActiveMaskDtype, param.xActiveMaskFormat);
    TensorDesc sharedExpertX =
        TensorDesc(param.sharedExpertXShape, param.sharedExpertXDtype, param.sharedExpertXFormat);
    const char *groupEp = "test_group";
    const char *commAlg = nullptr;

    // 输出信息
    TensorDesc outputX = TensorDesc(param.outputXShape, param.outputXDtype, param.outputXFormat);

    auto ut = OP_API_UT(aclnnMoeDistributeCombineSetup,
                        INPUT(expandX, quantExpandX, expertIds, expandIdx, expertScales, commCmdInfo, xActiveMask,
                              sharedExpertX, groupEp, param.epWorldSize, param.epRankId, param.moeExpertNum,
                              param.expertShardType, param.sharedExpertNum, param.sharedExpertRankNum, param.globalBs,
                              param.commQuantMode, param.commType, commAlg),
                        OUTPUT(outputX));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, param.expectStatus);
}

TEST_F(test_aclnn_moe_distribute_combine_setup, test_cases)
{
    if (std::size(test_cases) != 0) {
        uint64_t numCases = sizeof(test_cases) / sizeof(test_cases[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestOneParamCase(test_cases[idx]);
        }
    }
}

} // namespace
