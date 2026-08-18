/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "../../../op_api/aclnn_all_gather_matmul_v3.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

constexpr int64_t MX_GROUP_SIZE_VAL = static_cast<int64_t>((1ULL << 32) | (1ULL << 16) | 32);

class TestAclnnAllGatherMatmulV3 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(NpuArch::DAV_3510);
        cout << "TestAclnnAllGatherMatmulV3 SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "TestAclnnAllGatherMatmulV3 TearDown" << endl;
    }
};

struct AllGatherMatmulV3AclnnTestParam {
    string case_name;
    vector<int64_t> x1Shape;
    vector<int64_t> x2Shape;
    vector<int64_t> x1ScaleShape;
    vector<int64_t> x2ScaleShape;
    vector<int64_t> outputShape;
    char *group;
    aclDataType x1Dtype;
    aclDataType x2Dtype;
    aclDataType outputDtype;
    bool hasBias;
    aclnnStatus aclnnStatusUt;
};

static AllGatherMatmulV3AclnnTestParam g_casesParams[] = {
    {"test_aclnn_all_gather_matmul_v3_e4m3_e4m3_bf16_true",
     {2048, 4096}, {2560, 4096}, {2048, 64, 2}, {2560, 64, 2}, {8192, 2560},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_BF16, true, ACLNN_SUCCESS},
    {"test_aclnn_all_gather_matmul_v3_e4m3_e4m3_fp16_true",
     {2048, 4096}, {2560, 4096}, {2048, 64, 2}, {2560, 64, 2}, {8192, 2560},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT16, true, ACLNN_SUCCESS},
    {"test_aclnn_all_gather_matmul_v3_e5m2_e5m2_bf16_true",
     {2048, 4096}, {2560, 4096}, {2048, 64, 2}, {2560, 64, 2}, {8192, 2560},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_BF16, true, ACLNN_SUCCESS},
    {"test_aclnn_all_gather_matmul_v3_e5m2_e5m2_fp16_true",
     {2048, 4096}, {2560, 4096}, {2048, 64, 2}, {2560, 64, 2}, {8192, 2560},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT16, true, ACLNN_SUCCESS},
    {"test_aclnn_all_gather_matmul_v3_e4m3_e5m2_bf16_true",
     {2048, 4096}, {2560, 4096}, {2048, 64, 2}, {2560, 64, 2}, {8192, 2560},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_BF16, true, ACLNN_SUCCESS},
    {"test_aclnn_all_gather_matmul_v3_e5m2_e4m3_fp16_true",
     {2048, 4096}, {2560, 4096}, {2048, 64, 2}, {2560, 64, 2}, {8192, 2560},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT16, true, ACLNN_SUCCESS},
    {"test_aclnn_all_gather_matmul_v3_fp4_fp4_bf16_true",
     {2048, 4096}, {2560, 4096}, {2048, 64, 2}, {2560, 64, 2}, {8192, 2560},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT4_E2M1, ACL_FLOAT4_E2M1, ACL_BF16, true, ACLNN_SUCCESS},
    {"test_aclnn_all_gather_matmul_v3_fp4_fp4_fp16_true",
     {2048, 4096}, {2560, 4096}, {2048, 64, 2}, {2560, 64, 2}, {8192, 2560},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT4_E2M1, ACL_FLOAT4_E2M1, ACL_FLOAT16, true, ACLNN_SUCCESS},
    {"test_aclnn_all_gather_matmul_v3_e4m3_e4m3_bf16_no_bias_true",
     {2048, 4096}, {2560, 4096}, {2048, 64, 2}, {2560, 64, 2}, {8192, 2560},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_BF16, false, ACLNN_SUCCESS},
    {"test_aclnn_all_gather_matmul_v3_e4m3_e4m3_bf16_m4096_n6144_true",
     {4096, 4096}, {6144, 4096}, {4096, 64, 2}, {6144, 64, 2}, {16384, 6144},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_BF16, true, ACLNN_SUCCESS},
    {"test_aclnn_all_gather_matmul_v3_e4m3_e4m3_bf16_m8192_true",
     {8192, 4096}, {2560, 4096}, {8192, 64, 2}, {2560, 64, 2}, {32768, 2560},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_BF16, true, ACLNN_SUCCESS},
    {"test_aclnn_all_gather_matmul_v3_e4m3_e4m3_bf16_m2061_true",
     {2061, 4096}, {2560, 4096}, {2061, 64, 2}, {2560, 64, 2}, {8244, 2560},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_BF16, true, ACLNN_SUCCESS},
    {"test_aclnn_all_gather_matmul_v3_invalid_x1_dtype_false",
     {2048, 4096}, {2560, 4096}, {2048, 64, 2}, {2560, 64, 2}, {8192, 2560},
     "all_gather_matmul_v3_test_group",
     ACL_INT8, ACL_FLOAT8_E4M3FN, ACL_BF16, true, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_all_gather_matmul_v3_invalid_x2_dtype_false",
     {2048, 4096}, {2560, 4096}, {2048, 64, 2}, {2560, 64, 2}, {8192, 2560},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, true, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_all_gather_matmul_v3_fp4_mismatch_x1_false",
     {2048, 4096}, {2560, 4096}, {2048, 64, 2}, {2560, 64, 2}, {8192, 2560},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT4_E2M1, ACL_FLOAT8_E4M3FN, ACL_BF16, true, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_all_gather_matmul_v3_fp4_mismatch_x2_false",
     {2048, 4096}, {2560, 4096}, {2048, 64, 2}, {2560, 64, 2}, {8192, 2560},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_BF16, true, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_all_gather_matmul_v3_invalid_output_dtype_false",
     {2048, 4096}, {2560, 4096}, {2048, 64, 2}, {2560, 64, 2}, {8192, 2560},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_INT8, true, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_all_gather_matmul_v3_x1_1d_false",
     {4096}, {2560, 4096}, {2048, 64, 2}, {2560, 64, 2}, {8192, 2560},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_BF16, true, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_all_gather_matmul_v3_k_mismatch_false",
     {2048, 4096}, {2560, 4032}, {2048, 64, 2}, {2560, 63, 2}, {8192, 2560},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_BF16, true, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_all_gather_matmul_v3_k_not_divisible_64_false",
     {2048, 4033}, {2560, 4033}, {2048, 63, 2}, {2560, 63, 2}, {8192, 2560},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_BF16, true, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_all_gather_matmul_v3_x1_scale_wrong_shape_false",
     {2048, 4096}, {2560, 4096}, {1024, 64, 2}, {2560, 64, 2}, {8192, 2560},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_BF16, true, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_all_gather_matmul_v3_x2_scale_wrong_shape_false",
     {2048, 4096}, {2560, 4096}, {2048, 64, 2}, {1280, 64, 2}, {8192, 2560},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_BF16, true, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_all_gather_matmul_v3_output_n_mismatch_false",
     {2048, 4096}, {2560, 4096}, {2048, 64, 2}, {2560, 64, 2}, {8192, 1280},
     "all_gather_matmul_v3_test_group",
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_BF16, true, ACLNN_ERR_PARAM_INVALID},
};

static AllGatherMatmulV3AclnnTestParam g_groupCasesParams[] = {
    {"test_aclnn_all_gather_matmul_v3_group_length_127_true",
     {2048, 4096}, {2560, 4096}, {2048, 64, 2}, {2560, 64, 2}, {8192, 2560},
     "this_is_a_very_long_groupname_"
     "this_is_a_very_long_groupname_"
     "this_is_a_very_long_groupname_"
     "this_is_a_very_long_groupname_"
     "this_is",
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_BF16, true, ACLNN_SUCCESS},
    {"test_aclnn_all_gather_matmul_v3_group_length_128_false",
     {2048, 4096}, {2560, 4096}, {2048, 64, 2}, {2560, 64, 2}, {8192, 2560},
     "this_is_a_very_long_groupname_"
     "this_is_a_very_long_groupname_"
     "this_is_a_very_long_groupname_"
     "this_is_a_very_long_groupname_"
     "this_is_",
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_BF16, true, ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_all_gather_matmul_v3_group_length_129_false",
     {2048, 4096}, {2560, 4096}, {2048, 64, 2}, {2560, 64, 2}, {8192, 2560},
     "this_is_a_very_long_groupname_"
     "this_is_a_very_long_groupname_"
     "this_is_a_very_long_groupname_"
     "this_is_a_very_long_groupname_"
     "this_is_a",
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_BF16, true, ACLNN_ERR_PARAM_NULLPTR},
};

static void TestOneParamCase(const AllGatherMatmulV3AclnnTestParam &param)
{
    std::cout << "run case " << param.case_name << std::endl;
    if (param.group == nullptr) {
        std::cerr << "[ERROR]: group is null" << std::endl;
        return;
    }
    vector<int64_t> x1Shape = param.x1Shape;
    vector<int64_t> x2Shape = param.x2Shape;
    vector<int64_t> x1ScaleShape = param.x1ScaleShape;
    vector<int64_t> x2ScaleShape = param.x2ScaleShape;
    vector<int64_t> outputShape = param.outputShape;
    char *group = param.group;
    aclDataType x1Dtype = param.x1Dtype;
    aclDataType x2Dtype = param.x2Dtype;
    aclDataType outputDtype = param.outputDtype;
    aclnnStatus retStatus = param.aclnnStatusUt;
    int64_t rankSize = 4;
    int64_t hcclBufferSize = 200;
    int64_t groupSize = MX_GROUP_SIZE_VAL;
    const char *commMode = "urma";

    TensorDesc contextDesc = TensorDesc({100}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc x1Desc = TensorDesc(x1Shape, x1Dtype, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc(x2Shape, x2Dtype, ACL_FORMAT_ND);
    int64_t mVal = x1Shape.size() >= 1 ? x1Shape[0] : 0;
    int64_t kVal = x1Shape.size() >= 2 ? x1Shape[1] : 0;
    TensorDesc biasDesc = TensorDesc({param.x2Shape.size() >= 1 ? param.x2Shape[0] : 0}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x1ScaleDesc = TensorDesc(x1ScaleShape, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2ScaleDesc = TensorDesc(x2ScaleShape, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc outputDesc = TensorDesc(outputShape, outputDtype, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({rankSize * mVal, kVal}, x1Dtype, ACL_FORMAT_ND);

    aclnnStatus aclRet;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    if (param.hasBias) {
        auto ut = OP_API_UT(aclnnAllGatherQuantMatmulV3,
                            INPUT(contextDesc, x1Desc, x2Desc, biasDesc, x1ScaleDesc, x2ScaleDesc,
                                  group, rankSize, hcclBufferSize, groupSize, commMode),
                            OUTPUT(outputDesc, gatherOutDesc));
        aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    } else {
        auto ut = OP_API_UT(aclnnAllGatherQuantMatmulV3,
                            INPUT(contextDesc, x1Desc, x2Desc, nullptr, x1ScaleDesc, x2ScaleDesc,
                                  group, rankSize, hcclBufferSize, groupSize, commMode),
                            OUTPUT(outputDesc, gatherOutDesc));
        aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    }

    if (retStatus == ACLNN_SUCCESS) {
        EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, retStatus);
    }
}

TEST_F(TestAclnnAllGatherMatmulV3, CasesParamsTest)
{
    if (std::size(g_casesParams) != 0) {
        uint64_t numCases = sizeof(g_casesParams) / sizeof(g_casesParams[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestOneParamCase(g_casesParams[idx]);
        }
    }
}

TEST_F(TestAclnnAllGatherMatmulV3, GroupCasesParamsTest)
{
    if (std::size(g_groupCasesParams) != 0) {
        uint64_t numCases = sizeof(g_groupCasesParams) / sizeof(g_groupCasesParams[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestOneParamCase(g_groupCasesParams[idx]);
        }
    }
}

TEST_F(TestAclnnAllGatherMatmulV3, NullContextTest)
{
    TensorDesc x1Desc = TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({2560, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1ScaleDesc = TensorDesc({2048, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2ScaleDesc = TensorDesc({2560, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc outputDesc = TensorDesc({8192, 2560}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({8192, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    int64_t rankSize = 4;
    int64_t hcclBufferSize = 200;
    int64_t groupSize = MX_GROUP_SIZE_VAL;
    auto ut = OP_API_UT(aclnnAllGatherQuantMatmulV3,
                        INPUT(nullptr, x1Desc, x2Desc, nullptr, x1ScaleDesc, x2ScaleDesc,
                              "test_group", rankSize, hcclBufferSize, groupSize, "urma"),
                        OUTPUT(outputDesc, gatherOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(TestAclnnAllGatherMatmulV3, NullX1Test)
{
    TensorDesc contextDesc = TensorDesc({100}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({2560, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1ScaleDesc = TensorDesc({2048, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2ScaleDesc = TensorDesc({2560, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc outputDesc = TensorDesc({8192, 2560}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({8192, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    int64_t rankSize = 4;
    int64_t hcclBufferSize = 200;
    int64_t groupSize = MX_GROUP_SIZE_VAL;
    auto ut = OP_API_UT(aclnnAllGatherQuantMatmulV3,
                        INPUT(contextDesc, nullptr, x2Desc, nullptr, x1ScaleDesc, x2ScaleDesc,
                              "test_group", rankSize, hcclBufferSize, groupSize, "urma"),
                        OUTPUT(outputDesc, gatherOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(TestAclnnAllGatherMatmulV3, NullX2Test)
{
    TensorDesc contextDesc = TensorDesc({100}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc x1Desc = TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1ScaleDesc = TensorDesc({2048, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2ScaleDesc = TensorDesc({2560, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc outputDesc = TensorDesc({8192, 2560}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({8192, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    int64_t rankSize = 4;
    int64_t hcclBufferSize = 200;
    int64_t groupSize = MX_GROUP_SIZE_VAL;
    auto ut = OP_API_UT(aclnnAllGatherQuantMatmulV3,
                        INPUT(contextDesc, x1Desc, nullptr, nullptr, x1ScaleDesc, x2ScaleDesc,
                              "test_group", rankSize, hcclBufferSize, groupSize, "urma"),
                        OUTPUT(outputDesc, gatherOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(TestAclnnAllGatherMatmulV3, NullOutputTest)
{
    TensorDesc contextDesc = TensorDesc({100}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc x1Desc = TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({2560, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1ScaleDesc = TensorDesc({2048, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2ScaleDesc = TensorDesc({2560, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({8192, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    int64_t rankSize = 4;
    int64_t hcclBufferSize = 200;
    int64_t groupSize = MX_GROUP_SIZE_VAL;
    auto ut = OP_API_UT(aclnnAllGatherQuantMatmulV3,
                        INPUT(contextDesc, x1Desc, x2Desc, nullptr, x1ScaleDesc, x2ScaleDesc,
                              "test_group", rankSize, hcclBufferSize, groupSize, "urma"),
                        OUTPUT(nullptr, gatherOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(TestAclnnAllGatherMatmulV3, NullGroupTest)
{
    TensorDesc contextDesc = TensorDesc({100}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc x1Desc = TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({2560, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1ScaleDesc = TensorDesc({2048, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2ScaleDesc = TensorDesc({2560, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc outputDesc = TensorDesc({8192, 2560}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({8192, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    int64_t rankSize = 4;
    int64_t hcclBufferSize = 200;
    int64_t groupSize = MX_GROUP_SIZE_VAL;
    auto ut = OP_API_UT(aclnnAllGatherQuantMatmulV3,
                        INPUT(contextDesc, x1Desc, x2Desc, nullptr, x1ScaleDesc, x2ScaleDesc,
                              (const char *)nullptr, rankSize, hcclBufferSize, groupSize, "urma"),
                        OUTPUT(outputDesc, gatherOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(TestAclnnAllGatherMatmulV3, NullCommModeTest)
{
    TensorDesc contextDesc = TensorDesc({100}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc x1Desc = TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({2560, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1ScaleDesc = TensorDesc({2048, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2ScaleDesc = TensorDesc({2560, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc outputDesc = TensorDesc({8192, 2560}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({8192, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    int64_t rankSize = 4;
    int64_t hcclBufferSize = 200;
    int64_t groupSize = MX_GROUP_SIZE_VAL;
    auto ut = OP_API_UT(aclnnAllGatherQuantMatmulV3,
                        INPUT(contextDesc, x1Desc, x2Desc, nullptr, x1ScaleDesc, x2ScaleDesc,
                              "test_group", rankSize, hcclBufferSize, groupSize, (const char *)nullptr),
                        OUTPUT(outputDesc, gatherOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(TestAclnnAllGatherMatmulV3, InvalidX1ScaleDtypeTest)
{
    TensorDesc contextDesc = TensorDesc({100}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc x1Desc = TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({2560, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1ScaleDesc = TensorDesc({2048, 64, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2ScaleDesc = TensorDesc({2560, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc outputDesc = TensorDesc({8192, 2560}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({8192, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    int64_t rankSize = 4;
    int64_t hcclBufferSize = 200;
    int64_t groupSize = MX_GROUP_SIZE_VAL;
    auto ut = OP_API_UT(aclnnAllGatherQuantMatmulV3,
                        INPUT(contextDesc, x1Desc, x2Desc, nullptr, x1ScaleDesc, x2ScaleDesc,
                              "test_group", rankSize, hcclBufferSize, groupSize, "urma"),
                        OUTPUT(outputDesc, gatherOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(TestAclnnAllGatherMatmulV3, InvalidX2ScaleDtypeTest)
{
    TensorDesc contextDesc = TensorDesc({100}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc x1Desc = TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({2560, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1ScaleDesc = TensorDesc({2048, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2ScaleDesc = TensorDesc({2560, 64, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc outputDesc = TensorDesc({8192, 2560}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({8192, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    int64_t rankSize = 4;
    int64_t hcclBufferSize = 200;
    int64_t groupSize = MX_GROUP_SIZE_VAL;
    auto ut = OP_API_UT(aclnnAllGatherQuantMatmulV3,
                        INPUT(contextDesc, x1Desc, x2Desc, nullptr, x1ScaleDesc, x2ScaleDesc,
                              "test_group", rankSize, hcclBufferSize, groupSize, "urma"),
                        OUTPUT(outputDesc, gatherOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(TestAclnnAllGatherMatmulV3, InvalidBiasDtypeTest)
{
    TensorDesc contextDesc = TensorDesc({100}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc x1Desc = TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({2560, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc biasDesc = TensorDesc({2560}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x1ScaleDesc = TensorDesc({2048, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2ScaleDesc = TensorDesc({2560, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc outputDesc = TensorDesc({8192, 2560}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({8192, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    int64_t rankSize = 4;
    int64_t hcclBufferSize = 200;
    int64_t groupSize = MX_GROUP_SIZE_VAL;
    auto ut = OP_API_UT(aclnnAllGatherQuantMatmulV3,
                        INPUT(contextDesc, x1Desc, x2Desc, biasDesc, x1ScaleDesc, x2ScaleDesc,
                              "test_group", rankSize, hcclBufferSize, groupSize, "urma"),
                        OUTPUT(outputDesc, gatherOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(TestAclnnAllGatherMatmulV3, X1FormatNzTest)
{
    TensorDesc contextDesc = TensorDesc({100}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc x1Desc = TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_FRACTAL_NZ);
    TensorDesc x2Desc = TensorDesc({2560, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1ScaleDesc = TensorDesc({2048, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2ScaleDesc = TensorDesc({2560, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc outputDesc = TensorDesc({8192, 2560}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({8192, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    int64_t rankSize = 4;
    int64_t hcclBufferSize = 200;
    int64_t groupSize = MX_GROUP_SIZE_VAL;
    auto ut = OP_API_UT(aclnnAllGatherQuantMatmulV3,
                        INPUT(contextDesc, x1Desc, x2Desc, nullptr, x1ScaleDesc, x2ScaleDesc,
                              "test_group", rankSize, hcclBufferSize, groupSize, "urma"),
                        OUTPUT(outputDesc, gatherOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
