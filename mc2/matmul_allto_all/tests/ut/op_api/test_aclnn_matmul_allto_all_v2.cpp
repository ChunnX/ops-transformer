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
#include "../../../op_api/aclnn_matmul_allto_all_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class TestAclnnMatmulAlltoAllV2 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "TestAclnnMatmulAlltoAllV2 SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "TestAclnnMatmulAlltoAllV2 TearDown" << endl;
    }
};

// ut用例结构体，在V1基础上新增commMode字段
struct MatmulAlltoAllV2AclnnTestParam {
    string caseName;
    int worldSize;
    vector<int64_t> x1Shape;
    vector<int64_t> x2Shape;
    vector<int64_t> biasShape;
    vector<int64_t> outputShape;
    aclDataType x1Dtype;
    aclDataType x2Dtype;
    aclDataType biasDtype;
    aclDataType outputDtype;
    aclFormat x1Format;
    aclFormat x2Format;
    aclFormat biasFormat;
    aclFormat outputFormat;
    vector<int64_t> alltoAllAxesOptional;
    char* group;
    const char* commMode; // 新增：通信引擎参数
    bool transposeX1;
    bool transposeX2;
    aclnnStatus aclnnStatusUt;
    bool x1Null = false;
    bool x2Null = false;
    bool outputNull = false;
};

static aclTensor* CreateAclTensor(const std::vector<int64_t>& shape, aclDataType dataType, aclFormat format)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
        strides.data(), 0, format, shape.data(), shape.size(), nullptr);
    assert(tensor != nullptr);
    return tensor;
}

static void TestV2OneParamCase(const MatmulAlltoAllV2AclnnTestParam& param)
{
    std::cout << "run case " << param.caseName << std::endl;
    vector<int64_t> x1Shape = param.x1Shape;
    vector<int64_t> x2Shape = param.x2Shape;
    vector<int64_t> biasShape = param.biasShape;
    vector<int64_t> outputShape = param.outputShape;
    aclDataType x1Dtype = param.x1Dtype;
    aclDataType x2Dtype = param.x2Dtype;
    aclDataType biasDtype = param.biasDtype;
    aclDataType outputDtype = param.outputDtype;
    aclFormat x1Format = param.x1Format;
    aclFormat x2Format = param.x2Format;
    aclFormat biasFormat = param.biasFormat;
    aclFormat outputFormat = param.outputFormat;
    vector<int64_t> axesAcl = param.alltoAllAxesOptional;
    aclIntArray *alltoAllAxesOptional = aclCreateIntArray(axesAcl.data(), axesAcl.size());
    const char* group = param.group;
    const char* commMode = param.commMode;
    bool transposeX1 = param.transposeX1;
    bool transposeX2 = param.transposeX2;
    aclnnStatus retStatus = param.aclnnStatusUt;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet;

    if (param.x1Null || param.x2Null || param.outputNull) {
        aclTensor* x1Ptr = param.x1Null ? nullptr : CreateAclTensor(x1Shape, x1Dtype, x1Format);
        aclTensor* x2Ptr = param.x2Null ? nullptr : CreateAclTensor(x2Shape, x2Dtype, x2Format);
        aclTensor* outputPtr = param.outputNull ? nullptr : CreateAclTensor(outputShape, outputDtype, outputFormat);
        aclTensor* biasPtr = biasShape.empty() ? nullptr : CreateAclTensor(biasShape, biasDtype, biasFormat);
        auto ut = OP_API_UT(aclnnMatmulAlltoAllV2,
                        INPUT(x1Ptr, x2Ptr, biasPtr, alltoAllAxesOptional, group, commMode, transposeX1, transposeX2),
                        OUTPUT(outputPtr));
        aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
        if (x1Ptr != nullptr) aclDestroyTensor(x1Ptr);
        if (x2Ptr != nullptr) aclDestroyTensor(x2Ptr);
        if (outputPtr != nullptr) aclDestroyTensor(outputPtr);
        if (biasPtr != nullptr) aclDestroyTensor(biasPtr);
    } else {
        TensorDesc x1 = TensorDesc(x1Shape, x1Dtype, x1Format);
        TensorDesc x2 = TensorDesc(x2Shape, x2Dtype, x2Format);
        TensorDesc output = TensorDesc(outputShape, outputDtype, outputFormat);
        if (biasShape.empty()) {
            auto ut = OP_API_UT(aclnnMatmulAlltoAllV2,
                            INPUT(x1, x2, nullptr, alltoAllAxesOptional, group, commMode, transposeX1, transposeX2),
                            OUTPUT(output));
            aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
        } else {
            TensorDesc bias = TensorDesc(biasShape, biasDtype, biasFormat);
            auto ut = OP_API_UT(aclnnMatmulAlltoAllV2,
                            INPUT(x1, x2, bias, alltoAllAxesOptional, group, commMode, transposeX1, transposeX2),
                            OUTPUT(output));
            aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
        }
    }
    if (retStatus == ACLNN_SUCCESS) {
        EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, retStatus);
    }
    std::cout << "end case " <<  param.caseName << std::endl;
}

// DAV_3510 正常用例：覆盖 comm_mode = "default", "aicpu", "ccu"
static MatmulAlltoAllV2AclnnTestParam g_v2_casesParams_950[] = {
    // ======================== commMode = "default" ========================
    // BF16 + bias=FLOAT32 + notrans
    {"V2-950-default-bf16-biasf32-nd-notrans-01", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", "default", false, false, ACLNN_SUCCESS},
    // BF16 + bias=null + trans
    {"V2-950-default-bf16-biasnull-nd-trans-02", 2, {256, 128}, {256, 128}, {}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_DT_UNDEFINED, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", "default", false, true, ACLNN_SUCCESS},
    // FP16 + bias=FP16 + notrans
    {"V2-950-default-fp16-biasfp16-nd-notrans-03", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", "default", false, false, ACLNN_SUCCESS},
    // FP16 + bias=null + notrans
    {"V2-950-default-fp16-biasnull-nd-notrans-04", 2, {256, 128}, {128, 256}, {}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_DT_UNDEFINED, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", "default", false, false, ACLNN_SUCCESS},
    // 空tensor
    {"V2-950-default-x1_empty_tensor-05", 2, {0, 128}, {128, 256}, {256}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", "default", false, false, ACLNN_SUCCESS},

    // ======================== commMode = "ccu" ========================
    {"V2-950-ccu-bf16-biasf32-nd-notrans-06", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", "ccu", false, false, ACLNN_SUCCESS},
    {"V2-950-ccu-fp16-biasnull-nd-trans-07", 2, {256, 128}, {256, 128}, {}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_DT_UNDEFINED, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", "ccu", false, true, ACLNN_SUCCESS},

    // ======================== commMode = "aicpu" ========================
    {"V2-950-aicpu-bf16-biasf32-nd-notrans-08", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", "aicpu", false, false, ACLNN_SUCCESS},
    {"V2-950-aicpu-fp16-biasfp16-nd-trans-09", 2, {256, 128}, {256, 128}, {256}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", "aicpu", false, true, ACLNN_SUCCESS},

    // ======================== DAV_3510 异常用例 ========================
    // commMode为非法值
    {"error-V2-950-invalid_commode-01", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", "invalid_mode", false, false, ACLNN_ERR_PARAM_INVALID},
    // commMode为空字符串
    {"error-V2-950-empty_commode-02", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", "", false, false, ACLNN_ERR_PARAM_INVALID},
};

TEST_F(TestAclnnMatmulAlltoAllV2, CasesParamsTest950)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    if (std::size(g_v2_casesParams_950) != 0) {
        uint64_t numCases = sizeof(g_v2_casesParams_950) / sizeof(g_v2_casesParams_950[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestV2OneParamCase(g_v2_casesParams_950[idx]);
        }
    }
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
}

// 910B 正常用例 + 异常用例
static MatmulAlltoAllV2AclnnTestParam g_v2_casesParams_910B[] = {
    // ========================910B 正常用例 commMode = "default" ========================
    {"V2-910B-default-bf16-biasf32-nd-notrans-01", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", "default", false, false, ACLNN_SUCCESS},
    {"V2-910B-default-fp16-biasfp16-nd-notrans-02", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", "default", false, false, ACLNN_SUCCESS},
    {"V2-910B-default-bf16-biasnull-nd-notrans-03", 2, {256, 128}, {128, 256}, {}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_DT_UNDEFINED, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", "default", false, false, ACLNN_SUCCESS},

    // ========================910B 异常用例：910B不支持ccu/aicpu ========================
    // commMode = "ccu" 在910B上不支持
    {"error-V2-910B-ccu_not_support-01", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", "ccu", false, false, ACLNN_ERR_PARAM_INVALID},
    // commMode = "aicpu" 在910B上不支持
    {"error-V2-910B-aicpu_not_support-02", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", "aicpu", false, false, ACLNN_ERR_PARAM_INVALID},
    // commMode为非法值
    {"error-V2-910B-invalid_commode-03", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", "invalid", false, false, ACLNN_ERR_PARAM_INVALID},
};

TEST_F(TestAclnnMatmulAlltoAllV2, CasesParamsTest910B)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    if (std::size(g_v2_casesParams_910B) != 0) {
        uint64_t numCases = sizeof(g_v2_casesParams_910B) / sizeof(g_v2_casesParams_910B[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestV2OneParamCase(g_v2_casesParams_910B[idx]);
        }
    }
}
