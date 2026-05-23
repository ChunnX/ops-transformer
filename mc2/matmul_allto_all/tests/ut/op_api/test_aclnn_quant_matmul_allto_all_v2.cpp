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
#include "../../../op_api/aclnn_quant_matmul_allto_all_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

constexpr int64_t MX_GROUP_SIZE = 4295032864;

static aclTensor* CreateAclTensor(const std::vector<int64_t> shape, aclDataType dataType, aclFormat format) {
    void* storage_data = nullptr;
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
        strides.data(), 0, format, shape.data(), shape.size(), storage_data);
    assert(tensor != nullptr);
    return tensor;
}

static aclTensor* CreateAclTensorOrNull(const std::vector<int64_t> shape, aclDataType dataType, aclFormat format) {
    if (shape.empty()) {
        return nullptr;
    }
    return CreateAclTensor(shape, dataType, format);
}

class TestAclnnQuantMatmulAlltoAllV2 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(NpuArch::DAV_3510);
        cout << "TestAclnnQuantMatmulAlltoAllV2 SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "TestAclnnQuantMatmulAlltoAllV2 TearDown" << endl;
    }
};

struct QuantMatmulAlltoAllV2AclnnTestParam {
    string caseName;
    int worldSize;
    int64_t x1Quantmode;
    int64_t x2Quantmode;
    vector<int64_t> x1Shape;
    vector<int64_t> x2Shape;
    vector<int64_t> biasShape;
    vector<int64_t> x1ScaleShape;
    vector<int64_t> x2ScaleShape;
    vector<int64_t> outputShape;
    aclDataType x1Dtype;
    aclDataType x2Dtype;
    aclDataType biasDtype;
    aclDataType x1ScaleDtype;
    aclDataType x2ScaleDtype;
    aclDataType outputDtype;
    aclFormat x1Format;
    aclFormat x2Format;
    aclFormat biasFormat;
    aclFormat x1ScaleFormat;
    aclFormat x2ScaleFormat;
    aclFormat outputFormat;
    int64_t group_size;
    vector<int64_t> alltoAllAxesOptional;
    char* group;
    const char* commMode; // 新增：通信引擎参数
    bool transposeX1;
    bool transposeX2;
    aclnnStatus aclnnStatusUt;
};

static void TestQuantV2OneParamCase(const QuantMatmulAlltoAllV2AclnnTestParam& param)
{
    std::cout << "run case " << param.caseName << std::endl;
    int64_t x1quantmode = param.x1Quantmode;
    int64_t x2quantmode = param.x2Quantmode;
    vector<int64_t> x1Shape = param.x1Shape;
    vector<int64_t> x2Shape = param.x2Shape;
    vector<int64_t> biasShape = param.biasShape;
    vector<int64_t> x1scalesShape = param.x1ScaleShape;
    vector<int64_t> x2scalesShape = param.x2ScaleShape;
    vector<int64_t> outputShape = param.outputShape;
    aclDataType x1Dtype = param.x1Dtype;
    aclDataType x2Dtype = param.x2Dtype;
    aclDataType biasDtype = param.biasDtype;
    aclDataType x1scalesDtype = param.x1ScaleDtype;
    aclDataType x2scalesDtype = param.x2ScaleDtype;
    aclDataType outputDtype = param.outputDtype;
    aclFormat x1Format = param.x1Format;
    aclFormat x2Format = param.x2Format;
    aclFormat biasFormat = param.biasFormat;
    aclFormat x1ScaleFormat = param.x1ScaleFormat;
    aclFormat x2ScaleFormat = param.x2ScaleFormat;
    aclFormat outputFormat = param.outputFormat;
    int64_t groupSize = param.group_size;
    vector<int64_t> axesAcl = param.alltoAllAxesOptional;
    aclIntArray *alltoAllAxesOptional = aclCreateIntArray(axesAcl.data(), axesAcl.size());
    const char* group = param.group;
    const char* commMode = param.commMode;
    bool transposeX1 = param.transposeX1;
    bool transposeX2 = param.transposeX2;
    aclnnStatus retStatus = param.aclnnStatusUt;
    aclTensor* x1 = CreateAclTensorOrNull(x1Shape, x1Dtype, x1Format);
    aclTensor* x2 = CreateAclTensorOrNull(x2Shape, x2Dtype, x2Format);
    aclTensor* bias = CreateAclTensorOrNull(biasShape, biasDtype, biasFormat);
    aclTensor* x1scales = CreateAclTensorOrNull(x1scalesShape, x1scalesDtype, x1ScaleFormat);
    aclTensor* x2scales = CreateAclTensorOrNull(x2scalesShape, x2scalesDtype, x2ScaleFormat);
    aclTensor* output = CreateAclTensorOrNull(outputShape, outputDtype, outputFormat);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet;
    auto ut = OP_API_UT(aclnnQuantMatmulAlltoAllV2,
                        INPUT(x1, x2, bias, x1scales, x2scales, nullptr, nullptr, nullptr, alltoAllAxesOptional, group,
                               commMode, x1quantmode, x2quantmode, 0, -1, groupSize, transposeX1, transposeX2),
                        OUTPUT(output));
    aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    if (retStatus == ACLNN_SUCCESS) {
        EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, retStatus);
    }
    std::cout << "end case " <<  param.caseName << std::endl;
}

// KC量化 + commMode 覆盖用例
static QuantMatmulAlltoAllV2AclnnTestParam KCQuant_v2_cases_params[] = {
    // ======================== commMode = "default" 正常用例 ========================
    {"V2-KC-default-e4m3-e4m3-f32-bf16-01", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, {-1, -2}, "ut_test_quant_matmul_allto_all", "default", false, false, ACLNN_SUCCESS},
    {"V2-KC-default-e4m3-e5m2-f32-fp16-02", 2, 3, 2, {256, 128}, {128, 256}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, {-1, -2}, "ut_test_quant_matmul_allto_all", "default", false, false, ACLNN_SUCCESS},
    {"V2-KC-default-e5m2-e4m3-f32-fp32-03", 2, 3, 2, {256, 128}, {256, 128}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, {-1, -2}, "ut_test_quant_matmul_allto_all", "default", false, true, ACLNN_SUCCESS},

    // ======================== commMode = "ccu" 正常用例 ========================
    {"V2-KC-ccu-e4m3-e4m3-f32-bf16-04", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, {-1, -2}, "ut_test_quant_matmul_allto_all", "ccu", false, false, ACLNN_SUCCESS},
    {"V2-KC-ccu-e5m2-e5m2-f32-fp16-05", 2, 3, 2, {256, 128}, {256, 128}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, {-1, -2}, "ut_test_quant_matmul_allto_all", "ccu", false, true, ACLNN_SUCCESS},

    // ======================== commMode = "aicpu" 正常用例 ========================
    {"V2-KC-aicpu-e4m3-e4m3-f32-fp32-06", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, {-1, -2}, "ut_test_quant_matmul_allto_all", "aicpu", false, false, ACLNN_SUCCESS},
    {"V2-KC-aicpu-e5m2-e4m3-f32-bf16-07", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, {-1, -2}, "ut_test_quant_matmul_allto_all", "aicpu", false, false, ACLNN_SUCCESS},

    // ======================== 异常用例：commMode非法 ========================
    {"error-V2-KC-invalid_commode-01", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, {-1, -2}, "ut_test_quant_matmul_allto_all", "invalid_mode", false, false, ACLNN_ERR_PARAM_INVALID},
    {"error-V2-KC-empty_commode-02", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, {-1, -2}, "ut_test_quant_matmul_allto_all", "", false, false, ACLNN_ERR_PARAM_INVALID},
};

// MX量化 + commMode 覆盖用例
static QuantMatmulAlltoAllV2AclnnTestParam MXQuant_v2_cases_params[] = {
    // ======================== commMode = "default" 正常用例 ========================
    {"V2-MX-default-e5m2-e5m2-bf16-01", 2, 6, 6, {256, 128}, {256, 128}, {}, {256, 2, 2}, {256, 2, 2}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        MX_GROUP_SIZE, {-1, -2}, "ut_test_quant_matmul_allto_all", "default", false, true, ACLNN_SUCCESS},
    {"V2-MX-default-e4m3-e4m3-fp16-02", 2, 6, 6, {256, 128}, {256, 128}, {256}, {256, 2, 2}, {256, 2, 2}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        MX_GROUP_SIZE, {-1, -2}, "ut_test_quant_matmul_allto_all", "default", false, true, ACLNN_SUCCESS},

    // ======================== commMode = "ccu" 正常用例 ========================
    {"V2-MX-ccu-e5m2-e4m3-fp32-03", 2, 6, 6, {256, 128}, {256, 128}, {256}, {256, 2, 2}, {256, 2, 2}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        MX_GROUP_SIZE, {-1, -2}, "ut_test_quant_matmul_allto_all", "ccu", false, true, ACLNN_SUCCESS},
    {"V2-MX-ccu-e4m3-e5m2-bf16-04", 2, 6, 6, {256, 128}, {256, 128}, {}, {256, 2, 2}, {256, 2, 2}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        MX_GROUP_SIZE, {-1, -2}, "ut_test_quant_matmul_allto_all", "ccu", false, true, ACLNN_SUCCESS},

    // ======================== commMode = "aicpu" 正常用例 ========================
    {"V2-MX-aicpu-e4m3-e4m3-fp32-05", 2, 6, 6, {256, 128}, {256, 128}, {256}, {256, 2, 2}, {256, 2, 2}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        MX_GROUP_SIZE, {-1, -2}, "ut_test_quant_matmul_allto_all", "aicpu", false, true, ACLNN_SUCCESS},

    // ======================== 异常用例：commMode非法 ========================
    {"error-V2-MX-invalid_commode-01", 2, 6, 6, {256, 128}, {256, 128}, {256}, {256, 2, 2}, {256, 2, 2}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        MX_GROUP_SIZE, {-1, -2}, "ut_test_quant_matmul_allto_all", "invalid_mode", false, true, ACLNN_ERR_PARAM_INVALID},
};

TEST_F(TestAclnnQuantMatmulAlltoAllV2, KCQuant_v2_cases_params)
{
    if (std::size(KCQuant_v2_cases_params) != 0) {
        uint64_t numCases = sizeof(KCQuant_v2_cases_params) / sizeof(KCQuant_v2_cases_params[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestQuantV2OneParamCase(KCQuant_v2_cases_params[idx]);
        }
    }
}

TEST_F(TestAclnnQuantMatmulAlltoAllV2, MXQuant_v2_cases_params)
{
    if (std::size(MXQuant_v2_cases_params) != 0) {
        uint64_t numCases = sizeof(MXQuant_v2_cases_params) / sizeof(MXQuant_v2_cases_params[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestQuantV2OneParamCase(MXQuant_v2_cases_params[idx]);
        }
    }
}
