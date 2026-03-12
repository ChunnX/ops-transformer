/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_allto_all_quant_matmul.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

constexpr int64_t MX_GROUP_SIZE = 4295032864;

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
    int64_t shapeSize = 1;
    for (auto i: shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

aclTensor* CreateAclTensor(const std::vector<int64_t> shape, aclDataType dataType, aclFormat format) {
    // 定义存放tensor数据的内存指针
    void* storage_data = nullptr;
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // 调用aclCreateTensor接口创建aclTensor
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
        strides.data(), 0, format, shape.data(), shape.size(), storage_data);
    assert(tensor != nullptr);
    return tensor;
}

class test_aclnn_allto_all_quant_matmul : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(NpuArch::DAV_3510);
        cout << "test_aclnn_allto_all_quant_matmul SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "test_aclnn_allto_all_quant_matmul TearDown" << endl;
    }
};

// ut用例结构体
struct AlltoAllQuantMatmulAclnnTestParam {
    string case_name; // 用例名
    int world_size; // 通信域卡数，ut测试默认为2
    // 数据形状
    int64_t x1_quantmode; // x1量化模式
    int64_t x2_quantmode; // x2量化模式
    vector<int64_t> x1_shape; // x1数据shape，正常为（BS，H）
    vector<int64_t> x2_shape; // x2数据shape，正常为（H * world_size，N）
    vector<int64_t> bias_shape; // bias数据shape，正常为（N）
    vector<int64_t> x1_scale_optional_shape; // x1ScaleOptional数据shape，kc动态量化为空，mx量化为（BS，ceil(H/64)，2）
    vector<int64_t> x2_scale_shape; // x2scales数据shape，正常为（N），mx量化为（N，ceil(H*rankSize/64)，2）
    vector<int64_t> output_shape; // output数据shape，正常为（BS / world_size，N）
    vector<int64_t> alltoalloutput_shape; // alltoalloutput数据shape，正常为（BS / ranksize，H * ranksize）
    // 数据类型
    aclDataType x1_dtype; // x1数据dtype，仅支持bfloat16和float16，float8_e5m2和float8_e4m3fn
    aclDataType x2_dtype; // x2数据dtype，仅支持float8_e5m2和float8_e4m3fn
    aclDataType bias_dtype; // bias数据dtype，仅支持float32
    aclDataType x1_scale_optional_dtype; // x1ScaleOptional数据dtype，仅支持bfloat16和float16，float8_e8m0
    aclDataType x2_scale_dtype; // x2scales数据dtype，仅支持float32和float8_e8m0
    aclDataType output_dtype; // output数据dtype，支持bfloat16、float16和float32
    aclDataType alltoalloutput_dtype; // alltoalloutput数据dtype，仅支持bfloat16和float16，float8_e5m2和float8_e4m3fn，要求和x1Dtype一致
    // 数据格式
    aclFormat x1_format; // x1数据format，仅支持ND
    aclFormat x2_format; // x2数据format，仅支持ND
    aclFormat bias_format; // bias数据format，仅支持ND
    aclFormat x1_scale_optional_format; // x1ScaleOptional数据format，仅支持ND
    aclFormat x2_scale_format; // x2Scale数据format，仅支持ND
    aclFormat output_format; // output数据format，仅支持ND
    aclFormat alltoalloutput_format; // alltoalloutputoutput数据format，仅支持ND
    // 其它属性
    int64_t x1_quantdtype; // x1量化数据类型，KC动态量化场景下仅支持配置35（表示ACL_FLOAT8_E5M2）或36（表示ACL_FLOAT8_E4M3FN），mx量化场景下不生效，配置0即可
    int64_t group_size; // groupSize，仅在perGroup（MX）量化时产生意义，表示分组量化时分组的大小，其余场景默认传0即可
    vector<int64_t> alltoAllAxesOptional; // alltoall数据交换的方向，只能为空或者[-2,-1]
    char* group; // 通信域标识，字符串，长度要求（0，128）
    bool transposeX1; // x1是否转置，现不支持为true
    bool transposeX2; // x2是否转置，为true时x2shape为（N，H * world_size），mx量化时必须配置为true
    aclnnStatus aclnn_status; //期望状态
};

// KC动态量化UT用例表
static AlltoAllQuantMatmulAclnnTestParam KCDynQuant_cases_params[] = {
    // 正常用例
    {"AAQMM_DYNKC-succ1", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_SUCCESS}, // 合法场景：bf16/fp8_e4m3fn + bias为fp32 + y输出bf16 + 量化后e5m2 + 无转置 + 有alltoallout
    {"AAQMM_DYNKC-succ2", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        36, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_SUCCESS}, // 合法场景：bf16/fp8_e4m3fn + bias为fp32 + y输出bf16 + 量化后e4m3 + 无转置 + 有alltoallout
    {"AAQMM_DYNKC-succ3", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_FLOAT16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_SUCCESS}, // 合法场景：fp16/fp8_e4m3fn + bias为fp32 + y输出bf16 + 量化后e5m2 + 无转置 + 有alltoallout
    {"AAQMM_DYNKC-succ4", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_FLOAT16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_SUCCESS}, // 合法场景：bf16/fp8_e4m3fn + bias为fp32 + y输出fp16 + 量化后e5m2 + 无转置 + 有alltoallout
    {"AAQMM_DYNKC-succ5", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_FLOAT16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        36, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_SUCCESS}, // 合法场景：fp16/fp8_e4m3fn + bias为fp32 + y输出bf16 + 量化后e4m3 + 无转置 + 有alltoallout
    {"AAQMM_DYNKC-succ6", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_SUCCESS}, // 合法场景：bf16/fp8_e5m2 + bias为fp32 + y输出bf16 + 量化后e5m2 + 无转置 + 有alltoallout
    {"AAQMM_DYNKC-succ7", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        36, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_SUCCESS}, // 合法场景：bf16/fp8_e5m2 + bias为fp32 + y输出bf16 + 量化后e4m3 + 无转置 + 有alltoallout
    {"AAQMM_DYNKC-succ8", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_SUCCESS}, // 合法场景：bf16/fp8_e5m2 + bias为fp32 + y输出fp32 + 量化后e5m2 + 无转置 + 有alltoallout
    {"AAQMM_DYNKC-succ9", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        36, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_SUCCESS}, // 合法场景：bf16/fp8_e4m3fn + bias为fp32 + y输出fp32 + 量化后e4m3 + 无转置 + 有alltoallout
    {"AAQMM_DYNKC-succ10", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_FLOAT16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        36, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_SUCCESS}, // 合法场景：bf16/fp8_e4m3fn + bias为fp32 + y输出fp16 + 量化后e4m3 + 无转置 + 有alltoallout
    {"AAQMM_DYNKC-succ11", 2, 7, 2, {256, 64}, {128, 256}, {}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_SUCCESS}, // 合法场景：bf16/fp8_e4m3fn + bias为空 + y输出bf16 + 量化后e5m2 + 无转置 + 有alltoallout
    {"AAQMM_DYNKC-succ12", 2, 7, 2, {256, 64}, {128, 256}, {}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_FLOAT16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_SUCCESS}, // 合法场景：bf16/fp8_e4m3fn + bias为空 + y输出fp16 + 量化后e5m2 + 无转置 + 有alltoallout
    {"AAQMM_DYNKC-succ13", 2, 7, 2, {256, 64}, {128, 256}, {}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_SUCCESS}, // 合法场景：bf16/fp8_e5m2 + bias为空 + y输出bf16 + 量化后e5m2 + 无转置 + 有alltoallout
    {"AAQMM_DYNKC-succ14", 2, 7, 2, {256, 64}, {256, 128}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：bf16/fp8_e4m3fn + bias为fp32 + y输出bf16 + 量化后e5m2 + 有转置 + 有alltoallout
    {"AAQMM_DYNKC-succ15", 2, 7, 2, {256, 64}, {256, 128}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：bf16/fp8_e5m2 + bias为fp32 + y输出bf16 + 量化后e5m2 + 有转置 + 有alltoallout
    {"AAQMM_DYNKC-succ16", 2, 7, 2, {256, 64}, {256, 128}, {}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：bf16/fp8_e4m3fn + bias为空 + y输出bf16 + 量化后e5m2 + 有转置 + 有alltoallout
    {"AAQMM_DYNKC-succ17", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_SUCCESS}, // 合法场景：bf16/fp8_e4m3fn + bias为fp32 + y输出bf16 + 量化后e5m2 + 无转置 + 无alltoallout
    {"AAQMM_DYNKC-succ18", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        36, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_SUCCESS}, // 合法场景：bf16/fp8_e4m3fn + bias为fp32 + y输出bf16 + 量化后e4m3 + 无转置 + 无alltoallout
    {"AAQMM_DYNKC-succ19", 2, 7, 2, {256, 64}, {256, 128}, {256}, {}, {256}, {128, 256}, {},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：bf16/fp8_e4m3fn + bias为fp32 + y输出bf16 + 量化后e5m2 + 有转置 + 无alltoallout
    {"AAQMM_DYNKC-succ20", 2, 7, 2, {256, 64}, {256, 128}, {}, {}, {256}, {128, 256}, {},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：bf16/fp8_e4m3fn + bias为空 + y输出bf16 + 量化后e5m2 + 有转置 + 无alltoallout
    // 异常场景
    {"AAQMM_DYNKC-error1", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x1数据类型非法
    {"AAQMM_DYNKC-error2", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x2数据类型非法
    {"AAQMM_DYNKC-error3", 2, 7, 2, {256, 64, 32}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x1形状不满足shape约束
    {"AAQMM_DYNKC-error4", 2, 7, 2, {256, 64}, {128, 2, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x2形状不满足shape约束
    {"AAQMM_DYNKC-error5", 2, 7, 2, {0, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x1存在维度为0
    {"AAQMM_DYNKC-error6", 2, 7, 2, {256, 64}, {128, 0}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x2存在维度为0
    {"AAQMM_DYNKC-error7", 2, 7, 2, {256, 0}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x1为空tensor
    {"AAQMM_DYNKC-error8", 2, 7, 2, {256, 64}, {0, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x2为空tensor
    {"AAQMM_DYNKC-error9", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        30, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x1QuantDtype非法
    {"AAQMM_DYNKC-error10", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x2scale数据类型不是float32
    {"AAQMM_DYNKC-error11", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x2scale维度不为1D
    {"AAQMM_DYNKC-error12", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {123}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x2Scale形状不等于(N)
    {"AAQMM_DYNKC-error13", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_FLOAT8_E4M3FN, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：output数据类型不在float16, bf16, float32内
    {"AAQMM_DYNKC-error14", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256, 2}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：output维度不为2D
    {"AAQMM_DYNKC-error15", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_BF16, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：bias数据类型不等于float32
    {"AAQMM_DYNKC-error16", 2, 7, 2, {256, 64}, {128, 256}, {122}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：bias形状不等于N
    {"AAQMM_DYNKC-error17", 2, 9, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x1QuantMode取值非法
    {"AAQMM_DYNKC-error18", 2, 7, 19, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x2QuantMode取值非法
    {"AAQMM_DYNKC-error19", 2, 17, 22, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x1QuantMode和x2QuantMode组合非法
    {"AAQMM_DYNKC-error20", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -112}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：alltoAllAxesOptional不为[-2, -1]
    {"AAQMM_DYNKC-error21", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", true, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：transposex1为true
    {"AAQMM_DYNKC-error22", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "01234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：group长度超过128
    {"AAQMM_DYNKC-error23", 2, 7, 2, {256, 64}, {128, 256}, {256}, {}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "", false, false, ACLNN_ERR_PARAM_INVALID} // 异常场景：group长度等于0
};

// MX量化UT用例表
static AlltoAllQuantMatmulAclnnTestParam MXQuant_cases_params[] = {
    // 正常用例
    {"AAQMM_MX-succ1", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e5m2/fp8_e5m2 + bias为空 + y输出fp32 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ2", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e5m2/fp8_e5m2 + bias为空 + y输出fp16 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ3", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e5m2/fp8_e5m2 + bias为空 + y输出bf16 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ4", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e5m2/fp8_e4m3fn + bias为fp32 + y输出fp32 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ5", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e5m2/fp8_e4m3fn + bias为fp32 + y输出fp16 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ6", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e5m2/fp8_e4m3fn + bias为fp32 + y输出bf16 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ7", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e5m2/fp8_e4m3fn + bias为空 + y输出fp32 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ8", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e5m2/fp8_e4m3fn + bias为空 + y输出fp16 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ9", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e5m2/fp8_e4m3fn + bias为空 + y输出bf16 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ10", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e4m3fn/fp8_e5m2 + bias为fp32 + y输出fp32 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ11", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e4m3fn/fp8_e5m2 + bias为fp32 + y输出fp16 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ12", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e4m3fn/fp8_e5m2 + bias为fp32 + y输出bf16 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ13", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e4m3fn/fp8_e5m2 + bias为空 + y输出fp32 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ14", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e4m3fn/fp8_e5m2 + bias为空 + y输出fp16 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ15", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e4m3fn/fp8_e5m2 + bias为空 + y输出bf16 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ16", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e4m3fn/fp8_e4m3fn + bias为fp32 + y输出fp32 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ17", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e4m3fn/fp8_e4m3fn + bias为fp32 + y输出fp16 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ18", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e4m3fn/fp8_e4m3fn + bias为fp32 + y输出bf16 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ19", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS},  // 合法场景：fp8_e4m3fn/fp8_e4m3fn + bias为空 + y输出fp32 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ20", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS},  // 合法场景：fp8_e4m3fn/fp8_e4m3fn + bias为空 + y输出fp16 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ21", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS},  // 合法场景：fp8_e4m3fn/fp8_e4m3fn + bias为空 + y输出bf16 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ22", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS},  // 合法场景：fp8_e5m2/fp8_e5m2 + bias为fp32 + y输出fp32 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ23", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS},  // 合法场景：fp8_e5m2/fp8_e5m2 + bias为fp32 + y输出fp16 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ24", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS},  // 合法场景：fp8_e5m2/fp8_e5m2 + bias为fp32 + y输出bf16 + x2转置 + 有alltoallout
    {"AAQMM_MX-succ25", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS},  // 合法场景：fp8_e5m2/fp8_e5m2 + bias为空 + y输出fp32 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ26", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS},  // 合法场景：fp8_e5m2/fp8_e5m2 + bias为空 + y输出fp16 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ27", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS},  // 合法场景：fp8_e5m2/fp8_e5m2 + bias为空 + y输出bf16 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ28", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS},  // 合法场景：fp8_e5m2/fp8_e4m3fn + bias为fp32 + y输出fp32 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ29", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS},  // 合法场景：fp8_e5m2/fp8_e4m3fn + bias为fp32 + y输出fp16 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ30", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS},  // 合法场景：fp8_e5m2/fp8_e4m3fn + bias为fp32 + y输出bf16 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ31", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS},  // 合法场景：fp8_e5m2/fp8_e4m3fn + bias为空 + y输出fp32 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ32", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS},  // 合法场景：fp8_e5m2/fp8_e4m3fn + bias为空 + y输出fp16 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ33", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS},  // 合法场景：fp8_e5m2/fp8_e4m3fn + bias为空 + y输出bf16 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ34", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS},  // 合法场景：fp8_e4m3fn/fp8_e5m2 + bias为fp32 + y输出fp32 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ35", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS},  // 合法场景：fp8_e4m3fn/fp8_e5m2 + bias为fp32 + y输出fp16 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ36", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS},  // 合法场景：fp8_e4m3fn/fp8_e5m2 + bias为fp32 + y输出bf16 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ37", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e4m3fn/fp8_e5m2 + bias为空 + y输出fp32 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ38", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e4m3fn/fp8_e5m2 + bias为空 + y输出fp16 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ39", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e4m3fn/fp8_e5m2 + bias为空 + y输出bf16 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ40", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e4m3fn/fp8_e4m3fn + bias为fp32 + y输出fp32 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ41", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e4m3fn/fp8_e4m3fn + bias为fp32 + y输出fp16 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ42", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e4m3fn/fp8_e4m3fn + bias为fp32 + y输出bf16 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ43", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e4m3fn/fp8_e4m3fn + bias为空 + y输出fp32 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ44", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e4m3fn/fp8_e4m3fn + bias为空 + y输出fp16 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ45", 2, 6, 6, {256, 64}, {256, 128}, {}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e4m3fn/fp8_e4m3fn + bias为空 + y输出bf16 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ46", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e5m2/fp8_e5m2 + bias为fp32 + y输出fp32 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ47", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e5m2/fp8_e5m2 + bias为fp32 + y输出fp16 + x2转置 + 无alltoallout
    {"AAQMM_MX-succ48", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 合法场景：fp8_e5m2/fp8_e5m2 + bias为fp32 + y输出bf16 + x2转置 + 无alltoallout
    // 异常场景
    {"AAQMM_MX-error1", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x1数据类型非法
    {"AAQMM_MX-error2", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x2数据类型非法
    {"AAQMM_MX-error3", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：bias数据类型非法
    {"AAQMM_MX-error4", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E5M2, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x1Scale数据类型非法
    {"AAQMM_MX-error5", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x2Scale数据类型非法
    {"AAQMM_MX-error6", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：output数据类型非法
    {"AAQMM_MX-error7", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：alltoalloutput数据类型非法
    {"AAQMM_MX-error8", 2, 6, 6, {0, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x1为空tensor，第一维度为0
    {"AAQMM_MX-error9", 2, 6, 6, {256, 0}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x1为空tensor，第二维度为0
    {"AAQMM_MX-error10", 2, 6, 6, {256, 64}, {0, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x2为空tensor，N维度为0
    {"AAQMM_MX-error11", 2, 6, 6, {256, 64}, {256, 0}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x2为空tensor，K维度为0
    {"AAQMM_MX-error12", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x1数据格式非法
    {"AAQMM_MX-error13", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x2数据格式非法
    {"AAQMM_MX-error14", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：bias数据格式非法
    {"AAQMM_MX-error15", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x1Scale数据格式非法
    {"AAQMM_MX-error16", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x2Scale数据格式非法
    {"AAQMM_MX-error17", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：output数据格式非法
    {"AAQMM_MX-error18", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：alltoalloutput数据格式非法
    {"AAQMM_MX-error19", 2, 6, 6, {256, 64, 3}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x1维度不为2D
    {"AAQMM_MX-error20", 2, 6, 6, {256, 64}, {256, 2, 3, 4, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x2维度不为2D
    {"AAQMM_MX-error21", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：alltoallout维度不为2D
    {"AAQMM_MX-error22", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 2, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：output维度不为2D
    {"AAQMM_MX-error23", 2, 6, 6, {256, 64}, {256, 128}, {256, 3}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：bias维度不为1D
    {"AAQMM_MX-error24", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x1Scale维度不为3D
    {"AAQMM_MX-error25", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x2Scale维度不为3D
    {"AAQMM_MX-error26", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 5}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x1Scale最后一个维度的值不为2
    {"AAQMM_MX-error27", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 6}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x2Scale最后一个维度的值不为2
    {"AAQMM_MX-error28", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -88}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：alltoAllAxesOptional取值非法
    {"AAQMM_MX-error29", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：group为空
    {"AAQMM_MX-error30", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "01234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：group长度超过127
    {"AAQMM_MX-error31", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", true, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：transposeX1为true
    {"AAQMM_MX-error32", 2, 6, 6, {256, 64}, {128, 256}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID}, // 异常场景：transposeX2为false
    {"AAQMM_MX-error34", 2, 6, 6, {256}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x1的维度出错
    {"AAQMM_MX-error35", 2, 6, 6, {256, 64}, {256, 128}, {2560}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：bias的shape和x2不匹配
    {"AAQMM_MX-error36", 2, 6, 6, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {2560, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x2Scale的shape和x2不匹配
    {"AAQMM_MX-error37", 2, 6, 6, {256, 64}, {256, 128, 3}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // 异常场景：x2的维度出错
    {"AAQMM_MX-error38", 2, 190, 3, {256, 64}, {256, 128}, {256}, {256, 1, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        0, MX_GROUP_SIZE, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID} // 异常场景：quantmode组合非法，不是（6,6）
};

static void TestOneParamCase(const AlltoAllQuantMatmulAclnnTestParam& param)
{
    std::cout << "run case " << param.case_name << std::endl;
    // 从结构体list中获取实际用例属性
    int64_t x1quantmode = param.x1_quantmode;
    int64_t x2quantmode = param.x2_quantmode;
    vector<int64_t> x1Shape = param.x1_shape;
    vector<int64_t> x2Shape = param.x2_shape;
    vector<int64_t> biasShape = param.bias_shape;
    vector<int64_t> x1scalesShape = param.x1_scale_optional_shape;
    vector<int64_t> x2scalesShape = param.x2_scale_shape;
    vector<int64_t> outputShape = param.output_shape;
    vector<int64_t> alltoalloutShape = param.alltoalloutput_shape;
    aclDataType x1Dtype = param.x1_dtype;
    aclDataType x2Dtype = param.x2_dtype;
    aclDataType biasDtype = param.bias_dtype;
    aclDataType x1scalesDtype = param.x1_scale_optional_dtype;
    aclDataType x2scalesDtype = param.x2_scale_dtype;
    aclDataType outputDtype = param.output_dtype;
    aclDataType alltoalloutDtype = param.alltoalloutput_dtype;
    aclFormat x1Format = param.x1_format;
    aclFormat x2Format = param.x2_format;
    aclFormat biasFormat = param.bias_format;
    aclFormat x1_scale_format = param.x1_scale_optional_format;
    aclFormat x2_scale_format = param.x2_scale_format;
    aclFormat outputFormat = param.output_format;
    aclFormat alltoalloutFormat = param.alltoalloutput_format;
    int64_t x1quantdtype = param.x1_quantdtype;
    int64_t groupSize = param.group_size;
    vector<int64_t> axes_acl = param.alltoAllAxesOptional;
    aclIntArray *alltoAllAxesOptional = aclCreateIntArray(axes_acl.data(), axes_acl.size());
    const char* group = param.group;
    bool transposeX1 = param.transposeX1;
    bool transposeX2 = param.transposeX2;
    aclnnStatus retStatus = param.aclnn_status;
    TensorDesc x1 = TensorDesc(x1Shape, x1Dtype, x1Format);
    TensorDesc x2 = TensorDesc(x2Shape, x2Dtype, x2Format);
    TensorDesc x2scales = TensorDesc(x2scalesShape, x2scalesDtype, x2_scale_format);
    TensorDesc output = TensorDesc(outputShape, outputDtype, outputFormat);
    // 可能为nullptr的tensor通过手动创建
    aclTensor* bias = nullptr;
    aclTensor* x1scales = nullptr;
    aclTensor* alltoallout = nullptr;
    if (!biasShape.empty()) {
        bias = CreateAclTensor(biasShape, biasDtype, biasFormat);
    }
    if (!x1scalesShape.empty()) {
        x1scales = CreateAclTensor(x1scalesShape, x1scalesDtype, x1_scale_format);
    }
    if (!alltoalloutShape.empty()) {
        alltoallout = CreateAclTensor(alltoalloutShape, alltoalloutDtype, alltoalloutFormat);
    }
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet;
    auto ut = OP_API_UT(aclnnAlltoAllQuantMatmul,
        INPUT(x1, x2, bias, x1scales, x2scales, nullptr, nullptr, nullptr, group, alltoAllAxesOptional,
            x1quantmode, x2quantmode, 0, -1, x1quantdtype, groupSize, transposeX1, transposeX2),
        OUTPUT(output, alltoallout));
    aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, retStatus);
    std::cout << "end case " <<  param.case_name << std::endl;
}

// 测试KC动态量化场景下的UT用例
TEST_F(test_aclnn_allto_all_quant_matmul, KCDynQuant_cases_params)
{
    if (std::size(KCDynQuant_cases_params) != 0) {
        uint64_t numCases = sizeof(KCDynQuant_cases_params) / sizeof(KCDynQuant_cases_params[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            // 这是注释TestOneParamCase(KCDynQuant_cases_params[idx]);
        }
    }
}
// 测试MX量化场景下的UT用例
TEST_F(test_aclnn_allto_all_quant_matmul, MXQuant_cases_params)
{
    if (std::size(MXQuant_cases_params) != 0) {
        uint64_t numCases = sizeof(MXQuant_cases_params) / sizeof(MXQuant_cases_params[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            // 这是注释TestOneParamCase(MXQuant_cases_params[idx]);
        }
    }
}