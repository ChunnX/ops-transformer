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
 * \file test_aclnn_moe_distribute_combine_teardown.cpp
 * \brief aclnn ut
 */

#include <float.h>
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_moe_distribute_combine_teardown.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

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

class TestAclnnMoeDistributeCombineTeardown : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(NpuArch::DAV_3510);
        cout << "TestAclnnMoeDistributeCombineTeardown SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "TestAclnnMoeDistributeCombineTeardown TearDown" << endl;
    }
};

struct MoeDistributeCombineTeardownAclnnTestParam {
    string case_name;
    vector<int64_t> expandXShape;
    vector<int64_t> quantExpandXShape;
    vector<int64_t> expertIdsShape;
    vector<int64_t> expandIdxShape;
    vector<int64_t> expertScalesShape;
    vector<int64_t> commCmdInfoShape;
    vector<int64_t> xActiveMaskOptionalShape;
    vector<int64_t> sharedExpertXOptionalShape;
    vector<int64_t> xOutShape;
    char* groupEp;
    int64_t epWorldSize;
    int64_t epRankId;
    int64_t moeExpertNum;
    int64_t expertShardType;
    int64_t sharedExpertNum;
    int64_t sharedExpertRankNum;
    int64_t globalBs;
    int64_t commQuantMode;
    int64_t commType;
    char* commAlg;
    aclDataType expandXDtype;
    aclDataType quantExpandXDtype;
    aclDataType expertIdsDtype;
    aclDataType expandIdxDtype;
    aclDataType expertScalesDtype;
    aclDataType commCmdInfoDtype;
    aclDataType xActiveMaskOptionalDtype;
    aclDataType sharedExpertXOptionalDtype;
    aclDataType xOutDtype;
    aclFormat expandXFormat;
    aclFormat quantExpandXFormat;
    aclFormat expertIdsFormat;
    aclFormat expandIdxFormat;
    aclFormat expertScalesFormat;
    aclFormat commCmdInfoFormat;
    aclFormat xActiveMaskOptionalFormat;
    aclFormat sharedExpertXOptionalFormat;
    aclFormat xOutFormat;
    aclnnStatus aclnnStatusUt;
};

static MoeDistributeCombineTeardownAclnnTestParam g_casesParams[] = {
    {"test_aclnn_moe_distribute_combine_teardown_BS48_H4096_FLOAT16_INT8_FLOAT16_true", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_SUCCESS},
    {"test_aclnn_moe_distribute_combine_teardown_BS512_H4096_FLOAT16_INT8_FLOAT16_true", {512, 4096}, {512, 4096}, {512, 1}, {512, 1}, {512, 1}, {1024}, {512}, {512, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 8, 0, 8, 0, 0, 0, 64, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_SUCCESS},
    {"test_aclnn_moe_distribute_combine_teardown_BS48_H7168_BF16_INT8_BF16_true", {48, 7168}, {48, 7168}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 7168}, {8, 7168},
        "moe_distribute_combine_teardown_test_group", 8, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_BF16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_SUCCESS},
    {"test_aclnn_moe_distribute_combine_teardown_BS64_H7168_BF16_INT8_BF16_true", {64, 7168}, {64, 7168}, {64, 1}, {64, 1}, {64, 1}, {1024}, {64}, {64, 7168}, {8, 7168},
        "moe_distribute_combine_teardown_test_group", 2, 0, 8, 0, 0, 0, 16, 0, 0, nullptr,
        ACL_BF16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_SUCCESS},
    {"test_aclnn_moe_distribute_combine_teardown_BS48_H4096_INT32_INT8_FLOAT16_false_invalid_expandXDtype", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_INT32, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_moe_distribute_combine_teardown_BS48_H4096_FLOAT16_INT8_FLOAT16_false_invalid_xActiveMaskOptionalDtype", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_moe_distribute_combine_teardown_empty_expandXShape", {}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_teardown_BS48_H4096_FLOAT16_FLOAT16_FLOAT16_false_invalid_quantExpandXDtype", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_moe_distribute_combine_teardown_BS48_H4096_FLOAT16_INT8_FLOAT16_false_invalid_quantExpandXFormat", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_moe_distribute_combine_teardown_empty_quantExpandXShape", {48, 4096}, {}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_teardown_BS48_H4096_FLOAT16_INT8_FLOAT16_false_invalid_expertIdsDtype", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_moe_distribute_combine_teardown_BS48_H4096_FLOAT16_INT8_FLOAT16_false_invalid_expertIdsFormat", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_moe_distribute_combine_teardown_empty_expertIdsShape", {48, 4096}, {48, 4096}, {}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_teardown_BS48_H4096_FLOAT16_INT8_FLOAT16_false_invalid_expandIdxDtype", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_FLOAT16, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_moe_distribute_combine_teardown_BS48_H4096_FLOAT16_INT8_FLOAT16_false_invalid_expandIdxFormat", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_moe_distribute_combine_teardown_empty_expandIdxShape", {48, 4096}, {48, 4096}, {48, 1}, {}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_teardown_BS48_H4096_FLOAT16_INT8_FLOAT16_false_invalid_expertScalesDtype", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT16, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_moe_distribute_combine_teardown_BS48_H4096_FLOAT16_INT8_FLOAT16_false_invalid_expertScalesFormat", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_moe_distribute_combine_teardown_empty_expertScalesShape", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_teardown_BS48_H4096_FLOAT16_INT8_FLOAT16_false_invalid_commCmdInfoDtype", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_FLOAT16, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_moe_distribute_combine_teardown_BS48_H4096_FLOAT16_INT8_FLOAT16_false_invalid_commCmdInfoFormat", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_moe_distribute_combine_teardown_empty_commCmdInfoShape", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_teardown_BS48_H4096_FLOAT16_INT8_FLOAT16_false_invalid_xActiveMaskOptionalDtype", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_moe_distribute_combine_teardown_BS48_H4096_FLOAT16_INT8_FLOAT16_false_invalid_xActiveMaskOptionalFormat", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_moe_distribute_combine_teardown_BS48_H4096_FLOAT16_INT8_INT32_false_invalid_xOutDtype", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_INT32,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_moe_distribute_combine_teardown_BS48_H4096_FLOAT16_INT8_FLOAT16_false_invalid_xOutFormat", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z,
        ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_moe_distribute_combine_teardown_empty_xOutShape", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_teardown_null_groupEp", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        nullptr, 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        ACLNN_ERR_PARAM_NULLPTR},
};

static void TestOneParamCase(const MoeDistributeCombineTeardownAclnnTestParam& param)
{
    std::cout << "run case " << param.case_name << std::endl;
    if (param.groupEp == nullptr) {
        std::cerr << "[ERROR]: groupEp is null" << std::endl;
        return;
    }
    vector<int64_t> expandXShape = param.expandXShape;
    vector<int64_t> quantExpandXShape = param.quantExpandXShape;
    vector<int64_t> expertIdsShape = param.expertIdsShape;
    vector<int64_t> expandIdxShape = param.expandIdxShape;
    vector<int64_t> expertScalesShape = param.expertScalesShape;
    vector<int64_t> commCmdInfoShape = param.commCmdInfoShape;
    vector<int64_t> xActiveMaskOptionalShape = param.xActiveMaskOptionalShape;
    vector<int64_t> sharedExpertXOptionalShape = param.sharedExpertXOptionalShape;
    vector<int64_t> xOutShape = param.xOutShape;
    char* groupEp = param.groupEp;
    int64_t epWorldSize = param.epWorldSize;
    int64_t epRankId = param.epRankId;
    int64_t moeExpertNum = param.moeExpertNum;
    int64_t expertShardType = param.expertShardType;
    int64_t sharedExpertNum = param.sharedExpertNum;
    int64_t sharedExpertRankNum = param.sharedExpertRankNum;
    int64_t globalBs = param.globalBs;
    int64_t commQuantMode = param.commQuantMode;
    int64_t commType = param.commType;
    char* commAlg = param.commAlg;
    aclDataType expandXDtype = param.expandXDtype;
    aclDataType quantExpandXDtype = param.quantExpandXDtype;
    aclDataType expertIdsDtype = param.expertIdsDtype;
    aclDataType expandIdxDtype = param.expandIdxDtype;
    aclDataType expertScalesDtype = param.expertScalesDtype;
    aclDataType commCmdInfoDtype = param.commCmdInfoDtype;
    aclDataType xActiveMaskOptionalDtype = param.xActiveMaskOptionalDtype;
    aclDataType sharedExpertXOptionalDtype = param.sharedExpertXOptionalDtype;
    aclDataType xOutDtype = param.xOutDtype;
    aclFormat expandXFormat = param.expandXFormat;
    aclFormat quantExpandXFormat = param.quantExpandXFormat;
    aclFormat expertIdsFormat = param.expertIdsFormat;
    aclFormat expandIdxFormat = param.expandIdxFormat;
    aclFormat expertScalesFormat = param.expertScalesFormat;
    aclFormat commCmdInfoFormat = param.commCmdInfoFormat;
    aclFormat xActiveMaskOptionalFormat = param.xActiveMaskOptionalFormat;
    aclFormat sharedExpertXOptionalFormat = param.sharedExpertXOptionalFormat;
    aclFormat xOutFormat = param.xOutFormat;
    aclnnStatus retStatus = param.aclnnStatusUt;
    aclTensor* expandX = CreateAclTensorOrNull(expandXShape, expandXDtype, expandXFormat);
    aclTensor* quantExpandX = CreateAclTensorOrNull(quantExpandXShape, quantExpandXDtype, quantExpandXFormat);
    aclTensor* expertIds = CreateAclTensorOrNull(expertIdsShape, expertIdsDtype, expertIdsFormat);
    aclTensor* expandIdx = CreateAclTensorOrNull(expandIdxShape, expandIdxDtype, expandIdxFormat);
    aclTensor* expertScales = CreateAclTensorOrNull(expertScalesShape, expertScalesDtype, expertScalesFormat);
    aclTensor* commCmdInfo = CreateAclTensorOrNull(commCmdInfoShape, commCmdInfoDtype, commCmdInfoFormat);
    aclTensor* xActiveMaskOptional = CreateAclTensorOrNull(xActiveMaskOptionalShape, xActiveMaskOptionalDtype, xActiveMaskOptionalFormat);
    aclTensor* sharedExpertXOptional = CreateAclTensorOrNull(sharedExpertXOptionalShape, sharedExpertXOptionalDtype, sharedExpertXOptionalFormat);
    aclTensor* xOut = CreateAclTensorOrNull(xOutShape, xOutDtype, xOutFormat);
    auto ut = OP_API_UT(aclnnMoeDistributeCombineTeardown,
                        INPUT(expandX, quantExpandX, expertIds, expandIdx, expertScales, commCmdInfo,
                              xActiveMaskOptional, sharedExpertXOptional,
                              groupEp, epWorldSize, epRankId, moeExpertNum, expertShardType, sharedExpertNum,
                              sharedExpertRankNum, globalBs, commQuantMode, commType, commAlg),
                        OUTPUT(xOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    if (retStatus == ACLNN_SUCCESS) {
        EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, retStatus);
    }
}

TEST_F(TestAclnnMoeDistributeCombineTeardown, CasesParamsTest)
{
    if (std::size(g_casesParams) != 0) {
        uint64_t numCases = sizeof(g_casesParams) / sizeof(g_casesParams[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestOneParamCase(g_casesParams[idx]);
        }
    }
}