/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "../../../../op_host/op_api/aclnn_grouped_matmul_finalize_routing_v3.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;
using namespace op;

namespace {
struct V3Case {
    string name;
    vector<int64_t> x1Shape;
    vector<int64_t> x2Shape;
    vector<int64_t> scaleShape;
    vector<int64_t> biasShape;
    vector<int64_t> offsetShape;
    vector<int64_t> perTokenShape;
    vector<int64_t> groupListShape;
    vector<int64_t> sharedInputShape;
    vector<int64_t> logitShape;
    vector<int64_t> rowIndexShape;
    aclDataType x1Type;
    aclDataType x2Type;
    aclDataType scaleType;
    aclDataType offsetType;
    aclDataType perTokenType;
    aclFormat x2Format;
    bool passOffset;
    bool passLogit;
    bool passTuning;
    bool passAntiquant;
    int64_t dtypeAttr;
    int64_t tuningVal;
    bool transposeW;
    bool transposeX;
    aclnnStatus expect;
};

TensorDesc BuildX2(const V3Case &c)
{
    if (c.x2Format == ACL_FORMAT_FRACTAL_NZ) {
        return TensorDesc(c.x2Shape, c.x2Type, c.x2Format).ValueRange(-1, 1);
    }
    return TensorDesc(c.x2Shape, c.x2Type, c.x2Format).ValueRange(-1, 1);
}

aclnnStatus RunCase(const V3Case &c)
{
    TensorDesc x1Desc = TensorDesc(c.x1Shape, c.x1Type, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2Desc = BuildX2(c);
    TensorDesc scaleDesc = TensorDesc(c.scaleShape, c.scaleType, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc biasDesc = TensorDesc(c.biasShape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc offsetDesc = TensorDesc(c.offsetShape, c.offsetType, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc antiquantDesc = TensorDesc({4, 7168}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc perTokenDesc = TensorDesc(c.perTokenShape, c.perTokenType, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupListDesc = TensorDesc(c.groupListShape, ACL_INT64, ACL_FORMAT_ND).ValueRange(24, 24);
    TensorDesc sharedInputDesc = TensorDesc(c.sharedInputShape, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logitDesc = TensorDesc(c.logitShape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc rowIndexDesc = TensorDesc(c.rowIndexShape, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc outDesc = TensorDesc({24, 7168}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    aclIntArray *tuning = nullptr;
    vector<int64_t> tuningVec = {c.tuningVal};
    if (c.passTuning) {
        tuning = aclCreateIntArray(tuningVec.data(), tuningVec.size());
    }

    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingV3,
                        INPUT(x1Desc, x2Desc, scaleDesc, biasDesc, c.passOffset ? &offsetDesc : nullptr,
                              c.passAntiquant ? &antiquantDesc : nullptr, nullptr, perTokenDesc, groupListDesc, sharedInputDesc,
                              c.passLogit ? &logitDesc : nullptr, rowIndexDesc, c.dtypeAttr, 1.0, 0, c.transposeW, c.transposeX, 1,
                              tuning),
                        OUTPUT(outDesc));
    uint64_t workspace = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspace);
    if (tuning != nullptr) {
        aclDestroyIntArray(tuning);
    }
    return ret;
}

vector<V3Case> BuildCases()
{
    return {
        {"w4a8_offset_normal", {192, 2048}, {4, 2048, 7168 / 8}, {4, 1, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168},
         {192}, {192}, ACL_INT8, ACL_INT32, ACL_INT64, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, true, true, true, false, 0, 10, false,
         false, ACLNN_SUCCESS},
        {"offset_dim_value_error", {192, 2048}, {4, 2048, 7168 / 8}, {4, 1, 7168}, {4, 7168}, {4, 1, 7169}, {192}, {4}, {8, 7168},
         {192}, {192}, ACL_INT8, ACL_INT32, ACL_INT64, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, true, true, true, false, 0, 0, false,
         false, ACLNN_ERR_PARAM_INVALID},
        {"dtype_attr_error", {192, 2048}, {4, 2048, 7168 / 8}, {4, 1, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192},
         {192}, ACL_INT8, ACL_INT32, ACL_INT64, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, true, true, true, false, 1, 0, false, false,
         ACLNN_ERR_PARAM_INVALID},
        {"transpose_w_not_support", {192, 2048}, {4, 2048, 7168 / 8}, {4, 8, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168},
         {192}, {192}, ACL_INT8, ACL_INT32, ACL_INT64, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, true, true, true, false, 0, 0,
         true, false, ACLNN_ERR_PARAM_INVALID},
        {"x2_dim_num_error", {192, 2048}, {4}, {4, 8, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192}, {192}, ACL_INT8,
         ACL_INT32, ACL_INT64, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, true, true, true, false, 0, 0, true, false,
         ACLNN_ERR_PARAM_INVALID},
        {"x2_not_int32", {192, 2048}, {4, 2048, 7168 / 8}, {4, 8, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192},
         {192}, ACL_INT8, ACL_INT4, ACL_INT64, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, true, true, true, false, 0, 0, true, false,
         ACLNN_ERR_PARAM_INVALID},
        {"input_antiquant_wrong", {192, 2048}, {4, 2048, 7168 / 8}, {4, 1, 7168}, {4, 7168}, {4, 7168}, {192}, {4}, {8, 7168},
         {192}, {192}, ACL_INT8, ACL_INT32, ACL_INT64, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, true, true, true, true, 0, 0, false,
         false, ACLNN_ERR_PARAM_NULLPTR},
        {"offset_dtype_error", {192, 2048}, {4, 2048, 7168 / 8}, {4, 1, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168},
         {192}, {192}, ACL_INT8, ACL_INT32, ACL_INT64, ACL_INT32, ACL_FLOAT, ACL_FORMAT_ND, true, true, true, false, 0, 0, false,
         false, ACLNN_ERR_PARAM_INVALID},
        {"transpose_x_not_support", {192, 2048}, {4, 2048, 7168 / 8}, {4, 8, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168},
         {192}, {192}, ACL_INT8, ACL_INT32, ACL_INT64, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, true, true, true, false, 0, 0,
         false, true, ACLNN_ERR_PARAM_INVALID},
        {"weight_dtype_not_support", {192, 2048}, {4, 2048, 7168 / 8}, {4, 8, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168},
         {192}, {192}, ACL_INT8, ACL_INT8, ACL_INT64, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, true, true, true, false, 0, 0,
         false, false, ACLNN_ERR_PARAM_INVALID},
        {"tuning_invalid", {192, 2048}, {4, 2048, 7168 / 8}, {4, 8, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192},
         {192}, ACL_INT8, ACL_INT8, ACL_INT64, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, true, true, true, false, 0, -5, false,
         false, ACLNN_ERR_PARAM_INVALID},
        {"logits_null", {192, 2048}, {4, 2048, 7168 / 8}, {4, 1, 7168}, {4, 7168}, {5, 1, 7169}, {192}, {4}, {8, 7168}, {192},
         {192}, ACL_INT8, ACL_INT32, ACL_INT64, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, true, false, true, false, 0, 0, false, false,
         ACLNN_ERR_PARAM_NULLPTR},
        {"mxfp8_normal", {192, 2048}, {4, 2048, 7168}, {4, 2048 / 64, 7168, 2}, {4, 7168}, {4, 1, 7168}, {192, 2048 / 64, 2}, {4},
         {8, 7168}, {192}, {192}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FORMAT_ND,
         false, true, false, false, 0, 0, false, false, ACLNN_SUCCESS},
    };
}

class l2_GroupedMatmulFinalizeRoutingV3_test : public testing::TestWithParam<V3Case> {};

TEST_P(l2_GroupedMatmulFinalizeRoutingV3_test, tableDriven)
{
    const auto ret = RunCase(GetParam());
    EXPECT_EQ(ret, GetParam().expect);
}

string MakeName(const testing::TestParamInfo<V3Case> &info)
{
    string name = info.param.name;
    transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return isalnum(c) ? c : '_'; });
    return name;
}

INSTANTIATE_TEST_SUITE_P(grouped_matmul_finalize_routing_v3, l2_GroupedMatmulFinalizeRoutingV3_test,
                         testing::ValuesIn(BuildCases()), MakeName);
} // namespace
