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

#include "../../../../op_host/op_api/aclnn_grouped_matmul_finalize_routing_weight_nz_v2.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;
using namespace op;

namespace {
struct WeightNzV2Case {
    string name;
    vector<int64_t> x1Shape;
    vector<int64_t> x2Shape;
    vector<int64_t> scaleShape;
    vector<int64_t> biasShape;
    vector<int64_t> offsetShape;
    vector<int64_t> perTokenShape;
    vector<int64_t> groupListShape;
    vector<int64_t> sharedInputShape;
    vector<int64_t> logitsShape;
    vector<int64_t> rowIndexShape;
    vector<int64_t> outShape;
    aclDataType x1Type;
    aclDataType x2Type;
    aclDataType scaleType;
    aclDataType biasType;
    aclDataType offsetType;
    aclDataType perTokenType;
    aclDataType sharedInputType;
    aclFormat x2Format;
    bool passBias;
    bool passOffset;
    bool passScale;
    bool passLogits;
    bool passRowIndex;
    bool passTuning;
    int64_t dtypeAttr;
    int64_t tuningVal;
    bool transposeW;
    bool transposeX;
    aclnnStatus expect;
};

aclnnStatus RunCase(const WeightNzV2Case &c)
{
    TensorDesc x1Desc = TensorDesc(c.x1Shape, c.x1Type, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2Desc = TensorDesc(c.x2Shape, c.x2Type, c.x2Format).ValueRange(-1, 1);
    TensorDesc scaleDesc = TensorDesc(c.scaleShape, c.scaleType, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc biasDesc = TensorDesc(c.biasShape, c.biasType, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc offsetDesc = TensorDesc(c.offsetShape, c.offsetType, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc perTokenDesc = TensorDesc(c.perTokenShape, c.perTokenType, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupListDesc = TensorDesc(c.groupListShape, ACL_INT64, ACL_FORMAT_ND).ValueRange(24, 24);
    TensorDesc sharedInputDesc = TensorDesc(c.sharedInputShape, c.sharedInputType, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logitsDesc = TensorDesc(c.logitsShape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc rowIndexDesc = TensorDesc(c.rowIndexShape, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc outDesc = TensorDesc(c.outShape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    aclIntArray *tuning = nullptr;
    vector<int64_t> tuningVec = {c.tuningVal};
    if (c.passTuning) {
        tuning = aclCreateIntArray(tuningVec.data(), tuningVec.size());
    }

    auto ut = c.passScale
                  ? OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                              INPUT(x1Desc, x2Desc, scaleDesc, c.passBias ? &biasDesc : nullptr,
                                    c.passOffset ? &offsetDesc : nullptr, nullptr, nullptr, perTokenDesc, groupListDesc,
                                    sharedInputDesc, c.passLogits ? &logitsDesc : nullptr,
                                    c.passRowIndex ? &rowIndexDesc : nullptr, c.dtypeAttr, 1.0, 0, c.transposeW,
                                    c.transposeX, 1, tuning),
                              OUTPUT(outDesc))
                  : OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                              INPUT(x1Desc, x2Desc, nullptr, c.passBias ? &biasDesc : nullptr,
                                    c.passOffset ? &offsetDesc : nullptr, nullptr, nullptr, perTokenDesc, groupListDesc,
                                    sharedInputDesc, c.passLogits ? &logitsDesc : nullptr,
                                    c.passRowIndex ? &rowIndexDesc : nullptr, c.dtypeAttr, 1.0, 0, c.transposeW,
                                    c.transposeX, 1, tuning),
                              OUTPUT(outDesc));
    uint64_t workspace = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspace);

    if (tuning != nullptr) {
        aclDestroyIntArray(tuning);
    }
    return ret;
}

vector<WeightNzV2Case> BuildCases()
{
    return {
        {"transpose_not_support", {192, 2048}, {4, 2048, 7168}, {4, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192},
         {192}, {24, 7168}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, false,
         false, true, true, true, true, 0, 48, true, true, ACLNN_ERR_PARAM_INVALID},
        {"shared_input_nullptr", {192, 2048}, {4, 2048, 7168}, {4, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192},
         {192}, {24, 7168}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, false,
         false, true, false, false, true, 0, 48, false, false, ACLNN_ERR_PARAM_NULLPTR},
        {"x2_not_nz", {192, 2048}, {4, 2048, 7168}, {4, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192}, {192},
         {24, 7168}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_ND, false, false, true,
         true, true, true, 0, 48, false, false, ACLNN_ERR_PARAM_INVALID},
        {"dtype_attr_not_zero", {192, 2048}, {4, 2048, 7168}, {4, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192},
         {192}, {24, 7168}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, false,
         false, true, true, true, true, 1, 0, false, false, ACLNN_ERR_PARAM_INVALID},
        {"scale_first_dim_not_e", {192, 2048}, {4, 2048, 7168}, {5, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192},
         {192}, {24, 7168}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, false,
         false, true, true, true, true, 0, 0, false, false, ACLNN_ERR_PARAM_INVALID},
        {"per_token_not_m", {192, 2048}, {4, 2048, 7168}, {4, 7168}, {4, 7168}, {4, 1, 7168}, {193}, {4}, {8, 7168}, {192}, {192},
         {24, 7168}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, false, false,
         true, true, true, true, 0, 0, false, false, ACLNN_ERR_PARAM_INVALID},
        {"shared_input_bs_exceed", {192, 2048}, {4, 2048, 7168}, {4, 7168}, {4, 7168}, {4, 1, 7168}, {193}, {4}, {25, 7168}, {192},
         {192}, {24, 7168}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, false,
         false, true, true, true, true, 0, 0, false, false, ACLNN_ERR_PARAM_INVALID},
        {"m_zero", {0, 2048}, {4, 2048, 7168}, {4, 7168}, {4, 7168}, {4, 1, 7168}, {193}, {4}, {25, 7168}, {192}, {192}, {24, 7168},
         ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, false, false, true, true,
         true, true, 0, 0, false, false, ACLNN_ERR_PARAM_INVALID},
        {"x1_dim_invalid", {192}, {4, 2048, 7168}, {4, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192}, {192},
         {24, 7168}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, false, false,
         true, true, true, true, 0, 0, false, false, ACLNN_ERR_PARAM_INVALID},
        {"shared_input_out_bs_mismatch", {192, 2048}, {4, 2048, 7168}, {4, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {9, 7169},
         {192}, {192}, {24, 7168}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ,
         false, false, true, true, true, true, 0, 0, false, false, ACLNN_ERR_PARAM_INVALID},
        {"logits_not_m", {192, 2048}, {4, 2048, 7168}, {4, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {193}, {192},
         {24, 7168}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, false, false,
         true, true, true, true, 0, 0, false, false, ACLNN_ERR_PARAM_INVALID},
        {"row_index_not_m", {192, 2048}, {4, 2048, 7168}, {4, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192}, {193},
         {24, 7168}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, false, false,
         true, true, true, true, 0, 0, false, false, ACLNN_ERR_PARAM_INVALID},
        {"group_list_not_e", {192, 2048}, {4, 2048, 7168}, {4, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {5}, {8, 7168}, {192}, {192},
         {24, 7168}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, false, false,
         true, true, true, true, 0, 0, false, false, ACLNN_ERR_PARAM_INVALID},
        {"x2_shape_zero", {192, 2048}, {0}, {4, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192}, {192}, {24, 7168},
         ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, false, false, true, true,
         true, true, 0, 0, false, false, ACLNN_ERR_PARAM_INVALID},
        {"n_invalid", {192, 2048}, {0}, {4, 7000}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7000}, {192}, {192}, {24, 7000},
         ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, false, false, true, true,
         true, true, 0, 0, false, false, ACLNN_ERR_PARAM_INVALID},
        {"k_not_supported", {192, 4098}, {4, 4098, 7168}, {4, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192}, {192},
         {24, 7168}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, false, false,
         true, true, true, true, 0, 0, false, false, ACLNN_ERR_PARAM_INVALID},
        {"x1_x2_type_mismatch", {192, 2048}, {4, 2048, 7168}, {4, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192},
         {192}, {24, 7168}, ACL_INT8, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, false,
         false, true, true, true, true, 0, 0, false, false, ACLNN_ERR_PARAM_INVALID},
        {"scale_shape_invalid", {192, 2048}, {4, 2048, 7168}, {5, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192},
         {192}, {24, 7168}, ACL_INT8, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, false,
         false, true, true, true, true, 0, 0, false, false, ACLNN_ERR_PARAM_INVALID},
        {"w8a8_kn_not_support", {192, 2047}, {4, 2047, 896}, {4, 7, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192},
         {192}, {24, 7168}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, false,
         false, true, true, true, true, 0, 0, false, false, ACLNN_ERR_PARAM_INVALID},
        {"tuning_invalid", {192, 2048}, {4, 2048, 7168}, {4, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192}, {192},
         {24, 7168}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, false, false,
         true, true, true, true, 0, -5, false, false, ACLNN_ERR_PARAM_INVALID},
        {"w4a8_normal", {192, 2048}, {4, 2048, 896}, {4, 1, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192}, {192},
         {24, 7168}, ACL_INT8, ACL_INT32, ACL_INT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, true, false,
         true, true, true, true, 0, 10, false, false, ACLNN_SUCCESS},
        {"w4a8_with_offset", {192, 2048}, {4, 2048, 896}, {4, 1, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192},
         {192}, {24, 7168}, ACL_INT8, ACL_INT32, ACL_INT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, true,
         true, true, true, true, true, 0, 10, false, false, ACLNN_SUCCESS},
        {"w4a8_x2_not_int32", {192, 2048}, {4, 2048, 896}, {4, 1, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192},
         {192}, {24, 7168}, ACL_INT8, ACL_INT4, ACL_INT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, true,
         false, true, true, true, true, 0, 10, false, false, ACLNN_ERR_PARAM_INVALID},
        {"w4a8_scale_not_int64", {192, 2048}, {4, 2048, 896}, {4, 1, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192},
         {192}, {24, 7168}, ACL_INT8, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, true,
         false, true, true, true, true, 0, 10, false, false, ACLNN_ERR_PARAM_INVALID},
        {"w4a8_scale_dim_not_3", {192, 2048}, {4, 2048, 896}, {4, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192},
         {192}, {24, 7168}, ACL_INT8, ACL_INT32, ACL_INT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, true,
         false, true, true, true, true, 0, 10, false, false, ACLNN_ERR_PARAM_INVALID},
        {"ascend950_normal", {192, 2048}, {4, 2048, 7168}, {4, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192}, {192},
         {24, 7168}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, false, false,
         true, true, true, false, 0, 0, false, false, ACLNN_SUCCESS},
        {"ascend950_scale_nullptr", {192, 2048}, {4, 2048, 7168}, {}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192}, {192},
         {24, 7168}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, false, false,
         false, true, true, false, 0, 0, false, false, ACLNN_ERR_PARAM_NULLPTR},
        {"ascend950_dim_mismatch", {192, 2048}, {4, 7168, 2048}, {4, 7168}, {4, 7168}, {4, 1, 7168}, {192}, {4}, {8, 7168}, {192},
         {192}, {24, 7168}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, false,
         false, true, true, true, false, 0, 0, false, false, ACLNN_ERR_PARAM_INVALID},
    };
}

class l2_GroupedMatmulFinalizeRoutingWeightNzV2_test : public testing::TestWithParam<WeightNzV2Case> {};

TEST_P(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, tableDriven)
{
    const auto ret = RunCase(GetParam());
    EXPECT_EQ(ret, GetParam().expect);
}

string MakeName(const testing::TestParamInfo<WeightNzV2Case> &info)
{
    string name = info.param.name;
    transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return isalnum(c) ? c : '_'; });
    return name;
}

INSTANTIATE_TEST_SUITE_P(grouped_matmul_finalize_routing_weight_nz_v2, l2_GroupedMatmulFinalizeRoutingWeightNzV2_test,
                         testing::ValuesIn(BuildCases()), MakeName);
} // namespace
