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
#include <thread>
#include <gmock/gmock.h>
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../../op_api/aclnn_grouped_matmul_finalize_routing_weight_nz_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_GroupedMatmulFinalizeRoutingWeightNzV2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_GroupedMatmulFinalizeRoutingWeightNzV2_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_GroupedMatmulFinalizeRoutingWeightNzV2_test TearDown" << endl;
    }
};

// TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_add_value)
// {
//     int64_t m = 192;
//     int64_t k = 2048;
//     int64_t n = 7168;
//     int64_t e = 4;
//     int64_t bs = 24;
//     int64_t bsdp = 8;
//     TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
//     TensorDesc x2_desc =
//         TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
//     TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
//     TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
//     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
//     TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
//     TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
//     TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     vector<int64_t> tuningConfigVal = { m / e };
//     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
//     auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
//                         INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
//                               groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
//                               1, tuningConfig),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_transpose_test)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { m / e };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, true, true,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_shared_input_null)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> tuningConfigVal = { m / e };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, nullptr, nullptr, 0, 1.0, 0, false, false, 1,
                              tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_bias_not_null)
// {
//     int64_t m = 192;
//     int64_t k = 2048;
//     int64_t n = 7168;
//     int64_t e = 4;
//     int64_t bs = 24;
//     int64_t bsdp = 8;
//     TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
//     TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
//     TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
//     TensorDesc bias_desc = TensorDesc({e, n}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
//     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
//     TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
//     TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
//     TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
//     vector<int64_t> tuningConfigVal = { m / e };
//     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
//     auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
//                         INPUT(x1_desc, x2_desc, scale_desc, bias_desc, nullptr, nullptr, nullptr, perTokenScale_desc,
//                               groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
//                               1, tuningConfig),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_x2_not_nz)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> tuningConfigVal = { m / e };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_dtype_not_0)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 1, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_scale_noteq_e)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e + 1, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);\
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_perTokenScale_noteq_m)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m + 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_shared_input_more_than_bs)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m + 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bs + 1, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_group_list_type_not_1)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m + 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bs + 1, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_view_shape_0)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({0, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m + 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bs + 1, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_x12_viewdims_wrong)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> tuningConfigVal = {0};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_shareinput_outputBS_noeq)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp + 1, n + 1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_logit_1stdim_noeq_xMdim)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m + 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_rowindex1stdim_noeq_xMdim)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m + 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_groupListdim_wrong)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e + 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_weightview0)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({0}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_nvalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7000;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({0}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_not_support_k_shape_case)
{
    int64_t m = 192;
    int64_t k = 4098;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_x1_type_not_same_with_x2)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_scale_shape_invalid_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e + 1, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_k_n_not_support_w8a8_case)
{
    int64_t m = 192;
    int64_t k = 2048 - 1;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n / 8}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, k / quantGroupSize, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_tuning_config_invalid_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { -5 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_w4a8_normal_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n / 8}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 64, k / 16, 16, 8}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, 1, n}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, nullptr, nullptr, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_w4a8_offset_normal_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n / 8}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 64, k / 16, 16, 8}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, 1, n}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc offset_desc = TensorDesc({e, 1, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, offset_desc, nullptr, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_w4a8_x2_not_int32_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    // NOTE: This test case is adapted from 'l2_GroupedMatmulFinalizeRoutingV3_test',
    // not sure what the original intention was.
    TensorDesc x2_desc =
        TensorDesc({e, k, n / 8}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 64, k / 16, 16, 8}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, 1, n}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, nullptr, nullptr, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_w4a8_scale_not_int64_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n / 8}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 64, k / 16, 16, 8}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, 1, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, nullptr, nullptr, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_w4a8_scale_dim_not_3_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n / 8}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 64, k / 16, 16, 8}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, nullptr, nullptr, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}


// TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_w4a8_pertoken_scale_null_case)
// {
//     int64_t m = 192;
//     int64_t k = 2048;
//     int64_t n = 7168;
//     int64_t e = 4;
//     int64_t bs = 24;
//     int64_t bsdp = 8;
//     int64_t quantGroupSize = 256;

//     TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
//     TensorDesc x2_desc =
//         TensorDesc({e, k, n / 8}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 64, k / 16, 16, 8}).ValueRange(-1, 1);
//     TensorDesc scale_desc = TensorDesc({e, 1, n}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
//     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
//     TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
//     TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
//     TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     vector<int64_t> tuningConfigVal = { 10 };
//     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
//     auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
//                         INPUT(x1_desc, x2_desc, scale_desc, bias_desc, nullptr, nullptr, nullptr, nullptr, groupList_desc,
//                               shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1, tuningConfig),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
// }

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend950_test_normal_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend950_test_invalid_case_scale_nullptr)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, nullptr, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet,ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend950_test_invalid_case_dim_mismatch)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ==================== MxA8W4 Test Cases (Ascend950 Only) ====================

class l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950 SetUp" << endl;
        op::SetPlatformNpuArch(NpuArch::DAV_3510);
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950 TearDown" << endl;
    }
};

static inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_normal_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_with_bias)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_scale_dim_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    // scale维度错误：3维而非4维
    TensorDesc scale_desc = TensorDesc({e, 1, n}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, 32, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_x2_format_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    // x2格式错误：ND而非NZ
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_shared_input_offset_overflow)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 20;
    int64_t sharedInputOffset = 10;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    // sharedInputOffset + bsdp = 10 + 20 = 30 > outputBS = 24
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, sharedInputOffset,
                              false, true, 1, nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ==================== 约束条件负用例 ====================

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_k_not_aligned)
{
    int64_t m = 192;
    int64_t k = 2050;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_n_not_aligned)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7170;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_experts_exceeded)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 1025;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_bsdp_exceed_output)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 30;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ==================== 必选参数nullptr负用例 ====================

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_pertoken_scale_nullptr)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1, nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ==================== dtype错误负用例 ====================

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_x1_dtype_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_x2_dtype_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc =
        TensorDesc({e, n, k}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_scale_dtype_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ==================== 维度错误负用例 ====================

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_x1_dim_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, 1, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_pertoken_scale_dim_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, 32, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, 32}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_group_list_dim_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e, 1}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_out_dim_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ==================== shape一致性负用例 ====================

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_scale_shape_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64 + 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_pertoken_scale_shape_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64 + 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_bias_shape_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({e, n + 1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_x2_shape_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(n, 32), CeilDiv(k, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ==================== 格式错误负用例 ====================

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_x1_format_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_FRACTAL_NZ);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_scale_format_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_FRACTAL_NZ);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ==================== 边界值正向用例 ====================

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_min_k_n)
{
    int64_t m = 192;
    int64_t k = 32;
    int64_t n = 32;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_max_experts)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 1024;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_single_token)
{
    int64_t m = 1;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 1;
    int64_t bsdp = 1;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_no_shared_input)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, nullptr, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1, nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_cumsum_group_list)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 0,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// ==================== 泛化参数正向用例 ====================

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_large_k_n)
{
    int64_t m = 256;
    int64_t k = 4096;
    int64_t n = 16384;
    int64_t e = 8;
    int64_t bs = 32;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_many_experts)
{
    int64_t m = 512;
    int64_t k = 2048;
    int64_t n = 4096;
    int64_t e = 128;
    int64_t bs = 64;
    int64_t bsdp = 16;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_zero_offset)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 24;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
