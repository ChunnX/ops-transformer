/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../op_api/aclnn_mega_moe.h"

#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "opdev/op_log.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "common/op_api_def.h"

using namespace op;
using namespace std;

class l2_mega_moe_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "mega_moe_test Setup" << std::endl; }
    static void TearDownTestCase() { std::cout << "mega_moe_test TearDown" << std::endl; }
};

TEST_F(l2_mega_moe_test, ascend950_test_mega_moe_bf16_basic) {
    vector<int64_t> x_dims = {128, 512};
    vector<int64_t> expert_ids_dims = {128, 2};
    vector<int64_t> scales_dims = {128, 2};
    vector<int64_t> x_active_mask_dims = {128};
    vector<int64_t> expert_scales_dims = {128, 2};
    vector<int64_t> weight1_dims = {4, 512, 256};
    vector<int64_t> weightscales1_dims = {4, 256, 8};
    vector<int64_t> weight2_dims = {4, 128, 512};
    vector<int64_t> weightscales2_dims = {4, 512, 2};
    vector<int64_t> y_dims = {128, 512};

    auto x_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expert_ids_desc = TensorDesc(expert_ids_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto scales_desc = TensorDesc(scales_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x_active_mask_desc = TensorDesc(x_active_mask_dims, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
    auto expert_scales_desc = TensorDesc(expert_scales_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weight1_desc = TensorDesc(weight1_dims, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightscales1_desc = TensorDesc(weightscales1_dims, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2);
    auto weight2_desc = TensorDesc(weight2_dims, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightscales2_desc = TensorDesc(weightscales2_dims, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2);
    auto y_desc = TensorDesc(y_dims, ACL_BF16, ACL_FORMAT_ND);

    const char* group_ep = "ep_group";
    int64_t ep_world_size = 1;
    int64_t ep_rank_id = 0;
    int64_t moe_expert_num = 4;
    int64_t max_recv_token_num = 256;
    int64_t share_expert_num = 1;
    int64_t share_expert_rank_num = 0;
    int64_t dispatch_quant_mode = 0;
    int64_t dispatch_quant_out_type = ACL_DT_UNDEFINED;
    int64_t combine_quant_mode = 0;
    const char* comm_alg = "";
    int64_t global_bs = 128;

    auto ut = OP_API_UT(aclnnMegaMoeGetWorkspaceSize,
                        INPUT(x_desc, expert_ids_desc, scales_desc, x_active_mask_desc, expert_scales_desc,
                              weight1_desc, weightscales1_desc, weight2_desc, weightscales2_desc,
                              group_ep, ep_world_size, ep_rank_id, moe_expert_num, max_recv_token_num,
                              share_expert_num, share_expert_rank_num, dispatch_quant_mode, dispatch_quant_out_type,
                              combine_quant_mode, comm_alg, global_bs),
                        OUTPUT(y_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_mega_moe_test, ascend950_test_mega_moe_fp16_basic) {
    vector<int64_t> x_dims = {128, 512};
    vector<int64_t> expert_ids_dims = {128, 2};
    vector<int64_t> scales_dims = {128, 2};
    vector<int64_t> x_active_mask_dims = {128};
    vector<int64_t> expert_scales_dims = {128, 2};
    vector<int64_t> weight1_dims = {4, 512, 256};
    vector<int64_t> weightscales1_dims = {4, 256, 8};
    vector<int64_t> weight2_dims = {4, 128, 512};
    vector<int64_t> weightscales2_dims = {4, 512, 2};
    vector<int64_t> y_dims = {128, 512};

    auto x_desc = TensorDesc(x_dims, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expert_ids_desc = TensorDesc(expert_ids_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto scales_desc = TensorDesc(scales_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x_active_mask_desc = TensorDesc(x_active_mask_dims, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
    auto expert_scales_desc = TensorDesc(expert_scales_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weight1_desc = TensorDesc(weight1_dims, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightscales1_desc = TensorDesc(weightscales1_dims, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2);
    auto weight2_desc = TensorDesc(weight2_dims, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightscales2_desc = TensorDesc(weightscales2_dims, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2);
    auto y_desc = TensorDesc(y_dims, ACL_FLOAT16, ACL_FORMAT_ND);

    const char* group_ep = "ep_group";
    int64_t ep_world_size = 1;
    int64_t ep_rank_id = 0;
    int64_t moe_expert_num = 4;
    int64_t max_recv_token_num = 256;

    auto ut = OP_API_UT(aclnnMegaMoeGetWorkspaceSize,
                        INPUT(x_desc, expert_ids_desc, scales_desc, x_active_mask_desc, expert_scales_desc,
                              weight1_desc, weightscales1_desc, weight2_desc, weightscales2_desc,
                              group_ep, ep_world_size, ep_rank_id, moe_expert_num, max_recv_token_num,
                              1, 0, 0, ACL_DT_UNDEFINED, 0, "", 128),
                        OUTPUT(y_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_mega_moe_test, ascend950_test_mega_moe_ep_2cards) {
    vector<int64_t> x_dims = {128, 512};
    vector<int64_t> expert_ids_dims = {128, 2};
    vector<int64_t> weight1_dims = {2, 512, 256};
    vector<int64_t> weightscales1_dims = {2, 256, 8};
    vector<int64_t> weight2_dims = {2, 128, 512};
    vector<int64_t> weightscales2_dims = {2, 512, 2};
    vector<int64_t> y_dims = {128, 512};

    auto x_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expert_ids_desc = TensorDesc(expert_ids_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto scales_desc = TensorDesc(scales_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto x_active_mask_desc = TensorDesc(x_active_mask_dims, ACL_BOOL, ACL_FORMAT_ND);
    auto expert_scales_desc = TensorDesc(expert_scales_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weight1_desc = TensorDesc(weight1_dims, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightscales1_desc = TensorDesc(weightscales1_dims, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2);
    auto weight2_desc = TensorDesc(weight2_dims, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightscales2_desc = TensorDesc(weightscales2_dims, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2);
    auto y_desc = TensorDesc(y_dims, ACL_BF16, ACL_FORMAT_ND);

    const char* group_ep = "ep_group_2";
    int64_t ep_world_size = 2;
    int64_t ep_rank_id = 0;
    int64_t moe_expert_num = 4;

    auto ut = OP_API_UT(aclnnMegaMoeGetWorkspaceSize,
                        INPUT(x_desc, expert_ids_desc, scales_desc, x_active_mask_desc, expert_scales_desc,
                              weight1_desc, weightscales1_desc, weight2_desc, weightscales2_desc,
                              group_ep, ep_world_size, ep_rank_id, moe_expert_num, 512,
                              1, 0, 0, ACL_DT_UNDEFINED, 0, "", 128),
                        OUTPUT(y_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_mega_moe_test, ascend950_test_mega_moe_invalid_x_dtype) {
    vector<int64_t> x_dims = {128, 512};
    vector<int64_t> expert_ids_dims = {128, 2};

    auto x_desc = TensorDesc(x_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expert_ids_desc = TensorDesc(expert_ids_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto y_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoeGetWorkspaceSize,
                        INPUT(x_desc, expert_ids_desc, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              "ep_group", 1, 0, 4, 256, 1, 0, 0, ACL_DT_UNDEFINED, 0, "", 128),
                        OUTPUT(y_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mega_moe_test, ascend950_test_mega_moe_invalid_expert_ids_dtype) {
    vector<int64_t> x_dims = {128, 512};
    vector<int64_t> expert_ids_dims = {128, 2};

    auto x_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expert_ids_desc = TensorDesc(expert_ids_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    auto y_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoeGetWorkspaceSize,
                        INPUT(x_desc, expert_ids_desc, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              "ep_group", 1, 0, 4, 256, 1, 0, 0, ACL_DT_UNDEFINED, 0, "", 128),
                        OUTPUT(y_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mega_moe_test, ascend950_test_mega_moe_invalid_x_shape) {
    vector<int64_t> x_dims = {128};
    vector<int64_t> expert_ids_dims = {128, 2};

    auto x_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expert_ids_desc = TensorDesc(expert_ids_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto y_desc = TensorDesc(vector<int64_t>{128, 512}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoeGetWorkspaceSize,
                        INPUT(x_desc, expert_ids_desc, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              "ep_group", 1, 0, 4, 256, 1, 0, 0, ACL_DT_UNDEFINED, 0, "", 128),
                        OUTPUT(y_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mega_moe_test, ascend950_test_mega_moe_invalid_ep_rank_id) {
    vector<int64_t> x_dims = {128, 512};
    vector<int64_t> expert_ids_dims = {128, 2};

    auto x_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expert_ids_desc = TensorDesc(expert_ids_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto y_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND);

    const char* group_ep = "ep_group";
    int64_t ep_world_size = 2;
    int64_t ep_rank_id = 3;

    auto ut = OP_API_UT(aclnnMegaMoeGetWorkspaceSize,
                        INPUT(x_desc, expert_ids_desc, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              group_ep, ep_world_size, ep_rank_id, 4, 256, 1, 0, 0, ACL_DT_UNDEFINED, 0, "", 128),
                        OUTPUT(y_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mega_moe_test, ascend950_test_mega_moe_output_dtype_mismatch) {
    vector<int64_t> x_dims = {128, 512};
    vector<int64_t> expert_ids_dims = {128, 2};

    auto x_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expert_ids_desc = TensorDesc(expert_ids_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto y_desc = TensorDesc(x_dims, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoeGetWorkspaceSize,
                        INPUT(x_desc, expert_ids_desc, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              "ep_group", 1, 0, 4, 256, 1, 0, 0, ACL_DT_UNDEFINED, 0, "", 128),
                        OUTPUT(y_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mega_moe_test, ascend950_test_mega_moe_optional_params_nullptr) {
    vector<int64_t> x_dims = {128, 512};
    vector<int64_t> expert_ids_dims = {128, 2};
    vector<int64_t> weight1_dims = {4, 512, 256};
    vector<int64_t> weightscales1_dims = {4, 256, 8};
    vector<int64_t> weight2_dims = {4, 128, 512};
    vector<int64_t> weightscales2_dims = {4, 512, 2};

    auto x_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expert_ids_desc = TensorDesc(expert_ids_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto expert_scales_desc = TensorDesc(expert_ids_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weight1_desc = TensorDesc(weight1_dims, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightscales1_desc = TensorDesc(weightscales1_dims, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2);
    auto weight2_desc = TensorDesc(weight2_dims, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightscales2_desc = TensorDesc(weightscales2_dims, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2);
    auto y_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoeGetWorkspaceSize,
                        INPUT(x_desc, expert_ids_desc, nullptr, nullptr, expert_scales_desc,
                              weight1_desc, weightscales1_desc, weight2_desc, weightscales2_desc,
                              "ep_group", 1, 0, 4, 256, 1, 0, 0, ACL_DT_UNDEFINED, 0, "", 128),
                        OUTPUT(y_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_mega_moe_test, ascend950_test_mega_moe_topK_1) {
    vector<int64_t> x_dims = {128, 512};
    vector<int64_t> expert_ids_dims = {128, 1};
    vector<int64_t> weight1_dims = {4, 512, 256};
    vector<int64_t> weightscales1_dims = {4, 256, 8};
    vector<int64_t> weight2_dims = {4, 128, 512};
    vector<int64_t> weightscales2_dims = {4, 512, 2};

    auto x_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expert_ids_desc = TensorDesc(expert_ids_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto expert_scales_desc = TensorDesc(expert_ids_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weight1_desc = TensorDesc(weight1_dims, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightscales1_desc = TensorDesc(weightscales1_dims, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2);
    auto weight2_desc = TensorDesc(weight2_dims, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightscales2_desc = TensorDesc(weightscales2_dims, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2);
    auto y_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoeGetWorkspaceSize,
                        INPUT(x_desc, expert_ids_desc, nullptr, nullptr, expert_scales_desc,
                              weight1_desc, weightscales1_desc, weight2_desc, weightscales2_desc,
                              "ep_group", 1, 0, 4, 128, 1, 0, 0, ACL_DT_UNDEFINED, 0, "", 128),
                        OUTPUT(y_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}