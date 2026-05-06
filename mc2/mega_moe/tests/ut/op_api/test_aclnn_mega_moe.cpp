/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "../../../op_api/aclnn_mega_moe.h"

using namespace op;
using namespace std;

class AclnnMegaMoeTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MegaMoe AclnnMegaMoeTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MegaMoe AclnnMegaMoeTest TearDown" << std::endl;
    }
};

TEST_F(AclnnMegaMoeTest, ascend950_bf16_basic) {
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    
    vector<int64_t> x_dims = {128, 5120};
    vector<int64_t> topk_ids_dims = {128, 8};
    vector<int64_t> topk_weights_dims = {128, 8};
    vector<int64_t> weight1_dims = {4, 4096, 5120};
    vector<int64_t> weightscales1_dims = {4, 4096, 80, 2};
    vector<int64_t> weight2_dims = {4, 5120, 2048};
    vector<int64_t> weightscales2_dims = {4, 5120, 32, 2};
    vector<int64_t> y_dims = {128, 5120};
    vector<int64_t> expert_token_nums_dims = {4};

    auto context_desc = TensorDesc({32}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_ids_desc = TensorDesc(topk_ids_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto topk_weights_desc = TensorDesc(topk_weights_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto weight1_desc = TensorDesc(weight1_dims, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightscales1_desc = TensorDesc(weightscales1_dims, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2);
    auto weight2_desc = TensorDesc(weight2_dims, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightscales2_desc = TensorDesc(weightscales2_dims, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2);
    auto y_desc = TensorDesc(y_dims, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc(expert_token_nums_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(context_desc, x_desc, topk_ids_desc, topk_weights_desc,
                              weight1_desc, weight2_desc, weightscales1_desc, weightscales2_desc,
                              nullptr, nullptr,
                              16, 4, 2097152, 0, 4, 24, 0, "", 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = ut.TestGetWorkspaceSize(&workspace_size, &executor);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

TEST_F(AclnnMegaMoeTest, ascend950_fp32_weights) {
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    
    vector<int64_t> x_dims = {256, 4096};
    vector<int64_t> topk_ids_dims = {256, 6};
    vector<int64_t> topk_weights_dims = {256, 6};
    vector<int64_t> weight1_dims = {8, 2048, 4096};
    vector<int64_t> weightscales1_dims = {8, 2048, 64, 2};
    vector<int64_t> weight2_dims = {8, 4096, 1024};
    vector<int64_t> weightscales2_dims = {8, 4096, 16, 2};
    vector<int64_t> y_dims = {256, 4096};
    vector<int64_t> expert_token_nums_dims = {8};

    auto context_desc = TensorDesc({32}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_ids_desc = TensorDesc(topk_ids_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);
    auto topk_weights_desc = TensorDesc(topk_weights_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto weight1_desc = TensorDesc(weight1_dims, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightscales1_desc = TensorDesc(weightscales1_dims, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2);
    auto weight2_desc = TensorDesc(weight2_dims, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightscales2_desc = TensorDesc(weightscales2_dims, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2);
    auto y_desc = TensorDesc(y_dims, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc(expert_token_nums_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(context_desc, x_desc, topk_ids_desc, topk_weights_desc,
                              weight1_desc, weight2_desc, weightscales1_desc, weightscales2_desc,
                              nullptr, nullptr,
                              64, 8, 2097152, 0, 4, 23, 0, "", 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = ut.TestGetWorkspaceSize(&workspace_size, &executor);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

TEST_F(AclnnMegaMoeTest, ascend950_large_bs) {
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    
    vector<int64_t> x_dims = {512, 7168};
    vector<int64_t> topk_ids_dims = {512, 8};
    vector<int64_t> topk_weights_dims = {512, 8};
    vector<int64_t> weight1_dims = {2, 7168, 7168};
    vector<int64_t> weightscales1_dims = {2, 7168, 112, 2};
    vector<int64_t> weight2_dims = {2, 7168, 3584};
    vector<int64_t> weightscales2_dims = {2, 7168, 56, 2};
    vector<int64_t> y_dims = {512, 7168};
    vector<int64_t> expert_token_nums_dims = {2};

    auto context_desc = TensorDesc({32}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_ids_desc = TensorDesc(topk_ids_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto topk_weights_desc = TensorDesc(topk_weights_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto weight1_desc = TensorDesc(weight1_dims, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightscales1_desc = TensorDesc(weightscales1_dims, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2);
    auto weight2_desc = TensorDesc(weight2_dims, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightscales2_desc = TensorDesc(weightscales2_dims, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2);
    auto y_desc = TensorDesc(y_dims, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc(expert_token_nums_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(context_desc, x_desc, topk_ids_desc, topk_weights_desc,
                              weight1_desc, weight2_desc, weightscales1_desc, weightscales2_desc,
                              nullptr, nullptr,
                              4, 2, 2097152, 0, 4, 24, 0, "", 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = ut.TestGetWorkspaceSize(&workspace_size, &executor);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

TEST_F(AclnnMegaMoeTest, ascend950_nullptr_context) {
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    
    vector<int64_t> x_dims = {128, 4096};
    vector<int64_t> y_dims = {128, 4096};
    vector<int64_t> expert_token_nums_dims = {4};

    auto x_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y_desc = TensorDesc(y_dims, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc(expert_token_nums_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT((aclTensor*)nullptr, x_desc, TensorDesc({128, 6}, ACL_INT32, ACL_FORMAT_ND),
                              TensorDesc({128, 6}, ACL_BF16, ACL_FORMAT_ND),
                              TensorDesc({4, 1024, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND),
                              TensorDesc({4, 4096, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND),
                              TensorDesc({4, 1024, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND),
                              TensorDesc({4, 4096, 8, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND),
                              nullptr, nullptr,
                              16, 4, 2097152, 0, 4, 24, 0, "", 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = ut.TestGetWorkspaceSize(&workspace_size, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMegaMoeTest, ascend950_nullptr_x) {
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    
    vector<int64_t> y_dims = {128, 4096};
    vector<int64_t> expert_token_nums_dims = {4};

    auto y_desc = TensorDesc(y_dims, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc(expert_token_nums_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(TensorDesc({32}, ACL_INT32, ACL_FORMAT_ND), (aclTensor*)nullptr,
                              TensorDesc({128, 6}, ACL_INT32, ACL_FORMAT_ND),
                              TensorDesc({128, 6}, ACL_BF16, ACL_FORMAT_ND),
                              TensorDesc({4, 1024, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND),
                              TensorDesc({4, 4096, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND),
                              TensorDesc({4, 1024, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND),
                              TensorDesc({4, 4096, 8, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND),
                              nullptr, nullptr,
                              16, 4, 2097152, 0, 4, 24, 0, "", 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = ut.TestGetWorkspaceSize(&workspace_size, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMegaMoeTest, ascend950_invalid_dtype_x) {
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    
    vector<int64_t> x_dims = {128, 4096};
    vector<int64_t> y_dims = {128, 4096};
    vector<int64_t> expert_token_nums_dims = {4};

    auto x_desc = TensorDesc(x_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y_desc = TensorDesc(y_dims, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc(expert_token_nums_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(TensorDesc({32}, ACL_INT32, ACL_FORMAT_ND), x_desc,
                              TensorDesc({128, 6}, ACL_INT32, ACL_FORMAT_ND),
                              TensorDesc({128, 6}, ACL_BF16, ACL_FORMAT_ND),
                              TensorDesc({4, 1024, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND),
                              TensorDesc({4, 4096, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND),
                              TensorDesc({4, 1024, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND),
                              TensorDesc({4, 4096, 8, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND),
                              nullptr, nullptr,
                              16, 4, 2097152, 0, 4, 24, 0, "", 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = ut.TestGetWorkspaceSize(&workspace_size, &executor);
    EXPECT_NE(ret, ACLNN_SUCCESS);
}

TEST_F(AclnnMegaMoeTest, ascend950_invalid_shape_h) {
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    
    vector<int64_t> x_dims = {128, 2048};
    vector<int64_t> y_dims = {128, 2048};
    vector<int64_t> expert_token_nums_dims = {4};

    auto x_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y_desc = TensorDesc(y_dims, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc(expert_token_nums_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(TensorDesc({32}, ACL_INT32, ACL_FORMAT_ND), x_desc,
                              TensorDesc({128, 6}, ACL_INT32, ACL_FORMAT_ND),
                              TensorDesc({128, 6}, ACL_BF16, ACL_FORMAT_ND),
                              TensorDesc({4, 1024, 2048}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND),
                              TensorDesc({4, 2048, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND),
                              TensorDesc({4, 1024, 32, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND),
                              TensorDesc({4, 2048, 8, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND),
                              nullptr, nullptr,
                              16, 4, 2097152, 0, 4, 24, 0, "", 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = ut.TestGetWorkspaceSize(&workspace_size, &executor);
    EXPECT_NE(ret, ACLNN_SUCCESS);
}

TEST_F(AclnnMegaMoeTest, ascend950_invalid_topk) {
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    
    vector<int64_t> x_dims = {128, 4096};
    vector<int64_t> topk_ids_dims = {128, 10};
    vector<int64_t> y_dims = {128, 4096};
    vector<int64_t> expert_token_nums_dims = {4};

    auto x_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_ids_desc = TensorDesc(topk_ids_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto y_desc = TensorDesc(y_dims, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc(expert_token_nums_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(TensorDesc({32}, ACL_INT32, ACL_FORMAT_ND), x_desc, topk_ids_desc,
                              TensorDesc({128, 10}, ACL_BF16, ACL_FORMAT_ND),
                              TensorDesc({4, 1024, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND),
                              TensorDesc({4, 4096, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND),
                              TensorDesc({4, 1024, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND),
                              TensorDesc({4, 4096, 8, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND),
                              nullptr, nullptr,
                              16, 4, 2097152, 0, 4, 24, 0, "", 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = ut.TestGetWorkspaceSize(&workspace_size, &executor);
    EXPECT_NE(ret, ACLNN_SUCCESS);
}

TEST_F(AclnnMegaMoeTest, ascend950_weight_dim_mismatch) {
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    
    vector<int64_t> x_dims = {128, 4096};
    vector<int64_t> weight1_dims = {4, 1024, 4096};
    vector<int64_t> weight2_dims = {4, 4096, 1024};
    vector<int64_t> y_dims = {128, 4096};
    vector<int64_t> expert_token_nums_dims = {4};

    auto x_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weight1_desc = TensorDesc(weight1_dims, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weight2_desc = TensorDesc(weight2_dims, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y_desc = TensorDesc(y_dims, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc(expert_token_nums_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(TensorDesc({32}, ACL_INT32, ACL_FORMAT_ND), x_desc,
                              TensorDesc({128, 6}, ACL_INT32, ACL_FORMAT_ND),
                              TensorDesc({128, 6}, ACL_BF16, ACL_FORMAT_ND),
                              weight1_desc, weight2_desc,
                              TensorDesc({4, 1024, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND),
                              TensorDesc({4, 4096, 16, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND),
                              nullptr, nullptr,
                              16, 4, 2097152, 0, 4, 24, 0, "", 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = ut.TestGetWorkspaceSize(&workspace_size, &executor);
    EXPECT_NE(ret, ACLNN_SUCCESS);
}

TEST_F(AclnnMegaMoeTest, ascend950_invalid_ep_world_size) {
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    
    vector<int64_t> x_dims = {128, 4096};
    vector<int64_t> weight1_dims = {4, 1024, 4096};
    vector<int64_t> y_dims = {128, 4096};
    vector<int64_t> expert_token_nums_dims = {4};

    auto x_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weight1_desc = TensorDesc(weight1_dims, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y_desc = TensorDesc(y_dims, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc(expert_token_nums_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(TensorDesc({32}, ACL_INT32, ACL_FORMAT_ND), x_desc,
                              TensorDesc({128, 6}, ACL_INT32, ACL_FORMAT_ND),
                              TensorDesc({128, 6}, ACL_BF16, ACL_FORMAT_ND),
                              weight1_desc, TensorDesc({4, 4096, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND),
                              TensorDesc({4, 1024, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND),
                              TensorDesc({4, 4096, 8, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND),
                              nullptr, nullptr,
                              16, 1, 2097152, 0, 4, 24, 0, "", 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = ut.TestGetWorkspaceSize(&workspace_size, &executor);
    EXPECT_NE(ret, ACLNN_SUCCESS);
}

TEST_F(AclnnMegaMoeTest, ascend950_invalid_quant_mode) {
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    
    vector<int64_t> x_dims = {128, 4096};
    vector<int64_t> weight1_dims = {4, 1024, 4096};
    vector<int64_t> y_dims = {128, 4096};
    vector<int64_t> expert_token_nums_dims = {4};

    auto x_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weight1_desc = TensorDesc(weight1_dims, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y_desc = TensorDesc(y_dims, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc(expert_token_nums_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(TensorDesc({32}, ACL_INT32, ACL_FORMAT_ND), x_desc,
                              TensorDesc({128, 6}, ACL_INT32, ACL_FORMAT_ND),
                              TensorDesc({128, 6}, ACL_BF16, ACL_FORMAT_ND),
                              weight1_desc, TensorDesc({4, 4096, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND),
                              TensorDesc({4, 1024, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND),
                              TensorDesc({4, 4096, 8, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND),
                              nullptr, nullptr,
                              16, 4, 2097152, 0, 0, 24, 0, "", 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = ut.TestGetWorkspaceSize(&workspace_size, &executor);
    EXPECT_NE(ret, ACLNN_SUCCESS);
}