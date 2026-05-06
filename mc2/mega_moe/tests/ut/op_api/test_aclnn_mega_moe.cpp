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
#include "opdev/platform.h"
#include "aclnn_mega_moe.h"

using namespace op;
using namespace std;

class AclnnMegaMoeTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        std::cout << "MegaMoe AclnnMegaMoeTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        std::cout << "MegaMoe AclnnMegaMoeTest TearDown" << std::endl;
    }
};

TEST_F(AclnnMegaMoeTest, ascend950_bf16_basic) {
    auto context_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_ids_desc = TensorDesc({128, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto topk_weights_desc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    
    vector<TensorDesc> weight1_descs;
    for (int i = 0; i < 4; i++) {
        weight1_descs.push_back(TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    }
    
    vector<TensorDesc> weight2_descs;
    for (int i = 0; i < 4; i++) {
        weight2_descs.push_back(TensorDesc({4096, 1024}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    }
    
    vector<TensorDesc> weight_scales1_descs;
    for (int i = 0; i < 4; i++) {
        weight_scales1_descs.push_back(TensorDesc({2048, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    }
    
    vector<TensorDesc> weight_scales2_descs;
    for (int i = 0; i < 4; i++) {
        weight_scales2_descs.push_back(TensorDesc({4096, 16, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    }
    
    auto y_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(context_desc, x_desc, topk_ids_desc, topk_weights_desc,
                              weight1_descs, weight2_descs, weight_scales1_descs, weight_scales2_descs,
                              nullptr, nullptr,
                              16, 4, 2097152, 0, 4, 24, 0, "", 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AclnnMegaMoeTest, ascend950_fp8_e5m2) {
    auto context_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x_desc = TensorDesc({256, 5120}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_ids_desc = TensorDesc({256, 6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);
    auto topk_weights_desc = TensorDesc({256, 6}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    
    vector<TensorDesc> weight1_descs;
    for (int i = 0; i < 8; i++) {
        weight1_descs.push_back(TensorDesc({3072, 5120}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1));
    }
    
    vector<TensorDesc> weight2_descs;
    for (int i = 0; i < 8; i++) {
        weight2_descs.push_back(TensorDesc({5120, 1536}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1));
    }
    
    vector<TensorDesc> weight_scales1_descs;
    for (int i = 0; i < 8; i++) {
        weight_scales1_descs.push_back(TensorDesc({3072, 80, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    }
    
    vector<TensorDesc> weight_scales2_descs;
    for (int i = 0; i < 8; i++) {
        weight_scales2_descs.push_back(TensorDesc({5120, 24, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    }
    
    auto y_desc = TensorDesc({256, 5120}, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(context_desc, x_desc, topk_ids_desc, topk_weights_desc,
                              weight1_descs, weight2_descs, weight_scales1_descs, weight_scales2_descs,
                              nullptr, nullptr,
                              64, 8, 2097152, 0, 4, 23, 0, "", 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AclnnMegaMoeTest, ascend950_invalid_ep_world_size) {
    auto context_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_ids_desc = TensorDesc({128, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto topk_weights_desc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    
    vector<TensorDesc> weight1_descs;
    for (int i = 0; i < 4; i++) {
        weight1_descs.push_back(TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    }
    
    vector<TensorDesc> weight2_descs;
    for (int i = 0; i < 4; i++) {
        weight2_descs.push_back(TensorDesc({4096, 1024}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    }
    
    vector<TensorDesc> weight_scales1_descs;
    for (int i = 0; i < 4; i++) {
        weight_scales1_descs.push_back(TensorDesc({2048, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    }
    
    vector<TensorDesc> weight_scales2_descs;
    for (int i = 0; i < 4; i++) {
        weight_scales2_descs.push_back(TensorDesc({4096, 16, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    }
    
    auto y_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(context_desc, x_desc, topk_ids_desc, topk_weights_desc,
                              weight1_descs, weight2_descs, weight_scales1_descs, weight_scales2_descs,
                              nullptr, nullptr,
                              16, 1, 2097152, 0, 4, 24, 0, "", 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

TEST_F(AclnnMegaMoeTest, ascend950_invalid_moe_expert_num) {
    auto context_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_ids_desc = TensorDesc({128, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto topk_weights_desc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    
    vector<TensorDesc> weight1_descs;
    for (int i = 0; i < 4; i++) {
        weight1_descs.push_back(TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    }
    
    vector<TensorDesc> weight2_descs;
    for (int i = 0; i < 4; i++) {
        weight2_descs.push_back(TensorDesc({4096, 1024}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    }
    
    vector<TensorDesc> weight_scales1_descs;
    for (int i = 0; i < 4; i++) {
        weight_scales1_descs.push_back(TensorDesc({2048, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    }
    
    vector<TensorDesc> weight_scales2_descs;
    for (int i = 0; i < 4; i++) {
        weight_scales2_descs.push_back(TensorDesc({4096, 16, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    }
    
    auto y_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(context_desc, x_desc, topk_ids_desc, topk_weights_desc,
                              weight1_descs, weight2_descs, weight_scales1_descs, weight_scales2_descs,
                              nullptr, nullptr,
                              15, 4, 2097152, 0, 4, 24, 0, "", 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}