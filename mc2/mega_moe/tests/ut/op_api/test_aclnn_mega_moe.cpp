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
    vector<int64_t> x_dims = {128, 512};
    vector<int64_t> expert_ids_dims = {128, 2};
    vector<int64_t> weight1_dims = {4, 512, 256};
    vector<int64_t> weightscales1_dims = {4, 256, 8};
    vector<int64_t> weight2_dims = {4, 128, 512};
    vector<int64_t> weightscales2_dims = {4, 512, 2};
    vector<int64_t> y_dims = {128, 512};

    auto x_desc = TensorDesc(x_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expert_ids_desc = TensorDesc(expert_ids_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto weight1_desc = TensorDesc(weight1_dims, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightscales1_desc = TensorDesc(weightscales1_dims, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2);
    auto weight2_desc = TensorDesc(weight2_dims, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightscales2_desc = TensorDesc(weightscales2_dims, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2);
    auto y_desc = TensorDesc(y_dims, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(x_desc, expert_ids_desc, nullptr, nullptr, nullptr,
                              weight1_desc, weightscales1_desc, weight2_desc, weightscales2_desc,
                              "ep_group", 1, 0, 4, 256, 1, 0, 0, ACL_DT_UNDEFINED, 0, "", 128),
                        OUTPUT(y_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnMegaMoeTest, ascend950_fp16_basic) {
    vector<int64_t> x_dims = {128, 512};
    vector<int64_t> expert_ids_dims = {128, 2};
    vector<int64_t> weight1_dims = {4, 512, 256};
    vector<int64_t> weightscales1_dims = {4, 256, 8};
    vector<int64_t> weight2_dims = {4, 128, 512};
    vector<int64_t> weightscales2_dims = {4, 512, 2};

    auto x_desc = TensorDesc(x_dims, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expert_ids_desc = TensorDesc(expert_ids_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto weight1_desc = TensorDesc(weight1_dims, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightscales1_desc = TensorDesc(weightscales1_dims, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2);
    auto weight2_desc = TensorDesc(weight2_dims, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightscales2_desc = TensorDesc(weightscales2_dims, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2);
    auto y_desc = TensorDesc(x_dims, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(x_desc, expert_ids_desc, nullptr, nullptr, nullptr,
                              weight1_desc, weightscales1_desc, weight2_desc, weightscales2_desc,
                              "ep_group", 1, 0, 4, 256, 1, 0, 0, ACL_DT_UNDEFINED, 0, "", 128),
                        OUTPUT(y_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}