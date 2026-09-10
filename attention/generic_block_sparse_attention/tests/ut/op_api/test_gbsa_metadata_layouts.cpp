/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"
#include "attention/generic_block_sparse_attention_metadata/op_host/generic_block_sparse_attention_metadata_check.h"

namespace {
class GbsaMetadataLayoutTest : public testing::Test {
protected:
    void SetUp() override
    {
        int64_t cuShape[] = {3}, usedShape[] = {2}, stride[] = {1};
        cu = aclCreateTensor(cuShape, 1, ACL_INT64, stride, 0, ACL_FORMAT_ND, cuShape, 1, nullptr);
        used = aclCreateTensor(usedShape, 1, ACL_INT32, stride, 0, ACL_FORMAT_ND, usedShape, 1, nullptr);
        ASSERT_NE(cu, nullptr);
        ASSERT_NE(used, nullptr);
    }
    void TearDown() override
    {
        aclDestroyTensor(cu);
        aclDestroyTensor(used);
    }
    aclTensor *cu = nullptr;
    aclTensor *used = nullptr;
};

TEST_F(GbsaMetadataLayoutTest, AcceptsPagedAndTnd)
{
    EXPECT_EQ(CheckGbsaLayouts("TND", "PA_BBND"), ACLNN_SUCCESS);
    EXPECT_EQ(CheckGbsaLayouts("TND", "TND"), ACLNN_SUCCESS);
    EXPECT_EQ(CheckGbsaLayouts("TND", "PA_BNBD"), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(CheckGbsaLayouts("BNSD", "TND"), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(GbsaMetadataLayoutTest, PaAndTndLengthContracts)
{
    EXPECT_EQ(CheckGbsaSeqLengthInputs(cu, nullptr, nullptr, used, 2, "TND", "PA_BBND"), ACLNN_SUCCESS);
    EXPECT_EQ(CheckGbsaSeqLengthInputs(cu, cu, nullptr, nullptr, 2, "TND", "TND"), ACLNN_SUCCESS);
    EXPECT_EQ(CheckGbsaSeqLengthInputs(cu, cu, used, used, 2, "TND", "TND"), ACLNN_SUCCESS);
    EXPECT_EQ(CheckGbsaSeqLengthInputs(cu, nullptr, nullptr, used, 2, "TND", "TND"), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(CheckGbsaSeqLengthInputs(cu, cu, nullptr, used, 2, "TND", "PA_BBND"), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(CheckGbsaSeqLengthInputs(cu, nullptr, nullptr, nullptr, 2, "TND", "PA_BBND"), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(GbsaMetadataLayoutTest, RejectsWrongLengthShapeAndDtype)
{
    EXPECT_EQ(CheckGbsaSeqLengthInputs(cu, cu, nullptr, used, 1, "TND", "TND"), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(CheckGbsaSeqLengthInputs(cu, used, nullptr, used, 2, "TND", "TND"), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(CheckGbsaSeqLengthInputs(cu, cu, nullptr, cu, 2, "TND", "TND"), ACLNN_ERR_PARAM_INVALID);
}
} // namespace
