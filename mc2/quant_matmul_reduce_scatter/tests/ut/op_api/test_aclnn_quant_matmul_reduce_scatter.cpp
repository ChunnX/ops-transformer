/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "test_quant_matmul_reduce_scatter_api_ut_param.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

extern "C" aclnnStatus aclnnQuantMatmulReduceScatterGetWorkspaceSize(
    const aclTensor *context, const aclTensor *x1, const aclTensor *x2, const aclTensor *x1ScaleOptional,
    const aclTensor *x2ScaleOptional, const char *group, int64_t rankSize, int64_t hcclBufferSize,
    int64_t groupSize, const char *commMode, const aclTensor *output, uint64_t *workspaceSize,
    aclOpExecutor **executor);
extern "C" aclnnStatus aclnnQuantMatmulReduceScatter(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                     aclrtStream stream);

namespace QuantMatmulReduceScatterUT {

class TestAclnnQuantMatmulReduceScatter : public testing::TestWithParam<QuantMatmulReduceScatterApiUtParam> {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(NpuArch::DAV_3510);
        cout << "TestAclnnQuantMatmulReduceScatter SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "TestAclnnQuantMatmulReduceScatter TearDown" << endl;
    }
};

TEST_P(TestAclnnQuantMatmulReduceScatter, param)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

    aclTensor *context = param.context.GetViewDims().empty() ? nullptr : param.context.ToAclTypeRawPtr();
    aclTensor *x1 = param.x1.GetViewDims().empty() ? nullptr : param.x1.ToAclTypeRawPtr();
    aclTensor *x2 = param.x2.GetViewDims().empty() ? nullptr : param.x2.ToAclTypeRawPtr();
    aclTensor *x1Scale = param.x1Scale.GetViewDims().empty() ? nullptr : param.x1Scale.ToAclTypeRawPtr();
    aclTensor *x2Scale = param.x2Scale.GetViewDims().empty() ? nullptr : param.x2Scale.ToAclTypeRawPtr();
    aclTensor *output = param.output.GetViewDims().empty() ? nullptr : param.output.ToAclTypeRawPtr();

    auto ut = OP_API_UT(aclnnQuantMatmulReduceScatter,
                        INPUT(context, x1, x2, x1Scale, x2Scale, param.group.c_str(), param.rankSize,
                              param.hcclBufferSize, param.groupSize, param.commMode.c_str()),
                        OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    if (param.expectResult == ACLNN_SUCCESS) {
        EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, param.expectResult);
    }
}

TEST_F(TestAclnnQuantMatmulReduceScatter, SecondPhaseNullExecutorTest)
{
    aclnnStatus aclRet = aclnnQuantMatmulReduceScatter(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

TEST_F(TestAclnnQuantMatmulReduceScatter, NullGroupTest)
{
    TensorDesc contextDesc = TensorDesc({100}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc x1Desc = TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({2560, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1ScaleDesc = TensorDesc({2048, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2ScaleDesc = TensorDesc({2560, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc outputDesc = TensorDesc({512, 2560}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulReduceScatter,
                        INPUT(contextDesc, x1Desc, x2Desc, x1ScaleDesc, x2ScaleDesc, (const char *)nullptr, 4, 200,
                              4295032864L, "aiv_urma"),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(TestAclnnQuantMatmulReduceScatter, EmptyGroupTest)
{
    TensorDesc contextDesc = TensorDesc({100}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc x1Desc = TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({2560, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1ScaleDesc = TensorDesc({2048, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2ScaleDesc = TensorDesc({2560, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc outputDesc = TensorDesc({512, 2560}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnQuantMatmulReduceScatter,
        INPUT(contextDesc, x1Desc, x2Desc, x1ScaleDesc, x2ScaleDesc, "", 4, 200, 4295032864L, "aiv_urma"),
        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

INSTANTIATE_TEST_SUITE_P(
    QuantMatmulReduceScatter, TestAclnnQuantMatmulReduceScatter,
    testing::ValuesIn(GetCasesFromCsv<QuantMatmulReduceScatterApiUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<QuantMatmulReduceScatterApiUtParam>);

} // namespace QuantMatmulReduceScatterUT
