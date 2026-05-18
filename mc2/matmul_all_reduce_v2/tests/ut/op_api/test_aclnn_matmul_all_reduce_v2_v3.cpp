/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "matmul_all_reduce_v2_api_ut_param.h"
#include "op_api_ut_common/op_api_ut.h"
#include "../../../op_api/aclnn_matmul_all_reduce_v2_v3.h"

namespace MatmulAllReduceV2UT {

class AclnnMatmulAllReduceV2V3Test : public testing::TestWithParam<MatmulAllReduceV2ApiUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduceV2 AclnnMatmulAllReduceV2V3Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduceV2 AclnnMatmulAllReduceV2V3Test TearDown" << std::endl;
    }
};

TEST_P(AclnnMatmulAllReduceV2V3Test, param)
{
    auto param = GetParam();
    op::SetPlatformSocVersion(param.soc);
    auto ut = OP_API_UT(
        aclnnMatmulAllReduceV2V3,
        INPUT(param.x1, param.x2, param.bias, param.x3, param.group.c_str(), param.reduceOp.c_str(),
              param.commMode.c_str(), param.commTurn, param.streamMode),
        OUTPUT(param.output)
    );
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    if (param.expectResult == ACLNN_SUCCESS) {
        EXPECT_NE(ACLNN_ERR_PARAM_INVALID, aclnnRet);
        if (aclnnRet == ACLNN_SUCCESS) {
            aclnnMatmulAllReduceV2V3(nullptr, workspace_size, executor, nullptr);
        }
    } else {
        EXPECT_EQ(param.expectResult, aclnnRet);
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAllReduceV2,
    AclnnMatmulAllReduceV2V3Test,
    testing::ValuesIn(GetCasesFromCsv<MatmulAllReduceV2ApiUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MatmulAllReduceV2ApiUtParam>
);

} // namespace MatmulAllReduceV2UT