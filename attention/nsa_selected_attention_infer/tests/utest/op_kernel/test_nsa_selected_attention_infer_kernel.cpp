/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "tiling_case_executor.h"
#include "../op_host/nsai_tiling_ut_common.h"

using namespace nsai_tiling_ut;

class NsaSelectedAttentionInferKernel : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NsaSelectedAttentionInferKernel SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NsaSelectedAttentionInferKernel TearDown" << std::endl;
    }
};

namespace {

void RunNegativeKernelCase(const CaseParams& params, bool expectGraphFailure)
{
    CaseShapes shapes;
    auto para = BuildTilingContextPara(params, shapes);
    TilingInfo tilingInfo;
    if (expectGraphFailure) {
        ASSERT_FALSE(ExecuteTiling(para, tilingInfo));
        return;
    }
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
}

}  // namespace

TEST_F(NsaSelectedAttentionInferKernel, NsaSelectAttentionInfer_Case_1)
{
    RunNegativeKernelCase(LegacyCaseQ3Tnd(), true);
}

TEST_F(NsaSelectedAttentionInferKernel, NsaSelectAttentionInfer_Case_2)
{
    RunNegativeKernelCase(LegacyCaseQ1Bsnd(), false);
}

TEST_F(NsaSelectedAttentionInferKernel, NsaSelectAttentionInfer_Case_3)
{
    RunNegativeKernelCase(LegacyCaseQ1Batch1(), false);
}

TEST_F(NsaSelectedAttentionInferKernel, NsaSelectAttentionInfer_Case_4)
{
    RunNegativeKernelCase(LegacyCaseQ1Batch1(), false);
}
