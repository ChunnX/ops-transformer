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
#include <iostream>
#include "tiling_case_executor.h"
#include "nsai_tiling_ut_common.h"

using namespace nsai_tiling_ut;

class NsaSelectedAttentionInferTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NsaSelectedAttentionInferTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NsaSelectedAttentionInferTiling TearDown" << std::endl;
    }
};

namespace {

void RunTilingLegacyNegativeCase(const CaseParams& params, bool expectGraphFailure)
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

TEST_F(NsaSelectedAttentionInferTiling, NsaSelectAttentionInfer_Case_q3_tnd)
{
    RunTilingLegacyNegativeCase(LegacyCaseQ3Tnd(), true);
}

TEST_F(NsaSelectedAttentionInferTiling, NsaSelectAttentionInfer_Case_q1_bsnd_fp16)
{
    RunTilingLegacyNegativeCase(LegacyCaseQ1Bsnd(), false);
}

TEST_F(NsaSelectedAttentionInferTiling, NsaSelectAttentionInfer_Case_q1_bsh_fp16)
{
    RunTilingLegacyNegativeCase(LegacyCaseQ1Batch1(), false);
}

TEST_F(NsaSelectedAttentionInferTiling, NsaSelectAttentionInfer_Case_q1_tnd_fp16)
{
    RunTilingLegacyNegativeCase(LegacyCaseQ1Batch1(), false);
}
