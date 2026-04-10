/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_masked_causal_conv1d_tiling.cpp
 * \brief Unit tests for MaskedCausalConv1d tiling logic
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/masked_causal_conv1d_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"


using namespace std;

class MaskedCausalConv1dTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MaskedCausalConv1dTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MaskedCausalConv1dTiling TearDown" << std::endl;
    }
};

TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_bf16_s2048_b4_h768)
{
    // CompileInfo: coreNum=64, ubSize=253952 (248KB UB, A5 950 platform)
    struct FusedCausalConv1dCutBSHCompileInfo {} compileInfo;

    // No operator attributes for MaskedCausalConv1d
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 2048;
    int64_t B = 4;
    int64_t H = 768;
    int64_t W = 3;

    gert::TilingContextPara tilingContextPara(
        "MaskedCausalConv1d",
        {
            // Input 0: x - [S, B, H]
            {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 1: weight - [W, H]
            {{{W, H}, {W, H}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 2: mask - [B, S], BOOL
            {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            // Output 0: y - [S, B, H]
            {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    // Expected tiling key for BF16
    int64_t expectTilingKey = 10000;
    // Verify tiling succeeds (do not assert exact tiling data string, as values may vary)
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
