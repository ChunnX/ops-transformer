/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include <string>

#include "../../../op_host/kv_cache_turbo_quant_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class KvCacheTurboQuantTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "--- KvCacheTurboQuant Tiling UT SetUp ---" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "--- KvCacheTurboQuant Tiling UT TearDown ---" << std::endl;
    }
};

// GQA decode batch=1: [1, 8, 128], mse_bits=3
TEST_F(KvCacheTurboQuantTilingTest, tiling_gqa_decode_b1)
{
    struct KvCacheTurboQuantCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara("KvCacheTurboQuant",
                                              {{{{1, 8, 128}, {1, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{{{1, 8, 48}, {1, 8, 48}}, ge::DT_UINT8, ge::FORMAT_ND},
                                               {{{1, 8, 16}, {1, 8, 16}}, ge::DT_UINT8, ge::FORMAT_ND},
                                               {{{1, 8}, {1, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{1, 8}, {1, 8}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"mse_bits", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}},
                                              &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0UL);
}

// GQA decode batch=64: [64, 8, 128], mse_bits=3
TEST_F(KvCacheTurboQuantTilingTest, tiling_gqa_decode_b64)
{
    struct KvCacheTurboQuantCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara("KvCacheTurboQuant",
                                              {{{{64, 8, 128}, {64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{{{64, 8, 48}, {64, 8, 48}}, ge::DT_UINT8, ge::FORMAT_ND},
                                               {{{64, 8, 16}, {64, 8, 16}}, ge::DT_UINT8, ge::FORMAT_ND},
                                               {{{64, 8}, {64, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{64, 8}, {64, 8}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"mse_bits", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}},
                                              &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0UL);
}

// prefill: [2048, 8, 128], mse_bits=2
TEST_F(KvCacheTurboQuantTilingTest, tiling_gqa_mse2_t2048)
{
    struct KvCacheTurboQuantCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara("KvCacheTurboQuant",
                                              {{{{2048, 8, 128}, {2048, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{{{2048, 8, 32}, {2048, 8, 32}}, ge::DT_UINT8, ge::FORMAT_ND},
                                               {{{2048, 8, 16}, {2048, 8, 16}}, ge::DT_UINT8, ge::FORMAT_ND},
                                               {{{2048, 8}, {2048, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{2048, 8}, {2048, 8}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"mse_bits", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}},
                                              &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0UL);
}

// head_dim != 128 -> tiling fail
TEST_F(KvCacheTurboQuantTilingTest, tiling_invalid_head_dim_fail)
{
    struct KvCacheTurboQuantCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara("KvCacheTurboQuant",
                                              {{{{1, 8, 64}, {1, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{{{1, 8, 24}, {1, 8, 24}}, ge::DT_UINT8, ge::FORMAT_ND},
                                               {{{1, 8, 16}, {1, 8, 16}}, ge::DT_UINT8, ge::FORMAT_ND},
                                               {{{1, 8}, {1, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{1, 8}, {1, 8}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"mse_bits", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}},
                                              &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0UL);
}

// mse_bits = 5 -> tiling fail
TEST_F(KvCacheTurboQuantTilingTest, tiling_invalid_mse_bits_fail)
{
    struct KvCacheTurboQuantCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara("KvCacheTurboQuant",
                                              {{{{1, 8, 128}, {1, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{{{1, 8, 80}, {1, 8, 80}}, ge::DT_UINT8, ge::FORMAT_ND},
                                               {{{1, 8, 16}, {1, 8, 16}}, ge::DT_UINT8, ge::FORMAT_ND},
                                               {{{1, 8}, {1, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{1, 8}, {1, 8}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"mse_bits", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)}},
                                              &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0UL);
}
