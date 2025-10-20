/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/apply_rotary_pos_emb_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class ApplyRotaryPosEmbTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ApplyRotaryPosEmbTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ApplyRotaryPosEmbTiling TearDown" << std::endl;
    }
};

TEST_F(ApplyRotaryPosEmbTiling, test_tiling_fp16_001)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "24 128 64 0 0 0 0 0 0 0 3072 3072 256 256 0 0 0 0 0 1408 128 128 1408 128 128 12 1536 1 8 4 0 0 128 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, test_tiling_bf16_001)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "24 128 64 0 0 0 0 0 0 0 6144 6144 256 512 0 0 0 0 0 1408 128 128 1408 128 128 12 1536 2 16 8 0 0 64 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, test_tiling_fp32_001)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "24 128 64 0 0 0 0 0 0 0 6144 6144 512 512 0 0 0 0 0 1408 128 128 1408 128 128 12 1536 2 16 8 0 0 64 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}