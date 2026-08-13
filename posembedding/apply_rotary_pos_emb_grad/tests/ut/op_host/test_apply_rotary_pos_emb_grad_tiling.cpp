/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// BAB 模板 Tiling UT — golden-match 格式 (参考 rotary_position_embedding_grad)
//
// 检查: tiling 成功 + tilingKey + tilingData 严格比对
// 跳过: workspace (框架跨 case workspace 指针污染, 值不可靠)
//
// 平台: Ascend950, coreNum=48, ubSize=253952

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/apply_rotary_pos_emb_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class ApplyRotaryPosEmbGradBabTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ApplyRotaryPosEmbGradBabTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ApplyRotaryPosEmbGradBabTiling TearDown" << std::endl; }
};

// dCosFlag=0 (4 inputs, 2 outputs) — tilingKey=1
TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_fp16_bsnd_dcos0)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo, "Ascend950", 48, 253952);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 64 128 4 4 2 1 22 3 3 1 4 4 44 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {});
}

TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_fp32_bsnd_dcos0)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo, "Ascend950", 48, 253952);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 64 128 4 4 2 1 22 3 3 1 4 4 44 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {});
}

// dCosFlag=1 (6 inputs, 4 outputs) — tilingKey=17179869185
TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_fp16_bsnd_dcos1)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo, "Ascend950", 48, 253952);
    uint64_t expectTilingKey = 17179869185;
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 64 128 4 4 2 1 22 3 3 1 4 4 44 "
        "0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {});
}

TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_bf16_bsnd_dcos1)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo, "Ascend950", 48, 253952);
    uint64_t expectTilingKey = 17179869185;
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 64 128 4 4 2 1 22 3 3 1 4 4 44 "
        "0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {});
}

// nQ != nK — maxN=8
TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_fp16_bsnd_nq_neq_nk)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
        {
            {{{2, 64, 8, 128}, {2, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 8, 128}, {2, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 8, 128}, {2, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo, "Ascend950", 48, 253952);
    uint64_t expectTilingKey = 17179869185;
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 64 128 8 4 2 1 22 3 3 1 8 8 44 "
        "0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {});
}

// TND (3D) — B=1
TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_fp16_tnd_dcos1)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
        {
            {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
        },
        &compileInfo, "Ascend950", 48, 253952);
    uint64_t expectTilingKey = 17179869185;
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 64 128 4 4 1 1 32 2 2 1 4 4 32 "
        "0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {});
}

// D 非对齐 — D=48
TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_fp16_bsnd_d48_unaligned)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
        {
            {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 48}, {1, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 48}, {1, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 48}, {1, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 48}, {1, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo, "Ascend950", 48, 253952);
    uint64_t expectTilingKey = 17179869185;
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 64 48 4 4 2 1 22 3 3 1 4 4 44 "
        "0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {});
}

TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_fp16_bsnd_d48_unaligned_dcos0)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
        {
            {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 48}, {1, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 48}, {1, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo, "Ascend950", 48, 253952);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 64 48 4 4 2 1 22 3 3 1 4 4 44 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {});
}
