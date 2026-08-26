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
 * \file test_quant_lightning_indexer_v2_tiling.cpp
 * \brief Tests Quant Lightning Indexer V2 tiling validation on A2 and A3 platforms.
 */

#include <gtest/gtest.h>

#include <limits>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
struct QLIV2CompileInfo {};

void RunArch22TilingCase(int64_t topk, int64_t quantMode, int64_t maskMode, int64_t cmpRatio, int64_t returnValue,
                         ge::graphStatus expected)
{
    QLIV2CompileInfo compileInfo;
    int32_t sequsedK[] = {16, 16};
    gert::TilingContextPara tilingContextPara(
        "QuantLightningIndexerV2",
        {{{{2, 4, 64, 128}, {2, 4, 64, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{4, 16, 1, 128}, {4, 16, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2, 4, 64}, {2, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 4, 64}, {2, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 16, 1}, {4, 16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, sequsedK},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{2, 4, 1, topk}, {2, 4, 1, topk}}, ge::DT_INT32, ge::FORMAT_ND}, {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(topk)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maskMode)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(cmpRatio)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(returnValue)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    if (expected == ge::GRAPH_SUCCESS) {
        ExecuteTestCase(tilingContextPara, expected, std::numeric_limits<uint64_t>::max());
    } else {
        ExecuteTestCase(tilingContextPara, expected);
    }
}
} // namespace

TEST(QuantLightningIndexerV2Arch22Tiling, AcceptsSupportedBaseline)
{
    RunArch22TilingCase(2048, 2, 0, 1, 0, ge::GRAPH_SUCCESS);
}

TEST(QuantLightningIndexerV2Arch22Tiling, RejectsUnsupportedQuantModes)
{
    for (const int64_t quantMode : {1, 3, 4, 5}) {
        RunArch22TilingCase(2048, quantMode, 0, 1, 0, ge::GRAPH_FAILED);
    }
}

TEST(QuantLightningIndexerV2Arch22Tiling, RejectsTopkOutsideA2A3Range)
{
    RunArch22TilingCase(0, 2, 0, 1, 0, ge::GRAPH_FAILED);
    RunArch22TilingCase(2049, 2, 0, 1, 0, ge::GRAPH_FAILED);
}

TEST(QuantLightningIndexerV2Arch22Tiling, RejectsA2A3OnlyAttributeViolations)
{
    RunArch22TilingCase(2048, 2, 0, 3, 0, ge::GRAPH_FAILED);
    RunArch22TilingCase(2048, 2, 0, 1, 1, ge::GRAPH_FAILED);
}
