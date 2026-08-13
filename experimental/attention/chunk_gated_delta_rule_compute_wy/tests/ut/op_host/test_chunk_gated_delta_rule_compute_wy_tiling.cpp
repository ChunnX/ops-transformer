/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "../../../op_host/chunk_gated_delta_rule_compute_wy_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
gert::TilingContextPara CreateTilingContext(int64_t batch, int64_t tokens, int64_t qHeads, int64_t vHeads,
                                            int64_t kDim, int64_t vDim, int64_t chunkSize)
{
    static optiling::ChunkGatedDeltaRuleComputeWyCompileInfo compileInfo = {};
    return gert::TilingContextPara(
        "ChunkGatedDeltaRuleComputeWy",
        {
            {{{batch, tokens, qHeads, kDim}, {batch, tokens, qHeads, kDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch, tokens, qHeads, kDim}, {batch, tokens, qHeads, kDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch, tokens, vHeads, vDim}, {batch, tokens, vHeads, vDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch, tokens, vHeads}, {batch, tokens, vHeads}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{batch, tokens, vHeads}, {batch, tokens, vHeads}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{batch, qHeads, tokens, kDim}, {batch, qHeads, tokens, kDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch, qHeads, tokens, kDim}, {batch, qHeads, tokens, kDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch, vHeads, tokens, kDim}, {batch, vHeads, tokens, kDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch, vHeads, tokens, vDim}, {batch, vHeads, tokens, vDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch, vHeads, tokens}, {batch, vHeads, tokens}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"chunk_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(chunkSize)}}, &compileInfo);
}
} // namespace

TEST(ChunkGatedDeltaRuleComputeWyTiling, SupportsGroupedQueryAttention)
{
    auto context = CreateTilingContext(/*batch=*/2, /*tokens=*/128, /*qHeads=*/4, /*vHeads=*/8,
                                       /*kDim=*/64, /*vDim=*/128, /*chunkSize=*/64);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS);
}

TEST(ChunkGatedDeltaRuleComputeWyTiling, RejectsUnsupportedChunkSize)
{
    auto context = CreateTilingContext(1, 128, 2, 4, 64, 64, 32);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(ChunkGatedDeltaRuleComputeWyTiling, RejectsUnalignedSequenceLength)
{
    auto context = CreateTilingContext(1, 96, 2, 4, 64, 64, 64);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(ChunkGatedDeltaRuleComputeWyTiling, RejectsInvalidHeadGrouping)
{
    auto context = CreateTilingContext(1, 64, 3, 4, 64, 64, 64);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(ChunkGatedDeltaRuleComputeWyTiling, RejectsUnsupportedDimensions)
{
    auto context = CreateTilingContext(1, 64, 2, 4, 144, 64, 64);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}
