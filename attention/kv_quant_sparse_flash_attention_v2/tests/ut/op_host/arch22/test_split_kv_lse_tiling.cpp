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
#include <algorithm>
#include <cstring>
#include <tuple>
#include "tiling_case_executor.h"
#include "../../../../../kv_quant_sparse_flash_attention/op_host/kv_quant_sparse_flash_attention_tiling.h"

TEST(KvQuantSparseFlashAttentionV2SplitKv, Arch22LseAndFallback)
{
    struct CompileInfo {} compileInfo;
    for (int64_t heads : {8, 64}) {
        for (bool returnLse : {false, true}) {
            for (const auto &[queries, batches, topK, expectedBlocks] :
                 {std::tuple<int64_t, int64_t, int64_t, uint32_t>{1, 1, 2048, 4}, {4, 1, 2048, 16},
                  {1, 1, 1024, 2}, {4, 1, 1536, 8}, {1, 1, 512, 24}, {16, 1, 2048, 24},
                  {8, 4, 1024, 16}, {8, 4, 2048, 16}}) {
                int32_t qLengths[] = {1, 2, 3, static_cast<int32_t>(queries)};
                qLengths[batches - 1] = queries;
                int32_t kvLengths[] = {4096, 4096, 4096, 4096};
                gert::TilingContextPara para(
                    "KvQuantSparseFlashAttentionV2",
                    {{{{queries, heads, 576}, {queries, heads, 576}}, ge::DT_BF16, ge::FORMAT_ND},
                     {{{batches * 32, 128, 1, 656}, {batches * 32, 128, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
                     {{{batches * 32, 128, 1, 656}, {batches * 32, 128, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
                     {{{queries, 1, topK}, {queries, 1, topK}}, ge::DT_INT32, ge::FORMAT_ND},
                     {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                     {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                     {{{batches, 32}, {batches, 32}}, ge::DT_INT32, ge::FORMAT_ND},
                     {{{batches}, {batches}}, ge::DT_INT32, ge::FORMAT_ND, true, qLengths},
                     {{{batches}, {batches}}, ge::DT_INT32, ge::FORMAT_ND, true, kvLengths}},
                    {{{{queries, heads, 512}, {queries, heads, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                     {{{1, returnLse ? queries : 0, heads}, {1, returnLse ? queries : 0, heads}}, ge::DT_FLOAT, ge::FORMAT_ND},
                     {{{1, returnLse ? queries : 0, heads}, {1, returnLse ? queries : 0, heads}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                    {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
                     {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                     {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                     {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                     {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                     {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
                     {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                     {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                     {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                     {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                     {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                     {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
                     {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                     {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(returnLse)}},
                    &compileInfo, "Ascend910B", 24, 262144, 16384);
                TilingInfo info;
                ASSERT_TRUE(ExecuteTiling(para, info));
                uint32_t expected = expectedBlocks;
                if (heads == 64 && returnLse && topK >= 1024) {
                    uint32_t headSplits = 1;
                    while (headSplits * 2 <= std::min<int64_t>(8, 24 / queries)) {
                        headSplits *= 2;
                    }
                    if (headSplits > 1) {
                        expected = queries * headSplits;
                    }
                }
                EXPECT_EQ(info.blockNum, expected);
                // Read the serialized split block (s2, FD sizes, staging offset).
                // GetUserWorkspace skips the 16 MiB Ascend910B system workspace.
                optiling::KvQuantSparseFlashAttentionBaseParamsMla base;
                uint32_t splitFields[4];
                ASSERT_GE(info.tilingDataSize, base.GetDataSize() + sizeof(splitFields));
                std::memcpy(splitFields, info.tilingData.get() + base.GetDataSize(), sizeof(splitFields));
                if (splitFields[0] > 1) {
                    ASSERT_EQ(info.workspaceSizes.size(), 1);
                    const uint64_t stagingBytes = static_cast<uint64_t>(splitFields[0]) * queries * heads *
                                                  (512 * sizeof(uint16_t) + 2 * sizeof(float));
                    EXPECT_LE(static_cast<uint64_t>(splitFields[3]) + stagingBytes + (16U << 20),
                              static_cast<uint64_t>(info.workspaceSizes[0]));
                }
            }
        }
    }
}
