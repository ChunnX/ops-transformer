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
#include <tuple>
#include "tiling_case_executor.h"

TEST(KvQuantSparseFlashAttentionSplitKv, Arch22SplitAndHeadFallback)
{
    struct CompileInfo {} compileInfo;
    for (const auto &[queries, heads, topK, expectedBlocks] :
         {std::tuple<int64_t, int64_t, int64_t, uint32_t>{1, 8, 2048, 4},
          {4, 8, 2048, 16}, {1, 8, 1024, 2}, {4, 8, 1536, 8},
          {1, 8, 512, 24}, {1, 64, 2048, 4}, {4, 64, 2048, 16},
          {1, 64, 512, 24}, {16, 64, 2048, 24}, {16, 8, 2048, 24}}) {
        int32_t qLengths[] = {static_cast<int32_t>(queries)};
        int32_t kvLengths[] = {4096};
        gert::TilingContextPara para(
            "KvQuantSparseFlashAttention",
            {{{{queries, heads, 576}, {queries, heads, 576}}, ge::DT_BF16, ge::FORMAT_ND},
             {{{32, 128, 1, 656}, {32, 128, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
             {{{32, 128, 1, 656}, {32, 128, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
             {{{queries, 1, topK}, {queries, 1, topK}}, ge::DT_INT32, ge::FORMAT_ND},
             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
             {{{1, 32}, {1, 32}}, ge::DT_INT32, ge::FORMAT_ND},
             {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, qLengths},
             {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kvLengths}},
            {{{{queries, heads, 512}, {queries, heads, 512}}, ge::DT_BF16, ge::FORMAT_ND}},
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
             {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
            &compileInfo, "Ascend910B", 24, 262144, 16384);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(para, info));
        EXPECT_EQ(info.blockNum, expectedBlocks) << "Q=" << queries << ", G=" << heads << ", K=" << topK;
    }
}
