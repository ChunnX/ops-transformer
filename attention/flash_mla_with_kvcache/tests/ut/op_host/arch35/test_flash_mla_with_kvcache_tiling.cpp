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
#include "../flash_mla_with_kvcache_param.h"
#include "tiling_case_executor.h"

namespace FlashMlaWithKvcacheUT {

class FlashMlaWithKvcacheArch35TilingTest : public testing::TestWithParam<FlashMlaWithKvcacheTilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        // std::cout << "FlashMlaWithKvcache Arch35 TilingTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        // std::cout << "FlashMlaWithKvcache Arch35 TilingTest TearDown" << std::endl;
    }
};

TEST_P(FlashMlaWithKvcacheArch35TilingTest, tiling)
{
    auto param = GetParam();

    gert::TilingContextPara tilingContextPara(
        "FlashMlaWithKvcache",
        {
            param.q,
            param.k_cache,
            param.block_table,
            param.cache_seqlens,
            param.cu_seqlens_q,
            param.seqused_q,
            param.attn_mask,
            param.metadata,
        },
        {
            param.attn_out,
            param.softmax_lse,
        },
        {
            {"head_dim_v", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.head_dim_v)},
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(param.softmax_scale)},
            {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mask_mode)},
            {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.max_seqlen_q)},
            {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.max_seqlen_kv)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_q)},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_kv)},
            {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_out)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.return_softmax_lse)},
        },
        param.inputInstance, param.outputInstance, &param.compileInfo, "Ascend950", 64, 262144, 8192);

    ExecuteTestCase(tilingContextPara, param.expectResult, param.expectTilingKey, param.expectTilingDataHash, {}, 0,
                    true);
}

INSTANTIATE_TEST_SUITE_P(FlashMlaWithKvcache, FlashMlaWithKvcacheArch35TilingTest,
                         testing::ValuesIn(GetCasesFromCsv<FlashMlaWithKvcacheTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
                         PrintCaseInfoString<FlashMlaWithKvcacheTilingUtParam>);

} // namespace FlashMlaWithKvcacheUT
