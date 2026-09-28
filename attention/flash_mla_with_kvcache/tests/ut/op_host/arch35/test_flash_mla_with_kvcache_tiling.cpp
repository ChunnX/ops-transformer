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
#include <cstring>
#include "../../../../op_kernel/arch35/flash_mla_with_kvcache_tiling_data.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "../flash_mla_with_kvcache_param.h"
#include "tiling_case_executor.h"

namespace FlashMlaWithKvcacheUT {

class FlashMlaWithKvcacheArch35TilingTest : public testing::TestWithParam<FlashMlaWithKvcacheTilingUtParam> {};

TEST_P(FlashMlaWithKvcacheArch35TilingTest, tiling)
{
    auto param = GetParam();

    gert::TilingContextPara tilingContextPara(
        "FlashMlaWithKvcache",
        {
            param.q,
            param.k,
            param.block_table,
            param.cache_seqlens,
            param.cu_seqlens_q,
            param.seqused_q,
            param.attn_mask,
            param.metadata,
            param.sinks,
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
            {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.win_left)},
            {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.win_right)},
        },
        param.inputInstance, param.outputInstance, &param.compileInfo, "Ascend950", 64, 262144, 8192);

    const auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr) << "Set BUILD_PATH to the CMake UT build directory.";
    ASSERT_NE(registry->GetOpImpl("FlashMlaWithKvcache"), nullptr);
    TilingInfo result;
    ASSERT_EQ(ExecuteTiling(tilingContextPara, result), param.expectResult == ge::GRAPH_SUCCESS);
    if (param.expectResult != ge::GRAPH_SUCCESS) {
        return;
    }
    ASSERT_EQ(result.tilingDataSize, sizeof(optiling::FlashMlaWithKvcacheTilingData));
    optiling::FlashMlaWithKvcacheTilingData data;
    std::memcpy(&data, result.tilingData.get(), sizeof(data));
    const auto &mask = data.baseTiling.flashMlaWithKvcacheAttenMaskParams;
    EXPECT_EQ(mask.sparseMode, param.mask_mode);
    const int64_t expectedLeft = param.mask_mode == 4 && param.win_left != -1 ? param.win_left : INT32_MAX;
    const int64_t expectedRight =
        param.mask_mode == 3 ? 0 : (param.mask_mode == 4 && param.win_right != -1 ? param.win_right : INT32_MAX);
    EXPECT_EQ(mask.preTokens, expectedLeft);
    EXPECT_EQ(mask.nextTokens, expectedRight);
    if (param.mask_mode != 0) {
        EXPECT_EQ(mask.attenMaskBatch, 1U);
        EXPECT_EQ(mask.attenMaskS1Size, 2048U);
        EXPECT_EQ(mask.attenMaskS2Size, 2048U);
    }
    const auto &base = data.baseTiling.flashMlaWithKvcacheBaseParams;
    EXPECT_EQ(base.isSoftMaxLseEnable, param.return_softmax_lse);
    EXPECT_EQ(base.n2Size, 1U);
    EXPECT_EQ(base.dSize, 512U);
    EXPECT_EQ(base.dSizeV, 512U);
    EXPECT_EQ(base.dSizeRope, 64U);
    EXPECT_GT(result.blockNum, 0U);
}

INSTANTIATE_TEST_SUITE_P(
    FlashMlaWithKvcache, FlashMlaWithKvcacheArch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<FlashMlaWithKvcacheTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<FlashMlaWithKvcacheTilingUtParam>);

} // namespace FlashMlaWithKvcacheUT
