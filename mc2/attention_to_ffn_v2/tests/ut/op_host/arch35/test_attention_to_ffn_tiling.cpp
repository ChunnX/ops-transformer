/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

namespace AttentionToFFnUT {

class AttentionToFFnArch35TilingTest : public testing::Test {};

TEST_F(AttentionToFFnArch35TilingTest, NoQuantSyncNoActiveMask)
{
    struct AttentionToFFnCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFnV2",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend950"
    );

    constexpr uint64_t expectTilingKey = 8UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

} // namespace AttentionToFFnUT
