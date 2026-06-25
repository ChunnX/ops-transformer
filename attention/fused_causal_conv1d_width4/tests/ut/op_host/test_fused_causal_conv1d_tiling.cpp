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
 * \file test_fused_causal_conv1d_width4_tiling.cpp
 * \brief W=4 tiling smoke tests for FusedCausalConv1dWidth4.
 */

#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/fused_causal_conv1d_width4_cut_bh_tiling_arch35.h"
#include "../../../op_host/fused_causal_conv1d_width4_cut_bsh_tiling_arch35.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {

std::vector<gert::TilingContextPara::OpAttr> MakeAttrs(int64_t maxQueryLen)
{
    return {
        {"activation_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"pad_slot_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"run_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"max_query_len", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maxQueryLen)},
        {"residual_connection", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
        {"conv_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
    };
}

} // namespace

class FusedCausalConv1dWidth4Tiling : public testing::Test {};

TEST_F(FusedCausalConv1dWidth4Tiling, Bh3dFp16AcceptsKernelWidth4)
{
    optiling::FusedCausalConv1dWidth4CutBHCompileInfo compileInfo = {64, 261888};
    gert::TilingContextPara tilingPara(
        "FusedCausalConv1dWidth4",
        {
            {{{1, 5, 128}, {1, 5, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
            {{{4, 128}, {4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // weight, W=4
            {{{1, 7, 128}, {1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states: K-1+S-1
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // query_start_loc
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // cache_indices
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // initial_state_mode
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                  // bias
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // num_accepted_tokens
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // num_computed_tokens
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // block_idx_first_scheduled_token
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // block_idx_last_scheduled_token
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // initial_state_idx
        },
        {
            {{{1, 7, 128}, {1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states output
            {{{1, 5, 128}, {1, 5, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
        },
        MakeAttrs(5),
        &compileInfo);

    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, optiling::TILING_KEY_BH_FP16);
}

TEST_F(FusedCausalConv1dWidth4Tiling, Bh3dFp16RejectsKernelWidth3)
{
    optiling::FusedCausalConv1dWidth4CutBHCompileInfo compileInfo = {64, 261888};
    gert::TilingContextPara tilingPara(
        "FusedCausalConv1dWidth4",
        {
            {{{1, 5, 128}, {1, 5, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
            {{{3, 128}, {3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // invalid: W=3
            {{{1, 7, 128}, {1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 7, 128}, {1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 5, 128}, {1, 5, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        MakeAttrs(5),
        &compileInfo);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

TEST_F(FusedCausalConv1dWidth4Tiling, Bsh2dBf16AcceptsKernelWidth4)
{
    struct BshCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingPara(
        "FusedCausalConv1dWidth4",
        {
            {{{16, 128}, {16, 128}}, ge::DT_BF16, ge::FORMAT_ND},        // x, flattened token rows
            {{{4, 128}, {4, 128}}, ge::DT_BF16, ge::FORMAT_ND},          // weight, W=4
            {{{2, 7, 128}, {2, 7, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // conv_states
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},                  // query_start_loc: [0, 8, 16]
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                  // cache_indices
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // initial_state_mode
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                     // bias
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // num_accepted_tokens
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // num_computed_tokens
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // block_idx_first_scheduled_token
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // block_idx_last_scheduled_token
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // initial_state_idx
        },
        {
            {{{2, 7, 128}, {2, 7, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // conv_states output
            {{{16, 128}, {16, 128}}, ge::DT_BF16, ge::FORMAT_ND},        // y
        },
        MakeAttrs(16),
        &compileInfo);

    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, 10000);
}
