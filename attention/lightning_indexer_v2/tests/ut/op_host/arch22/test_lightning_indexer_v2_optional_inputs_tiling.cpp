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
 * \file test_lightning_indexer_v2_optional_inputs_tiling.cpp
 * \brief Tests compatibility behavior for ignored Lightning Indexer V2 optional inputs.
 */

#include <gtest/gtest.h>

#include <limits>

#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
constexpr char OP_NAME[] = "LightningIndexerV2";
constexpr char TND_LAYOUT[] = "TND";
constexpr char A2_A3_PLATFORM[] = "Ascend910B";
constexpr int64_t FIRST_BATCH_QUERY_END = 4;
constexpr int64_t QUERY_TOKEN_NUM = 8;
constexpr int64_t FIRST_BATCH_KEY_END = 8;
constexpr int64_t KEY_TOKEN_NUM = 16;
constexpr int64_t QUERY_HEAD_NUM = 64;
constexpr int64_t KEY_HEAD_NUM = 1;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t CU_SEQLENS_SIZE = 3;
constexpr int64_t TOPK = 32;
constexpr uint32_t A2_A3_CORE_NUM = 64;
constexpr uint64_t A2_A3_UB_SIZE = 262144;
constexpr uint64_t A2_A3_TILING_DATA_SIZE = 16384;
} // namespace

TEST(LightningIndexerV2OptionalInputs, AcceptsIgnoredA2A3Inputs)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int32_t cuSeqlensQ[] = {0, FIRST_BATCH_QUERY_END, QUERY_TOKEN_NUM};
    int32_t cuSeqlensK[] = {0, FIRST_BATCH_KEY_END, KEY_TOKEN_NUM};
    float ignoredSequsedQ[] = {-1.0F};
    float ignoredOutputOffset[] = {-1.0F};
    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        {{{{QUERY_TOKEN_NUM, QUERY_HEAD_NUM, HEAD_DIM}, {QUERY_TOKEN_NUM, QUERY_HEAD_NUM, HEAD_DIM}},
          ge::DT_BF16,
          ge::FORMAT_ND},
         {{{KEY_TOKEN_NUM, KEY_HEAD_NUM, HEAD_DIM}, {KEY_TOKEN_NUM, KEY_HEAD_NUM, HEAD_DIM}},
          ge::DT_BF16,
          ge::FORMAT_ND},
         {{{QUERY_TOKEN_NUM, QUERY_HEAD_NUM}, {QUERY_TOKEN_NUM, QUERY_HEAD_NUM}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{CU_SEQLENS_SIZE}, {CU_SEQLENS_SIZE}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensQ},
         {{{CU_SEQLENS_SIZE}, {CU_SEQLENS_SIZE}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensK},
         {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, ignoredSequsedQ},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, ignoredOutputOffset},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{QUERY_TOKEN_NUM, KEY_HEAD_NUM, TOPK}, {QUERY_TOKEN_NUM, KEY_HEAD_NUM, TOPK}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(TOPK)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(TND_LAYOUT)},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>(TND_LAYOUT)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, A2_A3_PLATFORM, A2_A3_CORE_NUM, A2_A3_UB_SIZE, A2_A3_TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max());
}
