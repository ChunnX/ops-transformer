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
 * \brief Tests compatibility behavior for ignored
 * Lightning Indexer V2 optional inputs.
 */

#include <gtest/gtest.h>

#include <limits>

#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

TEST(LightningIndexerV2OptionalInputs, AcceptsIgnoredA2A3Inputs)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int32_t cuSeqlensQ[] = {0, 4, 8};
    int32_t cuSeqlensK[] = {0, 8, 16};
    float ignoredSequsedQ[] = {-1.0F};
    float ignoredOutputOffset[] = {-1.0F};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {{{{8, 64, 128}, {8, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{16, 1, 128}, {16, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensQ},
         {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensK},
         {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, ignoredSequsedQ},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, ignoredOutputOffset},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{8, 1, 32}, {8, 1, 32}}, ge::DT_INT32, ge::FORMAT_ND}, {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max());
}
