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
 * \file test_quant_lightning_indexer_v2_infershape.cpp
 * \brief Tests Quant Lightning Indexer V2 inference shape validation.
 */

#include <gtest/gtest.h>

#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

class QuantLightningIndexerV2InfershapeTopkTest : public testing::TestWithParam<int64_t> {};

TEST_P(QuantLightningIndexerV2InfershapeTopkTest, ChecksCommonTopkRange)
{
    const int64_t topk = GetParam();
    gert::InfershapeContextPara infershapeContextPara(
        "QuantLightningIndexerV2",
        {{{{2, 4, 64, 128}, {2, 4, 64, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2, 16, 1, 128}, {2, 16, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2, 4, 64}, {2, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 4, 64}, {2, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 16, 1}, {2, 16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},
         {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},
         {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},
         {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},
         {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},
         {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},
         {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},
         {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true}},
        {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(topk)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    const bool valid = topk >= 1 && topk <= 8192;
    const std::vector<std::vector<int64_t>> expectedShapes = {{2, 4, 1, topk}, {0}};
    ExecuteTestCase(infershapeContextPara, valid ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED, expectedShapes);
}

INSTANTIATE_TEST_SUITE_P(TopkBoundaries, QuantLightningIndexerV2InfershapeTopkTest,
                         testing::Values<int64_t>(0, 1, 8192, 8193));
