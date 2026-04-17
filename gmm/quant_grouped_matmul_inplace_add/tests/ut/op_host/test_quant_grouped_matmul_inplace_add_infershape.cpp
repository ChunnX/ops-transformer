/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "infer_shape_case_executor.h"

namespace {

class QuantGroupedMatmulInplaceAddInfershape : public testing::Test {};

TEST_F(QuantGroupedMatmulInplaceAddInfershape, infer_shape_success_copies_yref_shape)
{
    gert::InfershapeContextPara para(
        "QuantGroupedMatmulInplaceAdd",
        {
            {{{96, 512}, {96, 512}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},   // x1
            {{{512, 128}, {512, 128}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND}, // x2
            {{{12, 128, 2}, {12, 128, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // scale2
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                      // group_list
            {{{4, 96, 128}, {4, 96, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},    // yRef
            {{{96, 12, 2}, {96, 12, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND} // scale1
        },
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{4, 96, 128}});
}

TEST_F(QuantGroupedMatmulInplaceAddInfershape, infer_shape_fail_when_optional_scale1_missing)
{
    // InferShape implementation requires optional scale1 shape not null.
    gert::InfershapeContextPara para(
        "QuantGroupedMatmulInplaceAdd",
        {
            {{{96, 512}, {96, 512}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},   // x1
            {{{512, 128}, {512, 128}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND}, // x2
            {{{12, 128, 2}, {12, 128, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // scale2
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                      // group_list
            {{{4, 96, 128}, {4, 96, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}     // yRef
        },
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

} // namespace
