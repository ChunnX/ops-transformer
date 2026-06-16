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

#include "infer_shape_case_executor.h"

namespace {

std::vector<gert::InfershapeContextPara::TensorDescription> BuildInputs()
{
    return {
        {{{20, 17, 128}, {20, 17, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{256, 128}, {256, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{17}, {17}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{8, 2, 128, 128}, {8, 2, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{{8, 2, 128, 128}, {8, 2, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{{8, 2, 128, 1}, {8, 2, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{128, 128}, {128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{128, 128}, {128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
}

std::vector<gert::InfershapeContextPara::TensorDescription> BuildOutputs()
{
    return {
        {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
}

} // namespace

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, InfersOutputAndCacheShapes)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    gert::InfershapeContextPara para("QkvRmsNormRopeCacheWithKScale", BuildInputs(), BuildOutputs(),
        {{"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(headNums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)}});

    std::vector<std::vector<int64_t>> expected = {
        {16, 17, 128},
        {16, 17},
        {8, 2, 128, 128},
        {8, 2, 128, 128},
        {8, 2, 128, 1},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, RejectsInvalidHeadNumsAttr)
{
    std::vector<int64_t> headNums = {16, 2};
    gert::InfershapeContextPara para("QkvRmsNormRopeCacheWithKScale", BuildInputs(), BuildOutputs(),
        {{"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(headNums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)}});

    std::vector<std::vector<int64_t>> expected;
    ExecuteTestCase(para, ge::GRAPH_FAILED, expected);

    headNums = {0, 2, 2};
    gert::InfershapeContextPara zeroHeadPara("QkvRmsNormRopeCacheWithKScale", BuildInputs(), BuildOutputs(),
        {{"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(headNums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)}});
    ExecuteTestCase(zeroHeadPara, ge::GRAPH_FAILED, expected);

    headNums = {16, -1, 2};
    gert::InfershapeContextPara negativeHeadPara("QkvRmsNormRopeCacheWithKScale", BuildInputs(), BuildOutputs(),
        {{"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(headNums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)}});
    ExecuteTestCase(negativeHeadPara, ge::GRAPH_FAILED, expected);
}
