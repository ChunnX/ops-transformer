/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if defined(MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950)

#include <gtest/gtest.h>

#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

namespace {
using Attr = gert::InfershapeContextPara::OpAttr;

class MoeTokenUnpermuteWithEpAscend950InferShape : public testing::Test {};

void RunNoProbsCase(const gert::StorageShape &tokensShape, const gert::StorageShape &indicesShape, int64_t numTopK,
                    const std::vector<int64_t> *range, ge::graphStatus expectedStatus,
                    const std::vector<std::vector<int64_t>> &expectedOutputShape = {}, bool paddedMode = false,
                    const std::vector<int64_t> &restoreShape = {})
{
    std::vector<Attr> attrs = {
        {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numTopK)},
    };
    if (range != nullptr) {
        attrs.emplace_back("range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(*range));
        attrs.emplace_back("padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(paddedMode));
        attrs.emplace_back("restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(restoreShape));
    }

    gert::InfershapeContextPara para("MoeTokenUnpermuteWithEp",
                                     {
                                         {tokensShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     },
                                     attrs);
    ExecuteTestCase(para, expectedStatus, expectedOutputShape);
}

void RunProbsCase(const gert::StorageShape &tokensShape, const gert::StorageShape &indicesShape,
                  const gert::StorageShape &probsShape, int64_t numTopK, const std::vector<int64_t> &range,
                  const std::vector<std::vector<int64_t>> &expectedOutputShape)
{
    gert::InfershapeContextPara para(
        "MoeTokenUnpermuteWithEp",
        {
            {tokensShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
            {probsShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numTopK)},
            {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})},
        });
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectedOutputShape);
}
} // namespace

TEST_F(MoeTokenUnpermuteWithEpAscend950InferShape, FallbackZeroRangeIgnoresNumTopK)
{
    const std::vector<int64_t> range = {0, 0};
    RunNoProbsCase({{17, 64}, {17, 64}}, {{17}, {17}}, 512, &range, ge::GRAPH_SUCCESS, {{17, 64}});
}

TEST_F(MoeTokenUnpermuteWithEpAscend950InferShape, FallbackEmptyRangeAllowsZeroNumTopK)
{
    const std::vector<int64_t> range = {};
    RunNoProbsCase({{17, 64}, {17, 64}}, {{17}, {17}}, 0, &range, ge::GRAPH_SUCCESS, {{17, 64}});
}

TEST_F(MoeTokenUnpermuteWithEpAscend950InferShape, FallbackMissingRangeAllowsZeroNumTopK)
{
    RunNoProbsCase({{17, 64}, {17, 64}}, {{17}, {17}}, 0, nullptr, ge::GRAPH_SUCCESS, {{17, 64}});
}

TEST_F(MoeTokenUnpermuteWithEpAscend950InferShape, FallbackWithProbsUsesProbsRows)
{
    RunProbsCase({{18, 64}, {18, 64}}, {{18}, {18}}, {{9, 2}, {9, 2}}, 0, {0, 0}, {{9, 64}});
}

TEST_F(MoeTokenUnpermuteWithEpAscend950InferShape, NonFallbackRejectsZeroNumTopK)
{
    const std::vector<int64_t> range = {0, 17};
    RunNoProbsCase({{17, 64}, {17, 64}}, {{17}, {17}}, 0, &range, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteWithEpAscend950InferShape, NonFallbackRejectsNumTopKAboveLimit)
{
    const std::vector<int64_t> range = {0, 17};
    RunNoProbsCase({{17, 64}, {17, 64}}, {{513}, {513}}, 513, &range, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteWithEpAscend950InferShape, RejectsPaddedMode)
{
    const std::vector<int64_t> range = {0, 17};
    RunNoProbsCase({{17, 64}, {17, 64}}, {{17}, {17}}, 1, &range, ge::GRAPH_FAILED, {}, true);
}

TEST_F(MoeTokenUnpermuteWithEpAscend950InferShape, RejectsNonEmptyRestoreShape)
{
    const std::vector<int64_t> range = {0, 17};
    RunNoProbsCase({{17, 64}, {17, 64}}, {{17}, {17}}, 1, &range, ge::GRAPH_FAILED, {}, false, {17, 64});
}

TEST_F(MoeTokenUnpermuteWithEpAscend950InferShape, NonFallbackRejectsIndivisibleIndices)
{
    const std::vector<int64_t> range = {0, 17};
    RunNoProbsCase({{17, 64}, {17, 64}}, {{17}, {17}}, 8, &range, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteWithEpAscend950InferShape, UnknownRankIsSafe)
{
    const std::vector<int64_t> range = {0, 0};
    RunNoProbsCase({{-2}, {-2}}, {{-2}, {-2}}, 0, &range, ge::GRAPH_SUCCESS, {{-2}});
}

TEST_F(MoeTokenUnpermuteWithEpAscend950InferShape, DynamicShapeKeepsRankTwo)
{
    const std::vector<int64_t> range = {};
    RunNoProbsCase({{-1, -1}, {-1, -1}}, {{-1}, {-1}}, 0, &range, ge::GRAPH_SUCCESS, {{-1, -1}});
}

#endif
