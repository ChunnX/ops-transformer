/**
 * Copyright (c) 2026 fk2323
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <initializer_list>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {

gert::StorageShape MakeShape(std::initializer_list<int64_t> dims)
{
    gert::StorageShape shape;
    auto &origin = shape.MutableOriginShape();
    auto &storage = shape.MutableStorageShape();
    origin.SetDimNum(dims.size());
    storage.SetDimNum(dims.size());
    size_t index = 0;
    for (const auto dim : dims) {
        origin.SetDim(index, dim);
        storage.SetDim(index, dim);
        ++index;
    }
    return shape;
}

gert::TilingContextPara BuildCase(const char *opName, std::initializer_list<int64_t> qDims,
                                  std::initializer_list<int64_t> kDims, std::initializer_list<int64_t> weightDims,
                                  int64_t topK, bool setTopK = true, uint32_t coreNum = 64, uint64_t ubSize = 253952)
{
    const int64_t outputTopK = topK > 0 ? topK : 1;
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {MakeShape(qDims), ge::DT_FLOAT, ge::FORMAT_ND},
        {MakeShape(kDims), ge::DT_FLOAT, ge::FORMAT_ND},
        {MakeShape(weightDims), ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {MakeShape({1, 1, 1, outputTopK}), ge::DT_INT32, ge::FORMAT_ND},
        {MakeShape({1, 1, 1, outputTopK}), ge::DT_UINT8, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::OpAttr> attrs;
    if (setTopK) {
        attrs.emplace_back("top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(topK));
    }
    struct CompileInfo {
    } compileInfo;
    return gert::TilingContextPara(opName, inputs, outputs, attrs, &compileInfo, "Ascend910B", coreNum, ubSize);
}

class Fmm016StreamingTopKBlockedV4TilingTest : public testing::Test {};

TEST_F(Fmm016StreamingTopKBlockedV4TilingTest, ValidShortAndDefaultTopK)
{
    ExecuteTestCase(BuildCase("Fmm016StreamingTopKBlockedV4", {1, 2, 9, 2, 8}, {1, 2, 8, 1024}, {1, 2, 9, 2}, 4),
                    ge::GRAPH_SUCCESS, UINT64_MAX);
    ExecuteTestCase(BuildCase("Fmm016StreamingTopKBlockedV4", {1, 1, 9, 1, 8}, {1, 1, 8, 1024}, {1, 1, 9, 1}, 1, false),
                    ge::GRAPH_SUCCESS, UINT64_MAX);
}

TEST_F(Fmm016StreamingTopKBlockedV4TilingTest, ValidCrossSourceTileAndBhBoundary)
{
    ExecuteTestCase(
        BuildCase("Fmm016StreamingTopKBlockedV4", {2, 2, 1025, 4, 16}, {2, 2, 16, 2048}, {2, 2, 1025, 4}, 16),
        ge::GRAPH_SUCCESS, UINT64_MAX);
}

TEST_F(Fmm016StreamingTopKBlockedV4TilingTest, RejectsInvalidRank)
{
    ExecuteTestCase(BuildCase("Fmm016StreamingTopKBlockedV4", {1, 2, 9, 2}, {1, 2, 8, 1024}, {1, 2, 9, 2}, 4),
                    ge::GRAPH_FAILED);
}

TEST_F(Fmm016StreamingTopKBlockedV4TilingTest, RejectsInvalidLimitsAndPadding)
{
    ExecuteTestCase(BuildCase("Fmm016StreamingTopKBlockedV4", {1, 1, 9, 5, 8}, {1, 1, 8, 1024}, {1, 1, 9, 5}, 4),
                    ge::GRAPH_FAILED);
    ExecuteTestCase(BuildCase("Fmm016StreamingTopKBlockedV4", {1, 1, 9, 1, 17}, {1, 1, 17, 1024}, {1, 1, 9, 1}, 4),
                    ge::GRAPH_FAILED);
    ExecuteTestCase(BuildCase("Fmm016StreamingTopKBlockedV4", {1, 1, 9, 1, 8}, {1, 1, 8, 1024}, {1, 1, 9, 1}, 17),
                    ge::GRAPH_FAILED);
    ExecuteTestCase(BuildCase("Fmm016StreamingTopKBlockedV4", {1, 1, 9, 1, 8}, {1, 1, 8, 1025}, {1, 1, 9, 1}, 4),
                    ge::GRAPH_FAILED);
    ExecuteTestCase(BuildCase("Fmm016StreamingTopKBlockedV4", {1, 1, 1025, 1, 8}, {1, 1, 8, 1024}, {1, 1, 1025, 1}, 4),
                    ge::GRAPH_FAILED);
}

TEST_F(Fmm016StreamingTopKBlockedV4TilingTest, RejectsShapeMismatch)
{
    ExecuteTestCase(BuildCase("Fmm016StreamingTopKBlockedV4", {1, 1, 9, 1, 8}, {2, 1, 8, 1024}, {1, 1, 9, 1}, 4),
                    ge::GRAPH_FAILED);
    ExecuteTestCase(BuildCase("Fmm016StreamingTopKBlockedV4", {1, 1, 9, 1, 8}, {1, 1, 8, 1024}, {1, 1, 8, 1}, 4),
                    ge::GRAPH_FAILED);
}

} // namespace
