/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace {
struct ShapeCase {
    const char *name;
    const char *layout;
    size_t input;
    std::vector<int64_t> shape;
    ge::graphStatus expected = ge::GRAPH_FAILED;
};

gert::TilingContextPara::TensorDescription MakeShapeTensor(const std::vector<int64_t> &dims, ge::DataType dtype)
{
    gert::StorageShape shape;
    shape.MutableShape().SetDimNum(dims.size());
    shape.MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.MutableShape().SetDim(i, dims[i]);
        shape.MutableStorageShape().SetDim(i, dims[i]);
    }
    return {shape, dtype, ge::FORMAT_ND};
}

void RunShapeCase(const ShapeCase &test)
{
    const bool tnd = std::string(test.layout) == "TND" || std::string(test.layout) == "TND_PA";
    const bool paged = std::string(test.layout) == "PA_BBND" || std::string(test.layout) == "TND_PA";
    std::vector<std::vector<int64_t>> shapes = {
        tnd ? std::vector<int64_t>{8, 2, 128} : std::vector<int64_t>{2, 4, 2, 128},
        tnd && !paged ? std::vector<int64_t>{16, 1, 128} : std::vector<int64_t>{2, 16, 1, 128},
        tnd ? std::vector<int64_t>{8, 2} : std::vector<int64_t>{2, 4, 2},
        tnd ? std::vector<int64_t>{3} : std::vector<int64_t>{},
        tnd && !paged ? std::vector<int64_t>{3} : std::vector<int64_t>{},
        {2}, {2}, {2}, paged ? std::vector<int64_t>{2, 1} : std::vector<int64_t>{},
        tnd ? std::vector<int64_t>{8, 1} : std::vector<int64_t>{2, 4, 1}, {1024}};
    shapes[test.input] = test.shape;
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    for (size_t i = 0; i < shapes.size(); ++i) {
        const ge::DataType dtype = i < 2 ? ge::DT_FLOAT16 : (i == 2 ? ge::DT_FLOAT : ge::DT_INT32);
        inputs.push_back(MakeShapeTensor(shapes[i], dtype));
    }
    const auto output = tnd ? std::vector<int64_t>{8, 1, 4} : std::vector<int64_t>{2, 4, 1, 4};
    struct CompileInfo {} compileInfo;
    gert::TilingContextPara para(
        "LightningIndexerV2", inputs,
        {MakeShapeTensor(output, ge::DT_INT32), MakeShapeTensor({0}, ge::DT_FLOAT)},
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(tnd ? "TND" : "BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>(paged ? "PA_BBND" : test.layout)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(para, test.expected, UINT64_MAX);
}
} // namespace

class LightningIndexerV2ShapeContract : public testing::TestWithParam<ShapeCase> {};

TEST_P(LightningIndexerV2ShapeContract, RejectInvalidShape)
{
    RunShapeCase(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Ascend950, LightningIndexerV2ShapeContract, testing::Values(
    ShapeCase{"BSND_seqused_q_rank2", "BSND", 5, {1, 2}},
    ShapeCase{"BSND_seqused_q_short", "BSND", 5, {1}},
    ShapeCase{"BSND_seqused_q_long", "BSND", 5, {3}},
    ShapeCase{"BSND_seqused_k_rank2", "BSND", 6, {1, 2}},
    ShapeCase{"BSND_seqused_k_short", "BSND", 6, {1}},
    ShapeCase{"BSND_seqused_k_long", "BSND", 6, {3}},
    ShapeCase{"BSND_cmp_residual_k_rank2", "BSND", 7, {1, 2}},
    ShapeCase{"BSND_cmp_residual_k_short", "BSND", 7, {1}},
    ShapeCase{"BSND_cmp_residual_k_long", "BSND", 7, {3}},
    ShapeCase{"BSND_q_rank", "BSND", 0, {8, 2, 128}},
    ShapeCase{"BSND_k_rank", "BSND", 1, {32, 1, 128}},
    ShapeCase{"BSND_w_rank", "BSND", 2, {8, 2}},
    ShapeCase{"BSND_output_idx_offset_rank", "BSND", 9, {8, 1}},
    ShapeCase{"BSND_valid", "BSND", 5, {2}, ge::GRAPH_SUCCESS},
    ShapeCase{"BSND_optional_q_absent", "BSND", 5, {}, ge::GRAPH_SUCCESS},
    ShapeCase{"BSND_optional_k_absent", "BSND", 6, {}, ge::GRAPH_SUCCESS},
    ShapeCase{"TND_seqused_q_rank2", "TND", 5, {1, 2}},
    ShapeCase{"TND_seqused_q_short", "TND", 5, {1}},
    ShapeCase{"TND_seqused_q_long", "TND", 5, {3}},
    ShapeCase{"TND_seqused_k_rank2", "TND", 6, {1, 2}},
    ShapeCase{"TND_seqused_k_short", "TND", 6, {1}},
    ShapeCase{"TND_seqused_k_long", "TND", 6, {3}},
    ShapeCase{"TND_cmp_residual_k_rank2", "TND", 7, {1, 2}},
    ShapeCase{"TND_cmp_residual_k_short", "TND", 7, {1}},
    ShapeCase{"TND_cmp_residual_k_long", "TND", 7, {3}},
    ShapeCase{"TND_q_rank", "TND", 0, {2, 4, 2, 128}},
    ShapeCase{"TND_k_rank", "TND", 1, {2, 8, 1, 128}},
    ShapeCase{"TND_w_rank", "TND", 2, {2, 4, 2}},
    ShapeCase{"TND_output_idx_offset_rank", "TND", 9, {2, 4, 1}},
    ShapeCase{"TND_valid", "TND", 5, {2}, ge::GRAPH_SUCCESS},
    ShapeCase{"TND_optional_q_absent", "TND", 5, {}, ge::GRAPH_SUCCESS},
    ShapeCase{"TND_optional_k_absent", "TND", 6, {}, ge::GRAPH_SUCCESS},
    ShapeCase{"PA_BBND_seqused_q_rank2", "PA_BBND", 5, {1, 2}},
    ShapeCase{"PA_BBND_seqused_q_short", "PA_BBND", 5, {1}},
    ShapeCase{"PA_BBND_seqused_q_long", "PA_BBND", 5, {3}},
    ShapeCase{"PA_BBND_seqused_k_rank2", "PA_BBND", 6, {1, 2}},
    ShapeCase{"PA_BBND_seqused_k_short", "PA_BBND", 6, {1}},
    ShapeCase{"PA_BBND_seqused_k_long", "PA_BBND", 6, {3}},
    ShapeCase{"PA_BBND_cmp_residual_k_rank2", "PA_BBND", 7, {1, 2}},
    ShapeCase{"PA_BBND_cmp_residual_k_short", "PA_BBND", 7, {1}},
    ShapeCase{"PA_BBND_cmp_residual_k_long", "PA_BBND", 7, {3}},
    ShapeCase{"PA_BBND_q_rank", "PA_BBND", 0, {8, 2, 128}},
    ShapeCase{"PA_BBND_k_rank", "PA_BBND", 1, {32, 1, 128}},
    ShapeCase{"PA_BBND_w_rank", "PA_BBND", 2, {8, 2}},
    ShapeCase{"PA_BBND_output_idx_offset_rank", "PA_BBND", 9, {8, 1}},
    ShapeCase{"PA_BBND_valid", "PA_BBND", 5, {2}, ge::GRAPH_SUCCESS},
    ShapeCase{"PA_BBND_optional_q_absent", "PA_BBND", 5, {}, ge::GRAPH_SUCCESS},
    ShapeCase{"TND_PA_seqused_q_rank2", "TND_PA", 5, {1, 2}},
    ShapeCase{"TND_PA_seqused_q_short", "TND_PA", 5, {1}},
    ShapeCase{"TND_PA_seqused_q_long", "TND_PA", 5, {3}},
    ShapeCase{"TND_PA_seqused_k_rank2", "TND_PA", 6, {1, 2}},
    ShapeCase{"TND_PA_seqused_k_short", "TND_PA", 6, {1}},
    ShapeCase{"TND_PA_seqused_k_long", "TND_PA", 6, {3}},
    ShapeCase{"TND_PA_cmp_residual_k_rank2", "TND_PA", 7, {1, 2}},
    ShapeCase{"TND_PA_cmp_residual_k_short", "TND_PA", 7, {1}},
    ShapeCase{"TND_PA_cmp_residual_k_long", "TND_PA", 7, {3}},
    ShapeCase{"TND_PA_q_rank", "TND_PA", 0, {2, 4, 2, 128}},
    ShapeCase{"TND_PA_k_rank", "TND_PA", 1, {32, 1, 128}},
    ShapeCase{"TND_PA_w_rank", "TND_PA", 2, {2, 4, 2}},
    ShapeCase{"TND_PA_output_idx_offset_rank", "TND_PA", 9, {2, 4, 1}},
    ShapeCase{"TND_PA_valid", "TND_PA", 5, {2}, ge::GRAPH_SUCCESS},
    ShapeCase{"TND_PA_optional_q_absent", "TND_PA", 5, {}, ge::GRAPH_SUCCESS},
    ShapeCase{"TND_cu_seqlens_q_rank2", "TND", 3, {1, 3}},
    ShapeCase{"TND_cu_seqlens_k_rank2", "TND", 4, {1, 3}},
    ShapeCase{"BSND_cu_seqlens_k_forbidden", "BSND", 4, {3}}
), [](const testing::TestParamInfo<ShapeCase> &info) { return info.param.name; });
