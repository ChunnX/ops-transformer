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
 * \file test_moe_tutel_combine_gates_tiling.cpp
 * \brief Ascend 950 tiling contract and branch tests.
 */

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <vector>

#include "../../../op_host/arch35/moe_tutel_combine_gates_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;

namespace {
const std::string kAscend950SocInfo = "{\n"
                                      "  \"hardware_info\": {\n"
                                      "    \"UB_SIZE\": 262144,\n"
                                      "    \"CORE_NUM\": 32,\n"
                                      "    \"socVersion\": \"Ascend910_95\"\n"
                                      "  }\n"
                                      "}";

constexpr size_t SYS_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;

struct TensorMeta {
    ge::DataType yType = ge::DT_UNDEFINED;
    ge::DataType outType = ge::DT_UNDEFINED;
    ge::DataType indicesType = ge::DT_INT32;
    ge::DataType locationsType = ge::DT_INT32;
    ge::Format xFormat = ge::FORMAT_ND;
    ge::Format yFormat = ge::FORMAT_ND;
    ge::Format indicesFormat = ge::FORMAT_ND;
    ge::Format locationsFormat = ge::FORMAT_ND;
    ge::Format outFormat = ge::FORMAT_ND;
};

optiling::MoeTutelCombineGatesCompileInfo MakeCompileInfo(uint64_t coreNum = 32, uint64_t ubSize = 262144,
                                                          uint32_t vectorLength = 256, uint32_t ubBlockSize = 32)
{
    optiling::MoeTutelCombineGatesCompileInfo info;
    info.coreNum = coreNum;
    info.ubSize = ubSize;
    info.vectorLength = vectorLength;
    info.ubBlockSize = ubBlockSize;
    info.workspaceSize = SYS_WORKSPACE_SIZE;
    return info;
}

gert::StorageShape MakeStorageShape(const vector<int64_t> &dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

// x[S,H], y_grad[E,C,H], and route/output tensors [K,S].
gert::TilingContextPara MakePara(const vector<int64_t> &xShape, const vector<int64_t> &yShape,
                                 const vector<int64_t> &indicesShape, const vector<int64_t> &locationsShape,
                                 const vector<int64_t> &outShape, ge::DataType floatType,
                                 optiling::MoeTutelCombineGatesCompileInfo *compileInfo,
                                 const TensorMeta &meta = TensorMeta())
{
    const ge::DataType yType = meta.yType == ge::DT_UNDEFINED ? floatType : meta.yType;
    const ge::DataType outType = meta.outType == ge::DT_UNDEFINED ? floatType : meta.outType;
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    inputs.emplace_back(MakeStorageShape(xShape), floatType, meta.xFormat);
    inputs.emplace_back(MakeStorageShape(yShape), yType, meta.yFormat);
    inputs.emplace_back(MakeStorageShape(indicesShape), meta.indicesType, meta.indicesFormat);
    inputs.emplace_back(MakeStorageShape(locationsShape), meta.locationsType, meta.locationsFormat);

    std::vector<gert::TilingContextPara::TensorDescription> outputs;
    outputs.emplace_back(MakeStorageShape(outShape), outType, meta.outFormat);
    return gert::TilingContextPara("MoeTutelCombineGates", inputs, outputs,
                                   std::vector<gert::TilingContextPara::OpAttr>{}, compileInfo, "Ascend950",
                                   kAscend950SocInfo, 4096);
}

void ExpectSuccess(gert::TilingContextPara &para, uint64_t tilingKey)
{
    std::vector<size_t> expectWorkspaces = {SYS_WORKSPACE_SIZE};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, tilingKey, "", expectWorkspaces);
}
} // namespace

class MoeTutelCombineGatesTilingTest : public testing::Test {};

TEST_F(MoeTutelCombineGatesTilingTest, not_cut_h_float32_top1)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({8, 64}, {2, 4, 64}, {1, 8}, {1, 8}, {1, 8}, ge::DT_FLOAT, &compileInfo);
    ExpectSuccess(para, optiling::MTCG_TILING_KEY_NOT_CUT_H);
}

TEST_F(MoeTutelCombineGatesTilingTest, not_cut_h_float16_k_less_than_e)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({100, 1024}, {4, 16, 1024}, {2, 100}, {2, 100}, {2, 100}, ge::DT_FLOAT16, &compileInfo);
    ExpectSuccess(para, optiling::MTCG_TILING_KEY_NOT_CUT_H);
}

TEST_F(MoeTutelCombineGatesTilingTest, not_cut_h_bfloat16_k_equals_e)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({7, 33}, {3, 5, 33}, {3, 7}, {3, 7}, {3, 7}, ge::DT_BF16, &compileInfo);
    ExpectSuccess(para, optiling::MTCG_TILING_KEY_NOT_CUT_H);
}

TEST_F(MoeTutelCombineGatesTilingTest, cut_h_float32_large_hidden)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({4, 65536}, {2, 3, 65536}, {1, 4}, {1, 4}, {1, 4}, ge::DT_FLOAT, &compileInfo);
    ExpectSuccess(para, optiling::MTCG_TILING_KEY_CUT_H);
}

TEST_F(MoeTutelCombineGatesTilingTest, empty_samples_is_supported)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({0, 64}, {2, 4, 64}, {1, 0}, {1, 0}, {1, 0}, ge::DT_FLOAT, &compileInfo);
    ExpectSuccess(para, optiling::MTCG_TILING_KEY_NOT_CUT_H);
}

TEST_F(MoeTutelCombineGatesTilingTest, empty_samples_and_hidden_are_supported)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({0, 0}, {2, 4, 0}, {1, 0}, {1, 0}, {1, 0}, ge::DT_FLOAT, &compileInfo);
    ExpectSuccess(para, optiling::MTCG_TILING_KEY_NOT_CUT_H);
}

TEST_F(MoeTutelCombineGatesTilingTest, zero_hidden_is_supported)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({8, 0}, {2, 4, 0}, {1, 8}, {1, 8}, {1, 8}, ge::DT_FLOAT, &compileInfo);
    ExpectSuccess(para, optiling::MTCG_TILING_KEY_NOT_CUT_H);
}

TEST_F(MoeTutelCombineGatesTilingTest, empty_samples_with_zero_experts_fails)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({0, 64}, {0, 4, 64}, {1, 0}, {1, 0}, {1, 0}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, empty_samples_with_zero_capacity_fails)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({0, 64}, {2, 0, 64}, {1, 0}, {1, 0}, {1, 0}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, empty_samples_with_zero_routes_fails)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({0, 64}, {2, 4, 64}, {0, 0}, {0, 0}, {0, 0}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, all_non_optional_axes_zero_fail)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({0, 64}, {0, 0, 64}, {0, 0}, {0, 0}, {0, 0}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, capacity_may_exceed_samples)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({2, 64}, {3, 8, 64}, {2, 2}, {2, 2}, {2, 2}, ge::DT_FLOAT16, &compileInfo);
    ExpectSuccess(para, optiling::MTCG_TILING_KEY_NOT_CUT_H);
}

TEST_F(MoeTutelCombineGatesTilingTest, route_count_greater_than_experts_fails)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({8, 64}, {2, 4, 64}, {3, 8}, {3, 8}, {3, 8}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, zero_route_count_fails)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({8, 64}, {2, 4, 64}, {0, 8}, {0, 8}, {0, 8}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, zero_expert_count_fails)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({8, 64}, {0, 4, 64}, {1, 8}, {1, 8}, {1, 8}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, zero_capacity_fails)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({8, 64}, {2, 0, 64}, {1, 8}, {1, 8}, {1, 8}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, indices_locations_shape_mismatch_fails)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({8, 64}, {2, 4, 64}, {1, 8}, {1, 4}, {1, 8}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, output_shape_mismatch_fails)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({8, 64}, {2, 4, 64}, {1, 8}, {1, 8}, {2, 8}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, hidden_mismatch_fails)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({8, 64}, {2, 4, 32}, {1, 8}, {1, 8}, {1, 8}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, invalid_y_grad_rank_fails)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({8, 64}, {2, 64}, {1, 8}, {1, 8}, {1, 8}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, one_dimensional_empty_x_fails)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({0}, {2, 4, 64}, {1, 0}, {1, 0}, {1, 0}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, invalid_route_rank_fails)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({8, 64}, {2, 4, 64}, {8}, {8}, {8}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, mixed_input_float_dtype_fails)
{
    auto compileInfo = MakeCompileInfo();
    TensorMeta meta;
    meta.yType = ge::DT_FLOAT16;
    auto para = MakePara({8, 64}, {2, 4, 64}, {1, 8}, {1, 8}, {1, 8}, ge::DT_FLOAT, &compileInfo, meta);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, mismatched_output_dtype_fails)
{
    auto compileInfo = MakeCompileInfo();
    TensorMeta meta;
    meta.outType = ge::DT_FLOAT16;
    auto para = MakePara({8, 64}, {2, 4, 64}, {1, 8}, {1, 8}, {1, 8}, ge::DT_FLOAT, &compileInfo, meta);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, non_int32_route_dtype_fails)
{
    auto compileInfo = MakeCompileInfo();
    TensorMeta meta;
    meta.indicesType = ge::DT_INT64;
    auto para = MakePara({8, 64}, {2, 4, 64}, {1, 8}, {1, 8}, {1, 8}, ge::DT_FLOAT, &compileInfo, meta);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, non_nd_format_fails)
{
    auto compileInfo = MakeCompileInfo();
    TensorMeta meta;
    meta.xFormat = ge::FORMAT_NCHW;
    auto para = MakePara({8, 64}, {2, 4, 64}, {1, 8}, {1, 8}, {1, 8}, ge::DT_FLOAT, &compileInfo, meta);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, unknown_dimension_fails_at_tiling)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({8, 64}, {2, 4, 64}, {-1, 8}, {-1, 8}, {-1, 8}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, tensor_element_count_overflow_fails)
{
    auto compileInfo = MakeCompileInfo();
    constexpr int64_t huge = std::numeric_limits<int64_t>::max();
    auto para = MakePara({huge, 2}, {1, 1, 2}, {1, huge}, {1, huge}, {1, huge}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, zero_core_count_fails)
{
    auto compileInfo = MakeCompileInfo(0);
    auto para = MakePara({8, 64}, {2, 4, 64}, {1, 8}, {1, 8}, {1, 8}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, invalid_vector_length_fails)
{
    auto compileInfo = MakeCompileInfo(32, 262144, 192);
    auto para = MakePara({8, 64}, {2, 4, 64}, {1, 8}, {1, 8}, {1, 8}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, non_integral_fp32_vector_length_fails)
{
    auto compileInfo = MakeCompileInfo(32, 262144, 258);
    auto para = MakePara({8, 64}, {2, 4, 64}, {1, 8}, {1, 8}, {1, 8}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, insufficient_ub_fails)
{
    auto compileInfo = MakeCompileInfo(32, 1024);
    auto para = MakePara({8, 64}, {2, 4, 64}, {1, 8}, {1, 8}, {1, 8}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, ub_larger_than_kernel_length_type_fails)
{
    const uint64_t oversizedUb = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
    auto compileInfo = MakeCompileInfo(32, oversizedUb);
    auto para = MakePara({8, 64}, {2, 4, 64}, {1, 8}, {1, 8}, {1, 8}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesTilingTest, cut_h_chunk_limit_boundary_succeeds)
{
    constexpr int64_t hFactorForMockedPlatform = 15936;
    constexpr int64_t maxChunkCount = 1024;
    const int64_t hidden = hFactorForMockedPlatform * maxChunkCount;
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({1, hidden}, {1, 1, hidden}, {1, 1}, {1, 1}, {1, 1}, ge::DT_FLOAT, &compileInfo);
    ExpectSuccess(para, optiling::MTCG_TILING_KEY_CUT_H);
}

TEST_F(MoeTutelCombineGatesTilingTest, cut_h_chunk_limit_overflow_fails)
{
    constexpr int64_t hFactorForMockedPlatform = 15936;
    constexpr int64_t maxChunkCount = 1024;
    const int64_t hidden = hFactorForMockedPlatform * maxChunkCount + 1;
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara({1, hidden}, {1, 1, hidden}, {1, 1}, {1, 1}, {1, 1}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
