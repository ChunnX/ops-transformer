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
 * \brief arch35 tiling UT：模板选择 + 各条非法输入的失败路径。
 */

#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "../../../op_host/moe_tutel_combine_gates_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

namespace {
const std::string kA5SocInfo = "{\n"
                               "  \"hardware_info\": {\n"
                               "    \"UB_SIZE\": 262144,\n"
                               "    \"CORE_NUM\": 32,\n"
                               "    \"socVersion\": \"Ascend910_95\"\n"
                               "  }\n"
                               "}";

constexpr size_t SYS_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;

// x[S,H] / y_grad[E,C,H] / indices[E,S] / locations[E,S] -> gates_grad[E,S]
gert::TilingContextPara MakePara(const vector<int64_t> &xShape, const vector<int64_t> &yShape,
                                 const vector<int64_t> &indicesShape, const vector<int64_t> &locationsShape,
                                 const vector<int64_t> &outShape, ge::DataType floatType,
                                 optiling::MoeTutelCombineGatesCompileInfo *compileInfo,
                                 ge::DataType yType = ge::DT_UNDEFINED)
{
    const ge::DataType realYType = (yType == ge::DT_UNDEFINED) ? floatType : yType;
    return gert::TilingContextPara("MoeTutelCombineGates",
                                   {
                                       {{xShape, xShape}, floatType, ge::FORMAT_ND},
                                       {{yShape, yShape}, realYType, ge::FORMAT_ND},
                                       {{indicesShape, indicesShape}, ge::DT_INT32, ge::FORMAT_ND},
                                       {{locationsShape, locationsShape}, ge::DT_INT32, ge::FORMAT_ND},
                                   },
                                   {
                                       {{outShape, outShape}, floatType, ge::FORMAT_ND},
                                   },
                                   {}, compileInfo, "Ascend950", kA5SocInfo, 4096);
}
} // namespace

class MoeTutelCombineGatesTilingTest : public testing::Test {};

// H 可全载 -> NOT_CUT_H
TEST_F(MoeTutelCombineGatesTilingTest, not_cut_h_float32)
{
    optiling::MoeTutelCombineGatesCompileInfo compileInfo = {32, 262144, 256, 32};
    auto para = MakePara({8, 64}, {2, 4, 64}, {2, 8}, {2, 8}, {2, 8}, ge::DT_FLOAT, &compileInfo);
    std::vector<size_t> expectWorkspaces = {SYS_WORKSPACE_SIZE};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, optiling::MTCG_TILING_KEY_NOT_CUT_H, "", expectWorkspaces);
}

TEST_F(MoeTutelCombineGatesTilingTest, not_cut_h_float16)
{
    optiling::MoeTutelCombineGatesCompileInfo compileInfo = {32, 262144, 256, 32};
    auto para = MakePara({100, 1024}, {4, 16, 1024}, {4, 100}, {4, 100}, {4, 100}, ge::DT_FLOAT16, &compileInfo);
    std::vector<size_t> expectWorkspaces = {SYS_WORKSPACE_SIZE};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, optiling::MTCG_TILING_KEY_NOT_CUT_H, "", expectWorkspaces);
}

TEST_F(MoeTutelCombineGatesTilingTest, not_cut_h_bfloat16)
{
    optiling::MoeTutelCombineGatesCompileInfo compileInfo = {32, 262144, 256, 32};
    auto para = MakePara({7, 33}, {3, 5, 33}, {3, 7}, {3, 7}, {3, 7}, ge::DT_BF16, &compileInfo);
    std::vector<size_t> expectWorkspaces = {SYS_WORKSPACE_SIZE};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, optiling::MTCG_TILING_KEY_NOT_CUT_H, "", expectWorkspaces);
}

// H 超 UB 预算 -> CUT_H（fp32 下 H = 65536 需要 512KB/行，单核 UB 只有 256KB）
TEST_F(MoeTutelCombineGatesTilingTest, cut_h_float32_large_hidden)
{
    optiling::MoeTutelCombineGatesCompileInfo compileInfo = {32, 262144, 256, 32};
    auto para = MakePara({4, 65536}, {2, 3, 65536}, {2, 4}, {2, 4}, {2, 4}, ge::DT_FLOAT, &compileInfo);
    std::vector<size_t> expectWorkspaces = {SYS_WORKSPACE_SIZE};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, optiling::MTCG_TILING_KEY_CUT_H, "", expectWorkspaces);
}

// 空 Tensor：S == 0
TEST_F(MoeTutelCombineGatesTilingTest, empty_samples)
{
    optiling::MoeTutelCombineGatesCompileInfo compileInfo = {32, 262144, 256, 32};
    auto para = MakePara({0, 64}, {2, 4, 64}, {2, 0}, {2, 0}, {2, 0}, ge::DT_FLOAT, &compileInfo);
    std::vector<size_t> expectWorkspaces = {SYS_WORKSPACE_SIZE};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, optiling::MTCG_TILING_KEY_NOT_CUT_H, "", expectWorkspaces);
}

// 空 Tensor：E == 0（专家数为 0，输出元素数为 0）
TEST_F(MoeTutelCombineGatesTilingTest, empty_experts)
{
    optiling::MoeTutelCombineGatesCompileInfo compileInfo = {32, 262144, 256, 32};
    auto para = MakePara({8, 64}, {0, 4, 64}, {0, 8}, {0, 8}, {0, 8}, ge::DT_FLOAT, &compileInfo);
    std::vector<size_t> expectWorkspaces = {SYS_WORKSPACE_SIZE};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, optiling::MTCG_TILING_KEY_NOT_CUT_H, "", expectWorkspaces);
}

// 空 Tensor：C == 0（capacity 为 0，任何 location 都越界 → 全部路由非法）
TEST_F(MoeTutelCombineGatesTilingTest, empty_capacity)
{
    optiling::MoeTutelCombineGatesCompileInfo compileInfo = {32, 262144, 256, 32};
    auto para = MakePara({8, 64}, {2, 0, 64}, {2, 8}, {2, 8}, {2, 8}, ge::DT_FLOAT, &compileInfo);
    std::vector<size_t> expectWorkspaces = {SYS_WORKSPACE_SIZE};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, optiling::MTCG_TILING_KEY_NOT_CUT_H, "", expectWorkspaces);
}

// 空内积：H == 0
TEST_F(MoeTutelCombineGatesTilingTest, zero_hidden)
{
    optiling::MoeTutelCombineGatesCompileInfo compileInfo = {32, 262144, 256, 32};
    auto para = MakePara({8, 0}, {2, 4, 0}, {2, 8}, {2, 8}, {2, 8}, ge::DT_FLOAT, &compileInfo);
    std::vector<size_t> expectWorkspaces = {SYS_WORKSPACE_SIZE};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, optiling::MTCG_TILING_KEY_NOT_CUT_H, "", expectWorkspaces);
}

// K != E：canndev 在此会对 gates_grad 越界写，950 判非法。
TEST_F(MoeTutelCombineGatesTilingTest, invalid_route_axis_not_equal_experts)
{
    optiling::MoeTutelCombineGatesCompileInfo compileInfo = {32, 262144, 256, 32};
    auto para = MakePara({8, 64}, {2, 4, 64}, {3, 8}, {3, 8}, {3, 8}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// indices 与 locations shape 不一致
TEST_F(MoeTutelCombineGatesTilingTest, invalid_indices_locations_shape_mismatch)
{
    optiling::MoeTutelCombineGatesCompileInfo compileInfo = {32, 262144, 256, 32};
    auto para = MakePara({8, 64}, {2, 4, 64}, {2, 8}, {2, 4}, {2, 8}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// x.shape[1] != y_grad.shape[2]
TEST_F(MoeTutelCombineGatesTilingTest, invalid_hidden_mismatch)
{
    optiling::MoeTutelCombineGatesCompileInfo compileInfo = {32, 262144, 256, 32};
    auto para = MakePara({8, 64}, {2, 4, 32}, {2, 8}, {2, 8}, {2, 8}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// rank 不符：y_grad 必须是 3 维
TEST_F(MoeTutelCombineGatesTilingTest, invalid_rank)
{
    optiling::MoeTutelCombineGatesCompileInfo compileInfo = {32, 262144, 256, 32};
    auto para = MakePara({8, 64}, {2, 64}, {2, 8}, {2, 8}, {2, 8}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// 混合浮点 dtype：x 为 fp32、y_grad 为 fp16
TEST_F(MoeTutelCombineGatesTilingTest, invalid_mixed_float_dtype)
{
    optiling::MoeTutelCombineGatesCompileInfo compileInfo = {32, 262144, 256, 32};
    auto para = MakePara({8, 64}, {2, 4, 64}, {2, 8}, {2, 8}, {2, 8}, ge::DT_FLOAT, &compileInfo, ge::DT_FLOAT16);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
