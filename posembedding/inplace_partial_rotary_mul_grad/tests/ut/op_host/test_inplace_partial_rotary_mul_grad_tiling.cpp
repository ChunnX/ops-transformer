/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/inplace_partial_rotary_mul_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class InplacePartialRotaryMulGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "InplacePartialRotaryMulGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "InplacePartialRotaryMulGradTiling TearDown" << std::endl;
    }

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 48}
                                    })";
};

TEST_F(InplacePartialRotaryMulGradTiling, tiling_fp16_mode0_half)
{
    optiling::InplacePartialRotaryMulGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InplacePartialRotaryMulGrad",
                                              {
                                                  // input info: dy, cos, sin
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info: dyOut, dcosOut, dsinOut
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attrs
                                                  {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 64})},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "1 1 1 64 0 64 0 1 1 0 0 0 1 1 1 1";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(InplacePartialRotaryMulGradTiling, tiling_fp16_mode1_interleave)
{
    optiling::InplacePartialRotaryMulGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InplacePartialRotaryMulGrad",
                                              {
                                                  // input info: dy, cos, sin
                                                  {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attrs
                                                  {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                  {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 128})},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "2 4 8 128 0 128 1 1 1 0 1 0 32 2 2 0";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(InplacePartialRotaryMulGradTiling, tiling_fp32_mode2_quarter)
{
    optiling::InplacePartialRotaryMulGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InplacePartialRotaryMulGrad",
                                              {
                                                  {{{2, 64, 4, 64}, {2, 64, 4, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 1, 1, 64}, {2, 1, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 1, 1, 64}, {2, 1, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 64, 4, 64}, {2, 64, 4, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 1, 1, 64}, {2, 1, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 1, 1, 64}, {2, 1, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                  {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 64})},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "2 64 4 64 0 64 2 0 0 1 0 0 16 32 4 0";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(InplacePartialRotaryMulGradTiling, tiling_fp32_mode3_interleave_half)
{
    optiling::InplacePartialRotaryMulGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InplacePartialRotaryMulGrad",
                                              {
                                                  {{{1, 32, 2, 64}, {1, 32, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 32, 1, 64}, {1, 32, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 32, 1, 64}, {1, 32, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 32, 2, 64}, {1, 32, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 32, 1, 64}, {1, 32, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 32, 1, 64}, {1, 32, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                  {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 64})},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "1 32 2 64 0 64 3 0 0 0 1 0 2 32 2 0";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(InplacePartialRotaryMulGradTiling, tiling_mixed_dtype_fp16_dy_fp32_cos)
{
    optiling::InplacePartialRotaryMulGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InplacePartialRotaryMulGrad",
                                              {
                                                  {{{2, 8, 4, 128}, {2, 8, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{2, 8, 1, 128}, {2, 8, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 8, 1, 128}, {2, 8, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 8, 4, 128}, {2, 8, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{2, 8, 1, 128}, {2, 8, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 8, 1, 128}, {2, 8, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 128})},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "2 8 4 128 0 128 0 1 0 0 1 0 2 32 2 0";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Error case: x is not nullptr (unsupported)
TEST_F(InplacePartialRotaryMulGradTiling, tiling_error_x_not_null)
{
    optiling::InplacePartialRotaryMulGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InplacePartialRotaryMulGrad",
                                              {
                                                  // input info: dy, cos, sin, x (4 inputs!)
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 64})},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Error case: partial_slice out of range
TEST_F(InplacePartialRotaryMulGradTiling, tiling_error_bad_slice)
{
    optiling::InplacePartialRotaryMulGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InplacePartialRotaryMulGrad",
                                              {
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 32}, {1, 1, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 32}, {1, 1, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 32}, {1, 1, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 32}, {1, 1, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 128})},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Error case: cos D != sliceLen
TEST_F(InplacePartialRotaryMulGradTiling, tiling_error_cos_dim_mismatch)
{
    optiling::InplacePartialRotaryMulGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InplacePartialRotaryMulGrad",
                                              {
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 32}, {1, 1, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 32}, {1, 1, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 32}, {1, 1, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 32}, {1, 1, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 64})},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
