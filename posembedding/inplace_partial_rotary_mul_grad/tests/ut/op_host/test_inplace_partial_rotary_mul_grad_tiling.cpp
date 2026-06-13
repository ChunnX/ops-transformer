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
#include "../../../op_host/inplace_partial_rotary_mul_grad_regbase_tiling.h"
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

// ============================================================================
// 同精度 — 4 个 mode 全覆盖
// ============================================================================

// mode=0 half, layout=NO_BROADCAST → AAndB → TILING_KEY_A(205)
TEST_F(InplacePartialRotaryMulGradTiling, tiling_fp16_mode0_half_nobroadcast)
{
    optiling::InplacePartialRotaryMulGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InplacePartialRotaryMulGrad",
                                              {
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
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 205);
}

// mode=1 interleave, layout=BSND(cosb_==1) → BAB → TILING_KEY_BAB(203)
TEST_F(InplacePartialRotaryMulGradTiling, tiling_fp16_mode1_interleave_bsnd)
{
    optiling::InplacePartialRotaryMulGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InplacePartialRotaryMulGrad",
                                              {
                                                  {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                  {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 128})},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 203);
}

// mode=2 quarter, layout=BROADCAST_BSN → AAndB(B) → TILING_KEY_B(206)
TEST_F(InplacePartialRotaryMulGradTiling, tiling_fp32_mode2_quarter_broadcast)
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
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 206);
}

// mode=3 interleave_half, layout=SBND → AB → TILING_KEY_AB(204)
TEST_F(InplacePartialRotaryMulGradTiling, tiling_fp32_mode3_interleavehalf_sbnd)
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
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 204);
}

// ============================================================================
// 混合精度 — 两种方向全覆盖
// ============================================================================

// dy=fp16 + cos=fp32, mode=0, layout=SBND → AB → TILING_KEY_AB(204)
TEST_F(InplacePartialRotaryMulGradTiling, tiling_mixed_fp16dy_fp32cos)
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
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 204);
}

// dy=fp32 + cos=fp16, mode=0, layout=BSND(cosb_==1) → BAB → TILING_KEY_BAB(203)
TEST_F(InplacePartialRotaryMulGradTiling, tiling_mixed_fp32dy_fp16cos)
{
    optiling::InplacePartialRotaryMulGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InplacePartialRotaryMulGrad",
                                              {
                                                  {{{1, 4, 8, 128}, {1, 4, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 4, 8, 128}, {1, 4, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 128})},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 203);
}

// dy=fp32 + cos=bf16, mode=1, layout=BROADCAST_BSN → AAndB(B) → TILING_KEY_B(206)
TEST_F(InplacePartialRotaryMulGradTiling, tiling_mixed_fp32dy_bf16cos)
{
    optiling::InplacePartialRotaryMulGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InplacePartialRotaryMulGrad",
                                              {
                                                  {{{2, 8, 4, 64}, {2, 8, 4, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 8, 4, 64}, {2, 8, 4, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                  {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 64})},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 206);
}

// ============================================================================
// 错误 case
// ============================================================================

// x 不为 nullptr — CheckShapeLimit 设置 dCosFlag_=1，校验 x shape == dy shape
TEST_F(InplacePartialRotaryMulGradTiling, tiling_error_x_with_shape_mismatch)
{
    optiling::InplacePartialRotaryMulGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InplacePartialRotaryMulGrad",
                                              {
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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

// partial_slice end-start != cos D — CheckAttr 会报错
TEST_F(InplacePartialRotaryMulGradTiling, tiling_error_slice_mismatch_cos_d)
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

// cos D > dy D — CheckShape 会报错
TEST_F(InplacePartialRotaryMulGradTiling, tiling_error_cos_d_larger_than_dy_d)
{
    optiling::InplacePartialRotaryMulGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InplacePartialRotaryMulGrad",
                                              {
                                                  {{{1, 1, 1, 32}, {1, 1, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 1, 1, 32}, {1, 1, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
