/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "../../../op_host/lrn_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

// Ascend950 SoC info. UB_SIZE = 253952 B is the real a5 vector-core UB span the
// tiling feasibility check compares against (worst tier/dtype need 229376 B).
std::string LRN_tiling_A5SocInfo =
    "{\n"
    "  \"hardware_info\": {\n"
    "    \"BT_SIZE\": 0,\n"
    "    \"load3d_constraints\": \"1\",\n"
    "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
    "    \"Intrinsic_data_move_l12ub\": true,\n"
    "    \"Intrinsic_data_move_l0c2ub\": true,\n"
    "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
    "    \"UB_SIZE\": 253952,\n"
    "    \"L2_SIZE\": 201326592,\n"
    "    \"L1_SIZE\": 524288,\n"
    "    \"L0A_SIZE\": 65536,\n"
    "    \"L0B_SIZE\": 65536,\n"
    "    \"L0C_SIZE\": 131072,\n"
    "    \"vector_core_cnt\": 40,\n"
    "    \"cube_core_cnt\": 20,\n"
    "    \"socVersion\": \"Ascend950\"\n"
    "  }\n"
    "}";

// ====================================================================
// Tiling key encoding (generated LRNTilingKey_tilingkey.h; the host packs
// the ACTUAL candidate values via GET_TPL_TILING_KEY, which encodes each
// field's candidate INDEX in declaration order from bit 0):
//   HwTile: 2 bits, values [128]              -> index 0 (bits 1-0)
//   CChunk: 2 bits, values [64]               -> index 0 (bits 3-2)
//   RTier:  3 bits, values [0,1,2,4,8,16,32,48] -> index j (bits 6-4)
//   key = j << 4
// The host selects tier = smallest candidate >= depth_radius (lrn_tiling.cpp):
//   r=0        -> tier 0  -> j=0 -> key   0
//   r=1        -> tier 1  -> j=1 -> key  16
//   r=2        -> tier 2  -> j=2 -> key  32
//   r=3..4     -> tier 4  -> j=3 -> key  48
//   r=5..8     -> tier 8  -> j=4 -> key  64
//   r=9..16    -> tier 16 -> j=5 -> key  80
//   r=17..32   -> tier 32 -> j=6 -> key  96
//   r=33..48   -> tier 48 -> j=7 -> key 112
// ====================================================================

constexpr uint64_t SKIP_TILING_KEY = UINT64_MAX;

class LRNTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "LRNTilingTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "LRNTilingTest TearDown" << std::endl;
    }
};

// ====================================================================
// Success cases — one per RTier candidate (keys 0/16/32/48/64/80/96/112)
// ====================================================================

// fp16, r=2 -> tier 2 -> key 32; C=64 (one c_chunk), HW=128 (one hw_tile)
TEST_F(LRNTilingTest, fp16_r2_tier2)
{
    optiling::LRNCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "LRN",
        {
            {{{2, 64, 8, 16}, {2, 64, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // x
        },
        {
            {{{2, 64, 8, 16}, {2, 64, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // y
        },
        {
            {"depth_radius", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"bias", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"alpha", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
            {"norm_region", Ops::Transformer::AnyValue::CreateFrom<std::string>("ACROSS_CHANNELS")},
        },
        &compileInfo, "Ascend950", LRN_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 32);
}

// fp32, r=5 -> tier 8 -> key 64; C=128 (two c_chunks), HW=128
TEST_F(LRNTilingTest, fp32_r5_tier8)
{
    optiling::LRNCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "LRN",
        {
            {{{1, 128, 4, 32}, {1, 128, 4, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 128, 4, 32}, {1, 128, 4, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"depth_radius", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"bias", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"alpha", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
            {"norm_region", Ops::Transformer::AnyValue::CreateFrom<std::string>("ACROSS_CHANNELS")},
        },
        &compileInfo, "Ascend950", LRN_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 64);
}

// fp16, r=0 -> tier 0 -> key 0; HW=6 < hw_tile=128 (column tail block)
TEST_F(LRNTilingTest, fp16_r0_tier0_hw_tail)
{
    optiling::LRNCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "LRN",
        {
            {{{1, 8, 2, 3}, {1, 8, 2, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 8, 2, 3}, {1, 8, 2, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"depth_radius", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"bias", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"alpha", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
            {"norm_region", Ops::Transformer::AnyValue::CreateFrom<std::string>("ACROSS_CHANNELS")},
        },
        &compileInfo, "Ascend950", LRN_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

// fp32, r=48 (contract max) -> tier 48 -> key 112; C=100 (tail c_chunk of 36
// rows), HW=35 (column tail). Worst-case UB span (fp32, tier48):
// (2*160 + 2*64) * 128 * 4 = 229376 B <= UB 253952 B.
TEST_F(LRNTilingTest, fp32_r48_tier48_worst_ub)
{
    optiling::LRNCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "LRN",
        {
            {{{1, 100, 5, 7}, {1, 100, 5, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 100, 5, 7}, {1, 100, 5, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"depth_radius", Ops::Transformer::AnyValue::CreateFrom<int64_t>(48)},
            {"bias", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"alpha", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
            {"norm_region", Ops::Transformer::AnyValue::CreateFrom<std::string>("ACROSS_CHANNELS")},
        },
        &compileInfo, "Ascend950", LRN_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 112);
}

// fp16, r=33 -> tier 48 -> key 112; C=65 (tail c_chunk), HW=130 (2 hw blocks,
// second one a 2-column tail), N=3 -> totalTasks = 3*2*2 = 12
TEST_F(LRNTilingTest, fp16_r33_tier48_both_tails)
{
    optiling::LRNCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "LRN",
        {
            {{{3, 65, 1, 130}, {3, 65, 1, 130}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{3, 65, 1, 130}, {3, 65, 1, 130}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"depth_radius", Ops::Transformer::AnyValue::CreateFrom<int64_t>(33)},
            {"bias", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"alpha", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
            {"norm_region", Ops::Transformer::AnyValue::CreateFrom<std::string>("ACROSS_CHANNELS")},
        },
        &compileInfo, "Ascend950", LRN_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 112);
}

// fp32, r=32 -> tier 32 -> key 96; C=200, HW=9 (column tail)
TEST_F(LRNTilingTest, fp32_r32_tier32)
{
    optiling::LRNCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "LRN",
        {
            {{{2, 200, 3, 3}, {2, 200, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 200, 3, 3}, {2, 200, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"depth_radius", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"bias", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"alpha", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
            {"norm_region", Ops::Transformer::AnyValue::CreateFrom<std::string>("ACROSS_CHANNELS")},
        },
        &compileInfo, "Ascend950", LRN_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 96);
}

// fp16, r=16 -> tier 16 -> key 80; C=300 (5 c_chunks), HW=4
TEST_F(LRNTilingTest, fp16_r16_tier16)
{
    optiling::LRNCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "LRN",
        {
            {{{1, 300, 2, 2}, {1, 300, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 300, 2, 2}, {1, 300, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"depth_radius", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"bias", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"alpha", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
            {"norm_region", Ops::Transformer::AnyValue::CreateFrom<std::string>("ACROSS_CHANNELS")},
        },
        &compileInfo, "Ascend950", LRN_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 80);
}

// fp16, r=3 -> tier 4 -> key 48 (tier ladder: r=3,4 both select tier 4)
TEST_F(LRNTilingTest, fp16_r3_tier4)
{
    optiling::LRNCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "LRN",
        {
            {{{1, 300, 2, 2}, {1, 300, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 300, 2, 2}, {1, 300, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"depth_radius", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"bias", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"alpha", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
            {"norm_region", Ops::Transformer::AnyValue::CreateFrom<std::string>("ACROSS_CHANNELS")},
        },
        &compileInfo, "Ascend950", LRN_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 48);
}

// fp32, r=1 -> tier 1 -> key 16; minimal 1x1x1x1 tensor
TEST_F(LRNTilingTest, fp32_r1_tier1_minimal)
{
    optiling::LRNCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "LRN",
        {
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"depth_radius", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"bias", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"alpha", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
            {"norm_region", Ops::Transformer::AnyValue::CreateFrom<std::string>("ACROSS_CHANNELS")},
        },
        &compileInfo, "Ascend950", LRN_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 16);
}

// ====================================================================
// Error cases — the tiling fail-closes with ge::GRAPH_PARAM_INVALID
// ====================================================================

// depth_radius = -1 (must be in [0, 48])
TEST_F(LRNTilingTest, err_depth_radius_negative)
{
    optiling::LRNCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "LRN",
        {
            {{{2, 64, 8, 16}, {2, 64, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 8, 16}, {2, 64, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"depth_radius", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"bias", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"alpha", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
            {"norm_region", Ops::Transformer::AnyValue::CreateFrom<std::string>("ACROSS_CHANNELS")},
        },
        &compileInfo, "Ascend950", LRN_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_PARAM_INVALID, SKIP_TILING_KEY);
}

// depth_radius = 49 (above the contract upper bound 48)
TEST_F(LRNTilingTest, err_depth_radius_over_max)
{
    optiling::LRNCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "LRN",
        {
            {{{2, 64, 8, 16}, {2, 64, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 8, 16}, {2, 64, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"depth_radius", Ops::Transformer::AnyValue::CreateFrom<int64_t>(49)},
            {"bias", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"alpha", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
            {"norm_region", Ops::Transformer::AnyValue::CreateFrom<std::string>("ACROSS_CHANNELS")},
        },
        &compileInfo, "Ascend950", LRN_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_PARAM_INVALID, SKIP_TILING_KEY);
}

// bias = 0 (must be > 0 so that norm > 0 for the ln path)
TEST_F(LRNTilingTest, err_bias_zero)
{
    optiling::LRNCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "LRN",
        {
            {{{2, 64, 8, 16}, {2, 64, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 8, 16}, {2, 64, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"depth_radius", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"bias", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
            {"alpha", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
            {"norm_region", Ops::Transformer::AnyValue::CreateFrom<std::string>("ACROSS_CHANNELS")},
        },
        &compileInfo, "Ascend950", LRN_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_PARAM_INVALID, SKIP_TILING_KEY);
}

// alpha = -1.0 (must be > 0)
TEST_F(LRNTilingTest, err_alpha_negative)
{
    optiling::LRNCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "LRN",
        {
            {{{2, 64, 8, 16}, {2, 64, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 8, 16}, {2, 64, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"depth_radius", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"bias", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"alpha", Ops::Transformer::AnyValue::CreateFrom<float>(-1.0f)},
            {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
            {"norm_region", Ops::Transformer::AnyValue::CreateFrom<std::string>("ACROSS_CHANNELS")},
        },
        &compileInfo, "Ascend950", LRN_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_PARAM_INVALID, SKIP_TILING_KEY);
}

// norm_region = "WITHIN_CHANNELS" (only ACROSS_CHANNELS is supported)
TEST_F(LRNTilingTest, err_norm_region_within_channels)
{
    optiling::LRNCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "LRN",
        {
            {{{2, 64, 8, 16}, {2, 64, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 8, 16}, {2, 64, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"depth_radius", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"bias", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"alpha", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
            {"norm_region", Ops::Transformer::AnyValue::CreateFrom<std::string>("WITHIN_CHANNELS")},
        },
        &compileInfo, "Ascend950", LRN_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_PARAM_INVALID, SKIP_TILING_KEY);
}

// x is 3-D (must be 4-D NCHW)
TEST_F(LRNTilingTest, err_x_rank3)
{
    optiling::LRNCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "LRN",
        {
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"depth_radius", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"bias", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"alpha", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
            {"norm_region", Ops::Transformer::AnyValue::CreateFrom<std::string>("ACROSS_CHANNELS")},
        },
        &compileInfo, "Ascend950", LRN_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_PARAM_INVALID, SKIP_TILING_KEY);
}

// x dtype = BF16 (only FLOAT16 / FLOAT are supported)
TEST_F(LRNTilingTest, err_x_dtype_bf16)
{
    optiling::LRNCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "LRN",
        {
            {{{2, 64, 8, 16}, {2, 64, 8, 16}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 8, 16}, {2, 64, 8, 16}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"depth_radius", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"bias", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"alpha", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
            {"norm_region", Ops::Transformer::AnyValue::CreateFrom<std::string>("ACROSS_CHANNELS")},
        },
        &compileInfo, "Ascend950", LRN_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_PARAM_INVALID, SKIP_TILING_KEY);
}

// SoC version = Ascend910B (the kernel is a regbase-arch a5 delivery)
TEST_F(LRNTilingTest, err_soc_910b)
{
    optiling::LRNCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "LRN",
        {
            {{{2, 64, 8, 16}, {2, 64, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 8, 16}, {2, 64, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"depth_radius", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"bias", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"alpha", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
            {"norm_region", Ops::Transformer::AnyValue::CreateFrom<std::string>("ACROSS_CHANNELS")},
        },
        &compileInfo, "Ascend910B", LRN_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_PARAM_INVALID, SKIP_TILING_KEY);
}
