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
 * \file test_masked_causal_conv1d_tiling.cpp
 * \brief Unit tests for MaskedCausalConv1d tiling logic
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/masked_causal_conv1d_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"


using namespace std;

class MaskedCausalConv1dTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MaskedCausalConv1dTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MaskedCausalConv1dTiling TearDown" << std::endl;
    }
};

// Test 1: BF16 baseline, S=2048 B=4 H=768
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_bf16_s2048_b4_h768)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 2048, B = 4, H = 768, W = 3;

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{W, H}, {W, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);
    int64_t expectTilingKey = 10000;
    std::string expectTilingData =
        "2048 4 768 8 4 128 64 4 0 1 1 2 0 1024 1024 64 1 480 2 64 1 1 3 64 1 64 1 1 3 64 64 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test 2: FP16 dtype → GetTilingKey returns TILING_KEY_FP16 (10001)
// Shape S=64 B=1 H=64 used to keep the test lightweight; tiling data differs from BF16
// only in the key, confirming the dtype→key mapping.
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_fp16_min_shape)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 64, B = 1, H = 64, W = 3;

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{W, H}, {W, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    int64_t expectTilingKey = 10001;
    std::string expectTilingData = "64 1 64 1 0 64 64 1 0 1 1 64 0 1 1 64 1 1 1 64 1 1 1 1 1 64 1 1 1 1 64 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test 3: Mask absent (empty shape) → isMaskNone_ = 1
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_bf16_mask_none_s1024_b2_h128)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 1024, B = 2, H = 128, W = 3;

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{W, H}, {W, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND}, // absent optional mask
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    int64_t expectTilingKey = 10000;
    std::string expectTilingData = "1024 2 128 2 0 64 64 2 0 1 1 16 0 64 64 64 1 64 1 64 1 1 1 64 1 64 1 1 1 64 64 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test 4: H not divisible by H_REG=64 → CheckInputParams GRAPH_FAILED
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_fail_h_not_aligned)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 1024, B = 2, H = 100, W = 3; // H=100 not multiple of 64

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{W, H}, {W, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test 5: Weight K != CONV_WINDOW_SIZE=3 → CheckInputParams GRAPH_FAILED
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_fail_wrong_weight_k)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 1024, B = 2, H = 128, W = 4; // W=4 != 3

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{W, H}, {W, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test 6: Unsupported dtype FP32 → GetInputDtypes GRAPH_FAILED
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_fail_invalid_dtype_fp32)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 1024, B = 2, H = 128, W = 3;

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{W, H}, {W, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test 7: Minimal valid shape S=64 B=1 H=64 (exactly H_REG)
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_bf16_min_shape_s64_b1_h64)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 64, B = 1, H = 64, W = 3;

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{W, H}, {W, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    int64_t expectTilingKey = 10000;
    std::string expectTilingData = "64 1 64 1 0 64 64 1 0 1 1 64 0 1 1 64 1 1 1 64 1 1 1 1 1 64 1 1 1 1 64 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test 8: Situation A + hUb expansion
// H=8192 → hCoreCnt_=64, hBlockFactor_=128; small BS fits at hUb=64,
// then the k-loop expands ubFactorH_ to 128.
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_bf16_situation_a_hexpand_s32_b1_h8192)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 32, B = 1, H = 8192, W = 3;

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{W, H}, {W, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    int64_t expectTilingKey = 10000;
    std::string expectTilingData =
        "32 1 8192 64 0 128 128 1 0 1 1 1 0 32 32 128 1 32 1 128 1 1 1 32 1 128 1 1 1 32 64 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test 9: Situation B – reduce B
// coreNum=1 → bBlockFactor_=3, sBlockFactor_=64; ubSize=65536 (net 57344):
//   calcUsedUb(64,3,64)=68608 > 57344 → Situation B
//   bUb=2: calcUsedUb(64,2,64)=50176 ≤ 57344 → reduce-B branch taken
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_bf16_situation_b_reduce_b)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 64, B = 3, H = 64, W = 3;

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{W, H}, {W, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    int64_t expectTilingKey = 10000;
    std::string expectTilingData = "64 3 64 1 0 64 64 2 1 2 1 32 0 2 2 64 2 2 1 64 1 2 1 2 1 64 1 1 1 2 64 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test 10: Situation B – cut S
// coreNum=1 → bBlockFactor_=1, sBlockFactor_=256; ubSize=65536 (net 57344):
//   calcUsedUb(64,1,256)=133376 > 57344 → Situation B, bUb already 1
//   → findSUbMax(64,1) → ubFactorS_ < 256 (cut S path)
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_bf16_situation_b_cut_s)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 256, B = 1, H = 64, W = 3;

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{W, H}, {W, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    int64_t expectTilingKey = 10000;
    std::string expectTilingData = "256 1 64 1 0 64 64 1 0 1 1 64 0 4 4 64 1 4 1 64 1 1 1 4 1 64 1 1 1 4 64 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test 11: Irregular core split – non-zero sMainCnt_ and tailBlockLoop* params
// S=101, B=6, H=128, coreNum=64 → sCoreCnt_=5, sBlockFactor_=21, sBlockTailFactor_=20
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_bf16_irregular_split_s101_b6_h128)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 101, B = 6, H = 128, W = 3;

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{W, H}, {W, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    int64_t expectTilingKey = 10000;
    std::string expectTilingData = "101 6 128 2 0 64 64 4 2 2 1 8 5 13 12 64 2 13 1 64 1 2 1 13 1 64 1 1 1 12 64 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test 12: x is 2-D (not 3-D) → GetInputShapes GRAPH_FAILED
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_fail_x_not_3d)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 1024, H = 128, W = 3;

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, H}, {S, H}}, ge::DT_BF16, ge::FORMAT_ND}, // 2-D, should be 3-D
                                                  {{{W, H}, {W, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{S, H}, {S, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test 13: weight dtype != x dtype (x=BF16, weight=FP16) → GetInputDtypes GRAPH_FAILED
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_fail_weight_dtype_mismatch)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 1024, B = 2, H = 128, W = 3;

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{W, H}, {W, H}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // FP16 != BF16
                                                  {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test 14: mask dtype != DT_BOOL (mask=FP16) → GetInputDtypes GRAPH_FAILED
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_fail_mask_dtype_not_bool)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 1024, B = 2, H = 128, W = 3;

    gert::TilingContextPara tilingContextPara(
        "MaskedCausalConv1d",
        {
            {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{W, H}, {W, H}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // should be DT_BOOL
        },
        {
            {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        attrs, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test 15: weight is 1-D (not 2-D) → CheckInputParams GRAPH_FAILED
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_fail_weight_not_2d)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 1024, B = 2, H = 128;

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{H}, {H}}, ge::DT_BF16, ge::FORMAT_ND}, // 1-D, should be 2-D [K, H]
                                                  {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test 16:
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_fail_weight_h_mismatch)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 1024, B = 2, H = 128, W = 3;

    gert::TilingContextPara tilingContextPara(
        "MaskedCausalConv1d",
        {
            {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{W, 256}, {W, 256}}, ge::DT_BF16, ge::FORMAT_ND}, // weight H=256 != x H=128
            {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        attrs, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test 17: mask shape != x shape (mask B or S mismatch) → CheckInputParams GRAPH_FAILED
// Uses mask B=3 while x B=2; the S-mismatch branch is symmetric and not separately tested.
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_fail_mask_shape_mismatch)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 1024, B = 2, H = 128, W = 3;

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{W, H}, {W, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{3, S}, {3, S}}, ge::DT_BOOL, ge::FORMAT_ND}, // mask B=3 != x B=2
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}
