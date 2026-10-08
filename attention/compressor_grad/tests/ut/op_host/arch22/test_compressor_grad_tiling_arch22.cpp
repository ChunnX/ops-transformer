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
#include <iostream>

#include "../../../../op_host/compressor_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

// Ascend910_93 (A3 / arch22) SoC info: 24 cube cores, 48 vector cores, UB 192KB
std::string CompressorGrad_tiling_A3SocInfo =
    "{\n"
    "  \"hardware_info\": {\n"
    "    \"BT_SIZE\": 0,\n"
    "    \"load3d_constraints\": \"1\",\n"
    "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
    "    \"Intrinsic_data_move_l12ub\": true,\n"
    "    \"Intrinsic_data_move_l0c2ub\": true,\n"
    "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
    "    \"4096\": 196608,\n"
    "    \"L2_SIZE\": 201326592,\n"
    "    \"L1_SIZE\": 524288,\n"
    "    \"L0A_SIZE\": 65536,\n"
    "    \"L0B_SIZE\": 65536,\n"
    "    \"L0C_SIZE\": 131072,\n"
    "    \"vector_core_cnt\": 48,\n"
    "    \"cube_core_cnt\": 24,\n"
    "    \"socVersion\": \"Ascend910_93\"\n"
    "  }\n"
    "}";

// ====================================================================
// arch22 tiling key space (op_kernel/arch22/compressor_grad_template_tiling_key.h):
//   single field D_TYPE (2bit, LIST {0, 1}): 0 = BF16, 1 = FP16
//   GET_TPL_TILING_KEY encodes the list index, so key == dtype code.
// ====================================================================

class CompressorGradTilingArch22Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CompressorGradTilingArch22Test SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "CompressorGradTilingArch22Test TearDown" << std::endl;
    }
};

// ====================================================================
// BSH Layout Tiling Tests
// ====================================================================

// C4A bf16: B=2, S=8, H=4096, D=512, coff=2, cmp_ratio=4
TEST_F(CompressorGradTilingArch22Test, bsh_c4a_bf16)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            // Required inputs
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},        // x
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},        // wkv
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},        // wgate
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},          // d_cmp_kv
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},   // softmax_score
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},   // kv
            // Optional inputs (empty for BSH)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                           // cu_seqlens
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                           // seqused
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                           // start_pos
        },
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},        // d_x
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},        // d_wkv
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},        // d_wgate
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},             // d_ape
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    int64_t expectTilingKey = 0; // BF16
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// C4A fp16: B=2, S=8, H=4096, D=512, coff=2, cmp_ratio=4
TEST_F(CompressorGradTilingArch22Test, bsh_c4a_fp16)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    int64_t expectTilingKey = 1; // FP16
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// C4Li bf16: B=2, S=8, H=2048, D=128, coff=2, cmp_ratio=4
TEST_F(CompressorGradTilingArch22Test, bsh_c4li_bf16)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{2, 8, 2048}, {2, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2, 128}, {2, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2, 8, 128}, {2, 2, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 8, 128}, {2, 2, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 8, 2048}, {2, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    int64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// C128A bf16: B=2, S=16, H=4096, D=512, coff=1, cmp_ratio=128
TEST_F(CompressorGradTilingArch22Test, bsh_c128a_bf16)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{2, 16, 4096}, {2, 16, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1, 512}, {2, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1, 128, 512}, {2, 1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1, 128, 512}, {2, 1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 16, 4096}, {2, 16, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 512}, {128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    int64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// Non-divisible S: B=1, S=6, H=2048, D=128, coff=2, cmp_ratio=4 (nblk=1, nblkIn=2)
TEST_F(CompressorGradTilingArch22Test, bsh_non_divisible_s)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{1, 6, 2048}, {1, 6, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 2, 128}, {1, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 2, 8, 128}, {1, 2, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 2, 8, 128}, {1, 2, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 6, 2048}, {1, 6, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    int64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// ====================================================================
// Error Cases — expected FAIL
// ====================================================================

// TH layout: arch22 only supports BSH -> reject
TEST_F(CompressorGradTilingArch22Test, err_th_layout_rejected)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},             // x (T, H)
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},       // wkv
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},       // wgate
            {{{4, 512}, {4, 512}}, ge::DT_BF16, ge::FORMAT_ND},               // d_cmp_kv (cmpKvRows, D)
            {{{2, 8, 512}, {2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},        // softmax_score (B, coff*cmpRatio, D)
            {{{2, 8, 512}, {2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},        // kv
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},                        // cu_seqlens
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0);
}

// No complete block: S=2 < cmp_ratio=4 -> arch22 rejects
TEST_F(CompressorGradTilingArch22Test, err_no_complete_block)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{1, 2, 2048}, {1, 2, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 128}, {1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 8, 128}, {1, 1, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 8, 128}, {1, 1, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 2, 2048}, {1, 2, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0);
}

// Unsupported cmp_ratio=3 (interval check: [2, 128])
TEST_F(CompressorGradTilingArch22Test, err_cmp_ratio_3)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0);
}

// Unsupported coff=3 (only 1 and 2 supported)
TEST_F(CompressorGradTilingArch22Test, err_coff_3)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0);
}

// Unsupported headDim=256 (only 128 and 512 supported)
TEST_F(CompressorGradTilingArch22Test, err_head_dim_256)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{2, 8, 2048}, {2, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 2048}, {512, 2048}}, ge::DT_BF16, ge::FORMAT_ND},       // coff*headDim = 2*256 = 512
            {{{512, 2048}, {512, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2, 256}, {2, 2, 256}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2, 8, 256}, {2, 2, 8, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 8, 256}, {2, 2, 8, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 8, 2048}, {2, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 2048}, {512, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 2048}, {512, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 512}, {4, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0);
}

// Unsupported hiddenSize=3000 (not 512-aligned, must be in [1024, 10240])
TEST_F(CompressorGradTilingArch22Test, err_hidden_size_not_aligned)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{2, 8, 3000}, {2, 8, 3000}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 3000}, {1024, 3000}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 3000}, {1024, 3000}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 8, 3000}, {2, 8, 3000}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 3000}, {1024, 3000}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 3000}, {1024, 3000}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0);
}

// Empty tensor: x has zero-size shape -> should fail (backward does not support empty tensors)
TEST_F(CompressorGradTilingArch22Test, err_empty_x_tensor)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{0, 8, 4096}, {0, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{0, 2, 512}, {0, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{0, 2, 8, 512}, {0, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0, 2, 8, 512}, {0, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{0, 8, 4096}, {0, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0);
}
