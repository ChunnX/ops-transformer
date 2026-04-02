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
 * \file test_fused_causal_conv1d_tiling.cpp
 * \brief Unit tests for FusedCausalConv1dCutBH tiling logic
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/fused_causal_conv1d_cut_bh_tiling_arch35.h"
#include "../../../op_host/fused_causal_conv1d_cut_bsh_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class FusedCausalConv1dCutBHTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedCausalConv1dCutBHTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedCausalConv1dCutBHTiling TearDown" << std::endl;
    }
};


TEST_F(FusedCausalConv1dCutBHTiling, FusedCausalConv1dCutBH_950_tiling_bf_b4_s1_d512)
{
    optiling::FusedCausalConv1dCutBHCompileInfo compileInfo = {
        64, 261888};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activation_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"pad_slot_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"run_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"residual_connection", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "FusedCausalConv1d",
        {
            // Input 0: x - (batch=4, seq_len=1, dim=512)
            {{{4, 1, 512}, {4, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_size=3, dim=512)
            {{{3, 512}, {3, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 2: convStates - (batch=4, cache_len=3+1-2=2, dim=512)
            {{{4, 2, 512}, {4, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - optional for 3D x
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=4)
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: hasInitialState - (batch=4)
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 6: bias - optional, (dim=512)
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 7: numAcceptedTokens - optional, (batch=4)
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // Output 0: y - (batch=4, seq_len=1, dim=512)
            {{{4, 1, 512}, {4, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Output 1: cacheStates - (batch=4, cache_len=2, dim=512)
            {{{4, 2, 512}, {4, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 20000;
    std::string expectTilingData = "16 4 4 4 0 128 128 4 0 1 1 1 1 1 1 128 128 1 1 1 1 128 128 4 1 0 512 3 2 512 1024 512 -1 0 1 0 ";

    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FusedCausalConv1dCutBHTiling, FusedCausalConv1dCutBH_950_tiling_bf_b1_s4_d1024)
{
    optiling::FusedCausalConv1dCutBHCompileInfo compileInfo = {
        64, 261888};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activation_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"pad_slot_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"run_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"residual_connection", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "FusedCausalConv1d",
        {
            // Input 0: x - (batch=1, seq_len=4, dim=1024)
            {{{1, 4, 1024}, {1, 4, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_size=3, dim=1024)
            {{{3, 1024}, {3, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 2: convStates - (batch=1, cache_len=3+4-2=5, dim=1024)
            // cache_len = kernel_size + seq_len - 2 = 3 + 4 - 2 = 5
            {{{1, 5, 1024}, {1, 5, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1=2)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=1)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: hasInitialState - (batch=1)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 6: bias - optional, (dim=1024)
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 7: numAcceptedTokens - optional, (batch=1)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            },
        {
            // Output 0: y - (batch=1, seq_len=4, dim=1024)
            {{{1, 4, 1024}, {1, 4, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // Output 1: cacheStates - (batch=1, cache_len=5, dim=1024)
            {{{1, 5, 1024}, {1, 5, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 20000;
    std::string expectTilingData = "8 8 1 8 0 128 128 1 0 1 1 1 1 1 1 128 128 1 1 1 1 128 128 1 4 0 1024 3 5 1024 5120 1024 -1 0 1 0 ";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Helper to build a 2D-Input TilingContextPara
static gert::TilingContextPara Make2DTilingPara(int64_t batch, int64_t cuSeqLen, int64_t dim)
{
    optiling::FusedCausalConv1dCutBHCompileInfo compileInfo = {64, 261888};
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activation_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"pad_slot_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"run_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"residual_connection", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
    };

    const int64_t k = 3;
    const int64_t state_len = 5; 

    return gert::TilingContextPara(
        "FusedCausalConv1d",
        {
            // Input 0: x - (cu_seq_len, dim)
            {{{cuSeqLen, dim}, {cuSeqLen, dim}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 1: weight - (k, dim)
            {{{k, dim}, {k, dim}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 2: convStates - (batch, state_len, dim)
            {{{batch, state_len, dim}, {batch, state_len, dim}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1)
            {{{batch + 1}, {batch + 1}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch)
            {{{batch}, {batch}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: hasInitialState - (batch) [empty as existing case]
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 6: bias - optional
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 7: numAcceptedTokens - (batch)
            {{{batch}, {batch}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // Output 0: y - (cu_seq_len, dim) for 2D path
            {{{cuSeqLen, dim}, {cuSeqLen, dim}}, ge::DT_BF16, ge::FORMAT_ND},
            // Output 1: cacheStates - (batch, state_len, dim)
            {{{batch, state_len, dim}, {batch, state_len, dim}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);
}

TEST_F(FusedCausalConv1dCutBHTiling, FusedCausalConv1dCutBH_950_tiling_bf_b1_s4_d512_x2d)
{
    auto para = Make2DTilingPara(1, 4, 512);
    int64_t expectTilingKey = 20000;
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, "4 4 1 4 0 128 128 1 0 1 1 1 1 1 1 128 128 1 1 1 1 128 128 1 6 4 512 3 5 512 2560 512 -1 1 1 1 ", {});
}

TEST_F(FusedCausalConv1dCutBHTiling, FusedCausalConv1dCutBH_950_tiling_bf_b1_s4_d768_x2d)
{
    auto para = Make2DTilingPara(1, 4, 768);
    int64_t expectTilingKey = 20000;
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, "6 6 1 6 0 128 128 1 0 1 1 1 1 1 1 128 128 1 1 1 1 128 128 1 6 4 768 3 5 768 3840 768 -1 1 1 1 ", {});
}

TEST_F(FusedCausalConv1dCutBHTiling, FusedCausalConv1dCutBH_950_tiling_bf_b1_s4_d4096_x2d)
{
    auto para = Make2DTilingPara(1, 4, 4096);
    int64_t expectTilingKey = 20000;
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, "32 32 1 32 0 128 128 1 0 1 1 1 1 1 1 128 128 1 1 1 1 128 128 1 6 4 4096 3 5 4096 20480 4096 -1 1 1 1 ", {});
}

TEST_F(FusedCausalConv1dCutBHTiling, FusedCausalConv1dCutBH_950_tiling_bf_b1_s4_d8192_x2d)
{
    auto para = Make2DTilingPara(1, 4, 8192);
    int64_t expectTilingKey = 20000;
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, "64 64 1 64 0 128 128 1 0 1 1 1 1 1 1 128 128 1 1 1 1 128 128 1 6 4 8192 3 5 8192 40960 8192 -1 1 1 1 ", {});
}

// batch=32, cuSeqLen=128
TEST_F(FusedCausalConv1dCutBHTiling, FusedCausalConv1dCutBH_950_tiling_bf_b32_s4_d512_x2d)
{
    auto para = Make2DTilingPara(32, 128, 512);
    int64_t expectTilingKey = 20000;
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, "64 4 16 4 0 128 128 16 0 2 2 1 1 2 2 128 128 1 1 2 2 128 128 32 6 128 512 3 5 512 2560 512 -1 1 1 1 ", {});
}

TEST_F(FusedCausalConv1dCutBHTiling, FusedCausalConv1dCutBH_950_tiling_bf_b32_s4_d768_x2d)
{
    auto para = Make2DTilingPara(32, 128, 768);
    int64_t expectTilingKey = 20000;
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, "64 4 16 2 2 256 128 16 0 2 2 1 1 2 2 256 256 1 1 2 2 128 128 32 6 128 768 3 5 768 3840 768 -1 1 1 1 ", {});
}

TEST_F(FusedCausalConv1dCutBHTiling, FusedCausalConv1dCutBH_950_tiling_bf_b32_s4_d4096_x2d)
{
    auto para = Make2DTilingPara(32, 128, 4096);
    int64_t expectTilingKey = 20000;
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, "64 32 2 32 0 128 128 2 0 16 16 1 1 16 16 128 128 1 1 16 16 128 128 32 6 128 4096 3 5 4096 20480 4096 -1 1 1 1 ", {});
}

TEST_F(FusedCausalConv1dCutBHTiling, FusedCausalConv1dCutBH_950_tiling_bf_b32_s4_d8192_x2d)
{
    auto para = Make2DTilingPara(32, 128, 8192);
    int64_t expectTilingKey = 20000;
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, "64 64 1 64 0 128 128 1 0 32 32 1 1 32 32 128 128 1 1 32 32 128 128 32 6 128 8192 3 5 8192 40960 8192 -1 1 1 1 ", {});
}


// ============================================================
// FusedCausalConv1d CutBSH Mode Tests (run_mode=0)
// ============================================================

class FusedCausalConv1dCutBSHTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedCausalConv1dCutBSHTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedCausalConv1dCutBSHTiling TearDown" << std::endl;
    }
};

// Test case 1: Basic FP16 test with small batch and sequence
// batch=2, each sequence length=128, dim=512, kernel_width=3
// cu_seq_len = 2 * 128 = 256
// cache_max_size=4 (> batch=2, allows some buffer)
TEST_F(FusedCausalConv1dCutBSHTiling, FusedCausalConv1d_950_tiling_basic_fp16)
{
    struct FusedCausalConv1dCutBSHCompileInfo {} compileInfo;

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activation_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"pad_slot_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"run_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"residual_connection", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "FusedCausalConv1d",
        {
            // Input 0: x - (cu_seq_len=256, dim=512)
            {{{256, 512}, {256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_width=3, dim=512)
            {{{3, 512}, {3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 2: convStates/cacheStates - (cache_max_size=4, cache_width=2, dim=512)
            {{{4, 2, 512}, {4, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1=3)
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=2)
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: initialStateMode - (batch=2)
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // Output 0: y - (cu_seq_len=256, dim=512)
            {{{256, 512}, {256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Output 1: cacheStates - (cache_max_size=4, cache_width=2, dim=512)
            {{{4, 2, 512}, {4, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 10001;  // FP16 类型
    std::string expectTilingData = "1 1 18 18 128 128 1 1 17 17 128 128 4 0 128 128 16 14 18 17 64 3 256 512 2 -1 512 1024 512 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case 2: Edge case with single batch
// batch=1, sequence length=64, dim=256, kernel_width=3
// cu_seq_len = 64
TEST_F(FusedCausalConv1dCutBSHTiling, FusedCausalConv1d_950_tiling_single_batch)
{
    struct FusedCausalConv1dCutBSHCompileInfo {} compileInfo;

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activation_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"pad_slot_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"run_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"residual_connection", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "FusedCausalConv1d",
        {
            // Input 0: x - (cu_seq_len=64, dim=256)
            {{{64, 256}, {64, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_width=3, dim=256)
            {{{3, 256}, {3, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 2: convStates/cacheStates - (cache_max_size=2, cache_width=2, dim=256)
            {{{2, 2, 256}, {2, 2, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1=2)
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=1)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: initialStateMode - (batch=1)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // Output 0: y - (cu_seq_len=64, dim=256)
            {{{64, 256}, {64, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Output 1: cacheStates - (cache_max_size=2, cache_width=2, dim=256)
            {{{2, 2, 256}, {2, 2, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 10001;
    std::string expectTilingData = "1 1 4 4 128 128 1 1 3 3 128 128 2 0 128 128 32 30 4 3 64 3 64 256 1 -1 256 512 256 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case 3: Medium size test with FP16
// batch=8, each sequence length=512, dim=768, kernel_width=3
// cu_seq_len = 8 * 512 = 4096
TEST_F(FusedCausalConv1dCutBSHTiling, FusedCausalConv1d_950_tiling_medium_fp16)
{
    struct FusedCausalConv1dCutBSHCompileInfo {} compileInfo;

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activation_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"pad_slot_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"run_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"residual_connection", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "FusedCausalConv1d",
        {
            // Input 0: x - (cu_seq_len=4096, dim=768)
            {{{4096, 768}, {4096, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_width=3, dim=768)
            {{{3, 768}, {3, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 2: convStates/cacheStates - (cache_max_size=16, cache_width=2, dim=768)
            {{{16, 2, 768}, {16, 2, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1=9)
            {{{9}, {9}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=8)
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: initialStateMode - (batch=8)
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // Output 0: y - (cu_seq_len=4096, dim=768)
            {{{4096, 768}, {4096, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Output 1: cacheStates - (cache_max_size=16, cache_width=2, dim=768)
            {{{16, 2, 768}, {16, 2, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 10001;
    std::string expectTilingData = "1 2 258 258 128 128 1 1 257 257 128 128 4 2 256 128 16 14 258 257 64 3 4096 768 8 -1 768 1536 768 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case 4: Variable sequence lengths test
// batch=3 with variable sequence lengths (128, 256, 64), dim=512, kernel_width=3
// cu_seq_len = 128 + 256 + 64 = 448
TEST_F(FusedCausalConv1dCutBSHTiling, FusedCausalConv1d_950_tiling_variable_seqlen)
{
    struct FusedCausalConv1dCutBSHCompileInfo {} compileInfo;

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activation_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"pad_slot_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"run_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"residual_connection", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "FusedCausalConv1d",
        {
            // Input 0: x - (cu_seq_len=448, dim=512)
            {{{448, 512}, {448, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_width=3, dim=512)
            {{{3, 512}, {3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 2: convStates/cacheStates - (cache_max_size=6, cache_width=2, dim=512)
            {{{6, 2, 512}, {6, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1=4)
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=3)
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: initialStateMode - (batch=3)
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // Output 0: y - (cu_seq_len=448, dim=512)
            {{{448, 512}, {448, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Output 1: cacheStates - (cache_max_size=6, cache_width=2, dim=512)
            {{{6, 2, 512}, {6, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 10001;
    std::string expectTilingData = "1 1 30 30 128 128 1 1 29 29 128 128 4 0 128 128 16 14 30 29 64 3 448 512 3 -1 512 1024 512 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case 5: Large size test with BF16
// batch=4, cu_seq_len=32768, dim=512, kernel_width=3
TEST_F(FusedCausalConv1dCutBSHTiling, FusedCausalConv1d_950_tiling_large_bf16)
{
    struct FusedCausalConv1dCutBSHCompileInfo {} compileInfo;

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activation_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"pad_slot_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"run_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"residual_connection", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "FusedCausalConv1d",
        {
            // Input 0: x - (cu_seq_len=32768, dim=512)
            {{{32768, 512}, {32768, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_width=3, dim=512)
            {{{3, 512}, {3, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 2: convStates/cacheStates - (cache_max_size=256, cache_width=2, dim=512)
            {{{256, 2, 512}, {256, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1=5)
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=4)
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: initialStateMode - (batch=4)
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // Output 0: y - (cu_seq_len=32768, dim=512)
            {{{32768, 512}, {32768, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Output 1: cacheStates - (cache_max_size=256, cache_width=2, dim=512)
            {{{256, 2, 512}, {256, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 10000;
    std::string expectTilingData = "5 1 492 90 128 128 5 1 492 89 128 128 4 0 128 128 16 14 2050 2049 64 3 32768 512 4 -1 512 1024 512 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case 6: Extra large size test with FP16 (dim split mode)
// batch=256, cu_seq_len=65536, dim=8192, kernel_width=3
TEST_F(FusedCausalConv1dCutBSHTiling, FusedCausalConv1d_950_tiling_xlarge_fp16)
{
    struct FusedCausalConv1dCutBSHCompileInfo {} compileInfo;

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activation_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"pad_slot_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"run_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"residual_connection", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "FusedCausalConv1d",
        {
            // Input 0: x - (cu_seq_len=65536, dim=8192)
            {{{65536, 8192}, {65536, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_width=3, dim=8192)
            {{{3, 8192}, {3, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 2: convStates/cacheStates - (cache_max_size=1024, cache_width=2, dim=8192)
            {{{1024, 2, 8192}, {1024, 2, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1=257)
            {{{257}, {257}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=256)
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: initialStateMode - (batch=256)
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // Output 0: y - (cu_seq_len=65536, dim=8192)
            {{{65536, 8192}, {65536, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Output 1: cacheStates - (cache_max_size=1024, cache_width=3, dim=8192)
            {{{1024, 3, 8192}, {1024, 3, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 10001;
    std::string expectTilingData = "136 1 486 196 128 128 136 1 486 196 128 128 64 0 128 128 1 0 65536 65536 64 3 65536 8192 256 -1 8192 16384 8192 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case 7: Small size test with FP16
TEST_F(FusedCausalConv1dCutBSHTiling, FusedCausalConv1d_950_tiling_small_fp16)
{
    struct FusedCausalConv1dCutBSHCompileInfo {} compileInfo;

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activation_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"pad_slot_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"run_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"residual_connection", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "FusedCausalConv1d",
        {
            // Input 0: x - (cu_seq_len=8, dim=512)
            {{{8, 512}, {8, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_width=3, dim=512)
            {{{3, 512}, {3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 2: convStates/cacheStates - (cache_max_size=4, cache_width=2, dim=512)
            {{{4, 2, 512}, {4, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1=3)
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=2)
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: initialStateMode - (batch=2)
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // Output 0: y - (cu_seq_len=8, dim=512)
            {{{8, 512}, {8, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Output 1: cacheStates - (cache_max_size=4, cache_width=3, dim=512)
            {{{4, 2, 512}, {4, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 10001;
    std::string expectTilingData = "1 1 3 3 128 128 1 1 3 3 128 128 4 0 128 128 6 0 3 3 24 3 8 512 2 -1 512 1024 512 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
