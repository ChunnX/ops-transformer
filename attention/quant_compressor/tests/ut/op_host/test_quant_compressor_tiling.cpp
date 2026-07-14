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
#include <limits>
#include "tiling/platform/platform_ascendc.h"
#include "tiling_case_executor.h"

struct QuantCompressorUtCompileInfo {
    int64_t core_num;
};

class QuantCompressorTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantCompressorTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantCompressorTiling TearDown" << std::endl;
    }
};

// C4Li: D=128, coff=2, cmp_ratio=4; BSH layout; cache_mode=1 (LINEAR_BUFFER).
TEST_F(QuantCompressorTiling, quant_compressor_tiling_bsh_c4li)
{
    QuantCompressorUtCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara("QuantCompressor",
        {
            {{{1, 128, 4096}, {1, 128, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},   // x
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},         // wkv
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},         // wgate
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},        // state_cache
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // ape
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // x_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // wkv_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // wgate_descale
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},                      // state_block_table
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                              // cu_seqlens (absent for BSH)
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                              // seqused (absent)
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                              // start_pos (absent)
        },
        {
            {{{1, 32, 128}, {1, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},           // cmp_kv
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},        // state_cache out
        },
        {
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950"
    );
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max());
}

// C128A: D=512, coff=1, cmp_ratio=128; BSH layout; cache_mode=2 (RING_BUFFER).
TEST_F(QuantCompressorTiling, quant_compressor_tiling_bsh_c128a)
{
    QuantCompressorUtCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara("QuantCompressor",
        {
            {{{1, 128, 4096}, {1, 128, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},   // x
            {{{512, 4096}, {512, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},         // wkv
            {{{512, 4096}, {512, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},         // wgate
            {{{1, 256, 1024}, {1, 256, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},      // state_cache
            {{{128, 512}, {128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},              // ape
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // x_descale
            {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // wkv_descale
            {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // wgate_descale
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                            // state_block_table [B] for RING
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                              // cu_seqlens (absent for BSH)
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                              // seqused (absent)
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                              // start_pos (absent)
        },
        {
            {{{1, 1, 512}, {1, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},             // cmp_kv
            {{{1, 256, 1024}, {1, 256, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},      // state_cache out
        },
        {
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950"
    );
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max());
}

// FULL_LOAD template: BSH, seqSize<=4 && tokenSize<=256. coreNum=8 to fit splitCoreParam[36].
TEST_F(QuantCompressorTiling, quant_compressor_tiling_bsh_full_load)
{
    QuantCompressorUtCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara("QuantCompressor",
        {
            {{{1, 4, 4096}, {1, 4, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},       // x (S=4)
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},         // wkv
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},         // wgate
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},        // state_cache
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // ape
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // x_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // wkv_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // wgate_descale
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},                      // state_block_table
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                                // cu_seqlens (absent for BSH)
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                                // seqused (absent)
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                                // start_pos (absent)
        },
        {
            {{{1, 1, 128}, {1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},             // cmp_kv (Sr=ceil(4/4)=1)
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},        // state_cache out
        },
        {
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 8
    );
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max());
}

// TH layout: x=[T,H], cu_seqlens=[B+1] present.
TEST_F(QuantCompressorTiling, quant_compressor_tiling_th_c4li)
{
    QuantCompressorUtCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara("QuantCompressor",
        {
            {{{128, 4096}, {128, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},         // x [T,H]
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},         // wkv
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},         // wgate
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},        // state_cache
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // ape
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // x_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // wkv_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // wgate_descale
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},                      // state_block_table
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens [B+1]
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                                // seqused (absent)
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                                // start_pos (absent)
        },
        {
            {{{33, 128}, {33, 128}}, ge::DT_BF16, ge::FORMAT_ND},                 // cmp_kv [Sr,D]
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},        // state_cache out
        },
        {
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950"
    );
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max());
}

// Error case: cmp_ratio=3 not in supported {2,4,8,16,32,64,128} -> GRAPH_FAILED.
TEST_F(QuantCompressorTiling, quant_compressor_tiling_bad_cmp_ratio)
{
    QuantCompressorUtCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara("QuantCompressor",
        {
            {{{1, 128, 4096}, {1, 128, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},   // x
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},         // wkv
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},         // wgate
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},        // state_cache
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // ape
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // x_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // wkv_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // wgate_descale
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},                      // state_block_table
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                                // cu_seqlens (absent for BSH)
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                                // seqused (absent)
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                                // start_pos (absent)
        },
        {
            {{{1, 32, 128}, {1, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},           // cmp_kv
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},        // state_cache out
        },
        {
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950"
    );
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
