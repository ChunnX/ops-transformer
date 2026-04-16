/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_flash_attn_tiling.cpp
 * \brief Unit tests for npu_flash_attn tiling based on document constraints
 *
 * 测试覆盖文档约束（npu_flash_attn.md）:
 * 1. 公共参数组 - 一致性约束: B≤65536, D≤512且16对齐, N_Q是N_KV整数倍
 * 2. Mask参数组 - 单参数约束: mask_mode=1/3/4, band模式win_left≤win_right
 * 3. SeqLengths参数组 - 单参数约束、存在性约束
 * 4. Paged Attention参数组 - 单参数约束、存在性约束
 * 5. LearnableSink参数组 - sinks dtype=FP32, shape=(N_Q)
 * 6. SoftmaxLSE参数组 - dtype/shape/layout约束
 * 7. Layout匹配关系表
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/flash_attn_tiling_common.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class FlashAttnTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FlashAttnTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FlashAttnTiling TearDown" << std::endl;
    }
};

// =====================================================================
// 公共参数组 - 一致性约束测试
// =====================================================================

// FP16 BSND 正常场景 B=2, N_Q=8, N_KV=2, S=128, D=128
TEST_F(FlashAttnTiling, FlashAttn_950_tiling_BSND_FP16_Normal)
{
    optiling::FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {
         // q: B=2, S_Q=128, N_Q=8, D=128
        {{{2, 128, 8, 128}, {2, 128, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // k: B=2, S_KV=256, N_KV=2, D=128
         {{{2, 256, 2, 128}, {2, 256, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // v: B=2, S_KV=256, N_KV=2, DV=128
         {{{2, 256, 2, 128}, {2, 256, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // block_table (optional)
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // cu_seqlens_q (optional)
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // cu_seqlens_kv (optional)
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // seqused_q (optional)
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // seqused_kv (optional)
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // sinks (optional)
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // metadata (optional)
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         },
        {
         // attention_out: B=2, S_Q=128, N_Q=8, D=128
         {{{2, 128, 8, 128}, {2, 128, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // softmax_lse (optional)
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         },
        {
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", {});
}

// FP16 BNSD 正常场景 B=1, N_Q=4, N_KV=1, S=256, D=192
TEST_F(FlashAttnTiling, FlashAttn_950_tiling_BNSD_FP16_Normal)
{
    optiling::FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {
         // q: B=1, N_Q=4, S_Q=256, D=192
        {{{1, 4, 256, 192}, {1, 4, 256, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // k: B=1, N_KV=1, S_KV=256, D=192
         {{{1, 1, 256, 192}, {1, 1, 256, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // v: B=1, N_KV=1, S_KV=256, DV=192
         {{{1, 1, 256, 192}, {1, 1, 256, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // cu_seqlens_q
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // cu_seqlens_kv
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // seqused_q
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // seqused_kv
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // sinks
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // metadata
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         },
        {
         // attention_out
         {{{1, 4, 256, 192}, {1, 4, 256, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // softmax_lse
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         },
        {
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0721687f)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", {});
}

// =====================================================================
// TND Layout + cu_seqlens测试
// =====================================================================

// FP16 TND 正常场景 + cu_seqlens
TEST_F(FlashAttnTiling, FlashAttn_950_tiling_TND_FP16_WithCuSeqlens)
{
    int64_t cu_seqlens_q_list[] = {0, 128, 384};  // B=2, T_Q=384
    int64_t cu_seqlens_kv_list[] = {0, 256, 640}; // B=2, T_KV=640
    optiling::FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {
         // q: T_Q=384, N_Q=4, D=128
        {{{384, 4, 128}, {384, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // k: T_KV=640, N_KV=1, D=128
         {{{640, 1, 128}, {640, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // v: T_KV=640, N_KV=1, DV=128
         {{{640, 1, 128}, {640, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // cu_seqlens_q: B+1=3
         {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, cu_seqlens_q_list},
         // cu_seqlens_kv: B+1=3
         {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, cu_seqlens_kv_list},
         // seqused_q
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // seqused_kv
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // sinks
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // metadata
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         },
        {
         // attention_out: T_Q=384, N_Q=4, D=128
         {{{384, 4, 128}, {384, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // softmax_lse
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         },
        {
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", {});
}

// =====================================================================
// Mask参数组测试 - mask_mode=1/3/4
// =====================================================================

// mask_mode=1 (全计算模式)
TEST_F(FlashAttnTiling, FlashAttn_950_tiling_MaskMode_1)
{
    optiling::FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {
        {{{2, 128, 4, 128}, {2, 128, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{2, 256, 1, 128}, {2, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{2, 256, 1, 128}, {2, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
        {{{2, 128, 4, 128}, {2, 128, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", {});
}

// mask_mode=3 (Causal模式)
TEST_F(FlashAttnTiling, FlashAttn_950_tiling_MaskMode_3_Causal)
{
    optiling::FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {
        {{{1, 256, 8, 128}, {1, 256, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{1, 256, 2, 128}, {1, 256, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{1, 256, 2, 128}, {1, 256, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
        {{{1, 256, 8, 128}, {1, 256, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", {});
}

// mask_mode=4 (band模式) win_left < win_right
TEST_F(FlashAttnTiling, FlashAttn_950_tiling_MaskMode_4_Band_Valid)
{
    optiling::FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {
        {{{1, 128, 4, 128}, {1, 128, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
        {{{1, 128, 4, 128}, {1, 128, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", {});
}

// =====================================================================
// Paged Attention参数组测试
// =====================================================================

// PagedAttention场景 + seqused_kv + block_table
TEST_F(FlashAttnTiling, FlashAttn_950_tiling_PagedAttention_Valid)
{
    int64_t seqused_kv_list[] = {256, 512};  // B=2
    optiling::FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {
        // q: B=2, S_Q=128, N_Q=8, D=128
        {{{2, 128, 8, 128}, {2, 128, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // k: PA layout - num_blocks=64, N_KV=2, block_size=16, D=128
        {{{64, 2, 16, 128}, {64, 2, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // v: PA layout
        {{{64, 2, 16, 128}, {64, 2, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // block_table: B=2, max_num_blocks=32
        {{{2, 32}, {2, 32}}, ge::DT_INT64, ge::FORMAT_ND},
        // cu_seqlens_q
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        // cu_seqlens_kv
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        // seqused_q
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        // seqused_kv: B=2
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, seqused_kv_list},
        // sinks
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // metadata
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
        {{{2, 128, 8, 128}, {2, 128, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNBD")},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", {});
}

// =====================================================================
// LearnableSink参数组测试 - sinks
// =====================================================================

// sinks参数提供 - dtype=FP32, shape=(N_Q)
TEST_F(FlashAttnTiling, FlashAttn_950_tiling_Sinks_Valid)
{
    optiling::FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {
        {{{2, 128, 8, 128}, {2, 128, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{2, 256, 2, 128}, {2, 256, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{2, 256, 2, 128}, {2, 256, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        // sinks: shape=(N_Q=8), dtype=FP32
        {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
        {{{2, 128, 8, 128}, {2, 128, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", {});
}

// =====================================================================
// SoftmaxLSE参数组测试
// =====================================================================

// return_softmax_lse=1 + BSND layout -> lse shape=(B, N_Q, S_Q)
TEST_F(FlashAttnTiling, FlashAttn_950_tiling_SoftmaxLSE_BSND)
{
    optiling::FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {
        {{{2, 128, 4, 128}, {2, 128, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{2, 256, 1, 128}, {2, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{2, 256, 1, 128}, {2, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
        {{{2, 128, 4, 128}, {2, 128, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // softmax_lse: (B=2, N_Q=4, S_Q=128), dtype=FP32
        {{{2, 4, 128}, {2, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", {});
}

// return_softmax_lse=1 + TND layout -> lse shape=(N_Q, T_Q)
TEST_F(FlashAttnTiling, FlashAttn_950_tiling_SoftmaxLSE_TND)
{
    int64_t cu_seqlens_q_list[] = {0, 128, 256};  // B=2, T_Q=256
    int64_t cu_seqlens_kv_list[] = {0, 256, 512}; // B=2, T_KV=512
    optiling::FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {
        {{{256, 4, 128}, {256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{512, 1, 128}, {512, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{512, 1, 128}, {512, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, cu_seqlens_q_list},
        {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, cu_seqlens_kv_list},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
        {{{256, 4, 128}, {256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // softmax_lse: (N_Q=4, T_Q=256), dtype=FP32
        {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", {});
}

// =====================================================================
// Layout匹配关系测试
// =====================================================================

// BNSD + BNSD kv + BNSD out
TEST_F(FlashAttnTiling, FlashAttn_950_tiling_Layout_BNSD_BNSD_BNSD)
{
    optiling::FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {
        {{{1, 4, 128, 128}, {1, 4, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{1, 1, 256, 128}, {1, 1, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{1, 1, 256, 128}, {1, 1, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
        {{{1, 4, 128, 128}, {1, 4, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", {});
}

// BNSD + BNSD kv + BSND out (允许)
TEST_F(FlashAttnTiling, FlashAttn_950_tiling_Layout_BNSD_BNSD_BSND)
{
    optiling::FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {
        {{{1, 4, 128, 128}, {1, 4, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{1, 1, 256, 128}, {1, 1, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{1, 1, 256, 128}, {1, 1, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
        {{{1, 128, 4, 128}, {1, 128, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", {});
}

// =====================================================================
// BF16测试
// =====================================================================

// BF16 BSND
TEST_F(FlashAttnTiling, FlashAttn_950_tiling_BF16_BSND)
{
    optiling::FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {
        {{{2, 128, 8, 128}, {2, 128, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{2, 256, 2, 128}, {2, 256, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{2, 256, 2, 128}, {2, 256, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
        {{{2, 128, 8, 128}, {2, 128, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", {});
}

// =====================================================================
// 参数约束失败测试（期望GRAPH_FAILED）
// =====================================================================

// mask_mode=0 (无效值，期望失败)
TEST_F(FlashAttnTiling, FlashAttn_950_tiling_MaskMode_0_Invalid)
{
    optiling::FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {
        {{{2, 128, 4, 128}, {2, 128, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{2, 256, 1, 128}, {2, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{2, 256, 1, 128}, {2, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
        {{{2, 128, 4, 128}, {2, 128, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},  // 无效值
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// mask_mode=4 (band) + win_left > win_right (期望失败)
TEST_F(FlashAttnTiling, FlashAttn_950_tiling_BandMode_WinLeftGreaterWinRight)
{
    optiling::FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {
        {{{1, 128, 4, 128}, {1, 128, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
        {{{1, 128, 4, 128}, {1, 128, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(100)},   // win_left > win_right
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(50)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// TND layout但没有cu_seqlens_q (期望失败)
TEST_F(FlashAttnTiling, FlashAttn_950_tiling_TND_NoCuSeqlens)
{
    optiling::FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {
        {{{256, 4, 128}, {256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{512, 1, 128}, {512, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{512, 1, 128}, {512, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},  // 缺少cu_seqlens_q
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
        {{{256, 4, 128}, {256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// cu_seqlens_q长度不等于B+1 (期望失败)
TEST_F(FlashAttnTiling, FlashAttn_950_tiling_CuSeqlensQ_LengthInvalid)
{
    int64_t cu_seqlens_q_list[] = {0, 128};  // 长度=2, 但B=2需要B+1=3
    optiling::FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {
        {{{256, 4, 128}, {256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{512, 1, 128}, {512, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{512, 1, 128}, {512, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, cu_seqlens_q_list},  // 长度错误
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
        {{{256, 4, 128}, {256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// cu_seqlens_q首元素不为0 (期望失败)
TEST_F(FlashAttnTiling, FlashAttn_950_tiling_CuSeqlensQ_FirstNotZero)
{
    int64_t cu_seqlens_q_list[] = {1, 128, 256};  // 首元素=1，应该为0
    int64_t cu_seqlens_kv_list[] = {0, 256, 512};
    optiling::FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {
        {{{256, 4, 128}, {256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{512, 1, 128}, {512, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{512, 1, 128}, {512, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, cu_seqlens_q_list},
        {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, cu_seqlens_kv_list},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
        {{{256, 4, 128}, {256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// N_Q不是N_KV的整数倍 (期望失败)
TEST_F(FlashAttnTiling, FlashAttn_950_tiling_NHeadRatio_Invalid)
{
    optiling::FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {
        {{{2, 128, 7, 128}, {2, 128, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // N_Q=7
        {{{2, 256, 2, 128}, {2, 256, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // N_KV=2, 7%2!=0
        {{{2, 256, 2, 128}, {2, 256, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
        {{{2, 128, 7, 128}, {2, 128, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// D不满足16对齐 (期望失败)
TEST_F(FlashAttnTiling, FlashAttn_950_tiling_D_Not16Aligned)
{
    optiling::FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {
        {{{2, 128, 4, 127}, {2, 128, 4, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // D=127, 不满足16对齐
        {{{2, 256, 1, 127}, {2, 256, 1, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{2, 256, 1, 127}, {2, 256, 1, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
        {{{2, 128, 4, 127}, {2, 128, 4, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Layout不匹配: BSND q + BSND kv + BNSD out (期望失败)
TEST_F(FlashAttnTiling, FlashAttn_950_tiling_Layout_BSND_to_BNSD_Invalid)
{
    optiling::FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {
        {{{2, 128, 4, 128}, {2, 128, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{2, 256, 1, 128}, {2, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{2, 256, 1, 128}, {2, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
        {{{2, 4, 128, 128}, {2, 4, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // BNSD output
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},  // 无效组合
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}