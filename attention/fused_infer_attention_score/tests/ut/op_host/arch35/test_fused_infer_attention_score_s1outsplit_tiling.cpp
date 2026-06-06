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
#include "../fused_infer_attention_score_param.h"
#include "../../../../op_host/fused_infer_attention_score_tiling_compile_info.h"
#include "tiling_case_executor.h"

namespace FusedInferAttentionScoreUT {
namespace {

const char *A5_SOC_INFO =
    "{\n"
    "  \"hardware_info\": {\n"
    "    \"BT_SIZE\": 0,\n"
    "    \"load3d_constraints\": \"1\",\n"
    "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
    "    \"Intrinsic_data_move_l12ub\": true,\n"
    "    \"Intrinsic_data_move_l0c2ub\": true,\n"
    "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
    "    \"UB_SIZE\": 196608,\n"
    "    \"L2_SIZE\": 117440512,\n"
    "    \"L1_SIZE\": 524288,\n"
    "    \"L0A_SIZE\": 65536,\n"
    "    \"L0B_SIZE\": 65536,\n"
    "    \"L0C_SIZE\": 65536,\n"
    "    \"vector_core_cnt\": 64,\n"
    "    \"cube_core_cnt\": 32,\n"
    "    \"socVersion\": \"Ascend950\"\n"
    "  }\n"
    "}";

static optiling::FusedInferAttentionScoreCompileInfo g_compileInfo;

class FusedInferAttentionScoreS1OutSplitTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedInferAttentionScore S1OutSplit Arch35 TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedInferAttentionScore S1OutSplit Arch35 TilingTest TearDown" << std::endl;
    }
};

} // namespace

// S1OutSplit 触发：B=8, n2Size=2, S2=8192, L2需求超过一半 L2Size
// 覆盖 CalcRequiredL2Size、CheckS1OutSplit、SplitOutSeq
TEST_F(FusedInferAttentionScoreS1OutSplitTiling, S1OutSplit_large_S1S2_triggers_S1OutSplit)
{
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, B=8, H=16, S1=2048, D=128 (BSND layout: S1 is dim1)
            {{{8, 16, 2048, 128}, {8, 16, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key, B=8, n2=2, S2=8192, D=128
            {{{8, 2, 8192, 128}, {8, 2, 8192, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{8, 2, 8192, 128}, {8, 2, 8192, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            TD_DEFAULT,
            // 5: actual_seq_lengths
            TD_DEFAULT,
            // 6: actual_seq_lengths_kv
            TD_DEFAULT,
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            TD_DEFAULT,
            // 15: query_padding_size
            TD_DEFAULT,
            // 16: kv_padding_size
            TD_DEFAULT,
            // 17-31: remaining optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            // 0: attention_out
            {{{8, 16, 2048, 128}, {8, 16, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo, "Ascend950", 64, 196608, 16384, A5_SOC_INFO);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// S1OutSplit 不触发：S2 较小，CalcRequiredL2Size 不足，走 SplitNBSeq
TEST_F(FusedInferAttentionScoreS1OutSplitTiling, S1OutSplit_small_S2_falls_back_to_SplitNBSeq)
{
    int64_t actualSeqQ[] = {512, 512, 512, 512};
    int64_t actualSeqKv[] = {512, 512, 512, 512};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{4, 16, 512, 128}, {4, 16, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{4, 4, 512, 128}, {4, 4, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{4, 4, 512, 128}, {4, 4, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            TD_DEFAULT,
            // 5: actual_seq_lengths_q
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            TD_DEFAULT,
            // 15-31
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            {{{4, 16, 512, 128}, {4, 16, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo, "Ascend950", 64, 196608, 16384, A5_SOC_INFO);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// CheckS1OutSplit 返回 false：maxActualSeqLenQ < sOuterSize
TEST_F(FusedInferAttentionScoreS1OutSplitTiling, CheckS1OutSplit_small_S1_below_sOuterSize)
{
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, S1=32 < sOuterSize=64
            {{{4, 16, 32, 128}, {4, 16, 32, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key, S2=4096 足够大但 S1 太小
            {{{4, 4, 4096, 128}, {4, 4, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{4, 4, 4096, 128}, {4, 4, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3-31
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            {{{4, 16, 32, 128}, {4, 16, 32, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo, "Ascend950", 64, 196608, 16384, A5_SOC_INFO);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// CheckS1OutSplit 返回 false：sparse_mode=BAND 被 IsCapable 拦截前检查
TEST_F(FusedInferAttentionScoreS1OutSplitTiling, CheckS1OutSplit_sparseMode_BAND_returns_false)
{
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{4, 16, 2048, 128}, {4, 16, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{4, 4, 4096, 128}, {4, 4, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{4, 4, 4096, 128}, {4, 4, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask, BAND 模式需要 mask
            {{{4, 1, 2048, 4096}, {4, 1, 2048, 4096}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5-31
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            {{{4, 16, 2048, 128}, {4, 16, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)}, // SPARSE_MODE_BAND=4
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo, "Ascend950", 64, 196608, 16384, A5_SOC_INFO);

    // sparse_mode=BAND 被 IsCapable 拦截，tiling 失败
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// CheckS1OutSplit 返回 false：kvStorageMode=TENSOR_LIST 被拦截
TEST_F(FusedInferAttentionScoreS1OutSplitTiling, CheckS1OutSplit_TENSOR_LIST_returns_false)
{
    std::vector<uint32_t> inputInstanceNum = {
        1,  // 0: query
        2,  // 1: key tensorlist
        2,  // 2: value tensorlist
        0,  // 3-31
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    };
    std::vector<uint32_t> outputInstanceNum = {1, 0};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, S1=2048 large
            {{{4, 16, 2048, 128}, {4, 16, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1-2: key/value tensorlist
            {{{4, 4, 4096, 128}, {4, 4, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 4, 4096, 128}, {4, 4, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3-4
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            {{{4, 16, 2048, 128}, {4, 16, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        inputInstanceNum,
        outputInstanceNum,
        &g_compileInfo, "Ascend950", 64, 196608, 16384, A5_SOC_INFO);

    // TENSOR_LIST 被 CheckS1OutSplit 拦截，走 SplitNBSeq 路径
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// SplitOutSeq 多batch场景覆盖
TEST_F(FusedInferAttentionScoreS1OutSplitTiling, SplitOutSeq_multi_batch_coverage)
{
    int64_t actualSeqQ[] = {1024, 2048, 512, 768};
    int64_t actualSeqKv[] = {4096, 4096, 4096, 4096};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, B=4
            {{{4, 16, 2048, 128}, {4, 16, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{4, 4, 4096, 128}, {4, 4, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{4, 4, 4096, 128}, {4, 4, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            TD_DEFAULT,
            // 5: actual_seq_lengths_q, different seq lens per batch
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-31
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            {{{4, 16, 2048, 128}, {4, 16, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo, "Ascend950", 64, 196608, 16384, A5_SOC_INFO);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

} // namespace FusedInferAttentionScoreUT
