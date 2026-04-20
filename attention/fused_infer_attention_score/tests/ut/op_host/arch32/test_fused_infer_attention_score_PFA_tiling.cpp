/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "../fused_infer_attention_score_param.h"
#include "../../../../op_host/fused_infer_attention_score_tiling.h"
#include "tiling_case_executor.h"
#include "softmax_tiling_mocker.h"

namespace FusedInferAttentionScoreUT {
namespace {

const char *A2_SOC_INFO =
    "{\n"
    "  \"hardware_info\": {\n"
    "    \"BT_SIZE\": 0,\n"
    "    \"load3d_constraints\": \"1\",\n"
    "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
    "    \"Intrinsic_data_move_l12ub\": true,\n"
    "    \"Intrinsic_data_move_l0c2ub\": true,\n"
    "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
    "    \"UB_SIZE\": 196608,\n"
    "    \"L2_SIZE\": 201326592,\n"
    "    \"L1_SIZE\": 524288,\n"
    "    \"L0A_SIZE\": 65536,\n"
    "    \"L0B_SIZE\": 65536,\n"
    "    \"L0C_SIZE\": 131072,\n"
    "    \"vector_core_cnt\": 40,\n"
    "    \"cube_core_cnt\": 20,\n"
    "    \"socVersion\": \"Ascend910_B3\"\n"
    "  }\n"
    "}";

struct EmptyCompileInfo {};
static EmptyCompileInfo g_compileInfo;

class FusedInferAttentionScorePfaTiling : public testing::Test {
protected:
    void SetUp() override
    {
        SoftmaxTilingMocker::GetInstance().SetSocVersion("Ascend910B");
    }

    void TearDown() override
    {
        SoftmaxTilingMocker::GetInstance().Reset();
    }
};

} // namespace

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_0)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3-5: pse_shift / atten_mask / actual_seq_lengths
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 6: actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-31: remaining optional inputs
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
            // 0: attention_out
            {{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // Head and scale config.
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            // Layout / KV head config.
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            // PA path config: invalid block_size=127 covers FIA/PFA reject logic.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            // Quant / output config.
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_1)
{
    int64_t actualSeqQ[] = {128, 256, 384, 512};
    int64_t actualSeqKv[] = {256, 512, 768, 1024};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{16, 128, 1024}, {16, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{16, 128, 1024}, {16, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            TD_DEFAULT,
            // 5: actual_seq_lengths_q
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-31: remaining optional inputs
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
            // 0: attention_out
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // Head and scale config.
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            // Layout / KV head config.
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            // PA sparse-mode coverage: sparse_mode=1 while atten_mask is absent.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            // Quant / output config.
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_2)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{1, 16, 1024}, {1, 16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3-5: pse_shift / atten_mask / actual_seq_lengths
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 6: actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15: query_padding_size
            TD_DEFAULT,
            // 16: kv_padding_size
            TD_DEFAULT,
            // 17: key_antiquant_scale
            {{{1, 1024}, {1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 18: key_antiquant_offset
            TD_DEFAULT,
            // 19: value_antiquant_scale
            {{{1, 1024}, {1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 20-31: remaining optional inputs
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
            {{{1, 16, 1024}, {1, 16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // Head and scale config.
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            // Layout / KV head config.
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            // Exercise PA + split key/value antiquant routing.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            // Quant / output config.
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_3)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{1, 16, 1024}, {1, 16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3-5: pse_shift / atten_mask / actual_seq_lengths
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 6: actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15: query_padding_size
            TD_DEFAULT,
            // 16: kv_padding_size
            TD_DEFAULT,
            // 17: key_antiquant_scale
            {{{1, 1024}, {1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 18: key_antiquant_offset
            TD_DEFAULT,
            // 19: value_antiquant_scale
            {{{1, 1024}, {1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 20-31: remaining optional inputs
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
            {{{1, 16, 1024}, {1, 16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // Head and scale config.
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            // Layout / KV head config.
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            // Keep antiquant tensors present but antiquant modes disabled.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            // Quant / output config.
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_4)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3-5: pse_shift / atten_mask / actual_seq_lengths
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 6: actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-31: remaining optional inputs
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
            // 0: attention_out
            {{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // Head and scale config.
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            // Layout / KV head config.
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            // Let the case pass FIA PA checker and continue into PFA PA path.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            // Quant / output config.
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 103000000000200000);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_5)
{
    int64_t actualSeqQ[] = {128, 256, 384, 512};
    int64_t actualSeqKv[] = {256, 512, 768, 1024};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{16, 128, 1024}, {16, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{16, 128, 1024}, {16, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            {{{1, 1, 2048, 2048}, {1, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5: actual_seq_lengths_q
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-31: remaining optional inputs
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
            // 0: attention_out
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // Head and scale config.
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            // Layout / KV head config.
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            // sparse_mode=1 is legal in FIA with mask, but should be rejected by PFA PA TND branch.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            // Quant / output config.
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 103000000010300003);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_6)
{
    int64_t actualSeqQ[] = {128, 256, 384, 512};
    int64_t actualSeqKv[] = {256, 512, 768, 1024};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{16, 128, 1024}, {16, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{16, 128, 1024}, {16, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3-4: pse_shift / atten_mask
            TD_DEFAULT,
            TD_DEFAULT,
            // 5: actual_seq_lengths_q
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table, intentionally make batch axis smaller than actual batch count
            {{{3, 8}, {3, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-31: remaining optional inputs
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
            // 0: attention_out
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // Head and scale config.
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            // Layout / KV head config.
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            // sparse_mode=0 lets the case enter PFA PA base-api branch and fail on seq-len/block-table consistency.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            // Quant / output config.
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5000000000010210100);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_7)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{1, 16, 1024}, {1, 16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5: actual_seq_lengths
            TD_DEFAULT,
            // 6: actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-11: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 12: antiquant_scale, use generic KV antiquant path instead of split key/value scales
            {{{2, 1024}, {2, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 13: antiquant_offset
            TD_DEFAULT,
            // 14: block_table
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-31: remaining optional inputs
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
            // 0: attention_out
            {{{1, 16, 1024}, {1, 16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // Head and scale config.
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            // Layout / KV head config.
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            // PA + generic antiquant branch coverage.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            // Quant / output config.
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_8)
{
    int64_t actualSeqQ[] = {128, 256, 384, 512};
    int64_t actualSeqKv[] = {256, 512, 768, 1024};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{16, 128, 1024}, {16, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{16, 128, 1024}, {16, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask, keep mask non-null so sparse_mode=0 can exercise the PA TND mask check
            {{{1, 1, 2048, 2048}, {1, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5: actual_seq_lengths_q
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-31: remaining optional inputs
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
            // 0: attention_out
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // Head and scale config.
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            // Layout / KV head config.
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            // Target the TND + PA sparse_mode=0 mask rejection branch in PFA.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            // Quant / output config.
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 103000000010200003);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_9)
{
    int64_t actualSeqQ[] = {128, 256, 384, 512};
    int64_t actualSeqKv[] = {256, 512, 768, 1024};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{16, 128, 1024}, {16, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{16, 128, 1024}, {16, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            {{{1, 1, 2048, 2048}, {1, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5: actual_seq_lengths_q
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table, keep dim2 smaller than max block count to probe PA block_table checks
            {{{4, 4}, {4, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-31: remaining optional inputs
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
            // 0: attention_out
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // Head and scale config.
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            // Layout / KV head config.
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            // sparse_mode=4 keeps a legal sparse mode while forcing the PA block_table path.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            // Quant / output config.
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5000000000010210105);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_10)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{1, 16, 1024}, {1, 16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5: actual_seq_lengths
            TD_DEFAULT,
            // 6: actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15: query_padding_size
            TD_DEFAULT,
            // 16: kv_padding_size
            TD_DEFAULT,
            // 17: key_antiquant_scale, use per-channel split KV antiquant
            {{{1, 1024}, {1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 18: key_antiquant_offset
            TD_DEFAULT,
            // 19: value_antiquant_scale
            {{{1, 1024}, {1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 20-31: remaining optional inputs
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
            {{{1, 16, 1024}, {1, 16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // Head and scale config.
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            // Layout / KV head config.
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            // Probe the PA per-channel antiquant support branch with valid split KV antiquant shapes.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            // Quant / output config.
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_11)
{
    int64_t actualSeqQ[] = {128, 256, 384, 512};
    int64_t actualSeqKv[] = {256, 512, 768, 1024};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, TND layout
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key, BNBD paged-cache layout to avoid routing into FAI
            {{{20, 8, 128, 128}, {20, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{20, 8, 128, 128}, {20, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask, keep null so sparse_mode=1 fails inside PFA PA sparse-mode check
            TD_DEFAULT,
            // 5: actual_seq_lengths
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-31: remaining optional inputs
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
            // 0: attention_out
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // Head and scale config.
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            // Layout / KV head config.
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            // PFA PA TND path: sparse_mode=1 is intentionally invalid for PA.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            // Quant / output config.
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_12)
{
    int64_t actualSeqQ[] = {128, 256, 384, 512};
    int64_t actualSeqKv[] = {256, 512, 768, 1024};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, TND layout
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key, BNBD paged-cache layout to route into PFA PA checks
            {{{20, 8, 128, 128}, {20, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{20, 8, 128, 128}, {20, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            TD_DEFAULT,
            // 5: actual_seq_lengths
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-31: remaining optional inputs
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
            // 0: attention_out
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // Head and scale config.
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            // Layout / KV head config.
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            // PFA PA TND path: sparse_mode is valid, invalid block_size probes deeper PA checks.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            // Quant / output config.
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_13)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, BSH layout with seq large enough to enable KV antiquant path
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key, BBH paged-cache layout
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask, sparse_mode=3 requires a sparse mask tensor
            {{{1, 1, 2048, 2048}, {1, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5: actual_seq_lengths
            TD_DEFAULT,
            // 6: actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15: query_padding_size
            TD_DEFAULT,
            // 16: kv_padding_size
            TD_DEFAULT,
            // 17: key_antiquant_scale, use BF16 to match per-channel PA antiquant requirement
            {{{1, 1024}, {1, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 18: key_antiquant_offset
            TD_DEFAULT,
            // 19: value_antiquant_scale
            {{{1, 1024}, {1, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 20-31: remaining optional inputs
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
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // Head and scale config.
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            // Layout / KV head config.
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            // PFA PA + per-channel antiquant path: sparse_mode=3 is valid, inner_precise=0 should fail in support check.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            // Quant / output config.
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_14)
{
    int64_t actualSeqQ[] = {128, 256, 384, 512};
    int64_t actualSeqKv[] = {256, 512, 768, 1024};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, TND layout
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key, BNBD paged-cache layout to route into PFA PA checks
            {{{20, 8, 128, 128}, {20, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{20, 8, 128, 128}, {20, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            TD_DEFAULT,
            // 5: actual_seq_lengths
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-31: remaining optional inputs
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
            // 0: attention_out
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // Head and scale config.
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            // Layout / KV head config.
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            // Valid TND + PA probe to verify PFA PA path can be entered.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            // Quant / output config.
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 103000000000300003);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_15)
{
    int64_t actualSeqQ[] = {128, 256, 384, 512};
    int64_t actualSeqKv[] = {256, 512, 768, 1024};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, TND layout
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key, BNBD paged-cache layout to route into PFA PA checks
            {{{20, 8, 128, 128}, {20, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{20, 8, 128, 128}, {20, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask, sparse_mode=3 needs a valid sparse mask
            {{{1, 1, 2048, 2048}, {1, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5: actual_seq_lengths
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-31: remaining optional inputs
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
            // 0: attention_out
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // Head and scale config.
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            // Layout / KV head config.
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            // Valid TND + PA + sparse mask probe.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            // Quant / output config.
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 103000000000300003);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_16)
{
    int64_t actualSharedPrefixLen[] = {1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, BSH layout
            // Use qs=1 and head_dim=80 so FIA falls back to the legacy PFA prefix path.
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            TD_DEFAULT,
            // 5: actual_seq_lengths_q
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
            // 14: block_table, keep null so the case stays on the non-PA PFA prefix path
            TD_DEFAULT,
            // 15: query_padding_size
            TD_DEFAULT,
            // 16: kv_padding_size
            TD_DEFAULT,
            // 17: key_antiquant_scale
            TD_DEFAULT,
            // 18: key_antiquant_offset
            TD_DEFAULT,
            // 19: value_antiquant_scale
            TD_DEFAULT,
            // 20: value_antiquant_offset
            TD_DEFAULT,
            // 21: key_shared_prefix, BSH prefix tensor [B, prefixS, H]
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 22: value_shared_prefix
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 23: actual_shared_prefix_len
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSharedPrefixLen},
            // 24-31: remaining optional inputs
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
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // Head and scale config.
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            // Layout / KV head config.
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            // Prefix path stays on regular PFA, not PA.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            // Quant / output config.
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20000000000100001);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_17)
{
    int64_t actualSharedPrefixLen[] = {2};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, BSH layout
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3-20: optional inputs unrelated to prefix remain empty
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
            // 21: key_shared_prefix, prefixS=1
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 22: value_shared_prefix
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 23: actual_shared_prefix_len, intentionally larger than prefixS
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSharedPrefixLen},
            // 24-31: remaining optional inputs
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
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_18)
{
    int64_t actualSharedPrefixLen[] = {1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, BSH layout
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3-20: optional inputs unrelated to prefix remain empty
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
            // 21: key_shared_prefix, intentionally use mismatched H to hit the BSH prefix shape check
            {{{1, 1, 64}, {1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 22: value_shared_prefix
            {{{1, 1, 64}, {1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 23: actual_shared_prefix_len
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSharedPrefixLen},
            // 24-31: remaining optional inputs
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
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_19)
{
    int64_t actualSharedPrefixLen[] = {1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, BSH layout
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3-20: optional inputs unrelated to prefix remain empty
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
            // 21: key_shared_prefix, intentionally use batch=2 to hit the prefix batch check
            {{{2, 1, 80}, {2, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 22: value_shared_prefix
            {{{2, 1, 80}, {2, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 23: actual_shared_prefix_len
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSharedPrefixLen},
            // 24-31: remaining optional inputs
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
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_20)
{
    int64_t actualSeqQ[] = {128, 256, 384, 512};
    int64_t actualSeqKv[] = {256, 512, 768, 1024};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, TND layout, use BF16 so FIA v3 routing is disabled and the case falls through to PFA
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key, BNBD paged-cache layout
            {{{20, 8, 128, 128}, {20, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{20, 8, 128, 128}, {20, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask, keep null so sparse_mode=1 is rejected by PFA PA sparse-mode helper
            TD_DEFAULT,
            // 5: actual_seq_lengths_q
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table, keep valid so the failure happens inside PFA PA sparse-mode logic
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-31: remaining optional inputs
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
            // 0: attention_out
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            // PFA PA TND helper coverage: sparse_mode=1 is invalid inside CheckPASparseMode.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_21)
{
    int64_t actualSeqQ[] = {128, 256, 384, 512};
    int64_t actualSeqKv[] = {256, 512, 768, 1024};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, TND layout, use BF16 so FIA v3 routing is disabled and the case falls through to PFA
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key, BNBD paged-cache layout
            {{{20, 8, 128, 128}, {20, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{20, 8, 128, 128}, {20, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            TD_DEFAULT,
            // 5: actual_seq_lengths_q
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-31: remaining optional inputs
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
            // 0: attention_out
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            // PFA PA TND helper coverage: sparse_mode is legal, invalid block_size forces CheckPAWhenBaseApi.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_22)
{
    int64_t actualSharedPrefixLen[] = {1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, BSH layout, keep qs=2 so FIA does not route to IFA.
            {{{1, 2, 80}, {1, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{1, 2, 80}, {1, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{1, 2, 80}, {1, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3-20: optional inputs unrelated to prefix remain empty
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
            // 21: key_shared_prefix, BSH prefix tensor [B, prefixS, H]
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 22: value_shared_prefix
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 23: actual_shared_prefix_len, keep non-null so PFA enters the explicit prefix-len branch
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSharedPrefixLen},
            // 24-31: remaining optional inputs
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
            {{{1, 2, 80}, {1, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 104000000010100001);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_23)
{
    int64_t actualSharedPrefixLen[] = {1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, BNSD layout, keep qs=2 so the path stays out of IFA.
            {{{1, 1, 2, 80}, {1, 1, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{1, 1, 2, 80}, {1, 1, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{1, 1, 2, 80}, {1, 1, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3-20: optional inputs unrelated to prefix remain empty
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
            // 21: key_shared_prefix, BNSD prefix tensor [B, N, prefixS, D]
            {{{1, 1, 1, 80}, {1, 1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 22: value_shared_prefix
            {{{1, 1, 1, 80}, {1, 1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 23: actual_shared_prefix_len
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSharedPrefixLen},
            // 24-31: remaining optional inputs
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
            {{{1, 1, 2, 80}, {1, 1, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 104000000000100000);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_24)
{
    int64_t actualSeqQ[] = {128, 256, 384, 512};
    int64_t actualSeqKv[] = {256, 512, 768, 1024};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, TND layout, keep BF16 so the path stays on PFA instead of FIA v3.
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key, BNBD paged-cache layout
            {{{20, 8, 128, 128}, {20, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{20, 8, 128, 128}, {20, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask, sparse_mode=0 keeps it optional
            TD_DEFAULT,
            // 5: actual_seq_lengths_q
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table, keep valid so PFA PA helper can continue past block-size validation
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-31: remaining optional inputs
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
            // 0: attention_out
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_25)
{
    int64_t actualSeqQ[] = {6, 12};
    int64_t actualSeqKv[] = {6, 12};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, TND layout so FIA routes into PFA instead of IFA.
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            TD_DEFAULT,
            // 5: actual_seq_lengths_q
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-23: quant / antiquant / prefix related optional inputs remain empty
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
            // 24: query_rope, valid TND rope tensor [T, N, 64]
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 25: key_rope
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 26-31: remaining optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            // 0: attention_out
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
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
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_26)
{
    int64_t actualSeqQ[] = {6, 12};
    int64_t actualSeqKv[] = {6, 12};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, TND layout
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value, intentionally use D=64 so FIA passes rope checks but PFA rejects rope+valueD!=128
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            TD_DEFAULT,
            // 5: actual_seq_lengths_q
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-23: unrelated optional inputs remain empty
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
            // 24: query_rope
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 25: key_rope
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 26-31: remaining optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            // 0: attention_out
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
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
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_27)
{
    int64_t actualSeqQ[] = {6, 12};
    int64_t actualSeqKv[] = {6, 12};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, TND layout
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{12, 32, 128}, {12, 32, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{12, 32, 128}, {12, 32, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            TD_DEFAULT,
            // 5: actual_seq_lengths_q
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-23: unrelated optional inputs remain empty
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
            // 24: query_rope
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 25: key_rope, keep N aligned with KV heads to exercise the GQA rope success path
            {{{12, 32, 64}, {12, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 26-31: remaining optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            // 0: attention_out
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
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
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_28)
{
    int64_t actualSeqQ[] = {6, 12};
    int64_t actualSeqKv[] = {6, 12};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, use BF16 so RouteToFia() stays false and the rope case falls through to old PFA
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            TD_DEFAULT,
            // 5: actual_seq_lengths_q
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-23: unrelated optional inputs remain empty
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
            // 24: query_rope, keep dtype aligned with query BF16
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            // 25: key_rope, keep dtype aligned with key FP16
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 26-31: remaining optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            // 0: attention_out
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
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
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_29)
{
    int64_t actualSeqQ[] = {6, 12};
    int64_t actualSeqKv[] = {6, 12};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, keep BF16 so the rope case stays on old PFA instead of RouteToFia()
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            TD_DEFAULT,
            // 5: actual_seq_lengths_q
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-23: unrelated optional inputs remain empty
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
            // 24: query_rope
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            // 25: key_rope, intentionally mismatch N to probe PFA rope N-axis rejection
            {{{12, 32, 64}, {12, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 26-31: remaining optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            // 0: attention_out
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
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
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_30)
{
    int64_t actualSeqQ[] = {6, 12};
    int64_t actualSeqKv[] = {6, 12};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, keep BF16 so the rope case stays on old PFA instead of RouteToFia()
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            TD_DEFAULT,
            // 5: actual_seq_lengths_q
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-23: unrelated optional inputs remain empty
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
            // 24: query_rope, intentionally use D=32 to probe PFA rope D-axis rejection
            {{{12, 64, 32}, {12, 64, 32}}, ge::DT_BF16, ge::FORMAT_ND},
            // 25: key_rope
            {{{12, 64, 32}, {12, 64, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 26-31: remaining optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            // 0: attention_out
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
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
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_31)
{
    int64_t actualSeqQ[] = {6, 12};
    int64_t actualSeqKv[] = {6, 12};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, keep BF16 so the case stays on old PFA instead of RouteToFia()
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            TD_DEFAULT,
            // 5: actual_seq_lengths_q
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-23: unrelated optional inputs remain empty
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
            // 24: query_rope, keep null so old PFA hits the "keyRope exists but queryRope is null" branch
            TD_DEFAULT,
            // 25: key_rope
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 26-31: remaining optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            // 0: attention_out
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
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
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_32)
{
    int64_t actualSeqQ[] = {6, 12};
    int64_t actualSeqKv[] = {6, 12};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, keep BF16 so the case stays on old PFA instead of RouteToFia()
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            TD_DEFAULT,
            // 5: actual_seq_lengths_q
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-23: unrelated optional inputs remain empty
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
            // 24: query_rope
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            // 25: key_rope, keep null so old PFA hits the "queryRope exists but keyRope is null" branch
            TD_DEFAULT,
            // 26-31: remaining optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            // 0: attention_out
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
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
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_33)
{
    int64_t actualSharedPrefixLen[] = {2};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, keep BF16 so RouteToFia() stays false and queryS=2 avoids the IFA path.
            {{{1, 2, 80}, {1, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key
            {{{1, 2, 80}, {1, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{1, 2, 80}, {1, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3-20: optional inputs unrelated to prefix remain empty
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
            // 21: key_shared_prefix, BSH prefix tensor [B, prefixS, H]
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 22: value_shared_prefix
            {{{1, 1, 80}, {1, 1, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 23: actual_shared_prefix_len, intentionally larger than prefixS to hit old-PFA range check
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSharedPrefixLen},
            // 24-31: remaining optional inputs
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
            {{{1, 2, 80}, {1, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_34)
{
    int64_t actualSharedPrefixLen[] = {1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, keep BF16 so RouteToFia() stays false and queryS=2 avoids the IFA path.
            {{{1, 2, 80}, {1, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key
            {{{1, 2, 80}, {1, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{1, 2, 80}, {1, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3-20: optional inputs unrelated to prefix remain empty
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
            // 21: key_shared_prefix, intentionally use mismatched H to hit old-PFA BSH prefix shape check
            {{{1, 1, 64}, {1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 22: value_shared_prefix
            {{{1, 1, 64}, {1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 23: actual_shared_prefix_len
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSharedPrefixLen},
            // 24-31: remaining optional inputs
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
            {{{1, 2, 80}, {1, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_35)
{
    int64_t actualSharedPrefixLen[] = {1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, keep BF16 so RouteToFia() stays false and queryS=2 avoids the IFA path.
            {{{1, 1, 2, 80}, {1, 1, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key
            {{{1, 1, 2, 80}, {1, 1, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{1, 1, 2, 80}, {1, 1, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3-20: optional inputs unrelated to prefix remain empty
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
            // 21: key_shared_prefix, intentionally mismatch D to hit old-PFA BNSD prefix N/D check
            {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 22: value_shared_prefix
            {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 23: actual_shared_prefix_len
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSharedPrefixLen},
            // 24-31: remaining optional inputs
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
            {{{1, 1, 2, 80}, {1, 1, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_36)
{
    int64_t actualSeqQ[] = {2};
    int64_t actualSeqKv[] = {4};
    int64_t queryPaddingSize[] = {1};
    int64_t kvPaddingSize[] = {1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, keep BF16 so RouteToFia() stays false and queryS=2 avoids the IFA path.
            {{{1, 2, 80}, {1, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key
            {{{1, 4, 80}, {1, 4, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{1, 4, 80}, {1, 4, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            TD_DEFAULT,
            // 5: actual_seq_lengths_q, required once left padding is enabled
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant / antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table, keep null so the case stays on the non-PA old PFA path
            TD_DEFAULT,
            // 15: query_padding_size
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, queryPaddingSize},
            // 16: kv_padding_size
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, kvPaddingSize},
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
            {{{1, 2, 80}, {1, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_37)
{
    std::vector<uint32_t> inputInstanceNum = {
        1,  // 0: query
        2,  // 1: key tensorlist
        2,  // 2: value tensorlist
        0,  // 3: pse_shift
        0,  // 4: atten_mask
        0,  // 5: actual_seq_lengths_q
        0,  // 6: actual_seq_lengths_kv
        0,  // 7: dequant_scale1
        0,  // 8: quant_scale1
        0,  // 9: dequant_scale2
        0,  // 10: quant_scale2
        0,  // 11: quant_offset2
        0,  // 12: antiquant_scale
        0,  // 13: antiquant_offset
        0,  // 14: block_table
        0,  // 15: query_padding_size
        0,  // 16: kv_padding_size
        0,  // 17: key_antiquant_scale
        0,  // 18: key_antiquant_offset
        0,  // 19: value_antiquant_scale
        0,  // 20: value_antiquant_offset
        0,  // 21: key_shared_prefix
        0,  // 22: value_shared_prefix
        0,  // 23: actual_shared_prefix_len
        0,  // 24: query_rope
        0,  // 25: key_rope
        0,  // 26: latent_cache
        0,  // 27: block_table_offset
        0,  // 28: query_padding_size_as
        0,  // 29: kv_padding_size_as
        0,  // 30: query_rope_offset
        0,  // 31: key_rope_offset
    };
    std::vector<uint32_t> outputInstanceNum = {1, 0};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, use BF16 so RouteToFia() stays false; queryS=2 avoids the IFA route.
            {{{2, 2, 80}, {2, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1-2: key tensorlist, BSH tensorlist makes old PFA read seq_len from dim 1.
            {{{1, 3, 80}, {1, 3, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 80}, {1, 4, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3-4: value tensorlist
            {{{1, 3, 80}, {1, 3, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 80}, {1, 4, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // 0: attention_out
            {{{2, 2, 80}, {2, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
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
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_38)
{
    std::vector<uint32_t> inputInstanceNum = {
        1,  // 0: query
        2,  // 1: key tensorlist
        2,  // 2: value tensorlist
        0,  // 3: pse_shift
        0,  // 4: atten_mask
        0,  // 5: actual_seq_lengths_q
        0,  // 6: actual_seq_lengths_kv
        0,  // 7: dequant_scale1
        0,  // 8: quant_scale1
        0,  // 9: dequant_scale2
        0,  // 10: quant_scale2
        0,  // 11: quant_offset2
        0,  // 12: antiquant_scale
        0,  // 13: antiquant_offset
        0,  // 14: block_table
        0,  // 15: query_padding_size
        0,  // 16: kv_padding_size
        0,  // 17: key_antiquant_scale
        0,  // 18: key_antiquant_offset
        0,  // 19: value_antiquant_scale
        0,  // 20: value_antiquant_offset
        0,  // 21: key_shared_prefix
        0,  // 22: value_shared_prefix
        0,  // 23: actual_shared_prefix_len
        0,  // 24: query_rope
        0,  // 25: key_rope
        0,  // 26: latent_cache
        0,  // 27: block_table_offset
        0,  // 28: query_padding_size_as
        0,  // 29: kv_padding_size_as
        0,  // 30: query_rope_offset
        0,  // 31: key_rope_offset
    };
    std::vector<uint32_t> outputInstanceNum = {1, 0};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, BNSD still stays on old PFA because q dtype != k/v dtype and queryS=2.
            {{{2, 1, 2, 80}, {2, 1, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1-2: key tensorlist, BNSD tensorlist makes old PFA read seq_len from dim 2.
            {{{1, 1, 3, 80}, {1, 1, 3, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 4, 80}, {1, 1, 4, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3-4: value tensorlist
            {{{1, 1, 3, 80}, {1, 1, 3, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 4, 80}, {1, 1, 4, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // 0: attention_out
            {{{2, 1, 2, 80}, {2, 1, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
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
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_39)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, BSH layout keeps the old PFA antiquant path simple and readable.
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key, paged-cache KV with int8 dtype to enable PA antiquant checks in old PFA.
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask, keep sparse mask non-null so this case can fail exactly on sparse_mode support.
            {{{1, 1, 2048, 2048}, {1, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5: actual_seq_lengths
            TD_DEFAULT,
            // 6: actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15: query_padding_size
            TD_DEFAULT,
            // 16: kv_padding_size
            TD_DEFAULT,
            // 17: key_antiquant_scale
            {{{1, 1024}, {1, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 18: key_antiquant_offset
            TD_DEFAULT,
            // 19: value_antiquant_scale
            {{{1, 1024}, {1, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 20-31: remaining optional inputs
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
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            // Keep high performance valid, then use sparse_mode=0 to hit the old-PFA per-channel antiquant sparse-mode reject.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_40)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask, keep null so the case reaches old-PFA mask-null reject inside PA antiquant support checks.
            TD_DEFAULT,
            // 5: actual_seq_lengths
            TD_DEFAULT,
            // 6: actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15: query_padding_size
            TD_DEFAULT,
            // 16: kv_padding_size
            TD_DEFAULT,
            // 17: key_antiquant_scale
            {{{1, 1024}, {1, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 18: key_antiquant_offset
            TD_DEFAULT,
            // 19: value_antiquant_scale
            {{{1, 1024}, {1, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 20-31: remaining optional inputs
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
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            // sparse_mode=3 is valid, so the next missing branch is the null-mask reject.
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_41)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3: pse_shift, keep non-null to hit the PA antiquant PSE reject in old PFA.
            {{{1, 1, 2048, 2048}, {1, 1, 2048, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 4: atten_mask
            {{{1, 1, 2048, 2048}, {1, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5: actual_seq_lengths
            TD_DEFAULT,
            // 6: actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15: query_padding_size
            TD_DEFAULT,
            // 16: kv_padding_size
            TD_DEFAULT,
            // 17: key_antiquant_scale
            {{{1, 1024}, {1, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 18: key_antiquant_offset
            TD_DEFAULT,
            // 19: value_antiquant_scale
            {{{1, 1024}, {1, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 20-31: remaining optional inputs
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
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_42)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, BNSD with qS=128 keeps the case on old PFA and enables KV antiquant.
            {{{1, 8, 128, 128}, {1, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3-6: keep PA inputs minimal and valid so old PFA can reject on antiquant shape.
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and generic antiquant optional inputs stay empty.
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15: query_padding_size
            TD_DEFAULT,
            // 16: kv_padding_size
            TD_DEFAULT,
            // 17: key_antiquant_scale, intentionally use 3 dims to hit old-PFA BNSD dim-num reject.
            {{{1, 8, 128}, {1, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 18: key_antiquant_offset
            TD_DEFAULT,
            // 19: value_antiquant_scale
            {{{1, 8, 128}, {1, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 20: value_antiquant_offset
            TD_DEFAULT,
            // 21-31: remaining optional inputs
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
            {{{1, 8, 128, 128}, {1, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_43)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{1, 8, 128, 128}, {1, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3-6
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-16
            TD_DEFAULT,
            TD_DEFAULT,
            // 17: key_antiquant_scale, keep 4 dims but break dim[2] so old PFA fails on shape body.
            {{{1, 8, 2, 128}, {1, 8, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 18: key_antiquant_offset
            TD_DEFAULT,
            // 19: value_antiquant_scale
            {{{1, 8, 2, 128}, {1, 8, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 20: value_antiquant_offset
            TD_DEFAULT,
            // 21-31
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
            {{{1, 8, 128, 128}, {1, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_44)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{1, 8, 128, 128}, {1, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3-6
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-16
            TD_DEFAULT,
            TD_DEFAULT,
            // 17: key_antiquant_scale, valid 4D BNSD antiquant shape.
            {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 18: key_antiquant_offset, intentionally 3D to hit old-PFA offset dim-num reject.
            {{{1, 8, 128}, {1, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 19: value_antiquant_scale
            {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 20: value_antiquant_offset
            {{{1, 8, 128}, {1, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 21-31
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
            {{{1, 8, 128, 128}, {1, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_45)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{1, 8, 128, 128}, {1, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3-6
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-16
            TD_DEFAULT,
            TD_DEFAULT,
            // 17: key_antiquant_scale
            {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 18: key_antiquant_offset, keep 4D but break dim[2].
            {{{1, 8, 2, 128}, {1, 8, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 19: value_antiquant_scale
            {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 20: value_antiquant_offset
            {{{1, 8, 2, 128}, {1, 8, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 21-31
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
            {{{1, 8, 128, 128}, {1, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_46)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, BSND also keeps qS=128 so the shared old-PFA path is still chosen.
            {{{1, 128, 8, 128}, {1, 128, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3-6
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-16
            TD_DEFAULT,
            TD_DEFAULT,
            // 17: key_antiquant_scale, intentionally use 2 dims to hit old-PFA BSND dim-num reject.
            {{{1, 1024}, {1, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 18: key_antiquant_offset
            TD_DEFAULT,
            // 19: value_antiquant_scale
            {{{1, 1024}, {1, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 20: value_antiquant_offset
            TD_DEFAULT,
            // 21-31
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
            {{{1, 128, 8, 128}, {1, 128, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_47)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{1, 128, 8, 128}, {1, 128, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3-6
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-16
            TD_DEFAULT,
            TD_DEFAULT,
            // 17: key_antiquant_scale, keep 3 dims but break D so old PFA fails on shape body.
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            // 18: key_antiquant_offset
            TD_DEFAULT,
            // 19: value_antiquant_scale
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            // 20: value_antiquant_offset
            TD_DEFAULT,
            // 21-31
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
            {{{1, 128, 8, 128}, {1, 128, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_48)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{1, 128, 8, 128}, {1, 128, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3-6
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-16
            TD_DEFAULT,
            TD_DEFAULT,
            // 17: key_antiquant_scale, valid BSND antiquant shape.
            {{{1, 8, 128}, {1, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 18: key_antiquant_offset, intentionally use 2 dims to hit old-PFA offset dim-num reject.
            {{{1, 1024}, {1, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 19: value_antiquant_scale
            {{{1, 8, 128}, {1, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 20: value_antiquant_offset
            {{{1, 1024}, {1, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 21-31
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
            {{{1, 128, 8, 128}, {1, 128, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_49)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{1, 128, 8, 128}, {1, 128, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3-6
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-16
            TD_DEFAULT,
            TD_DEFAULT,
            // 17: key_antiquant_scale
            {{{1, 8, 128}, {1, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 18: key_antiquant_offset, keep 3 dims but break D to hit old-PFA offset shape body.
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            // 19: value_antiquant_scale
            {{{1, 8, 128}, {1, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 20: value_antiquant_offset
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            // 21-31
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
            {{{1, 128, 8, 128}, {1, 128, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_50)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, reuse simple BSH PA antiquant routing but stay on generic antiquant input path.
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3-6
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-11
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 12: antiquant_scale, 1D size=3 should hit old-PFA per-tensor dim[0] reject.
            {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 13: antiquant_offset
            TD_DEFAULT,
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
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
            TD_DEFAULT,
        },
        {
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_51)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3-6
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-11
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 12: antiquant_scale, valid per-tensor first dim.
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 13: antiquant_offset, break dim[0] to hit old-PFA per-tensor offset reject.
            {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
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
            TD_DEFAULT,
        },
        {
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_52)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, keep the old-PFA PA antiquant route on simple BSH inputs.
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key, paged-cache KV in int8 enables PA per-channel antiquant validation.
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3: pse_shift, use a fully valid 4D shape so the case can reach old-PFA's "not support PSE" branch.
            {{{1, 8, 128, 256}, {1, 8, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 4: atten_mask, keep a valid sparse_mode=3 mask so the case is not intercepted by earlier mask-null checks.
            {{{1, 1, 2048, 2048}, {1, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5: actual_seq_lengths
            TD_DEFAULT,
            // 6: actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: generic quant / antiquant optional inputs remain empty.
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15: query_padding_size
            TD_DEFAULT,
            // 16: kv_padding_size
            TD_DEFAULT,
            // 17: key_antiquant_scale
            {{{1, 1024}, {1, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 18: key_antiquant_offset
            TD_DEFAULT,
            // 19: value_antiquant_scale
            {{{1, 1024}, {1, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 20-31: remaining optional inputs
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
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_53)
{
    int64_t actualSeqQ[] = {16};
    int64_t actualSeqKv[] = {16};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, standard old-PFA BNSD query layout.
            {{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask, valid 2D no-mask shape lets FIA fall through and old PFA execute its 2D mask branch.
            {{{16, 16}, {16, 16}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5: actual_seq_lengths_q
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-31: remaining optional inputs stay empty so FIA can fall through to old PFA directly.
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
            // 0: attention_out
            {{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            // Force the shared PFA BNSD path and keep the case away from PA / antiquant special handling.
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            // Deliberately mismatch kv heads so the case keeps a stable GRAPH_FAILED after entering old PFA.
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_54)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, BNSD keeps the shared old-PFA paged antiquant route but fails the PA layout whitelist.
            {{{1, 8, 128, 128}, {1, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key, paged-cache KV in int8.
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask, keep the same valid sparse_mode=3 mask used by the working PA antiquant route.
            {{{1, 1, 2048, 2048}, {1, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5: actual_seq_lengths_q
            TD_DEFAULT,
            // 6: actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: generic quant / antiquant optional inputs remain empty.
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15: query_padding_size
            TD_DEFAULT,
            // 16: kv_padding_size
            TD_DEFAULT,
            // 17: key_antiquant_scale, valid BNSD per-channel antiquant shape.
            {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 18: key_antiquant_offset
            TD_DEFAULT,
            // 19: value_antiquant_scale
            {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 20-31: remaining optional inputs
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
            {{{1, 8, 128, 128}, {1, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_55)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, stay on the already validated BSH paged antiquant path.
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            {{{1, 1, 2048, 2048}, {1, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5: actual_seq_lengths_q
            TD_DEFAULT,
            // 6: actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: generic quant / antiquant optional inputs remain empty.
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15: query_padding_size
            TD_DEFAULT,
            // 16: kv_padding_size
            TD_DEFAULT,
            // 17: key_antiquant_scale, valid BSH per-channel antiquant shape.
            {{{1, 1024}, {1, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 18: key_antiquant_offset
            TD_DEFAULT,
            // 19: value_antiquant_scale
            {{{1, 1024}, {1, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 20-31: remaining optional inputs
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
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            // Deliberately leave the shared PA antiquant route but fail its high-performance-only check.
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_56)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask, keep a valid mask so the case reaches old PFA and fails specifically on sparse_mode.
            {{{1, 1, 2048, 2048}, {1, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5: actual_seq_lengths_q
            TD_DEFAULT,
            // 6: actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: generic quant / antiquant optional inputs remain empty.
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15: query_padding_size
            TD_DEFAULT,
            // 16: kv_padding_size
            TD_DEFAULT,
            // 17: key_antiquant_scale
            {{{1, 1024}, {1, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 18: key_antiquant_offset
            TD_DEFAULT,
            // 19: value_antiquant_scale
            {{{1, 1024}, {1, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 20-31: remaining optional inputs
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
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            // Keep the rest of the PA antiquant setup valid, then fail only on the shared sparse_mode=3 requirement.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_57)
{
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, keep the simple shared old-PFA BSH route with q/k headDim=192.
            {{{1, 128, 1536}, {1, 128, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key, match query headDim so FIA's own checker can fall through.
            {{{1, 128, 1536}, {1, 128, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value, use the only q|k != v combination FIA explicitly allows, so old PFA can reject it later.
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask, leave empty so FIA falls through without sparse-mask side effects.
            TD_DEFAULT,
            // 5-31: no extra optional inputs; keep the cross-op route as plain old-PFA BSH as possible.
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
            // 0: attention_out
            {{{1, 128, 1536}, {1, 128, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

} // namespace FusedInferAttentionScoreUT
