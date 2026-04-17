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

} // namespace FusedInferAttentionScoreUT
