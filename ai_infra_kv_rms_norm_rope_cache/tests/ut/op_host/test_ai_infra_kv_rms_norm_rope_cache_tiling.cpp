/**

Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
/*!

\file repeat_interleave_grad_proto.h
\brief
*/
#include "../../../op_host/ai_infra_kv_rms_norm_rope_cache_tiling.h"
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
// using namespace ut_util;
using namespace std;
// using namespace ge;

class AiInfraKvRmsNormRopeCacheTiling : public testing::Test
{
protected:
static void SetUpTestCase()
{
std::cout << "AiInfraKvRmsNormRopeCacheTiling SetUp" << std::endl;
}

static void TearDownTestCase()
{
    std::cout << "AiInfraKvRmsNormRopeCacheTiling TearDown" << std::endl;
}
};

TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_PA_static_quant_contiguous_default_5011_pangu_000000)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "PA_BNSD";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "static";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 5011;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_PA_static_quant_contiguous_default_5011_pangu_000001)
{
int64_t batch_size = 32;
int64_t num_head = 1;
int64_t seq_len = 15;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 125;
int64_t block_size = 10;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "PA_BNSD";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "static";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{32, 1, 15, 192}, {32, 1, 15, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{32, 1, 15, 64}, {32, 1, 15, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{32, 1, 15, 64}, {32, 1, 15, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{480}, {480}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{125, 10, 1, 192}, {125, 10, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{125, 10, 1, 128}, {125, 10, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{32, 1, 15, 128}, {32, 1, 15, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{125, 10, 1, 192}, {125, 10, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{125, 10, 1, 128}, {125, 10, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{32, 1, 15, 192}, {32, 1, 15, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{32, 1, 15, 128}, {32, 1, 15, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 5011;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_PA_BLK_BNSD_static_quant_contiguous_default_5010_pangu_000000)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "PA_BLK_BNSD";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "static";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 5010;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_PA_BLK_BNSD_static_quant_contiguous_default_5010_pangu_000001)
{
int64_t batch_size = 9;
int64_t num_head = 1;
int64_t seq_len = 14;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 21;
int64_t block_size = 16;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "PA_BLK_BNSD";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "static";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{9, 1, 14, 192}, {9, 1, 14, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{9, 1, 14, 64}, {9, 1, 14, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{9, 1, 14, 64}, {9, 1, 14, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{9}, {9}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{21, 16, 1, 192}, {21, 16, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{21, 16, 1, 128}, {21, 16, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{9, 1, 14, 128}, {9, 1, 14, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{21, 16, 1, 192}, {21, 16, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{21, 16, 1, 128}, {21, 16, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{9, 1, 14, 192}, {9, 1, 14, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{9, 1, 14, 128}, {9, 1, 14, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 5010;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_PA_noquant_contiguous_default_5001_pangu_000002)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "PA_BNSD";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 5001;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_PA_noquant_contiguous_default_5001_pangu_000003)
{
int64_t batch_size = 30;
int64_t num_head = 1;
int64_t seq_len = 10;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 42;
int64_t block_size = 12;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "PA_BNSD";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{30, 1, 10, 192}, {30, 1, 10, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{30, 1, 10, 64}, {30, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{30, 1, 10, 64}, {30, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{300}, {300}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{42, 12, 1, 192}, {42, 12, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{42, 12, 1, 128}, {42, 12, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{30, 1, 10, 128}, {30, 1, 10, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{42, 12, 1, 192}, {42, 12, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{42, 12, 1, 128}, {42, 12, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{30, 1, 10, 192}, {30, 1, 10, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{30, 1, 10, 128}, {30, 1, 10, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 5001;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_PA_BLK_BNSD_noquant_contiguous_default_5000_pangu_000002)
{
int64_t batch_size = 38;
int64_t num_head = 1;
int64_t seq_len = 3809;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 230;
int64_t block_size = 745;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "PA_BLK_BNSD";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{38, 1, 3809, 576}, {38, 1, 3809, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{38, 1, 3809, 64}, {38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{38, 1, 3809, 64}, {38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{228}, {228}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{230, 745, 1, 64}, {230, 745, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{230, 745, 1, 512}, {230, 745, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{230, 745, 1, 64}, {230, 745, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{230, 745, 1, 512}, {230, 745, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{38, 1, 3809, 64}, {38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{38, 1, 3809, 512}, {38, 1, 3809, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 5000;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_PA_BLK_BNSD_noquant_contiguous_default_5000_pangu_000003)
{
int64_t batch_size = 9;
int64_t num_head = 1;
int64_t seq_len = 14;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 21;
int64_t block_size = 16;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "PA_BLK_BNSD";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{9, 1, 14, 192}, {9, 1, 14, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{9, 1, 14, 64}, {9, 1, 14, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{9, 1, 14, 64}, {9, 1, 14, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{9}, {9}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{21, 16, 1, 192}, {21, 16, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{21, 16, 1, 128}, {21, 16, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{9, 1, 14, 128}, {9, 1, 14, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{21, 16, 1, 192}, {21, 16, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{21, 16, 1, 128}, {21, 16, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{9, 1, 14, 192}, {9, 1, 14, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{9, 1, 14, 128}, {9, 1, 14, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 5000;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_PA_NZ_static_quant_contiguous_default_4011_pangu_000000)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "PA_NZ";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "static";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 4011;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_PA_NZ_static_quant_contiguous_default_4011_pangu_000001)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 192;
int64_t block_size = 128;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "PA_NZ";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "static";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{192}, {192}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 4011;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_PA_BLK_NZ_static_quant_contiguous_default_4010_pangu_000001)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "PA_BLK_NZ";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "static";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 4010;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_PA_BLK_NZ_static_quant_contiguous_default_4010_pangu_000002)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 192;
int64_t block_size = 128;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "PA_BLK_NZ";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "static";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{192}, {192}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 4010;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_PA_NZ_noquant_contiguous_default_4001_pangu_000002)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "PA_NZ";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 4001;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_PA_NZ_noquant_contiguous_default_4001_pangu_000003)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 192;
int64_t block_size = 128;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "PA_NZ";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 4001;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_PA_BLK_NZ_noquant_contiguous_default_4000_pangu_000003)
{
int64_t batch_size = 38;
int64_t num_head = 1;
int64_t seq_len = 3809;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 230;
int64_t block_size = 745;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "PA_BLK_NZ";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{38, 1, 3809, 576}, {38, 1, 3809, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{38, 1, 3809, 64}, {38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{38, 1, 3809, 64}, {38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{228}, {228}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{230, 745, 1, 64}, {230, 745, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{230, 745, 1, 512}, {230, 745, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{230, 745, 1, 64}, {230, 745, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{230, 745, 1, 512}, {230, 745, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{38, 1, 3809, 64}, {38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{38, 1, 3809, 512}, {38, 1, 3809, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 4000;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_PA_BLK_NZ_noquant_contiguous_default_4000_pangu_000004)
{
int64_t batch_size = 38;
int64_t num_head = 1;
int64_t seq_len = 3809;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 230;
int64_t block_size = 745;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "PA_BLK_NZ";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{38, 1, 3809, 192}, {38, 1, 3809, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{38, 1, 3809, 64}, {38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{38, 1, 3809, 64}, {38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{228}, {228}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{230, 745, 1, 192}, {230, 745, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{230, 745, 1, 128}, {230, 745, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{38, 1, 3809, 128}, {38, 1, 3809, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{230, 745, 1, 192}, {230, 745, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{230, 745, 1, 128}, {230, 745, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{38, 1, 3809, 192}, {38, 1, 3809, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{38, 1, 3809, 128}, {38, 1, 3809, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 4000;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_norm_static_quant_contiguous_default_3010_pangu_000000)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 7;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 192;
int64_t block_size = 128;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "static";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 3010;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_norm_asym_quant_contiguous_default_3010_pangu_000001)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 7;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 192;
int64_t block_size = 128;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "static";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 3010;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_PA_noquant_contiguous_default_3001_pangu_000002)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 7;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "PA";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 7, 576}, {64, 1, 7, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 7, 512}, {64, 1, 7, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 3001;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_PA_noquant_contiguous_default_3001_pangu_000003)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 7;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 192;
int64_t block_size = 128;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "PA";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 3001;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_contiguous_default_3000_pangu_000004)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 7;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 7, 576}, {64, 1, 7, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 7, 512}, {64, 1, 7, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 3000;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_norm_noquant_contiguous_default_3000_pangu_000005)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 7;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 192;
int64_t block_size = 128;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 3000;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_PA_noquant_contiguous_default_2001_pangu_000000)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 7;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "PA";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 7, 576}, {64, 1, 7, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 7, 512}, {64, 1, 7, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 2001;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_PA_noquant_contiguous_default_2001_pangu_000001)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 7;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 192;
int64_t block_size = 128;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "PA";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 2001;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_PA_noquant_contiguous_default_3001_pangu_000006)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 192;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "PA";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64, 1}, {64, 1}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{192, 128, 1, 64}, {192, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{192, 128, 1, 512}, {192, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{192, 128, 1, 64}, {192, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{192, 128, 1, 512}, {192, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 3001;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_PA_noquant_contiguous_default_3001_pangu_000007)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 192;
int64_t block_size = 128;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "PA";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64, 1}, {64, 1}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 3001;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_norm_static_quant_contiguous_default_1011_pangu_000000)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 7;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 192;
int64_t block_size = 128;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "static";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 1011;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_norm_static_quant_contiguous_default_3010_pangu_000008)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 192;
int64_t block_size = 128;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "static";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64, 1}, {64, 1}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 3010;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_contiguous_default_3000_pangu_000009)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64, 1}, {64, 1}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 3000;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_norm_noquant_contiguous_default_3000_pangu_000010)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 192;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 128;
// 属性
string cache_mode = "";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64, 1}, {64, 1}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 3000;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_norm_noquant_contiguous_default_1001_pangu_000004)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 7;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 192;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 128;
// 属性
string cache_mode = "";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{192, 128, 7, 192}, {192, 128, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 1001;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}

TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_PA_noquant_SFA_noquant_default_5001_pangu_000004)
{
int64_t batch_size = 1;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "PA_BNSD";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{1, 1, 1, 576}, {1, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 5001;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_PA_static_quant_SFA_c8_default_5011_pangu_000005)
{
int64_t batch_size = 1;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "PA_BNSD";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "static";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{1, 1, 1, 576}, {1, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 5011;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_empty_kv_in_empty_out_pangu_000000)
{
int64_t batch_size = 0;
int64_t num_head = 1;
int64_t seq_len = 10;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 10, 64}, {64, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 10, 64}, {64, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_contiguous_default_err_kv_dim_not576_pangu_000000)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 500;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 500}, {64, 1, 1, 500}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_contiguous_default_err_rope_dim_not64_pangu_000001)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 32;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 32}, {64, 1, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_empty_kv_in_nonempty_out_pangu_000001)
{
int64_t batch_size = 0;
int64_t num_head = 1;
int64_t seq_len = 10;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 10, 64}, {64, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 10, 64}, {64, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{0, 128, 1, 64}, {0, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{0, 128, 1, 512}, {0, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{0, 1, 10, 64}, {0, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{0, 1, 10, 512}, {0, 1, 10, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}

TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_contiguous_default_err_dv_not512_pangu_000002)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_norm_noquant_contiguous_default_err_vlen_not128_pangu_000003)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_int32_BNSD_norm_noquant_contiguous_default_err_dtype_int32_pangu_000000)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_INT32, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_INT32, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_INT32, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp64_BNSD_norm_noquant_contiguous_default_err_dtype_float64_pangu_000001)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_DOUBLE, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_contiguous_default_err_neg_dim_pangu_000000)
{
int64_t batch_size = -1;
int64_t num_head = 1;
int64_t seq_len = 10;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{-1, 1, 10, 576}, {-1, 1, 10, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 10, 64}, {64, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 10, 64}, {64, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_contiguous_default_err_kv_shape_5dim_pangu_000002)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 1;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 1, 576}, {64, 1, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BN1D_norm_noquant_contiguous_default_err_format_pangu_000000)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 10;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 10}, {64, 1, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 10, 64}, {64, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 10, 64}, {64, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_scalar_norm_noquant_contiguous_default_err_scalar_tensor_pangu_000000)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 10;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 10, 64}, {64, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 10, 64}, {64, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_contiguous_default_err_null_kv_pangu_000000)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_contiguous_default_err_null_gamma_pangu_000001)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_contiguous_default_err_null_cos_pangu_000002)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_contiguous_default_err_null_sin_pangu_000003)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_norm_noquant_contiguous_default_err_null_index_pangu_000004)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 128;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_norm_noquant_contiguous_default_err_null_k_cache_pangu_000005)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 128;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_norm_noquant_contiguous_default_err_null_ckv_cache_pangu_000006)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 128;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_err_N_zero_pangu_000004)
{
int64_t batch_size = 1;
int64_t num_head = 0;
int64_t seq_len = 10;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{1, 0, 10, 576}, {1, 0, 10, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{1, 0, 10, 64}, {1, 0, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{1, 0, 10, 64}, {1, 0, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{0, 128, 1, 64}, {0, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{0, 128, 1, 512}, {0, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{1, 0, 10, 64}, {1, 0, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{1, 0, 10, 512}, {1, 0, 10, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_err_S_zero_pangu_000005)
{
int64_t batch_size = 1;
int64_t num_head = 1;
int64_t seq_len = 0;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{1, 1, 0, 576}, {1, 1, 0, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 0, 64}, {64, 1, 0, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 0, 64}, {64, 1, 0, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{0, 128, 1, 64}, {0, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{0, 128, 1, 512}, {0, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{1, 1, 0, 64}, {1, 1, 0, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{1, 1, 0, 512}, {1, 1, 0, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_err_D_zero_pangu_000006)
{
int64_t batch_size = 1;
int64_t num_head = 1;
int64_t seq_len = 10;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 0;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 0;
int64_t v_dim = 0;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{1, 1, 10, 0}, {1, 1, 10, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{1, 1, 10, 64}, {1, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{1, 1, 10, 64}, {1, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{0, 128, 1, 64}, {0, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{0, 128, 1, 512}, {0, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{1, 1, 10, 0}, {1, 1, 10, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{1, 1, 10, 0}, {1, 1, 10, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_err_all_zero_pangu_000007)
{
int64_t batch_size = 0;
int64_t num_head = 0;
int64_t seq_len = 0;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 0;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 0;
int64_t v_dim = 0;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{0, 0, 0, 0}, {0, 0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{0, 128, 1, 64}, {0, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{0, 128, 1, 512}, {0, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{0, 0, 0, 0}, {0, 0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{0, 0, 0, 0}, {0, 0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_empty_with_invalid_dim_pangu_000000)
{
int64_t batch_size = 0;
int64_t num_head = 1;
int64_t seq_len = 10;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{0, 1, 10, 576}, {0, 1, 10, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 10, 64}, {64, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 10, 64}, {64, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{0, 128, 1, 64}, {0, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{0, 128, 1, 512}, {0, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{0, 1, 10, 64}, {0, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{0, 1, 10, 512}, {0, 1, 10, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_err_B_zero_pangu_000003)
{
int64_t batch_size = 0;
int64_t num_head = 1;
int64_t seq_len = 10;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{0, 1, 10, 576}, {0, 1, 10, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{0, 1, 10, 64}, {0, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{0, 1, 10, 64}, {0, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{0, 128, 1, 64}, {0, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{0, 128, 1, 512}, {0, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{1, 0, 10, 64}, {1, 0, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{1, 0, 10, 512}, {1, 0, 10, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_norm_noquant_empty_actualS_B0_pangu_000008)
{
int64_t batch_size = 0;
int64_t num_head = 1;
int64_t seq_len = 10;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{0, 1, 10, 192}, {0, 1, 10, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{0, 1, 10, 64}, {0, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{0, 1, 10, 64}, {0, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 192}, {576, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 128}, {576, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{0, 1, 10, 128}, {0, 1, 10, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{0, 128, 1, 192}, {0, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{0, 128, 1, 128}, {0, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{0, 1, 10, 192}, {0, 1, 10, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{0, 1, 10, 128}, {0, 1, 10, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_norm_noquant_empty_actualS_allB0_pangu_000009)
{
int64_t batch_size = 0;
int64_t num_head = 1;
int64_t seq_len = 0;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{0, 1, 0, 192}, {0, 1, 0, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{0, 1, 0, 64}, {0, 1, 0, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{0, 1, 0, 64}, {0, 1, 0, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 192}, {576, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 128}, {576, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{0, 1, 0, 128}, {0, 1, 0, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{0, 128, 1, 192}, {0, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{0, 128, 1, 128}, {0, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{0, 1, 0, 192}, {0, 1, 0, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{0, 1, 0, 128}, {0, 1, 0, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_empty_with_invalid_dtype_pangu_000001)
{
int64_t batch_size = 0;
int64_t num_head = 1;
int64_t seq_len = 10;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_INT32, ge::FORMAT_ND},
        // 输入 2: cos
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        // 输入 3: sin
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        // 输入 4: index
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{0, 128, 1, 64}, {0, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{0, 128, 1, 512}, {0, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{0, 1, 10, 64}, {0, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{0, 1, 10, 512}, {0, 1, 10, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_PA_static_quant_empty_with_invalid_quant_pangu_000003)
{
int64_t batch_size = 0;
int64_t num_head = 1;
int64_t seq_len = 10;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "PA_BNSD";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "invalid_quant";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{0, 1, 10, 192}, {0, 1, 10, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{0, 1, 10, 64}, {0, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{0, 1, 10, 64}, {0, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 192}, {576, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 128}, {576, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{0, 1, 10, 128}, {0, 1, 10, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{0, 128, 1, 192}, {0, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{0, 128, 1, 128}, {0, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{0, 1, 10, 192}, {0, 1, 10, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{0, 1, 10, 128}, {0, 1, 10, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_contiguous_default_err_numHead_not1_pangu_000004)
{
int64_t batch_size = 64;
int64_t num_head = 2;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 2, 1, 576}, {64, 2, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 2, 1, 64}, {64, 2, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 2, 1, 64}, {64, 2, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_norm_noquant_contiguous_default_err_numHead_invalid_pangu_000005)
{
int64_t batch_size = 64;
int64_t num_head = 3;
int64_t seq_len = 1;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 3, 1, 192}, {64, 3, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 3, 1, 64}, {64, 3, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 3, 1, 64}, {64, 3, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 192}, {576, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 128}, {576, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{64, 3, 1, 128}, {64, 3, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_static_quant_contiguous_default_err_norm_quant_pangu_000006)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "static";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_norm_noquant_contiguous_default_err_quant_mode_invalid_pangu_000007)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "invalid_mode";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 192}, {576, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 128}, {576, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_PA_noquant_contiguous_default_err_none_quant_with_scale_pangu_000008)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "PA_BNSD";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_contiguous_default_err_oversize_dim_pangu_000001)
{
int64_t batch_size = 1;
int64_t num_head = 1;
int64_t seq_len = 2147483647;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{1, 1, 2147483647, 576}, {1, 1, 2147483647, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{1, 1, 2147483647, 64}, {1, 1, 2147483647, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{1, 1, 2147483647, 64}, {1, 1, 2147483647, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{2147483647}, {2147483647}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_norm_noquant_contiguous_default_err_kv_shape_3dim_pangu_000003)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 192}, {64, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 192}, {576, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 128}, {576, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_mixed_dtype_BNSD_norm_noquant_contiguous_default_err_dtype_mismatch_pangu_000002)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 192}, {576, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 128}, {576, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_NCHW_norm_noquant_contiguous_default_err_format_pangu_000001)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 10;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 576, 1, 10}, {64, 576, 1, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 10, 64}, {64, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 10, 64}, {64, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{640}, {640}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_contiguous_default_err_scalar_gamma_pangu_000001)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 10;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 10, 576}, {64, 1, 10, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 10, 64}, {64, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 10, 64}, {64, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{640}, {640}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_norm_noquant_contiguous_default_err_scalar_rope_pangu_000002)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 10;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "Norm";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 10, 576}, {64, 1, 10, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 10, 64}, {64, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{640}, {640}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}

// CheckScaleValid DIM_TWO (methodMode_=1): c_kv_scale shape=[1,128], c_kv_offset shape=[1,128]
// Covers: CheckScaleValid scale2Shape DIM_TWO branch + CheckOffsetValid methodMode_=1 DIM_TWO branch
// tilingKey=3010: v2, Norm, CutT, static quant (asym: offset non-empty → quantMode_>1 → CheckOffsetValid called)
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_norm_asym_quant_scale2d_offset2d_3010_pangu_000000)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 7;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 192;
int64_t block_size = 128;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "static";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale, shape=[1,128] (DIM_TWO) → covers CheckScaleValid scale2Shape DIM_TWO
        {{{1, 128}, {1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset, shape=[1,128] (DIM_TWO) → covers CheckOffsetValid offset2Shape DIM_TWO
        {{{1, 128}, {1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 3010;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
// GetShapeAttrsInfo: inputQuantMode_=None but k_rope_scale is non-empty → GRAPH_FAILED
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_norm_noquant_scale_nonempty_failed_pangu_000000)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 7;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 192;
int64_t block_size = 128;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale non-empty with quant_mode=none → GRAPH_FAILED
        {{{192}, {192}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 0;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr);
}
// Covers CheckCosSinValid INT8 branch (line 127) + CheckGammaValid INT8 branch (line 149):
// cache_mode=PA_NZ, k_cache/v_cache dtype=INT8 → kcacheDtype==DT_INT8 / vcacheDtype==DT_INT8
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_PA_NZ_static_quant_int8cache_4011_pangu_000000)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
string cache_mode = "PA_NZ";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "static";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma, dim=512, 512%32==0 → valid for INT8
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos, D=64, 64%32==0 → valid for INT8
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache, dtype=INT8 → covers CheckCosSinValid INT8 branch
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},
        // 输入 6: ckv_cache, dtype=INT8 → covers CheckGammaValid INT8 branch
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},
        // 输入 7: k_rope_scale
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 4011;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
// Covers CheckCosSinValid: cosSeq != seqLen && cosSeq != 1 → return false (line 118)
// Note: ds::GetShapeAttrsInfo discards base return value, so tiling still succeeds.
// ropeSeqLength stays 0 (set after return false), seqLen(7)!=ropeSeqLength(0) → CutB, isMTP_=true
// v2, Norm, no quant, CutB, isMTP_=true → tilingKey=1001

// ======================== SFA New Feature Tests ========================
// Test SFA_NO_QUANT: K_CACHE is 1-dim (null), quantMode=none, rotaryMode=half
// Covers: RotaryMode::HALF, CacheLayout::SFA_NO_QUANT, K_CACHE null detection
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_PA_noquant_SFA_noquant_half_5001_pangu_000006)
{
int64_t batch_size = 1;
int64_t num_head = 1;
int64_t seq_len = 2;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 5000;
int64_t block_size = 64;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "PA_BNSD";
float epsilon = 1e-05f;
string rotary_mode = "half";
string quant_mode = "none";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{1, 1, 2, 576}, {1, 1, 2, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{1, 1, 2, 64}, {1, 1, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{1, 1, 2, 64}, {1, 1, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache (1-dim, SFA: kCache==None triggers GetDimNum()==1)
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{5000, 64, 1, 512}, {5000, 64, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale (empty, no quant)
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale (empty, no quant)
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{5000, 64, 1, 64}, {5000, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{5000, 64, 1, 512}, {5000, 64, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{1, 1, 2, 64}, {1, 1, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{1, 1, 2, 512}, {1, 1, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 5001;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
// Test SFA_C8: K_CACHE is 1-dim (null), quantMode=pertile128, rotaryMode=half
// Covers: QuantClass::Pertile_128, CacheLayout::SFA_C8, pertileSize=128, TilingKeyAttachQuantStat→5021
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_PA_pertile128_SFA_c8_half_5021_pangu_000007)
{
int64_t batch_size = 4;
int64_t num_head = 1;
int64_t seq_len = 2;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 16384;
int64_t block_size = 64;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "PA_BNSD";
float epsilon = 1e-05f;
string rotary_mode = "half";
string quant_mode = "pertile128";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{4, 1, 2, 576}, {4, 1, 2, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{4, 1, 2, 64}, {4, 1, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{4, 1, 2, 64}, {4, 1, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache (1-dim, SFA: kCache==None triggers GetDimNum()==1)
        {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{16384, 64, 1, 512}, {16384, 64, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},
        // 输入 7: k_rope_scale (provided for pertile128 quant)
        {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale (provided for pertile128 quant)
        {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{16384, 64, 1, 64}, {16384, 64, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{16384, 64, 1, 512}, {16384, 64, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{4, 1, 2, 64}, {4, 1, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{4, 1, 2, 512}, {4, 1, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 5021;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
// Covers CheckScaleValid mode0 DIM_TWO branches:
// scale1(k_rope_scale)=[1,64] → lines 98-100, scale2(c_kv_scale)=[1,512] → lines 105-107
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v1_kv_fp16_BNSD_PA_static_quant_scale2d_5011_pangu_000008)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 1;
int64_t rms_size = 512;
int64_t rope_size = 64;
int64_t hidden_size = 576;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 576;
int64_t block_size = 128;
int64_t k_dim = 64;
int64_t v_dim = 512;
// 属性
string cache_mode = "PA_BNSD";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "static";
bool is_output_kv = false;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale, shape=[1,64] (DIM_TWO) → covers mode0 scale1 DIM_TWO
        {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale, shape=[1,512] (DIM_TWO) → covers mode0 scale2 DIM_TWO
        {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 5011;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}
// Covers CheckScaleValid mode1 DIM_TWO scale1 branch:
// scale1(k_rope_scale)=[1,192] → lines 114-115 (GetDim(0)==numHead, GetDim(1)==RMS_NORM_LENGTH)
TEST_F(AiInfraKvRmsNormRopeCacheTiling, ai_infra_kv_RNRC_v2_kv_fp16_BNSD_norm_static_quant_scale2d_3010_pangu_000009)
{
int64_t batch_size = 64;
int64_t num_head = 1;
int64_t seq_len = 7;
int64_t rms_size = 128;
int64_t rope_size = 64;
int64_t hidden_size = 192;
int64_t blockDim;
int64_t cache_length;
int64_t block_num = 192;
int64_t block_size = 128;
int64_t k_dim = 192;
int64_t v_dim = 128;
// 属性
string cache_mode = "";
float epsilon = 1e-05f;
string rotary_mode = "interleave_half";
string quant_mode = "static";
bool is_output_kv = true;

optiling::AiInfraKvRmsNormRopeCacheCompileInfo compileInfo = {};
gert::TilingContextPara tilingContextPara(
    "AiInfraKvRmsNormRopeCache",
    {
        // 输入 0: kv
        {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 1: gamma
        {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 2: cos
        {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 3: sin
        {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 4: index
        {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
        // 输入 5: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 6: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输入 7: k_rope_scale, shape=[1,192] (DIM_TWO) → covers mode1 scale1 DIM_TWO
        {{{1, 192}, {1, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 8: c_kv_scale, shape=[1,128] (DIM_TWO) → covers mode1 scale2 DIM_TWO
        {{{1, 128}, {1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 9: k_rope_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // 输入 10: c_kv_offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        //输入 11：v
        {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        // 输出 1: k_cache
        {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 2: ckv_cache
        {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 3: k_rope_out
        {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // 输出 4: c_kv_out
        {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(rotary_mode)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(quant_mode)},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(is_output_kv)},
    },
    &compileInfo);
int64_t expectTilingKey = 3010;
string expectTilingDataStr = "";
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr);
}