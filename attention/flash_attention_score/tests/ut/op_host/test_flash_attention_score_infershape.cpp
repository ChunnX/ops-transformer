/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class FlashAttentionScoreProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FlashAttentionScoreProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FlashAttentionScoreProto TearDown" << std::endl;
    }
};

TEST_F(FlashAttentionScoreProto, flash_attention_score_infershape_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FlashAttentionScore",
        // 输入Tensor
        {
        // q
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // k: S2=256, B=1, H2=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // v: S2=256, B=1, H2=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // real_shift
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // atten_mask: S1=256, S2=256
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// 输出Tensor
        // softmaxMax: B=1, N1=1, S1=256, 8
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // softmaxSum: B=1, N1=1, S1=256, 8
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // softmaxOut
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // attentionOut
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {// 属性
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
        {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
        {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
        {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
        {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 1, 256, 8},  // softmaxMax
        {1, 1, 256, 8},  // softmaxSum
        {0, 0, 0, 0},    // softmaxOut
        {256, 1, 128},   // attentionOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FlashAttentionScoreProto, flash_attention_score_infershape_invalid_layout)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FlashAttentionScore",
        {
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("INVALID")},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(FlashAttentionScoreProto, flash_attention_score_infershape_bsh)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FlashAttentionScore",
        {
            {{{1, 256, 128}, {1, 256, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 256, 128}, {1, 256, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 256, 128}, {1, 256, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 1, 256, 8},
        {1, 1, 256, 8},
        {0, 0, 0, 0},
        {1, 256, 128},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FlashAttentionScoreProto, flash_attention_score_infershape_bsnd)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FlashAttentionScore",
        {
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 1, 256, 8},
        {1, 1, 256, 8},
        {0, 0, 0, 0},
        {1, 256, 1, 64},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FlashAttentionScoreProto, flash_attention_score_infershape_bnsd)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FlashAttentionScore",
        {
            {{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 128}, {128, 128}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
            {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 1, 128, 8},
        {1, 1, 128, 8},
        {0, 0, 0, 0},
        {1, 1, 128, 64},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FlashAttentionScoreProto, flash_attention_score_infershape_tnd)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FlashAttentionScore",
        {
            {{{81, 2, 192}, {81, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{81, 2, 192}, {81, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{81, 2, 192}, {81, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0721687f)},
            {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {81, 2, 8},
        {81, 2, 8},
        {0, 0, 0, 0},
        {81, 2, 192},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FlashAttentionScoreProto, flash_attention_score_infershape_hifloat8)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FlashAttentionScore",
        {
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.25f)},
            {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 1, 128, 1},
        {1, 1, 128, 1},
        {0, 0, 0, 0},
        {1, 128, 1, 128},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FlashAttentionScoreProto, flash_attention_score_infershape_sbh_n1_zero)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FlashAttentionScore",
        {
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 0, 256, 8},
        {1, 0, 256, 8},
        {0, 0, 0, 0},
        {256, 1, 0},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FlashAttentionScoreProto, flash_attention_score_infershape_sbh_n2_zero)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FlashAttentionScore",
        {
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 1, 0}, {256, 1, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 1, 256, 8},
        {1, 1, 256, 8},
        {0, 0, 0, 0},
        {256, 1, 128},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

class FlashAttentionScoreInferDType : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FlashAttentionScoreInferDType SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "FlashAttentionScoreInferDType TearDown" << std::endl; }
};

static ge::graphStatus RunInferDType(ge::DataType inputDtype, int64_t outDtypeAttr,
                                     std::vector<ge::DataType>& outputDtypes)
{
    std::vector<void*> inputDataTypes;
    inputDataTypes.emplace_back(&inputDtype);
    inputDataTypes.emplace_back(&inputDtype);
    inputDataTypes.emplace_back(&inputDtype);

    ge::DataType out0 = ge::DT_UNDEFINED;
    ge::DataType out1 = ge::DT_UNDEFINED;
    ge::DataType out2 = ge::DT_UNDEFINED;
    ge::DataType out3 = ge::DT_UNDEFINED;
    std::vector<void*> outputDataTypes;
    outputDataTypes.emplace_back(&out0);
    outputDataTypes.emplace_back(&out1);
    outputDataTypes.emplace_back(&out2);
    outputDataTypes.emplace_back(&out3);

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .SetOpType("FlashAttentionScore")
            .IrInstanceNum({1, 1, 1}, {1, 1, 1, 1})
            .InputDataTypes(inputDataTypes)
            .OutputDataTypes(outputDataTypes)
            .NodeAttrs({
                {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
                {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
                {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
                {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
                {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtypeAttr)},
                {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("FlashAttentionScore")->infer_datatype;
    auto ret = inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>());
    if (ret == ge::GRAPH_SUCCESS) {
        outputDtypes.resize(4);
        for (int i = 0; i < 4; i++) {
            outputDtypes[i] = contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(i);
        }
    }
    return ret;
}

TEST_F(FlashAttentionScoreInferDType, fp16_normal)
{
    std::vector<ge::DataType> outputDtypes;
    auto ret = RunInferDType(ge::DT_FLOAT16, 0, outputDtypes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(outputDtypes[0], ge::DT_FLOAT);
    EXPECT_EQ(outputDtypes[1], ge::DT_FLOAT);
    EXPECT_EQ(outputDtypes[2], ge::DT_FLOAT16);
    EXPECT_EQ(outputDtypes[3], ge::DT_FLOAT16);
}

TEST_F(FlashAttentionScoreInferDType, bf16_normal)
{
    std::vector<ge::DataType> outputDtypes;
    auto ret = RunInferDType(ge::DT_BF16, 0, outputDtypes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(outputDtypes[0], ge::DT_FLOAT);
    EXPECT_EQ(outputDtypes[1], ge::DT_FLOAT);
    EXPECT_EQ(outputDtypes[2], ge::DT_BF16);
    EXPECT_EQ(outputDtypes[3], ge::DT_BF16);
}

TEST_F(FlashAttentionScoreInferDType, fp32_normal)
{
    std::vector<ge::DataType> outputDtypes;
    auto ret = RunInferDType(ge::DT_FLOAT, 0, outputDtypes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(outputDtypes[0], ge::DT_FLOAT);
    EXPECT_EQ(outputDtypes[1], ge::DT_FLOAT);
    EXPECT_EQ(outputDtypes[2], ge::DT_FLOAT);
    EXPECT_EQ(outputDtypes[3], ge::DT_FLOAT);
}

TEST_F(FlashAttentionScoreInferDType, fp8_e4m3fn_quant)
{
    std::vector<ge::DataType> outputDtypes;
    auto ret = RunInferDType(ge::DT_FLOAT8_E4M3FN, 0, outputDtypes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(outputDtypes[0], ge::DT_FLOAT);
    EXPECT_EQ(outputDtypes[1], ge::DT_FLOAT);
    EXPECT_EQ(outputDtypes[2], ge::DT_BF16);
    EXPECT_EQ(outputDtypes[3], ge::DT_BF16);
}

TEST_F(FlashAttentionScoreInferDType, fp8_e5m2_quant)
{
    std::vector<ge::DataType> outputDtypes;
    auto ret = RunInferDType(ge::DT_FLOAT8_E5M2, 0, outputDtypes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(outputDtypes[0], ge::DT_FLOAT);
    EXPECT_EQ(outputDtypes[1], ge::DT_FLOAT);
    EXPECT_EQ(outputDtypes[2], ge::DT_BF16);
    EXPECT_EQ(outputDtypes[3], ge::DT_BF16);
}

TEST_F(FlashAttentionScoreInferDType, hifloat8_quant)
{
    std::vector<ge::DataType> outputDtypes;
    auto ret = RunInferDType(ge::DT_HIFLOAT8, 0, outputDtypes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(outputDtypes[0], ge::DT_FLOAT);
    EXPECT_EQ(outputDtypes[1], ge::DT_FLOAT);
    EXPECT_EQ(outputDtypes[2], ge::DT_BF16);
    EXPECT_EQ(outputDtypes[3], ge::DT_BF16);
}