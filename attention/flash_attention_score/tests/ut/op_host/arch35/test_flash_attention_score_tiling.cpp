/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/flash_attention_score_tiling_common.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class FlashAttentionScoreTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FlashAttentionScoreTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FlashAttentionScoreTiling TearDown" << std::endl;
    }
};

// BSH
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_tiling_0)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND910_95};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         // q
        {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // k
         {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // v
         {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // real_shift
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // drop_mask
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // padding_mask
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // atten_mask
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
        {
         // 输出Tensor
         // softmaxMax
         {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum
         {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{0, 0, 0, 0}, {0, 0, 0, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // attentionOut
         {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}
         },
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
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo,"Ascend910_95",64,262144,8192);
    int64_t expectTilingKey = 2400418603268571936;
    std::string expectTilingData = "1 1 1 256 256 0 128 128 0 4446465452318654464 65536 65536 0 0 0 16908546 1099511627776 1 0 0 0 0 0 0 255 0 0 0 0 0 0 0 0 0 0 0 2 2 2 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}


// BNSD
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_tiling_1)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND910_95};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         // q
        {{{1, 1, 128, 256}, {1, 1, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // k
         {{{1, 1, 128, 256}, {1, 1, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // v
         {{{1, 1, 128, 256}, {1, 1, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // real_shift
         {{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // drop_mask
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // padding_mask
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // atten_mask
         {{{128, 128}, {128, 128}}, ge::DT_UINT8, ge::FORMAT_ND},
         // prefix
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // actual_seq_qlen
          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // actual_seq_kvlen
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // q_start_idx
          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // kv_start_idx
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},         
         // dScaleQ
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // dScaleK
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // dScaleV
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // queryRope
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // keyRope
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
         // 输出Tensor
         // softmaxMax
         {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum
         {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{1, 1, 128, 256}, {1, 1, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{1, 1, 128, 256}, {1, 1, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.8f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo,"Ascend910_95",64,262144,8192);
    int64_t expectTilingKey = 2526519393640251952;
    std::string expectTilingData = "1 1 1 128 128 0 256 256 0 4467570831413529805 2147483647 2147483647 128 128 1 281474993618947 549755813888 1 0 0 2147483647 2147483647 0 0 204 0 0 0 0 0 0 0 0 0 0 0 2 2 2 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 ";
    std::vector<size_t> expectWorkspaces = {17170432};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// TND D=192
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_tiling_2)
{
    int64_t actual_seq_qlist[] = {13,14,81};
    int64_t actual_seq_kvlist[] = {13,14,81};
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND910_95};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         // q
        {{{81, 2, 192}, {81, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
         // k
         {{{81, 2, 192}, {81, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
         // v
         {{{81, 2, 192}, {81, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
         // real_shift
         {{{1,2,1024,67}, {1,2,1024,67}}, ge::DT_BF16, ge::FORMAT_ND},
         // drop_mask
         {{{81,2,192}, {81,2,192}}, ge::DT_UINT8, ge::FORMAT_ND},
         // padding_mask
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // atten_mask
        {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
         // prefix
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // actual_seq_qlen
          {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND,true,actual_seq_qlist},
         // actual_seq_kvlen
         {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND,true,actual_seq_kvlist},
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
        {
         // 输出Tensor
         // softmaxMax
        {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum
         {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{0,0,0}, {0,0,0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{81, 2, 192}, {81, 2, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0721687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.9f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo,"Ascend910_95",64,262144,8192);
    int64_t expectTilingKey = 2522015793744446016;
    std::string expectTilingData = "3 2 1 81 67 0 192 192 0 4437115660700903014 67 0 1024 67 1 74027927481745412 8796093026561 0 0 0 67 0 2 0 229 0 0 0 0 0 0 0 0 0 0 0 8 8 2 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 7 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 1179648 8 8 8 ";
    std::vector<size_t> expectWorkspaces = {17966592};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// TND d=126
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_tiling_3)
{
    int64_t actual_seq_qlist[] = {169,333};
    int64_t actual_seq_kvlist[] = {169,297};
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND910_95};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         // q
        {{{333, 4, 126}, {333, 4, 126}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // k: S2=256, B=1, H2=128
         {{{297, 4, 126}, {297, 4, 126}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // v: S2=256, B=1, H2=128
         {{{297, 4, 126}, {297, 4, 126}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // real_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // drop_mask
         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         // padding_mask
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // atten_mask: S1=256, S2=256
        {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         // prefix
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // actual_seq_qlen
          {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND,true,actual_seq_qlist},
         // actual_seq_kvlen
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND,true,actual_seq_kvlist},
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
        {
         // 输出Tensor
         // softmaxMax: B=1, N1=1, S1=256, 8
        {{{333, 4, 8}, {333, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: B=1, N1=1, S1=256, 8
         {{{333, 4, 8}, {333, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{0,0,0}, {0,0,0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{333, 4, 126}, {333, 4, 126}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.1767766f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo,"Ascend910_95",64,262144,8192);
    int64_t expectTilingKey = 2328361009230643776;
    std::string expectTilingData = "2 4 1 333 169 0 126 126 0 4482494421136310272 65536 65536 0 0 0 260 0 0 0 0 65536 65536 2 0 255 0 0 0 0 0 0 0 0 0 0 0 24 24 3 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// BF16
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_tiling_4)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND910_95};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         // q
        {{{1, 1, 128, 384}, {1, 1, 128, 384}}, ge::DT_BF16, ge::FORMAT_ND},
         // k: S2=256, B=1, H2=128
         {{{1, 1, 128, 384}, {1, 1, 128, 384}}, ge::DT_BF16, ge::FORMAT_ND},
         // v: S2=256, B=1, H2=128
         {{{1, 1, 128, 384}, {1, 1, 128, 384}}, ge::DT_BF16, ge::FORMAT_ND},
         // real_shift
         {{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         // drop_mask
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // padding_mask
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // atten_mask: S1=256, S2=256
         {{{128, 128}, {128, 128}}, ge::DT_UINT8, ge::FORMAT_ND},
         // prefix
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // actual_seq_qlen
          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // actual_seq_kvlen
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // q_start_idx
          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // kv_start_idx
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},         
         // dScaleQ
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // dScaleK
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // dScaleV
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // queryRope
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         // keyRope
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         },
        {
         // 输出Tensor
         // softmaxMax: B=1, N1=1, S1=256, 8
         {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: B=1, N1=1, S1=256, 8
         {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{1, 1, 128, 384}, {1, 1, 128, 384}}, ge::DT_BF16, ge::FORMAT_ND},
         // attentionOut
         {{{1, 1, 128, 384}, {1, 1, 128, 384}}, ge::DT_BF16, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(12)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(33)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo,"Ascend910_95",64,262144,8192);
    int64_t expectTilingKey = 2526519393908687664;
    std::string expectTilingData = "1 1 1 128 128 0 384 384 0 4467570831408496640 2147483647 2147483647 128 128 1 281474993618947 549755813888 1 0 0 2147483647 2147483647 12 33 127 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {18546688};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
// basic.cpp cov
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_tiling_5)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND910_95};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         // q
        {{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         // k: S2=256, B=1, H2=128
         {{{1, 1, 255, 64}, {1, 1, 255, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         // v: S2=256, B=1, H2=128
         {{{1, 1, 255, 64}, {1, 1, 255, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         // real_shift
         {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},
         // drop_mask bnss
         {{{1, 1, 128,255}, {1, 1, 128,255}}, ge::DT_UINT8, ge::FORMAT_ND},
         // padding_mask
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // atten_mask: S1=256, S2=256
         {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
         // prefix
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // actual_seq_qlen
          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // actual_seq_kvlen
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // q_start_idx
          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // kv_start_idx
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},         
         // dScaleQ
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // dScaleK
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // dScaleV
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // queryRope
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         // keyRope
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         },
        {
         // 输出Tensor
         // softmaxMax: B=1, N1=1, S1=256, 8
         {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: B=1, N1=1, S1=256, 8
         {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         // attentionOut
         {{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.6f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo,"Ascend910_95",64,262144,8192);
    int64_t expectTilingKey = 2531022991657009712;
    std::string expectTilingData = "1 1 1 128 255 0 64 64 0 4467570831410174362 65409 127 0 0 1 73183506846581507 8796093022209 2 0 0 128 128 6 24 153 0 0 0 0 0 0 0 0 0 0 0 2 2 2 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2 ";
    std::vector<size_t> expectWorkspaces = {16809984};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
// TND d=384
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_tiling_6)
{
    int64_t actual_seq_qlist[] = {64,192};
    int64_t actual_seq_kvlist[] = {64,192};
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND910_95};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         // q
        {{{192, 4, 384}, {192, 4, 384}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // k: S2=256, B=1, H2=128
         {{{192, 4, 384}, {192, 4, 384}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // v: S2=256, B=1, H2=128
         {{{192, 4, 384}, {192, 4, 384}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // real_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // drop_mask
         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         // padding_mask
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // atten_mask: S1=256, S2=256
        {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         // prefix
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // actual_seq_qlen
          {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND,true,actual_seq_qlist},
         // actual_seq_kvlen
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND,true,actual_seq_kvlist},
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
        {
         // 输出Tensor
         // softmaxMax: B=1, N1=1, S1=256, 8
        {{{192, 4, 8}, {192, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: B=1, N1=1, S1=256, 8
         {{{192, 4, 8}, {192, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{0,0,0}, {0,0,0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{192, 4, 384}, {192, 4, 384}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0510310f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo,"Ascend910_95",64,262144,8192);
    int64_t expectTilingKey = 2328361010304385856;
    std::string expectTilingData = "2 4 1 192 128 0 384 384 0 4418319178713268224 65536 65536 0 0 0 260 0 0 0 0 65536 65536 2 0 255 0 0 0 0 0 0 0 0 0 0 0 8 8 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 7 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 ";
    std::vector<size_t> expectWorkspaces = {30932992};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
// TND d=256
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_tiling_7)
{
    int64_t actual_seq_qlist[3] = {13,14,81};
    int64_t actual_seq_kvlist[3] = {13,14,81};
    int64_t prefix[3] = {10, 12, 60};
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND910_95};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         // q
        {{{81,2,256}, {81,2,256}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // k: S2=256, B=1, H2=128
         {{{81,2,256}, {81,2,256}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // v: S2=256, B=1, H2=128
         {{{81,2,256}, {81,2,256}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // real_shift
         {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // drop_mask
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // padding_mask
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // atten_mask: S1=256, S2=256
        //  {{{3072,2048}, {3072,2048}}, ge::DT_UINT8, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         // prefix
         {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND,true,prefix},
         // actual_seq_qlen
          {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND,true,actual_seq_qlist},
         // actual_seq_kvlen
         {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND,true,actual_seq_kvlist},
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
        {
         // 输出Tensor
         // softmaxMax: B=1, N1=1, S1=256, 8
         {{{81,2,8}, {81,2,8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: B=1, N1=1, S1=256, 8
         {{{81,2,8}, {81,2,8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{0, 0, 0}, {0, 0, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // attentionOut
         {{{81,2,256}, {81,2,256}}, ge::DT_FLOAT, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0625f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.7f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo,"Ascend910_95",64,262144,8192);
    int64_t expectTilingKey = 2458965399229694784;
    std::string expectTilingData = "3 2 1 81 67 0 256 256 0 4431542034392888115 65536 65536 0 0 1 17179869956 0 2 0 0 65536 65536 2 0 178 0 0 0 0 0 0 0 0 0 0 0 6 6 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 ";
    std::vector<size_t> expectWorkspaces = {19136512};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}