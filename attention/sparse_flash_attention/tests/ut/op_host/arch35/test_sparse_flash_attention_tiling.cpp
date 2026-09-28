/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <tuple>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "register/tilingdata_base.h"

using namespace std;

// DAV_3510 (Ascend950) tiling cases for SparseFlashAttention
class SparseFlashAttentionTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SparseFlashAttentionTilingArch35 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SparseFlashAttentionTilingArch35 TearDown" << std::endl;
    }
};

// BSND query and PA_BSND key layout with valid params, success on Ascend950
TEST_F(SparseFlashAttentionTilingArch35, SparseFlashAttention_950_tiling_0)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {113};
    int64_t actual_seq_kvlist[] = {113};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query            input0
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // key              input1
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // value            input2
            {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},  // sparse_indices   input3
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},                        // block_table      input4
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_query
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
            {{{1, 128, 128, 64}, {1, 128, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope       input5
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}        // key_rope         input6
        },
        {
            {{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // attention_out
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // softmax_max
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                               // softmax_sum
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// layout_kv is invalid
TEST_F(SparseFlashAttentionTilingArch35, SparseFlashAttention_950_tiling_1)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 128, 128, 64}, {1, 128, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("INVALID")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// return_softmax_lse is true while layout_kv is PA_BSND, supported on Ascend950 only
TEST_F(SparseFlashAttentionTilingArch35, SparseFlashAttention_950_tiling_2)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {113};
    int64_t actual_seq_kvlist[] = {113};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query            input0
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // key              input1
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // value            input2
            {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},  // sparse_indices   input3
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},                        // block_table      input4
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_query
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
            {{{1, 128, 128, 64}, {1, 128, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope       input5
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}        // key_rope         input6
        },
        {
            {{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // attention_out
            {{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},    // softmax_max
            {{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}     // softmax_sum
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// Tiling data classes are registered for the op
TEST_F(SparseFlashAttentionTilingArch35, SparseFlashAttention_tiling_data_class_registered)
{
    auto &factory = optiling::CTilingDataClassFactory::GetInstance();
    EXPECT_NE(factory.CreateTilingDataInstance("SparseFlashAttention"), nullptr);
    EXPECT_NE(factory.CreateTilingDataInstance("SparseFlashAttentionBaseParamsMlaOp"), nullptr);
    EXPECT_NE(factory.CreateTilingDataInstance("SparseFlashAttentionSingleCoreParamsMlaOp"), nullptr);
    EXPECT_NE(factory.CreateTilingDataInstance("SparseFlashAttentionSingleCoreTensorSizeMlaOp"), nullptr);
    EXPECT_NE(factory.CreateTilingDataInstance("SparseFlashAttentionSplitKVParamsMlaOp"), nullptr);
    EXPECT_NE(factory.CreateTilingDataInstance("SparseFlashAttentionInnerSplitParamsOp"), nullptr);
}

TEST_F(SparseFlashAttentionTilingArch35, SplitKvRespectsAvailableAicCores)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    for (const auto &[queryCount, topK, aicCores, expectedBlocks] :
         {std::tuple<int64_t, int64_t, int64_t, uint32_t>{1, 2048, 8, 8},
          {1, 2048, 24, 16},
          {4, 2048, 24, 16},
          {4, 2048, 56, 32},
          {1, 1024, 24, 8},
          {4, 1024, 24, 16},
          {1, 256, 24, 2},
          {4, 256, 24, 8},
          {1, 768, 24, 4},
          {1, 1000, 24, 8},
          {4, 512, 24, 16},
          {4, 1000, 24, 16},
          {3, 1024, 24, 24},
          {8, 2048, 24, 16},
          {16, 2048, 56, 32},
          {1, 128, 24, 24}}) {
        int32_t actualSeqQ[] = {static_cast<int32_t>(queryCount)};
        int32_t actualSeqKV[] = {4096};
        gert::TilingContextPara para(
            "SparseFlashAttention",
            {{{{queryCount, 8, 512}, {queryCount, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
             {{{32, 128, 1, 512}, {32, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
             {{{32, 128, 1, 512}, {32, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
             {{{queryCount, 1, topK}, {queryCount, 1, topK}}, ge::DT_INT32, ge::FORMAT_ND},
             {{{1, 32}, {1, 32}}, ge::DT_INT32, ge::FORMAT_ND},
             {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQ},
             {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKV},
             {{{queryCount, 8, 64}, {queryCount, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
             {{{32, 128, 1, 64}, {32, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}},
            {{{{queryCount, 8, 512}, {queryCount, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
             {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
             {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}},
            {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
             {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
             {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
             {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
             {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
             {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
            &compileInfo, "Ascend950", aicCores, 262144, 16384);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(para, info)) << "Q=" << queryCount << ", TopK=" << topK << ", AIC=" << aicCores;
        EXPECT_EQ(info.blockNum, expectedBlocks) << "Q=" << queryCount << ", TopK=" << topK << ", AIC=" << aicCores;
    }
}
