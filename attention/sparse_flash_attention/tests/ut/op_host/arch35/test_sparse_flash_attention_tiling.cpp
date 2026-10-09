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
#include "../../../../op_kernel/sparse_flash_attention_template_tiling_key.h"

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
    for (int64_t ropeDim : {0, 64}) {
        for (int64_t heads : {1, 3, 8, 64, 65, 97, 128}) {
            for (int64_t sparseMode : {0, 3}) {
                for (const auto& [batchCount, queryCount, topK, aicCores, expectedBlocks] :
                     {std::tuple<int64_t, int64_t, int64_t, int64_t, uint32_t>{1, 1, 2048, 8, 8},
                      {1, 1, 2048, 24, 16},
                      {1, 4, 2048, 24, 16},
                      {2, 4, 2048, 24, 16},
                      {2, 4, 2048, 56, 32},
                      {1, 4, 2048, 56, 32},
                      {1, 1, 1024, 24, 8},
                      {1, 4, 1024, 24, 16},
                      {2, 4, 1024, 24, 16},
                      {4, 4, 1024, 24, 16},
                      {8, 8, 1024, 24, 16},
                      {1, 1, 256, 24, 2},
                      {1, 4, 256, 24, 8},
                      {1, 1, 768, 24, 4},
                      {1, 1, 1000, 24, 8},
                      {1, 4, 512, 24, 16},
                      {1, 4, 1000, 24, 16},
                      {1, 3, 1024, 24, 24},
                      {1, 8, 2048, 24, 16},
                      {1, 16, 2048, 56, 32},
                      {2, 20, 2048, 24, 24},
                      {2, 4, 256, 24, 8},
                      {1, 1, 4096, 24, 24},
                      {1, 1, 128, 24, 24}}) {
                    int32_t actualSeqQ[8] = {};
                    int32_t actualSeqKV[8] = {};
                    for (int64_t batch = 0; batch < batchCount; ++batch) {
                        actualSeqQ[batch] = static_cast<int32_t>((batch + 1) * queryCount / batchCount);
                        actualSeqKV[batch] = 4096;
                    }
                    const gert::StorageShape queryRopeShape =
                        ropeDim ? gert::StorageShape({queryCount, heads, ropeDim}, {queryCount, heads, ropeDim}) :
                                  gert::StorageShape({}, {});
                    const gert::StorageShape keyRopeShape =
                        ropeDim ? gert::StorageShape({32, 128, 1, ropeDim}, {32, 128, 1, ropeDim}) :
                                  gert::StorageShape({}, {});
                    gert::TilingContextPara para(
                        "SparseFlashAttention",
                        {{{{queryCount, heads, 512}, {queryCount, heads, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                         {{{32, 128, 1, 512}, {32, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                         {{{32, 128, 1, 512}, {32, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                         {{{queryCount, 1, topK}, {queryCount, 1, topK}}, ge::DT_INT32, ge::FORMAT_ND},
                         {{{batchCount, 32}, {batchCount, 32}}, ge::DT_INT32, ge::FORMAT_ND},
                         {{{batchCount}, {batchCount}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQ},
                         {{{batchCount}, {batchCount}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKV},
                         {queryRopeShape, ge::DT_BF16, ge::FORMAT_ND},
                         {keyRopeShape, ge::DT_BF16, ge::FORMAT_ND}},
                        {{{{queryCount, heads, 512}, {queryCount, heads, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
                         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
                         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sparseMode)},
                         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
                        &compileInfo, "Ascend950", aicCores, 262144, 16384);
                    TilingInfo info;
                    ASSERT_TRUE(ExecuteTiling(para, info))
                        << "B=" << batchCount << ", Q=" << queryCount << ", TopK=" << topK;
                    const uint32_t headParts = heads > 64 ? 2U : 1U;
                    const uint32_t maxSplits = std::min({16U, static_cast<uint32_t>(aicCores / queryCount / headParts),
                                                         static_cast<uint32_t>((topK + 127) / 128)});
                    uint32_t splits = 1;
                    while (splits * 2 <= maxSplits) {
                        splits *= 2;
                    }
                    const bool useFd = topK >= 256 && topK <= 2048 && splits > 1;
                    const uint32_t genericBlocks = useFd ? queryCount * headParts * splits : aicCores;
                    EXPECT_EQ(info.blockNum, heads <= 64 ? expectedBlocks : genericBlocks)
                        << "B=" << batchCount << ", Q=" << queryCount << ", G=" << heads << ", TopK=" << topK
                        << ", AIC=" << aicCores;
                    EXPECT_EQ(info.tilingKey, GET_TPL_TILING_KEY(static_cast<uint32_t>(useFd), 1U, 1U, 2U, 1U,
                                                                 static_cast<uint32_t>(heads > 64),
                                                                 static_cast<uint32_t>(ropeDim != 0)));
                    if (useFd) {
                        const uint64_t scratch = genericBlocks * 3ULL * 128 * (512 + ropeDim) * sizeof(uint16_t);
                        const uint64_t rows = (heads + headParts * 2 - 1) / (headParts * 2);
                        const uint64_t staging = genericBlocks * 2ULL * rows * (512 + 2 * 8) * sizeof(float);
                        ASSERT_EQ(info.workspaceSizes.size(), 1U);
                        EXPECT_GE(info.workspaceSizes[0], scratch + staging);
                    }
                }
            }
        }
    }
}

TEST_F(SparseFlashAttentionTilingArch35, SplitKvFeaturesRespectKernelSupport)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    for (int64_t ropeDim : {0, 32, 64, 128}) {
        for (const auto& [queryType, kvType, heads, sparseMode, returnLse, expectedBlocks] :
             {std::tuple<ge::DataType, ge::DataType, int64_t, int64_t, bool, uint32_t>{ge::DT_BF16, ge::DT_BF16, 8, 0,
                                                                                       false, 16},
              {ge::DT_FLOAT16, ge::DT_FLOAT16, 8, 0, false, 16},
              {ge::DT_FLOAT16, ge::DT_FLOAT16, 8, 3, false, 16},
              {ge::DT_BF16, ge::DT_BF16, 1, 0, false, 16},
              {ge::DT_BF16, ge::DT_BF16, 3, 0, true, 16},
              {ge::DT_BF16, ge::DT_BF16, 16, 0, false, 16},
              {ge::DT_BF16, ge::DT_BF16, 64, 0, false, 16},
              {ge::DT_BF16, ge::DT_BF16, 64, 3, true, 16},
              {ge::DT_FLOAT16, ge::DT_FLOAT16, 64, 3, true, 16},
              {ge::DT_BF16, ge::DT_BF16, 65, 0, true, 16},
              {ge::DT_FLOAT16, ge::DT_FLOAT16, 127, 3, true, 16},
              {ge::DT_BF16, ge::DT_BF16, 128, 0, false, 16},
              {ge::DT_BF16, ge::DT_BF16, 8, 3, false, 16},
              {ge::DT_BF16, ge::DT_BF16, 8, 0, true, 16},
              {ge::DT_FLOAT16, ge::DT_FLOAT16, 8, 3, true, 16}}) {
            int32_t actualSeqQ[] = {1};
            int32_t actualSeqKV[] = {4096};
            const gert::StorageShape statsShape =
                returnLse ? gert::StorageShape({1, 1, heads}, {1, 1, heads}) : gert::StorageShape({0}, {0});
            const gert::StorageShape queryRopeShape =
                ropeDim ? gert::StorageShape({1, heads, ropeDim}, {1, heads, ropeDim}) : gert::StorageShape({}, {});
            const gert::StorageShape keyRopeShape =
                ropeDim ? gert::StorageShape({32, 128, 1, ropeDim}, {32, 128, 1, ropeDim}) : gert::StorageShape({}, {});
            gert::TilingContextPara para(
                "SparseFlashAttention",
                {{{{1, heads, 512}, {1, heads, 512}}, queryType, ge::FORMAT_ND},
                 {{{32, 128, 1, 512}, {32, 128, 1, 512}}, kvType, ge::FORMAT_ND},
                 {{{32, 128, 1, 512}, {32, 128, 1, 512}}, kvType, ge::FORMAT_ND},
                 {{{1, 1, 2048}, {1, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
                 {{{1, 32}, {1, 32}}, ge::DT_INT32, ge::FORMAT_ND},
                 {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQ},
                 {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKV},
                 {queryRopeShape, queryType, ge::FORMAT_ND},
                 {keyRopeShape, kvType, ge::FORMAT_ND}},
                {{{{1, heads, 512}, {1, heads, 512}}, queryType, ge::FORMAT_ND},
                 {statsShape, ge::DT_FLOAT, ge::FORMAT_ND},
                 {statsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
                {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
                 {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                 {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                 {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
                 {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sparseMode)},
                 {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                 {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                 {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                 {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(returnLse)}},
                &compileInfo, "Ascend950", 24, 262144, 16384);
            if (ropeDim != 0 && ropeDim != 64) {
                ExecuteTestCase(para, ge::GRAPH_FAILED);
                continue;
            }
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(para, info));
            EXPECT_EQ(info.blockNum, expectedBlocks)
                << "Q dtype=" << queryType << ", KV dtype=" << kvType << ", G=" << heads << ", mode=" << sparseMode
                << ", LSE=" << returnLse;
            EXPECT_EQ(info.tilingKey, GET_TPL_TILING_KEY(1U, 1U, 1U, 2U, 1U, static_cast<uint32_t>(heads > 64),
                                                         static_cast<uint32_t>(ropeDim != 0)));
        }
    }
}

TEST_F(SparseFlashAttentionTilingArch35, InvalidSparseBlockSizeRejectedBeforeSplitKv)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    for (ge::DataType dtype : {ge::DT_BF16, ge::DT_FLOAT16}) {
        for (int64_t sparseBlockSize : {0, 2, 128}) {
            int32_t actualSeqQ[] = {1};
            int32_t actualSeqKV[] = {4096};
            gert::TilingContextPara para(
                "SparseFlashAttention",
                {{{{1, 8, 512}, {1, 8, 512}}, dtype, ge::FORMAT_ND},
                 {{{32, 128, 1, 512}, {32, 128, 1, 512}}, dtype, ge::FORMAT_ND},
                 {{{32, 128, 1, 512}, {32, 128, 1, 512}}, dtype, ge::FORMAT_ND},
                 {{{1, 1, 2048}, {1, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
                 {{{1, 32}, {1, 32}}, ge::DT_INT32, ge::FORMAT_ND},
                 {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQ},
                 {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKV},
                 {{{1, 8, 64}, {1, 8, 64}}, dtype, ge::FORMAT_ND},
                 {{{32, 128, 1, 64}, {32, 128, 1, 64}}, dtype, ge::FORMAT_ND}},
                {{{{1, 8, 512}, {1, 8, 512}}, dtype, ge::FORMAT_ND},
                 {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                 {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
                 {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sparseBlockSize)},
                 {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                 {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
                 {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                 {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                 {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                 {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
                &compileInfo, "Ascend950", 24, 262144, 16384);
            ExecuteTestCase(para, ge::GRAPH_FAILED);
        }
    }
}
