/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "attention/generic_block_sparse_attention/op_kernel/kernel_common.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "attention/generic_block_sparse_attention/op_host/generic_block_sparse_attention_tiling.h"

using namespace std;
using namespace ge;
using namespace optiling;

class generic_block_sparse_attention_tiling_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "generic_block_sparse_attention_tiling_ut SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "generic_block_sparse_attention_tiling_ut TearDown" << endl;
    }
};

namespace {
constexpr int64_t kBatch = 1;
constexpr int64_t kS1 = 4;
constexpr int64_t kS2 = 256;
constexpr int64_t kN1 = 4;
constexpr int64_t kN2 = 1;
constexpr int64_t kD = 128;
constexpr int64_t kTopK = 2;
constexpr int64_t kBlockSize = 128;
constexpr int64_t kBlockShapeX = 1;
constexpr int64_t kT = kBatch * kS1;
constexpr int64_t kMaxBlocks = (kS2 + kBlockSize - 1) / kBlockSize;
constexpr int64_t kTotalQBlocks = kT;
constexpr float kScale = 1.0f / sqrt(static_cast<float>(kD));

// Expected host tiling keys (BSA-style bitfields; must match GenerateTilingKey).
constexpr uint64_t kFp16Tiling910B = 9200000071001002ULL;
constexpr uint64_t kBf16Tiling910B = 9200000071023222ULL;
constexpr uint64_t kFp16HalfSmTiling910B = 9200000071101002ULL;
constexpr uint64_t kFp16Tiling950 = 9250000071001002ULL;
constexpr uint64_t kBf16Tiling950 = 9250000071023222ULL;
constexpr uint64_t kFp8Fp16Tiling950 = 9250000071001012ULL;
constexpr uint64_t kFp8Bf16Tiling950 = 9250000071001022ULL;

std::vector<int64_t> kBlockShape = {kBlockShapeX, kBlockSize};

std::vector<gert::TilingContextPara::TensorDescription> MakeInputs(ge::DataType qkvDtype)
{
    return {
        {{{kT, kN1, kD}, {kT, kN1, kD}}, qkvDtype, ge::FORMAT_ND},
        {{{kMaxBlocks, kBlockSize, kN2, kD}, {kMaxBlocks, kBlockSize, kN2, kD}}, qkvDtype, ge::FORMAT_ND},
        {{{kMaxBlocks, kBlockSize, kN2, kD}, {kMaxBlocks, kBlockSize, kN2, kD}}, qkvDtype, ge::FORMAT_ND},
        {{{kN2, kTotalQBlocks, kTopK}, {kN2, kTotalQBlocks, kTopK}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{kN2, kTotalQBlocks}, {kN2, kTotalQBlocks}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{{kBatch + 1}, {kBatch + 1}}, ge::DT_INT64, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{{kBatch}, {kBatch}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{kBatch, kMaxBlocks}, {kBatch, kMaxBlocks}}, ge::DT_INT32, ge::FORMAT_ND},
    };
}

std::vector<gert::TilingContextPara::TensorDescription> MakeOutputs(ge::DataType outDtype, bool returnLse)
{
    if (returnLse) {
        return {
            {{{kT, kN1, kD}, {kT, kN1, kD}}, outDtype, ge::FORMAT_ND},
            {{{kT, kN1, 1}, {kT, kN1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        };
    }
    return {
        {{{kT, kN1, kD}, {kT, kN1, kD}}, outDtype, ge::FORMAT_ND},
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
}

std::vector<gert::TilingContextPara::OpAttr> MakeAttrs(int64_t maskType, int64_t quantType, int64_t softmaxPrec,
                                                       int64_t returnLse, float dstTypeMax = 0.0f, int64_t winLeft = -1,
                                                       int64_t winRight = -1, int64_t isPackedGqa = 1,
                                                       const std::string &layoutQ = "TND",
                                                       const std::string &layoutKv = "PA_BBND",
                                                       const std::vector<int64_t> &blockShape = kBlockShape,
                                                       float scaleValue = kScale)
{
    return {
        {"block_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(blockShape)},
        {"is_packed_gqa", Ops::Transformer::AnyValue::CreateFrom<int64_t>(isPackedGqa)},
        {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutQ)},
        {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutKv)},
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
        {"mask_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maskType)},
        {"quant_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantType)},
        {"dst_type_max", Ops::Transformer::AnyValue::CreateFrom<float>(dstTypeMax)},
        {"softmax_precision", Ops::Transformer::AnyValue::CreateFrom<int64_t>(softmaxPrec)},
        {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(winLeft)},
        {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(winRight)},
        {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(returnLse)},
    };
}
} // namespace

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_fp16_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 0, 0, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp16Tiling910B);
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_fp16_halfsm_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 0, 1, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp16HalfSmTiling910B);
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_bf16_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_BF16),
                                              MakeOutputs(ge::DT_BF16, false), MakeAttrs(1, 0, 0, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kBf16Tiling910B);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_return_softmax_lse)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, true), MakeAttrs(1, 0, 0, 1), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_fp16_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 0, 1, 0), &compileInfo,
                                              "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp16Tiling950);
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_bf16_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_BF16),
                                              MakeOutputs(ge::DT_BF16, false), MakeAttrs(1, 0, 1, 0), &compileInfo,
                                              "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kBf16Tiling950);
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_fp8_out_fp16_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT8_E4M3FN),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 5, 1, 0), &compileInfo,
                                              "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp8Fp16Tiling950);
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_fp8_out_bf16_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT8_E4M3FN),
                                              MakeOutputs(ge::DT_BF16, false), MakeAttrs(1, 5, 1, 0), &compileInfo,
                                              "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp8Bf16Tiling950);
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_rejects_block_table)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 1, "TND", "TND"), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_is_packed_gqa)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_mask_type)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(0, 0, 0, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_block_shape_x)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    std::vector<int64_t> badBlockShape = {128, 128};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 1, "TND", "PA_BBND", badBlockShape),
                                              &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_win_left)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 0, 0, 0, 0.0f, 127, -1),
                                              &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_bf16_halfsm_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_BF16),
                                              MakeOutputs(ge::DT_BF16, false), MakeAttrs(1, 0, 1, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_fp8_on_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT8_E4M3FN),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 5, 0, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, quant_type_dtype_mismatch)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 5, 0, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, missing_metadata)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[5] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, missing_cu_seq_lengths_q)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[11] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, missing_seqused_kv)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[14] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, pa_rejects_cu_seq_lengths_kv)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[12] = {{{kBatch + 1}, {kBatch + 1}}, ge::DT_INT64, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, scale_value_zero_defaults_to_rsqrt_d)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 1, "TND", "PA_BBND", kBlockShape, 0.0f), &compileInfo, "Ascend910B", 40,
        196608);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(kFp16Tiling910B));
    ASSERT_NE(tilingInfo.tilingData, nullptr);
    // scaleValue follows the 10 uint32 fields in the POD tiling layout.
    constexpr size_t kScaleValueOffset = 10U * sizeof(uint32_t);
    ASSERT_GT(tilingInfo.tilingDataSize, kScaleValueOffset + sizeof(float));
    float scaleValue = 0.0f;
    std::memcpy(&scaleValue, tilingInfo.tilingData.get() + kScaleValueOffset, sizeof(float));
    EXPECT_FLOAT_EQ(scaleValue, kScale);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_softmax_precision_0_on_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 0, 0, 0), &compileInfo,
                                              "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, missing_block_table)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[15] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, invalid_metadata_size)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[5] = {{{512}, {512}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_fp8_with_lse_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT8_E4M3FN),
                                              MakeOutputs(ge::DT_FLOAT16, true), MakeAttrs(1, 5, 1, 1), &compileInfo,
                                              "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

namespace {
auto MakeTndInputs(ge::DataType dtype, bool hasUsedKv = false)
{
    auto inputs = MakeInputs(dtype);
    inputs[1] = {{{kS2, kN2, kD}, {kS2, kN2, kD}}, dtype, ge::FORMAT_ND};
    inputs[2] = inputs[1];
    inputs[12] = {{{kBatch + 1}, {kBatch + 1}}, ge::DT_INT64, ge::FORMAT_ND};
    if (!hasUsedKv) {
        inputs[14] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
    }
    inputs[15] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
    return inputs;
}

auto MakeTndAttrs(int64_t precision = 0, int64_t quant = 0)
{
    return MakeAttrs(1, quant, precision, 0, 0.0f, -1, -1, 1, "TND", "TND");
}
} // namespace

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_all_supported_keys)
{
    const struct {
        ge::DataType input;
        ge::DataType output;
        const char *soc;
        int64_t precision;
        int64_t quant;
        uint64_t key;
    } cases[] = {
        {ge::DT_FLOAT16, ge::DT_FLOAT16, "Ascend910B", 0, 0, 9200000030001002ULL},
        {ge::DT_FLOAT16, ge::DT_FLOAT16, "Ascend910B", 1, 0, 9200000030101002ULL},
        {ge::DT_BF16, ge::DT_BF16, "Ascend910B", 0, 0, 9200000030023222ULL},
        {ge::DT_FLOAT16, ge::DT_FLOAT16, "Ascend950", 1, 0, 9250000030001002ULL},
        {ge::DT_BF16, ge::DT_BF16, "Ascend950", 1, 0, 9250000030023222ULL},
        {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16, "Ascend950", 1, 5, 9250000030001012ULL},
        {ge::DT_FLOAT8_E4M3FN, ge::DT_BF16, "Ascend950", 1, 5, 9250000030001022ULL},
    };
    for (const auto &test : cases) {
        for (bool used : {false, true}) {
            GenericBlockSparseAttentionCompileInfo info;
            gert::TilingContextPara para("GenericBlockSparseAttention", MakeTndInputs(test.input, used),
                                        MakeOutputs(test.output, false), MakeTndAttrs(test.precision, test.quant),
                                        &info, test.soc, 40, 262144);
            ExecuteTestCase(para, ge::GRAPH_SUCCESS, test.key);
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_invalid_lengths_and_kv_shapes)
{
    for (int invalid = 0; invalid < 11; ++invalid) {
        SCOPED_TRACE(invalid);
        auto inputs = MakeTndInputs(ge::DT_FLOAT16);
        switch (invalid) {
            case 0: inputs[12] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND}; break;
            case 1: inputs[12] = {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND}; break;
            case 2: inputs[12] = {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}; break;
            case 3: inputs[14] = {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}; break;
            case 4: inputs[14] = {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND}; break;
            case 5: inputs[2] = {{{kS2 - 1, kN2, kD}, {kS2 - 1, kN2, kD}}, ge::DT_FLOAT16, ge::FORMAT_ND}; break;
            case 6: inputs[1] = inputs[2] = {{{kS2, 2, kD}, {kS2, 2, kD}}, ge::DT_FLOAT16, ge::FORMAT_ND}; break;
            case 7: inputs[1] = inputs[2] = {{{kS2, kN2, 64}, {kS2, kN2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}; break;
            case 8: inputs[1] = inputs[2] = MakeInputs(ge::DT_FLOAT16)[1]; break;
            case 9: inputs[2].dtype_ = ge::DT_BF16; break;
            case 10: inputs[12] = {{{1, 2}, {1, 2}}, ge::DT_INT64, ge::FORMAT_ND}; break;
        }
        GenericBlockSparseAttentionCompileInfo info;
        gert::TilingContextPara para("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                    MakeTndAttrs(), &info, "Ascend910B", 40, 196608);
        ExecuteTestCase(para, ge::GRAPH_FAILED);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_rejects_noncontiguous_direct_graph_kv)
{
    for (int input : {1, 2}) {
        for (int axis = 0; axis < 3; ++axis) {
            auto inputs = MakeTndInputs(ge::DT_FLOAT16);
            inputs[input].hasStride_ = true;
            inputs[input].stride_.SetDimNum(3);
            inputs[input].stride_.SetStride(0, kN2 * kD);
            inputs[input].stride_.SetStride(1, kD);
            inputs[input].stride_.SetStride(2, 1);
            inputs[input].stride_.SetStride(axis, inputs[input].stride_.GetStride(axis) * 2);
            GenericBlockSparseAttentionCompileInfo info;
            gert::TilingContextPara para("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                        MakeTndAttrs(), &info, "Ascend910B", 40, 196608);
            ExecuteTestCase(para, ge::GRAPH_FAILED);
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_multibatch_tail_and_fd_workspace_match_pa)
{
    for (int64_t topK : {1, 11, 12, 16}) {
        auto tnd = MakeTndInputs(ge::DT_FLOAT16, true);
        tnd[11] = {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND};
        tnd[12] = tnd[11];
        tnd[13] = tnd[14] = {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND};
        tnd[1] = tnd[2] = {{{259, kN2, kD}, {259, kN2, kD}}, ge::DT_FLOAT16, ge::FORMAT_ND};
        tnd[3] = {{{kN2, kTotalQBlocks, topK}, {kN2, kTotalQBlocks, topK}}, ge::DT_INT32, ge::FORMAT_ND};
        auto pa = tnd;
        pa[1] = pa[2] = MakeInputs(ge::DT_FLOAT16)[1];
        pa[12] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
        pa[15] = {{{2, kMaxBlocks}, {2, kMaxBlocks}}, ge::DT_INT32, ge::FORMAT_ND};
        GenericBlockSparseAttentionCompileInfo info;
        gert::TilingContextPara tndPara("GenericBlockSparseAttention", tnd, MakeOutputs(ge::DT_FLOAT16, false),
                                       MakeTndAttrs(), &info, "Ascend910B", 40, 196608);
        gert::TilingContextPara paPara("GenericBlockSparseAttention", pa, MakeOutputs(ge::DT_FLOAT16, false),
                                      MakeAttrs(1, 0, 0, 0), &info, "Ascend910B", 40, 196608);
        TilingInfo tndInfo, paInfo;
        ASSERT_TRUE(ExecuteTiling(tndPara, tndInfo));
        ASSERT_TRUE(ExecuteTiling(paPara, paInfo));
        GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData tndData{}, paData{};
        ASSERT_EQ(tndInfo.tilingDataSize, sizeof(tndData));
        ASSERT_EQ(paInfo.tilingDataSize, sizeof(paData));
        std::memcpy(&tndData, tndInfo.tilingData.get(), sizeof(tndData));
        std::memcpy(&paData, paInfo.tilingData.get(), sizeof(paData));
        EXPECT_EQ(tndData.workSpaceSize, paData.workSpaceSize);
        EXPECT_EQ(tndData.batch, 2U);
        EXPECT_EQ(tndData.kStride0, kN2 * kD);
        EXPECT_EQ(tndData.vStride0, kN2 * kD);
        EXPECT_EQ(tndData.maxBlocksPerBatch, 0U);
        EXPECT_EQ(tndData.blockSize, 0U);
        EXPECT_EQ(paData.blockSize, 128U);
        EXPECT_EQ(tndData.blockShapeY, 128U);
        EXPECT_EQ(tndData.kvBaseTile, 128U);
        EXPECT_EQ(paData.kvBaseTile, 128U);
        EXPECT_EQ(tndData.mm1L1TileN, tndData.kvBaseTile);
        EXPECT_EQ(tndData.mm2L1TileKLeft, tndData.kvBaseTile);
        EXPECT_EQ(tndData.fdStaticEnabled, topK >= 12 ? 1U : 0U);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, pa_preserves_dim0_strides_and_rejects_inner_holes)
{
    for (bool innerHole : {false, true}) {
        auto inputs = MakeInputs(ge::DT_FLOAT16);
        for (int index : {1, 2}) {
            inputs[index].hasStride_ = true;
            inputs[index].stride_.SetDimNum(4);
            inputs[index].stride_.SetStride(0, 2 * kBlockSize * kN2 * kD);
            inputs[index].stride_.SetStride(1, kN2 * kD);
            inputs[index].stride_.SetStride(2, kD);
            inputs[index].stride_.SetStride(3, innerHole ? 2 : 1);
        }
        GenericBlockSparseAttentionCompileInfo info;
        gert::TilingContextPara para("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                    MakeAttrs(1, 0, 0, 0), &info, "Ascend910B", 40, 196608);
        if (innerHole) {
            ExecuteTestCase(para, ge::GRAPH_FAILED);
        } else {
            TilingInfo result;
            ASSERT_TRUE(ExecuteTiling(para, result));
            GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData data{};
            ASSERT_EQ(result.tilingDataSize, sizeof(data));
            std::memcpy(&data, result.tilingData.get(), sizeof(data));
            EXPECT_EQ(data.kStride0, 2U * kBlockSize * kN2 * kD);
            EXPECT_EQ(data.vStride0, data.kStride0);
            EXPECT_EQ(result.tilingKey, kFp16Tiling910B);
        }
    }
}
