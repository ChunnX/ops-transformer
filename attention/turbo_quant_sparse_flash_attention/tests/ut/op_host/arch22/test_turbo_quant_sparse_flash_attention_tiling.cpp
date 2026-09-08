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
#include <algorithm>
#include <tuple>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
using AnyValue = Ops::Transformer::AnyValue;
using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

// TND 布局下 actual_seq_lengths 为累加和：单 batch、query 1 个 token、KV 有效长度 512。
int32_t g_actualSeqQuery[] = {1};
int32_t g_actualSeqKv[] = {512};

// query [T=1, N1=8, D=576]；KV 为 PA_BSND [blockNum=4, blockSize=128, N2=1, 386]，
// slot 386B = 256B packed nibbles + 128B rope in query dtype + 2B float16 scale.
std::vector<TensorDesc> BaseInputs()
{
    return {{{{1, 8, 576}, {1, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1, 386}, {4, 128, 1, 386}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4, 128, 1, 386}, {4, 128, 1, 386}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1, 1, 256}, {1, 1, 256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // key_dequant_scale：COMBINE 模式下不消费
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // value_dequant_scale
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, g_actualSeqQuery},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, g_actualSeqKv}};
}

std::vector<TensorDesc> BaseOutputs()
{
    return {{{{1, 8, 512}, {1, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}};
}

// 属性基线取 host 侧唯一合法值；负向用例在此基础上单点改写，以定位到具体校验项。
std::vector<OpAttr> BaseAttrs()
{
    return {{"scale_value", AnyValue::CreateFrom<float>(0.044194173f)},
            {"key_quant_mode", AnyValue::CreateFrom<int64_t>(3)},
            {"value_quant_mode", AnyValue::CreateFrom<int64_t>(3)},
            {"sparse_block_size", AnyValue::CreateFrom<int64_t>(1)},
            {"layout_query", AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_mode", AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", AnyValue::CreateFrom<int64_t>(2)},
            {"quant_scale_repo_mode", AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", AnyValue::CreateFrom<int64_t>(128)},
            {"rope_head_dim", AnyValue::CreateFrom<int64_t>(64)},
            {"return_softmax_lse", AnyValue::CreateFrom<bool>(false)}};
}

// 单点改写某个属性
std::vector<OpAttr> AttrsWith(const std::string &name, const AnyValue &value)
{
    auto attrs = BaseAttrs();
    for (auto &attr : attrs) {
        if (attr.attrName_ == name) {
            attr.attr_ = value;
        }
    }
    return attrs;
}
} // namespace

class TurboQuantSparseFlashAttentionTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TurboQuantSparseFlashAttentionTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "TurboQuantSparseFlashAttentionTiling TearDown" << std::endl;
    }
};

struct TurboQuantSparseFlashAttentionCompileInfo {};

// ---------------- 正向 ----------------

TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_base)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(), BaseAttrs(),
                                 &compileInfo, "Ascend910B", 64, 262144, 16384);
    // 期望值由本用例实跑采集，用于锁定 tiling 结果不被后续改动无声改变。
    int64_t expectTilingKey = 1158;
    std::string expectTilingData = "2199023255553 1 128 4294967300 4410436851802832897 4294967304 3 1 "
                                   "256 386 274877907456 3 3 128 0 0 49408 40532396646334464 262144 64 "
                                   "39582418608128 2199023255560 ";
    std::vector<size_t> expectWorkspaces = {184090624};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ---------------- 负向：属性取值 ----------------

// tile_size 未被 kernel 消费，slot 布局按 128 写死，非 128 不产生对应语义，须直接拒绝。
TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_tile_size_64)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(),
                                 AttrsWith("tile_size", AnyValue::CreateFrom<int64_t>(64)), &compileInfo, "Ascend910B",
                                 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_rope_head_dim_32)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(),
                                 AttrsWith("rope_head_dim", AnyValue::CreateFrom<int64_t>(32)), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_key_quant_mode_1)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(),
                                 AttrsWith("key_quant_mode", AnyValue::CreateFrom<int64_t>(1)), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_attention_mode_0)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(),
                                 AttrsWith("attention_mode", AnyValue::CreateFrom<int64_t>(0)), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_quant_scale_repo_mode_0)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(),
                                 AttrsWith("quant_scale_repo_mode", AnyValue::CreateFrom<int64_t>(0)), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ---------------- 负向：layout ----------------

// query 仅支持 TND：launcher 强制 query 为 3D，输出固定按 TND 构造。
TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_layout_query_bsnd)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(),
                                 AttrsWith("layout_query", AnyValue::CreateFrom<std::string>("BSND")), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// Non-paged KV layouts remain unsupported.
TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_layout_kv_bsnd)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(),
                                 AttrsWith("layout_kv", AnyValue::CreateFrom<std::string>("BSND")), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_layout_kv_tnd)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(),
                                 AttrsWith("layout_kv", AnyValue::CreateFrom<std::string>("TND")), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ---------------- 负向：dtype 与 shape ----------------

TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_query_fp16)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    auto inputs = BaseInputs();
    inputs[0].dtype_ = ge::DT_FLOAT16;
    auto outputs = BaseOutputs();
    outputs[0].dtype_ = ge::DT_FLOAT16;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", inputs, outputs, BaseAttrs(), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 1158;
    std::string expectTilingData = "2199023255553 1 128 4294967300 4410436851802832897 4294967304 3 1 "
                                   "256 386 274877907456 3 3 128 0 0 49408 40532396646334464 262144 64 "
                                   "39582418608128 2199023255560 ";
    std::vector<size_t> expectWorkspaces = {184090624};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_query_output_dtype_mismatch)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    auto inputs = BaseInputs();
    inputs[0].dtype_ = ge::DT_FLOAT16;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", inputs, BaseOutputs(), BaseAttrs(), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_output_query_dtype_mismatch)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    auto outputs = BaseOutputs();
    outputs[0].dtype_ = ge::DT_FLOAT16;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), outputs, BaseAttrs(), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_query_fp32)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    auto inputs = BaseInputs();
    inputs[0].dtype_ = ge::DT_FLOAT;
    auto outputs = BaseOutputs();
    outputs[0].dtype_ = ge::DT_FLOAT;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", inputs, outputs, BaseAttrs(), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_actual_seq_kv_int64)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    auto inputs = BaseInputs();
    inputs[8].dtype_ = ge::DT_INT64;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", inputs, BaseOutputs(), BaseAttrs(), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// query 的 head dim 写死为 576（512 latent + 64 rope），kernel 的 MM1 K 宽度按此固定。
TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_query_head_dim_640)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    auto inputs = BaseInputs();
    inputs[0].shape_ = {{1, 8, 640}, {1, 8, 640}};
    auto outputs = BaseOutputs();
    outputs[0].shape_ = {{1, 8, 576}, {1, 8, 576}};
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", inputs, outputs, BaseAttrs(), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// 当前 DSA 路径的 K/V 共用 latent，value 的 shape 必须与 key 完全一致。
TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_value_shape_mismatch)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    auto inputs = BaseInputs();
    inputs[2].shape_ = {{2, 128, 1, 386}, {2, 128, 1, 386}};
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", inputs, BaseOutputs(), BaseAttrs(), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// block_table 的第二维（每 batch 的最大 block 数）须大于 0：PA_BSND 下
// s2Size = block_table.dim1 * block_size，该校验保证 s2Size 恒大于 0，
// 因此 tiling 无需再为 s2Size == 0 兜底。
TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_block_table_dim1_zero)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    auto inputs = BaseInputs();
    inputs[6].shape_ = {{1, 0}, {1, 0}};
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", inputs, BaseOutputs(), BaseAttrs(), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// block_table 首维须等于 batch size。
TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_block_table_batch_mismatch)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    auto inputs = BaseInputs();
    inputs[6].shape_ = {{2, 4}, {2, 4}};
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", inputs, BaseOutputs(), BaseAttrs(), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

class TurboQuantSparseFlashAttentionBnbdTiling
    : public testing::TestWithParam<std::tuple<ge::DataType, int64_t, int64_t>> {};

TEST_P(TurboQuantSparseFlashAttentionBnbdTiling, MatchesBsndTiling)
{
    const auto [dtype, blockSize, queryHeads] = GetParam();
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    auto inputs = BaseInputs();
    auto outputs = BaseOutputs();
    inputs[0].dtype_ = outputs[0].dtype_ = dtype;
    inputs[0].shape_ = {{1, queryHeads, 576}, {1, queryHeads, 576}};
    outputs[0].shape_ = {{1, queryHeads, 512}, {1, queryHeads, 512}};
    inputs[1].shape_ = inputs[2].shape_ = {{4, blockSize, 1, 386}, {4, blockSize, 1, 386}};
    int32_t seqKv[] = {static_cast<int32_t>(4 * blockSize)};
    inputs[8] = {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, seqKv};
    gert::TilingContextPara bsnd("TurboQuantSparseFlashAttention", inputs, outputs, BaseAttrs(), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    TilingInfo bsndInfo;
    ASSERT_TRUE(ExecuteTiling(bsnd, bsndInfo));
    inputs[1].shape_ = inputs[2].shape_ = {{4, 1, blockSize, 386}, {4, 1, blockSize, 386}};
    gert::TilingContextPara bnbd("TurboQuantSparseFlashAttention", inputs, outputs,
                                 AttrsWith("layout_kv", AnyValue::CreateFrom<std::string>("PA_BNBD")), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    TilingInfo bnbdInfo;
    ASSERT_TRUE(ExecuteTiling(bnbd, bnbdInfo));
    if (queryHeads <= 64) {
        EXPECT_EQ(bsndInfo.tilingKey, 1158);
    }
    EXPECT_EQ(bnbdInfo.tilingKey, bsndInfo.tilingKey);
    EXPECT_EQ(bnbdInfo.workspaceSizes, bsndInfo.workspaceSizes);
    EXPECT_EQ(bnbdInfo.blockNum, bsndInfo.blockNum);
    ASSERT_EQ(bnbdInfo.tilingDataSize, bsndInfo.tilingDataSize);
    EXPECT_TRUE(std::equal(bnbdInfo.tilingData.get(), bnbdInfo.tilingData.get() + bnbdInfo.tilingDataSize,
                           bsndInfo.tilingData.get()));
}

INSTANTIATE_TEST_SUITE_P(LayoutDtypeBlockSize, TurboQuantSparseFlashAttentionBnbdTiling,
                         testing::Combine(testing::Values(ge::DT_FLOAT16, ge::DT_BF16), testing::Values(16, 128, 1024),
                                          testing::Values(8, 64, 128)));

class TurboQuantSparseFlashAttentionBnbdInvalid : public testing::TestWithParam<std::vector<int64_t>> {};

TEST_P(TurboQuantSparseFlashAttentionBnbdInvalid, RejectsInvalidKeyShape)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    auto inputs = BaseInputs();
    const auto &shape = GetParam();
    gert::Shape keyShape;
    keyShape.SetDimNum(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        keyShape.SetDim(i, shape[i]);
    }
    inputs[1].shape_.MutableOriginShape() = keyShape;
    inputs[1].shape_.MutableStorageShape() = keyShape;
    inputs[2].shape_ = inputs[1].shape_;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", inputs, BaseOutputs(),
                                 AttrsWith("layout_kv", AnyValue::CreateFrom<std::string>("PA_BNBD")), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

INSTANTIATE_TEST_SUITE_P(LayoutAxesAndBounds, TurboQuantSparseFlashAttentionBnbdInvalid,
                         testing::Values(std::vector<int64_t>{4, 128, 1, 386}, std::vector<int64_t>{4, 2, 128, 386},
                                         std::vector<int64_t>{4, 0, 128, 386}, std::vector<int64_t>{4, 1, 0, 386},
                                         std::vector<int64_t>{4, 1, 8, 386}, std::vector<int64_t>{4, 1, 1040, 386},
                                         std::vector<int64_t>{4, 1, 128, 385}, std::vector<int64_t>{4, 128, 386}));

class TurboQuantSparseFlashAttentionEmptyTiling
    : public testing::TestWithParam<std::tuple<ge::DataType, bool, bool, int64_t>> {};

TEST_P(TurboQuantSparseFlashAttentionEmptyTiling, EmptyQueryHasNoWorkspace)
{
    const auto [dtype, bnbd, returnLse, heads] = GetParam();
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    auto inputs = BaseInputs();
    auto outputs = BaseOutputs();
    inputs[0].dtype_ = outputs[0].dtype_ = dtype;
    inputs[0].shape_ = {{0, heads, 576}, {0, heads, 576}};
    inputs[3].shape_ = {{0, 1, 256}, {0, 1, 256}};
    outputs[0].shape_ = {{0, heads, 512}, {0, heads, 512}};
    outputs[1].shape_ = outputs[2].shape_ =
        returnLse ? gert::StorageShape({1, 0, heads}, {1, 0, heads}) : gert::StorageShape({0}, {0});
    int32_t seqZero[] = {0};
    inputs[7] = inputs[8] = {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, seqZero};
    if (bnbd) {
        inputs[1].shape_ = inputs[2].shape_ = {{4, 1, 128, 386}, {4, 1, 128, 386}};
    }
    auto attrs = AttrsWith("layout_kv", AnyValue::CreateFrom<std::string>(bnbd ? "PA_BNBD" : "PA_BSND"));
    attrs.back().attr_ = AnyValue::CreateFrom<bool>(returnLse);
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", inputs, outputs, attrs, &compileInfo, "Ascend910B",
                                 64, 262144, 16384);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ASSERT_EQ(info.workspaceSizes.size(), 1);
    EXPECT_EQ(info.workspaceSizes[0], 0);
    EXPECT_EQ(info.blockNum, 1);
    EXPECT_GT(info.tilingDataSize, 0);

    // Empty output must still reject invalid non-token dimensions.
    inputs[0].shape_ = {{0, 0, 576}, {0, 0, 576}};
    gert::TilingContextPara invalid("TurboQuantSparseFlashAttention", inputs, outputs, attrs, &compileInfo,
                                    "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(invalid, ge::GRAPH_FAILED);
}

INSTANTIATE_TEST_SUITE_P(DtypeLayoutLseHeads, TurboQuantSparseFlashAttentionEmptyTiling,
                         testing::Combine(testing::Values(ge::DT_FLOAT16, ge::DT_BF16), testing::Bool(),
                                          testing::Bool(), testing::Values(1, 128)));

TEST_F(TurboQuantSparseFlashAttentionTiling, MixedZeroSequenceLengths)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    auto inputs = BaseInputs();
    auto outputs = BaseOutputs();
    int32_t seqQuery[] = {0, 2, 2, 3, 3};
    int32_t seqKv[] = {17, 0, 29, 32, 0};
    inputs[0].shape_ = {{3, 8, 576}, {3, 8, 576}};
    inputs[3].shape_ = {{3, 1, 256}, {3, 1, 256}};
    inputs[6].shape_ = {{5, 4}, {5, 4}};
    inputs[7] = {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, seqQuery};
    inputs[8] = {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, seqKv};
    outputs[0].shape_ = {{3, 8, 512}, {3, 8, 512}};
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", inputs, outputs, BaseAttrs(), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 1158);
}

namespace {
void SetCacheStride(TensorDesc &tensor, const std::vector<int64_t> &strides)
{
    tensor.hasStride_ = true;
    tensor.stride_ = gert::Stride();
    tensor.stride_.SetDimNum(strides.size());
    for (size_t axis = 0; axis < strides.size(); ++axis) {
        tensor.stride_.SetStride(axis, strides[axis]);
    }
}
} // namespace

class TurboQuantSparseFlashAttentionStride : public testing::TestWithParam<std::tuple<ge::DataType, bool, int64_t>> {};

TEST_P(TurboQuantSparseFlashAttentionStride, PreservesByteStrideWithoutTruncation)
{
    const auto [dtype, bnbd, stride0] = GetParam();
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    auto inputs = BaseInputs();
    auto outputs = BaseOutputs();
    inputs[0].dtype_ = outputs[0].dtype_ = dtype;
    if (bnbd) {
        inputs[1].shape_ = inputs[2].shape_ = {{4, 1, 128, 386}, {4, 1, 128, 386}};
    }
    auto strides = bnbd ? std::vector<int64_t>{stride0, 7, 386, 1} : std::vector<int64_t>{stride0, 386, 7, 1};
    SetCacheStride(inputs[1], strides);
    strides[0] += 256;
    SetCacheStride(inputs[2], strides);
    auto attrs = AttrsWith("layout_kv", AnyValue::CreateFrom<std::string>(bnbd ? "PA_BNBD" : "PA_BSND"));
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", inputs, outputs, attrs, &compileInfo, "Ascend910B",
                                 64, 262144, 16384);
    std::string expected = "2199023255553 1 128 4294967300 4410436851802832897 4294967304 3 1 "
                           "256 386 274877907456 3 3 128 0 0 " +
                           std::to_string(stride0) + " 40532396646334464 262144 64 39582418608128 2199023255560 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 1158, expected, {184090624});
}

INSTANTIATE_TEST_SUITE_P(DtypeLayoutStride, TurboQuantSparseFlashAttentionStride,
                         testing::Combine(testing::Values(ge::DT_FLOAT16, ge::DT_BF16), testing::Bool(),
                                          testing::Values(int64_t{49408}, int64_t{98816}, int64_t{49664},
                                                          int64_t{1} << 33, INT64_MAX / 4)));

class TurboQuantSparseFlashAttentionInvalidStride
    : public testing::TestWithParam<std::tuple<ge::DataType, bool, int, int>> {};

TEST_P(TurboQuantSparseFlashAttentionInvalidStride, RejectsUnsupportedCacheView)
{
    const auto [dtype, bnbd, inputIndex, invalidKind] = GetParam();
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    auto inputs = BaseInputs();
    auto outputs = BaseOutputs();
    inputs[0].dtype_ = outputs[0].dtype_ = dtype;
    if (bnbd) {
        inputs[1].shape_ = inputs[2].shape_ = {{4, 1, 128, 386}, {4, 1, 128, 386}};
    }
    auto strides = bnbd ? std::vector<int64_t>{98816, 49408, 386, 1} : std::vector<int64_t>{98816, 386, 386, 1};
    switch (invalidKind) {
        case 0:
            strides[bnbd ? 2 : 1] = 387;
            break;
        case 1:
            strides[3] = 2;
            break;
        case 2:
            strides[0] = 49407;
            break;
        case 3:
            strides[0] = 0;
            break;
        case 4:
            strides[0] = -1;
            break;
        case 5:
            strides[0] = INT64_MAX;
            break;
        case 6:
            strides.pop_back();
            break;
    }
    SetCacheStride(inputs[inputIndex], strides);
    auto attrs = AttrsWith("layout_kv", AnyValue::CreateFrom<std::string>(bnbd ? "PA_BNBD" : "PA_BSND"));
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", inputs, outputs, attrs, &compileInfo, "Ascend910B",
                                 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

INSTANTIATE_TEST_SUITE_P(DtypeLayoutInputViolation, TurboQuantSparseFlashAttentionInvalidStride,
                         testing::Combine(testing::Values(ge::DT_FLOAT16, ge::DT_BF16), testing::Bool(),
                                          testing::Values(1, 2), testing::Range(0, 7)));
