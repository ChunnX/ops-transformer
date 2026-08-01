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
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/und_gen_qkv_rms_norm_rope_cache_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

namespace {
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t MROPE_AXIS_NUM = 3;
constexpr float NORM_EPS = 1e-6f;

class UndGenQkvRmsNormRopeCacheTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "UndGenQkvRmsNormRopeCacheTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UndGenQkvRmsNormRopeCacheTiling TearDown" << std::endl;
    }
};

// 全部可选输入均提供（hasGen = true, hasCatIndices = true）
gert::TilingContextPara BuildTilingContext(int64_t undLen, int64_t genLen, int64_t numHeadQ, int64_t numHeadK,
                                           int64_t numHeadV, int64_t headDim = HEAD_DIM,
                                           int64_t blockSize = BLOCK_SIZE, int64_t maxPos = 4096,
                                           const std::vector<int64_t>& mropeSection = {16, 16, 16})
{
    const int64_t total = undLen + genLen;
    const int64_t numHead = numHeadQ + numHeadK + numHeadV;
    const int64_t blockNum = (total + blockSize - 1) / blockSize + 1;

    static optiling::UndGenQkvRmsNormRopeCacheCompileInfo compileInfo = {};

    return gert::TilingContextPara(
        "UndGenQkvRmsNormRopeCache",
        {
            {{{undLen, numHead, headDim}, {undLen, numHead, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{headDim}, {headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{headDim}, {headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{maxPos, headDim}, {maxPos, headDim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{blockNum, blockSize, numHeadK, headDim}, {blockNum, blockSize, numHeadK, headDim}},
             ge::DT_BF16,
             ge::FORMAT_ND},
            {{{blockNum, blockSize, numHeadV, headDim}, {blockNum, blockSize, numHeadV, headDim}},
             ge::DT_BF16,
             ge::FORMAT_ND},
            {{{total}, {total}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{MROPE_AXIS_NUM, total}, {MROPE_AXIS_NUM, total}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{genLen, numHead, headDim}, {genLen, numHead, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{headDim}, {headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{headDim}, {headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{total}, {total}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{total, numHeadQ, headDim}, {total, numHeadQ, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{blockNum, blockSize, numHeadK, headDim}, {blockNum, blockSize, numHeadK, headDim}},
             ge::DT_BF16,
             ge::FORMAT_ND},
            {{{blockNum, blockSize, numHeadV, headDim}, {blockNum, blockSize, numHeadV, headDim}},
             ge::DT_BF16,
             ge::FORMAT_ND},
        },
        {
            {"num_heads_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadQ)},
            {"num_heads_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadK)},
            {"num_heads_v", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadV)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(NORM_EPS)},
            {"mrope_section", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(mropeSection)},
        },
        &compileInfo,
        "Ascend950");
}
} // namespace

// ==================== 成功路径 ====================
// NOTE: 多核/UB 切分尚未实现（见 op_host/*_tiling.cpp 的 TODO），
//       故此处只校验 tilingKey / workspace / tilingData 落盘；
//       切分补齐后需要补 usedCoreNum、blockFactor、ubFactor 的期望值。

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_success_h8_1_1)
{
    auto para = BuildTilingContext(5, 3, 8, 1, 1);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0U);
    EXPECT_GT(tilingInfo.tilingDataSize, 0U);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_success_h16_2_2)
{
    auto para = BuildTilingContext(6652, 4100, 16, 2, 2);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0);
    EXPECT_GT(tilingInfo.tilingDataSize, 0U);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_success_plain_rope_empty_mrope_section)
{
    // mrope_section 为空时退化为 [D/2, 0, 0]，等价标准 RoPE
    auto para = BuildTilingContext(128, 0, 8, 1, 1, HEAD_DIM, BLOCK_SIZE, 4096, {});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0);
}

// ==================== 失败路径 ====================

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_head_dim_unsupported)
{
    // 仅支持 headDim = 128
    auto para = BuildTilingContext(8, 0, 8, 1, 1, 64);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_num_heads_mismatch_n_dim)
{
    // und_qkv 的 N 维按 (8,1,1) 造，但属性声明 (16,2,2)
    auto para = BuildTilingContext(8, 0, 8, 1, 1);
    auto badPara = para;
    // 直接改属性，使 num_heads_q + k + v != N
    badPara.attrs_[0] = {"num_heads_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)};
    ExecuteTestCase(badPara, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_mrope_section_size_invalid)
{
    auto para = BuildTilingContext(8, 0, 8, 1, 1, HEAD_DIM, BLOCK_SIZE, 4096, {16, 16});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_mrope_section_sum_exceeds_half)
{
    auto para = BuildTilingContext(8, 0, 8, 1, 1, HEAD_DIM, BLOCK_SIZE, 4096, {32, 32, 32});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_kv_cache_capacity_insufficient)
{
    // Bn * Bs 小于 total：block_size 造成容量不足
    auto para = BuildTilingContext(8, 0, 8, 1, 1, HEAD_DIM, BLOCK_SIZE, 4096);
    auto badPara = para;
    // k_cache/v_cache 改成只有 1 页 1 行
    badPara.inputTensorDesc_[4] = {{{1, 1, 1, HEAD_DIM}, {1, 1, 1, HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND};
    badPara.inputTensorDesc_[5] = {{{1, 1, 1, HEAD_DIM}, {1, 1, 1, HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND};
    ExecuteTestCase(badPara, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_positions_shape_invalid)
{
    auto para = BuildTilingContext(8, 0, 8, 1, 1);
    auto badPara = para;
    // positions 第一维必须是 3
    badPara.inputTensorDesc_[7] = {{{2, 8}, {2, 8}}, ge::DT_INT64, ge::FORMAT_ND};
    ExecuteTestCase(badPara, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_slot_mapping_length_mismatch)
{
    auto para = BuildTilingContext(8, 0, 8, 1, 1);
    auto badPara = para;
    badPara.inputTensorDesc_[6] = {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND};
    ExecuteTestCase(badPara, ge::GRAPH_FAILED);
}

// ==================== dtype 校验 ====================

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_und_qkv_dtype_invalid)
{
    auto para = BuildTilingContext(8, 0, 8, 1, 1);
    para.inputTensorDesc_[0] = {{{8, 10, HEAD_DIM}, {8, 10, HEAD_DIM}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_cos_sin_cache_dtype_invalid)
{
    // cos_sin_cache 必须是 FLOAT32
    auto para = BuildTilingContext(8, 0, 8, 1, 1);
    para.inputTensorDesc_[3] = {{{4096, HEAD_DIM}, {4096, HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_kv_cache_dtype_invalid)
{
    auto para = BuildTilingContext(8, 0, 8, 1, 1);
    para.inputTensorDesc_[4] = {{{2, BLOCK_SIZE, 1, HEAD_DIM}, {2, BLOCK_SIZE, 1, HEAD_DIM}},
                                ge::DT_FLOAT16, ge::FORMAT_ND};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_slot_mapping_dtype_invalid)
{
    // 索引类张量统一 int64，int32 应被拒
    auto para = BuildTilingContext(8, 0, 8, 1, 1);
    para.inputTensorDesc_[6] = {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_positions_dtype_invalid)
{
    auto para = BuildTilingContext(8, 0, 8, 1, 1);
    para.inputTensorDesc_[7] = {{{MROPE_AXIS_NUM, 8}, {MROPE_AXIS_NUM, 8}}, ge::DT_INT32, ge::FORMAT_ND};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ==================== 支持范围校验 ====================

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_head_combo_unsupported)
{
    // 本期只支持 (8,1,1) 与 (16,2,2)
    auto para = BuildTilingContext(8, 0, 4, 2, 2);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_block_size_unsupported)
{
    // block_size 只支持 128
    auto para = BuildTilingContext(8, 0, 8, 1, 1, HEAD_DIM, 64);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_kv_cache_head_num_mismatch_attr)
{
    // k_cache 的 Hk 与 num_heads_k 不一致
    auto para = BuildTilingContext(8, 0, 8, 1, 1);
    para.inputTensorDesc_[4] = {{{2, BLOCK_SIZE, 2, HEAD_DIM}, {2, BLOCK_SIZE, 2, HEAD_DIM}},
                                ge::DT_BF16, ge::FORMAT_ND};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_gen_qkv_n_dim_mismatch)
{
    // gen_qkv 的 N 维与 und_qkv 不一致
    auto para = BuildTilingContext(8, 4, 8, 1, 1);
    para.inputTensorDesc_[8] = {{{4, 20, HEAD_DIM}, {4, 20, HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_weights_length_mismatch)
{
    auto para = BuildTilingContext(8, 0, 8, 1, 1);
    para.inputTensorDesc_[1] = {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_cos_sin_cache_dim_mismatch)
{
    auto para = BuildTilingContext(8, 0, 8, 1, 1);
    para.inputTensorDesc_[3] = {{{4096, 64}, {4096, 64}}, ge::DT_FLOAT, ge::FORMAT_ND};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ==================== 可选输入缺省的退化路径 ====================
// gen_qkv / gen_weights_q / gen_weights_k / cat_indices 全部不传：
// 通过 IrInstanceNum 把这 4 个可选输入的实例数置 0 表达"未提供"。
TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_success_prefill_only_without_optional_inputs)
{
    constexpr int64_t undLen = 128;
    constexpr int64_t numHeadQ = 8;
    constexpr int64_t numHeadK = 1;
    constexpr int64_t numHeadV = 1;
    constexpr int64_t numHead = numHeadQ + numHeadK + numHeadV;
    constexpr int64_t maxPos = 4096;
    const int64_t blockNum = (undLen + BLOCK_SIZE - 1) / BLOCK_SIZE + 1;

    static optiling::UndGenQkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "UndGenQkvRmsNormRopeCache",
        {
            {{{undLen, numHead, HEAD_DIM}, {undLen, numHead, HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{HEAD_DIM}, {HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{HEAD_DIM}, {HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{maxPos, HEAD_DIM}, {maxPos, HEAD_DIM}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{blockNum, BLOCK_SIZE, numHeadK, HEAD_DIM}, {blockNum, BLOCK_SIZE, numHeadK, HEAD_DIM}},
             ge::DT_BF16,
             ge::FORMAT_ND},
            {{{blockNum, BLOCK_SIZE, numHeadV, HEAD_DIM}, {blockNum, BLOCK_SIZE, numHeadV, HEAD_DIM}},
             ge::DT_BF16,
             ge::FORMAT_ND},
            {{{undLen}, {undLen}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{MROPE_AXIS_NUM, undLen}, {MROPE_AXIS_NUM, undLen}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{undLen, numHeadQ, HEAD_DIM}, {undLen, numHeadQ, HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{blockNum, BLOCK_SIZE, numHeadK, HEAD_DIM}, {blockNum, BLOCK_SIZE, numHeadK, HEAD_DIM}},
             ge::DT_BF16,
             ge::FORMAT_ND},
            {{{blockNum, BLOCK_SIZE, numHeadV, HEAD_DIM}, {blockNum, BLOCK_SIZE, numHeadV, HEAD_DIM}},
             ge::DT_BF16,
             ge::FORMAT_ND},
        },
        {
            {"num_heads_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadQ)},
            {"num_heads_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadK)},
            {"num_heads_v", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadV)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(NORM_EPS)},
            {"mrope_section", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 16, 16})},
        },
        &compileInfo,
        "Ascend950");
    para.inputInstanceNum_ = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0};
    para.outputInstanceNum_ = {1, 1, 1};

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0);
    EXPECT_GT(tilingInfo.tilingDataSize, 0U);
}

// gen_qkv 提供但 gen_weights 缺省时必须失败
TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_gen_weights_missing)
{
    auto para = BuildTilingContext(8, 4, 8, 1, 1);
    // TensorDescription 无默认构造，只能 erase 不能 resize：只保留到 gen_qkv，丢掉 gen_weights_q/k 与 cat_indices
    para.inputTensorDesc_.erase(para.inputTensorDesc_.begin() + 9, para.inputTensorDesc_.end());
    para.inputInstanceNum_ = {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0};
    para.outputInstanceNum_ = {1, 1, 1};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
