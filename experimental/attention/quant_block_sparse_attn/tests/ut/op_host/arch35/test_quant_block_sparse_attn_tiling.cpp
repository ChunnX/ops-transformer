/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

class QuantBlockSparseAttnTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantBlockSparseAttnTilingArch35 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantBlockSparseAttnTilingArch35 TearDown" << std::endl;
    }
};

namespace {
using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

struct QuantBlockSparseAttnCompileInfo {};

const TensorDesc EmptyInput({{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND});

std::vector<TensorDesc> MakeValidInputs(int64_t t = 256, int64_t n1 = 4, int64_t n2 = 4)
{
    return {
        {{{t, n1, 128}, {t, n1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{{4, n2, 128, 128}, {4, n2, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{{4, n2, 128, 128}, {4, n2, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{{n1}, {n1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{n2}, {n2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{n2}, {n2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        EmptyInput,
        EmptyInput,
        EmptyInput,
        EmptyInput,
        {{{1, n1, 2, 4}, {1, n1, 2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1, n1, 2}, {1, n1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        EmptyInput,
        {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
        EmptyInput,
    };
}

std::vector<TensorDesc> MakeValidOutputs(int64_t t = 256, int64_t n1 = 4)
{
    return {
        {{{t, n1, 128}, {t, n1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{n1, t}, {n1, t}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
}

std::vector<OpAttr> MakeValidAttrs(int64_t qBlockSize = 128, int64_t kvBlockSize = 128, int64_t maskMode = 3,
                                   int64_t maxSeqlenKv = 256)
{
    return {
        {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maxSeqlenKv)},
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"sparse_q_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(qBlockSize)},
        {"sparse_kv_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kvBlockSize)},
        {"paBlockStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
        {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
        {"layout_sparse_indices", Ops::Transformer::AnyValue::CreateFrom<std::string>("B_N_Qb_Kb")},
        {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maskMode)},
        {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
    };
}

gert::TilingContextPara BuildTilingPara(const std::vector<TensorDesc> &inputs, const std::vector<TensorDesc> &outputs,
                                        const std::vector<OpAttr> &attrs)
{
    static QuantBlockSparseAttnCompileInfo compileInfo;
    return gert::TilingContextPara("QuantBlockSparseAttn", inputs, outputs, attrs, &compileInfo, "Ascend950", 64,
                                   262144, 65536);
}

void ExpectTilingResult(const gert::TilingContextPara &para, bool expectSuccess)
{
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(para, tilingInfo);
    EXPECT_EQ(ok, expectSuccess);
    if (expectSuccess) {
        EXPECT_GT(tilingInfo.blockNum, 0U);
        EXPECT_GT(tilingInfo.tilingDataSize, 0U);
    }
}

std::vector<OpAttr> MakeAttrsEx(const std::string &layoutQ, int64_t paBlockStride, int64_t maskMode = 3,
                                int64_t maxSeqlenKv = 256, bool returnLse = false, int64_t qBlockSize = 128,
                                int64_t kvBlockSize = 128)
{
    return {
        {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maxSeqlenKv)},
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"sparse_q_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(qBlockSize)},
        {"sparse_kv_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kvBlockSize)},
        {"paBlockStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(paBlockStride)},
        {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
        {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutQ)},
        {"layout_sparse_indices", Ops::Transformer::AnyValue::CreateFrom<std::string>("B_N_Qb_Kb")},
        {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maskMode)},
        {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(returnLse)},
    };
}

std::vector<TensorDesc> MakeBSNDInputs(int64_t b = 4, int64_t s = 256, int64_t n1 = 4, int64_t n2 = 4)
{
    int64_t qb = (s + 127) / 128;
    return {
        {{{b, s, n1, 128}, {b, s, n1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{{4, n2, 128, 128}, {4, n2, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{{4, n2, 128, 128}, {4, n2, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{{n1}, {n1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{n2}, {n2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{n2}, {n2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        EmptyInput,
        EmptyInput,
        EmptyInput,
        EmptyInput,
        {{{b, n1, qb, 4}, {b, n1, qb, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{b, n1, qb}, {b, n1, qb}}, ge::DT_INT32, ge::FORMAT_ND},
        EmptyInput,
        {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
        EmptyInput,
    };
}

std::vector<TensorDesc> MakeNTDInputs(int64_t n1 = 4, int64_t t = 256, int64_t n2 = 4)
{
    return {
        {{{n1, t, 128}, {n1, t, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{{4, n2, 128, 128}, {4, n2, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{{4, n2, 128, 128}, {4, n2, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{{n1}, {n1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{n2}, {n2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{n2}, {n2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        EmptyInput,
        EmptyInput,
        EmptyInput,
        EmptyInput,
        {{{1, n1, 2, 4}, {1, n1, 2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1, n1, 2}, {1, n1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        EmptyInput,
        {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
        EmptyInput,
    };
}

std::vector<TensorDesc> Make1DKVInputs(int64_t t = 256, int64_t n1 = 4, int64_t n2 = 4, int64_t keyStorageSize = 65536)
{
    return {
        {{{t, n1, 128}, {t, n1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{{keyStorageSize}, {keyStorageSize}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{{4, n2, 128, 128}, {4, n2, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{{n1}, {n1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{n2}, {n2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{n2}, {n2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        EmptyInput,
        EmptyInput,
        EmptyInput,
        EmptyInput,
        {{{1, n1, 2, 4}, {1, n1, 2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1, n1, 2}, {1, n1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        EmptyInput,
        {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
        EmptyInput,
    };
}
} // namespace

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_basic)
{
    ExpectTilingResult(BuildTilingPara(MakeValidInputs(), MakeValidOutputs(), MakeValidAttrs()), true);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_gqa)
{
    ExpectTilingResult(BuildTilingPara(MakeValidInputs(256, 8, 2), MakeValidOutputs(256, 8), MakeValidAttrs()), true);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_mask_mode_0)
{
    ExpectTilingResult(BuildTilingPara(MakeValidInputs(), MakeValidOutputs(), MakeValidAttrs(128, 128, 0)), true);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_query_dtype)
{
    auto inputs = MakeValidInputs();
    inputs[0] = TensorDesc({{256, 4, 128}, {256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_block_size)
{
    ExpectTilingResult(BuildTilingPara(MakeValidInputs(), MakeValidOutputs(), MakeValidAttrs(64, 128, 3)), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_mask_mode)
{
    ExpectTilingResult(BuildTilingPara(MakeValidInputs(), MakeValidOutputs(), MakeValidAttrs(128, 128, 5)), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_bsnd_layout)
{
    ExpectTilingResult(BuildTilingPara(MakeBSNDInputs(), MakeValidOutputs(), MakeAttrsEx("BSND", 0)), true);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_ntd_layout)
{
    ExpectTilingResult(BuildTilingPara(MakeNTDInputs(), MakeValidOutputs(), MakeAttrsEx("NTD", 0)), true);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_1d_combined_kv)
{
    ExpectTilingResult(BuildTilingPara(Make1DKVInputs(), MakeValidOutputs(), MakeAttrsEx("TND", 16384)), true);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_kv_dtype)
{
    auto inputs = MakeValidInputs();
    inputs[1] = TensorDesc({{4, 4, 128, 128}, {4, 4, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    inputs[2] = TensorDesc({{4, 4, 128, 128}, {4, 4, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_with_block_table)
{
    auto inputs = MakeValidInputs();
    inputs[13] = TensorDesc({{1, 8}, {1, 8}}, ge::DT_INT32, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), true);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_return_softmax_lse)
{
    ExpectTilingResult(BuildTilingPara(MakeValidInputs(), MakeValidOutputs(), MakeAttrsEx("TND", 0, 3, 256, true)),
                       true);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_hifloat8_dtype)
{
    auto inputs = MakeValidInputs();
    inputs[0] = TensorDesc({{256, 4, 128}, {256, 4, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND);
    inputs[1] = TensorDesc({{4, 4, 128, 128}, {4, 4, 128, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND);
    inputs[2] = TensorDesc({{4, 4, 128, 128}, {4, 4, 128, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), true);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_query_layout_mismatch)
{
    ExpectTilingResult(BuildTilingPara(MakeValidInputs(), MakeValidOutputs(), MakeAttrsEx("BSND", 0)), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_key_3d)
{
    auto inputs = MakeValidInputs();
    inputs[1] = TensorDesc({{4, 128, 128}, {4, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    inputs[2] = TensorDesc({{4, 128, 128}, {4, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_sparse_indices_3d)
{
    auto inputs = MakeValidInputs();
    inputs[11] = TensorDesc({{1, 4, 4}, {1, 4, 4}}, ge::DT_INT32, ge::FORMAT_ND);
    inputs[12] = TensorDesc({{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_atten_mask_3d)
{
    auto inputs = MakeValidInputs();
    inputs[14] = TensorDesc({{256, 256, 1}, {256, 256, 1}}, ge::DT_UINT8, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_dsize)
{
    auto inputs = MakeValidInputs();
    inputs[0] = TensorDesc({{256, 4, 64}, {256, 4, 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_n1_n2_mismatch_4d)
{
    auto inputs = MakeValidInputs();
    inputs[1] = TensorDesc({{4, 3, 128, 128}, {4, 3, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    inputs[2] = TensorDesc({{4, 3, 128, 128}, {4, 3, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_block_table_b)
{
    auto inputs = MakeValidInputs();
    inputs[13] = TensorDesc({{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_1d_kv_pa_block_stride_zero)
{
    ExpectTilingResult(BuildTilingPara(Make1DKVInputs(), MakeValidOutputs(), MakeAttrsEx("TND", 0)), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_1d_kv_invalid_storage_size)
{
    auto inputs = Make1DKVInputs(256, 4, 4, 65537);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeAttrsEx("TND", 16384)), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_1d_kv_n1_n2_mismatch)
{
    auto inputs = Make1DKVInputs();
    inputs[5] = TensorDesc({{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeAttrsEx("TND", 16384)), false);
}
