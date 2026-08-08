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
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "../quant_block_sparse_attn_host_ut_param.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace QuantBlockSparseAttnUT {

static const std::string OP_NAME = "QuantBlockSparseAttn";

struct QuantBlockSparseAttnCompileInfo {
} compileInfo;

using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

const TensorDesc EmptyInput(gert::StorageShape({0}, {0}), ge::DT_INT32, ge::FORMAT_ND);
const TensorDesc DefaultMetadata(gert::StorageShape({1}, {1}), ge::DT_INT32, ge::FORMAT_ND);

static TensorDesc NormalizeTd(const TensorDesc &td)
{
    if (td.dtype_ == ge::DT_UNDEFINED) {
        return EmptyInput;
    }
    return td;
}

static void SetStride(TensorDesc &td, uint64_t stride0, uint64_t stride1, uint64_t stride2, uint64_t stride3)
{
    td.stride_.SetStride(0U, stride0);
    td.stride_.SetStride(1U, stride1);
    td.stride_.SetStride(2U, stride2);
    td.stride_.SetStride(3U, stride3);
    td.hasStride_ = true;
}

static bool HasPositiveDims(const gert::Shape &shape)
{
    for (size_t i = 0U; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) <= 0) {
            return false;
        }
    }
    return true;
}

static void SetFp8PaStrides(TensorDesc &key, TensorDesc &value, TensorDesc &kDescale)
{
    const gert::Shape &keyShape = key.shape_.GetStorageShape();
    const gert::Shape &valueShape = value.shape_.GetStorageShape();
    const gert::Shape &kDescaleShape = kDescale.shape_.GetStorageShape();
    SetStride(key, 1U, 1U, 1U, 1U);
    SetStride(value, 1U, 1U, 1U, 1U);
    SetStride(kDescale, 1U, 1U, 1U, 1U);
    if (keyShape.GetDimNum() != 4U || valueShape.GetDimNum() != 4U || !HasPositiveDims(keyShape) ||
        !HasPositiveDims(valueShape)) {
        return;
    }

    const uint64_t keyN = static_cast<uint64_t>(keyShape.GetDim(1U));
    const uint64_t keyBlockSize = static_cast<uint64_t>(keyShape.GetDim(2U));
    const uint64_t keyD = static_cast<uint64_t>(keyShape.GetDim(3U));
    const uint64_t valueN = static_cast<uint64_t>(valueShape.GetDim(1U));
    const uint64_t valueBlockSize = static_cast<uint64_t>(valueShape.GetDim(2U));
    const uint64_t valueD = static_cast<uint64_t>(valueShape.GetDim(3U));
    uint64_t paBlockStride = keyN * keyBlockSize * keyD + valueN * valueBlockSize * valueD;
    if (kDescaleShape.GetDimNum() == 4U && HasPositiveDims(kDescaleShape)) {
        const uint64_t kDescaleN = static_cast<uint64_t>(kDescaleShape.GetDim(1U));
        const uint64_t kDescaleBlockSize = static_cast<uint64_t>(kDescaleShape.GetDim(2U));
        const uint64_t kDescaleD = static_cast<uint64_t>(kDescaleShape.GetDim(3U));
        paBlockStride += kDescaleN * kDescaleBlockSize * kDescaleD * sizeof(float);
        SetStride(kDescale, paBlockStride / sizeof(float), kDescaleBlockSize * kDescaleD, kDescaleD, 1U);
    }

    SetStride(key, paBlockStride, keyBlockSize * keyD, keyD, 1U);
    SetStride(value, paBlockStride, valueBlockSize * valueD, valueD, 1U);
}

static QuantBlockSparseAttnTilingUtParam MakeValidFp8Param()
{
    const csv_map csvMap = {
        {"caseName", "direct_fp8_validation"},
        {"expectStatus", "SUCCESS"},
        {"queryShape", "256 4 128"}, {"queryDtype", "FLOAT8_E4M3FN"}, {"queryFormat", "ND"},
        {"keyShape", "4 4 128 128"}, {"keyDtype", "FLOAT8_E4M3FN"}, {"keyFormat", "ND"},
        {"valueShape", "4 4 128 128"}, {"valueDtype", "FLOAT8_E4M3FN"}, {"valueFormat", "ND"},
        {"qDescaleShape", "256 4"}, {"qDescaleDtype", "FLOAT"}, {"qDescaleFormat", "ND"},
        {"kDescaleShape", "4 4 128 1"}, {"kDescaleDtype", "FLOAT"}, {"kDescaleFormat", "ND"},
        {"vDescaleShape", "4"}, {"vDescaleDtype", "FLOAT"}, {"vDescaleFormat", "ND"},
        {"pScaleShape", "1"}, {"pScaleDtype", "FLOAT"}, {"pScaleFormat", "ND"},
        {"cuSeqlensQShape", "2"}, {"cuSeqlensQDtype", "INT32"}, {"cuSeqlensQFormat", "ND"},
        {"sequsedKvShape", "1"}, {"sequsedKvDtype", "INT32"}, {"sequsedKvFormat", "ND"},
        {"sparseIndicesShape", "1 4 2 4"}, {"sparseIndicesDtype", "INT32"},
        {"sparseIndicesFormat", "ND"},
        {"sparseSeqLenShape", "1 4 2"}, {"sparseSeqLenDtype", "INT32"}, {"sparseSeqLenFormat", "ND"},
        {"blockTableShape", "1 8"}, {"blockTableDtype", "INT32"}, {"blockTableFormat", "ND"},
        {"attenMaskShape", "2048 2048"}, {"attenMaskDtype", "UINT8"}, {"attenMaskFormat", "ND"},
        {"metadataShape", "1"}, {"metadataDtype", "INT32"}, {"metadataFormat", "ND"},
        {"attentionOutShape", "256 4 128"}, {"attentionOutDtype", "BF16"}, {"attentionOutFormat", "ND"},
        {"softmaxLseShape", "4 256"}, {"softmaxLseDtype", "FLOAT"}, {"softmaxLseFormat", "ND"},
        {"softmax_scale", "1.0"}, {"sparse_q_block_size", "128"}, {"sparse_kv_block_size", "128"},
        {"layout_kv", "PA_BNSD"}, {"layout_q", "TND"}, {"layout_sparse_indices", "B_N_Qb_Kb"},
        {"layout_out", "TND"}, {"quant_mode", "1"}, {"mask_mode", "3"},
        {"return_softmax_lse", "false"},
    };
    return QuantBlockSparseAttnTilingUtParam(csvMap);
}

static bool ExecuteCase(QuantBlockSparseAttnTilingUtParam param, TilingInfo &tilingInfo,
                        bool provideFp8Strides = true, bool provideFp8Metadata = true,
                        bool useMismatchedValueStride = false)
{
    TensorDesc key = NormalizeTd(param.key);
    TensorDesc value = NormalizeTd(param.value);
    TensorDesc kDescale = NormalizeTd(param.kDescale);
    TensorDesc metadata = NormalizeTd(param.metadata);
    if (param.quant_mode == 1) {
        if (provideFp8Strides) {
            SetFp8PaStrides(key, value, kDescale);
            if (useMismatchedValueStride) {
                value.stride_.SetStride(0U, value.stride_.GetStride(0U) + 1U);
            }
        }
        if (provideFp8Metadata && param.metadata.dtype_ == ge::DT_UNDEFINED) {
            metadata = DefaultMetadata;
        } else if (!provideFp8Metadata) {
            metadata = param.metadata;
        }
    }

    std::vector<TensorDesc> inputTensorDesc({
        NormalizeTd(param.query), key, value, NormalizeTd(param.qDescale), kDescale, NormalizeTd(param.vDescale),
        NormalizeTd(param.pScale), NormalizeTd(param.cuSeqlensQ), NormalizeTd(param.cuSeqlensKv),
        NormalizeTd(param.sequsedQ), NormalizeTd(param.sequsedKv), NormalizeTd(param.sparseIndices),
        NormalizeTd(param.sparseSeqLen), NormalizeTd(param.blockTable), NormalizeTd(param.attenMask), metadata,
    });

    std::vector<TensorDesc> outputTensorDesc({
        NormalizeTd(param.attentionOut),
        NormalizeTd(param.softmaxLse),
    });

    std::vector<OpAttr> attrs({
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(param.softmax_scale)},
        {"sparse_q_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sparse_q_block_size)},
        {"sparse_kv_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sparse_kv_block_size)},
        {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_kv)},
        {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_q)},
        {"layout_sparse_indices", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_sparse_indices)},
        {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_out)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.quant_mode)},
        {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mask_mode)},
        {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(param.return_softmax_lse)},
    });

    gert::TilingContextPara para(OP_NAME, inputTensorDesc, outputTensorDesc, attrs, &compileInfo, "Ascend950", 64,
                                 262144, 65536);
    return ExecuteTiling(para, tilingInfo);
}

class QuantBlockSparseAttnTilingArch35Test : public testing::TestWithParam<QuantBlockSparseAttnTilingUtParam> {
protected:
    static void SetUpTestCase() { std::cout << "QuantBlockSparseAttnTilingArch35Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "QuantBlockSparseAttnTilingArch35Test TearDown" << std::endl; }
};

TEST_P(QuantBlockSparseAttnTilingArch35Test, param)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;
    TilingInfo tilingInfo;
    bool ok = ExecuteCase(param, tilingInfo);

    if (param.expectResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(ok, true);
        EXPECT_GT(tilingInfo.blockNum, 0U);
        EXPECT_GT(tilingInfo.tilingDataSize, 0U);
    } else {
        EXPECT_EQ(ok, false);
    }
}

TEST(QuantBlockSparseAttnFp8ValidationTest, RejectsN1Above128)
{
    auto param = MakeValidFp8Param();
    param.query = TensorDesc(gert::StorageShape({256, 129, 128}, {256, 129, 128}), ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    param.key = TensorDesc(gert::StorageShape({4, 3, 128, 128}, {4, 3, 128, 128}), ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    param.value =
        TensorDesc(gert::StorageShape({4, 3, 128, 128}, {4, 3, 128, 128}), ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    param.qDescale = TensorDesc(gert::StorageShape({256, 129}, {256, 129}), ge::DT_FLOAT, ge::FORMAT_ND);
    param.kDescale = TensorDesc(gert::StorageShape({4, 3, 128, 1}, {4, 3, 128, 1}), ge::DT_FLOAT, ge::FORMAT_ND);
    param.vDescale = TensorDesc(gert::StorageShape({3}, {3}), ge::DT_FLOAT, ge::FORMAT_ND);
    param.sparseIndices = TensorDesc(gert::StorageShape({1, 129, 2, 4}, {1, 129, 2, 4}), ge::DT_INT32, ge::FORMAT_ND);
    param.sparseSeqLen = TensorDesc(gert::StorageShape({1, 129, 2}, {1, 129, 2}), ge::DT_INT32, ge::FORMAT_ND);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteCase(param, tilingInfo));
}

TEST(QuantBlockSparseAttnFp8ValidationTest, RejectsBatchAbove65536)
{
    auto param = MakeValidFp8Param();
    param.sparseIndices =
        TensorDesc(gert::StorageShape({65537, 4, 2, 4}, {65537, 4, 2, 4}), ge::DT_INT32, ge::FORMAT_ND);
    param.sparseSeqLen = TensorDesc(gert::StorageShape({65537, 4, 2}, {65537, 4, 2}), ge::DT_INT32, ge::FORMAT_ND);
    param.blockTable = TensorDesc(gert::StorageShape({65537, 8}, {65537, 8}), ge::DT_INT32, ge::FORMAT_ND);
    param.cuSeqlensQ = TensorDesc(gert::StorageShape({65538}, {65538}), ge::DT_INT32, ge::FORMAT_ND);
    param.sequsedKv = TensorDesc(gert::StorageShape({65537}, {65537}), ge::DT_INT32, ge::FORMAT_ND);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteCase(param, tilingInfo));
}

TEST(QuantBlockSparseAttnFp8ValidationTest, RejectsSparseIndicesN1Mismatch)
{
    auto param = MakeValidFp8Param();
    param.sparseIndices = TensorDesc(gert::StorageShape({1, 8, 2, 4}, {1, 8, 2, 4}), ge::DT_INT32, ge::FORMAT_ND);
    param.sparseSeqLen = TensorDesc(gert::StorageShape({1, 8, 2}, {1, 8, 2}), ge::DT_INT32, ge::FORMAT_ND);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteCase(param, tilingInfo));
}

TEST(QuantBlockSparseAttnFp8ValidationTest, RejectsSparseSeqLenBatchMismatch)
{
    auto param = MakeValidFp8Param();
    param.sparseSeqLen = TensorDesc(gert::StorageShape({2, 4, 2}, {2, 4, 2}), ge::DT_INT32, ge::FORMAT_ND);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteCase(param, tilingInfo));
}

TEST(QuantBlockSparseAttnFp8ValidationTest, RejectsSparseSeqLenN1Mismatch)
{
    auto param = MakeValidFp8Param();
    param.sparseSeqLen = TensorDesc(gert::StorageShape({1, 8, 2}, {1, 8, 2}), ge::DT_INT32, ge::FORMAT_ND);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteCase(param, tilingInfo));
}

TEST(QuantBlockSparseAttnFp8ValidationTest, RejectsSparseSeqLenMaxQbMismatch)
{
    auto param = MakeValidFp8Param();
    param.sparseSeqLen = TensorDesc(gert::StorageShape({1, 4, 3}, {1, 4, 3}), ge::DT_INT32, ge::FORMAT_ND);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteCase(param, tilingInfo));
}

TEST(QuantBlockSparseAttnFp8ValidationTest, RejectsBlockTableShorterThanMaxKb)
{
    auto param = MakeValidFp8Param();
    param.sparseIndices = TensorDesc(gert::StorageShape({1, 4, 2, 9}, {1, 4, 2, 9}), ge::DT_INT32, ge::FORMAT_ND);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteCase(param, tilingInfo));
}

TEST(QuantBlockSparseAttnFp8ValidationTest, RejectsMissingMetadata)
{
    auto param = MakeValidFp8Param();
    param.metadata = TD_DEFAULT;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteCase(param, tilingInfo, true, false));
}

TEST(QuantBlockSparseAttnFp8ValidationTest, RejectsEmptyMetadata)
{
    auto param = MakeValidFp8Param();
    param.metadata = TensorDesc(gert::StorageShape({0}, {0}), ge::DT_INT32, ge::FORMAT_ND);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteCase(param, tilingInfo));
}

TEST(QuantBlockSparseAttnFp8ValidationTest, RejectsEmptyQuery)
{
    auto param = MakeValidFp8Param();
    param.query = TensorDesc(gert::StorageShape({0, 4, 128}, {0, 4, 128}), ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteCase(param, tilingInfo));
}

TEST(QuantBlockSparseAttnFp8ValidationTest, RejectsMetadataNot1D)
{
    auto param = MakeValidFp8Param();
    param.metadata = TensorDesc(gert::StorageShape({1, 1}, {1, 1}), ge::DT_INT32, ge::FORMAT_ND);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteCase(param, tilingInfo));
}

TEST(QuantBlockSparseAttnFp8ValidationTest, RejectsPScaleRank2Singleton)
{
    auto param = MakeValidFp8Param();
    param.pScale = TensorDesc(gert::StorageShape({1, 1}, {1, 1}), ge::DT_FLOAT, ge::FORMAT_ND);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteCase(param, tilingInfo));
}

TEST(QuantBlockSparseAttnFp8ValidationTest, AllowsEmptyPScaleWithoutDtypeRestriction)
{
    auto param = MakeValidFp8Param();
    param.pScale = TensorDesc(gert::StorageShape({0}, {0}), ge::DT_FLOAT8_E8M0, ge::FORMAT_ND);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteCase(param, tilingInfo));
}

TEST(QuantBlockSparseAttnFp8ValidationTest, RejectsMissingKvStride)
{
    auto param = MakeValidFp8Param();
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteCase(param, tilingInfo, false));
}

TEST(QuantBlockSparseAttnFp8ValidationTest, RejectsMismatchedKvStride0)
{
    auto param = MakeValidFp8Param();
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteCase(param, tilingInfo, true, true, true));
}

TEST(QuantBlockSparseAttnFp8ValidationTest, RejectsNegativeMaskMode)
{
    auto param = MakeValidFp8Param();
    param.mask_mode = -1;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteCase(param, tilingInfo));
}

TEST(QuantBlockSparseAttnFp8ValidationTest, RejectsNegativeQuantMode)
{
    auto param = MakeValidFp8Param();
    param.quant_mode = -1;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteCase(param, tilingInfo));
}

TEST(QuantBlockSparseAttnFp8ValidationTest, RejectsNegativeQBlockSize)
{
    auto param = MakeValidFp8Param();
    param.sparse_q_block_size = -1;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteCase(param, tilingInfo));
}

TEST(QuantBlockSparseAttnFp8ValidationTest, RejectsNegativeKvBlockSize)
{
    auto param = MakeValidFp8Param();
    param.sparse_kv_block_size = -1;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteCase(param, tilingInfo));
}

INSTANTIATE_TEST_SUITE_P(
    QuantBlockSparseAttnTiling, QuantBlockSparseAttnTilingArch35Test,
    testing::ValuesIn(GetCasesFromCsv<QuantBlockSparseAttnTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<QuantBlockSparseAttnTilingUtParam>);

} // namespace QuantBlockSparseAttnUT
