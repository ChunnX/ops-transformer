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
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "../quant_block_sparse_attn_host_ut_param.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../../op_kernel/arch35/quant_block_sparse_attn_tiling_data.h"
#include "../../../../op_host/quant_block_sparse_attn_check.h"

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
    td.stride_.SetDimNum(4U);
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

static bool ExecuteCase(QuantBlockSparseAttnTilingUtParam param, TilingInfo &tilingInfo)
{
    TensorDesc key = NormalizeTd(param.key);
    TensorDesc value = NormalizeTd(param.value);
    TensorDesc kDescale = NormalizeTd(param.kDescale);
    TensorDesc metadata = NormalizeTd(param.metadata);
    if (param.quant_mode == 1) {
        if (param.provideFp8Strides) {
            SetFp8PaStrides(key, value, kDescale);
            if (param.mismatchValueStride) {
                value.stride_.SetStride(0U, value.stride_.GetStride(0U) + 1U);
            }
        }
    }
    if (param.provideFp8Metadata && param.metadata.dtype_ == ge::DT_UNDEFINED) {
        metadata = DefaultMetadata;
    } else if (!param.provideFp8Metadata) {
        metadata = param.metadata;
    }
    std::vector<TensorDesc> inputTensorDesc({
        NormalizeTd(param.query),
        key,
        value,
        NormalizeTd(param.qDescale),
        kDescale,
        NormalizeTd(param.vDescale),
        NormalizeTd(param.pScale),
        NormalizeTd(param.cuSeqlensQ),
        NormalizeTd(param.cuSeqlensKv),
        NormalizeTd(param.sequsedQ),
        NormalizeTd(param.sequsedKv),
        NormalizeTd(param.sparseIndices),
        NormalizeTd(param.sparseSeqLen),
        NormalizeTd(param.blockTable),
        NormalizeTd(param.attenMask),
        metadata,
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
    static void SetUpTestCase()
    {
        std::cout << "QuantBlockSparseAttnTilingArch35Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantBlockSparseAttnTilingArch35Test TearDown" << std::endl;
    }
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
        ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(optiling::QuantBlockSparseAttnTilingData));
        optiling::QuantBlockSparseAttnTilingData data{};
        std::memcpy(&data, tilingInfo.tilingData.get(), sizeof(data));
        const auto &common = data.commonParams;
        EXPECT_EQ(common.baseParams.coreNum, tilingInfo.blockNum);
        EXPECT_EQ(common.baseParams.scaleValue, param.softmax_scale);
        EXPECT_EQ(common.paParams.qBlockSize, param.sparse_q_block_size);
        EXPECT_EQ(common.paParams.kvBlockSize, param.sparse_kv_block_size);
        EXPECT_EQ(common.sparseParams.sparseSeqLenStride, param.sparseIndices.shape_.GetStorageShape().GetDim(2));
        EXPECT_EQ(common.sparseParams.sparseIndicesStride, param.sparseIndices.shape_.GetStorageShape().GetDim(3));
        EXPECT_EQ(common.outputParams.totalOutputSize, param.attentionOut.shape_.GetStorageShape().GetShapeSize());
        if (param.quant_mode == 1) {
            EXPECT_EQ(data.fp8Params.needInit, 1U);
            EXPECT_EQ(data.mxParams.scaleGroupSize, 0U);
        } else if (param.quant_mode == 2) {
            EXPECT_EQ(data.mxParams.scaleGroupSize, 64U);
            EXPECT_EQ(data.mxParams.scaleLastDim, 2U);
            EXPECT_EQ(data.fp8Params.needInit, 0U);
        }
    } else {
        EXPECT_EQ(ok, false);
    }
}

INSTANTIATE_TEST_SUITE_P(
    QuantBlockSparseAttnTiling, QuantBlockSparseAttnTilingArch35Test,
    testing::ValuesIn(GetCasesFromCsv<QuantBlockSparseAttnTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<QuantBlockSparseAttnTilingUtParam>);

TEST(QuantBlockSparseAttnFeatureDispatch, UnsupportedModeDoesNotFallBackToFp8)
{
    // No feature-specific inputs are supplied: dispatch must reject these modes
    // before invoking any FP8/MXFP8 tensor checks.
    for (uint32_t mode : {0U, 3U, 4U, UINT32_MAX}) {
        optiling::QuantBlockSparseAttnTilingInfo info;
        info.quantModeVal = mode;
        optiling::QuantBlockSparseAttnCheck checker(info);
        EXPECT_EQ(checker.Process(), ge::GRAPH_FAILED);
    }
}

} // namespace QuantBlockSparseAttnUT
