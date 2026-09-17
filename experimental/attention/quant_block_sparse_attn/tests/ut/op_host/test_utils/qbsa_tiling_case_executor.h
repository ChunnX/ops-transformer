/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBSA_TILING_CASE_EXECUTOR_H
#define QBSA_TILING_CASE_EXECUTOR_H

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "../../../../op_host/tiling_common/qbsa_op_contract.h"
#include "../../../../op_kernel/arch35/qbsa_fp8/qbsa_fp8_tiling_data.h"
#include "../../../../op_kernel/arch35/qbsa_mxfp8/qbsa_mxfp8_tiling_data.h"
#include "../../../../op_kernel/arch35/common/quant_block_sparse_attn_tiling_key_registry.h"
#include "qbsa_host_ut_param.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace QuantBlockSparseAttnUT {

using QbsaTilingTensorDesc = gert::TilingContextPara::TensorDescription;
using QbsaTilingOpAttr = gert::TilingContextPara::OpAttr;

inline const QbsaTilingTensorDesc &GetQbsaEmptyInput()
{
    static const QbsaTilingTensorDesc emptyInput(gert::StorageShape({0}, {0}), ge::DT_INT32, ge::FORMAT_ND);
    return emptyInput;
}

inline const QbsaTilingTensorDesc &GetQbsaDefaultMetadata()
{
    static const QbsaTilingTensorDesc defaultMetadata(gert::StorageShape({1}, {1}), ge::DT_INT32, ge::FORMAT_ND);
    return defaultMetadata;
}

inline QbsaTilingTensorDesc NormalizeQbsaTensorDesc(const QbsaTilingTensorDesc &tensorDesc)
{
    return tensorDesc.dtype_ == ge::DT_UNDEFINED ? GetQbsaEmptyInput() : tensorDesc;
}

inline void SetQbsaStride(QbsaTilingTensorDesc &tensorDesc, uint64_t stride0, uint64_t stride1, uint64_t stride2,
                          uint64_t stride3)
{
    tensorDesc.stride_.SetStride(0U, stride0);
    tensorDesc.stride_.SetStride(1U, stride1);
    tensorDesc.stride_.SetStride(2U, stride2);
    tensorDesc.stride_.SetStride(3U, stride3);
    tensorDesc.stride_.SetDimNum(4U);
    tensorDesc.hasStride_ = true;
}

inline bool HasPositiveQbsaDims(const gert::Shape &shape)
{
    for (size_t i = 0U; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) <= 0) {
            return false;
        }
    }
    return true;
}

inline void SetQbsaFp8PaStrides(QbsaTilingTensorDesc &key, QbsaTilingTensorDesc &value, QbsaTilingTensorDesc &kDescale)
{
    const gert::Shape &keyShape = key.shape_.GetStorageShape();
    const gert::Shape &valueShape = value.shape_.GetStorageShape();
    const gert::Shape &kDescaleShape = kDescale.shape_.GetStorageShape();
    SetQbsaStride(key, 1U, 1U, 1U, 1U);
    SetQbsaStride(value, 1U, 1U, 1U, 1U);
    SetQbsaStride(kDescale, 1U, 1U, 1U, 1U);
    if (keyShape.GetDimNum() != 4U || valueShape.GetDimNum() != 4U || !HasPositiveQbsaDims(keyShape) ||
        !HasPositiveQbsaDims(valueShape)) {
        return;
    }

    const uint64_t keyN = static_cast<uint64_t>(keyShape.GetDim(1U));
    const uint64_t keyBlockSize = static_cast<uint64_t>(keyShape.GetDim(2U));
    const uint64_t keyD = static_cast<uint64_t>(keyShape.GetDim(3U));
    const uint64_t valueN = static_cast<uint64_t>(valueShape.GetDim(1U));
    const uint64_t valueBlockSize = static_cast<uint64_t>(valueShape.GetDim(2U));
    const uint64_t valueD = static_cast<uint64_t>(valueShape.GetDim(3U));
    uint64_t paBlockStride = keyN * keyBlockSize * keyD + valueN * valueBlockSize * valueD;
    if (kDescaleShape.GetDimNum() == 4U && HasPositiveQbsaDims(kDescaleShape)) {
        const uint64_t kDescaleN = static_cast<uint64_t>(kDescaleShape.GetDim(1U));
        const uint64_t kDescaleBlockSize = static_cast<uint64_t>(kDescaleShape.GetDim(2U));
        const uint64_t kDescaleD = static_cast<uint64_t>(kDescaleShape.GetDim(3U));
        paBlockStride += kDescaleN * kDescaleBlockSize * kDescaleD * sizeof(float);
        SetQbsaStride(kDescale, paBlockStride / sizeof(float), kDescaleBlockSize * kDescaleD, kDescaleD, 1U);
    }

    SetQbsaStride(key, paBlockStride, keyBlockSize * keyD, keyD, 1U);
    SetQbsaStride(value, paBlockStride, valueBlockSize * valueD, valueD, 1U);
}

inline void PrepareQbsaFp8TilingParam(QuantBlockSparseAttnTilingUtParam &param)
{
    if (!param.provideFp8Strides) {
        return;
    }
    SetQbsaFp8PaStrides(param.key, param.value, param.kDescale);
    if (param.mismatchValueStride) {
        param.value.stride_.SetStride(0U, param.value.stride_.GetStride(0U) + 1U);
    }
}

inline bool ExecuteQbsaTilingCase(QuantBlockSparseAttnTilingUtParam param, TilingInfo &tilingInfo,
                                  std::size_t tilingDataCapacity = 4096U)
{
    struct QbsaCompileInfo {};
    static QbsaCompileInfo compileInfo;

    QbsaTilingTensorDesc key = NormalizeQbsaTensorDesc(param.key);
    QbsaTilingTensorDesc value = NormalizeQbsaTensorDesc(param.value);
    QbsaTilingTensorDesc kDescale = NormalizeQbsaTensorDesc(param.kDescale);
    QbsaTilingTensorDesc metadata = NormalizeQbsaTensorDesc(param.metadata);
    if (param.provideMetadata && param.metadata.dtype_ == ge::DT_UNDEFINED) {
        metadata = GetQbsaDefaultMetadata();
    } else if (!param.provideMetadata) {
        metadata = param.metadata;
    }

    std::vector<QbsaTilingTensorDesc> inputTensorDesc({
        NormalizeQbsaTensorDesc(param.query),
        key,
        value,
        NormalizeQbsaTensorDesc(param.qDescale),
        kDescale,
        NormalizeQbsaTensorDesc(param.vDescale),
        NormalizeQbsaTensorDesc(param.pScale),
        NormalizeQbsaTensorDesc(param.cuSeqlensQ),
        NormalizeQbsaTensorDesc(param.cuSeqlensKv),
        NormalizeQbsaTensorDesc(param.sequsedQ),
        NormalizeQbsaTensorDesc(param.sequsedKv),
        NormalizeQbsaTensorDesc(param.sparseIndices),
        NormalizeQbsaTensorDesc(param.sparseSeqLen),
        NormalizeQbsaTensorDesc(param.blockTable),
        NormalizeQbsaTensorDesc(param.attenMask),
        metadata,
    });
    std::vector<QbsaTilingTensorDesc> outputTensorDesc({
        NormalizeQbsaTensorDesc(param.attentionOut),
        NormalizeQbsaTensorDesc(param.softmaxLse),
    });
    std::vector<QbsaTilingOpAttr> attrs({
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

    gert::TilingContextPara para("QuantBlockSparseAttn", inputTensorDesc, outputTensorDesc, attrs, &compileInfo,
                                 "Ascend950", 64, 262144, tilingDataCapacity);
    return ExecuteTiling(para, tilingInfo);
}

inline bool ExecuteQbsaFp8TilingCase(
    QuantBlockSparseAttnTilingUtParam param, TilingInfo &tilingInfo,
    std::size_t tilingDataCapacity = sizeof(optiling::QuantBlockSparseAttnFp8TilingData))
{
    PrepareQbsaFp8TilingParam(param);
    return ExecuteQbsaTilingCase(param, tilingInfo, tilingDataCapacity);
}

template <typename TilingData>
inline TilingData DecodeQbsaTilingData(const TilingInfo &tilingInfo)
{
    TilingData data{};
    if (tilingInfo.tilingDataSize != sizeof(data) || tilingInfo.tilingData == nullptr) {
        ADD_FAILURE() << "Serialized tiling data does not match the selected mode's structure";
        return data;
    }
    std::memcpy(&data, tilingInfo.tilingData.get(), sizeof(data));
    return data;
}

inline void ExpectQbsaCommonTilingData(const QuantBlockSparseAttnTilingUtParam &param, const TilingInfo &tilingInfo,
                                       const optiling::QuantBlockSparseAttnCommonParams &common)
{
    EXPECT_EQ(common.baseParams.coreNum, tilingInfo.blockNum);
    EXPECT_EQ(common.baseParams.scaleValue, param.softmax_scale);
    EXPECT_EQ(common.paParams.qBlockSize, param.sparse_q_block_size);
    EXPECT_EQ(common.paParams.kvBlockSize, param.sparse_kv_block_size);
    EXPECT_EQ(common.sparseParams.sparseSeqLenStride, param.sparseIndices.shape_.GetStorageShape().GetDim(2));
    EXPECT_EQ(common.sparseParams.sparseIndicesStride, param.sparseIndices.shape_.GetStorageShape().GetDim(3));
    EXPECT_EQ(common.outputParams.totalOutputSize, param.attentionOut.shape_.GetStorageShape().GetShapeSize());
}

} // namespace QuantBlockSparseAttnUT

#endif // QBSA_TILING_CASE_EXECUTOR_H
