/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_block_sparse_attn_check.cpp
 * \brief QuantBlockSparseAttn parameter validation implementation.
 */

#include <array>
#include <cstddef>
#include <string>

#include "quant_block_sparse_attn_check.h"
#include "quant_block_sparse_attn_tiling.h"
#include "log/log.h"

namespace optiling {
namespace {
constexpr const char *kOpName = "QuantBlockSparseAttn";
constexpr size_t DIM_NUM_1 = 1U;
constexpr size_t DIM_NUM_4 = 4U;
constexpr int64_t BSA_MXFP8_P_SCALE_SIZE = 1;

template <size_t N>
std::string ShapeToString(const std::array<int64_t, N> &shape)
{
    std::string result = "[";
    for (size_t i = 0U; i < N; ++i) {
        if (i != 0U) {
            result += ", ";
        }
        result += std::to_string(shape[i]);
    }
    result += "]";
    return result;
}

template <size_t N>
bool IsShapeEqual(const gert::Shape &actualShape, const std::array<int64_t, N> &expectedShape)
{
    if (actualShape.GetDimNum() != N) {
        return false;
    }
    for (size_t i = 0U; i < N; ++i) {
        if (actualShape.GetDim(i) != expectedShape[i]) {
            return false;
        }
    }
    return true;
}

bool IsShapeEqual(const gert::Shape &lhs, const gert::Shape &rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0U; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

template <size_t N>
ge::graphStatus CheckInputShape(const BSARequiredParaInfo &input, const char *inputName,
                                const std::array<int64_t, N> &expectedShape)
{
    if (input.shape == nullptr) {
        OP_LOGE(kOpName, "%s shape is nullptr", inputName);
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &actualShape = input.shape->GetStorageShape();
    if (!IsShapeEqual(actualShape, expectedShape)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            kOpName, inputName, Ops::Base::ToString(actualShape),
            "must be " + ShapeToString(expectedShape) + " in quant_mode=2 MXFP8 full-quant scenario");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckInputDtype(const BSARequiredParaInfo &input, const char *inputName, ge::DataType expectedType,
                                const char *expectedTypeName)
{
    if (input.desc == nullptr) {
        OP_LOGE(kOpName, "%s desc is nullptr", inputName);
        return ge::GRAPH_FAILED;
    }
    const ge::DataType actualType = input.desc->GetDataType();
    if (actualType != expectedType) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, inputName, std::to_string(static_cast<int32_t>(actualType)),
                                  expectedTypeName);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckMXFP8FullQuantScaleInputs(const QuantBlockSparseAttnParaInfo &opParamInfo)
{
    // MX matmul 通过硬件 LoadData 消费 Q/K/V/P scale，因此四个 scale tensor 都必须传入。
    // 该 helper 只检查是否存在；dtype/shape 分别由 CheckScaleDtype 和
    // CheckMXFP8FullQuantShape 校验。
    if (opParamInfo.pScale.shape == nullptr || opParamInfo.kDescale.shape == nullptr ||
        opParamInfo.vDescale.shape == nullptr || opParamInfo.qDescale.shape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "quantScale1/keyAntiquantScale/valueAntiquantScale/dequantScaleQuery", "nullptr",
            "all must be passed in quant_mode=2 MXFP8 full-quant scenario");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckKVDtypeConsistency(const QuantBlockSparseAttnParaInfo &opParamInfo)
{
    if (opParamInfo.key.desc == nullptr || opParamInfo.value.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const ge::DataType keyType = opParamInfo.key.desc->GetDataType();
    const ge::DataType valueType = opParamInfo.value.desc->GetDataType();
    if (keyType != valueType) {
        OP_LOGE_FOR_INVALID_DTYPE(
            kOpName, "key/value",
            std::to_string(static_cast<int32_t>(keyType)) + "/" + std::to_string(static_cast<int32_t>(valueType)),
            "key and value must have the same dtype");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

QuantBlockSparseAttnCheck::QuantBlockSparseAttnCheck(const QuantBlockSparseAttnTilingInfo &tilingInfo)
    : tilingInfo_(tilingInfo)
{
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckAttrs() const
{
    if (tilingInfo_.layoutKVStr != "PA_BNSD" || tilingInfo_.layoutSparseIndicesStr != "B_N_Qb_Kb" ||
        (tilingInfo_.quantModeVal != BSA_QUANT_MODE_FP8 &&
         tilingInfo_.quantModeVal != BSA_QUANT_MODE_MXFP8_FULL_QUANT)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "layout_kv/layout_sparse_indices/quant_mode",
            tilingInfo_.layoutKVStr + "/" + tilingInfo_.layoutSparseIndicesStr + "/" +
                std::to_string(tilingInfo_.quantModeVal),
            "layout_kv must be PA_BNSD, layout_sparse_indices must be B_N_Qb_Kb, quant_mode must be 1 or 2");
        return ge::GRAPH_FAILED;
    }

    if (tilingInfo_.quantModeVal == BSA_QUANT_MODE_MXFP8_FULL_QUANT) {
        // MXFP8 特性与原 FP8 约束分开处理。当前 MX 仅支持 TND query/output、
        // PA_BNSD KV 以及 B_N_Qb_Kb sparse indices。
        if (tilingInfo_.layoutQStr != "TND" || tilingInfo_.layoutOutStr != "TND") {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                kOpName, "layout_q/layout_out", tilingInfo_.layoutQStr + "/" + tilingInfo_.layoutOutStr,
                "must both be TND in quant_mode=2 MXFP8 full-quant scenario");
            return ge::GRAPH_FAILED;
        }
        OP_LOGD(kOpName, "quant_mode=2 maps to queryQuantMode/keyAntiquantMode/valueAntiquantMode=%u/%u/%u",
                BSA_MXFP8_PER_TOKEN_GROUP_MODE, BSA_MXFP8_PER_TOKEN_GROUP_MODE,
                BSA_MXFP8_PER_CHANNEL_GROUP_MODE);
    } else if (tilingInfo_.quantModeVal != BSA_QUANT_MODE_FP8) {
        OP_LOGE_FOR_INVALID_VALUE(kOpName, "quant_mode", std::to_string(tilingInfo_.quantModeVal), "1 or 2");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckDtype() const
{
    if (CheckKVDtypeConsistency(tilingInfo_.opParamInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.quantModeVal == BSA_QUANT_MODE_MXFP8_FULL_QUANT) {
        // MX 全量化场景中 Q/K/V 均为 fp8_e4m3fn。scale tensor 为 fp8_e8m0，
        // 且 shape 规则不同，因此单独在 scale 校验中处理。
        if (tilingInfo_.qDtype != ge::DT_FLOAT8_E4M3FN || tilingInfo_.kvDtype != ge::DT_FLOAT8_E4M3FN) {
            OP_LOGE_FOR_INVALID_DTYPE(
                kOpName, "query/key/value",
                std::to_string(static_cast<int>(tilingInfo_.qDtype)) + "/" +
                    std::to_string(static_cast<int>(tilingInfo_.kvDtype)),
                "FLOAT8_E4M3FN in quant_mode=2 MXFP8 full-quant scenario");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    } else {
        if (tilingInfo_.qDtype != ge::DT_FLOAT8_E4M3FN && tilingInfo_.qDtype != ge::DT_HIFLOAT8) {
            OP_LOGE_FOR_INVALID_DTYPE(kOpName, "query", std::to_string(static_cast<int>(tilingInfo_.qDtype)),
                                      "FLOAT8_E4M3FN or HIFLOAT8");
            return ge::GRAPH_FAILED;
        }
        if (tilingInfo_.kvDtype != ge::DT_FLOAT8_E4M3FN && tilingInfo_.kvDtype != ge::DT_HIFLOAT8) {
            OP_LOGE_FOR_INVALID_DTYPE(kOpName, "key/value", std::to_string(static_cast<int>(tilingInfo_.kvDtype)),
                                      "FLOAT8_E4M3FN or HIFLOAT8");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckScaleDtype() const
{
    const auto &opParamInfo = tilingInfo_.opParamInfo;
    if (tilingInfo_.quantModeVal == BSA_QUANT_MODE_FP8) {
        if (CheckInputDtype(opParamInfo.qDescale, "dequantScaleQuery", ge::DT_FLOAT, "FLOAT") != ge::GRAPH_SUCCESS ||
            CheckInputDtype(opParamInfo.kDescale, "keyAntiquantScale", ge::DT_FLOAT, "FLOAT") != ge::GRAPH_SUCCESS ||
            CheckInputDtype(opParamInfo.vDescale, "valueAntiquantScale", ge::DT_FLOAT, "FLOAT") !=
                ge::GRAPH_SUCCESS ||
            CheckInputDtype(opParamInfo.pScale, "quantScale1", ge::DT_FLOAT, "FLOAT") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else if (tilingInfo_.quantModeVal == BSA_QUANT_MODE_MXFP8_FULL_QUANT) {
        // MX 场景下 Q/K/V antiquant scale 与 P quant scale 都是 e8m0。
        // 它们不同于原 FP8 的 FLOAT scale，因为 LoadData 会将其作为 MX 指数 scale 消费。
        if (CheckInputDtype(opParamInfo.qDescale, "dequantScaleQuery", ge::DT_FLOAT8_E8M0, "FLOAT8_E8M0") !=
                ge::GRAPH_SUCCESS ||
            CheckInputDtype(opParamInfo.kDescale, "keyAntiquantScale", ge::DT_FLOAT8_E8M0, "FLOAT8_E8M0") !=
                ge::GRAPH_SUCCESS ||
            CheckInputDtype(opParamInfo.vDescale, "valueAntiquantScale", ge::DT_FLOAT8_E8M0, "FLOAT8_E8M0") !=
                ge::GRAPH_SUCCESS ||
            CheckInputDtype(opParamInfo.pScale, "quantScale1", ge::DT_FLOAT8_E8M0, "FLOAT8_E8M0") !=
                ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE_FOR_INVALID_VALUE(kOpName, "quant_mode", std::to_string(tilingInfo_.quantModeVal), "1 or 2");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckBlockSize() const
{
    // MX 路径将一个 sparse KV block 映射为 S2=512，随后拆成两个 256-token
    // C1/V1 subLoop。Q block size 仍保持原有 128。
    const uint32_t expectedKVBlockSize =
        (tilingInfo_.quantModeVal == BSA_QUANT_MODE_MXFP8_FULL_QUANT) ? BSA_MXFP8_PA_BLOCK_SIZE : BSA_BLOCK_SIZE;
    if (tilingInfo_.qBlockSizeVal != BSA_BLOCK_SIZE || tilingInfo_.kvBlockSizeVal != expectedKVBlockSize) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "q_block_size/kv_block_size",
            std::to_string(tilingInfo_.qBlockSizeVal) + "/" + std::to_string(tilingInfo_.kvBlockSizeVal),
            "must be " + std::to_string(BSA_BLOCK_SIZE) + "/" + std::to_string(expectedKVBlockSize));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckShapeConsistency() const
{
    if (tilingInfo_.bSize == 0U || tilingInfo_.n1Size == 0U || tilingInfo_.n2Size == 0U || tilingInfo_.gSize == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "bSize/n1Size/n2Size/gSize",
            std::to_string(tilingInfo_.bSize) + "/" + std::to_string(tilingInfo_.n1Size) + "/" +
                std::to_string(tilingInfo_.n2Size) + "/" + std::to_string(tilingInfo_.gSize),
            "all dimensions must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.n1Size % tilingInfo_.n2Size != 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "n1Size (query head num) ", std::to_string(tilingInfo_.n1Size),
                                              "must be divisible by n2Size (kv head num) " +
                                                  std::to_string(tilingInfo_.n2Size));
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.s1Size == 0U || tilingInfo_.s2Size == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "s1Size/s2Size", std::to_string(tilingInfo_.s1Size) + "/" + std::to_string(tilingInfo_.s2Size),
            "sequence lengths must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.qbMax == 0U || tilingInfo_.kbMax == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "qbMax/kbMax", std::to_string(tilingInfo_.qbMax) + "/" + std::to_string(tilingInfo_.kbMax),
            "block counts must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.dSize == 0U || tilingInfo_.dSizeV == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "dSize/dSizeV", std::to_string(tilingInfo_.dSize) + "/" + std::to_string(tilingInfo_.dSizeV),
            "head dimensions must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    const auto &opParamInfo = tilingInfo_.opParamInfo;
    const gert::Shape &keyShape = opParamInfo.key.shape->GetStorageShape();
    const gert::Shape &valueShape = opParamInfo.value.shape->GetStorageShape();
    if (keyShape.GetDimNum() == DIM_NUM_4 && valueShape.GetDimNum() == DIM_NUM_4 &&
        !IsShapeEqual(keyShape, valueShape)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            kOpName, "key/value",
            Ops::Base::ToString(keyShape) + "/" + Ops::Base::ToString(valueShape),
            "key and value must have the same shape when both are 4D PA BNBD tensors");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckMaskMode() const
{
    if (tilingInfo_.quantModeVal == BSA_QUANT_MODE_MXFP8_FULL_QUANT &&
        tilingInfo_.maskModeVal != BSA_MASK_MODE_CAUSAL) {
        OP_LOGE_WITH_INVALID_ATTR(kOpName, "mask_mode", std::to_string(tilingInfo_.maskModeVal),
                                  std::to_string(BSA_MASK_MODE_CAUSAL));
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.quantModeVal == BSA_QUANT_MODE_FP8 && tilingInfo_.maskModeVal != BSA_MASK_MODE_NONE &&
        tilingInfo_.maskModeVal != BSA_MASK_MODE_CAUSAL) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "mask_mode", std::to_string(tilingInfo_.maskModeVal),
                                              "must be 0 or 3 in quant_mode=1 FP8 scenario");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.maskModeVal > BSA_MASK_MODE_MAX) {
        OP_LOGE_WITH_INVALID_ATTR(kOpName, "mask_mode", std::to_string(tilingInfo_.maskModeVal), "0, 1, 2, 3 or 4");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckMXFP8FullQuantShape() const
{
    const auto &opParamInfo = tilingInfo_.opParamInfo;
    // quant_mode=2 在访问 shape 前要求所有 scale 输入都存在。
    // 空 tensor 支持的是输出初始化场景，不表示可以省略 MX scale tensor。
    if (CheckMXFP8FullQuantScaleInputs(opParamInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // MX kernel 只处理 PA BNBD K/V。Key 和 Value 的最后一维可以不同（D 与 DV），
    // 但二者都必须是 4D：[paBlockNumSum, kvHeadNum, blockSize, headDim]。
    const gert::Shape &keyShape = opParamInfo.key.shape->GetStorageShape();
    const gert::Shape &valueShape = opParamInfo.value.shape->GetStorageShape();
    if (keyShape.GetDimNum() != DIM_NUM_4 || valueShape.GetDimNum() != DIM_NUM_4) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            kOpName, "key/value",
            std::to_string(keyShape.GetDimNum()) + "D, " + std::to_string(valueShape.GetDimNum()) + "D",
            "must both be 4D PA BNBD [blockNum, kvHeadNum, blockSize, headDim] in quant_mode=2 MXFP8 "
            "full-quant scenario");
        return ge::GRAPH_FAILED;
    }

    const int64_t scaleDSize = static_cast<int64_t>(BSACeilDiv(tilingInfo_.dSize, BSA_MXFP8_SCALE_GROUP_SIZE));
    const int64_t kvTokenNum =
        static_cast<int64_t>(tilingInfo_.paBlockNumSum) * static_cast<int64_t>(tilingInfo_.kvBlockSizeVal);
    // valueAntiquantScale 每 64 个 KV token 分一组，因此首维是 token group 数，
    // 不是原始 KV token 数；这一点与 keyAntiquantScale 不同。
    const int64_t valueScaleTokenGroups =
        (kvTokenNum + static_cast<int64_t>(BSA_MXFP8_SCALE_GROUP_SIZE) - 1) /
        static_cast<int64_t>(BSA_MXFP8_SCALE_GROUP_SIZE);
    // Shape 约束：
    //   P scale：一个静态 e8m0 scale。
    //   Q/K scale：per-token-group，[T, N, D/64, 2]。
    //   V scale：per-channel-group，[T/64, N, DV, 2]。
    const std::array<int64_t, DIM_NUM_1> pScaleShape = {BSA_MXFP8_P_SCALE_SIZE};
    const std::array<int64_t, DIM_NUM_4> queryAntiquantScaleShape = {
        static_cast<int64_t>(tilingInfo_.qTokenNum), static_cast<int64_t>(tilingInfo_.n1Size), scaleDSize,
        BSA_MXFP8_SCALE_LAST_DIM};
    const std::array<int64_t, DIM_NUM_4> keyShapeExpect = {static_cast<int64_t>(tilingInfo_.paBlockNumSum),
                                                           static_cast<int64_t>(tilingInfo_.n2Size),
                                                           static_cast<int64_t>(tilingInfo_.kvBlockSizeVal),
                                                           static_cast<int64_t>(tilingInfo_.dSize)};
    const std::array<int64_t, DIM_NUM_4> valueShapeExpect = {static_cast<int64_t>(tilingInfo_.paBlockNumSum),
                                                             static_cast<int64_t>(tilingInfo_.n2Size),
                                                             static_cast<int64_t>(tilingInfo_.kvBlockSizeVal),
                                                             static_cast<int64_t>(tilingInfo_.dSizeV)};
    const std::array<int64_t, DIM_NUM_4> keyAntiquantScaleShape = {
        kvTokenNum, static_cast<int64_t>(tilingInfo_.n2Size), scaleDSize, BSA_MXFP8_SCALE_LAST_DIM};
    const std::array<int64_t, DIM_NUM_4> valueAntiquantScaleShape = {
        valueScaleTokenGroups, static_cast<int64_t>(tilingInfo_.n2Size), static_cast<int64_t>(tilingInfo_.dSizeV),
        BSA_MXFP8_SCALE_LAST_DIM};

    if (CheckInputShape(opParamInfo.pScale, "quantScale1", pScaleShape) != ge::GRAPH_SUCCESS ||
        CheckInputShape(opParamInfo.qDescale, "dequantScaleQuery", queryAntiquantScaleShape) != ge::GRAPH_SUCCESS ||
        CheckInputShape(opParamInfo.key, "key", keyShapeExpect) != ge::GRAPH_SUCCESS ||
        CheckInputShape(opParamInfo.value, "value", valueShapeExpect) != ge::GRAPH_SUCCESS ||
        CheckInputShape(opParamInfo.kDescale, "keyAntiquantScale", keyAntiquantScaleShape) != ge::GRAPH_SUCCESS ||
        CheckInputShape(opParamInfo.vDescale, "valueAntiquantScale", valueAntiquantScaleShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::Process()
{
    if (CheckAttrs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckScaleDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckMaskMode() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.quantModeVal == BSA_QUANT_MODE_MXFP8_FULL_QUANT &&
        CheckMXFP8FullQuantShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
