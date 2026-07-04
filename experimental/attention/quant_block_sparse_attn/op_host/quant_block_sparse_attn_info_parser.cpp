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
 * \file quant_block_sparse_attn_info_parser.cpp
 * \brief QuantBlockSparseAttn parameter parsing implementation.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "quant_block_sparse_attn_info_parser.h"
#include "quant_block_sparse_attn_tiling.h"
#include "log/log.h"

namespace optiling {
namespace {
constexpr const char *kOpName = "QuantBlockSparseAttn";
constexpr size_t DIM_NUM_1 = 1U;
constexpr size_t DIM_NUM_2 = 2U;
constexpr size_t DIM_NUM_3 = 3U;
constexpr size_t DIM_NUM_4 = 4U;
constexpr size_t DIM_NUM_5 = 5U;
constexpr size_t DIM_B = 0U;
constexpr size_t DIM_S = 1U;
constexpr size_t DIM_N = 2U;
constexpr size_t DIM_D = 3U;
constexpr size_t DIM_TND_T = 0U;
constexpr size_t DIM_TND_N = 1U;
constexpr size_t DIM_TND_D = 2U;
constexpr size_t DIM_NTD_N = 0U;
constexpr size_t DIM_NTD_T = 1U;
constexpr size_t DIM_NTD_D = 2U;
constexpr size_t DIM_PA_N = 1U;
constexpr size_t DIM_PA_BLOCK_SIZE = 2U;
constexpr size_t DIM_BLOCK_TABLE_MAX = 1U;
constexpr size_t DIM_SPARSE_N1 = 1U;
constexpr size_t DIM_SPARSE_QB = 2U;
constexpr size_t DIM_SPARSE_COUNT = 3U;
constexpr size_t DIM_KV_HEAD_DIM = 3U;
constexpr size_t DIM_V_DESCALE_N2 = 0U;
constexpr uint32_t BSA_LAYOUT_Q_BSND_VALUE = 0U;
constexpr uint32_t BSA_LAYOUT_Q_TND_VALUE = 2U;
constexpr uint32_t BSA_LAYOUT_Q_NTD_VALUE = 5U;
constexpr int64_t BSA_MXFP8_P_SCALE_SIZE = 1;

bool IsInputShapeReady(const BSARequiredParaInfo &input)
{
    return input.shape != nullptr;
}

void AppendMissingInput(const BSARequiredParaInfo &input, const char *inputName, std::string &missingInputs)
{
    if (IsInputShapeReady(input)) {
        return;
    }
    if (!missingInputs.empty()) {
        missingInputs += ", ";
    }
    missingInputs += inputName;
}

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
    std::string missingInputs;
    AppendMissingInput(opParamInfo.pScale, "quantScale1", missingInputs);
    AppendMissingInput(opParamInfo.kDescale, "keyAntiquantScale", missingInputs);
    AppendMissingInput(opParamInfo.vDescale, "valueAntiquantScale", missingInputs);
    AppendMissingInput(opParamInfo.qDescale, "dequantScaleQuery", missingInputs);
    if (!missingInputs.empty()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "quantScale1/keyAntiquantScale/valueAntiquantScale/dequantScaleQuery", missingInputs,
            "all must be passed in quant_mode=2 MXFP8 full-quant scenario");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckCommonAttrs(const QuantBlockSparseAttnTilingInfo &tilingInfo, const gert::RuntimeAttrs *attrs)
{
    const std::string layoutKV = BSAGetStringAttr(attrs, BSA_LAYOUT_KV_ATTR_INDEX, "PA_BNSD");
    const std::string layoutSparseIndices = BSAGetStringAttr(attrs, BSA_LAYOUT_SPARSE_INDICES_ATTR_INDEX, "B_N_Qb_Kb");
    if (layoutKV != "PA_BNSD" || layoutSparseIndices != "B_N_Qb_Kb" ||
        (tilingInfo.quantModeVal != BSA_QUANT_MODE_FP8 &&
         tilingInfo.quantModeVal != BSA_QUANT_MODE_MXFP8_FULL_QUANT)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "layout_kv/layout_sparse_indices/quant_mode",
            layoutKV + "/" + layoutSparseIndices + "/" + std::to_string(tilingInfo.quantModeVal),
            "layout_kv must be PA_BNSD, layout_sparse_indices must be B_N_Qb_Kb, quant_mode must be 1 or 2");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckFP8QuantModeAttrs(const QuantBlockSparseAttnTilingInfo &tilingInfo)
{
    if (tilingInfo.maskModeVal != BSA_MASK_MODE_NONE && tilingInfo.maskModeVal != BSA_MASK_MODE_CAUSAL) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "mask_mode", std::to_string(tilingInfo.maskModeVal),
                                              "must be 0 or 3 in quant_mode=1 FP8 scenario");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo.qBlockSizeVal != BSA_BLOCK_SIZE || tilingInfo.kvBlockSizeVal != BSA_BLOCK_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "q_block_size/kv_block_size",
            std::to_string(tilingInfo.qBlockSizeVal) + "/" + std::to_string(tilingInfo.kvBlockSizeVal),
            "must both be " + std::to_string(BSA_BLOCK_SIZE) + " in quant_mode=1 FP8 scenario");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckMXFP8FullQuantAttrs(const QuantBlockSparseAttnTilingInfo &tilingInfo,
                                         const gert::RuntimeAttrs *attrs)
{
    const std::string layoutQ = BSAGetStringAttr(attrs, BSA_LAYOUT_Q_ATTR_INDEX, "TND");
    const std::string layoutOut = BSAGetStringAttr(attrs, BSA_LAYOUT_OUT_ATTR_INDEX, "TND");
    if (layoutQ != "TND" || layoutOut != "TND") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "layout_q/layout_out", layoutQ + "/" + layoutOut,
            "must both be TND in quant_mode=2 MXFP8 full-quant scenario");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo.maskModeVal != BSA_MASK_MODE_CAUSAL) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "mask_mode", std::to_string(tilingInfo.maskModeVal),
                                              "must be 3 in quant_mode=2 MXFP8 full-quant scenario");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo.qBlockSizeVal != BSA_BLOCK_SIZE || tilingInfo.kvBlockSizeVal != BSA_MXFP8_PA_BLOCK_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "q_block_size/kv_block_size",
            std::to_string(tilingInfo.qBlockSizeVal) + "/" + std::to_string(tilingInfo.kvBlockSizeVal),
            "must be " + std::to_string(BSA_BLOCK_SIZE) + "/" + std::to_string(BSA_MXFP8_PA_BLOCK_SIZE) +
                " in quant_mode=2 MXFP8 full-quant PA scenario");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(kOpName, "quant_mode=2 maps to queryQuantMode/keyAntiquantMode/valueAntiquantMode=%u/%u/%u",
            BSA_MXFP8_PER_TOKEN_GROUP_MODE, BSA_MXFP8_PER_TOKEN_GROUP_MODE, BSA_MXFP8_PER_CHANNEL_GROUP_MODE);
    return CheckMXFP8FullQuantScaleInputs(tilingInfo.opParamInfo);
}

ge::graphStatus CheckQuantModeAttrs(const QuantBlockSparseAttnTilingInfo &tilingInfo, const gert::RuntimeAttrs *attrs)
{
    if (CheckCommonAttrs(tilingInfo, attrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo.quantModeVal == BSA_QUANT_MODE_FP8) {
        return CheckFP8QuantModeAttrs(tilingInfo);
    } else if (tilingInfo.quantModeVal == BSA_QUANT_MODE_MXFP8_FULL_QUANT) {
        return CheckMXFP8FullQuantAttrs(tilingInfo, attrs);
    } else {
        OP_LOGE_FOR_INVALID_VALUE(kOpName, "quant_mode", std::to_string(tilingInfo.quantModeVal), "1 or 2");
        return ge::GRAPH_FAILED;
    }
}

ge::graphStatus CheckFP8ScaleDtype(const QuantBlockSparseAttnParaInfo &opParamInfo)
{
    if (CheckInputDtype(opParamInfo.qDescale, "dequantScaleQuery", ge::DT_FLOAT, "FLOAT") != ge::GRAPH_SUCCESS ||
        CheckInputDtype(opParamInfo.kDescale, "keyAntiquantScale", ge::DT_FLOAT, "FLOAT") != ge::GRAPH_SUCCESS ||
        CheckInputDtype(opParamInfo.vDescale, "valueAntiquantScale", ge::DT_FLOAT, "FLOAT") != ge::GRAPH_SUCCESS ||
        CheckInputDtype(opParamInfo.pScale, "quantScale1", ge::DT_FLOAT, "FLOAT") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckMXFP8FullQuantScaleDtype(const QuantBlockSparseAttnParaInfo &opParamInfo)
{
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
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckQuantModeScaleDtype(const QuantBlockSparseAttnTilingInfo &tilingInfo)
{
    if (tilingInfo.quantModeVal == BSA_QUANT_MODE_FP8) {
        return CheckFP8ScaleDtype(tilingInfo.opParamInfo);
    } else if (tilingInfo.quantModeVal == BSA_QUANT_MODE_MXFP8_FULL_QUANT) {
        return CheckMXFP8FullQuantScaleDtype(tilingInfo.opParamInfo);
    } else {
        OP_LOGE_FOR_INVALID_VALUE(kOpName, "quant_mode", std::to_string(tilingInfo.quantModeVal), "1 or 2");
        return ge::GRAPH_FAILED;
    }
}

ge::graphStatus CheckMXFP8FullQuantKVLayout(const gert::Shape &keyShape, const gert::Shape &valueShape)
{
    if (keyShape.GetDimNum() != DIM_NUM_4 || valueShape.GetDimNum() != DIM_NUM_4) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            kOpName, "key/value",
            std::to_string(keyShape.GetDimNum()) + "D, " + std::to_string(valueShape.GetDimNum()) + "D",
            "must both be 4D PA BNBD [blockNum, kvHeadNum, blockSize, headDim] in quant_mode=2 MXFP8 "
            "full-quant scenario");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckMXFP8FullQuantShape(const QuantBlockSparseAttnTilingInfo &tilingInfo)
{
    const auto &opParamInfo = tilingInfo.opParamInfo;
    const int64_t scaleDSize = static_cast<int64_t>(BSACeilDiv(tilingInfo.dSize, BSA_MXFP8_SCALE_GROUP_SIZE));
    const int64_t valueScaleBlockSize =
        static_cast<int64_t>(BSACeilDiv(tilingInfo.kvBlockSizeVal, BSA_MXFP8_SCALE_GROUP_SIZE));

    const std::array<int64_t, DIM_NUM_1> pScaleShape = {BSA_MXFP8_P_SCALE_SIZE};
    const std::array<int64_t, DIM_NUM_4> qScaleShape = {static_cast<int64_t>(tilingInfo.qTokenNum),
                                                        static_cast<int64_t>(tilingInfo.n1Size), scaleDSize,
                                                        BSA_MXFP8_SCALE_LAST_DIM};
    const std::array<int64_t, DIM_NUM_4> kvShape = {static_cast<int64_t>(tilingInfo.paBlockNumSum),
                                                    static_cast<int64_t>(tilingInfo.n2Size),
                                                    static_cast<int64_t>(tilingInfo.kvBlockSizeVal),
                                                    static_cast<int64_t>(tilingInfo.dSizeV)};
    const std::array<int64_t, DIM_NUM_5> kScaleShape = {static_cast<int64_t>(tilingInfo.paBlockNumSum),
                                                        static_cast<int64_t>(tilingInfo.n2Size),
                                                        static_cast<int64_t>(tilingInfo.kvBlockSizeVal), scaleDSize,
                                                        BSA_MXFP8_SCALE_LAST_DIM};
    const std::array<int64_t, DIM_NUM_5> vScaleShape = {static_cast<int64_t>(tilingInfo.paBlockNumSum),
                                                        static_cast<int64_t>(tilingInfo.n2Size), valueScaleBlockSize,
                                                        static_cast<int64_t>(tilingInfo.dSizeV),
                                                        BSA_MXFP8_SCALE_LAST_DIM};

    if (CheckInputShape(opParamInfo.pScale, "quantScale1", pScaleShape) != ge::GRAPH_SUCCESS ||
        CheckInputShape(opParamInfo.qDescale, "dequantScaleQuery", qScaleShape) != ge::GRAPH_SUCCESS ||
        CheckInputShape(opParamInfo.key, "key", kvShape) != ge::GRAPH_SUCCESS ||
        CheckInputShape(opParamInfo.value, "value", kvShape) != ge::GRAPH_SUCCESS ||
        CheckInputShape(opParamInfo.kDescale, "keyAntiquantScale", kScaleShape) != ge::GRAPH_SUCCESS ||
        CheckInputShape(opParamInfo.vDescale, "valueAntiquantScale", vScaleShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

QuantBlockSparseAttnInfoParser::QuantBlockSparseAttnInfoParser(gert::TilingContext *context) : context_(context)
{
}

ge::graphStatus QuantBlockSparseAttnInfoParser::ParseQuery(QuantBlockSparseAttnTilingInfo &tilingInfo,
                                                           const gert::Shape &queryShape,
                                                           const gert::Shape &sparseIndicesShape,
                                                           const gert::RuntimeAttrs *attrs)
{
    const size_t queryDimNum = queryShape.GetDimNum();
    const std::string layoutQ = BSAGetStringAttr(attrs, BSA_LAYOUT_Q_ATTR_INDEX, "TND");
    if (queryDimNum == DIM_NUM_4 && layoutQ == "BSND") {
        tilingInfo.isBSND = true;
        tilingInfo.layoutQValue = BSA_LAYOUT_Q_BSND_VALUE;
        if (!BSAGetDimAsU32(queryShape, DIM_B, tilingInfo.bSize) ||
            !BSAGetDimAsU32(queryShape, DIM_S, tilingInfo.s1Size) ||
            !BSAGetDimAsU32(queryShape, DIM_N, tilingInfo.n1Size) ||
            !BSAGetDimAsU32(queryShape, DIM_D, tilingInfo.dSize)) {
            OP_LOGE(kOpName, "ParseQuery BSND: failed to get query dims, dimNum=%zu", queryDimNum);
            return ge::GRAPH_FAILED;
        }
        tilingInfo.qTokenNum = tilingInfo.bSize * tilingInfo.s1Size;
    } else if (queryDimNum == DIM_NUM_3 && layoutQ == "TND") {
        tilingInfo.isBSND = false;
        tilingInfo.layoutQValue = BSA_LAYOUT_Q_TND_VALUE;
        if (!BSAGetDimAsU32(sparseIndicesShape, DIM_B, tilingInfo.bSize) ||
            !BSAGetDimAsU32(queryShape, DIM_TND_T, tilingInfo.s1Size) ||
            !BSAGetDimAsU32(queryShape, DIM_TND_N, tilingInfo.n1Size) ||
            !BSAGetDimAsU32(queryShape, DIM_TND_D, tilingInfo.dSize)) {
            OP_LOGE(kOpName, "ParseQuery TND: failed to get query/sparseIndices dims, dimNum=%zu", queryDimNum);
            return ge::GRAPH_FAILED;
        }
        tilingInfo.qTokenNum = tilingInfo.s1Size;
        tilingInfo.s1Size = BSAGetPositiveAttr(attrs, BSA_MAX_SEQLEN_Q_ATTR_INDEX, tilingInfo.s1Size);
    } else if (queryDimNum == DIM_NUM_3 && layoutQ == "NTD") {
        tilingInfo.isBSND = false;
        tilingInfo.layoutQValue = BSA_LAYOUT_Q_NTD_VALUE;
        if (!BSAGetDimAsU32(sparseIndicesShape, DIM_B, tilingInfo.bSize) ||
            !BSAGetDimAsU32(queryShape, DIM_NTD_N, tilingInfo.n1Size) ||
            !BSAGetDimAsU32(queryShape, DIM_NTD_T, tilingInfo.s1Size) ||
            !BSAGetDimAsU32(queryShape, DIM_NTD_D, tilingInfo.dSize)) {
            OP_LOGE(kOpName, "ParseQuery NTD: failed to get query/sparseIndices dims, dimNum=%zu", queryDimNum);
            return ge::GRAPH_FAILED;
        }
        tilingInfo.qTokenNum = tilingInfo.s1Size;
        tilingInfo.s1Size = BSAGetPositiveAttr(attrs, BSA_MAX_SEQLEN_Q_ATTR_INDEX, tilingInfo.s1Size);
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "query",
                                                 std::to_string(queryDimNum) + "D with layout " + layoutQ,
                                                 "4D with layout BSND, 3D with layout TND or 3D with layout NTD");
        return ge::GRAPH_FAILED;
    }

    if (tilingInfo.dSize != BSA_D_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE(kOpName, "query head_dim (dSize)", std::to_string(tilingInfo.dSize),
                                  std::to_string(BSA_D_SIZE));
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnInfoParser::ParseKeyValue(QuantBlockSparseAttnTilingInfo &tilingInfo,
                                                              const gert::Shape &keyShape,
                                                              const gert::Shape &vDescaleShape,
                                                              const gert::RuntimeAttrs *attrs)
{
    if (keyShape.GetDimNum() == 1U) {
        if (tilingInfo.paBlockStrideVal == 0U) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "pa_block_stride", "0",
                                                  "must be greater than 0 for 1D combined KV storage");
            return ge::GRAPH_FAILED;
        }
        if (!BSAGetDimAsU32(vDescaleShape, DIM_V_DESCALE_N2, tilingInfo.n2Size) || tilingInfo.n2Size == 0U) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "v_descale", Ops::Base::ToString(vDescaleShape),
                                                  "v_descale must provide kvHeadNum in dim 0");
            return ge::GRAPH_FAILED;
        }
        const int64_t keyStorageSize = keyShape.GetShapeSize();
        if (keyStorageSize <= 0 || static_cast<uint64_t>(keyStorageSize) % tilingInfo.paBlockStrideVal != 0U) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                kOpName, "key", Ops::Base::ToString(keyShape),
                "1D combined KV storage size must be a positive multiple of pa_block_stride=" +
                    std::to_string(tilingInfo.paBlockStrideVal));
            return ge::GRAPH_FAILED;
        }

        if (tilingInfo.n1Size % tilingInfo.n2Size != 0U) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "n1Size (query head num)", std::to_string(tilingInfo.n1Size),
                                                  "must be divisible by n2Size (kv head num) " +
                                                      std::to_string(tilingInfo.n2Size));
            return ge::GRAPH_FAILED;
        }

        tilingInfo.gSize = tilingInfo.n1Size / tilingInfo.n2Size;
        tilingInfo.isGqa = (tilingInfo.gSize > 1U);
        tilingInfo.s2Size = BSAGetPositiveAttr(attrs, BSA_MAX_SEQLEN_KV_ATTR_INDEX, tilingInfo.kvBlockSizeVal);
        tilingInfo.kbMax = BSACeilDiv(tilingInfo.s2Size, tilingInfo.kvBlockSizeVal);
        tilingInfo.qbMax = BSACeilDiv(tilingInfo.s1Size, tilingInfo.qBlockSizeVal);
        tilingInfo.dSizeV = BSA_D_SIZE;
        tilingInfo.paBlockNumSum = static_cast<uint32_t>(keyStorageSize / tilingInfo.paBlockStrideVal);
        return ge::GRAPH_SUCCESS;
    }

    uint32_t physicalKvBlockSize = 0;
    uint32_t kvDSize = 0;
    if (keyShape.GetDimNum() != DIM_NUM_4 || !BSAGetDimAsU32(keyShape, DIM_PA_BLOCK_SIZE, physicalKvBlockSize) ||
        !BSAGetDimAsU32(keyShape, DIM_PA_N, tilingInfo.n2Size) || !BSAGetDimAsU32(keyShape, DIM_KV_HEAD_DIM, kvDSize) ||
        physicalKvBlockSize != tilingInfo.kvBlockSizeVal || kvDSize != BSA_D_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            kOpName, "key", Ops::Base::ToString(keyShape),
            "key must be 4D [blockNum, kvHeadNum, blockSize, headDim] with blockSize=" +
                std::to_string(tilingInfo.kvBlockSizeVal) + " and headDim=" + std::to_string(BSA_D_SIZE));
        return ge::GRAPH_FAILED;
    }

    if (tilingInfo.n2Size == 0U || tilingInfo.n1Size % tilingInfo.n2Size != 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "n1Size (query head num)", std::to_string(tilingInfo.n1Size),
                                              "must be divisible by n2Size (kv head num) " +
                                                  std::to_string(tilingInfo.n2Size));
        return ge::GRAPH_FAILED;
    }

    tilingInfo.gSize = tilingInfo.n1Size / tilingInfo.n2Size;
    tilingInfo.isGqa = (tilingInfo.gSize > 1U);

    tilingInfo.s2Size = BSAGetPositiveAttr(attrs, BSA_MAX_SEQLEN_KV_ATTR_INDEX, tilingInfo.kvBlockSizeVal);
    tilingInfo.kbMax = BSACeilDiv(tilingInfo.s2Size, tilingInfo.kvBlockSizeVal);
    tilingInfo.qbMax = BSACeilDiv(tilingInfo.s1Size, tilingInfo.qBlockSizeVal);

    tilingInfo.dSizeV = BSA_D_SIZE;

    if (!BSAGetDimAsU32(keyShape, DIM_B, tilingInfo.paBlockNumSum)) {
        OP_LOGE(kOpName, "ParseKeyValue: failed to get paBlockNumSum from key dim 0");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnInfoParser::ParseSparseIndices(QuantBlockSparseAttnTilingInfo &tilingInfo,
                                                                   const gert::Shape &sparseIndicesShape,
                                                                   const gert::Shape &sparseSeqLenShape)
{
    if (sparseIndicesShape.GetDimNum() != DIM_NUM_4 || sparseSeqLenShape.GetDimNum() != DIM_NUM_3) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            kOpName, "sparse_indices, sparse_seq_len",
            std::to_string(sparseIndicesShape.GetDimNum()) + "D, " + std::to_string(sparseSeqLenShape.GetDimNum()) +
                "D",
            "sparse_indices must be 4D [B, N1, Qb, Count], sparse_seq_len must be 3D [B, N1, Qb]");
        return ge::GRAPH_FAILED;
    }

    uint32_t sparseB = 0;
    uint32_t sparseN1 = 0;
    uint32_t sparseQb = 0;
    if (!BSAGetDimAsU32(sparseIndicesShape, DIM_B, sparseB) ||
        !BSAGetDimAsU32(sparseIndicesShape, DIM_SPARSE_N1, sparseN1) ||
        !BSAGetDimAsU32(sparseIndicesShape, DIM_SPARSE_QB, sparseQb) ||
        !BSAGetDimAsU32(sparseIndicesShape, DIM_SPARSE_COUNT, tilingInfo.sparseCount)) {
        OP_LOGE(kOpName, "ParseSparseIndices: failed to get dims from sparseIndices");
        return ge::GRAPH_FAILED;
    }
    if (sparseB != tilingInfo.bSize || sparseN1 != tilingInfo.n1Size || sparseQb != tilingInfo.qbMax) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            kOpName, "sparse_indices", Ops::Base::ToString(sparseIndicesShape),
            "dims [B=" + std::to_string(sparseB) + ", N1=" + std::to_string(sparseN1) + ", Qb=" +
                std::to_string(sparseQb) + "] must match query-derived values [B=" + std::to_string(tilingInfo.bSize) +
                ", N1=" + std::to_string(tilingInfo.n1Size) + ", Qb=" + std::to_string(tilingInfo.qbMax) + "]");
        return ge::GRAPH_FAILED;
    }

    uint32_t seqLenB = 0;
    uint32_t seqLenN1 = 0;
    uint32_t seqLenQb = 0;
    if (!BSAGetDimAsU32(sparseSeqLenShape, DIM_B, seqLenB) ||
        !BSAGetDimAsU32(sparseSeqLenShape, DIM_SPARSE_N1, seqLenN1) ||
        !BSAGetDimAsU32(sparseSeqLenShape, DIM_SPARSE_QB, seqLenQb)) {
        OP_LOGE(kOpName, "ParseSparseIndices: failed to get dims from sparseSeqLen");
        return ge::GRAPH_FAILED;
    }
    if (seqLenB != tilingInfo.bSize || seqLenN1 != tilingInfo.n1Size || seqLenQb != tilingInfo.qbMax) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            kOpName, "sparse_seq_len", Ops::Base::ToString(sparseSeqLenShape),
            "dims [B=" + std::to_string(seqLenB) + ", N1=" + std::to_string(seqLenN1) + ", Qb=" +
                std::to_string(seqLenQb) + "] must match query-derived values [B=" + std::to_string(tilingInfo.bSize) +
                ", N1=" + std::to_string(tilingInfo.n1Size) + ", Qb=" + std::to_string(tilingInfo.qbMax) + "]");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnInfoParser::ParseOptionalInputs(QuantBlockSparseAttnTilingInfo &tilingInfo)
{
    auto &opParamInfo = tilingInfo.opParamInfo;

    opParamInfo.cuSeqlensQ.desc = context_->GetInputDesc(BSA_CU_SEQLENS_Q_INDEX);
    const gert::StorageShape *cuSeqlensQShape = context_->GetOptionalInputShape(BSA_CU_SEQLENS_Q_INDEX);
    opParamInfo.cuSeqlensQ.tensor =
        (cuSeqlensQShape != nullptr && cuSeqlensQShape->GetStorageShape().GetShapeSize() > 0) ?
            reinterpret_cast<const gert::Tensor *>(cuSeqlensQShape) :
            nullptr;

    opParamInfo.cuSeqlensKV.desc = context_->GetInputDesc(BSA_CU_SEQLENS_KV_INDEX);
    const gert::StorageShape *cuSeqlensKVShape = context_->GetOptionalInputShape(BSA_CU_SEQLENS_KV_INDEX);
    opParamInfo.cuSeqlensKV.tensor =
        (cuSeqlensKVShape != nullptr && cuSeqlensKVShape->GetStorageShape().GetShapeSize() > 0) ?
            reinterpret_cast<const gert::Tensor *>(cuSeqlensKVShape) :
            nullptr;

    opParamInfo.seqUsedQ.desc = context_->GetInputDesc(BSA_SEQUSED_Q_INDEX);
    const gert::StorageShape *seqUsedQShape = context_->GetOptionalInputShape(BSA_SEQUSED_Q_INDEX);
    opParamInfo.seqUsedQ.tensor = (seqUsedQShape != nullptr && seqUsedQShape->GetStorageShape().GetShapeSize() > 0) ?
                                      reinterpret_cast<const gert::Tensor *>(seqUsedQShape) :
                                      nullptr;

    opParamInfo.seqUsedKV.desc = context_->GetInputDesc(BSA_SEQUSED_KV_INDEX);
    const gert::StorageShape *seqUsedKVShape = context_->GetOptionalInputShape(BSA_SEQUSED_KV_INDEX);
    opParamInfo.seqUsedKV.tensor = (seqUsedKVShape != nullptr && seqUsedKVShape->GetStorageShape().GetShapeSize() > 0) ?
                                       reinterpret_cast<const gert::Tensor *>(seqUsedKVShape) :
                                       nullptr;

    opParamInfo.blockTable.desc = context_->GetInputDesc(BSA_BLOCK_TABLE_INDEX);
    const gert::StorageShape *blockTableStorageShape = context_->GetOptionalInputShape(BSA_BLOCK_TABLE_INDEX);
    opParamInfo.blockTable.tensor =
        (blockTableStorageShape != nullptr && blockTableStorageShape->GetStorageShape().GetShapeSize() > 0) ?
            reinterpret_cast<const gert::Tensor *>(blockTableStorageShape) :
            nullptr;

    opParamInfo.metadata.desc = context_->GetInputDesc(BSA_METADATA_INDEX);
    const gert::StorageShape *metadataShape = context_->GetOptionalInputShape(BSA_METADATA_INDEX);
    opParamInfo.metadata.tensor = (metadataShape != nullptr && metadataShape->GetStorageShape().GetShapeSize() > 0) ?
                                      reinterpret_cast<const gert::Tensor *>(metadataShape) :
                                      nullptr;

    tilingInfo.maxBlockNumPerBatch = tilingInfo.kbMax;
    if (blockTableStorageShape != nullptr && blockTableStorageShape->GetStorageShape().GetShapeSize() > 0) {
        const gert::Shape &blockTableShape = blockTableStorageShape->GetStorageShape();
        uint32_t blockTableB = 0;
        if (blockTableShape.GetDimNum() != DIM_NUM_2 || !BSAGetDimAsU32(blockTableShape, DIM_B, blockTableB) ||
            !BSAGetDimAsU32(blockTableShape, DIM_BLOCK_TABLE_MAX, tilingInfo.maxBlockNumPerBatch) ||
            blockTableB != tilingInfo.bSize) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                kOpName, "block_table", std::to_string(blockTableShape.GetDimNum()) + "D",
                "2D [B=" + std::to_string(tilingInfo.bSize) + ", maxBlockNumPerBatch]");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnInfoParser::ParseAttributes(QuantBlockSparseAttnTilingInfo &tilingInfo,
                                                                const gert::RuntimeAttrs *attrs)
{
    auto &opParamInfo = tilingInfo.opParamInfo;

    opParamInfo.qBlockSize = attrs->GetAttrPointer<int64_t>(BSA_SPARSE_Q_BLOCK_SIZE_ATTR_INDEX);
    opParamInfo.kvBlockSize = attrs->GetAttrPointer<int64_t>(BSA_SPARSE_KV_BLOCK_SIZE_ATTR_INDEX);
    opParamInfo.paBlockStride = attrs->GetAttrPointer<int64_t>(BSA_PA_BLOCK_STRIDE_ATTR_INDEX);
    tilingInfo.qBlockSizeVal = BSAGetPositiveAttr(attrs, BSA_SPARSE_Q_BLOCK_SIZE_ATTR_INDEX, BSA_BLOCK_SIZE);
    tilingInfo.kvBlockSizeVal = BSAGetPositiveAttr(attrs, BSA_SPARSE_KV_BLOCK_SIZE_ATTR_INDEX, BSA_BLOCK_SIZE);
    tilingInfo.paBlockStrideVal = BSAGetUintAttr(attrs, BSA_PA_BLOCK_STRIDE_ATTR_INDEX, 0U);

    opParamInfo.softmaxScale = attrs->GetAttrPointer<float>(BSA_SOFTMAX_SCALE_ATTR_INDEX);
    opParamInfo.maskMode = attrs->GetAttrPointer<int64_t>(BSA_MASK_MODE_ATTR_INDEX);
    opParamInfo.returnSoftmaxLse = attrs->GetAttrPointer<bool>(BSA_RETURN_SOFTMAX_LSE_ATTR_INDEX);
    opParamInfo.maxSeqlenQ = attrs->GetAttrPointer<int64_t>(BSA_MAX_SEQLEN_Q_ATTR_INDEX);
    opParamInfo.maxSeqlenKV = attrs->GetAttrPointer<int64_t>(BSA_MAX_SEQLEN_KV_ATTR_INDEX);
    opParamInfo.quantMode = attrs->GetAttrPointer<int64_t>(BSA_QUANT_MODE_ATTR_INDEX);

    tilingInfo.softmaxScaleVal = BSAGetFloatAttr(attrs, BSA_SOFTMAX_SCALE_ATTR_INDEX, 1.0F);
    tilingInfo.maskModeVal = BSAGetUintAttr(attrs, BSA_MASK_MODE_ATTR_INDEX, 0U);
    tilingInfo.quantModeVal = BSAGetUintAttr(attrs, BSA_QUANT_MODE_ATTR_INDEX, BSA_QUANT_MODE_FP8);
    tilingInfo.returnSoftmaxLseVal = BSAGetBoolAttr(attrs, BSA_RETURN_SOFTMAX_LSE_ATTR_INDEX, false);

    opParamInfo.query.desc = context_->GetInputDesc(BSA_QUERY_INDEX);
    tilingInfo.qDtype =
        (opParamInfo.query.desc != nullptr) ? opParamInfo.query.desc->GetDataType() : ge::DT_FLOAT8_E4M3FN;
    opParamInfo.key.desc = context_->GetInputDesc(BSA_KEY_INDEX);
    tilingInfo.kvDtype = (opParamInfo.key.desc != nullptr) ? opParamInfo.key.desc->GetDataType() : ge::DT_FLOAT8_E4M3FN;
    opParamInfo.value.desc = context_->GetInputDesc(BSA_VALUE_INDEX);
    tilingInfo.vDtype =
        (opParamInfo.value.desc != nullptr) ? opParamInfo.value.desc->GetDataType() : ge::DT_FLOAT8_E4M3FN;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnInfoParser::Parse(QuantBlockSparseAttnTilingInfo &tilingInfo)
{
    if (context_ == nullptr) {
        OP_LOGE(kOpName, "Parse: tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(kOpName, "Parse: attrs is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto &opParamInfo = tilingInfo.opParamInfo;

    opParamInfo.query.shape = context_->GetInputShape(BSA_QUERY_INDEX);
    opParamInfo.key.shape = context_->GetInputShape(BSA_KEY_INDEX);
    opParamInfo.value.shape = context_->GetInputShape(BSA_VALUE_INDEX);
    opParamInfo.qDescale.desc = context_->GetInputDesc(BSA_Q_DESCALE_INDEX);
    opParamInfo.qDescale.shape = context_->GetInputShape(BSA_Q_DESCALE_INDEX);
    opParamInfo.kDescale.desc = context_->GetInputDesc(BSA_K_DESCALE_INDEX);
    opParamInfo.kDescale.shape = context_->GetInputShape(BSA_K_DESCALE_INDEX);
    opParamInfo.vDescale.desc = context_->GetInputDesc(BSA_V_DESCALE_INDEX);
    opParamInfo.vDescale.shape = context_->GetInputShape(BSA_V_DESCALE_INDEX);
    opParamInfo.pScale.desc = context_->GetInputDesc(BSA_P_SCALE_INDEX);
    opParamInfo.pScale.shape = context_->GetInputShape(BSA_P_SCALE_INDEX);
    opParamInfo.sparseIndices.desc = context_->GetInputDesc(BSA_SPARSE_INDICES_INDEX);
    opParamInfo.sparseIndices.shape = context_->GetInputShape(BSA_SPARSE_INDICES_INDEX);
    opParamInfo.sparseSeqLen.desc = context_->GetInputDesc(BSA_SPARSE_SEQ_LEN_INDEX);
    opParamInfo.sparseSeqLen.shape = context_->GetInputShape(BSA_SPARSE_SEQ_LEN_INDEX);
    opParamInfo.attenMask.desc = context_->GetInputDesc(BSA_ATTEN_MASK_INDEX);
    opParamInfo.attenMask.shape = context_->GetInputShape(BSA_ATTEN_MASK_INDEX);
    opParamInfo.attnOut.desc = context_->GetOutputDesc(BSA_ATTENTION_OUT_INDEX);
    opParamInfo.attnOut.shape = context_->GetOutputShape(BSA_ATTENTION_OUT_INDEX);
    opParamInfo.lseOut.desc = context_->GetOutputDesc(BSA_SOFTMAX_LSE_INDEX);
    opParamInfo.lseOut.shape = context_->GetOutputShape(BSA_SOFTMAX_LSE_INDEX);

    if (opParamInfo.query.shape == nullptr || opParamInfo.key.shape == nullptr || opParamInfo.value.shape == nullptr ||
        opParamInfo.qDescale.shape == nullptr || opParamInfo.kDescale.shape == nullptr ||
        opParamInfo.vDescale.shape == nullptr || opParamInfo.pScale.shape == nullptr ||
        opParamInfo.sparseIndices.shape == nullptr || opParamInfo.sparseSeqLen.shape == nullptr ||
        opParamInfo.attenMask.shape == nullptr) {
        OP_LOGE(kOpName,
                "Parse: required input shape is nullptr (query=%p, key=%p, value=%p, qDescale=%p, kDescale=%p, "
                "vDescale=%p, pScale=%p, sparseIndices=%p, sparseSeqLen=%p, attenMask=%p)",
                opParamInfo.query.shape, opParamInfo.key.shape, opParamInfo.value.shape, opParamInfo.qDescale.shape,
                opParamInfo.kDescale.shape, opParamInfo.vDescale.shape, opParamInfo.pScale.shape,
                opParamInfo.sparseIndices.shape, opParamInfo.sparseSeqLen.shape, opParamInfo.attenMask.shape);
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &queryShape = opParamInfo.query.shape->GetStorageShape();
    const gert::Shape &keyShape = opParamInfo.key.shape->GetStorageShape();
    const gert::Shape &sparseIndicesShape = opParamInfo.sparseIndices.shape->GetStorageShape();
    const gert::Shape &sparseSeqLenShape = opParamInfo.sparseSeqLen.shape->GetStorageShape();
    const gert::Shape &attenMaskShape = opParamInfo.attenMask.shape->GetStorageShape();
    if (attenMaskShape.GetDimNum() != DIM_NUM_2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(kOpName, "attention_mask", std::to_string(attenMaskShape.GetDimNum()) + "D", "2D");
        return ge::GRAPH_FAILED;
    }

    if (ParseAttributes(tilingInfo, attrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckQuantModeAttrs(tilingInfo, attrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckQuantModeScaleDtype(tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (ParseQuery(tilingInfo, queryShape, sparseIndicesShape, attrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &valueShape = opParamInfo.value.shape->GetStorageShape();
    if (tilingInfo.quantModeVal == BSA_QUANT_MODE_MXFP8_FULL_QUANT &&
        CheckMXFP8FullQuantKVLayout(keyShape, valueShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &vDescaleShape = opParamInfo.vDescale.shape->GetStorageShape();
    if (ParseKeyValue(tilingInfo, keyShape, vDescaleShape, attrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo.quantModeVal == BSA_QUANT_MODE_MXFP8_FULL_QUANT &&
        CheckMXFP8FullQuantShape(tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseSparseIndices(tilingInfo, sparseIndicesShape, sparseSeqLenShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseOptionalInputs(tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tilingInfo.gS1OuterSize = tilingInfo.qbMax;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
