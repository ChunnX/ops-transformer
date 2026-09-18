/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <limits>
#include <string>

#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "qbsa_mxfp8_tiling.h"
#include "../../tiling_common/qbsa_check_utils.h"
#include "../../../op_kernel/arch35/common/quant_block_sparse_attn_const.h"

namespace optiling {
using namespace qbsa_check;
namespace {
constexpr const char *kOpName = "QuantBlockSparseAttn";
constexpr size_t DIM_NUM_1 = 1U;
constexpr size_t DIM_NUM_3 = 3U;
constexpr size_t DIM_NUM_4 = 4U;
constexpr size_t DIM_NUM_5 = 5U;
constexpr int64_t QBSA_MXFP8_P_SCALE_SIZE = 1;

bool IsMXFP8SparseBlockSizeSupported(uint32_t blockSize)
{
    return blockSize == QBSA_MXFP8_SPARSE_BLOCK_SIZE_128 || blockSize == QBSA_MXFP8_SPARSE_BLOCK_SIZE_64;
}

ge::graphStatus CheckMXFP8FullQuantScaleInputs(const QuantBlockSparseAttnParaInfo &opParamInfo)
{
    if (opParamInfo.kDescale.shape == nullptr || opParamInfo.vDescale.shape == nullptr ||
        opParamInfo.qDescale.shape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "k_descale/v_descale/q_descale", "nullptr",
                                              "All must be passed in quant_mode=2 MXFP8 full-quant scenario");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

QbsaMxfp8Arch35Tiling::QbsaMxfp8Arch35Tiling(gert::TilingContext *context)
    : QbsaTilingStrategy(context)
{}

ge::graphStatus QbsaMxfp8Arch35Tiling::CheckAttributes() const
{
    if (CheckMxAttrs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckMxSparseBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckLayouts() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckSoftmaxScale();
}

ge::graphStatus QbsaMxfp8Arch35Tiling::CheckTensorMetadata() const
{
    if (CheckPaInputs() != ge::GRAPH_SUCCESS || CheckMetadata() != ge::GRAPH_SUCCESS ||
        CheckReservedInputs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckFormat() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckMxScaleDtype();
}

ge::graphStatus QbsaMxfp8Arch35Tiling::ParseShapeFacts(QbsaTilingContextParser &parser)
{
    if (parser.ParsePaShapeFacts(tilingInfo_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ResolveHeadGrouping();
}

ge::graphStatus QbsaMxfp8Arch35Tiling::CheckShapeRelations() const
{
    if (CheckMxPaBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeLimits() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckKeyValueShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckActualSeqLen() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNoneOrCausalMask() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckMxfp8FullQuantShape();
}

ge::graphStatus QbsaMxfp8Arch35Tiling::CheckMxAttrs() const
{
    if (tilingInfo_.layoutQStr != "TND" || tilingInfo_.layoutOutStr != "TND") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "layout_q/layout_out",
                                              tilingInfo_.layoutQStr + "/" + tilingInfo_.layoutOutStr,
                                              "Must both be TND in quant_mode=2 MXFP8 full-quant scenario");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaMxfp8Arch35Tiling::CheckMxScaleDtype() const
{
    const auto &opParamInfo = tilingInfo_.opParamInfo;
    if (CheckInputDtype(opParamInfo.qDescale, "q_descale", ge::DT_FLOAT8_E8M0, "FLOAT8_E8M0") != ge::GRAPH_SUCCESS ||
        CheckInputDtype(opParamInfo.kDescale, "k_descale", ge::DT_FLOAT8_E8M0, "FLOAT8_E8M0") != ge::GRAPH_SUCCESS ||
        CheckInputDtype(opParamInfo.vDescale, "v_descale", ge::DT_FLOAT8_E8M0, "FLOAT8_E8M0") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.pScale.desc != nullptr && opParamInfo.pScale.shape != nullptr &&
        opParamInfo.pScale.shape->GetStorageShape().GetShapeSize() > 0 &&
        opParamInfo.pScale.desc->GetDataType() != ge::DT_FLOAT8_E8M0 &&
        opParamInfo.pScale.desc->GetDataType() != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "p_scale",
                                  ge::TypeUtils::DataTypeToSerialString(opParamInfo.pScale.desc->GetDataType()),
                                  "FLOAT8_E8M0 or FLOAT");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaMxfp8Arch35Tiling::CheckMxSparseBlockSize() const
{
    const auto &opParamInfo = tilingInfo_.opParamInfo;
    if (opParamInfo.qBlockSize == nullptr || opParamInfo.kvBlockSize == nullptr || *opParamInfo.qBlockSize <= 0 ||
        *opParamInfo.kvBlockSize <= 0 || *opParamInfo.qBlockSize > std::numeric_limits<uint32_t>::max() ||
        *opParamInfo.kvBlockSize > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "sparse_q_block_size/sparse_kv_block_size", "invalid",
                                              "Both attributes must be positive uint32 values");
        return ge::GRAPH_FAILED;
    }
    if (!IsMXFP8SparseBlockSizeSupported(tilingInfo_.qBlockSizeVal) ||
        !IsMXFP8SparseBlockSizeSupported(tilingInfo_.kvBlockSizeVal)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "q_block_size/kv_block_size",
            std::to_string(tilingInfo_.qBlockSizeVal) + "/" + std::to_string(tilingInfo_.kvBlockSizeVal),
            "Must each be " + std::to_string(QBSA_MXFP8_SPARSE_BLOCK_SIZE_128) + " or " +
                std::to_string(QBSA_MXFP8_SPARSE_BLOCK_SIZE_64) + " in quant_mode=2 MXFP8 full-quant scenario");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.qBlockSizeVal != tilingInfo_.kvBlockSizeVal) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "q_block_size/kv_block_size",
            std::to_string(tilingInfo_.qBlockSizeVal) + "/" + std::to_string(tilingInfo_.kvBlockSizeVal),
            "Sparse_q_block_size must equal sparse_kv_block_size in quant_mode=2 MXFP8 full-quant scenario");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaMxfp8Arch35Tiling::CheckMxPaBlockSize() const
{
    if (tilingInfo_.paBlockSizeVal == 0U || tilingInfo_.paBlockSizeVal % tilingInfo_.kvBlockSizeVal != 0U ||
        tilingInfo_.paBlockSizeVal > QBSA_MXFP8_MAX_PA_BLOCK_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "pa_block_size (key dim 2)", std::to_string(tilingInfo_.paBlockSizeVal),
            "In quant_mode=2 MXFP8 full-quant scenario, pa_block_size must be a positive multiple of "
            "sparse_block_size (" +
                std::to_string(tilingInfo_.kvBlockSizeVal) +
                ") and <= " + std::to_string(QBSA_MXFP8_MAX_PA_BLOCK_SIZE));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaMxfp8Arch35Tiling::CheckMxfp8FullQuantShape() const
{
    const auto &opParamInfo = tilingInfo_.opParamInfo;
    if (CheckMXFP8FullQuantScaleInputs(opParamInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

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

    const int64_t scaleDSize = static_cast<int64_t>(QBSACeilDiv(tilingInfo_.dSize, QBSA_MXFP8_SCALE_GROUP_SIZE));
    const int64_t valueScaleBlockSize =
        static_cast<int64_t>(QBSACeilDiv(tilingInfo_.paBlockSizeVal, QBSA_MXFP8_SCALE_GROUP_SIZE));
    const int64_t paBlockSize = static_cast<int64_t>(tilingInfo_.paBlockSizeVal);
    const std::array<int64_t, DIM_NUM_1> pScaleShape = {QBSA_MXFP8_P_SCALE_SIZE};
    const std::array<int64_t, DIM_NUM_3> queryShapeExpect = {static_cast<int64_t>(tilingInfo_.qTokenNum),
                                                             static_cast<int64_t>(tilingInfo_.n1Size),
                                                             static_cast<int64_t>(tilingInfo_.dSize)};
    const std::array<int64_t, DIM_NUM_4> queryAntiquantScaleShape = {static_cast<int64_t>(tilingInfo_.qTokenNum),
                                                                     static_cast<int64_t>(tilingInfo_.n1Size),
                                                                     scaleDSize, QBSA_MXFP8_SCALE_LAST_DIM};
    const std::array<int64_t, DIM_NUM_4> keyShapeExpect = {static_cast<int64_t>(tilingInfo_.paBlockNumSum),
                                                           static_cast<int64_t>(tilingInfo_.n2Size), paBlockSize,
                                                           static_cast<int64_t>(tilingInfo_.dSize)};
    const std::array<int64_t, DIM_NUM_4> valueShapeExpect = {static_cast<int64_t>(tilingInfo_.paBlockNumSum),
                                                             static_cast<int64_t>(tilingInfo_.n2Size), paBlockSize,
                                                             static_cast<int64_t>(tilingInfo_.dSizeV)};
    const std::array<int64_t, DIM_NUM_5> keyAntiquantScaleShape = {static_cast<int64_t>(tilingInfo_.paBlockNumSum),
                                                                   static_cast<int64_t>(tilingInfo_.n2Size),
                                                                   paBlockSize, scaleDSize, QBSA_MXFP8_SCALE_LAST_DIM};
    const std::array<int64_t, DIM_NUM_5> valueAntiquantScaleShape = {
        static_cast<int64_t>(tilingInfo_.paBlockNumSum), static_cast<int64_t>(tilingInfo_.n2Size), valueScaleBlockSize,
        static_cast<int64_t>(tilingInfo_.dSizeV), QBSA_MXFP8_SCALE_LAST_DIM};

    if (CheckInputShape(opParamInfo.query, "query", queryShapeExpect) != ge::GRAPH_SUCCESS ||
        CheckInputShape(opParamInfo.qDescale, "q_descale", queryAntiquantScaleShape) != ge::GRAPH_SUCCESS ||
        CheckInputShape(opParamInfo.key, "key", keyShapeExpect) != ge::GRAPH_SUCCESS ||
        CheckInputShape(opParamInfo.value, "value", valueShapeExpect) != ge::GRAPH_SUCCESS ||
        CheckInputShape(opParamInfo.kDescale, "k_descale", keyAntiquantScaleShape) != ge::GRAPH_SUCCESS ||
        CheckInputShape(opParamInfo.vDescale, "v_descale", valueAntiquantScaleShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // pScale may be an empty tensor (shape size 0) to indicate default scale 1.0
    const int64_t pScaleShapeSize =
        (opParamInfo.pScale.shape != nullptr) ? opParamInfo.pScale.shape->GetStorageShape().GetShapeSize() : 0;
    if (pScaleShapeSize != 0) {
        if (CheckInputShape(opParamInfo.pScale, "quantScale1", pScaleShape) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
