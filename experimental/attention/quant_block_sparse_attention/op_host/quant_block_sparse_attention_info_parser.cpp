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
 * \file quant_block_sparse_attention_info_parser.cpp
 * \brief QuantBlockSparseAttention parameter parsing implementation.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "quant_block_sparse_attention_info_parser.h"
#include "quant_block_sparse_attention_tiling.h"

namespace optiling {
namespace {
constexpr size_t DIM_NUM_2 = 2U;
constexpr size_t DIM_NUM_3 = 3U;
constexpr size_t DIM_NUM_4 = 4U;
constexpr size_t DIM_NUM_5 = 5U;
constexpr size_t DIM_B = 0U;
constexpr size_t DIM_S = 1U;
constexpr size_t DIM_N = 2U;
constexpr size_t DIM_D = 3U;
constexpr size_t DIM_PA_N = 1U;
constexpr size_t DIM_PA_BLOCK_SIZE = 2U;
constexpr size_t DIM_BLOCK_TABLE_MAX = 1U;
constexpr size_t DIM_SPARSE_QB = 3U;
constexpr size_t DIM_SPARSE_COUNT = 4U;
constexpr uint32_t BSA_LAYOUT_Q_BSND_VALUE = 0U;
constexpr uint32_t BSA_LAYOUT_Q_TND_VALUE = 1U;
} // namespace

QuantBlockSparseAttentionInfoParser::QuantBlockSparseAttentionInfoParser(gert::TilingContext *context)
    : context_(context)
{
}

ge::graphStatus QuantBlockSparseAttentionInfoParser::ParseQuery(QuantBlockSparseAttentionTilingInfo &tilingInfo,
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
            return ge::GRAPH_FAILED;
        }
        tilingInfo.qTokenNum = tilingInfo.bSize * tilingInfo.s1Size;
    } else if (queryDimNum == DIM_NUM_3 && layoutQ == "TND") {
        tilingInfo.isBSND = false;
        tilingInfo.layoutQValue = BSA_LAYOUT_Q_TND_VALUE;
        if (!BSAGetDimAsU32(sparseIndicesShape, DIM_B, tilingInfo.bSize) ||
            !BSAGetDimAsU32(queryShape, DIM_B, tilingInfo.s1Size) ||
            !BSAGetDimAsU32(queryShape, DIM_S, tilingInfo.n1Size) ||
            !BSAGetDimAsU32(queryShape, DIM_N, tilingInfo.dSize)) {
            return ge::GRAPH_FAILED;
        }
        tilingInfo.qTokenNum = tilingInfo.s1Size;
        tilingInfo.s1Size = BSAGetPositiveAttr(attrs, BSA_MAX_SEQLEN_Q_ATTR_INDEX, tilingInfo.s1Size);
    } else {
        return ge::GRAPH_FAILED;
    }

    if (tilingInfo.dSize != BSA_D_SIZE) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttentionInfoParser::ParseKeyValue(QuantBlockSparseAttentionTilingInfo &tilingInfo,
                                                                   const gert::Shape &keyShape,
                                                                   const gert::RuntimeAttrs *attrs)
{
    uint32_t physicalKvBlockSize = 0;
    uint32_t kvDSize = 0;
    if (keyShape.GetDimNum() != DIM_NUM_4 || !BSAGetDimAsU32(keyShape, DIM_PA_BLOCK_SIZE, physicalKvBlockSize) ||
        !BSAGetDimAsU32(keyShape, DIM_PA_N, tilingInfo.n2Size) || !BSAGetDimAsU32(keyShape, DIM_SPARSE_QB, kvDSize) ||
        physicalKvBlockSize != tilingInfo.kvBlockSizeVal || kvDSize != BSA_D_SIZE) {
        return ge::GRAPH_FAILED;
    }

    if (tilingInfo.n2Size == 0U || tilingInfo.n1Size % tilingInfo.n2Size != 0U) {
        return ge::GRAPH_FAILED;
    }

    tilingInfo.gSize = tilingInfo.n1Size / tilingInfo.n2Size;
    tilingInfo.isGqa = (tilingInfo.gSize > 1U);

    tilingInfo.s2Size = BSAGetPositiveAttr(attrs, BSA_MAX_SEQLEN_KV_ATTR_INDEX, tilingInfo.kvBlockSizeVal);
    tilingInfo.kbMax = BSACeilDiv(tilingInfo.s2Size, tilingInfo.kvBlockSizeVal);
    tilingInfo.qbMax = BSACeilDiv(tilingInfo.s1Size, tilingInfo.qBlockSizeVal);

    tilingInfo.dSizeV = BSA_D_SIZE;

    if (!BSAGetDimAsU32(keyShape, DIM_B, tilingInfo.paBlockNumSum)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttentionInfoParser::ParseSparseIndices(QuantBlockSparseAttentionTilingInfo &tilingInfo,
                                                                        const gert::Shape &sparseIndicesShape,
                                                                        const gert::Shape &sparseSeqLenShape)
{
    if (sparseIndicesShape.GetDimNum() != DIM_NUM_5 || sparseSeqLenShape.GetDimNum() != DIM_NUM_4) {
        return ge::GRAPH_FAILED;
    }

    uint32_t sparseB = 0;
    uint32_t sparseN2 = 0;
    uint32_t sparseG = 0;
    uint32_t sparseQb = 0;
    if (!BSAGetDimAsU32(sparseIndicesShape, DIM_B, sparseB) || !BSAGetDimAsU32(sparseIndicesShape, DIM_S, sparseN2) ||
        !BSAGetDimAsU32(sparseIndicesShape, DIM_N, sparseG) ||
        !BSAGetDimAsU32(sparseIndicesShape, DIM_SPARSE_QB, sparseQb) ||
        !BSAGetDimAsU32(sparseIndicesShape, DIM_SPARSE_COUNT, tilingInfo.sparseCount)) {
        return ge::GRAPH_FAILED;
    }
    if (sparseB != tilingInfo.bSize || sparseN2 != tilingInfo.n2Size || sparseG != tilingInfo.gSize ||
        sparseQb != tilingInfo.qbMax) {
        return ge::GRAPH_FAILED;
    }

    uint32_t seqLenB = 0;
    uint32_t seqLenN2 = 0;
    uint32_t seqLenG = 0;
    uint32_t seqLenQb = 0;
    if (!BSAGetDimAsU32(sparseSeqLenShape, DIM_B, seqLenB) || !BSAGetDimAsU32(sparseSeqLenShape, DIM_S, seqLenN2) ||
        !BSAGetDimAsU32(sparseSeqLenShape, DIM_N, seqLenG) ||
        !BSAGetDimAsU32(sparseSeqLenShape, DIM_SPARSE_QB, seqLenQb)) {
        return ge::GRAPH_FAILED;
    }
    if (seqLenB != tilingInfo.bSize || seqLenN2 != tilingInfo.n2Size || seqLenG != tilingInfo.gSize ||
        seqLenQb != tilingInfo.qbMax) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus
QuantBlockSparseAttentionInfoParser::ParseOptionalInputs(QuantBlockSparseAttentionTilingInfo &tilingInfo)
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
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttentionInfoParser::ParseAttributes(QuantBlockSparseAttentionTilingInfo &tilingInfo,
                                                                     const gert::RuntimeAttrs *attrs)
{
    auto &opParamInfo = tilingInfo.opParamInfo;

    opParamInfo.qBlockSize = attrs->GetAttrPointer<int64_t>(BSA_SPARSE_Q_BLOCK_SIZE_ATTR_INDEX);
    opParamInfo.kvBlockSize = attrs->GetAttrPointer<int64_t>(BSA_SPARSE_KV_BLOCK_SIZE_ATTR_INDEX);
    opParamInfo.paBlockStride = attrs->GetAttrPointer<int64_t>(BSA_PA_BLOCK_STRIDE_ATTR_INDEX);
    tilingInfo.qBlockSizeVal = BSAGetPositiveAttr(attrs, BSA_SPARSE_Q_BLOCK_SIZE_ATTR_INDEX, BSA_BLOCK_SIZE);
    tilingInfo.kvBlockSizeVal = BSAGetPositiveAttr(attrs, BSA_SPARSE_KV_BLOCK_SIZE_ATTR_INDEX, BSA_BLOCK_SIZE);
    tilingInfo.paBlockStrideVal = BSAGetUintAttr(attrs, BSA_PA_BLOCK_STRIDE_ATTR_INDEX, 0U);
    if (tilingInfo.qBlockSizeVal != BSA_BLOCK_SIZE || tilingInfo.kvBlockSizeVal != BSA_BLOCK_SIZE) {
        return ge::GRAPH_FAILED;
    }

    opParamInfo.softmaxScale = attrs->GetAttrPointer<float>(BSA_SOFTMAX_SCALE_ATTR_INDEX);
    opParamInfo.maskMode = attrs->GetAttrPointer<int64_t>(BSA_MASK_MODE_ATTR_INDEX);
    opParamInfo.returnSoftmaxLse = attrs->GetAttrPointer<bool>(BSA_RETURN_SOFTMAX_LSE_ATTR_INDEX);
    opParamInfo.maxSeqlenQ = attrs->GetAttrPointer<int64_t>(BSA_MAX_SEQLEN_Q_ATTR_INDEX);
    opParamInfo.maxSeqlenKV = attrs->GetAttrPointer<int64_t>(BSA_MAX_SEQLEN_KV_ATTR_INDEX);
    opParamInfo.quantMode = attrs->GetAttrPointer<int64_t>(BSA_QUANT_MODE_ATTR_INDEX);

    tilingInfo.softmaxScaleVal = BSAGetFloatAttr(attrs, BSA_SOFTMAX_SCALE_ATTR_INDEX, 1.0F);
    tilingInfo.maskModeVal = BSAGetUintAttr(attrs, BSA_MASK_MODE_ATTR_INDEX, 0U);
    tilingInfo.quantModeVal = BSAGetUintAttr(attrs, BSA_QUANT_MODE_ATTR_INDEX, 1U);
    tilingInfo.returnSoftmaxLseVal = BSAGetBoolAttr(attrs, BSA_RETURN_SOFTMAX_LSE_ATTR_INDEX, false);

    opParamInfo.query.desc = context_->GetInputDesc(BSA_QUERY_INDEX);
    tilingInfo.qDtype =
        (opParamInfo.query.desc != nullptr) ? opParamInfo.query.desc->GetDataType() : ge::DT_FLOAT8_E4M3FN;
    opParamInfo.key.desc = context_->GetInputDesc(BSA_KEY_INDEX);
    tilingInfo.kvDtype = (opParamInfo.key.desc != nullptr) ? opParamInfo.key.desc->GetDataType() : ge::DT_FLOAT8_E4M3FN;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttentionInfoParser::Parse(QuantBlockSparseAttentionTilingInfo &tilingInfo)
{
    if (context_ == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto &opParamInfo = tilingInfo.opParamInfo;

    opParamInfo.query.shape = context_->GetInputShape(BSA_QUERY_INDEX);
    opParamInfo.key.shape = context_->GetInputShape(BSA_KEY_INDEX);
    opParamInfo.value.desc = context_->GetInputDesc(BSA_VALUE_INDEX);
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
        opParamInfo.sparseIndices.shape == nullptr || opParamInfo.sparseSeqLen.shape == nullptr ||
        opParamInfo.attenMask.shape == nullptr) { // TODOcly 这里的校验在加上其他的必传参数
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &queryShape = opParamInfo.query.shape->GetStorageShape();
    const gert::Shape &keyShape = opParamInfo.key.shape->GetStorageShape();
    const gert::Shape &sparseIndicesShape = opParamInfo.sparseIndices.shape->GetStorageShape();
    const gert::Shape &sparseSeqLenShape = opParamInfo.sparseSeqLen.shape->GetStorageShape();
    const gert::Shape &attenMaskShape = opParamInfo.attenMask.shape->GetStorageShape();
    if (attenMaskShape.GetDimNum() != DIM_NUM_2) {
        return ge::GRAPH_FAILED;
    }

    if (ParseAttributes(tilingInfo, attrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const std::string layoutKV = BSAGetStringAttr(attrs, BSA_LAYOUT_KV_ATTR_INDEX, "PA_BNSD");
    const std::string layoutSparseIndices = BSAGetStringAttr(attrs, BSA_LAYOUT_SPARSE_INDICES_ATTR_INDEX, "B_N_Qb_Kb");
    if (layoutKV != "PA_BNSD" || layoutSparseIndices != "B_N_Qb_Kb" || tilingInfo.quantModeVal != 1U ||
        (tilingInfo.maskModeVal != 0U && tilingInfo.maskModeVal != 3U)) {
        return ge::GRAPH_FAILED;
    }

    if (ParseQuery(tilingInfo, queryShape, sparseIndicesShape, attrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseKeyValue(tilingInfo, keyShape, attrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseSparseIndices(tilingInfo, sparseIndicesShape, sparseSeqLenShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseOptionalInputs(tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tilingInfo.gS1OuterSize = tilingInfo.gSize * tilingInfo.qbMax;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
