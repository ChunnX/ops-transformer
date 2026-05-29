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
 * \file block_sparse_attention_tiling.cpp
 * \brief TODO: 待实现
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "block_sparse_attention_tiling.h"
#include "register/op_impl_registry.h"

namespace optiling {
namespace {
constexpr size_t DIM_NUM_2 = 2U;
constexpr size_t DIM_NUM_3 = 3U;
constexpr size_t DIM_NUM_4 = 4U;
constexpr size_t DIM_NUM_5 = 5U;
constexpr size_t DIM_B = 0U;
constexpr size_t DIM_S = 1U;
constexpr size_t DIM_N = 2U;
constexpr size_t DIM_BLOCK_SIZE = 1U;
constexpr size_t DIM_BLOCK_TABLE_MAX = 1U;
constexpr size_t DIM_SPARSE_QB = 3U;
constexpr size_t DIM_SPARSE_COUNT = 4U;

uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return divisor == 0U ? 0U : (value + divisor - 1U) / divisor;
}

bool GetDimAsU32(const gert::Shape &shape, size_t dimIndex, uint32_t &value)
{
    if (dimIndex >= shape.GetDimNum()) {
        return false;
    }
    const auto dim = shape.GetDim(dimIndex);
    if (dim <= 0 || dim > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    value = static_cast<uint32_t>(dim);
    return true;
}

uint32_t GetPositiveAttr(const gert::RuntimeAttrs *attrs, uint32_t index, uint32_t defaultValue)
{
    const int64_t *attrPtr = attrs->GetAttrPointer<int64_t>(index);
    if (attrPtr == nullptr || *attrPtr <= 0) {
        return defaultValue;
    }
    if (*attrPtr > std::numeric_limits<uint32_t>::max()) {
        return defaultValue;
    }
    return static_cast<uint32_t>(*attrPtr);
}

uint32_t GetUintAttr(const gert::RuntimeAttrs *attrs, uint32_t index, uint32_t defaultValue)
{
    const int64_t *attrPtr = attrs->GetAttrPointer<int64_t>(index);
    if (attrPtr == nullptr || *attrPtr < 0) {
        return defaultValue;
    }
    if (*attrPtr > std::numeric_limits<uint32_t>::max()) {
        return defaultValue;
    }
    return static_cast<uint32_t>(*attrPtr);
}

float GetFloatAttr(const gert::RuntimeAttrs *attrs, uint32_t index, float defaultValue)
{
    const float *attrPtr = attrs->GetAttrPointer<float>(index);
    return attrPtr == nullptr ? defaultValue : *attrPtr;
}
} // namespace

ge::graphStatus TilingBlockSparseAttention(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    auto rawTilingData = context->GetRawTilingData();
    if (attrs == nullptr || rawTilingData == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *queryStorageShape = context->GetInputShape(BSA_QUERY_INDEX);
    const gert::StorageShape *keyStorageShape = context->GetInputShape(BSA_KEY_INDEX);
    const gert::StorageShape *sparseIndicesStorageShape = context->GetInputShape(BSA_SPARSE_INDICES_INDEX);
    const gert::StorageShape *sparseSeqLenStorageShape = context->GetInputShape(BSA_SPARSE_SEQ_LEN_INDEX);
    const gert::StorageShape *attenMaskStorageShape = context->GetInputShape(BSA_ATTEN_MASK_INDEX);
    if (queryStorageShape == nullptr || keyStorageShape == nullptr || sparseIndicesStorageShape == nullptr ||
        sparseSeqLenStorageShape == nullptr || attenMaskStorageShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &queryShape = queryStorageShape->GetStorageShape();
    const gert::Shape &keyShape = keyStorageShape->GetStorageShape();
    const gert::Shape &sparseIndicesShape = sparseIndicesStorageShape->GetStorageShape();
    const gert::Shape &sparseSeqLenShape = sparseSeqLenStorageShape->GetStorageShape();
    const gert::Shape &attenMaskShape = attenMaskStorageShape->GetStorageShape();
    if (sparseIndicesShape.GetDimNum() != DIM_NUM_5 || sparseSeqLenShape.GetDimNum() != DIM_NUM_4 ||
        attenMaskShape.GetDimNum() != DIM_NUM_2) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t qBlockSize = GetPositiveAttr(attrs, BSA_Q_BLOCK_SIZE_ATTR_INDEX, BSA_BLOCK_SIZE);
    const uint32_t kvBlockSize = GetPositiveAttr(attrs, BSA_KV_BLOCK_SIZE_ATTR_INDEX, BSA_BLOCK_SIZE);
    if (qBlockSize != BSA_BLOCK_SIZE || kvBlockSize != BSA_BLOCK_SIZE) {
        return ge::GRAPH_FAILED;
    }

    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t s1Size = 0;
    const size_t queryDimNum = queryShape.GetDimNum();
    if (queryDimNum == DIM_NUM_4) {
        if (!GetDimAsU32(queryShape, DIM_B, bSize) || !GetDimAsU32(queryShape, DIM_S, s1Size) ||
            !GetDimAsU32(queryShape, DIM_N, n1Size)) {
            return ge::GRAPH_FAILED;
        }
    } else if (queryDimNum == DIM_NUM_3) {
        if (!GetDimAsU32(sparseIndicesShape, DIM_B, bSize) || !GetDimAsU32(queryShape, DIM_B, s1Size) ||
            !GetDimAsU32(queryShape, DIM_S, n1Size)) {
            return ge::GRAPH_FAILED;
        }
        s1Size = GetPositiveAttr(attrs, BSA_MAX_SEQLEN_Q_ATTR_INDEX, s1Size);
    } else {
        return ge::GRAPH_FAILED;
    }

    uint32_t n2Size = 0;
    uint32_t physicalKvBlockSize = 0;
    if (keyShape.GetDimNum() != DIM_NUM_4 || !GetDimAsU32(keyShape, DIM_BLOCK_SIZE, physicalKvBlockSize) ||
        !GetDimAsU32(keyShape, DIM_N, n2Size) || physicalKvBlockSize != kvBlockSize) {
        return ge::GRAPH_FAILED;
    }
    if (n2Size == 0U || n1Size % n2Size != 0U) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t gSize = n1Size / n2Size;
    const uint32_t s2Size = GetPositiveAttr(attrs, BSA_MAX_SEQLEN_KV_ATTR_INDEX, kvBlockSize);
    const uint32_t kbMax = CeilDiv(s2Size, kvBlockSize);
    const uint32_t qbMax = CeilDiv(s1Size, qBlockSize);

    uint32_t sparseB = 0;
    uint32_t sparseN2 = 0;
    uint32_t sparseG = 0;
    uint32_t sparseQb = 0;
    uint32_t sparseCount = 0;
    if (!GetDimAsU32(sparseIndicesShape, DIM_B, sparseB) || !GetDimAsU32(sparseIndicesShape, DIM_S, sparseN2) ||
        !GetDimAsU32(sparseIndicesShape, DIM_N, sparseG) ||
        !GetDimAsU32(sparseIndicesShape, DIM_SPARSE_QB, sparseQb) ||
        !GetDimAsU32(sparseIndicesShape, DIM_SPARSE_COUNT, sparseCount)) {
        return ge::GRAPH_FAILED;
    }
    if (sparseB != bSize || sparseN2 != n2Size || sparseG != gSize || sparseQb != qbMax) {
        return ge::GRAPH_FAILED;
    }

    uint32_t seqLenB = 0;
    uint32_t seqLenN2 = 0;
    uint32_t seqLenG = 0;
    uint32_t seqLenQb = 0;
    if (!GetDimAsU32(sparseSeqLenShape, DIM_B, seqLenB) || !GetDimAsU32(sparseSeqLenShape, DIM_S, seqLenN2) ||
        !GetDimAsU32(sparseSeqLenShape, DIM_N, seqLenG) ||
        !GetDimAsU32(sparseSeqLenShape, DIM_SPARSE_QB, seqLenQb)) {
        return ge::GRAPH_FAILED;
    }
    if (seqLenB != bSize || seqLenN2 != n2Size || seqLenG != gSize || seqLenQb != qbMax) {
        return ge::GRAPH_FAILED;
    }

    uint32_t maxBlockNumPerBatch = kbMax;
    const gert::StorageShape *blockTableStorageShape = context->GetOptionalInputShape(BSA_BLOCK_TABLE_INDEX);
    if (blockTableStorageShape != nullptr && blockTableStorageShape->GetStorageShape().GetShapeSize() > 0) {
        const gert::Shape &blockTableShape = blockTableStorageShape->GetStorageShape();
        uint32_t blockTableB = 0;
        if (blockTableShape.GetDimNum() != DIM_NUM_2 || !GetDimAsU32(blockTableShape, DIM_B, blockTableB) ||
            !GetDimAsU32(blockTableShape, DIM_BLOCK_TABLE_MAX, maxBlockNumPerBatch) || blockTableB != bSize) {
            return ge::GRAPH_FAILED;
        }
    }

    const uint64_t totalTaskNum = static_cast<uint64_t>(bSize) * n2Size * gSize * qbMax;
    if (totalTaskNum == 0U || totalTaskNum > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t usedCoreNum = static_cast<uint32_t>(
        std::min<uint64_t>(static_cast<uint64_t>(BSA_MAX_CORE_NUM), totalTaskNum));
    uint32_t coreStart[BSA_MAX_CORE_NUM] = {};
    uint32_t coreEnd[BSA_MAX_CORE_NUM] = {};
    for (uint32_t coreIdx = 0U; coreIdx < usedCoreNum; ++coreIdx) {
        coreStart[coreIdx] = static_cast<uint32_t>(totalTaskNum * coreIdx / usedCoreNum);
        coreEnd[coreIdx] = static_cast<uint32_t>(totalTaskNum * (coreIdx + 1U) / usedCoreNum);
    }

    BlockSparseAttentionTilingData tilingData;
    tilingData.baseParams.set_bSize(bSize);
    tilingData.baseParams.set_n1Size(n1Size);
    tilingData.baseParams.set_n2Size(n2Size);
    tilingData.baseParams.set_gSize(gSize);
    tilingData.baseParams.set_s1Size(s1Size);
    tilingData.baseParams.set_s2Size(s2Size);
    tilingData.baseParams.set_qbMax(qbMax);
    tilingData.baseParams.set_kbMax(kbMax);
    tilingData.baseParams.set_sparseCount(sparseCount);
    tilingData.baseParams.set_blockSize(qBlockSize);
    tilingData.baseParams.set_maxBlockNumPerBatch(maxBlockNumPerBatch);
    tilingData.baseParams.set_attenMaskCompressMode(GetUintAttr(attrs, BSA_MASK_MODE_ATTR_INDEX, 0U));
    tilingData.baseParams.set_usedCoreNum(usedCoreNum);
    tilingData.baseParams.set_softmaxScale(GetFloatAttr(attrs, BSA_SOFTMAX_SCALE_ATTR_INDEX, 1.0F));
    tilingData.set_coreStart(coreStart);
    tilingData.set_coreEnd(coreEnd);

    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(0);
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
graphStatus TilingPrepareBlockSparseAttention(gert::TilingParseContext *context)
{
    // TODO: 待实现 - 填充 compile info。
    return context == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class BlockSparseAttentionTiling {
public:
    static ge::graphStatus Tiling(gert::TilingContext *context)
    {
        return optiling::TilingBlockSparseAttention(context);
    }
};
} // namespace ops

namespace optiling {
IMPL_OP_OPTILING(BlockSparseAttention)
    .Tiling(TilingBlockSparseAttention);
} // namespace optiling
