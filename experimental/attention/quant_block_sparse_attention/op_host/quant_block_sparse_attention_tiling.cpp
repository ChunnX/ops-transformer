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
 * \file quant_block_sparse_attention_tiling.cpp
 * \brief TODO: 待实现
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <tiling/platform/platform_ascendc.h>
#include "../op_kernel/quant_block_sparse_attention_template_tiling_key.h"
#include "quant_block_sparse_attention_tiling.h"
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
constexpr uint32_t BSA_LAYOUT_Q_BSND_VALUE = 0U;
constexpr uint32_t BSA_LAYOUT_Q_TND_VALUE = 1U;
constexpr uint32_t BSA_LAYOUT_KV_PA_BNSD_VALUE = 1U;
constexpr uint32_t BSA_LAYOUT_SPARSE_B_N_QB_KB_VALUE = 0U;
constexpr uint8_t BSA_LAYOUT_TYPE_BSH = 1U;
constexpr uint8_t BSA_LAYOUT_TYPE_TND = 4U;
constexpr uint8_t BSA_PA_LAYOUT_TYPE_BNBD = 0U;
constexpr uint8_t BSA_COMPAT_MASK_NONE = 0U;
constexpr uint8_t BSA_COMPAT_MASK_LEFT_UP_CAUSAL = 1U;

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

bool GetBoolAttr(const gert::RuntimeAttrs *attrs, uint32_t index, bool defaultValue)
{
    const bool *attrPtr = attrs->GetAttrPointer<bool>(index);
    return attrPtr == nullptr ? defaultValue : *attrPtr;
}

std::string GetStringAttr(const gert::RuntimeAttrs *attrs, uint32_t index, const char *defaultValue)
{
    const char *attrPtr = attrs->GetAttrPointer<char>(index);
    return attrPtr == nullptr ? std::string(defaultValue) : std::string(attrPtr);
}

uint32_t GetAicCoreNum(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return BSA_MAX_CORE_NUM;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    return aicNum == 0U ? BSA_MAX_CORE_NUM : std::min<uint32_t>(aicNum, BSA_MAX_CORE_NUM);
}
} // namespace

ge::graphStatus TilingQuantBlockSparseAttention(gert::TilingContext *context)
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

    const uint32_t qBlockSize = GetPositiveAttr(attrs, BSA_SPARSE_Q_BLOCK_SIZE_ATTR_INDEX, BSA_BLOCK_SIZE);
    const uint32_t kvBlockSize = GetPositiveAttr(attrs, BSA_SPARSE_KV_BLOCK_SIZE_ATTR_INDEX, BSA_BLOCK_SIZE);
    if (qBlockSize != BSA_BLOCK_SIZE || kvBlockSize != BSA_BLOCK_SIZE) {
        return ge::GRAPH_FAILED;
    }

    const std::string layoutKv = GetStringAttr(attrs, BSA_LAYOUT_KV_ATTR_INDEX, "PA_BNSD");
    const std::string layoutQ = GetStringAttr(attrs, BSA_LAYOUT_Q_ATTR_INDEX, "TND");
    const std::string layoutSparseIndices =
        GetStringAttr(attrs, BSA_LAYOUT_SPARSE_INDICES_ATTR_INDEX, "B_N_Qb_Kb");
    const uint32_t quantMode = GetUintAttr(attrs, BSA_QUANT_MODE_ATTR_INDEX, 1U);
    const uint32_t maskMode = GetUintAttr(attrs, BSA_MASK_MODE_ATTR_INDEX, 3U);
    const bool returnSoftmaxLse = GetBoolAttr(attrs, BSA_RETURN_SOFTMAX_LSE_ATTR_INDEX, false);
    if (layoutKv != "PA_BNSD" || layoutSparseIndices != "B_N_Qb_Kb" || quantMode != 1U ||
        (maskMode != 0U && maskMode != 3U)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t s1Size = 0;
    uint32_t qTokenNum = 0;
    uint32_t dSize = 0;
    const size_t queryDimNum = queryShape.GetDimNum();
    uint32_t layoutQValue = BSA_LAYOUT_Q_BSND_VALUE;
    if (queryDimNum == DIM_NUM_4 && layoutQ == "BSND") {
        if (!GetDimAsU32(queryShape, DIM_B, bSize) || !GetDimAsU32(queryShape, DIM_S, s1Size) ||
            !GetDimAsU32(queryShape, DIM_N, n1Size) || !GetDimAsU32(queryShape, DIM_NUM_3, dSize)) {
            return ge::GRAPH_FAILED;
        }
        qTokenNum = bSize * s1Size;
    } else if (queryDimNum == DIM_NUM_3 && layoutQ == "TND") {
        if (!GetDimAsU32(sparseIndicesShape, DIM_B, bSize) || !GetDimAsU32(queryShape, DIM_B, s1Size) ||
            !GetDimAsU32(queryShape, DIM_S, n1Size) || !GetDimAsU32(queryShape, DIM_N, dSize)) {
            return ge::GRAPH_FAILED;
        }
        qTokenNum = s1Size;
        s1Size = GetPositiveAttr(attrs, BSA_MAX_SEQLEN_Q_ATTR_INDEX, s1Size);
        layoutQValue = BSA_LAYOUT_Q_TND_VALUE;
    } else {
        return ge::GRAPH_FAILED;
    }
    if (dSize != BSA_D_SIZE) {
        return ge::GRAPH_FAILED;
    }

    uint32_t n2Size = 0;
    uint32_t physicalKvBlockSize = 0;
    uint32_t kvDSize = 0;
    if (keyShape.GetDimNum() != DIM_NUM_4 || !GetDimAsU32(keyShape, DIM_BLOCK_SIZE, physicalKvBlockSize) ||
        !GetDimAsU32(keyShape, DIM_N, n2Size) || !GetDimAsU32(keyShape, DIM_SPARSE_QB, kvDSize) ||
        physicalKvBlockSize != kvBlockSize || kvDSize != BSA_D_SIZE) {
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
    uint32_t paBlockNumSum = 0;
    if (!GetDimAsU32(keyShape, DIM_B, paBlockNumSum)) {
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape *blockTableStorageShape = context->GetOptionalInputShape(BSA_BLOCK_TABLE_INDEX);
    if (blockTableStorageShape != nullptr && blockTableStorageShape->GetStorageShape().GetShapeSize() > 0) {
        const gert::Shape &blockTableShape = blockTableStorageShape->GetStorageShape();
        uint32_t blockTableB = 0;
        if (blockTableShape.GetDimNum() != DIM_NUM_2 || !GetDimAsU32(blockTableShape, DIM_B, blockTableB) ||
            !GetDimAsU32(blockTableShape, DIM_BLOCK_TABLE_MAX, maxBlockNumPerBatch) || blockTableB != bSize) {
            return ge::GRAPH_FAILED;
        }
    }

    const uint32_t gS1OuterSize = gSize * qbMax;
    const uint32_t bnCount = bSize * n2Size;
    const uint64_t totalTaskNum = static_cast<uint64_t>(bnCount) * gS1OuterSize;
    if (totalTaskNum == 0U || totalTaskNum > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t maxAicCoreNum = GetAicCoreNum(context);
    const uint32_t usedCoreNum = static_cast<uint32_t>(
        std::min<uint64_t>(static_cast<uint64_t>(maxAicCoreNum), totalTaskNum));
    uint32_t bnStartIdx[BSA_CORE_SPLIT_NUM] = {};
    int64_t sparseStartIdx[BSA_CORE_SPLIT_NUM] = {};
    for (uint32_t boundaryIdx = 0U; boundaryIdx <= usedCoreNum; ++boundaryIdx) {
        const uint64_t taskOffset = totalTaskNum * boundaryIdx / usedCoreNum;
        bnStartIdx[boundaryIdx] = static_cast<uint32_t>(taskOffset / gS1OuterSize);
        sparseStartIdx[boundaryIdx] = static_cast<int64_t>(taskOffset % gS1OuterSize);
    }

    QuantBlockSparseAttentionTilingData tilingData;
    tilingData.baseParams.set_bSize(bSize);
    tilingData.baseParams.set_n1Size(n1Size);
    tilingData.baseParams.set_n2Size(n2Size);
    tilingData.baseParams.set_gSize(gSize);
    tilingData.baseParams.set_s1Size(s1Size);
    tilingData.baseParams.set_s2Size(s2Size);
    tilingData.baseParams.set_dSize(BSA_D_SIZE);
    tilingData.baseParams.set_dSizeV(BSA_D_SIZE);
    tilingData.baseParams.set_qbMax(qbMax);
    tilingData.baseParams.set_kbMax(kbMax);
    tilingData.baseParams.set_sparseCount(sparseCount);
    tilingData.baseParams.set_qBlockSize(qBlockSize);
    tilingData.baseParams.set_kvBlockSize(kvBlockSize);
    tilingData.baseParams.set_maxBlockNumPerBatch(maxBlockNumPerBatch);
    const float softmaxScale = GetFloatAttr(attrs, BSA_SOFTMAX_SCALE_ATTR_INDEX, 1.0F);
    tilingData.baseParams.set_softmaxScale(softmaxScale);
    tilingData.attrParams.set_layoutQ(layoutQValue);
    tilingData.attrParams.set_layoutKv(BSA_LAYOUT_KV_PA_BNSD_VALUE);
    tilingData.attrParams.set_layoutSparseIndices(BSA_LAYOUT_SPARSE_B_N_QB_KB_VALUE);
    tilingData.attrParams.set_quantMode(quantMode);
    tilingData.attrParams.set_maskMode(maskMode);
    tilingData.attrParams.set_returnSoftmaxLse(returnSoftmaxLse ? 1U : 0U);
    tilingData.paParams.set_blockSize(kvBlockSize);
    tilingData.paParams.set_blockTableDim2(maxBlockNumPerBatch);
    tilingData.paParams.set_paBlockNumSum(paBlockNumSum);
    tilingData.paParams.set_paLayoutType(BSA_PA_LAYOUT_TYPE_BNBD);
    tilingData.sparseParams.set_gS1OuterSize(gS1OuterSize);
    tilingData.sparseParams.set_sparseSeqLenStride(qbMax);
    tilingData.sparseParams.set_sparseIndicesStride(sparseCount);

    const uint8_t compatMaskMode = (maskMode == 3U) ? BSA_COMPAT_MASK_LEFT_UP_CAUSAL : BSA_COMPAT_MASK_NONE;
    auto &inputParams = tilingData.inputParamsRegbase;
    inputParams.set_bSize(bSize);
    inputParams.set_t1Size(qTokenNum);
    inputParams.set_t2Size(s2Size);
    inputParams.set_n2Size(n2Size);
    inputParams.set_gSize(gSize);
    inputParams.set_s1Size(s1Size);
    inputParams.set_s2Size(s2Size);
    inputParams.set_alignedS2(kbMax * kvBlockSize);
    inputParams.set_dSize(BSA_D_SIZE);
    inputParams.set_dSizeV(BSA_D_SIZE);
    inputParams.set_dSizeRope(0);
    inputParams.set_keepProb(1.0F);
    inputParams.set_scaleValue(softmaxScale);
    inputParams.set_preTokens(std::numeric_limits<int32_t>::max());
    inputParams.set_nextTokens(0);
    inputParams.set_pseS1Size(0);
    inputParams.set_pseS2Size(0);
    inputParams.set_pseBSize(0);
    inputParams.set_bandIndex(0);
    inputParams.set_layoutType(layoutQValue == BSA_LAYOUT_Q_TND_VALUE ? BSA_LAYOUT_TYPE_TND : BSA_LAYOUT_TYPE_BSH);
    inputParams.set_pseShapeType(0);
    inputParams.set_attenMaskShapeType(2);
    inputParams.set_attenMaskDataType(1);
    inputParams.set_attenMaskCompressMode(compatMaskMode);
    inputParams.set_implMode(0);
    inputParams.set_sparseType(0);
    inputParams.set_needDropMaskOp(0);
    inputParams.set_dropMaskOuter(0);
    inputParams.set_pseEncodeType(0);
    inputParams.set_tndSoftmaxOut(0);
    inputParams.set_remain(0);
    inputParams.set_attenMaskS2Size(s2Size);
    inputParams.set_pseType(0);
    inputParams.set_rsv1(0);
    inputParams.set_qStartIdx(0);
    inputParams.set_kvStartIdx(0);
    inputParams.set_s1SparseValidSize(s1Size);
    inputParams.set_s2SparseValidSize(s2Size);
    inputParams.set_seed(0);
    inputParams.set_offset(0);
    inputParams.set_keepProbUint8(255);
    inputParams.set_pseAlibiBaseS1(0);
    inputParams.set_pseAlibiBaseS2(0);
    inputParams.set_deqScaleFlag(1);
    inputParams.set_deqScale2Flag(1);
    inputParams.set_isActualSeqLengthsNull(1);
    inputParams.set_isActualSeqLengthsKVNull(1);
    inputParams.set_actualSeqLengthsSize(bSize);
    inputParams.set_actualSeqLengthsKVSize(bSize);
    inputParams.set_isKvContinuous(1);
    inputParams.set_fromFused(0);
    inputParams.set_isBSNDOut(0);
    inputParams.set_transposeLayout(0);
    inputParams.set_isGqa(gSize > 1U ? 1 : 0);
    inputParams.set_isSoftMaxLseEnable(returnSoftmaxLse ? 1 : 0);
    inputParams.set_isActualSharedPrefixLenNull(1);
    inputParams.set_isQHasLeftPadding(0);
    inputParams.set_isKVHasLeftPadding(0);
    inputParams.set_ropeHeadSize(0);
    inputParams.set_prefixSeqInnerSize(0);
    inputParams.set_headNumRatio(1U);
    inputParams.set_blockSize(kvBlockSize);
    inputParams.set_blockTableDim2(maxBlockNumPerBatch);
    inputParams.set_paBlockNumSum(paBlockNumSum);
    inputParams.set_attenMaskS1Size(s1Size);
    inputParams.set_paLayoutType(BSA_PA_LAYOUT_TYPE_BNBD);
    inputParams.set_isRowInvalid(0);
    inputParams.set_isPostQuantPerChnl(0);
    inputParams.set_isPostQuantBF16(0);
    inputParams.set_antiquantPageAttentionFlag(0);
    inputParams.set_antiquantPerTensorFlag(0);
    inputParams.set_antiquantPerHeadFlag(0);
    inputParams.set_antiquantParaSeqSize(0);

    auto &multiCoreParams = tilingData.multiCoreParamsRegbase;
    multiCoreParams.set_coreNum(static_cast<int32_t>(usedCoreNum));
    multiCoreParams.set_totalSize(static_cast<int64_t>(totalTaskNum));
    multiCoreParams.set_s1OuterSize(qbMax);
    multiCoreParams.set_splitFactorSize(0);
    multiCoreParams.set_splitFactorTailSize(0);
    multiCoreParams.set_bnStartIdx(bnStartIdx);
    multiCoreParams.set_sparseStartIdx(sparseStartIdx);
    multiCoreParams.set_firstFullLoadS1OuterIdx(0);
    multiCoreParams.set_splitCoreMode(0);
    uint8_t reserve[3] = {};
    multiCoreParams.set_reserve(reserve);

    tilingData.dropmaskParamsRegbase.set_multiCoreFactorSize(0);
    tilingData.dropmaskParamsRegbase.set_baseUbCalSize(0);
    tilingData.dropmaskParamsRegbase.set_multiCoreTotalSize(0);
    tilingData.dropmaskParamsRegbase.set_shapeTotalSize(0);
    tilingData.dropmaskParamsRegbase.set_dropMaskAddrOffset(0);

    const int64_t totalOutputSize = static_cast<int64_t>(qTokenNum) * n1Size * BSA_D_SIZE;
    const int64_t totalSoftmaxLseSize = static_cast<int64_t>(qTokenNum) * n1Size;
    tilingData.initOutputParams.set_singleCoreSize(0);
    tilingData.initOutputParams.set_needInit(0);
    tilingData.initOutputParams.set_isOneN(n1Size == 1U ? 1 : 0);
    uint8_t rsvd[2] = {};
    tilingData.initOutputParams.set_rsvd(rsvd);
    tilingData.initOutputParams.set_totalOutputSize(totalOutputSize);
    tilingData.initOutputParams.set_totalSoftMaxLseOutputSize(totalSoftmaxLseSize);

    context->SetBlockDim(usedCoreNum);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        BSA_DTYPE_FP8_E4M3FN, layoutQValue,
        BSA_KV_LAYOUT_PA_ND, maskMode, returnSoftmaxLse ? 1U : 0U);
    context->SetTilingKey(tilingKey);
    context->SetScheduleMode(1);
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
graphStatus TilingPrepareQuantBlockSparseAttention(gert::TilingParseContext *context)
{
    // TODO: 待实现 - 填充 compile info。
    return context == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class QuantBlockSparseAttentionTiling {
public:
    static ge::graphStatus Tiling(gert::TilingContext *context)
    {
        return optiling::TilingQuantBlockSparseAttention(context);
    }
};
} // namespace ops

namespace optiling {
IMPL_OP_OPTILING(QuantBlockSparseAttention)
    .Tiling(TilingQuantBlockSparseAttention);
} // namespace optiling
