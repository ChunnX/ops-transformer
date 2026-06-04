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
 * \file quant_block_sparse_attention_tiling.h
 * \brief TODO: 待实现
 */

#ifndef QUANT_BLOCK_SPARSE_ATTENTION_TILING_H_
#define QUANT_BLOCK_SPARSE_ATTENTION_TILING_H_

#include <cstdint>
#include <limits>
#include <string>

#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>
#include "register/tilingdata_base.h"
#include "quant_block_sparse_attention_info_parser.h"

namespace optiling {
constexpr uint32_t BSA_MAX_CORE_NUM = 48U;
constexpr uint32_t BSA_CORE_SPLIT_NUM = BSA_MAX_CORE_NUM + 1U;
constexpr uint32_t BSA_BLOCK_SIZE = 128U;
constexpr uint32_t BSA_D_SIZE = 128U;

inline uint32_t BSACeilDiv(uint32_t value, uint32_t divisor)
{
    return divisor == 0U ? 0U : (value + divisor - 1U) / divisor;
}

inline bool BSAGetDimAsU32(const gert::Shape &shape, size_t dimIndex, uint32_t &value)
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

inline uint32_t BSAGetPositiveAttr(const gert::RuntimeAttrs *attrs, uint32_t index, uint32_t defaultValue)
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

inline uint32_t BSAGetUintAttr(const gert::RuntimeAttrs *attrs, uint32_t index, uint32_t defaultValue)
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

inline float BSAGetFloatAttr(const gert::RuntimeAttrs *attrs, uint32_t index, float defaultValue)
{
    const float *attrPtr = attrs->GetAttrPointer<float>(index);
    return attrPtr == nullptr ? defaultValue : *attrPtr;
}

inline bool BSAGetBoolAttr(const gert::RuntimeAttrs *attrs, uint32_t index, bool defaultValue)
{
    const bool *attrPtr = attrs->GetAttrPointer<bool>(index);
    return attrPtr == nullptr ? defaultValue : *attrPtr;
}

inline std::string BSAGetStringAttr(const gert::RuntimeAttrs *attrs, uint32_t index, const char *defaultValue)
{
    const char *attrPtr = attrs->GetAttrPointer<char>(index);
    return attrPtr == nullptr ? std::string(defaultValue) : std::string(attrPtr);
}

BEGIN_TILING_DATA_DEF(QuantBlockSparseAttentionBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, n1Size)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, dSize)
TILING_DATA_FIELD_DEF(uint32_t, dSizeV)
TILING_DATA_FIELD_DEF(uint32_t, qbMax)
TILING_DATA_FIELD_DEF(uint32_t, kbMax)
TILING_DATA_FIELD_DEF(uint32_t, sparseCount)
TILING_DATA_FIELD_DEF(uint32_t, qBlockSize)
TILING_DATA_FIELD_DEF(uint32_t, kvBlockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, paBlockStride)
TILING_DATA_FIELD_DEF(float, softmaxScale)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(QuantBlockSparseAttentionBaseParamsOp, QuantBlockSparseAttentionBaseParams)

BEGIN_TILING_DATA_DEF(QuantBlockSparseAttentionAttrParams)
TILING_DATA_FIELD_DEF(uint32_t, layoutQ)
TILING_DATA_FIELD_DEF(uint32_t, layoutKv)
TILING_DATA_FIELD_DEF(uint32_t, layoutSparseIndices)
TILING_DATA_FIELD_DEF(uint32_t, quantMode)
TILING_DATA_FIELD_DEF(uint32_t, maskMode)
TILING_DATA_FIELD_DEF(uint32_t, returnSoftmaxLse)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(QuantBlockSparseAttentionAttrParamsOp, QuantBlockSparseAttentionAttrParams)

BEGIN_TILING_DATA_DEF(QuantBlockSparseAttentionPaParams)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, blockTableDim2)
TILING_DATA_FIELD_DEF(uint32_t, paBlockNumSum)
TILING_DATA_FIELD_DEF(uint32_t, paLayoutType)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(QuantBlockSparseAttentionPaParamsOp, QuantBlockSparseAttentionPaParams)

BEGIN_TILING_DATA_DEF(QuantBlockSparseAttentionSparseParams)
TILING_DATA_FIELD_DEF(uint32_t, gS1OuterSize)
TILING_DATA_FIELD_DEF(uint32_t, sparseSeqLenStride)
TILING_DATA_FIELD_DEF(uint32_t, sparseIndicesStride)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(QuantBlockSparseAttentionSparseParamsOp, QuantBlockSparseAttentionSparseParams)

BEGIN_TILING_DATA_DEF(QuantBlockSparseAttentionInputParamsRegbase)
TILING_DATA_FIELD_DEF(int64_t, bSize)
TILING_DATA_FIELD_DEF(int64_t, t1Size)
TILING_DATA_FIELD_DEF(int64_t, t2Size)
TILING_DATA_FIELD_DEF(int64_t, n2Size)
TILING_DATA_FIELD_DEF(int64_t, gSize)
TILING_DATA_FIELD_DEF(int64_t, s1Size)
TILING_DATA_FIELD_DEF(int64_t, s2Size)
TILING_DATA_FIELD_DEF(int64_t, alignedS2)
TILING_DATA_FIELD_DEF(int64_t, dSize)
TILING_DATA_FIELD_DEF(int64_t, dSizeV)
TILING_DATA_FIELD_DEF(int64_t, dSizeRope)
TILING_DATA_FIELD_DEF(float, keepProb)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(int64_t, preTokens)
TILING_DATA_FIELD_DEF(int64_t, nextTokens)
TILING_DATA_FIELD_DEF(int64_t, pseS1Size)
TILING_DATA_FIELD_DEF(int64_t, pseS2Size)
TILING_DATA_FIELD_DEF(uint32_t, pseBSize)
TILING_DATA_FIELD_DEF(uint32_t, bandIndex)
TILING_DATA_FIELD_DEF(uint8_t, layoutType)
TILING_DATA_FIELD_DEF(uint8_t, pseShapeType)
TILING_DATA_FIELD_DEF(uint8_t, attenMaskShapeType)
TILING_DATA_FIELD_DEF(uint8_t, attenMaskDataType)
TILING_DATA_FIELD_DEF(uint8_t, attenMaskCompressMode)
TILING_DATA_FIELD_DEF(uint8_t, implMode)
TILING_DATA_FIELD_DEF(uint8_t, sparseType)
TILING_DATA_FIELD_DEF(uint8_t, needDropMaskOp)
TILING_DATA_FIELD_DEF(uint8_t, dropMaskOuter)
TILING_DATA_FIELD_DEF(uint8_t, pseEncodeType)
TILING_DATA_FIELD_DEF(uint8_t, tndSoftmaxOut)
TILING_DATA_FIELD_DEF(uint8_t, remain)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskS2Size)
TILING_DATA_FIELD_DEF(uint32_t, pseType)
TILING_DATA_FIELD_DEF(uint32_t, rsv1)
TILING_DATA_FIELD_DEF(int64_t, qStartIdx)
TILING_DATA_FIELD_DEF(int64_t, kvStartIdx)
TILING_DATA_FIELD_DEF(int64_t, s1SparseValidSize)
TILING_DATA_FIELD_DEF(int64_t, s2SparseValidSize)
TILING_DATA_FIELD_DEF(int64_t, seed)
TILING_DATA_FIELD_DEF(int64_t, offset)
TILING_DATA_FIELD_DEF(int64_t, keepProbUint8)
TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS1)
TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS2)
TILING_DATA_FIELD_DEF(uint8_t, deqScaleFlag)
TILING_DATA_FIELD_DEF(uint8_t, deqScale2Flag)
TILING_DATA_FIELD_DEF(uint8_t, isActualSeqLengthsNull)
TILING_DATA_FIELD_DEF(uint8_t, isActualSeqLengthsKVNull)
TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsSize)
TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsKVSize)
TILING_DATA_FIELD_DEF(uint8_t, isKvContinuous)
TILING_DATA_FIELD_DEF(uint8_t, fromFused)
TILING_DATA_FIELD_DEF(uint8_t, isBSNDOut)
TILING_DATA_FIELD_DEF(uint8_t, transposeLayout)
TILING_DATA_FIELD_DEF(uint8_t, isGqa)
TILING_DATA_FIELD_DEF(uint8_t, isSoftMaxLseEnable)
TILING_DATA_FIELD_DEF(uint8_t, isActualSharedPrefixLenNull)
TILING_DATA_FIELD_DEF(uint8_t, isQHasLeftPadding)
TILING_DATA_FIELD_DEF(uint8_t, isKVHasLeftPadding)
TILING_DATA_FIELD_DEF(uint32_t, ropeHeadSize)
TILING_DATA_FIELD_DEF(uint32_t, prefixSeqInnerSize)
TILING_DATA_FIELD_DEF(uint32_t, headNumRatio)
TILING_DATA_FIELD_DEF(int32_t, blockSize)
TILING_DATA_FIELD_DEF(int32_t, blockTableDim2)
TILING_DATA_FIELD_DEF(int32_t, paBlockNumSum)
TILING_DATA_FIELD_DEF(uint32_t, paBlockStride)
TILING_DATA_FIELD_DEF(int32_t, attenMaskS1Size)
TILING_DATA_FIELD_DEF(uint8_t, paLayoutType)
TILING_DATA_FIELD_DEF(uint8_t, isRowInvalid)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(QuantBlockSparseAttentionInputParamsRegbaseOp, QuantBlockSparseAttentionInputParamsRegbase)

BEGIN_TILING_DATA_DEF(QuantBlockSparseAttentionMultiCoreParams)
TILING_DATA_FIELD_DEF(int32_t, coreNum)
TILING_DATA_FIELD_DEF(int64_t, totalSize)
TILING_DATA_FIELD_DEF(int64_t, s1OuterSize)
TILING_DATA_FIELD_DEF(int64_t, splitFactorSize)
TILING_DATA_FIELD_DEF(int64_t, splitFactorTailSize)
TILING_DATA_FIELD_DEF_ARR(uint32_t, BSA_CORE_SPLIT_NUM, bnStartIdx)
TILING_DATA_FIELD_DEF_ARR(int64_t, BSA_CORE_SPLIT_NUM, sparseStartIdx)
TILING_DATA_FIELD_DEF(int64_t, firstFullLoadS1OuterIdx)
TILING_DATA_FIELD_DEF(uint8_t, splitCoreMode)
TILING_DATA_FIELD_DEF_ARR(uint8_t, 3, reserve)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(QuantBlockSparseAttentionMultiCoreParamsOp, QuantBlockSparseAttentionMultiCoreParams)

BEGIN_TILING_DATA_DEF(QuantBlockSparseAttentionDropmaskParams)
TILING_DATA_FIELD_DEF(int32_t, multiCoreFactorSize)
TILING_DATA_FIELD_DEF(int32_t, baseUbCalSize)
TILING_DATA_FIELD_DEF(int64_t, multiCoreTotalSize)
TILING_DATA_FIELD_DEF(int64_t, shapeTotalSize)
TILING_DATA_FIELD_DEF(int64_t, dropMaskAddrOffset)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(QuantBlockSparseAttentionDropmaskParamsOp, QuantBlockSparseAttentionDropmaskParams)

BEGIN_TILING_DATA_DEF(QuantBlockSparseAttentionInitOutputParams)
TILING_DATA_FIELD_DEF(uint32_t, singleCoreSize)
TILING_DATA_FIELD_DEF(uint8_t, needInit)
TILING_DATA_FIELD_DEF(uint8_t, isOneN)
TILING_DATA_FIELD_DEF_ARR(uint8_t, 2, rsvd)
TILING_DATA_FIELD_DEF(int64_t, totalOutputSize)
TILING_DATA_FIELD_DEF(int64_t, totalSoftMaxLseOutputSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(QuantBlockSparseAttentionInitOutputParamsOp, QuantBlockSparseAttentionInitOutputParams)

BEGIN_TILING_DATA_DEF(QuantBlockSparseAttentionTilingData)
TILING_DATA_FIELD_DEF_STRUCT(QuantBlockSparseAttentionBaseParams, baseParams)
TILING_DATA_FIELD_DEF_STRUCT(QuantBlockSparseAttentionAttrParams, attrParams)
TILING_DATA_FIELD_DEF_STRUCT(QuantBlockSparseAttentionPaParams, paParams)
TILING_DATA_FIELD_DEF_STRUCT(QuantBlockSparseAttentionSparseParams, sparseParams)
TILING_DATA_FIELD_DEF_STRUCT(QuantBlockSparseAttentionInputParamsRegbase, inputParamsRegbase)
TILING_DATA_FIELD_DEF_STRUCT(QuantBlockSparseAttentionMultiCoreParams, multiCoreParamsRegbase)
TILING_DATA_FIELD_DEF_STRUCT(QuantBlockSparseAttentionDropmaskParams, dropmaskParamsRegbase)
TILING_DATA_FIELD_DEF_STRUCT(QuantBlockSparseAttentionInitOutputParams, initOutputParams)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(QuantBlockSparseAttention, QuantBlockSparseAttentionTilingData)

ge::graphStatus TilingQuantBlockSparseAttention(gert::TilingContext *context);

class QuantBlockSparseAttentionTiling {
public:
    explicit QuantBlockSparseAttentionTiling(gert::TilingContext *context);
    ~QuantBlockSparseAttentionTiling() = default;

    ge::graphStatus DoOpTiling(QuantBlockSparseAttentionTilingInfo *tilingInfo);

private:
    void FillBaseParams();
    void FillAttrParams();
    void FillPaParams();
    void FillSparseParams();
    void FillInputParams();
    void FillMultiCoreParams();
    void FillDropmaskParams();
    void FillInitOutputParams();
    void CalcTilingKey();
    void CalcWorkspaceSize();

    gert::TilingContext *context_ = nullptr;
    QuantBlockSparseAttentionTilingInfo *tilingInfo_ = nullptr;
    QuantBlockSparseAttentionTilingData tilingData_;
    uint32_t usedCoreNum_ = 0;
    uint64_t totalTaskNum_ = 0;
    uint64_t tilingKey_ = 0;
    uint64_t workspaceSize_ = 0;
};

} // namespace optiling

#endif // QUANT_BLOCK_SPARSE_ATTENTION_TILING_H_
