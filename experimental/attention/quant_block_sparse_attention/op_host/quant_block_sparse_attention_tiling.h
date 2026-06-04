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

#include <exe_graph/runtime/tiling_context.h>
#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint32_t BSA_MAX_CORE_NUM = 48U;
constexpr uint32_t BSA_CORE_SPLIT_NUM = BSA_MAX_CORE_NUM + 1U;
constexpr uint32_t BSA_BLOCK_SIZE = 128U;
constexpr uint32_t BSA_D_SIZE = 128U;

// Inputs Index
constexpr uint32_t BSA_QUERY_INDEX = 0U;
constexpr uint32_t BSA_KEY_INDEX = 1U;
constexpr uint32_t BSA_VALUE_INDEX = 2U;
constexpr uint32_t BSA_Q_DESCALE_INDEX = 3U;
constexpr uint32_t BSA_K_DESCALE_INDEX = 4U;
constexpr uint32_t BSA_V_DESCALE_INDEX = 5U;
constexpr uint32_t BSA_P_SCALE_INDEX = 6U;
constexpr uint32_t BSA_CU_SEQLENS_Q_INDEX = 7U;
constexpr uint32_t BSA_CU_SEQLENS_KV_INDEX = 8U;
constexpr uint32_t BSA_SEQUSED_Q_INDEX = 9U;
constexpr uint32_t BSA_SEQUSED_KV_INDEX = 10U;
constexpr uint32_t BSA_SPARSE_INDICES_INDEX = 11U;
constexpr uint32_t BSA_SPARSE_SEQ_LEN_INDEX = 12U;
constexpr uint32_t BSA_BLOCK_TABLE_INDEX = 13U;
constexpr uint32_t BSA_ATTEN_MASK_INDEX = 14U;
constexpr uint32_t BSA_METADATA_INDEX = 15U;

// Outputs Index
constexpr uint32_t BSA_ATTENTION_OUT_INDEX = 0U;
constexpr uint32_t BSA_SOFTMAX_LSE_INDEX = 1U;

// Attributes Index
constexpr uint32_t BSA_MAX_SEQLEN_Q_ATTR_INDEX = 0U;
constexpr uint32_t BSA_MAX_SEQLEN_KV_ATTR_INDEX = 1U;
constexpr uint32_t BSA_SOFTMAX_SCALE_ATTR_INDEX = 2U;
constexpr uint32_t BSA_SPARSE_Q_BLOCK_SIZE_ATTR_INDEX = 3U;
constexpr uint32_t BSA_SPARSE_KV_BLOCK_SIZE_ATTR_INDEX = 4U;
constexpr uint32_t BSA_LAYOUT_KV_ATTR_INDEX = 5U;
constexpr uint32_t BSA_LAYOUT_Q_ATTR_INDEX = 6U;
constexpr uint32_t BSA_LAYOUT_SPARSE_INDICES_ATTR_INDEX = 7U;
constexpr uint32_t BSA_QUANT_MODE_ATTR_INDEX = 8U;
constexpr uint32_t BSA_MASK_MODE_ATTR_INDEX = 9U;
constexpr uint32_t BSA_RETURN_SOFTMAX_LSE_ATTR_INDEX = 10U;

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
TILING_DATA_FIELD_DEF(int32_t, attenMaskS1Size)
TILING_DATA_FIELD_DEF(uint8_t, paLayoutType)
TILING_DATA_FIELD_DEF(uint8_t, isRowInvalid)
TILING_DATA_FIELD_DEF(uint8_t, isPostQuantPerChnl)
TILING_DATA_FIELD_DEF(uint8_t, isPostQuantBF16)
TILING_DATA_FIELD_DEF(uint8_t, antiquantPageAttentionFlag)
TILING_DATA_FIELD_DEF(uint16_t, antiquantPerTensorFlag)
TILING_DATA_FIELD_DEF(uint16_t, antiquantPerHeadFlag)
TILING_DATA_FIELD_DEF(uint32_t, antiquantParaSeqSize)
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
} // namespace optiling

#endif // QUANT_BLOCK_SPARSE_ATTENTION_TILING_H_
