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

#include <exe_graph/runtime/tiling_context.h>
#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint32_t BSA_MAX_CORE_NUM = 64U;
constexpr uint32_t BSA_BLOCK_SIZE = 128U;

// Inputs Index
constexpr uint32_t BSA_QUERY_INDEX = 0U;
constexpr uint32_t BSA_KEY_INDEX = 1U;
constexpr uint32_t BSA_VALUE_INDEX = 2U;
constexpr uint32_t BSA_Q_SCALE_INDEX = 3U;
constexpr uint32_t BSA_K_SCALE_INDEX = 4U;
constexpr uint32_t BSA_V_SCALE_INDEX = 5U;
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
constexpr uint32_t BSA_Q_BLOCK_SIZE_ATTR_INDEX = 3U;
constexpr uint32_t BSA_KV_BLOCK_SIZE_ATTR_INDEX = 4U;
constexpr uint32_t BSA_KV_LAYOUT_ATTR_INDEX = 5U;
constexpr uint32_t BSA_SPARSE_INDICES_LAYOUT_ATTR_INDEX = 6U;
constexpr uint32_t BSA_QUANT_MODE_ATTR_INDEX = 7U;
constexpr uint32_t BSA_MASK_MODE_ATTR_INDEX = 8U;
constexpr uint32_t BSA_RETURN_SOFTMAX_LSE_ATTR_INDEX = 9U;

BEGIN_TILING_DATA_DEF(QuantBlockSparseAttentionBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, n1Size)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, qbMax)
TILING_DATA_FIELD_DEF(uint32_t, kbMax)
TILING_DATA_FIELD_DEF(uint32_t, sparseCount)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskCompressMode)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(float, softmaxScale)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(QuantBlockSparseAttentionBaseParamsOp, QuantBlockSparseAttentionBaseParams)

BEGIN_TILING_DATA_DEF(QuantBlockSparseAttentionTilingData)
TILING_DATA_FIELD_DEF_STRUCT(QuantBlockSparseAttentionBaseParams, baseParams)
TILING_DATA_FIELD_DEF_ARR(uint32_t, BSA_MAX_CORE_NUM, coreStart)
TILING_DATA_FIELD_DEF_ARR(uint32_t, BSA_MAX_CORE_NUM, coreEnd)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(QuantBlockSparseAttention, QuantBlockSparseAttentionTilingData)

ge::graphStatus TilingQuantBlockSparseAttention(gert::TilingContext *context);
} // namespace optiling

#endif // QUANT_BLOCK_SPARSE_ATTENTION_TILING_H_
