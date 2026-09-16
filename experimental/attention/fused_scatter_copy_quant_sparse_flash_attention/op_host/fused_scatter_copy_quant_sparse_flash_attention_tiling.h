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
 * \file fused_scatter_copy_quant_sparse_flash_attention_tiling.h
 * \brief
 */

#ifndef FUSED_SCATTER_COPY_QUANT_SPARSE_FLASH_ATTENTION_TILING_H
#define FUSED_SCATTER_COPY_QUANT_SPARSE_FLASH_ATTENTION_TILING_H

#include <cstdint>
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FusedScatterCopyQuantSparseFlashAttentionBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, seqSize)
TILING_DATA_FIELD_DEF(uint32_t, qSeqSize)
TILING_DATA_FIELD_DEF(uint32_t, nNumOfQInOneGroup)
TILING_DATA_FIELD_DEF(uint32_t, sparseBlockCount)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(uint32_t, outputLayout)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, isActualLenDimsNull)
TILING_DATA_FIELD_DEF(uint32_t, isActualLenDimsKVNull)
TILING_DATA_FIELD_DEF(uint32_t, dSizeVInput)
TILING_DATA_FIELD_DEF(float, scaleValue)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedScatterCopyQuantSparseFlashAttentionBaseParamsOp,
    FusedScatterCopyQuantSparseFlashAttentionBaseParams)

BEGIN_TILING_DATA_DEF(FusedScatterCopyQuantSparseFlashAttentionSingleCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedScatterCopyQuantSparseFlashAttentionSingleCoreParamsOp,
    FusedScatterCopyQuantSparseFlashAttentionSingleCoreParams)

BEGIN_TILING_DATA_DEF(FusedScatterCopyQuantSparseFlashAttentionTilingData)
TILING_DATA_FIELD_DEF_STRUCT(FusedScatterCopyQuantSparseFlashAttentionBaseParams, baseParams)
TILING_DATA_FIELD_DEF_STRUCT(FusedScatterCopyQuantSparseFlashAttentionSingleCoreParams, singleCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, dramMaxBlocks)
TILING_DATA_FIELD_DEF(uint32_t, hbmPhysicalBlocks)
TILING_DATA_FIELD_DEF(uint32_t, dramPhysicalBlocks)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedScatterCopyQuantSparseFlashAttention,
    FusedScatterCopyQuantSparseFlashAttentionTilingData)
} // namespace optiling

#endif
