/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef L0_SPARSE_FLASH_MLA_METADATA_AICPU_H
#define L0_SPARSE_FLASH_MLA_METADATA_AICPU_H

#include "opdev/op_executor.h"

namespace l0op {
/**
 * @brief SparseFlashMlaMetadata L0 kernel interface
 * Function: Generates load-balanced tiling metadata for aclnnSparseFlashMla on AICPU.
 *           Parses batch sequence lengths and attention config, computes valid S2 ranges per S1G block,
 *           performs core assignment, and writes FA/FD metadata for downstream FlashAttention kernels.
 *
 * @param [in] cuSeqLensQOptional: Prefix-sum of valid q token counts per batch, shape is (B+1,), dtype is INT32
 * @param [in] cuSeqLensOriKvOptional: Prefix-sum of valid oriKv token counts per batch, shape is (B+1,), dtype is INT32
 * @param [in] cuSeqLensCmpKvOptional: Prefix-sum of valid cmpKv token counts per batch, shape is (B+1,), dtype is INT32
 * @param [in] sequsedQOptional: Actual q token counts per batch, shape is (B,), dtype is INT32
 * @param [in] sequsedOriKvOptional: Actual oriKv token counts per batch, shape is (B,), dtype is INT32
 * @param [in] numHeadsQ: Number of query heads (N1)
 * @param [in] numHeadsKv: Number of kv heads (N2)
 * @param [in] headDim: Head dimension (D)
 * @param [in] batchSize: Batch size (B), 0 means inferred from cuSeqLensQ when layoutQ is TND
 * @param [in] maxSeqlenQ: Maximum valid q sequence length across batches
 * @param [in] maxSeqlenKv: Maximum valid oriKv sequence length across batches
 * @param [in] oriTopK: Sparse token count selected from oriKv
 * @param [in] cmpTopK: Sparse token count selected from cmpKv
 * @param [in] cmpRatio: Compression ratio of oriKv
 * @param [in] oriMaskMode: Mask mode for q and oriKv
 * @param [in] cmpMaskMode: Mask mode for q and cmpKv
 * @param [in] oriWinLeft: Sliding-window left extension token count
 * @param [in] oriWinRight: Sliding-window right extension token count
 * @param [in] layoutQOptional: Layout string of q, e.g. "BSND" or "TND"
 * @param [in] layoutKvOptional: Layout string of kv, e.g. "PA_BNBD", "BSND" or "TND"
 * @param [in] hasOriKv: Whether oriKv is provided
 * @param [in] hasCmpKv: Whether cmpKv is provided
 * @param [in] socVersion: SoC version string for core capability query
 * @param [in] aicCoreNum: Number of AIC cores used for FA metadata generation
 * @param [in] aivCoreNum: Number of AIV cores used for FD metadata generation
 * @param [out] metadata: Tiling metadata output for SparseFlashMla, shape is (1024,), dtype is INT32
 * @param [in] executor: Op executor
 * @return aclTensor*: Output tensor metadata
 */
const aclTensor* SparseFlashMlaMetadata(
    const aclTensor* cuSeqLensQOptional, const aclTensor* cuSeqLensOriKvOptional,
    const aclTensor* cuSeqLensCmpKvOptional, const aclTensor* sequsedQOptional, const aclTensor* sequsedOriKvOptional,
    int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
    int64_t oriTopK, int64_t cmpTopK, int64_t cmpRatio, int64_t oriMaskMode, int64_t cmpMaskMode, int64_t oriWinLeft,
    int64_t oriWinRight, char *layoutQOptional, char *layoutKvOptional, bool hasOriKv, bool hasCmpKv,
    const char *socVersion, int64_t aicCoreNum, int64_t aivCoreNum, const aclTensor* metadata, aclOpExecutor* executor);
} // namespace l0op

#endif
