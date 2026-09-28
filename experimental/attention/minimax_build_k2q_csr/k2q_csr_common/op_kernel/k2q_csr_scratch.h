/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * scratch 布局解析（各阶段入口共用）。
 *
 * scratch(int32):
 *   row_map[B*maxKv] | token_batch[T] | tile_counts[G*H*R] | abs_base[G*H*R] | row_counts[H*R]
 *
 * 对齐 CUDA：Hist 写 tile_counts + atomic 累加 row_counts；PR 只扫 row_counts。
 */
#ifndef K2Q_CSR_SCRATCH_H
#define K2Q_CSR_SCRATCH_H

#include "k2q_csr_tiling.h"
#include "kernel_operator.h"

namespace K2qCsrScratch {

__aicore__ inline int64_t TileElems(const K2qCsrTilingData *tiling)
{
    int64_t G = tiling->numGroups > 0 ? tiling->numGroups : 1;
    int64_t H = tiling->H > 0 ? tiling->H : 1;
    int64_t R = tiling->totalRows > 0 ? tiling->totalRows : 1;
    int64_t n = G * H * R;
    return n > 0 ? n : 1;
}

__aicore__ inline void Split(GM_ADDR scratch, const K2qCsrTilingData *tiling, GM_ADDR &rowMap, GM_ADDR &tokenBatch,
                             GM_ADDR &histScratch)
{
    int64_t rowMapBytes = tiling->rowMapElems * static_cast<int64_t>(sizeof(int32_t));
    int64_t tokenBytes = tiling->T * static_cast<int64_t>(sizeof(int32_t));
    rowMap = scratch;
    tokenBatch = scratch + rowMapBytes;
    histScratch = scratch + rowMapBytes + tokenBytes;
}

/** histScratch 内：tile_counts | abs_base | row_counts */
__aicore__ inline GM_ADDR RowCountsAddr(GM_ADDR histScratch, const K2qCsrTilingData *tiling)
{
    int64_t tileBytes = TileElems(tiling) * static_cast<int64_t>(sizeof(int32_t));
    return histScratch + tileBytes * 2;
}

} // namespace K2qCsrScratch

#endif
