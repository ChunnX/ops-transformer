/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_GATED_DELTA_RULE_VECTOR_COPY_H
#define CHUNK_GATED_DELTA_RULE_VECTOR_COPY_H

#include "chunk_gated_delta_rule_layout.h"

namespace ChunkGatedDeltaRule {

using namespace AscendC;

// GM [rows,cols]（可跨 token/head）-> UB [rows,DIM]；步长以元素传入，内部换算为字节/块。
// DMA 仅补齐最后一个 32B 块，完整 tile 的尾部仍由各计算路径显式清零或用 mask 排除。
template <typename T>
__aicore__ inline void CopyRowsGmToUb(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t rows,
                                    uint32_t cols, uint32_t srcRowStride)
{
    constexpr uint32_t elementsPerBlock = BLOCK_SIZE / sizeof(T);
    const uint32_t alignedCols = MmAlign(cols, elementsPerBlock);
    DataCopyExtParams copy{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(T)),
                           static_cast<uint32_t>((srcRowStride - cols) * sizeof(T)),
                           static_cast<int64_t>((FusedLayout::DIM - alignedCols) / elementsPerBlock), 0};
    DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(alignedCols - cols), static_cast<T>(0)};
    DataCopyPad(dst, src, copy, pad);
}

// UB [validRows,DIM] -> 连续 GM [validRows,validCols]；只写回有效特征列。
template <typename T>
__aicore__ inline void CopyStateRowsUbToGm(GlobalTensor<T> dst, LocalTensor<T> src,
                                         uint32_t validRows, uint32_t validCols)
{
    if (validRows == 0) {
        return;
    }
    constexpr uint32_t elementsPerBlock = BLOCK_SIZE / sizeof(T);
    const uint32_t alignedCols = MmAlign(validCols, elementsPerBlock);
    DataCopyExtParams params{static_cast<uint16_t>(validRows), static_cast<uint32_t>(validCols * sizeof(T)),
                             static_cast<uint32_t>((FusedLayout::DIM - alignedCols) / elementsPerBlock), 0, 0};
    DataCopyPad(dst, src, params);
}

// VF 使用 UB 指针；LocalTensor 仍负责描述物理地址空间。
template <typename T>
__aicore__ inline __ubuf__ T *UbAddress(LocalTensor<T> tensor)
{
    return reinterpret_cast<__ubuf__ T *>(tensor.GetPhyAddr());
}

} // namespace ChunkGatedDeltaRule

#endif // CHUNK_GATED_DELTA_RULE_VECTOR_COPY_H
