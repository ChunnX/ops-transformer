/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef BSA_EFF_ROWS_TILE_HPP
#define BSA_EFF_ROWS_TILE_HPP

#include <cstdint>

namespace NpuArch::Gemm::Block {

__aicore__ inline uint32_t ClampEffectiveRows(uint32_t effectiveRows, uint32_t blockIndex, uint32_t blockRows,
                                              int64_t seqLen)
{
    int64_t start = static_cast<int64_t>(blockIndex) * blockRows;
    uint32_t rows = effectiveRows < blockRows ? effectiveRows : blockRows;
    return seqLen <= start ? 0 : static_cast<uint32_t>(seqLen - start < rows ? seqLen - start : rows);
}

struct EffRowsTile {
    uint32_t blockIndex = 0; // Original block index, also used to address its dequant scale.
    int64_t srcOffset = 0;
    uint32_t rows = 0;
};

// QK and delayed PV use independent cursors on both AIC and AIV.
// A tile never crosses a quantization block, including partially populated blocks.
template <class SparseTensor, class RowsTensor>
__aicore__ inline EffRowsTile NextEffRowsTile(SparseTensor sparse, RowsTensor effRows, uint64_t rowsBase,
                                              uint32_t blockCount, uint32_t blockRows, int64_t seqLen,
                                              uint32_t tileRows, uint32_t &blockCursor, uint32_t &rowCursor)
{
    while (blockCursor < blockCount) {
        uint32_t index = sparse.GetValue(blockCursor);
        uint32_t rows = ClampEffectiveRows(effRows.GetValue(rowsBase + index * 2 + 1), index, blockRows, seqLen);
        if (rowCursor >= rows) {
            ++blockCursor;
            rowCursor = 0;
            continue;
        }
        uint32_t count = rows - rowCursor < tileRows ? rows - rowCursor : tileRows;
        EffRowsTile tile{index, static_cast<int64_t>(index) * blockRows + rowCursor, count};
        rowCursor += count;
        if (rowCursor == rows) {
            ++blockCursor;
            rowCursor = 0;
        }
        return tile;
    }
    return {};
}

} // namespace NpuArch::Gemm::Block
#endif // BSA_EFF_ROWS_TILE_HPP
