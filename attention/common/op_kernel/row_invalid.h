/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file row_invalid.h
 * \brief 行无效判定公共函数, 判断当前 vec block 是否包含需要清零的无效行
 */
#ifndef ROW_INVALID_H
#define ROW_INVALID_H

#include "const_def.h"

struct RowInvalidTokenRange {
    int64_t s1FirstValidToken;
    int64_t s1LastValidToken;
    bool hasValidRow;
};

__aicore__ inline RowInvalidTokenRange CalcRowInvalidTokenRange(int64_t nextTokensLeftUp, int64_t preTokensLeftUp,
                                                                int64_t actS2Size, int64_t actS1Size)
{
    RowInvalidTokenRange r;
    r.s1FirstValidToken = AttentionCommon::Min(AttentionCommon::Max(-nextTokensLeftUp, 0), actS1Size);
    r.s1LastValidToken = AttentionCommon::Min(AttentionCommon::Max(preTokensLeftUp + actS2Size, 0), actS1Size);
    r.s1LastValidToken = AttentionCommon::Max(r.s1LastValidToken - 1, 0);
    r.hasValidRow = (r.s1FirstValidToken > 0) || (r.s1LastValidToken < actS1Size);
    return r;
}

__aicore__ inline bool CalcBatchNeedRowInvalid(const RowInvalidTokenRange &tokenRange, uint8_t sparseMode,
                                               bool isRowInvalidOpen)
{
    return isRowInvalidOpen ||
           ((sparseMode != static_cast<uint8_t>(SparseMode::LEFT_UP_CAUSAL)) && tokenRange.hasValidRow);
}

__aicore__ inline bool CalcS1GBlockNeedRowInvalid(int32_t vecMStartIdx, int32_t vecMSize, uint32_t groupSize,
                                                  const RowInvalidTokenRange &tokenRange)
{
    int32_t vecMEndIdx = vecMStartIdx + vecMSize - 1;
    int32_t s1StartIdx = vecMStartIdx / groupSize;
    int32_t s1EndIdx = vecMEndIdx / groupSize;
    return (s1StartIdx < tokenRange.s1FirstValidToken) || (s1EndIdx > tokenRange.s1LastValidToken);
}

__aicore__ inline bool CalcGS1BlockNeedRowInvalid(int32_t vecMStartIdx, int32_t vecMSize, int64_t actS1Size,
                                                  const RowInvalidTokenRange &tokenRange)
{
    int32_t vecMEndIdx = vecMStartIdx + vecMSize - 1;
    int32_t s1StartIdx = vecMStartIdx % actS1Size;
    int32_t s1EndIdx = vecMEndIdx % actS1Size;
    int32_t gStartIdx = vecMStartIdx / actS1Size;
    int32_t gEndIdx = vecMEndIdx / actS1Size;
    if (gStartIdx == gEndIdx) {
        return (s1StartIdx < tokenRange.s1FirstValidToken) || (s1EndIdx > tokenRange.s1LastValidToken);
    }
    return (tokenRange.s1FirstValidToken > 0) || (tokenRange.s1LastValidToken < actS1Size - 1);
}

#endif // ROW_INVALID_H
