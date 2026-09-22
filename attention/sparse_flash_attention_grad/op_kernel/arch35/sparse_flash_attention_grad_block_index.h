/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cstdint>

namespace SfagBaseApi {
template <typename IndexTensor>
__aicore__ inline int64_t FindSfagPartialBlock(IndexTensor &indices, int64_t baseOffset, int64_t count,
                                               int64_t blockSize, int64_t validTokens)
{
    if (count <= 0 || validTokens <= 0 || validTokens % blockSize == 0) {
        return -1;
    }
    const int64_t boundaryBlock = validTokens / blockSize;
    for (int64_t position = count - 1; position >= 0; --position) {
        if (indices.GetValue(baseOffset + position) == boundaryBlock) {
            return baseOffset + position;
        }
    }
    return -1;
}

__aicore__ inline int64_t MapSfagBlockOffset(int64_t offset, int64_t partialOffset, int64_t lastOffset)
{
    if (partialOffset >= 0) {
        if (offset == lastOffset) {
            return partialOffset;
        }
        if (offset == partialOffset) {
            return lastOffset;
        }
    }
    return offset;
}
} // namespace SfagBaseApi
