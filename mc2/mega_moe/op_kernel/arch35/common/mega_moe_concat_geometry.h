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

namespace MegaMoeImpl {
struct Gmm1ConcatZnCopy {
    uint32_t blockCount;
    uint32_t blockBytes;
    uint64_t sourceStrideBytes;
    uint32_t destinationStrideBytes;
};

// Inputs use logical elements. K is C0-aligned and singleN is a multiple of 128
// under the arch35 tiling contract. Each K/C0 row contributes gate then up.
#ifdef __CCE_AICORE__
__aicore__
#endif
    constexpr inline Gmm1ConcatZnCopy
    MakeGmm1ConcatZnCopy(uint64_t k, uint64_t singleN, uint64_t halfN, uint64_t c0, uint64_t elementsPerByte)
{
    const uint64_t bytesPerColumn = c0 / elementsPerByte;
    return {static_cast<uint32_t>(2U * (k / c0)), static_cast<uint32_t>(singleN * bytesPerColumn),
            halfN * bytesPerColumn, static_cast<uint32_t>(singleN * bytesPerColumn)};
}
} // namespace MegaMoeImpl
