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
 * \file flash_decode_s2_split.h
 * \brief S2-split Flash Decode staging layout helpers.
 */
#ifndef FLASH_DECODE_S2_SPLIT_H
#define FLASH_DECODE_S2_SPLIT_H

#include <stdint.h>

namespace AttentionCommon {

// The three regions are contiguous: partial O, max, then sum.
struct S2SplitFdStagingLayout {
    uint32_t stagingM;
    uint32_t dAlign;
    uint32_t slotCount;
    uint32_t broadcastElems;

    __aicore__ inline uint32_t StagingAttenOutBytes() const
    {
        return stagingM * dAlign * sizeof(float);
    }

    __aicore__ inline uint32_t StagingMaxSumBytes() const
    {
        return stagingM * broadcastElems * sizeof(float);
    }

    __aicore__ inline __gm__ uint8_t *AttenOutRegion(__gm__ uint8_t *base) const
    {
        return base;
    }

    __aicore__ inline __gm__ uint8_t *MaxRegion(__gm__ uint8_t *base) const
    {
        return AttenOutRegion(base) + slotCount * StagingAttenOutBytes();
    }

    __aicore__ inline __gm__ uint8_t *SumRegion(__gm__ uint8_t *base) const
    {
        return MaxRegion(base) + slotCount * StagingMaxSumBytes();
    }
};

} // namespace AttentionCommon

#endif  // FLASH_DECODE_S2_SPLIT_H
