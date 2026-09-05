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
 * \file buffer_channel.h
 * \brief BufferChannel — AIV(写)/AIC(读) 共用的 buffer 复用环形分配器。
 *
 * 作为通信生产者和计算消费者的桥梁：
 *   - AIV (生产者): GetNextSlot() → {offset, slotIdx}，推进游标，传给 Commit(offset)
 *   - AIC (消费者): GetSlotBase(slotIdx) → GM_ADDR，按需寻址，不触碰游标
 *   - GetBase() → GM_ADDR，获取整体基址（连续 GM tensor 场景）
 */

#pragma once

#include "kernel_operator.h"

namespace Apace {
namespace Basic {

using namespace AscendC;

class BufferChannel {
public:

    struct Slot {
        uint64_t offset;
        uint32_t slotIdx;
    };

    __aicore__ inline BufferChannel() = default;

    __aicore__ inline void Init(GM_ADDR baseAddr, uint64_t slotSize, uint32_t slotNum)
    {
        baseAddr_ = baseAddr;
        slotSize_ = slotSize;
        slotNum_ = slotNum;
        slotIdx_ = 0;
    }

    __aicore__ inline GM_ADDR GetBase() const
    {
        return baseAddr_;
    }

    __aicore__ inline GM_ADDR GetSlotBase(uint32_t slotIdx) const
    {
        return baseAddr_ + static_cast<GM_ADDR>(slotIdx) * slotSize_;
    }

    __aicore__ inline uint32_t GetSlotNum() const
    {
        return slotNum_;
    }

    __aicore__ inline Slot GetNextSlot()
    {
        Slot s{ static_cast<uint64_t>(slotIdx_) * slotSize_, slotIdx_ };
        slotIdx_ = (slotIdx_ + 1) % slotNum_;
        return s;
    }

private:
    GM_ADDR baseAddr_{0};
    uint64_t slotSize_{0};
    uint32_t slotNum_{1};
    uint32_t slotIdx_{0};
};

} // namespace Basic
} // namespace Apace
