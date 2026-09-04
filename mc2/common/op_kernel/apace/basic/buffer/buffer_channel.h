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
 * \brief BufferChannel：本卡内 AIC(读)/AIV(写) 共用的 buffer 复用环形分配器。
 *
 * 只负责"给出下一块可用 buffer 的逻辑偏移 + 逻辑 id"：
 *   - slotIdx 在内部编号 0..slotNum-1，环形回绕；
 *   - GetNextSlot 返回 {offset = id*slotSize, id}，并推进游标；
 *     以避开 SyncAll<true> 占用的 flagId 14；此约束由调用方(复用深度 bufferCount)保证。
 * slotSize 单位由调用方自定（读端行、写端字节）；CrossCoreSetFlag/WaitFlag 等同步原语由外部完成。
 */

#pragma once

#include "kernel_operator.h"

namespace Apace {
namespace Basic {

using namespace AscendC;

class BufferChannel {
public:

    struct Slot {
        uint64_t offset;   // 逻辑偏移 = slotIdx * slotSize
        uint32_t slotIdx;   // 逻辑 slot 索引，0..slotNum-1
    };

    __aicore__ inline BufferChannel() = default;

    __aicore__ inline void Init(uint64_t slotSize, uint32_t slotNum)
    {
        ascendc_assert(slotNum >= 1, "BufferChannel slotNum must be >= 1");
        slotSize_ = slotSize;
        slotNum_ = slotNum;
        slotIdx_ = 0;
    }

    __aicore__ inline uint32_t GetSlotNum() const
    {
        return slotNum_;
    }

    // 返回当前 buffer 的逻辑偏移与 id，并让游标回环前进一格
    __aicore__ inline Slot GetNextSlot()
    {
        Slot s{ static_cast<uint64_t>(slotIdx_) * slotSize_, slotIdx_ };
        slotIdx_ = (slotIdx_ + 1) % slotNum_;
        return s;
    }

private:
    uint64_t slotSize_{0};
    uint32_t slotNum_{1};
    uint32_t slotIdx_{0};
};

} // namespace Basic
} // namespace Apace