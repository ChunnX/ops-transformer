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
 * \brief BufferChannel：本卡内 AIC(读)/AIV(写) 共用的 buffer 复用环形寻址 + release flag 封装。
 *
 * 内部维护 tileIndex 游标，由外部「通信轮次」驱动前进（GetNextSlotAddr 返回当前 slot 偏移并让游标 +1、回环）。
 * slotBase/slotSize 单位由调用方自定（读端行、写端字节），GetNextSlotAddr/GetBufferAddr 返回同单位偏移。
 * 只封装 buffer 复用相关的一对 release flag（c 核 Set/ v 核 Wait）；SyncAll、CrossDevice 等定制化同步留在外部。
 */

#pragma once

#include "kernel_operator.h"

namespace Apace {
namespace Basic {

using namespace AscendC;

class BufferChannel {
public:
    static constexpr uint32_t FLAG_ID_MODULO = 14;

    __aicore__ inline BufferChannel() = default;

    __aicore__ inline void Init(uint64_t slotBase, uint64_t slotSize, uint32_t slotNum, uint32_t totalTiles)
    {
        slotBase_ = slotBase;
        slotSize_ = slotSize;
        slotNum_ = (slotNum == 0) ? 1 : slotNum;
        totalTiles_ = totalTiles;
        tileIndex_ = 0;
    }

    // 返回当前 slot 偏移，并让内部游标前进一格（回环）
    __aicore__ inline uint64_t GetNextSlotAddr()
    {
        uint64_t offset = slotBase_ + (static_cast<uint64_t>(tileIndex_) % slotNum_) * slotSize_;
        tileIndex_++;
        return offset;
    }

    // 返回当前 slot 偏移（不前进）
    __aicore__ inline uint64_t GetBufferAddr() const
    {
        return slotBase_ + (static_cast<uint64_t>(tileIndex_) % slotNum_) * slotSize_;
    }

    // c 核：round 已被消费完，释放对应 slot（不复用时 slotNum==totalTiles，恒不发）
    __aicore__ inline void SetRelease(uint32_t round)
    {
        if (round < totalTiles_ - slotNum_) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(round % FLAG_ID_MODULO);
        }
    }

    // v 核：写 round 前等对应 slot 被释放（不复用时恒不等待）
    __aicore__ inline void WaitRelease(uint32_t round)
    {
        if (round >= slotNum_) {
            CrossCoreWaitFlag<0x2, PIPE_MTE3>((round - slotNum_) % FLAG_ID_MODULO);
        }
    }

    __aicore__ inline uint32_t GetSlotNum() const { return slotNum_; }

private:
    uint64_t slotBase_{0};
    uint64_t slotSize_{0};
    uint32_t slotNum_{1};
    uint32_t totalTiles_{0};
    uint32_t tileIndex_{0};
};

} // namespace Basic
} // namespace Apace