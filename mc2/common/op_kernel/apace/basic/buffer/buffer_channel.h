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
 * \brief BufferChannel：本卡内 AIC(读)/AIV(写) 共用的 buffer 复用环形寻址 + release 进度封装。
 *
 * 内部维护两个独立游标：
 *   - tileIndex：数据寻址游标，由 GetNextSlotAddr 驱动（返回当前 slot 偏移并 +1、回环）；
 *   - lastReleaseIdx：cube 侧 release 进度游标，GetLastReleaseId 查询、GetCurrentReleaseId 推进并产出 flagId。
 * slotBase/slotSize 单位由调用方自定（读端行、写端字节），GetNextSlotAddr 返回同单位偏移。
 * 只封装环形寻址与 release 进度/flagId 计算；CrossCoreSetFlag/CrossCoreWaitFlag、SyncAll、CrossDevice 等同步由外部完成。
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

    // 返回归一化后的槽位数：slotNum 为 0(不复用)或超过 totalTiles 时取 totalTiles
    __aicore__ inline uint32_t Init(uint64_t slotBase, uint64_t slotSize, uint32_t slotNum, uint32_t totalTiles)
    {
        slotBase_ = slotBase;
        slotSize_ = slotSize;
        totalTiles_ = totalTiles;
        slotNum_ = (slotNum == 0 || slotNum > totalTiles_) ? totalTiles_ : slotNum;
        tileIndex_ = 0;
        lastReleaseIdx_ = -1;
        return slotNum_;
    }

    // 返回当前 slot 偏移，并让内部游标前进一格（回环）
    __aicore__ inline uint64_t GetNextSlotAddr()
    {
        uint64_t offset = slotBase_ + (static_cast<uint64_t>(tileIndex_) % slotNum_) * slotSize_;
        tileIndex_++;
        return offset;
    }

    // cube 上次已 release 到的 tile index（供外部 while 条件比较）
    __aicore__ inline int32_t GetLastReleaseId() const
    {
        return lastReleaseIdx_;
    }

    // 推进 release 游标并返回当前 release 的 flagId（index % FLAG_ID_MODULO）
    __aicore__ inline uint32_t GetCurrentReleaseId()
    {
        ++lastReleaseIdx_;
        return static_cast<uint32_t>(lastReleaseIdx_) % FLAG_ID_MODULO;
    }

    // 最后一个需要 release 的 tile index；不复用时 slotNum==totalTiles，返回 -1 表示无需 release
    __aicore__ inline int32_t GetReleaseBound() const
    {
        return static_cast<int32_t>(totalTiles_) - static_cast<int32_t>(slotNum_) - 1;
    }

private:
    uint64_t slotBase_{0};
    uint64_t slotSize_{0};
    uint32_t slotNum_{1};
    uint32_t totalTiles_{0};
    uint32_t tileIndex_{0};
    int32_t lastReleaseIdx_{-1};
};

} // namespace Basic
} // namespace Apace