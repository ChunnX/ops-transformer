/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file utils.h
 * \brief
 */

#ifndef UTILS_H
#define UTILS_H

namespace AscendC {
constexpr static int64_t MAX_RANK_NUM = 64;  // 支持的最大卡数
struct HcclA5OpResParam {
    uint64_t workSpace; // client 和server 之间通信的地址
    uint64_t workSpaceSize; // client和server之间通信空间的大小
    uint32_t rankId; // 当前卡rankId
    uint32_t rankDim; // 卡总数
    uint64_t winSzie; 
    uint64_t windowsIn[MAX_RANK_NUM];
    uint64_t windowsOut[MAX_RANK_NUM];     // MAX_RANK_NUM 最大到32

    // for ccu
    uint64_t xnAddr; // Xn寄存器起始地址
    uint64_t ckeAddr; // CKE寄存器起始地址
    uint64_t msAddr; // MS地址，预留
    uint64_t msSize; // 可写的MS个数，预留
};

// 向上取整除法：计算a除以b的向上取整结果
__aicore__ inline uint64_t CeilDiv(uint64_t a, uint32_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
};

// 向上对齐：将a向上对齐到b的倍数
__aicore__ inline uint64_t CeilAlign(uint64_t a, uint32_t b)
{
    uint64_t bTemp = static_cast<uint64_t>(b);
    return (bTemp == 0) ? a : CeilDiv(a, bTemp) * bTemp;
};

// 尾块模运算：计算a对b取模，模为0时返回b（用于数据块尾块的计算）
__aicore__ inline uint64_t BlockAlignMod(uint64_t a, uint32_t b)
{
    if (b == 0) {
        return 0;
    }
    
    uint64_t c = a % b;
    return c ? c : b;
}

template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
    AscendC::TEventID eventID = GetTPipePtr()->FetchEventID(event);
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

}  // namespace AscendC

#endif