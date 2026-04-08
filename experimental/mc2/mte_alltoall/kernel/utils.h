/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the CANN Open Software License Agreement Version 2.0.
 */

// ============================================================================
// MC2 Kernel 直调样例 - 通用工具函数
// ============================================================================

#ifndef MC2_UTILS_H
#define MC2_UTILS_H

namespace AscendC {
constexpr static int64_t MAX_RANK_NUM = 64;

// MC2 通信上下文结构体，由 Host 端通过 HcclAllocComResourceByTiling 生成后
// 拷贝到 Device 端。Kernel 通过此结构获取 Win 窗口地址和 rank 信息。
struct HcclA5OpResParam {
    uint64_t workSpace;
    uint64_t workSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
    uint64_t winSzie;
    uint64_t windowsIn[MAX_RANK_NUM];
    uint64_t windowsOut[MAX_RANK_NUM];

    uint64_t xnAddr;
    uint64_t ckeAddr;
    uint64_t msAddr;
    uint64_t msSize;
};

// 向上取整除法
__aicore__ inline uint64_t CeilDiv(uint64_t a, uint32_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

// 尾块模运算：模为 0 时返回 b（用于数据块尾块计算）
__aicore__ inline uint64_t BlockAlignMod(uint64_t a, uint32_t b)
{
    if (b == 0) {
        return 0;
    }
    uint64_t c = a % b;
    return c ? c : b;
}

// 事件同步工具：SetFlag + WaitFlag 封装
template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
    AscendC::TEventID eventID = GetTPipePtr()->FetchEventID(event);
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

}  // namespace AscendC

#endif
