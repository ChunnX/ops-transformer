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
 * \file barrier_ubmem.h
 * \brief
 */
#pragma once
#include "../collective_comm_context.h"
#include "basic_api/kernel_basic_intf.h"
#include "tensor_api/tensor.h"

namespace Atcoss {
namespace AivComm {
using namespace AscendC;

constexpr uint32_t UB_SIZE = 64;
constexpr uint32_t SYNC_SLOT_PEER_CORE = 8;
constexpr uint32_t BARRIER_FLAG_SIZE = 8;

static __aicore__ inline uint32_t GetCoreSyncPoolOffset(uint32_t threadCnt) {
    return BARRIER_FLAG_SIZE + threadCnt * SYNC_SLOT_PEER_CORE * BARRIER_FLAG_SIZE;
}

class TeamBarrier {
public:
    __aicore__ inline TeamBarrier()
    {}

    __aicore__ inline static uint32_t GetBarrierUbSize()
    {
        return UB_SIZE;
    }

    __aicore__ inline void Init(
        __ubuf__ uint8_t* syncBuf, __gm__ CommUbmemContext* ctx, uint32_t threadCnt, uint32_t threadId);

    __aicore__ inline void Execute();

private:
    __ubuf__ uint8_t* syncBuf_;
    __gm__ CommUbmemContext* ctx_;
    uint32_t threadCnt_;
    uint32_t threadId_;
    int32_t count_;
    __aicore__ inline void FlushCacheLine(__gm__ uint8_t* addr)
    {
        AscendC::GlobalTensor<uint8_t> global;
        global.SetGlobalBuffer(addr);
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<
            uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global);
        __asm__ __volatile__("");
    }

    __aicore__ inline void BarrierCrossCore();

    __aicore__ inline void BarrierCrossDevice();
};

__aicore__ inline void TeamBarrier::Init(
    __ubuf__ uint8_t* syncBuf, __gm__ CommUbmemContext* ctx, uint32_t threadCnt, uint32_t threadId)
{
    syncBuf_ = syncBuf;
    ctx_ = ctx;
    threadCnt_ = threadCnt;
    threadId_ = threadId;
}

__aicore__ inline void TeamBarrier::Execute()
{
    if ASCEND_IS_AIV {
        if (threadId_ >= threadCnt_) {
            return;
        }

        __gm__ uint64_t* teamSyncCounter = (__gm__ uint64_t*)(ctx_->commBufferAddrs[ctx_->rankId]
            + BARRIER_FLAG_SIZE + threadId_ * BARRIER_FLAG_SIZE);
        count_ = (int32_t)(*teamSyncCounter) + 1;

        BarrierCrossCore();
        SetFlag<HardEvent::S_V>(1);
        WaitFlag<HardEvent::S_V>(1);
        BarrierCrossDevice();

        auto copyUB2GM = Te::MakeCopy(Te::CopyUB2GM{});
        auto ubTmp = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::UB, int32_t>(reinterpret_cast<uint64_t>(syncBuf_)),
            Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<int32_t>>{}(1, 1));
        auto teamSyncGmTensor = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>((__gm__ int32_t*)(teamSyncCounter)),
            Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<int32_t>>{}(1, 1));
        ubTmp[Te::MakeCoord(0, 0)] = count_;
        SetFlag<HardEvent::V_MTE3>(1);
        WaitFlag<HardEvent::V_MTE3>(1);
        Te::Copy(copyUB2GM, teamSyncGmTensor, ubTmp);
        SetFlag<HardEvent::MTE3_S>(1);
        WaitFlag<HardEvent::MTE3_S>(1);

    }
}

__aicore__ inline void TeamBarrier::BarrierCrossCore()
{
    auto coreSyncPool = (__gm__ uint64_t*)(ctx_->commBufferAddrs[ctx_->rankId] + GetCoreSyncPoolOffset(threadCnt_));
    uint32_t shift = 1;
    uint32_t offset = 0;

    while (shift < threadCnt_) {
        uint32_t next = (threadId_ + shift) % threadCnt_;
        __gm__ uint64_t* dst = (__gm__ uint64_t*)(coreSyncPool + next * SYNC_SLOT_PEER_CORE + offset);
        *dst = count_;
        FlushCacheLine((__gm__ uint8_t*)dst);
        __gm__ uint64_t* src = (__gm__ uint64_t*)(coreSyncPool + threadId_ * SYNC_SLOT_PEER_CORE + offset);
        do {
            if (*src == count_) {
                break;
            }
        } while (true);

        shift *= 2;
        offset++;
    }
}

__aicore__ inline void TeamBarrier::BarrierCrossDevice()
{
    uint32_t nranks = ctx_->rankSize;
    uint32_t myRank = ctx_->rankId;
    int32_t step = threadCnt_ < nranks ? (int32_t)threadCnt_ : (int32_t)nranks;
    auto localFlag = (__gm__ int32_t*)(ctx_->commBufferAddrs[ctx_->rankId]);

    auto copyUB2GM = Te::MakeCopy(Te::CopyUB2GM{});
    auto ubTmp = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::UB, int32_t>(reinterpret_cast<uint64_t>(syncBuf_)),
        Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<int32_t>>{}(1, 1));
    ubTmp[Te::MakeCoord(0, 0)] = count_;
    SetFlag<HardEvent::V_MTE3>(1);
    WaitFlag<HardEvent::V_MTE3>(1);
    auto gmFlag = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>((__gm__ int32_t*)(localFlag)),
        Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<int32_t>>{}(1, 1));
    Te::Copy(copyUB2GM, gmFlag, ubTmp);
    SetFlag<HardEvent::MTE3_S>(1);
    WaitFlag<HardEvent::MTE3_S>(1);

    for (uint32_t i = threadId_; i < nranks; i += (uint32_t)step) {
        if (i == myRank) {
            continue;
        }
        __gm__ int32_t* remotePtr = (__gm__ int32_t*)(ctx_->commBufferAddrs[i]);
        do {
            FlushCacheLine((__gm__ uint8_t*)remotePtr);
            if (*remotePtr >= count_) {
                break;
            }
        } while (true);
    }
}
} // namespace AivComm
} // namespace Atcoss