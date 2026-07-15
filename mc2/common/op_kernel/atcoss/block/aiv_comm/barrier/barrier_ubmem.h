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

namespace Atcoss {
namespace AivComm {
using namespace AscendC;

constexpr uint32_t UB_SIZE = 32;
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
    if (threadId_ >= threadCnt_) {
        return;
    }

    __gm__ uint64_t* teamSyncCounter = (__gm__ uint64_t*)(ctx_->commBufferAddrs[ctx_->rankId]
        + BARRIER_FLAG_SIZE + threadId_ * BARRIER_FLAG_SIZE);
    count_ = (int32_t)(*teamSyncCounter) + 1;
    BarrierCrossCore();
    BarrierCrossDevice();

    LocalTensor<uint64_t> ubTmp;
    ubTmp.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    ubTmp.address_.bufferAddr = reinterpret_cast<uint64_t>(syncBuf_);
    ubTmp.address_.dataLen = UB_SIZE;
    GlobalTensor<uint64_t> teamSyncGmTensor;
    teamSyncGmTensor.SetGlobalBuffer(teamSyncCounter, 1);
    DataCopyExtParams copyParams{1U, sizeof(uint64_t), 0, 0, 0};
    ubTmp.SetValue(0, (uint64_t)count_);
    DataCopyPad(teamSyncGmTensor, ubTmp, copyParams);
    PipeBarrier<PIPE_MTE3>();
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

    DataCopyExtParams copyParams{1U, sizeof(int32_t), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams;
    LocalTensor<int32_t> ubTmp;
    ubTmp.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    ubTmp.address_.bufferAddr = reinterpret_cast<uint64_t>(syncBuf_);
    ubTmp.address_.dataLen = UB_SIZE;

    ubTmp.SetValue(0, count_);
    GlobalTensor<int32_t> gmFlag;
    gmFlag.SetGlobalBuffer(localFlag);
    DataCopyPad(gmFlag, ubTmp, copyParams);
    PipeBarrier<PIPE_MTE3>();

    int32_t val;
    for (uint32_t i = threadId_; i < nranks; i += (uint32_t)step) {
        if (i == myRank) {
            continue;
        }
        __gm__ int32_t* remotePtr = (__gm__ int32_t*)(ctx_->commBufferAddrs[i]);
        GlobalTensor<int32_t> remoteGmTensor;
        remoteGmTensor.SetGlobalBuffer(remotePtr);
        do {
            DataCopyPad(ubTmp, remoteGmTensor, copyParams, padParams);
            PipeBarrier<PIPE_MTE2>();
            val = ubTmp.GetValue(0);
            if (val >= count_) {
                break;
            }
        } while (true);
    }
}
} // namespace AivComm
} // namespace Atcoss