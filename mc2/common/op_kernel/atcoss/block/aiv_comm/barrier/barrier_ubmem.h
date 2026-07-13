/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file barrier_ubmem.h
 * \brief 部分核参与的多卡 Barrier
 *
 * 使用方式：
 *   Barrier barrier;
 *   barrier.Execute(threadCnt, threadId);
 *   1. 核间到达：参与的核之间做 dissemination barrier（O(logN)，自增 core_sync_counter）
 *   2. 跨卡阶段：单核写本地 flag，所有核分工读远端 flag（等远端到达）
 *   3. 使用 team sync_pool + sync_counter
 */

#pragma once

#include "kernel_operator.h"
#include "../collective_comm_context.h"

namespace Atcoss {
namespace AivComm {
using namespace AscendC;

constexpr uint32_t UB_SIZE = 32;
constexpr uint32_t SYNC_SLOT_PEER_CORE = 8;
constexpr uint32_t BARRIER_FLAG_SIZE = 8;

static __aicore__ inline uint32_t GetCoreSyncPoolOffset(uint32_t threadCnt) {
    return BARRIER_FLAG_SIZE + threadCnt * SYNC_SLOT_PEER_CORE * BARRIER_FLAG_SIZE;
}

/**
 * @brief 基于 UB Memory 的多卡 Barrier 同步类
 * 实现部分核参与的多卡 Barrier 同步，采用两阶段算法：
 *   1. 卡内阶段（BarrierIntraDevice）：同卡的多个核之间做 dissemination barrier
 *   2. 跨卡阶段（BarrierCrossDevice）：单核写本地 flag，所有核分工读远端 flag
 * 内存布局：
 *   - syncBuf_: UB buffer 地址，用于临时数据传输，32字节大小
 *   - st_->commBufferAddrs[st_->rankId]: 本地 GM 同步 buffer，结构为：
 *     [flag(BARRIER_FLAG_SIZE)][team_count(BARRIER_FLAG_SIZE*threadCnt)][core_pool(BARRIER_FLAG_SIZE*threadCnt*SYNC_SLOT_PEER_CORE)]
 *      - flag：远端读的信号标志位
 *      - team_count：记录本卡每个核的同步轮次
 *      - core_pool：记录卡内核同步标志位
 */
class Barrier {
public:
    __aicore__ inline Barrier()
    {}

    __aicore__ inline static uint32_t GetBarrierUbSize() {
        return UB_SIZE;
    }

    __aicore__ inline void Execute(
        __ubuf__ uint8_t* syncBuf_, __gm__ CommUbmemContext* st_, uint32_t threadCnt, uint32_t threadId);

private:
    __aicore__ inline void FlushCacheLine(__gm__ uint8_t* addr)
    {
        AscendC::GlobalTensor<uint8_t> global;
        global.SetGlobalBuffer(addr);
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<
            uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global);
        __asm__ __volatile__("");
    }

    __aicore__ inline void BarrierCrossCore(
        __gm__ CommUbmemContext* st_, uint32_t threadId, uint32_t threadCnt, int32_t count);

    __aicore__ inline void BarrierCrossDevice(
        __ubuf__ uint8_t* syncBuf_, __gm__ CommUbmemContext* st_, uint32_t threadId, uint32_t threadCnt, int32_t count);
};

__aicore__ inline void Barrier::Execute(
    __ubuf__ uint8_t* syncBuf_, __gm__ CommUbmemContext* st_, uint32_t threadCnt, uint32_t threadId)
{
    if (threadId >= threadCnt) {
        return;
    }

    __gm__ uint64_t* teamSyncCounter = (__gm__ uint64_t*)(st_->commBufferAddrs[st_->rankId] + BARRIER_FLAG_SIZE + threadId * BARRIER_FLAG_SIZE);
    int32_t count = (int32_t)(*teamSyncCounter) + 1;
    BarrierCrossCore(st_, threadId, threadCnt, count);
    BarrierCrossDevice((__ubuf__ uint8_t*)(syncBuf_), st_, threadId, threadCnt, count);

    LocalTensor<uint64_t> ubTmp;
    ubTmp.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    ubTmp.address_.bufferAddr = reinterpret_cast<uint64_t>(syncBuf_);
    ubTmp.address_.dataLen = UB_SIZE;
    GlobalTensor<uint64_t> teamSyncGmTensor;
    teamSyncGmTensor.SetGlobalBuffer(teamSyncCounter, 1);
    DataCopyExtParams copyParams{1U, sizeof(uint64_t), 0, 0, 0};
    ubTmp.SetValue(0, (uint64_t)count);
    DataCopyPad(teamSyncGmTensor, ubTmp, copyParams);
    PipeBarrier<PIPE_MTE3>();
}

__aicore__ inline void Barrier::BarrierCrossCore(
    __gm__ CommUbmemContext* st_, uint32_t threadId, uint32_t threadCnt, int32_t count)
{
    auto coreSyncPool = (__gm__ uint64_t*)(st_->commBufferAddrs[st_->rankId] + GetCoreSyncPoolOffset(threadCnt));
    uint32_t shift = 1;
    uint32_t offset = 0;

    while (shift < threadCnt) {
        uint32_t next = (threadId + shift) % threadCnt;
        __gm__ uint64_t* dst = (__gm__ uint64_t*)(coreSyncPool + next * SYNC_SLOT_PEER_CORE + offset);
        *dst = count;
        FlushCacheLine((__gm__ uint8_t*)dst);
        __gm__ uint64_t* src = (__gm__ uint64_t*)(coreSyncPool + threadId * SYNC_SLOT_PEER_CORE + offset);
        do {
            if (*src == count) {
                break;
            }
        } while (true);

        shift *= 2;
        offset++;
    }
}

__aicore__ inline void Barrier::BarrierCrossDevice(
    __ubuf__ uint8_t* syncBuf_, __gm__ CommUbmemContext* st_, uint32_t threadId, uint32_t threadCnt, int32_t count)
{
    uint32_t nranks = st_->rankSize;
    uint32_t myRank = st_->rankId;
    int32_t step = threadCnt < nranks ? (int32_t)threadCnt : (int32_t)nranks;
    auto localFlag = (__gm__ int32_t*)(st_->commBufferAddrs[st_->rankId]);

    DataCopyExtParams copyParams{1U, sizeof(int32_t), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams;
    LocalTensor<int32_t> ubTmp;
    ubTmp.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    ubTmp.address_.bufferAddr = reinterpret_cast<uint64_t>(syncBuf_);
    ubTmp.address_.dataLen = UB_SIZE;

    ubTmp.SetValue(0, count);
    GlobalTensor<int32_t> gmFlag;
    gmFlag.SetGlobalBuffer(localFlag);
    DataCopyPad(gmFlag, ubTmp, copyParams);
    PipeBarrier<PIPE_MTE3>();

    int32_t val;
    for (uint32_t i = threadId; i < nranks; i += (uint32_t)step) {
        if (i == myRank) {
            continue;
        }
        __gm__ int32_t* remotePtr = (__gm__ int32_t*)(st_->commBufferAddrs[i]);
        GlobalTensor<int32_t> remoteGmTensor;
        remoteGmTensor.SetGlobalBuffer(remotePtr);
        do {
            DataCopyPad(ubTmp, remoteGmTensor, copyParams, padParams);
            PipeBarrier<PIPE_MTE2>();
            val = ubTmp.GetValue(0);
            if (val >= count) {
                break;
            }
        } while (true);
    }
}

} // namespace AivComm
} // namespace Atcoss