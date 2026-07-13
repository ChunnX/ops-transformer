/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file barrier_ubmem.h
 * \brief 部分核参与的多卡 Barrier 一段式
 *
 * 使用方式：
 *   Barrier barrier;
 *   barrier.Execute(threadCnt, threadId);
 *   1. 核间到达：参与的核之间做 dissemination barrier（O(logN)，自增 core_sync_counter）
 *   2. 跨卡阶段：单核写本地 flag，所有核分工读远端 flag（等远端到达）
 *   3. 使用 team sync_pool + sync_counter（不额外分配内存）
 */

#pragma once

#include "kernel_operator.h"
#include "../asc_comm/asc_comm_context.h"


namespace Dista {
namespace AivComm {
using namespace AscendC;

// 团队同步计数器在 sync buffer 中的起始偏移（字节）
#define BARRIER_TEAM_COUNT_OFFSET 8

// 卡内同步池在 sync buffer 中的起始偏移（字节）
// 布局：8B(flag) + 28*8B(每核一个 slot，最多28核)
#define BARRIER_CORE_SYNC_POOL_OFFSET 8 + 28 * 8

// 每个核在同步池中的 slot 大小（元素数）
// 取 ceil(48/8) = 6，保证 dissemination barrier 的 rounds 不越界
#define SYNC_SLOT_PEER_CORE 6

// 将 val 向上对齐到 al 的倍数（al 必须是 2 的幂）
#define ALIGN_UP(val, al) (((val) + ((al) - 1)) & ~((al) - 1))
/**
 * @brief 基于 UB Memory 的多卡 Barrier 同步类
 *
 * 实现部分核参与的多卡 Barrier 同步，采用两阶段算法：
 *   1. 卡内阶段（BarrierIntraDevice）：同卡的多个核之间做 dissemination barrier
 *   2. 跨卡阶段（BarrierCrossDevice）：单核写本地 flag，所有核分工读远端 flag
 *
 * 内存布局：
 *   - syncBuf_: UB buffer 地址，用于临时数据传输
 *   - st_->localSyncBufferAddr: 本地 GM 同步 buffer，结构为 [flag(8B)][team_count(8B*28)][core_pool(6*8B*28)]
 */
class Barrier {
public:
    __aicore__ inline Barrier()
    {}

    /*!
     * \brief 执行多卡 Barrier 同步
     * \param syncBuf_     UB buffer 地址，用于临时数据传输
     * \param st_          同步 buffer 上下文，包含本地/远端 GM buffer 地址等信息
     * \param threadCnt    本卡参与同步的核数（≤ GetBlockNum）
     * \param threadId     当前核编号（0..threadCnt-1）
     *
     * 执行流程：
     *   1. 从 team sync counter 获取当前轮次 count
     *   2. 执行卡内 barrier（所有核同步到同一轮次）
     *   3. 执行跨卡 barrier（所有卡同步到同一轮次）
     *   4. 更新 team sync counter 为当前轮次
     */
    __aicore__ inline void Execute(
        __ubuf__ uint8_t* syncBuf_, const SyncBufferContext* st_, uint32_t threadCnt, uint32_t threadId);

private:
    /**
     * @brief 刷新指定地址的 cache line
     * @param addr 要刷新的 GM 地址
     *
     * 执行 DCCI 操作，将 cache line 写回并无效化，确保后续读取从 GM 获取最新值
     */
    __aicore__ inline void FlushCacheLine(__gm__ uint8_t* addr)
    {
        AscendC::GlobalTensor<uint8_t> global;
        global.SetGlobalBuffer(addr);
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<
            uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global);
        __asm__ __volatile__("");
    }

    /**
     * @brief 卡内 barrier 同步（Phase 1）
     * @param st_       同步 buffer 上下文
     * @param threadId  当前核编号
     * @param threadCnt 本卡参与核数
     * @param count     当前同步轮次
     *
     * 使用 dissemination barrier 算法，每轮每个核向相邻核发送信号并等待信号
     * 算法复杂度 O(log(threadCnt))，使用 core_sync_pool 存储同步状态
     */
    __aicore__ inline void BarrierIntraDevice(
        const SyncBufferContext* st_, uint32_t threadId, uint32_t threadCnt, int32_t count);

    /**
     * @brief 跨卡 barrier 同步（Phase 2）
     * @param syncBuf_  UB buffer 地址
     * @param st_       同步 buffer 上下文
     * @param threadId  当前核编号
     * @param threadCnt 本卡参与核数
     * @param count     当前同步轮次
     *
     * 只有 threadId=0 的核负责写本地 flag，所有核分工轮询远端卡的 flag
     * 每个核负责 nranks/threadCnt 张卡的 flag 读取
     */
    __aicore__ inline void BarrierCrossDevice(
        __ubuf__ uint8_t* syncBuf_, const SyncBufferContext* st_, uint32_t threadId, uint32_t threadCnt, int32_t count);
};

__aicore__ inline void Barrier::Execute(
    __ubuf__ uint8_t* syncBuf_, const SyncBufferContext* st_, uint32_t threadCnt, uint32_t threadId)
{
    if (threadId >= threadCnt) {
        return;
    }

    // 读取当前轮次的 team sync counter
    __gm__ uint64_t* teamSyncCounter = (__gm__ uint64_t*)(st_->localSyncBufferAddr + BARRIER_TEAM_COUNT_OFFSET + threadId * 8);
    int32_t count = (int32_t)(*teamSyncCounter) + 1;

    // Phase 1: 卡内 barrier，确保本卡所有核同步到同一轮次
    BarrierIntraDevice(st_, threadId, threadCnt, count);

    // Phase 2: 跨卡 barrier，确保所有卡同步到同一轮次
    BarrierCrossDevice((__ubuf__ uint8_t*)(syncBuf_), st_, threadId, threadCnt, count);

    // 更新 team sync counter，通知其他核当前轮次已完成
    LocalTensor<uint64_t> ubTmp;
    ubTmp.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    ubTmp.address_.bufferAddr = reinterpret_cast<uint64_t>(syncBuf_);
    ubTmp.address_.dataLen = ALIGN_UP(sizeof(uint64_t), 32);
    GlobalTensor<uint64_t> teamSyncGmTensor;
    teamSyncGmTensor.SetGlobalBuffer(teamSyncCounter, 1);
    DataCopyExtParams copyParams{1U, sizeof(uint64_t), 0, 0, 0};
    ubTmp.SetValue(0, (uint64_t)count);
    DataCopyPad(teamSyncGmTensor, ubTmp, copyParams);
    PipeBarrier<PIPE_MTE3>();
}

/**
 * Phase 1：卡内 barrier 同步
 *
 * 使用 dissemination barrier 算法，算法复杂度 O(log(threadCnt))
 * 核心思想：每轮每个核向距离 2^shift 的核发送自己的计数器值，并等待对方的计数器值
 *
 * 同步池布局：
 *   corePool[coreIdx][roundId]，roundSize = SYNC_SLOT_PEER_CORE = 6
 *   threadCnt <= coreNumMax = 28
 *   offset = 0 ~ ceil(threadCnt) - 1，offset 最大是 ceil(28) - 1 <= roundSize
 *   因此给每个核最少预留 6 个 slot，核内做地址偏移不会有越界和踩踏问题
 */
__aicore__ inline void Barrier::BarrierIntraDevice(
    const SyncBufferContext* st_, uint32_t threadId, uint32_t threadCnt, int32_t count)
{
    auto coreSyncPool = (__gm__ uint64_t*)(st_->localSyncBufferAddr + BARRIER_CORE_SYNC_POOL_OFFSET);
    uint32_t shift = 1;
    uint32_t offset = 0;

    // Dissemination barrier：每轮向距离 2^shift 的核发送信号并等待
    while (shift < threadCnt) {
        // 计算目标核编号（取模实现循环）
        uint32_t next = (threadId + shift) % threadCnt;

        // 向目标核的同步槽写入当前计数器值
        __gm__ uint64_t* dst = (__gm__ uint64_t*)(coreSyncPool + next * SYNC_SLOT_PEER_CORE + offset);
        *dst = count;
        FlushCacheLine((__gm__ uint8_t*)dst);

        // 自旋等待目标核写入相同的计数器值
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

/**
 * Phase 2：跨卡 barrier 同步
 *
 * 工作分配策略：
 *   - 只有 threadId=0 的核负责写本地 flag（写一次即可）
 *   - 所有核分工轮询远端卡的 flag，每个核负责 nranks/threadCnt 张卡
 *
 * 示例（threadCnt=4, rankSize=8, thisRank=3）：
 *   - threadId=0：写本地 flag，等 rank 0, 4 的 flag
 *   - threadId=1：不写 flag，等 rank 1, 5 的 flag
 *   - threadId=2：不写 flag，等 rank 2, 6 的 flag
 *   - threadId=3：不写 flag，跳过 rank 3（自己），等 rank 7 的 flag
 */
__aicore__ inline void Barrier::BarrierCrossDevice(
    __ubuf__ uint8_t* syncBuf_, const SyncBufferContext* st_, uint32_t threadId, uint32_t threadCnt, int32_t count)
{
    uint32_t nranks = st_->rankSize;
    uint32_t myRank = st_->myRankId;
    int32_t step = threadCnt < nranks ? (int32_t)threadCnt : (int32_t)nranks;
   
    // 本地 flag 地址（GM 空间起始位置）
    auto localFlag = (__gm__ int32_t*)(st_->localSyncBufferAddr);

    // 准备 UB buffer 用于数据传输
    DataCopyExtParams copyParams{1U, sizeof(int32_t), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams;
    LocalTensor<int32_t> ubTmp;
    ubTmp.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    ubTmp.address_.bufferAddr = reinterpret_cast<uint64_t>(syncBuf_);
    ubTmp.address_.dataLen = ALIGN_UP(sizeof(int32_t), 32);

    // 只有 threadId=0 负责写本地 flag
    if (threadId == 0) {
        ubTmp.SetValue(0, count);
        GlobalTensor<int32_t> gmFlag;
        gmFlag.SetGlobalBuffer(localFlag);
        DataCopyPad(gmFlag, ubTmp, copyParams);
        PipeBarrier<PIPE_MTE3>();
    }

    // 所有核分工轮询远端卡的 flag
    int32_t val;
    for (uint32_t i = threadId; i < nranks; i += (uint32_t)step) {
        if (i == myRank) {
            continue;  // 跳过本卡
        }
        __gm__ int32_t* remotePtr = (__gm__ int32_t*)(st_->remoteSyncBufferAddrs[i]);
        GlobalTensor<int32_t> remoteGmTensor;
        remoteGmTensor.SetGlobalBuffer(remotePtr);

        // 自旋等待远端卡到达同一轮次
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
} // namespace Dista