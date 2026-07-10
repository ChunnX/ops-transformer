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
 * \file barrier_ubmem.h
 * \brief 分层同步 Barrier - 对称分层策略 + 两阶段同步
 *
 * 使用场景：部分 vec core 参与的多卡同步
 * - 本卡和对端卡使用相同的分层策略（对称）
 * - 每个 vec core 负责一部分对端卡同步（并行化）
 * - 两阶段同步：跨卡 pairwise 同步 + 本卡 vec core 同步
 *
 * 技术特点：
 * - 对称分层同步网络：本卡和对端卡核数假设相同
 * - 部分同步语义：只保证参与的核同步，其他核不参与
 * - 简化参数设计：只需 threadCnt/threadId（本卡参数）
 * - 自动分核计算：Base 类自动计算 targetRank 分配
 * - 第一阶段并行化：多个核并行处理不同对端卡集合
 * - 第二阶段同步：本卡核同步确保第一阶段都完成
 *
 * 使用方式：
 * // 本卡所有核参与
 * uint32_t threadCnt = GetBlockDim();
 * uint32_t threadId = GetBlockIdx();
 * Barrier barrier;
 * barrier.Execute(threadCnt, threadId);
 *
 * // 本卡部分核参与（kernel启动时配置blockDim=4）
 * uint32_t threadCnt = 4;
 * uint32_t threadId = GetBlockIdx();
 * Barrier barrier;
 * barrier.Execute(threadCnt, threadId);
 */

#pragma once

#include "kernel_operator.h"
#include "shmem.h"

namespace Dista {
namespace AivComm {

using namespace AscendC;

constexpr int32_t SHIFT_MULTIPLIER = 2;
constexpr uint64_t ACLSHMEM_DATA_CACHE_LINE_SIZE = 64;

class Barrier {
public:
    __aicore__ inline Barrier() {}
    
    /*!
     * \brief 分层同步 Barrier（两阶段）
     * \param threadCnt 本卡参与核数
     * \param threadId 当前核编号（0 ~ threadCnt-1）
     * 
     * 执行流程：
     * 1. 自动计算 targetRank 分配（threadId -> targetRankStart/targetRankCnt）
     * 2. 第一阶段：跨卡 pairwise 同步（每个核与负责的对端卡同步）
     * 3. 第二阶段：本卡 vec core 同步（确保第一阶段都完成）
     * 
     * 约束：
     * - 所有参与的核都必须调用此接口
     * - 对端卡的分层策略与本卡相同（对称）
     * - 对端卡参与核数假设与本卡相同
     */
    __aicore__ inline void Execute(
        uint32_t threadCnt,
        uint32_t threadId);
    
private:
    /*!
     * \brief 第一阶段：跨卡 pairwise 同步
     * \param threadId 当前核编号（本卡）
     * \param targetRank 对端卡 rankId
     * \param targetVecCoreCnt 对端卡参与核数（假设与本卡相同）
     * \param count barrier 计数器（用于信号通知）
     * 
     * 执行流程：
     * 1. 本核通知对端卡的所有核（写远程 sync_pool）
     * 2. 等待对端卡的所有核通知（读本地 sync_pool）
     * 
     * 正确性保证：
     * - 对端卡的所有核都到达 barrier 后才能继续
     * - 对端卡的分层策略与本卡相同（对称）
     */
    __aicore__ inline void BarrierPairwise(
        uint32_t threadId,
        uint32_t targetRank,
        uint32_t targetVecCoreCnt,
        int32_t count);
    
    /*!
     * \brief 第二阶段：本卡 vec core 同步
     * \param threadId 当前核编号（本卡）
     * \param threadCnt 本卡参与核数
     * \param count barrier 计数器（用于信号通知）
     * 
     * 执行流程：
     * 1. 使用 dissemination barrier（O(logN) 算法）
     * 2. 每轮：通知后续核 + 等待前序核
     * 3. 距离指数增长：shift = 1, 2, 4, 8...
     * 
     * 正确性保证：
     * - 确保第一阶段所有核都完成跨卡同步
     * - 本卡所有参与的核都到达 barrier 后才能继续
     */
    __aicore__ inline void BarrierVecCore(
        uint32_t threadId,
        uint32_t threadCnt,
        int32_t count);
    
    /*!
     * \brief 写远程信号（通知对端卡）
     * \param syncPoolAddr 同步数组地址（远程）
     * \param count barrier 计数器值
     */
    __aicore__ inline void SignalSetRemote(
        __gm__ int32_t* syncPoolAddr,
        int32_t count);
    
    /*!
     * \brief 等待信号（读本地）
     * \param syncPoolAddr 同步数组地址（本地）
     * \param count 期望的 barrier 计数器值
     */
    __aicore__ inline void SignalWaitLocal(
        __gm__ int32_t* syncPoolAddr,
        int32_t count);
    
    /*!
     * \brief 写本地信号（通知本卡其他核）
     * \param syncPoolAddr 同步数组地址（本地）
     * \param count barrier 计数器值
     */
    __aicore__ inline void SignalSetLocal(
        __gm__ int32_t* syncPoolAddr,
        int32_t count);
    
    /*!
     * \brief 获取 team sync pool（参考 shmem）
     * \param team_idx team 索引
     * \return sync pool 地址
     */
    __aicore__ inline __gm__ aclshmemx_sync_bit* GetTeamSyncPool(aclshmem_team_t team_idx);
    
    /*!
     * \brief 获取 team sync counter（参考 shmem）
     * \param team_idx team 索引
     * \return sync counter 地址
     */
    __aicore__ inline __gm__ aclshmemx_sync_bit* GetTeamSyncCounter(aclshmem_team_t team_idx);
    
    /*!
     * \brief 获取 core sync pool（参考 shmem）
     * \return core sync pool 地址
     */
    __aicore__ inline __gm__ aclshmemx_sync_bit* GetCoreSyncPool();
    
    /*!
     * \brief 获取 core sync counter（参考 shmem）
     * \return core sync counter 地址
     */
    __aicore__ inline __gm__ aclshmemx_sync_bit* GetCoreSyncCounter();
};

__aicore__ inline void Barrier::Execute(
    uint32_t threadCnt,
    uint32_t threadId)
{
    uint32_t rankSize = aclshmem_n_pes();
    uint32_t myRankId = aclshmem_my_pe();
    
    // 自动计算 targetRank 分配（参考 aiv_udma）
    uint32_t targetRankPerCore = (rankSize + threadCnt - 1) / threadCnt;
    uint32_t targetRankStart = threadId * targetRankPerCore;
    
    uint32_t targetRankCnt;
    if (targetRankStart + targetRankPerCore <= rankSize) {
        targetRankCnt = targetRankPerCore;
    } else if (targetRankStart < rankSize) {
        targetRankCnt = rankSize - targetRankStart;
    } else {
        targetRankCnt = 0;
    }
    
    // 获取 barrier 计数器（参考 shmem）
    auto syncCounter = GetTeamSyncCounter(ACLSHMEM_TEAM_WORLD);
    int32_t count = aclshmemi_load((__gm__ int32_t*)syncCounter) + 1;
    
    // 第一阶段：跨卡 pairwise 同步
    for (uint32_t i = 0; i < targetRankCnt; i++) {
        uint32_t targetRank = targetRankStart + i;
        if (targetRank != myRankId) {
            BarrierPairwise(threadId, targetRank, threadCnt, count);
        }
    }
    
    // 第二阶段：本卡 vec core 同步
    BarrierVecCore(threadId, threadCnt, count);
    
    // 更新计数器（只有主核更新）
    if (threadId == 0) {
        aclshmemi_store((__gm__ int32_t*)syncCounter, count);
    }
}

__aicore__ inline void Barrier::BarrierPairwise(
    uint32_t threadId,
    uint32_t targetRank,
    uint32_t targetVecCoreCnt,
    int32_t count)
{
    auto syncPool = GetTeamSyncPool(ACLSHMEM_TEAM_WORLD);
    
    // 第一阶段：通知对端卡的所有核（写远程）
    // 每个核通知对端卡的对应核（threadId -> targetRank 的核 threadId）
    __gm__ int32_t* remoteSyncAddr = (__gm__ int32_t*)(syncPool + threadId);
    SignalSetRemote(remoteSyncAddr, count);
    
    // 第二阶段：等待对端卡的所有核（读本地）
    // 等待对端卡对应核的通知（threadId 的核收到 targetRank 的核 threadId 的通知）
    __gm__ int32_t* localSyncAddr = (__gm__ int32_t*)(syncPool + threadId);
    SignalWaitLocal(localSyncAddr, count);
}

__aicore__ inline void Barrier::BarrierVecCore(
    uint32_t threadId,
    uint32_t threadCnt,
    int32_t count)
{
    auto coreSyncPool = GetCoreSyncPool();
    auto coreSyncCounter = GetCoreSyncCounter();
    int32_t coreCount = aclshmemi_load((__gm__ int32_t*)coreSyncCounter) + 1;
    
    // Dissemination barrier（参考 shmem）
    int32_t shift = 1;
    int32_t offset = 0;
    while (shift < threadCnt) {
        int32_t next = (threadId + shift) % threadCnt;
        
        // 通知后续核
        __gm__ int32_t* notifyAddr = (__gm__ int32_t*)(coreSyncPool + next + offset);
        SignalSetLocal(notifyAddr, coreCount);
        
        // 等待前序核
        __gm__ int32_t* waitAddr = (__gm__ int32_t*)(coreSyncPool + threadId + offset);
        SignalWaitLocal(waitAddr, coreCount);
        
        shift *= SHIFT_MULTIPLIER;
        offset++;
    }
    
    // 更新 core sync counter（只有主核更新）
    if (threadId == 0) {
        aclshmemi_store((__gm__ int32_t*)coreSyncCounter, coreCount);
    }
}

__aicore__ inline void Barrier::SignalSetRemote(
    __gm__ int32_t* syncPoolAddr,
    int32_t count)
{
    aclshmemi_signal_set(syncPoolAddr, count);
}

__aicore__ inline void Barrier::SignalWaitLocal(
    __gm__ int32_t* syncPoolAddr,
    int32_t count)
{
    aclshmemi_signal_wait_until_eq_for_barrier(syncPoolAddr, count);
}

__aicore__ inline void Barrier::SignalSetLocal(
    __gm__ int32_t* syncPoolAddr,
    int32_t count)
{
    aclshmemi_signal_set(syncPoolAddr, count);
}

__aicore__ inline __gm__ aclshmemx_sync_bit* Barrier::GetTeamSyncPool(aclshmem_team_t team_idx)
{
    uint64_t addr = (uint64_t)aclshmemi_get_state()->sync_pool;
    addr += team_idx * SYNC_ARRAY_SIZE;
    return (__gm__ aclshmemx_sync_bit*)addr;
}

__aicore__ inline __gm__ aclshmemx_sync_bit* Barrier::GetTeamSyncCounter(aclshmem_team_t team_idx)
{
    uint64_t addr = (uint64_t)aclshmemi_get_state()->sync_counter;
    addr += team_idx * SYNC_COUNTER_SIZE;
    return (__gm__ aclshmemx_sync_bit*)addr;
}

__aicore__ inline __gm__ aclshmemx_sync_bit* Barrier::GetCoreSyncPool()
{
    return (__gm__ aclshmemx_sync_bit*)aclshmemi_get_state()->core_sync_pool;
}

__aicore__ inline __gm__ aclshmemx_sync_bit* Barrier::GetCoreSyncCounter()
{
    return (__gm__ aclshmemx_sync_bit*)aclshmemi_get_state()->core_sync_counter;
}

} // namespace AivComm
} // namespace Dista