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
 * \file mega_moe_utils.h
 * \brief MegaMoe 调度使用的通用 host/device 辅助函数。
 */

#ifndef MEGA_MOE_UTILS_H
#define MEGA_MOE_UTILS_H

#include <cstdint>

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "mega_moe_constants.h"
#include "mega_moe_types.h"

namespace MegaMoeImpl {

using namespace AscendC;

constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;

struct WorkRange {
    uint32_t start = 0;
    uint32_t count = 0;
};

__aicore__ inline WorkRange TilingByJobContext(uint32_t totalLen, uint32_t jobIndex, uint32_t totalJobs,
                                               uint32_t align = ALIGN_32)
{
    if (totalJobs == 0U || jobIndex >= totalJobs) {
        return {};
    }
    uint32_t lenPerJob = Ops::Base::CeilDiv(totalLen, totalJobs);
    uint32_t alignedLenPerJob = Ops::Base::CeilAlign(lenPerJob, align);
    uint32_t jobOffset = jobIndex * alignedLenPerJob;
    if (jobOffset >= totalLen) {
        return {};
    }
    uint32_t jobLen = jobOffset + alignedLenPerJob > totalLen ? totalLen - jobOffset : alignedLenPerJob;
    return {jobOffset, jobLen};
}

__aicore__ inline GroupSyncSlotLayout CalcGroupSyncSlotLayout(uint32_t expertTokenCount, uint32_t logicalCoreCount)
{
    uint32_t groupCount = Ops::Base::CeilDiv(expertTokenCount, COMBINE_TOKEN_GROUP_SIZE);
    uint32_t totalSyncSlotCount = groupCount > logicalCoreCount ? groupCount : logicalCoreCount;
    return {totalSyncSlotCount / groupCount, totalSyncSlotCount % groupCount};
}

__aicore__ inline __gm__ int32_t *GetCombineSyncCounterAddress(__gm__ int32_t *expertCounterBase,
                                                               uint32_t localSyncSlotIndex)
{
    return expertCounterBase + static_cast<uint64_t>(localSyncSlotIndex) * INT_CACHELINE;
}

__aicore__ inline void GetGroupSyncSlotRange(uint32_t groupIndex, const GroupSyncSlotLayout &slotLayout,
                                             uint32_t &firstSyncSlot, uint32_t &syncSlotCount)
{
    syncSlotCount = slotLayout.baseSlotCountPerGroup + (groupIndex < slotLayout.extraSlotGroupCount ? 1U : 0U);
    uint32_t precedingExtraSlotCount =
        groupIndex < slotLayout.extraSlotGroupCount ? groupIndex : slotLayout.extraSlotGroupCount;
    firstSyncSlot = groupIndex * slotLayout.baseSlotCountPerGroup + precedingExtraSlotCount;
}

__aicore__ inline void ComputeCombineGroupsForCore(uint32_t logicalCoreId, uint32_t groupCount,
                                                   uint32_t logicalCoreCount, uint32_t &firstGroupIndex,
                                                   uint32_t &groupStride, uint32_t &coreIndexWithinGroup,
                                                   uint32_t &coresAssignedToGroup)
{
    uint32_t totalSyncSlotCount = groupCount > logicalCoreCount ? groupCount : logicalCoreCount;
    uint32_t baseSlotCountPerGroup = totalSyncSlotCount / groupCount;
    uint32_t extraSlotGroupCount = totalSyncSlotCount % groupCount;
    uint32_t slotBoundaryAfterExtraSlotGroups = extraSlotGroupCount * (baseSlotCountPerGroup + 1U);

    if (logicalCoreId < slotBoundaryAfterExtraSlotGroups) {
        coresAssignedToGroup = baseSlotCountPerGroup + 1U;
        firstGroupIndex = logicalCoreId / coresAssignedToGroup;
        coreIndexWithinGroup = logicalCoreId % coresAssignedToGroup;
    } else {
        uint32_t slotOffsetAfterExtraSlotGroups = logicalCoreId - slotBoundaryAfterExtraSlotGroups;
        coresAssignedToGroup = baseSlotCountPerGroup;
        firstGroupIndex = extraSlotGroupCount + slotOffsetAfterExtraSlotGroups / coresAssignedToGroup;
        coreIndexWithinGroup = slotOffsetAfterExtraSlotGroups % coresAssignedToGroup;
    }

    groupStride = groupCount < logicalCoreCount ? groupCount : logicalCoreCount;
}

/*
 * 计算单个专家贡献的独立 M 分组数。不同专家使用不同权重，因此各专家不足
 * rowsPerMGroup 的尾块也必须分别占用一个分组。
 */
__aicore__ inline uint32_t GetMGroupCountForRows(uint64_t rowCount, uint32_t rowsPerMGroup)
{
    if (rowCount == 0U || rowsPerMGroup == 0U) {
        return 0U;
    }
    return static_cast<uint32_t>((rowCount - 1U) / rowsPerMGroup + 1U);
}

// 计算剩余分组预算允许当前 Wave 在本专家内推进到的结束行偏移。
__aicore__ inline uint32_t GetWaveEndRowOffsetInExpert(uint64_t expertRowCount, uint32_t currentRowOffsetInExpert,
                                                       uint32_t remainingMGroupCount, uint32_t rowsPerMGroup)
{
    if (remainingMGroupCount == 0U || rowsPerMGroup == 0U || currentRowOffsetInExpert >= expertRowCount) {
        return currentRowOffsetInExpert;
    }
    uint64_t maxWaveEndRowOffset =
        static_cast<uint64_t>(currentRowOffsetInExpert) + static_cast<uint64_t>(remainingMGroupCount) * rowsPerMGroup;
    return static_cast<uint32_t>(expertRowCount < maxWaveEndRowOffset ? expertRowCount : maxWaveEndRowOffset);
}

// 连续均衡分配 token，前 totalTokens % workerCount 个任务各多处理一个 token。
__aicore__ inline WorkRange GetBalancedTokenRange(uint32_t totalTokens, uint32_t workerIdx, uint32_t workerCount)
{
    if (workerCount == 0U || workerIdx >= workerCount) {
        return {};
    }
    uint32_t base = totalTokens / workerCount;
    uint32_t remainder = totalTokens % workerCount;
    uint32_t extraBefore = workerIdx < remainder ? workerIdx : remainder;
    return {workerIdx * base + extraBefore, base + static_cast<uint32_t>(workerIdx < remainder)};
}

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
// 使用调用方准备的全零 UB Tensor，清理当前 AIV 任务负责的连续 GM 区域。
template <int32_t JobAlignment>
__aicore__ inline void ResetWorkspaceRegion(const AivJobContext &job, GM_ADDR regionPtr, int32_t elementCount,
                                            int32_t batchElementCount, LocalTensor<int32_t> &resetTensor)
{
    WorkRange range = TilingByJobContext(static_cast<uint32_t>(elementCount), job.jobIndex, job.totalJobs,
                                         static_cast<uint32_t>(JobAlignment));
    GlobalTensor<int32_t> regionGm;
    regionGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(regionPtr));
    for (uint32_t offset = 0; offset < range.count; offset += static_cast<uint32_t>(batchElementCount)) {
        uint32_t batchCount = range.count - offset < static_cast<uint32_t>(batchElementCount) ?
                                  range.count - offset :
                                  static_cast<uint32_t>(batchElementCount);
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(batchCount * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(regionGm[range.start + offset], resetTensor, copyParams);
    }
}

#endif

__aicore__ inline void NotifyCube(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void WaitForVector(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void NotifyVector(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + value);
}

__aicore__ inline void WaitForCube(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + value);
}

template <bool IsPingPong = false>
__aicore__ inline void EndSync(int32_t vecSetSyncCom, uint16_t pingpongIdx = 0)
{
    if (vecSetSyncCom == 0) {
        return;
    }
    if constexpr (g_coreType == AIC) {
        if constexpr (IsPingPong) {
            if (vecSetSyncCom == 1) {
                WaitForVector(1U - pingpongIdx);
            } else {
                WaitForVector(pingpongIdx);
                WaitForVector(1U - pingpongIdx);
            }
        } else {
            WaitForVector();
        }
    }
}

__aicore__ inline void EndGMM2Sync(int32_t &vecSetSyncCom, uint16_t gmm2PingPongIdx)
{
    if constexpr (g_coreType == AIV) {
        return;
    }
    if (vecSetSyncCom <= 0) {
        return;
    } else if (vecSetSyncCom == 1) {
        WaitForVector();
    } else {
        WaitForVector(gmm2PingPongIdx);
        WaitForVector(1U - gmm2PingPongIdx);
    }
}

__aicore__ inline void TilingByCore(int32_t totalLen, int32_t &coreLen, int32_t &coreOffset, int32_t align = ALIGN_32)
{
    int32_t coreIdx = GetBlockIdx();
    int32_t coreNum = GetBlockNum() * 2;
    int32_t lenPerCore = Ops::Base::CeilDiv(static_cast<uint32_t>(totalLen), static_cast<uint32_t>(coreNum));
    int32_t lenPerCoreAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(lenPerCore), static_cast<uint32_t>(align));
    coreLen = lenPerCoreAlign;
    coreOffset = coreIdx * lenPerCoreAlign;
    if (coreOffset + coreLen >= totalLen) {
        coreLen = totalLen - coreOffset;
    }
    if (coreOffset >= totalLen) {
        coreLen = 0;
    }
}

__aicore__ inline void GmSignalWaitBarrier(__gm__ int32_t *sigAddr, int32_t compareValue)
{
    do {
        if (ReadGmByPassDCache(sigAddr) == compareValue) {
            return;
        }
    } while (true);
}

__aicore__ inline uint64_t GetUrmaCommHandle(__gm__ Mc2MoeContext *mc2Context, uint32_t rankId, uint32_t epRankId)
{
    uint32_t index = rankId > epRankId ? rankId - 1U : rankId;
    return mc2Context->hcommHandle_[index];
}

inline GM_ADDR g_winRankAddr_[HCCL_MAX_RANK_SIZE];

__aicore__ inline GM_ADDR GetRankWinAddrWithOffset(uint32_t rankId, uint64_t offset)
{
    return (GM_ADDR)(g_winRankAddr_[rankId] + offset);
}

__aicore__ inline GM_ADDR GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t *dataAddr = reinterpret_cast<__gm__ uint64_t *>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t *retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<GM_ADDR>(*(retPtr + index));
}

template <typename ElementType>
__aicore__ inline GM_ADDR GetExpertWeightAddr(GM_ADDR tensorListAddr, bool isPerExpertWeightTensor, uint32_t expertIdx,
                                              uint64_t elementOffset)
{
    uint16_t tensorIdx = isPerExpertWeightTensor ? static_cast<uint16_t>(expertIdx) : 0U;
    GM_ADDR tensorAddr = GetTensorAddr(tensorIdx, tensorListAddr);
    if (isPerExpertWeightTensor) {
        return tensorAddr;
    }
    return tensorAddr + elementOffset * sizeof(ElementType);
}

// ======================================================================================
// A8W4/A4W4 权重 L2 预取：等待 token count 的自旋窗口内，各 block 的 AIV0 用 fire-and-forget
// 的 GM->UB 拷贝把 W4 权重原始数据拉进 L2（只填 L2，UB 数据从不消费），使后续权重读命中 L2。
// 权重地址只依赖 expertIdx 与 tiling 常量（见 GetExpertWeightAddr 调用点），count 未就绪即可预取。
// 分片：全部 AIV0（每 block 一个，workerCount = blockNum）在每个权重区域内按 chunk 交错认领
// （worker c 取区域内偏移 c、c+N、... 的 chunk），聚合覆盖率随核数放大（上限为 HBM/L2 带宽）；
// 区域之间保持串行（先全部 W1 再全部 W2，镜像消费顺序），覆盖形状与消费序一致。偶发的同
// chunk 重叠由 L2 去重，无害。
// ======================================================================================

struct WeightL2PrefetchCursor {
    __gm__ uint8_t *chunkAddr = nullptr; // 本 worker 在当前区域内的下一 chunk 地址
    __gm__ uint8_t *regionEnd = nullptr; // 当前区域结束地址（exclusive）
    uint32_t regionIndex = 0U;           // 当前区域编号：[0, E)=W1 各专家，[E, 2E)=W2 各专家
    uint32_t regionCount = 0U;           // 2 * expertCount；regionIndex >= regionCount 表示耗尽
    uint32_t workerIdx = 0U;
    uint32_t workerCount = 1U;
    uint64_t budgetRemaining = 0U; // 本 worker 的字节预算（全局预算 / workerCount）
    GM_ADDR weight1List = nullptr;
    GM_ADDR weight2List = nullptr;
    bool isPerExpertWeightTensor = false;
    uint32_t expertCount = 0U;
    uint64_t w1BytesPerExpert = 0U;
    uint64_t w2BytesPerExpert = 0U;
};

// 把 regionIndex 映射到 (tensor list, expertIdx, 区域字节数) 并定位本 worker 的起始 chunk。
__aicore__ inline void WeightL2PrefetchSetupRegion(WeightL2PrefetchCursor &cursor)
{
    if (cursor.regionIndex >= cursor.regionCount) {
        return;
    }
    bool isW1 = cursor.regionIndex < cursor.expertCount;
    uint32_t expertIdx = isW1 ? cursor.regionIndex : cursor.regionIndex - cursor.expertCount;
    uint64_t regionBytes = isW1 ? cursor.w1BytesPerExpert : cursor.w2BytesPerExpert;
    __gm__ uint8_t *regionBase = reinterpret_cast<__gm__ uint8_t *>(
        GetExpertWeightAddr<uint8_t>(isW1 ? cursor.weight1List : cursor.weight2List, cursor.isPerExpertWeightTensor,
                                     expertIdx, static_cast<uint64_t>(expertIdx) * regionBytes));
    cursor.regionEnd = regionBase + regionBytes;
    // 区域内 chunk 交错认领：worker c 从偏移 (c * CHUNK) mod regionBytes 起步。区域小于
    // workerCount * CHUNK 时回绕会与低号 worker 重叠，L2 去重，无害。
    uint64_t startOffset = regionBytes == 0U ?
                               0U :
                               (static_cast<uint64_t>(cursor.workerIdx) * WEIGHT_L2_PREFETCH_CHUNK_BYTES) % regionBytes;
    cursor.chunkAddr = regionBase + startOffset;
}

// 初始化预取游标。W1 = hiddenDim * tokenHiddenDim / 2，W2 = tokenHiddenDim * (hiddenDim / 2) / 2
// （fp4 两个 element 共 1 字节，SwiGLU 把 GMM1 输出劈成 gate/up 两半），与
// UpdateMoeExpertGmm1/Gmm2GlobalBuffer 的 B 侧寻址公式同源。budgetBytes 为全部 worker 共享的
// 全局预算，内部按 workerCount 均摊（分摊不足一个 chunk 时保底 1 个 chunk）。
__aicore__ inline void WeightL2PrefetchInit(WeightL2PrefetchCursor &cursor, GM_ADDR weight1List, GM_ADDR weight2List,
                                            bool isPerExpertWeightTensor, uint64_t hiddenDim, uint64_t tokenHiddenDim,
                                            uint32_t expertCount, uint32_t workerIdx, uint32_t workerCount,
                                            uint64_t budgetBytes)
{
    cursor.weight1List = weight1List;
    cursor.weight2List = weight2List;
    cursor.isPerExpertWeightTensor = isPerExpertWeightTensor;
    cursor.expertCount = expertCount;
    cursor.regionCount = 2U * expertCount;
    cursor.workerIdx = workerIdx;
    cursor.workerCount = workerCount > 0U ? workerCount : 1U;
    cursor.w1BytesPerExpert = hiddenDim * tokenHiddenDim / 2U;
    cursor.w2BytesPerExpert = tokenHiddenDim * (hiddenDim / ACTIVATION_N_HALF) / 2U;
    cursor.budgetRemaining = budgetBytes / cursor.workerCount;
    if (budgetBytes != 0U && cursor.budgetRemaining == 0U) {
        cursor.budgetRemaining = WEIGHT_L2_PREFETCH_CHUNK_BYTES;
    }
    cursor.regionIndex = 0U;
    WeightL2PrefetchSetupRegion(cursor);
}

// 本 worker 在当前区域的 chunk 走完后推进到下一区域（全部 W1 之后是全部 W2）。
// 指针表解析（GetTensorAddr 两级 GM 读）每个区域只做一次，摊到该区域的 chunk 上。
__aicore__ inline void WeightL2PrefetchAdvanceRegion(WeightL2PrefetchCursor &cursor)
{
    ++cursor.regionIndex;
    WeightL2PrefetchSetupRegion(cursor);
}

// 发射一个 chunk 的 GM->UB fire-and-forget 拷贝（不等待完成）；返回 false 表示本 worker 游标
// 已耗尽（预算用完或全部区域走完）。单缓冲 scratch 即可：MTE2 队列按序执行保证 UB 写-写按
// 程序序，且 scratch 数据从不被消费。
__aicore__ inline bool WeightL2PrefetchTryIssue(WeightL2PrefetchCursor &cursor)
{
    while (cursor.regionIndex < cursor.regionCount && cursor.budgetRemaining != 0U) {
        if (cursor.chunkAddr >= cursor.regionEnd) {
            WeightL2PrefetchAdvanceRegion(cursor);
            continue;
        }
        uint64_t bytesToRegionEnd = static_cast<uint64_t>(cursor.regionEnd - cursor.chunkAddr);
        // 拷贝量同时被区域尾与剩余预算钳住，防止预算均摊后不足一个 chunk 时 uint64 回绕。
        uint64_t copyLimit = bytesToRegionEnd < cursor.budgetRemaining ? bytesToRegionEnd : cursor.budgetRemaining;
        uint32_t copyBytes = copyLimit < WEIGHT_L2_PREFETCH_CHUNK_BYTES ? static_cast<uint32_t>(copyLimit) :
                                                                          WEIGHT_L2_PREFETCH_CHUNK_BYTES;
        GlobalTensor<uint8_t> weightGm;
        weightGm.SetGlobalBuffer(cursor.chunkAddr);
        LocalTensor<uint8_t> scratch(TPosition::VECCALC, WEIGHT_L2_PREFETCH_UB_OFFSET, copyBytes);
        DataCopyPad(scratch, weightGm, {1U, copyBytes, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        cursor.budgetRemaining -= copyBytes;
        // 下一 chunk 跳过其它 worker 认领的 chunk（步长 = workerCount 个 chunk）。
        cursor.chunkAddr += static_cast<uint64_t>(cursor.workerCount) * WEIGHT_L2_PREFETCH_CHUNK_BYTES;
        return true;
    }
    return false;
}

// 排空在飞的预取拷贝：SetFlag 进入 MTE2 队列、按序在全部已发射拷贝完成后置位事件。每次预取会话
// 必须以 Drain 收尾，防止残留拷贝与后续 prologue/unpermute 的 MTE2 访问交错。
__aicore__ inline void WeightL2PrefetchDrain()
{
    SetFlag<HardEvent::MTE2_V>(WEIGHT_L2_PREFETCH_MTE2_V_EVENT_ID);
    WaitFlag<HardEvent::MTE2_V>(WEIGHT_L2_PREFETCH_MTE2_V_EVENT_ID);
}

template <size_t I, typename T>
__aicore__ constexpr inline decltype(auto) Get(T &&value)
{
    return AscendC::Std::get<I>(AscendC::Std::forward<T>(value));
}

template <size_t First, size_t Second, size_t... Rest, typename T>
__aicore__ constexpr inline decltype(auto) Get(T &&value)
{
    return Get<Second, Rest...>(AscendC::Std::get<First>(AscendC::Std::forward<T>(value)));
}

__aicore__ inline ExpertLoopState CreateExpertLoopState(const MoeStageCommonConfig &context)
{
    ExpertLoopState state;
    Get<N_VALUE>(state.problemShape) = context.gmm1OutputDim;
    Get<K_VALUE>(state.problemShape) = context.tokenHiddenDim;
    return state;
}

// 按专家顺序推进紧凑 token 行偏移，并用当前专家的 token 数更新 M 维度。
__aicore__ inline void UpdateExpertLoopState(ExpertLoopState &state, uint32_t expertIdx, uint64_t expertTokenCount)
{
    if (expertIdx != state.expertIdx) {
        state.globalTokenStartIndex += Get<M_VALUE>(state.problemShape);
    }
    state.expertIdx = expertIdx;
    Get<M_VALUE>(state.problemShape) = static_cast<int64_t>(expertTokenCount);
}

// 按当前 Wave 的剩余容量推进专家 token 位置，返回本次推进覆盖的 token 数。
template <uint32_t TileM>
__aicore__ inline uint32_t AdvanceExpertTokenPositionInWave(uint32_t expertTokenCount, uint32_t waveMGroupTarget,
                                                            uint32_t &waveMGroupCount, ExpertTokenPosition &position)
{
    uint32_t expertRemainingTokenCount = expertTokenCount - position.tokenIndexInExpert;
    uint32_t waveRemainingTokenCapacity = (waveMGroupTarget - waveMGroupCount) * TileM;
    uint32_t sliceTokenCount =
        expertRemainingTokenCount < waveRemainingTokenCapacity ? expertRemainingTokenCount : waveRemainingTokenCapacity;
    waveMGroupCount += Ops::Base::CeilDiv(sliceTokenCount, TileM);
    position.tokenIndexInExpert += sliceTokenCount;
    if (position.tokenIndexInExpert >= expertTokenCount) {
        ++position.expertIdx;
        position.tokenIndexInExpert = 0U;
    }
    return sliceTokenCount;
}

// 轮询 GM 中的 int32 ready flag，并在两次读取之间加入短暂退避。
__aicore__ inline void WaitUntilGmFlagIsNonZero(__gm__ int32_t *flagAddr)
{
    constexpr int64_t pollBackoffCycles = 100;
    while (AscendC::ReadGmByPassDCache(flagAddr) == 0) {
        int64_t startCycle = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - startCycle < pollBackoffCycles) {
        }
    }
}

/*
 * 等待 Dispatch 发布当前 expert/block 的 token 总数。这里只同步专家级元数据；每个 256-row group
 * 的 Dispatch 数据就绪依赖由 WaitForGmm1InputReady 处理。
 */
__aicore__ inline void WaitForMoeExpertTokenCountReady(GM_ADDR tokenCountReadyFlagWorkspace,
                                                       const BlockWorkspaceContext &countWorkspace, uint32_t expertIdx)
{
    __gm__ int32_t *tokenCountReadyFlag = reinterpret_cast<__gm__ int32_t *>(tokenCountReadyFlagWorkspace) +
                                          static_cast<uint64_t>(expertIdx) * countWorkspace.blockNum * INT_CACHELINE +
                                          static_cast<uint64_t>(countWorkspace.blockIdx) * INT_CACHELINE;
    WaitUntilGmFlagIsNonZero(tokenCountReadyFlag);
}

/*
 * WaitForMoeExpertTokenCountReady 的预取变体：自旋间隙发 fire-and-forget 权重预取。
 * 每轮先查 flag 再发块（预取不得推迟就绪检测）；flag 就绪即停发并 Drain——在飞 chunk 正是
 * prologue 最先要读的数据，Drain 时间不浪费。游标耗尽后退化为原 100-cycle 退避自旋。
 */
__aicore__ inline void WaitForMoeExpertTokenCountReadyWithPrefetch(GM_ADDR tokenCountReadyFlagWorkspace,
                                                                   const BlockWorkspaceContext &countWorkspace,
                                                                   uint32_t expertIdx,
                                                                   WeightL2PrefetchCursor &cursor)
{
    __gm__ int32_t *tokenCountReadyFlag = reinterpret_cast<__gm__ int32_t *>(tokenCountReadyFlagWorkspace) +
                                          static_cast<uint64_t>(expertIdx) * countWorkspace.blockNum * INT_CACHELINE +
                                          static_cast<uint64_t>(countWorkspace.blockIdx) * INT_CACHELINE;
    while (AscendC::ReadGmByPassDCache(tokenCountReadyFlag) == 0) {
        if (!WeightL2PrefetchTryIssue(cursor)) {
            WaitUntilGmFlagIsNonZero(tokenCountReadyFlag);
            break;
        }
    }
    WeightL2PrefetchDrain();
}

// 从当前 expert/block 对应的稳定 workspace 槽读取专家 token 数。
__aicore__ inline uint32_t GetExpertTokenCountFromWorkspace(GM_ADDR expertTokenCountWorkspace,
                                                            const BlockWorkspaceContext &countWorkspace,
                                                            uint32_t expertIdx)
{
    uint64_t countSlotIndex = static_cast<uint64_t>(expertIdx) * countWorkspace.blockNum + countWorkspace.blockIdx;
    __gm__ int32_t *expertTokenCountAddr =
        reinterpret_cast<__gm__ int32_t *>(expertTokenCountWorkspace) + countSlotIndex * INT32_PER_256B;
    return static_cast<uint32_t>(AscendC::ReadGmByPassDCache(expertTokenCountAddr));
}

// 轮询 GM 中的 int32 flag 直至等于期望值，并在两次读取之间加入短暂退避。
__aicore__ inline void WaitUntilGmFlagEquals(__gm__ int32_t *flagAddr, int32_t expectedValue)
{
    constexpr int64_t pollBackoffCycles = 100;
    while (AscendC::ReadGmByPassDCache(flagAddr) != expectedValue) {
        int64_t startCycle = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - startCycle < pollBackoffCycles) {
        }
    }
}

// 轮询 GM 中的 int32 计数直至不小于目标值，并在两次读取之间加入短暂退避。
__aicore__ inline void WaitUntilGmFlagAtLeast(__gm__ int32_t *flagAddr, int32_t targetValue)
{
    constexpr int64_t pollBackoffCycles = 100;
    while (AscendC::ReadGmByPassDCache(flagAddr) < targetValue) {
        int64_t startCycle = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - startCycle < pollBackoffCycles) {
        }
    }
}

// 全卡同步：各 AIV 按任务分工分摊 rank 握手，末尾执行本卡 AIV 同步。
__aicore__ inline void CrossRankSyncInWorldSize(GM_ADDR rankSyncInWorldPtr, uint32_t rankId, uint32_t worldSize,
                                                const AivJobContext &aivJob)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    __gm__ int32_t *syncRank = reinterpret_cast<__gm__ int32_t *>(rankSyncInWorldPtr);
    __gm__ int32_t *syncCount = reinterpret_cast<__gm__ int32_t *>(rankSyncInWorldPtr + 48U * 1024U +
                                                                   static_cast<uint64_t>(aivJob.jobIndex) * 64U);
    int32_t count = ReadGmByPassDCache(syncCount) + 1;
    for (uint32_t rankIdx = aivJob.jobIndex; rankIdx < worldSize; rankIdx += aivJob.totalJobs) {
        __gm__ int32_t *remoteSyncAddr =
            reinterpret_cast<__gm__ int32_t *>(g_winRankAddr_[rankIdx]) + rankId * INT_CACHELINE;
        WriteGmByPassDCache(remoteSyncAddr, count);
        GmSignalWaitBarrier(syncRank + rankIdx * INT_CACHELINE, count);
    }
    WriteGmByPassDCache(syncCount, count);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

template <AscendC::HardEvent event, int32_t eventId>
__aicore__ inline void SyncFuncStatic()
{
    AscendC::SetFlag<event>(static_cast<event_t>(eventId));
    AscendC::WaitFlag<event>(static_cast<event_t>(eventId));
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_UTILS_H
