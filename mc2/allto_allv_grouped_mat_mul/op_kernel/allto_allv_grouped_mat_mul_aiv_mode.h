/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 */

#ifndef ALLTO_ALLV_GROUPED_MAT_MUL_AIV_MODE_H
#define ALLTO_ALLV_GROUPED_MAT_MUL_AIV_MODE_H

#include <cstdint>

#include "allto_allv_grouped_mat_mul_aiv_comm.h"
#include "allto_allv_grouped_mat_mul_catlass.h"
#include "allto_allv_grouped_mat_mul_tiling.h"

namespace AlltoAllvGroupedMatMulAivMode {

using ExpertMeta = AlltoAllvGroupedMatMulAiv::ExpertMeta;

struct ElementRange {
    uint64_t offset = 0U;
    uint64_t count = 0U;
};

#if defined(__CCE_AICORE__)
#define A2AVGMM_AIV_MODE_HOST_DEVICE __aicore__ inline
#else
#define A2AVGMM_AIV_MODE_HOST_DEVICE inline
#endif

A2AVGMM_AIV_MODE_HOST_DEVICE bool BuildExpertMetaForIndex(
    const int64_t *recvCounts, uint32_t rankSize, uint32_t expertPerRank,
    uint32_t expertIdx, uint64_t tokenCapacity, ExpertMeta &expert)
{
    if (recvCounts == nullptr || rankSize == 0U || expertPerRank == 0U ||
        expertIdx >= expertPerRank) {
        return false;
    }

    uint64_t expertBase = 0U;
    for (uint32_t priorExpert = 0U; priorExpert < expertIdx; ++priorExpert) {
        for (uint32_t sourceRank = 0U; sourceRank < rankSize; ++sourceRank) {
            const int64_t count =
                recvCounts[static_cast<uint64_t>(sourceRank * expertPerRank + priorExpert)];
            if (count < 0 ||
                !AlltoAllvGroupedMatMulAiv::SafeAddU64(
                    expertBase, static_cast<uint64_t>(count), expertBase)) {
                return false;
            }
        }
    }

    uint64_t expertTokens = 0U;
    for (uint32_t sourceRank = 0U; sourceRank < rankSize; ++sourceRank) {
        const int64_t count =
            recvCounts[static_cast<uint64_t>(sourceRank * expertPerRank + expertIdx)];
        if (count < 0 ||
            !AlltoAllvGroupedMatMulAiv::SafeAddU64(
                expertTokens, static_cast<uint64_t>(count), expertTokens)) {
            return false;
        }
    }

    if (expertBase > tokenCapacity || expertTokens > tokenCapacity - expertBase ||
        expertTokens > 0xffffffffULL) {
        return false;
    }
    expert.recvTokenBase = expertBase;
    expert.tokenCount = static_cast<uint32_t>(expertTokens);
    expert.reserved = 0U;
    return true;
}

A2AVGMM_AIV_MODE_HOST_DEVICE bool GetDestinationSourceTokenOffset(
    const int64_t *recvCounts, uint32_t rankSize, uint32_t expertPerRank,
    uint32_t expertIdx, uint32_t sourceRank, uint64_t tokenCapacity,
    uint64_t &tokenOffset)
{
    ExpertMeta expert = {};
    if (!BuildExpertMetaForIndex(recvCounts, rankSize, expertPerRank, expertIdx,
                                 tokenCapacity, expert) ||
        sourceRank >= rankSize) {
        return false;
    }

    tokenOffset = expert.recvTokenBase;
    for (uint32_t source = 0U; source < sourceRank; ++source) {
        const int64_t count =
            recvCounts[static_cast<uint64_t>(source * expertPerRank + expertIdx)];
        if (count < 0 ||
            !AlltoAllvGroupedMatMulAiv::SafeAddU64(
                tokenOffset, static_cast<uint64_t>(count), tokenOffset)) {
            return false;
        }
    }
    return tokenOffset <= tokenCapacity;
}

A2AVGMM_AIV_MODE_HOST_DEVICE bool PartitionElements(
    uint64_t totalElements, uint32_t workerIdx, uint32_t workerNum,
    ElementRange &range)
{
    if (workerNum == 0U || workerIdx >= workerNum) {
        range = {};
        return false;
    }

    const uint64_t quotient = totalElements / workerNum;
    const uint64_t remainder = totalElements % workerNum;
    range.count = quotient + (workerIdx < remainder ? 1U : 0U);
    range.offset = workerIdx < remainder ? workerIdx : remainder;
    for (uint32_t worker = 0U; worker < workerIdx; ++worker) {
        range.offset += quotient;
    }
    return true;
}

A2AVGMM_AIV_MODE_HOST_DEVICE bool IsSourceWorker(
    uint32_t taskIdx, uint32_t taskRatio, uint32_t subBlockIdx,
    uint32_t rankSize, uint32_t sourceRank)
{
    if (taskRatio == 0U || subBlockIdx != 0U || sourceRank >= rankSize) {
        return false;
    }
    const uint32_t blockIdx = taskIdx / taskRatio;
    return blockIdx < rankSize && blockIdx == sourceRank;
}

#undef A2AVGMM_AIV_MODE_HOST_DEVICE

} // namespace AlltoAllvGroupedMatMulAivMode

#if defined(__CCE_AICORE__) && !defined(__CCE_KT_TEST__)

namespace AlltoAllvGroupedMatMulAivMode {
namespace detail {

constexpr uint64_t kControlRegionBytes = 16U * 1024U * 1024U;
constexpr uint32_t kAicDataReadyEventId = 12U;
constexpr uint32_t kFlagBufferBytes = 32U;

__aicore__ inline int32_t LoadInvocationEpoch(
    const AlltoAllvGroupedMatMulAiv::PeerWindowContext &context,
    const AlltoAllvGroupedMatMulAiv::A2avWindowLayout &layout,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer)
{
    __gm__ int32_t *epochAddress = AlltoAllvGroupedMatMulAiv::GetFlagAddress(
        context.Window(context.RankId()), layout.controlOffset,
        AlltoAllvGroupedMatMulAiv::kEpochOffset);
    return AlltoAllvGroupedMatMulAiv::LoadFlag(epochAddress, flagBuffer);
}

__aicore__ inline void SignalAicDataReady()
{
    Catlass::Arch::CrossCoreFlag flag{kAicDataReadyEventId};
    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag);
}

__aicore__ inline bool GetPeerSourceTokenOffset(
    AscendC::GlobalTensor<int64_t> &peerCounts, uint32_t rankSize,
    uint32_t expertPerRank, uint32_t destinationRank, uint32_t expertIdx,
    uint64_t &tokenOffset, uint32_t &tokenCount)
{
    if (rankSize == 0U || expertPerRank == 0U || destinationRank >= rankSize ||
        expertIdx >= expertPerRank) {
        return false;
    }

    const uint64_t target =
        static_cast<uint64_t>(destinationRank * expertPerRank + expertIdx);
    tokenOffset = 0U;
    for (uint64_t index = 0U; index < target; ++index) {
        const int64_t count = peerCounts.GetValue(index);
        if (count < 0 ||
            !AlltoAllvGroupedMatMulAiv::SafeAddU64(
                tokenOffset, static_cast<uint64_t>(count), tokenOffset)) {
            return false;
        }
    }

    const int64_t count = peerCounts.GetValue(target);
    if (count < 0 || static_cast<uint64_t>(count) > 0xffffffffULL) {
        return false;
    }
    tokenCount = static_cast<uint32_t>(count);
    return true;
}

__aicore__ inline bool BuildDeviceWindowLayout(
    uint64_t inputTokens, uint32_t hiddenSize, uint32_t countNum,
    uint64_t windowBytes,
    AlltoAllvGroupedMatMulAiv::A2avWindowLayout &layout)
{
    if (inputTokens > 0xffffffffULL) {
        return false;
    }
    const uint64_t inputElements =
        AlltoAllvGroupedMatMulAiv::MulU32ToU64(
            static_cast<uint32_t>(inputTokens), hiddenSize);
    if (inputElements > (~static_cast<uint64_t>(0U) >> 1U)) {
        return false;
    }

    uint64_t countAreaBytes = 0U;
    uint64_t payloadAndCounts = 0U;
    uint64_t requiredBytes = 0U;
    if (!AlltoAllvGroupedMatMulAiv::AlignUpU64(
            inputElements << 1U,
            AlltoAllvGroupedMatMulAiv::kWindowAlignment,
            layout.inputBytes) ||
        !AlltoAllvGroupedMatMulAiv::AlignUpU64(
            static_cast<uint64_t>(countNum) << 3U,
            AlltoAllvGroupedMatMulAiv::kWindowAlignment,
            countAreaBytes) ||
        !AlltoAllvGroupedMatMulAiv::SafeAddU64(
            layout.inputBytes, countAreaBytes, payloadAndCounts) ||
        !AlltoAllvGroupedMatMulAiv::SafeAddU64(
            payloadAndCounts, kControlRegionBytes, requiredBytes) ||
        requiredBytes > windowBytes ||
        kControlRegionBytes > windowBytes) {
        return false;
    }

    layout.controlOffset = AlltoAllvGroupedMatMulAiv::AlignDownU64(
        windowBytes - kControlRegionBytes,
        AlltoAllvGroupedMatMulAiv::kWindowAlignment);
    if (layout.controlOffset < countAreaBytes) {
        return false;
    }
    layout.countsOffset = layout.controlOffset - countAreaBytes;
    if (layout.inputBytes > layout.countsOffset) {
        return false;
    }
    layout.readyOffset = AlltoAllvGroupedMatMulAiv::kFirstExpertFlagOffset;
    layout.totalBytes = requiredBytes;
    return true;
}

__aicore__ inline uint32_t GetCopyQueueBytes(
    const AlltoAllvGmmAivTilingData &tiling)
{
    const uint64_t payloadBytes =
        static_cast<uint64_t>(tiling.gmmCocTiling.ubMoveNum) * sizeof(bfloat16_t);
    const uint64_t countBytes =
        static_cast<uint64_t>(A2AVGMM_MAX_COUNT_NUM) * sizeof(int64_t);
    const uint64_t bytes = payloadBytes > countBytes ? payloadBytes : countBytes;
    return static_cast<uint32_t>(
        bytes > static_cast<uint64_t>(0xffffffffU) ? 0xffffffffU : bytes);
}

__aicore__ inline bool CopyInitialWindowPayload(
    GM_ADDR gmmx,
    const AlltoAllvGmmAivTilingData &tiling,
    const AlltoAllvGroupedMatMulAiv::PeerWindowContext &context,
    const AlltoAllvGroupedMatMulAiv::A2avWindowLayout &layout,
    uint32_t taskIdx, uint32_t subBlockIdx,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &copyBuffer,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &countBuffer)
{
    uint64_t inputElements = 0U;
    for (uint32_t index = 0U; index < tiling.countNum; ++index) {
        const int64_t count = tiling.sendCounts[index];
        if (count < 0 ||
            !AlltoAllvGroupedMatMulAiv::SafeAddU64(
                inputElements, static_cast<uint64_t>(count), inputElements)) {
            return false;
        }
    }
    if (inputElements > 0xffffffffULL) {
        return false;
    }
    inputElements = AlltoAllvGroupedMatMulAiv::MulU32ToU64(
        static_cast<uint32_t>(inputElements), tiling.gmmInfo.K);

    ElementRange range = {};
    if (!PartitionElements(inputElements, taskIdx, tiling.gmmInfo.aivCoreNum,
                           range)) {
        return false;
    }

    AscendC::GlobalTensor<bfloat16_t> input;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(gmmx));
    AscendC::GlobalTensor<bfloat16_t> localWindow;
    localWindow.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
        context.Window(context.RankId())));
    AlltoAllvGroupedMatMulAiv::CopyLocalGmToWindow(
        localWindow[range.offset], input[range.offset], range.count,
        tiling.gmmCocTiling.ubMoveNum, copyBuffer);

    if (taskIdx == 0U && subBlockIdx == 0U) {
        AscendC::GlobalTensor<int64_t> windowCounts;
        windowCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(
            context.Window(context.RankId()) + layout.countsOffset));
        AscendC::LocalTensor<int64_t> localCounts = countBuffer.Get<int64_t>();
        constexpr uint32_t countsPerBatch =
            kFlagBufferBytes / sizeof(int64_t);
        for (uint32_t offset = 0U; offset < tiling.countNum;
             offset += countsPerBatch) {
            const uint32_t current =
                (tiling.countNum - offset) > countsPerBatch
                    ? countsPerBatch : (tiling.countNum - offset);
            for (uint32_t index = 0U; index < current; ++index) {
                localCounts.SetValue(index, tiling.sendCounts[offset + index]);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
            AscendC::DataCopyExtParams params(
                1U, current * sizeof(int64_t), 0U, 0U, 0U);
            AscendC::DataCopyPad(windowCounts[offset], localCounts, params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        }
    }
    return true;
}

__aicore__ inline bool CopyExpertFromSource(
    GM_ADDR recvBuffer, const AlltoAllvGmmAivTilingData &tiling,
    const AlltoAllvGroupedMatMulAiv::PeerWindowContext &context,
    const AlltoAllvGroupedMatMulAiv::A2avWindowLayout &layout,
    uint32_t expertIdx, uint32_t sourceRank,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &copyBuffer)
{
    uint64_t destinationTokenOffset = 0U;
    if (!GetDestinationSourceTokenOffset(
            tiling.recvCounts, tiling.gmmInfo.rankSize,
            tiling.gmmInfo.expertPerRank, expertIdx, sourceRank,
            tiling.gmmInfo.maxOutputSize, destinationTokenOffset)) {
        return false;
    }

    AscendC::GlobalTensor<int64_t> peerCounts;
    peerCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(
        context.Window(sourceRank) + layout.countsOffset));
    uint64_t sourceTokenOffset = 0U;
    uint32_t sourceTokenCount = 0U;
    if (!GetPeerSourceTokenOffset(
            peerCounts, tiling.gmmInfo.rankSize,
            tiling.gmmInfo.expertPerRank, context.RankId(), expertIdx,
            sourceTokenOffset, sourceTokenCount)) {
        return false;
    }

    const uint32_t countIndex =
        sourceRank * tiling.gmmInfo.expertPerRank + expertIdx;
    const int64_t expectedCount = tiling.recvCounts[countIndex];
    if (expectedCount < 0 ||
        static_cast<uint64_t>(expectedCount) != sourceTokenCount) {
        return false;
    }

    if (sourceTokenOffset > 0xffffffffULL ||
        destinationTokenOffset > 0xffffffffULL) {
        return false;
    }
    const uint64_t sourceElementOffset =
        AlltoAllvGroupedMatMulAiv::MulU32ToU64(
            static_cast<uint32_t>(sourceTokenOffset), tiling.gmmInfo.K);
    const uint64_t destinationElementOffset =
        AlltoAllvGroupedMatMulAiv::MulU32ToU64(
            static_cast<uint32_t>(destinationTokenOffset), tiling.gmmInfo.K);
    const uint64_t elementCount =
        AlltoAllvGroupedMatMulAiv::MulU32ToU64(
            sourceTokenCount, tiling.gmmInfo.K);

    AscendC::GlobalTensor<bfloat16_t> peerInput;
    peerInput.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
        context.Window(sourceRank)));
    AscendC::GlobalTensor<bfloat16_t> destination;
    destination.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t *>(recvBuffer));
    AlltoAllvGroupedMatMulAiv::CopyPeerGmToLocalGm(
        destination[destinationElementOffset],
        peerInput[sourceElementOffset], elementCount,
        tiling.gmmCocTiling.ubMoveNum, copyBuffer);
    return true;
}

template <bool GMM_TRANSPOSE_B, bool MM_TRANSPOSE_B>
__aicore__ inline void ProcessAic(
    GM_ADDR gmmweight, GM_ADDR mmx, GM_ADDR mmweight, GM_ADDR gmmy,
    GM_ADDR mmy, GM_ADDR recvBuffer,
    const AlltoAllvGmmAivTilingData &tiling)
{
    AscendC::SetLoadDataPaddingValue(0);
    AscendC::SetAtomicNone();
    AscendC::SetFixpipeNz2ndFlag(1, 0, 0);

    Catlass::Arch::CrossCoreFlag dataReady{kAicDataReadyEventId};
    Catlass::Arch::CrossCoreWaitFlag(dataReady);

    for (uint32_t expertIdx = 0U;
         expertIdx < tiling.gmmInfo.expertPerRank; ++expertIdx) {
        ExpertMeta expert = {};
        if (!BuildExpertMetaForIndex(
                tiling.recvCounts, tiling.gmmInfo.rankSize,
                tiling.gmmInfo.expertPerRank, expertIdx,
                tiling.gmmInfo.maxOutputSize, expert)) {
            return;
        }
        if (!AlltoAllvGroupedMatMulCatlass::RunExpertGemm<
                GMM_TRANSPOSE_B>(recvBuffer, gmmweight, gmmy, expertIdx,
                                 expert, tiling.gmmInfo,
                                 tiling.gmmCocTiling)) {
            return;
        }
    }

    if (tiling.gmmInfo.hasSharedExpert != 0U) {
        if (mmx == nullptr || mmweight == nullptr || mmy == nullptr ||
            !AlltoAllvGroupedMatMulCatlass::RunSharedExpertGemm<
                MM_TRANSPOSE_B>(mmx, mmweight, mmy, tiling.mmInfo,
                                tiling.mmCocTiling)) {
            return;
        }
    }
}

__aicore__ inline void ProcessAiv(
    GM_ADDR gmmx, GM_ADDR recvBuffer,
    const AlltoAllvGmmAivTilingData &tiling,
    const AlltoAllvGroupedMatMulAiv::PeerWindowContext &context,
    const AlltoAllvGroupedMatMulAiv::A2avWindowLayout &layout,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &copyBuffer,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer)
{
    const uint32_t taskIdx = AscendC::GetBlockIdx();
    const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
    const uint32_t taskRatio = AscendC::GetTaskRation();
    const uint32_t blockIdx =
        taskRatio == 0U ? 0xffffffffU : taskIdx / taskRatio;

    int32_t epoch = 0;
    if (taskIdx == 0U && subBlockIdx == 0U) {
        epoch = AlltoAllvGroupedMatMulAiv::NextInvocationEpoch(
            context, layout, flagBuffer);
    }
    AscendC::SyncAll<true>();
    epoch = LoadInvocationEpoch(context, layout, flagBuffer);

    (void)CopyInitialWindowPayload(
        gmmx, tiling, context, layout, taskIdx, subBlockIdx, copyBuffer,
        flagBuffer);
    AscendC::SyncAll<true>();
    AlltoAllvGroupedMatMulAiv::PublishPeerPhase(
        context, layout, AlltoAllvGroupedMatMulAiv::kPublishSlotsOffset,
        epoch, blockIdx, subBlockIdx, flagBuffer);
    AscendC::SyncAll<true>();
    if (taskIdx == 0U && subBlockIdx == 0U) {
        AlltoAllvGroupedMatMulAiv::WaitPeerPhase(
            context, layout, AlltoAllvGroupedMatMulAiv::kPublishSlotsOffset,
            epoch, flagBuffer);
    }
    AscendC::SyncAll<true>();

    for (uint32_t expertIdx = 0U;
         expertIdx < tiling.gmmInfo.expertPerRank; ++expertIdx) {
        const uint32_t sourceRank = blockIdx;
        if (IsSourceWorker(taskIdx, taskRatio, subBlockIdx,
                           context.RankSize(), sourceRank)) {
            (void)CopyExpertFromSource(
                recvBuffer, tiling, context, layout, expertIdx, sourceRank,
                copyBuffer);
        }
        AscendC::SyncAll<true>();
    }

    SignalAicDataReady();

    AlltoAllvGroupedMatMulAiv::PublishPeerPhase(
        context, layout, AlltoAllvGroupedMatMulAiv::kReleaseSlotsOffset,
        epoch, blockIdx, subBlockIdx, flagBuffer);
    AscendC::SyncAll<true>();
    if (taskIdx == 0U && subBlockIdx == 0U) {
        AlltoAllvGroupedMatMulAiv::WaitPeerPhase(
            context, layout, AlltoAllvGroupedMatMulAiv::kReleaseSlotsOffset,
            epoch, flagBuffer);
    }
}

} // namespace detail

template <bool GMM_TRANSPOSE_B, bool MM_TRANSPOSE_B>
__aicore__ inline void Run(
    GM_ADDR gmmx, GM_ADDR gmmweight, GM_ADDR mmx, GM_ADDR mmweight,
    GM_ADDR gmmy, GM_ADDR mmy, GM_ADDR permuteOut, GM_ADDR userWorkspace,
    const AlltoAllvGmmAivTilingData &tiling)
{
    GM_ADDR recvBuffer =
        tiling.gmmInfo.hasPermuteOut != 0U && permuteOut != nullptr
            ? permuteOut
            : userWorkspace + tiling.recvTokenOffset;

    if ASCEND_IS_AIV {
        AlltoAllvGroupedMatMulAiv::PeerWindowContext context;
        if (!context.Init() ||
            context.RankSize() != tiling.gmmInfo.rankSize) {
            return;
        }

        AlltoAllvGroupedMatMulAiv::A2avWindowLayout layout = {};
        uint64_t inputTokens = 0U;
        for (uint32_t index = 0U; index < tiling.countNum; ++index) {
            const int64_t count = tiling.sendCounts[index];
            if (count < 0 ||
                !AlltoAllvGroupedMatMulAiv::SafeAddU64(
                    inputTokens, static_cast<uint64_t>(count), inputTokens)) {
                return;
            }
        }
        if (!detail::BuildDeviceWindowLayout(
                inputTokens, tiling.gmmInfo.K, tiling.countNum,
                context.WindowBytes(), layout)) {
            return;
        }
        AscendC::TPipe pipe;

        AscendC::TBuf<AscendC::TPosition::VECCALC> flagBuffer;
        pipe.InitBuffer(flagBuffer, detail::kFlagBufferBytes);
        AscendC::TBuf<AscendC::TPosition::VECCALC> copyBuffer;
        const uint32_t queueBytes = detail::GetCopyQueueBytes(tiling);
        if (queueBytes == 0U) {
            return;
        }
        pipe.InitBuffer(copyBuffer, queueBytes);
        detail::ProcessAiv(
            gmmx, recvBuffer, tiling, context, layout, copyBuffer,
            flagBuffer);
    }
    if ASCEND_IS_AIC {
        detail::ProcessAic<GMM_TRANSPOSE_B, MM_TRANSPOSE_B>(
            gmmweight, mmx, mmweight, gmmy, mmy, recvBuffer, tiling);
    }
}

} // namespace AlltoAllvGroupedMatMulAivMode

#endif // defined(__CCE_AICORE__) && !defined(__CCE_KT_TEST__)

#endif // ALLTO_ALLV_GROUPED_MAT_MUL_AIV_MODE_H
