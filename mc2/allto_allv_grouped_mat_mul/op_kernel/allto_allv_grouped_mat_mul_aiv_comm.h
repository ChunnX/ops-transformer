/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 */

#ifndef ALLTO_ALLV_GROUPED_MAT_MUL_AIV_COMM_H
#define ALLTO_ALLV_GROUPED_MAT_MUL_AIV_COMM_H

#include <cstdint>

namespace AlltoAllvGroupedMatMulAiv {

constexpr uint32_t kMaxRankSize = 8U;
constexpr uint64_t kWindowAlignment = 512U;
constexpr uint64_t kDefaultWindowBytes = 200U * 1024U * 1024U;
constexpr uint64_t kControlFlagStride = 32U;
constexpr uint64_t kEpochOffset = 0U;
constexpr uint64_t kPublishSlotsOffset = kControlFlagStride;
constexpr uint64_t kReleaseSlotsOffset = kPublishSlotsOffset + kMaxRankSize * kControlFlagStride;
constexpr uint64_t kFirstExpertFlagOffset = kReleaseSlotsOffset + kMaxRankSize * kControlFlagStride;
constexpr uint64_t kBfloat16Bytes = 2U;

struct ExpertMeta {
    uint64_t recvTokenBase = 0U;
    uint32_t tokenCount = 0U;
    uint32_t reserved = 0U;
};

struct ExpertSourceMeta {
    uint64_t dstTokenOffset = 0U;
    uint64_t srcTokenOffset = 0U;
    uint32_t tokenCount = 0U;
    uint32_t sourceRank = 0U;
};

struct A2avWindowLayout {
    uint64_t inputBytes = 0U;
    uint64_t countsOffset = 0U;
    uint64_t controlOffset = 0U;
    uint64_t readyOffset = 0U;
    uint64_t totalBytes = 0U;
};

#if defined(__CCE_AICORE__)
#define A2AVGMM_HOST_DEVICE __aicore__ inline
#else
#define A2AVGMM_HOST_DEVICE inline
#endif

A2AVGMM_HOST_DEVICE bool SafeAddU64(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    constexpr uint64_t maxValue = ~static_cast<uint64_t>(0U);
    if (lhs > maxValue - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

A2AVGMM_HOST_DEVICE bool SafeMulU64(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    constexpr uint64_t maxValue = ~static_cast<uint64_t>(0U);
    if (lhs != 0U && rhs > maxValue / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}


A2AVGMM_HOST_DEVICE uint64_t MulU32ToU64(uint32_t lhs, uint32_t rhs)
{
    const uint32_t lhsLow = lhs & 0xffffU;
    const uint32_t lhsHigh = lhs >> 16U;
    const uint32_t rhsLow = rhs & 0xffffU;
    const uint32_t rhsHigh = rhs >> 16U;
    const uint64_t low = static_cast<uint64_t>(lhsLow * rhsLow);
    const uint64_t middle =
        static_cast<uint64_t>(lhsLow * rhsHigh) +
        static_cast<uint64_t>(lhsHigh * rhsLow);
    const uint64_t high = static_cast<uint64_t>(lhsHigh * rhsHigh);
    return low + (middle << 16U) + (high << 32U);
}

A2AVGMM_HOST_DEVICE bool AlignUpU64(uint64_t value, uint64_t alignment, uint64_t &result)
{
    if (alignment == 0U || (alignment & (alignment - 1U)) != 0U) {
        return false;
    }
    uint64_t valueWithPadding = 0U;
    if (!SafeAddU64(value, alignment - 1U, valueWithPadding)) {
        return false;
    }
    result = valueWithPadding & ~(alignment - 1U);
    return true;
}

A2AVGMM_HOST_DEVICE uint64_t AlignDownU64(uint64_t value, uint64_t alignment)
{
    return value & ~(alignment - 1U);
}

A2AVGMM_HOST_DEVICE uint64_t ExpertReadyBytes(uint32_t expertPerRank)
{
    return static_cast<uint64_t>(expertPerRank) * kControlFlagStride;
}

A2AVGMM_HOST_DEVICE uint64_t ExpertReadyOffset(uint64_t readyBase, uint32_t expertIdx)
{
    return readyBase + static_cast<uint64_t>(expertIdx) * kControlFlagStride;
}

A2AVGMM_HOST_DEVICE bool BuildExpertMetadata(const int64_t *recvCounts, uint32_t rankSize,
                                             uint32_t expertPerRank, uint64_t tokenCapacity,
                                             ExpertMeta *expertMeta, ExpertSourceMeta *sourceMeta)
{
    if (recvCounts == nullptr || expertMeta == nullptr || sourceMeta == nullptr || rankSize == 0U ||
        expertPerRank == 0U) {
        return false;
    }

    uint64_t expertBase = 0U;
    for (uint32_t expertIdx = 0U; expertIdx < expertPerRank; ++expertIdx) {
        uint64_t expertTokens = 0U;
        for (uint32_t sourceRank = 0U; sourceRank < rankSize; ++sourceRank) {
            const int64_t count = recvCounts[static_cast<uint64_t>(sourceRank) * expertPerRank + expertIdx];
            if (count < 0 || static_cast<uint64_t>(count) > 0xffffffffULL) {
                return false;
            }

            ExpertSourceMeta &source = sourceMeta[static_cast<uint64_t>(expertIdx) * rankSize + sourceRank];
            source.dstTokenOffset = expertBase + expertTokens;
            source.srcTokenOffset = 0U;
            source.tokenCount = static_cast<uint32_t>(count);
            source.sourceRank = sourceRank;
            if (!SafeAddU64(expertTokens, static_cast<uint64_t>(count), expertTokens)) {
                return false;
            }
        }

        if (expertTokens > 0xffffffffULL || expertBase > tokenCapacity ||
            expertTokens > tokenCapacity - expertBase) {
            return false;
        }
        expertMeta[expertIdx].recvTokenBase = expertBase;
        expertMeta[expertIdx].tokenCount = static_cast<uint32_t>(expertTokens);
        expertMeta[expertIdx].reserved = 0U;
        expertBase += expertTokens;
    }
    return expertBase == tokenCapacity;
}

A2AVGMM_HOST_DEVICE bool GetPeerSourceTokenOffset(const int64_t *peerSendCounts, uint32_t rankSize,
                                                  uint32_t expertPerRank, uint32_t dstRank,
                                                  uint32_t expertIdx, uint64_t &tokenOffset)
{
    if (peerSendCounts == nullptr || rankSize == 0U || expertPerRank == 0U || dstRank >= rankSize ||
        expertIdx >= expertPerRank) {
        return false;
    }

    const uint64_t target = static_cast<uint64_t>(dstRank) * expertPerRank + expertIdx;
    if (peerSendCounts[target] < 0 || static_cast<uint64_t>(peerSendCounts[target]) > 0xffffffffULL) {
        return false;
    }
    tokenOffset = 0U;
    for (uint64_t index = 0U; index < target; ++index) {
        const int64_t count = peerSendCounts[index];
        if (count < 0 || !SafeAddU64(tokenOffset, static_cast<uint64_t>(count), tokenOffset)) {
            return false;
        }
    }
    return true;
}

A2AVGMM_HOST_DEVICE bool BuildWindowLayout(uint64_t inputTokenNum, uint64_t hiddenSize, uint64_t countNum,
                                           uint64_t controlBytes, uint64_t windowBytes,
                                           A2avWindowLayout &layout)
{
    uint64_t inputElements = 0U;
    uint64_t rawInputBytes = 0U;
    uint64_t rawCountBytes = 0U;
    uint64_t countAreaBytes = 0U;
    uint64_t requiredBytes = 0U;
    uint64_t payloadAndCounts = 0U;
    if (!SafeMulU64(inputTokenNum, hiddenSize, inputElements) ||
        !SafeMulU64(inputElements, kBfloat16Bytes, rawInputBytes) ||
        !AlignUpU64(rawInputBytes, kWindowAlignment, layout.inputBytes) ||
        !SafeMulU64(countNum, sizeof(int64_t), rawCountBytes) ||
        !AlignUpU64(rawCountBytes, kWindowAlignment, countAreaBytes) ||
        !SafeAddU64(layout.inputBytes, countAreaBytes, payloadAndCounts) ||
        !SafeAddU64(payloadAndCounts, controlBytes, requiredBytes)) {
        return false;
    }
    if (controlBytes < kFirstExpertFlagOffset || requiredBytes > windowBytes ||
        controlBytes > windowBytes) {
        return false;
    }

    layout.controlOffset = AlignDownU64(windowBytes - controlBytes, kWindowAlignment);
    if (layout.controlOffset < countAreaBytes) {
        return false;
    }
    layout.countsOffset = layout.controlOffset - countAreaBytes;
    if (layout.inputBytes > layout.countsOffset) {
        return false;
    }
    layout.readyOffset = kFirstExpertFlagOffset;
    layout.totalBytes = requiredBytes;
    return true;
}

#undef A2AVGMM_HOST_DEVICE

} // namespace AlltoAllvGroupedMatMulAiv

#if defined(__CCE_AICORE__)

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"
#include "../../common/op_kernel/moe_distribute_base.h"

namespace AlltoAllvGroupedMatMulAiv {

class PeerWindowContext {
public:
    __aicore__ inline bool Init()
    {
        GM_ADDR contextAddress = AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>();
        if (contextAddress == nullptr) {
            return false;
        }
        context_ = reinterpret_cast<__gm__ HcclOpResParam *>(contextAddress);
        rankId_ = context_->localUsrRankId;
        rankSize_ = context_->rankSize;
        windowBytes_ = context_->winSize == 0U ? kDefaultWindowBytes : context_->winSize;
        if (rankSize_ == 0U || rankSize_ > kMaxRankSize || rankId_ >= rankSize_ || windowBytes_ == 0U) {
            return false;
        }

        for (uint32_t rank = 0U; rank < rankSize_; ++rank) {
            if (rank == rankId_) {
                windows_[rank] = reinterpret_cast<GM_ADDR>(context_->localWindowsIn);
            } else {
                __gm__ HcclRankRelationResV2 *relation =
                    reinterpret_cast<__gm__ HcclRankRelationResV2 *>(context_->remoteRes[rank].nextDevicePtr);
                windows_[rank] = reinterpret_cast<GM_ADDR>(relation->windowsIn);
            }
        }
        return true;
    }

    __aicore__ inline uint32_t RankId() const
    {
        return rankId_;
    }

    __aicore__ inline uint32_t RankSize() const
    {
        return rankSize_;
    }

    __aicore__ inline uint64_t WindowBytes() const
    {
        return windowBytes_;
    }

    __aicore__ inline GM_ADDR Window(uint32_t rank) const
    {
        return windows_[rank];
    }

private:
    __gm__ HcclOpResParam *context_ = nullptr;
    GM_ADDR windows_[kMaxRankSize] = {};
    uint32_t rankId_ = 0U;
    uint32_t rankSize_ = 0U;
    uint64_t windowBytes_ = 0U;
};

template <typename T>
__aicore__ inline void CopyPeerGmToLocalGm(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                            uint64_t count, uint32_t ubMoveNum,
                                            AscendC::TBuf<AscendC::TPosition::VECCALC> &buffer)
{
    if (count == 0U || ubMoveNum == 0U) {
        return;
    }

    AscendC::LocalTensor<T> local = buffer.template Get<T>();
    for (uint64_t offset = 0U; offset < count; offset += ubMoveNum) {
        const uint32_t current =
            static_cast<uint32_t>((count - offset) > ubMoveNum ? ubMoveNum : (count - offset));
        const uint32_t copyBytes = current * sizeof(T);
        AscendC::DataCopyExtParams copyParams(1U, copyBytes, 0U, 0U, 0U);
        AscendC::DataCopyPadExtParams<T> padParams;
        AscendC::DataCopyPad(local, src[offset], copyParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(dst[offset], local, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

template <typename T>
__aicore__ inline void CopyLocalGmToWindow(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                           uint64_t count, uint32_t ubMoveNum,
                                           AscendC::TBuf<AscendC::TPosition::VECCALC> &buffer)
{
    CopyPeerGmToLocalGm(dst, src, count, ubMoveNum, buffer);
}

__aicore__ inline __gm__ int32_t *GetFlagAddress(GM_ADDR window, uint64_t controlOffset, uint64_t flagOffset)
{
    return reinterpret_cast<__gm__ int32_t *>(window + controlOffset + flagOffset);
}

__aicore__ inline void StoreFlag(__gm__ int32_t *address,
                                  AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer, int32_t value)
{
    AscendC::LocalTensor<int32_t> local = flagBuffer.Get<int32_t>();
    local.SetValue(0U, value);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
    AscendC::GlobalTensor<int32_t> global;
    global.SetGlobalBuffer(address);
    AscendC::DataCopyExtParams copyParams(1U, sizeof(int32_t), 0U, 0U, 0U);
    AscendC::DataCopyPad(global, local, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
}

__aicore__ inline int32_t LoadFlag(__gm__ int32_t *address,
                                   AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer)
{
    AscendC::LocalTensor<int32_t> local = flagBuffer.Get<int32_t>();
    AscendC::GlobalTensor<int32_t> global;
    global.SetGlobalBuffer(address);
    AscendC::DataCopyExtParams copyParams(1U, sizeof(int32_t), 0U, 0U, 0U);
    AscendC::DataCopyPadExtParams<int32_t> padParams;
    AscendC::DataCopyPad(local, global, copyParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
    return local.GetValue(0U);
}

__aicore__ inline void WaitFlagValue(__gm__ int32_t *address,
                                     AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer, int32_t value)
{
    while (LoadFlag(address, flagBuffer) != value) {
    }
}

__aicore__ inline int32_t NextInvocationEpoch(const PeerWindowContext &context, const A2avWindowLayout &layout,
                                              AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer)
{
    __gm__ int32_t *epochAddress = GetFlagAddress(context.Window(context.RankId()), layout.controlOffset, kEpochOffset);
    const int32_t current = LoadFlag(epochAddress, flagBuffer);
    const int32_t next = (current <= 0 || current == 0x7fffffff) ? 1 : current + 1;
    StoreFlag(epochAddress, flagBuffer, next);
    return next;
}

__aicore__ inline void PublishPeerPhase(const PeerWindowContext &context, const A2avWindowLayout &layout,
                                        uint64_t phaseOffset, int32_t epoch, uint32_t blockIdx, uint32_t aivIdx,
                                        AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer)
{
    if (aivIdx != 0U || blockIdx >= context.RankSize()) {
        return;
    }
    __gm__ int32_t *slot = GetFlagAddress(
        context.Window(blockIdx), layout.controlOffset,
        phaseOffset + static_cast<uint64_t>(context.RankId()) * kControlFlagStride);
    StoreFlag(slot, flagBuffer, epoch);
}

__aicore__ inline void WaitPeerPhase(const PeerWindowContext &context, const A2avWindowLayout &layout,
                                     uint64_t phaseOffset, int32_t epoch,
                                     AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer)
{
    for (uint32_t sourceRank = 0U; sourceRank < context.RankSize(); ++sourceRank) {
        __gm__ int32_t *slot = GetFlagAddress(
            context.Window(context.RankId()), layout.controlOffset,
            phaseOffset + static_cast<uint64_t>(sourceRank) * kControlFlagStride);
        WaitFlagValue(slot, flagBuffer, epoch);
    }
}

__aicore__ inline void PublishExpertReady(GM_ADDR workspace, uint64_t readyBase, uint32_t expertIdx,
                                          int32_t epoch,
                                          AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer)
{
    __gm__ int32_t *ready = reinterpret_cast<__gm__ int32_t *>(
        workspace + ExpertReadyOffset(readyBase, expertIdx));
    StoreFlag(ready, flagBuffer, epoch);
}

__aicore__ inline void WaitExpertReady(GM_ADDR workspace, uint64_t readyBase, uint32_t expertIdx,
                                       int32_t epoch,
                                       AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer)
{
    __gm__ int32_t *ready = reinterpret_cast<__gm__ int32_t *>(
        workspace + ExpertReadyOffset(readyBase, expertIdx));
    WaitFlagValue(ready, flagBuffer, epoch);
}

} // namespace AlltoAllvGroupedMatMulAiv

#endif // defined(__CCE_AICORE__)

#endif // ALLTO_ALLV_GROUPED_MAT_MUL_AIV_COMM_H
