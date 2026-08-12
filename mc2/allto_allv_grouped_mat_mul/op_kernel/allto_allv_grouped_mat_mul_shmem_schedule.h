/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ALLTO_ALLV_GROUPED_MAT_MUL_SHMEM_SCHEDULE_H
#define ALLTO_ALLV_GROUPED_MAT_MUL_SHMEM_SCHEDULE_H

#include <cstdint>

#if defined(__CCE_AICORE__)
#define A2AV_GMM_SHMEM_HOST_DEVICE __aicore__
#else
#define A2AV_GMM_SHMEM_HOST_DEVICE
#endif

namespace AlltoAllvGmmShmem {

struct RoundExpertSlice {
    uint32_t stageRow;
    uint32_t expertRow;
    uint32_t rows;
};

A2AV_GMM_SHMEM_HOST_DEVICE inline uint64_t CeilDivU64(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

A2AV_GMM_SHMEM_HOST_DEVICE inline uint64_t ClipSubU64(uint64_t lhs, uint64_t rhs)
{
    return lhs > rhs ? lhs - rhs : 0;
}

A2AV_GMM_SHMEM_HOST_DEVICE inline uint64_t AlignUpU64(uint64_t value, uint64_t alignment)
{
    return CeilDivU64(value, alignment) * alignment;
}

// Intersect one expert's rows in a peer stream with a communication round.
A2AV_GMM_SHMEM_HOST_DEVICE inline RoundExpertSlice GetRoundExpertSlice(uint64_t expertBegin, uint64_t expertRows,
                                                                       uint64_t roundBegin, uint64_t roundRows)
{
    const uint64_t expertEnd = expertBegin + expertRows;
    const uint64_t roundEnd = roundBegin + roundRows;
    const uint64_t overlapBegin = expertBegin > roundBegin ? expertBegin : roundBegin;
    const uint64_t overlapEnd = expertEnd < roundEnd ? expertEnd : roundEnd;
    if (overlapEnd <= overlapBegin) {
        return {0, 0, 0};
    }
    return {static_cast<uint32_t>(overlapBegin - roundBegin),
            static_cast<uint32_t>(overlapBegin - expertBegin),
            static_cast<uint32_t>(overlapEnd - overlapBegin)};
}

// Source layout: [stage, source rank, comm row, K], with every K row aligned to 512 bytes.
A2AV_GMM_SHMEM_HOST_DEVICE inline uint64_t GetStagePeerOffset(uint32_t stage, uint32_t peer, uint32_t peerCount,
                                                              uint32_t commRows, uint32_t k, uint32_t elementBytes)
{
    const uint64_t elementsPer512B = 512U / elementBytes;
    const uint64_t alignedK = AlignUpU64(k, elementsPer512B);
    return (static_cast<uint64_t>(stage) * peerCount + peer) * commRows * alignedK;
}

A2AV_GMM_SHMEM_HOST_DEVICE inline uint32_t GetCompletedCommBlocks(uint32_t rows, uint32_t commBlockM)
{
    return static_cast<uint32_t>(CeilDivU64(rows, commBlockM));
}

A2AV_GMM_SHMEM_HOST_DEVICE inline uint32_t GetPeerTokenCount(const uint32_t *counts, uint32_t peer,
                                                             uint32_t localExpertNum)
{
    uint32_t result = 0;
    const uint32_t begin = peer * localExpertNum;
    for (uint32_t expert = 0; expert < localExpertNum; ++expert) {
        result += counts[begin + expert];
    }
    return result;
}

A2AV_GMM_SHMEM_HOST_DEVICE inline uint64_t GetPeerInputOffset(const uint32_t *sendCounts, uint32_t peer,
                                                              uint32_t localExpertNum)
{
    uint64_t result = 0;
    for (uint32_t index = 0; index < peer * localExpertNum; ++index) {
        result += sendCounts[index];
    }
    return result;
}

// The source operator emits expert-major rows, with source rank as the inner group.
A2AV_GMM_SHMEM_HOST_DEVICE inline uint64_t GetExpertOutputOffset(const uint32_t *recvCounts, uint32_t expert,
                                                                 uint32_t srcPeer, uint32_t peerCount,
                                                                 uint32_t localExpertNum)
{
    uint64_t result = 0;
    for (uint32_t previousExpert = 0; previousExpert < expert; ++previousExpert) {
        for (uint32_t peer = 0; peer < peerCount; ++peer) {
            result += recvCounts[peer * localExpertNum + previousExpert];
        }
    }
    for (uint32_t peer = 0; peer < srcPeer; ++peer) {
        result += recvCounts[peer * localExpertNum + expert];
    }
    return result;
}

A2AV_GMM_SHMEM_HOST_DEVICE inline uint64_t GetExpertBeginInPeer(const uint32_t *counts, uint32_t peer,
                                                                uint32_t expert, uint32_t localExpertNum)
{
    uint64_t result = 0;
    const uint32_t begin = peer * localExpertNum;
    for (uint32_t previousExpert = 0; previousExpert < expert; ++previousExpert) {
        result += counts[begin + previousExpert];
    }
    return result;
}

} // namespace AlltoAllvGmmShmem

#undef A2AV_GMM_SHMEM_HOST_DEVICE

#endif // ALLTO_ALLV_GROUPED_MAT_MUL_SHMEM_SCHEDULE_H
