/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_KERNEL_CONTEXT_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_KERNEL_CONTEXT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "compute/rms_norm_common.h"

namespace MegaMhc::Kernel {
using namespace AscendC;

constexpr uint32_t kRowsPerMSubblock = 16;
constexpr uint16_t kPairReadyFlag = 8;
constexpr uint16_t kPairConsumedFlag = 9;
constexpr uint16_t kSubblockFlagOffset = 16;
constexpr uint8_t kPairSyncMode = 4;

template <typename T>
struct Tensors {
    GlobalTensor<T> hOut;
    GlobalTensor<T> residual;
    GlobalTensor<float> hPost;
    GlobalTensor<float> hRes;
    GlobalTensor<float> phi;
    GlobalTensor<float> alpha;
    GlobalTensor<float> bias;
    GlobalTensor<float> gamma1;
    GlobalTensor<float> gamma0;
    GlobalTensor<float> gamma2;
    GlobalTensor<T> hInPrime;
    GlobalTensor<float> hPostNew;
    GlobalTensor<float> hResNew;
    GlobalTensor<T> x2Out;
    GlobalTensor<float> invRms0;
    GlobalTensor<float> partialMm;
    GlobalTensor<float> partialRms;
    GlobalTensor<float> stage;
};

/** Allocate UB with 256B padding: each DMA base is aligned and small vectors can load a full register. */
__aicore__ inline void InitUbBuffer(TPipe *pipe, TBuf<TPosition::VECCALC> &buffer, uint32_t bytes)
{
    pipe->InitBuffer(buffer, (bytes + 255U) / 256U * 256U);
}

/** Enqueue destination[i] = source[sourceOffset+i]; caller owns UB reuse and completion events. */
template <typename T>
__aicore__ inline void EnqueueGlobalToUb(const LocalTensor<T> &destination, const GlobalTensor<T> &source,
                                         uint64_t sourceOffset, uint32_t elementCount)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(elementCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(destination, source[sourceOffset], copyParams, padParams);
}

/** Copy a contiguous GM range into UB and wait until MTE2 has made it visible to RegBase. */
template <typename T>
__aicore__ inline void CopyGlobalToUb(const LocalTensor<T> &destination, const GlobalTensor<T> &source,
                                      uint64_t sourceOffset, uint32_t elementCount)
{
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    EnqueueGlobalToUb(destination, source, sourceOffset, elementCount);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
}

/** Publish a contiguous UB range to GM after RegBase and MTE3 have completed the store. */
template <typename T>
__aicore__ inline void CopyUbToGlobal(const GlobalTensor<T> &destination, uint64_t destinationOffset,
                                      const LocalTensor<T> &source, uint32_t elementCount)
{
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(elementCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(destination[destinationOffset], source, copyParams);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

/** Initialize one or more FP32 values before an ordered Vector reduction. */
__aicore__ inline void ZeroFp32Values(__ubuf__ float *values, uint32_t valueCount)
{
    const uint16_t repeats = static_cast<uint16_t>((valueCount + 63U) / 64U);
    __VEC_SCOPE__
    {
        Reg::MaskReg mask;
        Reg::RegTensor<float> zero;
        Reg::MaskReg fullMask = Compute::MaskForCount(64U);
        Reg::Duplicate(zero, 0.0f, fullMask);
        uint32_t remaining = valueCount;
        for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
            mask = Reg::UpdateMask<float>(remaining);
            Reg::StoreAlign<float>(values + repeat * 64U, zero, mask);
        }
    }
}

/** Copy count <= 64 FP32 values into aligned DMA scratch; scratch[i] = source[i]. */
__aicore__ inline void PackSmallFp32(__ubuf__ float *source, __ubuf__ float *scratch, uint32_t count)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> value;
        Reg::RegTensor<uint32_t> indices;
        Reg::MaskReg mask = Compute::MaskForCount(count);
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t> &>(indices), 0);
        Reg::Gather(value, source, indices, mask);
        Reg::Store<float>(scratch, value, count);
    }
}

/** Locate a shard from host-planned long/short lengths: offset=min(i,L)*long+max(i-L,0)*short. */
__aicore__ inline uint64_t PlannedKStart(const MegaMhcTilingData &tiling, uint32_t kIndex)
{
    return kIndex < tiling.largeKCoreCount ?
               static_cast<uint64_t>(kIndex) * tiling.splitK :
               static_cast<uint64_t>(tiling.largeKCoreCount) * tiling.splitK +
                   static_cast<uint64_t>(kIndex - tiling.largeKCoreCount) * tiling.shortSplitK;
}

/** Return the host-planned number of K tiles for this K shard without choosing new tile sizes. */
__aicore__ inline uint32_t PlannedL1TileCount(const MegaMhcTilingData &tiling, uint32_t kIndex)
{
    const bool lastKCore = kIndex + 1U == tiling.kGrid;
    const uint32_t fullLoops =
        lastKCore ? tiling.lastKCoreL1FullLoops :
                    (kIndex < tiling.largeKCoreCount ? tiling.l1KFullLoops : tiling.shortKCoreL1FullLoops);
    const uint32_t tail = lastKCore ? tiling.lastKCoreL1Tail : tiling.l1KTail;
    return fullLoops + static_cast<uint32_t>(tail != 0U);
}

/** Select the precomputed K length for one L1 tile without rebuilding its descriptor. */
__aicore__ inline uint32_t PlannedL1TileKCount(const MegaMhcTilingData &tiling, uint32_t kIndex, uint32_t tileIndex)
{
    const bool lastKCore = kIndex + 1U == tiling.kGrid;
    const uint32_t fullLoops =
        lastKCore ? tiling.lastKCoreL1FullLoops :
                    (kIndex < tiling.largeKCoreCount ? tiling.l1KFullLoops : tiling.shortKCoreL1FullLoops);
    const uint32_t tail = lastKCore ? tiling.lastKCoreL1Tail : tiling.l1KTail;
    return tileIndex < fullLoops ? tiling.kL1 : tail;
}

/** Select the precomputed K, L0 and Mmad descriptor for one L1 transfer index. */
__aicore__ inline void GetPlannedL1Tile(const MegaMhcTilingData &tiling, uint32_t kIndex, uint32_t tileIndex,
                                        uint32_t &kCount, uint32_t &l0FullLoops, uint32_t &l0Tail,
                                        uint32_t &tailLoadKStep, uint32_t &mmadFullLoops, uint32_t &mmadTailLoops,
                                        uint32_t &tailMmadKStep, uint32_t &tailMmadFinalKStep)
{
    const bool lastKCore = kIndex + 1U == tiling.kGrid;
    const uint32_t fullLoops =
        lastKCore ? tiling.lastKCoreL1FullLoops :
                    (kIndex < tiling.largeKCoreCount ? tiling.l1KFullLoops : tiling.shortKCoreL1FullLoops);
    const uint32_t tail = lastKCore ? tiling.lastKCoreL1Tail : tiling.l1KTail;
    if (tileIndex < fullLoops) {
        kCount = tiling.kL1;
        l0FullLoops = tiling.l0KFullLoops;
        l0Tail = tiling.l0KTail;
        tailLoadKStep = tiling.l0TailLoadKStep;
    } else {
        kCount = tail;
        if (lastKCore) {
            l0FullLoops = tiling.lastCoreTailL1L0FullLoops;
            l0Tail = tiling.lastCoreTailL1L0Tail;
            tailLoadKStep = tiling.lastCoreTailLoadKStep;
        } else {
            l0FullLoops = tiling.tailL1L0FullLoops;
            l0Tail = tiling.tailL1L0Tail;
            tailLoadKStep = tiling.tailL1TailLoadKStep;
        }
    }
    // The planner owns both loop counts; only the descriptor changes for the
    // final L1 transfer on the last K shard.
    mmadFullLoops = tiling.fullMmadKLoopCount;
    if (tileIndex < fullLoops) {
        mmadTailLoops = tiling.l0TailMmadKLoopCount;
        tailMmadKStep = tiling.l0TailMmadKStep;
        tailMmadFinalKStep = tiling.l0TailMmadFinalKStep;
    } else if (lastKCore) {
        mmadTailLoops = tiling.lastCoreTailMmadKLoopCount;
        tailMmadKStep = tiling.lastCoreTailMmadKStep;
        tailMmadFinalKStep = tiling.lastCoreTailMmadFinalKStep;
    } else {
        mmadTailLoops = tiling.tailL1TailMmadKLoopCount;
        tailMmadKStep = tiling.tailL1TailMmadKStep;
        tailMmadFinalKStep = tiling.tailL1TailMmadFinalKStep;
    }
    if (l0Tail == 0U) {
        mmadTailLoops = 0U;
        tailMmadFinalKStep = 0U;
    }
}

} // namespace MegaMhc::Kernel

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_KERNEL_CONTEXT_H
