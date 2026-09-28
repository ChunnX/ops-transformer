/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_TRANSFER_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_TRANSFER_H

#include "execution_policy.h"
#include "mega_mhc_kernel_context.h"
#include "compute/projection_cube.h"

namespace MegaMhc::Transfer {
using namespace AscendC;

/** GM-backed transfer policy for AIV UB -> staging GM -> AIC L1, with pair-mode credit handshakes. */
class GmStage {
public:
    /** Allocate two empty-slot credits for each of the paired AIV producers. */
    __aicore__ inline static void SeedPairCredits()
    {
        if ASCEND_IS_AIC {
            for (uint32_t subblock = 0; subblock < 2U; ++subblock) {
                const uint16_t flag = Kernel::kPairConsumedFlag + subblock * Kernel::kSubblockFlagOffset;
                CrossCoreSetFlag<Kernel::kPairSyncMode, PIPE_MTE1>(flag);
                CrossCoreSetFlag<Kernel::kPairSyncMode, PIPE_MTE1>(flag);
            }
        }
    }

    /** Store one AIV-owned FP32 row slice into the selected GM staging slot. */
    __aicore__ inline static void StoreAivRow(const GlobalTensor<float> &stage, uint64_t offset,
                                              const LocalTensor<float> &tile, uint32_t elements,
                                              const MegaMhcTilingData &tiling)
    {
        Kernel::CopyUbToGlobal<float>(stage, offset, tile, elements);
    }

    /** Enqueue stage[offset+i]=tile[i]; event 7 is consumed before this UB slot is reused.
     * The outer scheduler publishes READY on MTE3 after this transfer and drains the event at exit.
     */
    __aicore__ inline static void StoreAivRowAsync(const GlobalTensor<float> &stage, uint64_t offset,
                                                   const LocalTensor<float> &tile, uint32_t elements,
                                                   const MegaMhcTilingData &tiling)
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID7);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID7);
        DataCopyExtParams params{1U, static_cast<uint32_t>(elements * sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(stage[offset], tile, params);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
    }

    /** Acquire one FIFO slot credit BEFORE any MTE3 write can overwrite that slot. */
    __aicore__ inline static void AcquireSlot()
    {
        CrossCoreWaitFlag<Kernel::kPairSyncMode, PIPE_MTE3>(Kernel::kPairConsumedFlag);
    }

    /** Publish READY after all row stores; AIV uses local flag IDs in pair mode. */
    __aicore__ inline static void PublishReady()
    {
        CrossCoreSetFlag<Kernel::kPairSyncMode, PIPE_MTE3>(Kernel::kPairReadyFlag);
    }

    /** Drain the two returned credits, including credits for empty halves and waves. */
    __aicore__ inline static void DrainCredits()
    {
        CrossCoreWaitFlag<Kernel::kPairSyncMode, PIPE_MTE3>(Kernel::kPairConsumedFlag);
        CrossCoreWaitFlag<Kernel::kPairSyncMode, PIPE_MTE3>(Kernel::kPairConsumedFlag);
    }

    /** Wait for one AIV subblock's staging transfer to become visible to AIC. */
    __aicore__ inline static void WaitReady(uint32_t subblock)
    {
        const uint16_t ready = Kernel::kPairReadyFlag + subblock * Kernel::kSubblockFlagOffset;
        CrossCoreWaitFlag<Kernel::kPairSyncMode, PIPE_MTE2>(ready);
    }

    /** Return a slot credit only after AIC has consumed both staged halves. */
    __aicore__ inline static void ReturnCredit(uint32_t subblock)
    {
        const uint16_t consumed = Kernel::kPairConsumedFlag + subblock * Kernel::kSubblockFlagOffset;
        CrossCoreSetFlag<Kernel::kPairSyncMode, PIPE_MTE1>(consumed);
    }

    /** Copy one completed GM slot into the host-sized L1 A NZ buffer. */
    template <bool USE_HF32>
    __aicore__ inline static void LoadA1(Cube::ProjectionCube<USE_HF32> &cube, const GlobalTensor<float> &stage,
                                         uint64_t stageOffset, uint32_t mRows, uint32_t nzM, uint32_t kCount,
                                         uint32_t slot)
    {
        cube.LoadA(stage, stageOffset, mRows, nzM, kCount, slot);
    }
};

/** Pair-mode UB -> L1 transfer; each eight-float source block maps to NZ[k/8,m,8]. */
class UbL1Stage : public GmStage {
public:
    /** Store packed FP32 rows directly into the AIC L1 slot, without a GM staging round trip. */
    __aicore__ inline static void StoreAivRow(const GlobalTensor<float> &stage, uint64_t offset,
                                              const LocalTensor<float> &tile, uint32_t elements,
                                              const MegaMhcTilingData &tiling)
    {
        const uint64_t pairOffset = offset % (static_cast<uint64_t>(tiling.slotCount) * tiling.mTile * tiling.kL1);
        const uint32_t slot = pairOffset / (tiling.mTile * tiling.kL1);
        const uint32_t firstRow = (pairOffset % (tiling.mTile * tiling.kL1)) / tiling.kL1;
        LocalTensor<float> destination(
            TPosition::A1, (static_cast<uint64_t>(slot) * tiling.l1ASlotElements + firstRow * 8U) * sizeof(float),
            tiling.l1ASlotElements);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
        const uint32_t rows = (elements + tiling.kL1 - 1U) / tiling.kL1;
        DataCopyParams params{static_cast<uint16_t>(tiling.kL1 / 8U), 1, 0, static_cast<uint16_t>(tiling.nzM - 1U)};
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t columns =
                elements - row * tiling.kL1 < tiling.kL1 ? elements - row * tiling.kL1 : tiling.kL1;
            params.blockCount = static_cast<uint16_t>(columns / 8U);
            DataCopy(destination[row * 8U], tile[row * tiling.kL1], params);
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    }

    /** Complete UB -> L1 NZ publication, then issue the scheduler's reusable-buffer credit.
     * This conservative direct backend completes synchronously; it never inherits a GM-only store.
     */
    __aicore__ inline static void StoreAivRowAsync(const GlobalTensor<float> &stage, uint64_t offset,
                                                   const LocalTensor<float> &tile, uint32_t elements,
                                                   const MegaMhcTilingData &tiling)
    {
        StoreAivRow(stage, offset, tile, elements, tiling);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
    }

    /** Wait on MTE1 so direct L1 loads observe both paired producers' MTE3 writes. */
    __aicore__ inline static void WaitReady(uint32_t subblock)
    {
        CrossCoreWaitFlag<Kernel::kPairSyncMode, PIPE_MTE1>(Kernel::kPairReadyFlag +
                                                            subblock * Kernel::kSubblockFlagOffset);
    }

    /** Activate an already populated A L1 slot; synchronization is owned by the transfer policy. */
    template <bool USE_HF32>
    __aicore__ inline static void LoadA1(Cube::ProjectionCube<USE_HF32> &cube, const GlobalTensor<float> &stage,
                                         uint64_t offset, uint32_t rows, uint32_t nzM, uint32_t kCount, uint32_t slot)
    {
        cube.AcceptDirectA(slot);
    }
};

template <ProjectionTransport TRANSPORT>
struct StagePolicy;
template <>
struct StagePolicy<ProjectionTransport::GmStage> {
    using Type = GmStage;
};
template <>
struct StagePolicy<ProjectionTransport::UbToL1> {
    using Type = UbL1Stage;
};
template <ProjectionTransport TRANSPORT>
using StagePolicyT = typename StagePolicy<TRANSPORT>::Type;

} // namespace MegaMhc::Transfer

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_TRANSFER_H
