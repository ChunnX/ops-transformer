/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is distributed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_GM_STAGE_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_GM_STAGE_H

#include "../mega_mhc_kernel_context.h"
#include "../projection_cube.h"

namespace AiInfraMegaMhc::Transfer {
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
        const LocalTensor<float> &tile, uint32_t elements, const MegaMhcTilingData &tiling)
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
    __aicore__ inline static void LoadA1(Cube::ProjectionCube &cube,
                                         const GlobalTensor<float> &stage, uint64_t stageOffset,
                                         uint32_t mRows, uint32_t nzM, uint32_t kCount, uint32_t slot)
    {
        cube.LoadA(stage, stageOffset, mRows, nzM, kCount, slot);
    }
};

}  // namespace AiInfraMegaMhc::Transfer

#endif  // OPS_TRANSFORMER_MHC_MEGA_MHC_GM_STAGE_H
