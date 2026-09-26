/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_UB_L1_STAGE_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_UB_L1_STAGE_H
#include "gm_stage.h"

namespace AiInfraMegaMhc::Transfer {
/** Pair-mode UB -> L1 transfer; each eight-float source block maps to NZ[k/8,m,8]. */
class UbL1Stage : public GmStage {
public:
    /** Store packed FP32 rows directly into the AIC L1 slot, without a GM staging round trip. */
    __aicore__ inline static void StoreAivRow(const GlobalTensor<float> &stage, uint64_t offset,
        const LocalTensor<float> &tile, uint32_t elements, const MegaMhcTilingData &tiling)
    {
        const uint64_t pairOffset = offset % (static_cast<uint64_t>(tiling.slotCount) * tiling.mTile * tiling.kL1);
        const uint32_t slot = pairOffset / (tiling.mTile * tiling.kL1);
        const uint32_t firstRow = (pairOffset % (tiling.mTile * tiling.kL1)) / tiling.kL1;
        LocalTensor<float> destination(TPosition::A1,
            (static_cast<uint64_t>(slot) * tiling.l1ASlotElements + firstRow * 8U) * sizeof(float),
            tiling.l1ASlotElements);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
        const uint32_t rows = (elements + tiling.kL1 - 1U) / tiling.kL1;
        DataCopyParams params{static_cast<uint16_t>(tiling.kL1 / 8U), 1, 0,
            static_cast<uint16_t>(tiling.nzM - 1U)};
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t columns = elements - row * tiling.kL1 < tiling.kL1 ?
                elements - row * tiling.kL1 : tiling.kL1;
            params.blockCount = static_cast<uint16_t>(columns / 8U);
            DataCopy(destination[row * 8U], tile[row * tiling.kL1], params);
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    }

    /** Wait on MTE1 so direct L1 loads observe both paired producers' MTE3 writes. */
    __aicore__ inline static void WaitReady(uint32_t subblock)
    {
        CrossCoreWaitFlag<Kernel::kPairSyncMode, PIPE_MTE1>(
            Kernel::kPairReadyFlag + subblock * Kernel::kSubblockFlagOffset);
    }

    /** Activate an already populated A L1 slot; synchronization is owned by the transfer policy. */
    __aicore__ inline static void LoadA1(Cube::ProjectionCube &cube, const GlobalTensor<float> &stage,
        uint64_t offset, uint32_t rows, uint32_t nzM, uint32_t kCount, uint32_t slot)
    {
        cube.AcceptDirectA(slot);
    }
};
} // namespace AiInfraMegaMhc::Transfer
#endif
