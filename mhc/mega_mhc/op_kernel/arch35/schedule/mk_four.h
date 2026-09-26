/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is distributed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_MK_FOUR_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_MK_FOUR_H

#include "mk_projection.h"
#include "rms0_statistics.h"

namespace AiInfraMegaMhc::Schedule {

/** Pipeline each Post-produced K tile into Cube, then establish one projection-complete boundary. */
template <typename T>
__aicore__ inline void RunMkFourSchedule(Kernel::Tensors<T> &tensors,
                                          const MegaMhcTilingData &tiling, TPipe *pipe)
{
    if (tiling.kGrid != 1U) {
        RunRms0Statistics<T>(tensors, tiling, pipe);
        pipe->Destroy();
        SyncAll<false>();
    } else {
        pipe->Destroy();
    }
    TPipe projectionPipe;
    RunMKScheduler<T, true, Transfer::StagePolicyT<kFourProjectionTransport>, kOverlapFourEpilogue>(
        tensors, tiling, &projectionPipe);
    projectionPipe.Destroy();
    SyncAll<false>();
}

}  // namespace AiInfraMegaMhc::Schedule

#endif  // OPS_TRANSFORMER_MHC_MEGA_MHC_MK_FOUR_H
