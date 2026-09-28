/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_SCHEDULE_BS_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_SCHEDULE_BS_H

#include "schedule_common.h"
#include "epilogue.h"

namespace MegaMhc::Schedule {
using namespace AscendC;

/** BS owns distinct token tiles and full K; retain the staged barrier protocol.
 * Formula: x2=Post(RMS0(hOut),residual), optionally RMSMid(x2); y=RMS1(Pre(x2)).
 * Host owns all tile sizes. This entry owns stage lifetimes, synchronization and finalization.
 */
template <typename T, bool WITH_MID, bool USE_HF32>
__aicore__ inline void RunBsSchedule(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &tiling)
{
    using StageTransfer = Transfer::StagePolicyT<WITH_MID ? kFiveProjectionTransport : kFourProjectionTransport>;
    TPipe producerPipe;
    if constexpr (WITH_MID) {
        RunPrefetchedFiveProducer<T>(tensors, tiling, &producerPipe);
    } else {
        RunFourStagedProducer<T>(tensors, tiling, &producerPipe);
    }
    producerPipe.Destroy();
    SyncAll<false>();

    TPipe projectionPipe;
    RunProjectionSchedule<T, StageTransfer, USE_HF32>(tensors, tiling, &projectionPipe);
    projectionPipe.Destroy();
    SyncAll<false>();

    TPipe epiloguePipe;
    RunEpilogue<T>(tensors, tiling, &epiloguePipe);
    epiloguePipe.Destroy();
}

} // namespace MegaMhc::Schedule

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_SCHEDULE_BS_H
