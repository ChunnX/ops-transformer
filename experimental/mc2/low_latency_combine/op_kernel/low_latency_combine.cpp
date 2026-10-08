// Copyright (c) 2026, edconeone
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <cstdint>

#include "kernel_operator.h"
#include "shmem.h"

#include "mc2_kernel_launch.hpp"

#if CATLASS_ARCH == 3510
#include "low_latency/combine_server_dedup_kernel.h"
#include "low_latency/ep_memory_server_dedup_combine_tiling.h"
#include "low_latency/device/dispatch_server_dedup_magic.h"
#include "low_latency/launch.hpp"

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void combine_server_dedup_kernel(
    GM_ADDR commArgs, GM_ADDR expertOut, GM_ADDR destinationIndex, GM_ADDR sendCounts, GM_ADDR expertScales,
    GM_ADDR tokenType, GM_ADDR relayReadIndex, GM_ADDR sourceMask, GM_ADDR yOut, GM_ADDR tilingData)
{
    const auto* args = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs);
    util_set_ffts_config(args->fftsVal);
    uint32_t magic;
    {
        AscendC::TPipe pipe;
        magic = DispatchDedup::RotateDispatchServerDedupMagic(args, AscendC::GetBlockIdx(), &pipe);
    }
    const auto* td = reinterpret_cast<const volatile __gm__ CombineDedup::EpServerDedupCombineTilingData*>(tilingData);
    CombineDedup::CombineMemoryServerDedupKernel kernel;
    kernel.Init(commArgs, expertOut, destinationIndex, sendCounts, expertScales, tokenType, relayReadIndex, sourceMask,
                yOut, td->bs, td->h, td->topK, td->moeExpertNum, td->dtype, 0, magic);
    kernel.Process();
}
extern "C" DEEPEP_EXPORT void combine_server_dedup_kernel_do(void* stream,
                                                             const deepep::EpServerDedupCombineKernelArgs* args)
{
    const auto& a = *args;
    combine_server_dedup_kernel<<<64, DEEPEP_KERNEL_LAUNCH_UB(0), stream>>>(
        a.commArgs, a.expertOut, a.destinationIndex, a.sendCounts, a.expertScales, a.tokenType, a.relayReadIndex,
        a.sourceMask, a.yOut, a.tilingData);
}
#endif
