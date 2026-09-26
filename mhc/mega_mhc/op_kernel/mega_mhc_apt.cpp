/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310

#include "kernel_tiling/kernel_tiling.h"
#include "arch35/mega_mhc_kernel_context.h"
#include "arch35/epilogue.h"
#include "arch35/schedule_mk.h"

using namespace AscendC;
using namespace AiInfraMegaMhc;

/**
 * Total entry for the 950 four-op/five-op m-k families.
 * Four-op: complete RMS0/Post rows -> overlapped m-k projection when wave tiling is selected.
 * Five-op: complete RMS0/Post/Mid -> barrier -> m-k projection -> barrier.
 * Both variants reuse the same ordered Pre/RMS1 compute routines.
 */
__global__ __aicore__ void ai_infra_mega_mhc(GM_ADDR hOut, GM_ADDR residual, GM_ADDR hPost, GM_ADDR hRes, GM_ADDR phi,
                                             GM_ADDR alpha, GM_ADDR bias, GM_ADDR gamma1, GM_ADDR gamma0,
                                             GM_ADDR gamma2, GM_ADDR hInPrime, GM_ADDR hPostNew, GM_ADDR hResNew,
                                             GM_ADDR x2Out, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    using T = DTYPE_H_OUT;

    __gm__ uint8_t *userWorkspace = GetUserWorkspace(workspaceGM);
    Kernel::Tensors<T> tensors;
    tensors.hOut.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(hOut));
    tensors.residual.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(residual));
    tensors.hPost.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(hPost));
    tensors.hRes.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(hRes));
    tensors.phi.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(phi));
    tensors.alpha.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(alpha));
    tensors.bias.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bias));
    tensors.gamma1.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gamma1));
    tensors.gamma0.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gamma0));
    tensors.hInPrime.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(hInPrime));
    tensors.hPostNew.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(hPostNew));
    tensors.hResNew.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(hResNew));
    tensors.x2Out.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x2Out));
    tensors.invRms0.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace + tilingData.invRms0Offset));
    tensors.partialMm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace + tilingData.partialMmOffset));
    tensors.partialRms.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace + tilingData.partialRmsOffset));
    tensors.stage.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace + tilingData.stageOffset));
    if (tilingData.hasMid != 0U) {
        tensors.gamma2.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gamma2));
    }

    if (tilingData.hasMid == 0U) {
        TPipe fourOpPipe;
        Schedule::RunMkSchedule<T, false>(tensors, tilingData, &fourOpPipe);
    } else {
        TPipe fiveOpPipe;
        Schedule::RunMkSchedule<T>(tensors, tilingData, &fiveOpPipe);
    }

    TPipe epiloguePipe;
    Schedule::RunMkEpilogue<T>(tensors, tilingData, &epiloguePipe);
    epiloguePipe.Destroy();
}

#endif // defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
