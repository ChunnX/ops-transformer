/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_pre_apt.cpp
 * \brief
 */
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/mhc_pre_common.h"
#include "arch35/mhc_pre_split_bs.h"
#include "arch35/mhc_pre_m_k_split.h"
#include "arch35/mhc_pre_split_nd.h"
#include "arch35/mhc_pre_tiling_key.h"

using namespace AscendC;
using namespace MhcPre;

// CV_MODE is the AIV:AIC core ratio the host picked from the device core counts, expressed with
// the same macro values the tiling key carries: ASCENDC_TPL_MIX_AIC_1_2 pairs two AIVs with each
// AIC and splits the M tile between them, ASCENDC_TPL_MIX_AIC_1_1 gives each AIC a single AIV that
// owns the whole tile. It must stay the last template parameter so it lines up with the trailing
// KERNEL_TYPE_DECL field of the tiling key.
template <int8_t TILING_MODE, int8_t RESI_MODE, uint8_t CV_MODE>
__global__ __aicore__ void mhc_pre(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias, GM_ADDR gamma, GM_ADDR hin,
                                   GM_ADDR h_post, GM_ADDR h_res, GM_ADDR inv_rms, GM_ADDR h_mix, GM_ADDR h_pre,
                                   GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    GET_TILING_DATA(tilingData, tilingGM);
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

    // The task type is a compile-time attribute of the kernel binary, so the ratio has to be part
    // of the template arguments rather than a runtime decision: the host stamps CV_MODE into the
    // tiling key and the matching instantiation is the one that gets launched.
    KERNEL_TASK_TYPE_DEFAULT(CV_MODE == ASCENDC_TPL_MIX_AIC_1_1 ? KERNEL_TYPE_MIX_AIC_1_1 :
                                                                  KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    if constexpr (TILING_MODE == MHC_PRE_SPLIT_BS) {
        InitParams initParams{x, phi, alpha, bias, gamma, hin, h_post,
                              h_res, inv_rms, h_mix, h_pre, user, &pipe, &tilingData};
        MhcPreSplitBS<DTYPE_X, float32_t, RESI_MODE> op;
        op.Init(initParams);
        op.Process();
    } else if constexpr (TILING_MODE == MHC_PRE_SPLIT_M_K) {
        InitParams initParams{x, phi, alpha, bias, gamma, hin, h_post,
                              h_res, inv_rms, h_mix, h_pre, user, &pipe, &tilingData};
        MhcPreMKPart1<DTYPE_X, float32_t, RESI_MODE> part1;
        part1.Init(initParams);
        part1.Process();
        // Stage 1 and stage 2 use different UB layouts. Releasing stage-1 queues lets stage 2
        // reuse the full UB budget for reduction and postprocessing.
        pipe.Destroy();
        TPipe stage2Pipe;
        initParams.tPipeIn = &stage2Pipe;
        MhcPreMKPart2<DTYPE_X, float32_t, RESI_MODE> part2;
        part2.Init(initParams);
        part2.Process();
    } else if constexpr (TILING_MODE == MHC_PRE_SPLIT_ND) {
        InitParams initParams{x, phi, alpha, bias, gamma, hin, h_post,
                              h_res, inv_rms, h_mix, h_pre, user, &pipe, &tilingData};
        MhcPreSplitND<DTYPE_X, float32_t, RESI_MODE> op;
        op.Init(initParams);
        op.Process();
    }
}
#endif
