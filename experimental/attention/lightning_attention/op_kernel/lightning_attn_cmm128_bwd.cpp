/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lightning_attn_cmm128_bwd_kernel.h"
#include "common/lo_cmm128_tiling_data.h"

#include "kernel_operator.h"

using namespace LightningAttn::Common;

__aicore__ inline void CopyTiling(LoCmm128TilingData *dstTiling, const LoCmm128TilingData *srcTiling)
{
    uint32_t *dstPtr = reinterpret_cast<uint32_t *>(dstTiling);
    const uint32_t *srcPtr = reinterpret_cast<const uint32_t *>(srcTiling);

    uint32_t i = 0;
    for (; i < sizeof(LoCmm128TilingData) / sizeof(uint32_t); i++, dstPtr++) {
        *dstPtr = *(srcPtr + i);
    }
}


/**
 * @brief  la_cmm128_backward kernel function entry.
 * @param  query: Global memory address for the Query (Q) matrix.
 * @param  key: Global memory address for the Key (K) matrix.
 * @param  value: Global memory address for the Value (V) matrix.
 * @param  slopeRate: Global memory address for the slope matrix.
 * @param  kvState: Global memory address for the KV state saved during the forward pass.
 * @param  gradQueryOut:  Global memory address for the calculated gradient of Query.
 * @param  gradKeyOut:  Global memory address for the calculated gradient of Key.
 * @param  gradValueOut:  Global memory address for the calculated gradient of Value.
 * @param  workspace: Global memory address for the temporary scratchpad space.
 * @param  tiling:  Global memory address for a tiling configuration data.
 * @retval None
 */

__global__ __aicore__ void la_cmm128_backward(GM_ADDR gradResult,
                                                         GM_ADDR query,
                                                         GM_ADDR key,
                                                         GM_ADDR value,
                                                         GM_ADDR slopeRate,
                                                         GM_ADDR kvState,
                                                         GM_ADDR gradQueryOut,
                                                         GM_ADDR gradKeyOut,
                                                         GM_ADDR gradValueOut,
                                                         GM_ADDR workspace,
                                                         LoCmm128TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    AscendC::TPipe pipe;

    LightningAttn::Kernel::LaBackwardCMM128<bfloat16_t, float> op;
    op.Init(gradResult, query, key, value, slopeRate, kvState, gradQueryOut, gradKeyOut, gradValueOut, workspace, &tiling, &pipe);
    op.Process();
}
