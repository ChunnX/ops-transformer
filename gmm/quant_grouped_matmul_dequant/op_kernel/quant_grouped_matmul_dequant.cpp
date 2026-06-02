/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_grouped_matmul_dequant.cpp
 * \brief Template-dispatch entry for the unified qgmmd kernel.
 *
 * Reads the runtime flags from TilingData and instantiates the matching
 * QuantGroupedMatmulDequant<...> specialization. All tiling keys route here;
 * the tile-pick differences live in TilingData, not the class topology.
 */

#include "quant_grouped_matmul_dequant.h"
// Staged xZN split-distribution path: own stack frame (entry is at the
// 16000-byte aicore ceiling), dispatched via RunStagedB0.
#include "quant_grouped_matmul_dequant_staged_run.h"

using namespace AscendC;

// HasPertoken locked true; static-quant (pertensor) path dropped.
#define QGMMD_DISPATCH(hasSmooth, scaleIsU64, blockedZN, smallM, useXZN)                  \
    do {                                                                                   \
        QuantGroupedMatmulDequant<true, hasSmooth, scaleIsU64, blockedZN, smallM, useXZN>  \
            op;                                                                            \
        op.Init(x, quantized_weight, weight_scale, group_list, bias, x_scale,              \
                x_offset, smooth_scale, y, usr_workspace, tilingData, &tPipe);             \
        op.Process();                                                                      \
    } while (0)

// Runtime flag extraction -> pruned template dispatch.
#define QGMMD_DISPATCH_XZN(hasSmooth, scaleIsU64, blockedZN, smallM)                       \
    do {                                                                                   \
        if (useXZN) {                                                                      \
            QGMMD_DISPATCH(hasSmooth, scaleIsU64, blockedZN, smallM, true);                \
        } else {                                                                           \
            QGMMD_DISPATCH(hasSmooth, scaleIsU64, blockedZN, smallM, false);               \
        }                                                                                  \
    } while (0)

#define QGMMD_DISPATCH_SMOOTH(scaleIsU64, blockedZN, smallM)                               \
    do {                                                                                   \
        if (hasSmooth) {                                                                   \
            QGMMD_DISPATCH_XZN(true,  scaleIsU64, blockedZN, smallM);                      \
        } else {                                                                           \
            QGMMD_DISPATCH_XZN(false, scaleIsU64, blockedZN, smallM);                      \
        }                                                                                  \
    } while (0)

#define QGMMD_DISPATCH_SCALE(blockedZN, smallM)                                            \
    do {                                                                                   \
        if (scaleIsU64) {                                                                  \
            QGMMD_DISPATCH_SMOOTH(true,  blockedZN, smallM);                               \
        } else {                                                                           \
            QGMMD_DISPATCH_SMOOTH(false, blockedZN, smallM);                               \
        }                                                                                  \
    } while (0)

#define QGMMD_DISPATCH_BLOCKED(smallM)                                                     \
    do {                                                                                   \
        if (blockedZN) {                                                                   \
            QGMMD_DISPATCH_SCALE(true,  smallM);                                           \
        } else {                                                                           \
            QGMMD_DISPATCH_SCALE(false, smallM);                                           \
        }                                                                                  \
    } while (0)

extern "C" __global__ __aicore__ void quant_grouped_matmul_dequant(
    GM_ADDR x, GM_ADDR quantized_weight, GM_ADDR weight_scale, GM_ADDR group_list,
    GM_ADDR bias, GM_ADDR x_scale, GM_ADDR x_offset, GM_ADDR smooth_scale,
    GM_ADDR y, GM_ADDR usr_workspace, GM_ADDR qmmTiling)
{
    SetAtomicNone();
    GET_TILING_DATA(tiling_data, qmmTiling);
    const QuantMatmulDequantTilingData *__restrict tilingData =
        const_cast<QuantMatmulDequantTilingData *>(&tiling_data);

    // Staged path: own stack frame, own TPipe; dispatch before the entry's
    // tPipe is constructed so only one pipe is live.
    if (GetFlag(tilingData->flags, FLAG_BIT_USE_XZN_WORKSPACE)) {
        RunStagedB0(x, quantized_weight, weight_scale, group_list, bias, x_scale,
                    x_offset, smooth_scale, y, usr_workspace, tilingData);
        return;
    }

    TPipe tPipe;

    const bool hasSmooth  = GetFlag(tilingData->flags, FLAG_BIT_HAS_SMOOTH);
    const bool scaleIsU64 = GetFlag(tilingData->flags, FLAG_BIT_SCALE_IS_INT64);
    const bool blockedZN  = GetFlag(tilingData->flags, FLAG_BIT_BLOCKED_ZN);
    const bool smallM     = GetFlag(tilingData->flags, FLAG_BIT_SMALL_M);
    const bool useXZN     = GetFlag(tilingData->flags, FLAG_BIT_USE_XZN_WORKSPACE);

    if (smallM) {
        QGMMD_DISPATCH_BLOCKED(true);
    } else {
        QGMMD_DISPATCH_BLOCKED(false);
    }

    SetMaskNorm();
    ResetMask();
}
