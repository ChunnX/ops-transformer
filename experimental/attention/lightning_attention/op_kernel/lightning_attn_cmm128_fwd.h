/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "kernel_operator.h"

/**
 * @brief  la_cmm128_forward kernel function entry.
 * @param  query: Global memory address for the Query (Q) matrix.
 * @param  key: Global memory address for the Key (K) matrix.
 * @param  value: Global memory address for the Value (V) matrix.
 * @param  slopeRate: Global memory address for the slope matrix.
 * @param  kvStateOut: Output address for the inter-chunk KV stats (saved in a training mode for backward pass)
 * @param  resultOut: Global memory address for the forward pass output matrix.
 * @param  workspace: Global memory address for the temporary scratchpad space.
 * @param  tiling:  Global memory address for a tiling configuration data.
 * @retval None
 */
__global__ __aicore__ void la_cmm128_forward(
    GM_ADDR query,
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR slopeRate,
    GM_ADDR kvStateOut,
    GM_ADDR resultOut,
    GM_ADDR workspace,
    LightningAttn::Common::LoCmm128TilingData tiling
);
