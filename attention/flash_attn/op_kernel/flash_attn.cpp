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
 * \file flash_attn.cpp
 * \brief FlashAttn Kernel主入口（4字段）
 */

#include "kernel_operator.h"
#include "arch35/flash_attn_entry_regbase.h"
#include "arch35/flash_attn_template_tiling_key.h"
#include "arch35/flash_attn_tiling_data.h"

using namespace AscendC;

// kernel主入口：4字段 → 模板参数
template <uint8_t inOutLayoutType, uint8_t KvLayoutType, bool hasAttenMask, uint8_t config>
__global__ __aicore__ void
flash_attn(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockTable,
           __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv,
           __gm__ uint8_t *sinks, __gm__ uint8_t *attnMask, __gm__ uint8_t *metadata, __gm__ uint8_t *attnOut,
           __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(optiling::FlashAttnTilingData);
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if constexpr (inOutLayoutType == InOutLayoutType_TND && KvLayoutType == KvLayoutType_NO_PA) {
        dc_preload(reinterpret_cast<__gm__ uint64_t*>(cuSeqLensQ), 0);
        dc_preload(reinterpret_cast<__gm__ uint64_t*>(cuSeqLensKv), 0);
        dc_preload(reinterpret_cast<__gm__ uint64_t*>(sequsedQ), 0);
        dc_preload(reinterpret_cast<__gm__ uint64_t*>(sequsedKv), 0);
    }
    // dtype 分发 → flash_attn_kernel_run (类型推导 + kernel实例化执行)
#if (ORIG_DTYPE_Q == DT_BF16)
    flash_attn_kernel_run<bfloat16_t, bfloat16_t, inOutLayoutType, KvLayoutType, hasAttenMask, config>(
        query, key, value, attnMask, cuSeqLensQ, cuSeqLensKv, sequsedQ, sequsedKv, blockTable, sinks, attnOut,
        softmaxLse, user, tiling, metadata);
#elif (ORIG_DTYPE_Q == DT_FLOAT16)
    flash_attn_kernel_run<half, half, inOutLayoutType, KvLayoutType, hasAttenMask, config>(
        query, key, value, attnMask, cuSeqLensQ, cuSeqLensKv, sequsedQ, sequsedKv, blockTable, sinks, attnOut,
        softmaxLse, user, tiling, metadata);
#endif
}
