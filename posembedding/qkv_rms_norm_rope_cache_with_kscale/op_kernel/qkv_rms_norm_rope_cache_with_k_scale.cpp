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
 * \file qkv_rms_norm_rope_cache_with_k_scale.cpp
 * \brief Empty AscendC kernel used to validate packaging before T01 adds the real controller.
 */
#include "kernel_operator.h"

#define QKV_RMS_NORM_ROPE_CACHE_WITH_KSCALE_D128_FAST 350128

extern "C" __global__ __aicore__ void qkv_rms_norm_rope_cache_with_k_scale(
    GM_ADDR qkv, GM_ADDR q_gamma, GM_ADDR k_gamma, GM_ADDR cos_sin, GM_ADDR slot_mapping, GM_ADDR k_cache,
    GM_ADDR v_cache, GM_ADDR k_scale_cache, GM_ADDR cu_seq_len, GM_ADDR rotation, GM_ADDR v_scale, GM_ADDR q_out,
    GM_ADDR q_scale, GM_ADDR k_cache_out, GM_ADDR v_cache_out, GM_ADDR k_scale_cache_out, GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)qkv;
    (void)q_gamma;
    (void)k_gamma;
    (void)cos_sin;
    (void)slot_mapping;
    (void)k_cache;
    (void)v_cache;
    (void)k_scale_cache;
    (void)cu_seq_len;
    (void)rotation;
    (void)v_scale;
    (void)q_out;
    (void)q_scale;
    (void)k_cache_out;
    (void)v_cache_out;
    (void)k_scale_cache_out;
    (void)workspace;
    (void)tiling;

    if (TILING_KEY_IS(QKV_RMS_NORM_ROPE_CACHE_WITH_KSCALE_D128_FAST)) {
        KERNEL_TASK_TYPE(QKV_RMS_NORM_ROPE_CACHE_WITH_KSCALE_D128_FAST, KERNEL_TYPE_MIX_AIC_1_2);
    }
}
