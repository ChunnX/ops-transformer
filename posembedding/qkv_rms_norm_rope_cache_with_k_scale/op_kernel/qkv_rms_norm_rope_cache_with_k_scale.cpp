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
 * \brief Kernel entry for QkvRmsNormRopeCacheWithKScale.
 */
#include "qkv_rms_norm_rope_cache_with_k_scale_controller.h"

using namespace AscendC;
using QkvRmsNormRopeCacheWithKScale::D128FastController;
using QkvRmsNormRopeCacheWithKScale::GlobalTensors;
using QkvRmsNormRopeCacheWithKScaleKernelTiling::QkvRmsNormRopeCacheWithKScaleTilingData;

extern "C" __global__ __aicore__ void qkv_rms_norm_rope_cache_with_k_scale(
    GM_ADDR qkv, GM_ADDR q_gamma, GM_ADDR k_gamma, GM_ADDR cos_sin, GM_ADDR slot_mapping, GM_ADDR k_cache,
    GM_ADDR v_cache, GM_ADDR k_scale_cache, GM_ADDR cu_seq_len, GM_ADDR rotation, GM_ADDR v_scale, GM_ADDR q_out,
    GM_ADDR q_scale, GM_ADDR k_cache_out, GM_ADDR v_cache_out, GM_ADDR k_scale_cache_out, GM_ADDR workspace,
    GM_ADDR tiling)
{
    if (TILING_KEY_IS(QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_D128_FAST)) {
        KERNEL_TASK_TYPE(QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_D128_FAST, KERNEL_TYPE_MIX_AIC_1_2);

        REGISTER_TILING_DEFAULT(QkvRmsNormRopeCacheWithKScaleTilingData);
        GET_TILING_DATA_WITH_STRUCT(QkvRmsNormRopeCacheWithKScaleTilingData, tilingDataIn, tiling);
        const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData = &tilingDataIn;

        const GlobalTensors tensors = {
            qkv,       q_gamma, k_gamma,       cos_sin, slot_mapping, k_cache,     v_cache,
            k_scale_cache,     cu_seq_len,    rotation, v_scale,      q_out,       q_scale,
            k_cache_out,       v_cache_out,   k_scale_cache_out,      workspace,
        };

        D128FastController controller;
        controller.Process(tensors, tilingData);
    }
}
