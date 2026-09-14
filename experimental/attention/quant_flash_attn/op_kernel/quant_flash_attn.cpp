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
 * \file quant_flash_attn.cpp
 * \brief MxFP8 SOFTMAX FP16 最小骨架入口（与主线高精 MxFP8 隔离）
 */

#include "kernel_operator.h"
#include "arch35/quant_flash_attn_template_tiling_key.h"
#include "arch35/quant_flash_attn_tiling_data.h"

using namespace optiling;

template <uint8_t inOutLayoutType, uint16_t config, uint8_t quantMode, bool hasAttenMask, uint8_t KvLayoutType,
          bool isFd>
__global__ __aicore__ void quant_flash_attn(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dequantScaleQuery,
    __gm__ uint8_t *dequantScaleKey, __gm__ uint8_t *dequantScaleValue, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *pScale, __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv, __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sinks, __gm__ uint8_t *attnMask, __gm__ uint8_t *metadata,
    __gm__ uint8_t *attnOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(QuantFlashAttnTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    AscendC::printf("[QuantFlashAttn] SOFTMAX_FP16 kernel entered\n");

    // TODO(SOFTMAX_FP16 新方案): 在此实现 AIC/AIV 协同的 mm1 / softmax FP16 / mm2 流程
    (void)query;
    (void)key;
    (void)value;
    (void)dequantScaleQuery;
    (void)dequantScaleKey;
    (void)dequantScaleValue;
    (void)blockTable;
    (void)pScale;
    (void)cuSeqLensQ;
    (void)cuSeqLensKv;
    (void)sequsedQ;
    (void)sequsedKv;
    (void)sinks;
    (void)attnMask;
    (void)metadata;
    (void)attnOut;
    (void)softmaxLse;
    (void)workspace;
    (void)tiling;
}