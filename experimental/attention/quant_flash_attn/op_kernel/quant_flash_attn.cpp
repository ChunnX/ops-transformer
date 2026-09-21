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
 * \brief MxFP8 SOFTMAX FP16 kernel 入口（与主线高精 MxFP8 隔离，静态 Tensor 编程，无 TPipe）
 */

#include "kernel_operator.h"
#include "arch35/quant_flash_attn_template_tiling_key.h"
#include "arch35/quant_flash_attn_tiling_data.h"
#include "arch35/quant_flash_attn_kernel_mxfp8_softmax_fp16.h"

using namespace optiling;

// ─────────────────────────────────────────────────────────
// quant_flash_attn_mxfp8_softmax_fp16: MxFP8 Softmax FP16（S1/S2=256，V0/V1 切 S1 半）
// ─────────────────────────────────────────────────────────
template <uint8_t inOutLayoutType, uint16_t config, uint8_t quantMode, bool hasAttenMask, uint8_t KvLayoutType,
          bool isFd>
__aicore__ inline void quant_flash_attn_mxfp8_softmax_fp16(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dequantScaleQuery,
    __gm__ uint8_t *dequantScaleKey, __gm__ uint8_t *dequantScaleValue, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *pScale, __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv, __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sinks, __gm__ uint8_t *attnMask, __gm__ uint8_t *metadata,
    __gm__ uint8_t *attnOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    const __gm__ QuantFlashAttnTilingData *__restrict tilingData =
        (const __gm__ QuantFlashAttnTilingData *__restrict)tiling;
    using CubeBlock = QFA_KERNEL::QuantFlashAttnBlockCubeMxfp8SoftmaxFp16<QFA_KERNEL::QUANT_T, QFA_KERNEL::SCALE_T>;
    using VectorBlock = QFA_KERNEL::QuantFlashAttnBlockVectorMxfp8SoftmaxFp16<QFA_KERNEL::QUANT_T, QFA_KERNEL::SCALE_T, QFA_KERNEL::OUT_T>;
    QFA_KERNEL::QuantFlashAttnKernelMxfp8SoftmaxFp16<CubeBlock, VectorBlock> op;
    op.Init(query, key, value, dequantScaleQuery, dequantScaleKey, dequantScaleValue, blockTable, pScale,
            cuSeqLensQ, cuSeqLensKv, sequsedQ, sequsedKv, sinks, attnMask, metadata, attnOut, softmaxLse, workspace,
            tilingData);
    op.Process();
    PipeBarrier<PIPE_ALL>();
}

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

#if (ORIG_DTYPE_Q == DT_FLOAT8_E4M3FN)
    if constexpr (quantMode == QFA_MXFP8_SOFTMAX_FP16) {
        quant_flash_attn_mxfp8_softmax_fp16<inOutLayoutType, config, quantMode, hasAttenMask, KvLayoutType, isFd>(
            query, key, value, dequantScaleQuery, dequantScaleKey, dequantScaleValue, blockTable, pScale,
            cuSeqLensQ, cuSeqLensKv, sequsedQ, sequsedKv, sinks, attnMask, metadata, attnOut, softmaxLse, workspace,
            tiling);
    }
#endif
}
