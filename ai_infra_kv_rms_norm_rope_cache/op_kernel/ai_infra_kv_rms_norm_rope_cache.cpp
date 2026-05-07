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
 * \file ai_infra_kv_rms_norm_rope_cache.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ai_infra_kv_rms_norm_rope_cache_b16_mtp.h"
#include "ai_infra_kv_rms_norm_rope_cache_b16_mtp_quant.h"
#include "ai_infra_kv_rms_norm_rope_cache_b16_b1sd.h"
#include "ai_infra_kv_rms_norm_rope_cache_b16_pa_blk_nz.h"
#include "ai_infra_kv_rms_norm_rope_cache_b16_pa_bnsd_quant.h"
#include "ai_infra_kv_rms_norm_rope_cache_b16_pa_blk_bnsd_quant.h"
#include "ai_infra_kv_rms_norm_rope_cache_b16_pa_nz_quant.h"
#include "ai_infra_kv_rms_norm_rope_cache_b16_pa_blk_nz_quant.h"
#include "ai_infra_kv_rms_norm_rope_cache_b16_pa.h"
#include "ai_infra_kv_rms_norm_rope_cache_b16_pa_bnsd_c8.h"

#include "arch35/ai_infra_kv_rms_norm_rope_cache_regbase_full_load.h"
#include "arch35/ai_infra_kv_rms_norm_rope_cache_regbase_recompute.h"
using namespace AscendC;
using namespace AiInfraKvRmsNormRopeCache;
#define AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_NORM 1000
#define AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_NORM_MTP 1001
#define AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_NORM_QUANT 1010
#define AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_NORM_MTP_QUANT 1011
#define AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA 2000
#define AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_MTP 2001
#define AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_B1SD_NORM 3000
#define AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_B1SD_PA 3001
#define AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_B1SD_NORM_QUANT 3010
#define AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_NZ 4000
#define AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_NZ 4001
#define AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_BNSD 5000
#define AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_BNSD 5001
#define AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_NZ_QUANT 4010
#define AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_NZ_QUANT 4011
#define AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_BNSD_QUANT 5010
#define AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_BNSD_QUANT 5011
#define AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_BNSD_C8 5021

extern "C" __global__ __aicore__ void ai_infra_kv_rms_norm_rope_cache(
    GM_ADDR kv, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR k_cache, GM_ADDR ckv_cache,
    GM_ADDR k_rope_scale, GM_ADDR c_kv_scale, GM_ADDR k_rope_offset, GM_ADDR c_kv_offset, GM_ADDR v, GM_ADDR k_cache_out,
    GM_ADDR c_kv_cache_out, GM_ADDR k_rope, GM_ADDR c_kv, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
#if (__CCE_AICORE__ == 310)
    #define FLOAT_OVERFLOW_MODE_CTRL 60
    #if (__NPU_ARCH__ == 3510)
        int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    #endif
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    #if (__NPU_ARCH__ == 3510)
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    #endif
    if (TILING_KEY_IS(10000)) {
        GET_TILING_DATA_WITH_STRUCT(AiInfraKvRmsNormRopeCacheRegbaseFullLoadTilingData, tiling_data_in, tiling);
        const AiInfraKvRmsNormRopeCacheRegbaseFullLoadTilingData* __restrict tilingData = &tiling_data_in;
        AiInfraKvRmsNormRopeCacheRegbaseFullLoad<DTYPE_KV, DTYPE_K_CACHE, DTYPE_CKV_CACHE> op(&pipe, tilingData);
        op.Init(
            kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset,
            k_rope, c_kv);
        op.Process();
    } else if(TILING_KEY_IS(20000)) {
        GET_TILING_DATA_WITH_STRUCT(AiInfraKvRmsNormRopeCacheRegbaseRecomputeTilingData, tiling_data_in, tiling);
        const AiInfraKvRmsNormRopeCacheRegbaseRecomputeTilingData* __restrict tilingData = &tiling_data_in;
        AiInfraKvRmsNormRopeCacheRegbaseRecompute<DTYPE_KV, DTYPE_K_CACHE, DTYPE_CKV_CACHE> op(&pipe, tilingData);
        op.Init(
            kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset,
            k_rope, c_kv);
        op.Process();
    }
    #if (__NPU_ARCH__ == 3510)
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
    #endif
#else
    if (TILING_KEY_IS(AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_NORM)) {
        GET_TILING_DATA_WITH_STRUCT(AiInfraKvRmsNormRopeCacheTilingData, tiling_data_in, tiling);
        const AiInfraKvRmsNormRopeCacheTilingData* __restrict tilingData = &tiling_data_in;
        KernelAiInfraKvRmsNormRopeCacheB16MTP<false, DTYPE_KV> op(&pipe, tilingData);
        op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, v);
        op.Process();
    } else if (TILING_KEY_IS(AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_NORM_MTP)) {
        GET_TILING_DATA_WITH_STRUCT(AiInfraKvRmsNormRopeCacheTilingData, tiling_data_in, tiling);
        const AiInfraKvRmsNormRopeCacheTilingData* __restrict tilingData = &tiling_data_in;
        KernelAiInfraKvRmsNormRopeCacheB16MTP<false, DTYPE_KV> op(&pipe, tilingData);
        op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, v);
        op.Process();
    } else if (TILING_KEY_IS(AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_NORM_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(AiInfraKvRmsNormRopeCacheTilingData, tiling_data_in, tiling);
        const AiInfraKvRmsNormRopeCacheTilingData* __restrict tilingData = &tiling_data_in;
        KernelAiInfraKvRmsNormRopeCacheB16MTPQUANT<false, DTYPE_KV, DTYPE_K_CACHE, DTYPE_CKV_CACHE> op(&pipe, tilingData);
        op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope, c_kv, k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, v);
        op.Process();
    } else if (TILING_KEY_IS(AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_NORM_MTP_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(AiInfraKvRmsNormRopeCacheTilingData, tiling_data_in, tiling);
        const AiInfraKvRmsNormRopeCacheTilingData* __restrict tilingData = &tiling_data_in;
        KernelAiInfraKvRmsNormRopeCacheB16MTPQUANT<false, DTYPE_KV, DTYPE_K_CACHE, DTYPE_CKV_CACHE> op(&pipe, tilingData);
        op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope, c_kv, k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, v);
        op.Process();
    } else if (TILING_KEY_IS(AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA)) {
        GET_TILING_DATA_WITH_STRUCT(AiInfraKvRmsNormRopeCacheTilingData, tiling_data_in, tiling);
        const AiInfraKvRmsNormRopeCacheTilingData* __restrict tilingData = &tiling_data_in;
        KernelAiInfraKvRmsNormRopeCacheB16MTP<true, DTYPE_KV> op(&pipe, tilingData);
        op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, v);
        op.Process();
    } else if (TILING_KEY_IS(AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_MTP)) {
        GET_TILING_DATA_WITH_STRUCT(AiInfraKvRmsNormRopeCacheTilingData, tiling_data_in, tiling);
        const AiInfraKvRmsNormRopeCacheTilingData* __restrict tilingData = &tiling_data_in;
        KernelAiInfraKvRmsNormRopeCacheB16MTP<true, DTYPE_KV> op(&pipe, tilingData);
        op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, v);
        op.Process();
    } else if (TILING_KEY_IS(AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_B1SD_NORM)) {
        GET_TILING_DATA_WITH_STRUCT(AiInfraKvRmsNormRopeCacheTilingData, tiling_data_in, tiling);
        const AiInfraKvRmsNormRopeCacheTilingData* __restrict tilingData = &tiling_data_in;
        KernelAiInfraKvRmsNormRopeCacheB16B1SD<false, DTYPE_KV> op(&pipe, tilingData);
        op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, v);
        op.Process();
    } else if (TILING_KEY_IS(AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_B1SD_PA)) {
        GET_TILING_DATA_WITH_STRUCT(AiInfraKvRmsNormRopeCacheTilingData, tiling_data_in, tiling);
        const AiInfraKvRmsNormRopeCacheTilingData* __restrict tilingData = &tiling_data_in;
        KernelAiInfraKvRmsNormRopeCacheB16B1SD<true, DTYPE_KV> op(&pipe, tilingData);
        op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, v);
        op.Process();
    } else if (TILING_KEY_IS(AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_NZ)) {
        GET_TILING_DATA_WITH_STRUCT(AiInfraKvRmsNormRopeCacheTilingData, tiling_data_in, tiling);
        const AiInfraKvRmsNormRopeCacheTilingData* __restrict tilingData = &tiling_data_in;
        KernelAiInfraKvRmsNormRopeCacheB16PABLKNZ<true, DTYPE_KV> op(&pipe, tilingData);
        op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, v, k_rope, c_kv);
        op.Process();
    } else if (TILING_KEY_IS(AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_NZ_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(AiInfraKvRmsNormRopeCacheTilingData, tiling_data_in, tiling);
        const AiInfraKvRmsNormRopeCacheTilingData* __restrict tilingData = &tiling_data_in;
        KernelAiInfraKvRmsNormRopeCacheB16PANZQUANT<true, DTYPE_KV, DTYPE_K_CACHE, DTYPE_CKV_CACHE> op(&pipe, tilingData);
        op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, v, k_rope, c_kv);
        op.Process();
    } else if (TILING_KEY_IS(AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_BNSD_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(AiInfraKvRmsNormRopeCacheTilingData, tiling_data_in, tiling);
        const AiInfraKvRmsNormRopeCacheTilingData* __restrict tilingData = &tiling_data_in;
        KernelAiInfraKvRmsNormRopeCacheB16PABLKBNSDQUANT<true, DTYPE_KV, DTYPE_K_CACHE, DTYPE_CKV_CACHE> op(&pipe, tilingData);
        op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope, c_kv, k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, v);
        op.Process();
    } else if (TILING_KEY_IS(AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_BNSD_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(AiInfraKvRmsNormRopeCacheTilingData, tiling_data_in, tiling);
        const AiInfraKvRmsNormRopeCacheTilingData* __restrict tilingData = &tiling_data_in;
        KernelAiInfraKvRmsNormRopeCacheB16BNSDQUANT<true, DTYPE_KV, DTYPE_K_CACHE, DTYPE_CKV_CACHE> op(&pipe, tilingData);
        op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope, c_kv, k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, v);
        op.Process();
    } else if (TILING_KEY_IS(AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_B1SD_NORM_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(AiInfraKvRmsNormRopeCacheTilingData, tiling_data_in, tiling);
        const AiInfraKvRmsNormRopeCacheTilingData* __restrict tilingData = &tiling_data_in;
        KernelAiInfraKvRmsNormRopeCacheB16BNSDQUANT<false, DTYPE_KV, DTYPE_K_CACHE, DTYPE_CKV_CACHE> op(&pipe, tilingData);
        op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope, c_kv, k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, v);
        op.Process();
    } else if (TILING_KEY_IS(AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_NZ_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(AiInfraKvRmsNormRopeCacheTilingData, tiling_data_in, tiling);
        const AiInfraKvRmsNormRopeCacheTilingData* __restrict tilingData = &tiling_data_in;
        KernelAiInfraKvRmsNormRopeCacheQuantB16PABLKNZ<true, DTYPE_KV, DTYPE_K_CACHE, DTYPE_CKV_CACHE> op(&pipe, tilingData);
        op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope, c_kv, k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, v);
        op.Process();
    } else if (TILING_KEY_IS(AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_NZ)) {
        GET_TILING_DATA_WITH_STRUCT(AiInfraKvRmsNormRopeCacheTilingData, tiling_data_in, tiling);
        const AiInfraKvRmsNormRopeCacheTilingData* __restrict tilingData = &tiling_data_in;
        KernelAiInfraKvRmsNormRopeCacheB16PA<true, DTYPE_KV, PA_NZ_NO_QUANT> op(&pipe, tilingData);
        op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope, c_kv, v);
        op.Process();
    } else if (TILING_KEY_IS(AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_BNSD)) {
        GET_TILING_DATA_WITH_STRUCT(AiInfraKvRmsNormRopeCacheTilingData, tiling_data_in, tiling);
        const AiInfraKvRmsNormRopeCacheTilingData* __restrict tilingData = &tiling_data_in;
        KernelAiInfraKvRmsNormRopeCacheB16PA<true, DTYPE_KV, PA_BLK_BNSD_NO_QUANT> op(&pipe, tilingData);
        op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope, c_kv, v);
        op.Process();
    } else if (TILING_KEY_IS(AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_BNSD)) {
        GET_TILING_DATA_WITH_STRUCT(AiInfraKvRmsNormRopeCacheTilingData, tiling_data_in, tiling);
        const AiInfraKvRmsNormRopeCacheTilingData* __restrict tilingData = &tiling_data_in;
        KernelAiInfraKvRmsNormRopeCacheB16PA<true, DTYPE_KV, PA_BNSD_NO_QUANT> op(&pipe, tilingData);
        op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope, c_kv, v);
        op.Process();
    } else if (TILING_KEY_IS(AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_BNSD_C8)) {
        GET_TILING_DATA_WITH_STRUCT(AiInfraKvRmsNormRopeCacheTilingData, tiling_data_in, tiling);
        const AiInfraKvRmsNormRopeCacheTilingData* __restrict tilingData = &tiling_data_in;
        KernelAiInfraKvRmsNormRopeCacheB16PAC8<true, DTYPE_KV, PA_BNSD_NO_QUANT> op(&pipe, tilingData);
        op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope, c_kv, v);
        op.Process();
    }
#endif
}