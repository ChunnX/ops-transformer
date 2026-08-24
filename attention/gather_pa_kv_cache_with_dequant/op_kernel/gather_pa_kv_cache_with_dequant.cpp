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
 * \file gather_pa_kv_cache_with_dequant.cpp
 * \brief Kernel entry point for fused GatherPaKvCache + Dequant
 */

#include "arch35/gather_pa_kv_cache_with_dequant_nd.h"
#include "kernel_operator.h"

using namespace AscendC;

#define TILING_KEY_1111 1111
#define TILING_KEY_1110 1110
#define TILING_KEY_1101 1101
#define TILING_KEY_1100 1100

extern "C" __global__ __aicore__ void gather_pa_kv_cache_with_dequant(
    GM_ADDR key_cache, GM_ADDR key_scale_cache, GM_ADDR value_cache, GM_ADDR value_scale,
    GM_ADDR block_tables, GM_ADDR seq_lens, GM_ADDR key_in, GM_ADDR value_in,
    GM_ADDR seq_offset, GM_ADDR key_out, GM_ADDR value_out, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(GatherPaKvCacheWithDequantTilingDataV35, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (TILING_KEY_IS(TILING_KEY_1111)) {
        GatherPaKvCacheWithDequantV35::GatherPaKvCacheWithDequantNd<DTYPE_SEQ_LENS, true, true> op(&pipe, &tilingData);
        op.Init(key_cache, key_scale_cache, value_cache, value_scale, block_tables, seq_lens,
                key_in, value_in, seq_offset, key_out, value_out);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_1110)) {
        GatherPaKvCacheWithDequantV35::GatherPaKvCacheWithDequantNd<DTYPE_SEQ_LENS, true, false> op(&pipe, &tilingData);
        op.Init(key_cache, key_scale_cache, value_cache, value_scale, block_tables, seq_lens,
                key_in, value_in, seq_offset, key_out, value_out);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_1101)) {
        GatherPaKvCacheWithDequantV35::GatherPaKvCacheWithDequantNd<DTYPE_SEQ_LENS, false, true> op(&pipe, &tilingData);
        op.Init(key_cache, key_scale_cache, value_cache, value_scale, block_tables, seq_lens,
                key_in, value_in, seq_offset, key_out, value_out);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_1100)) {
        GatherPaKvCacheWithDequantV35::GatherPaKvCacheWithDequantNd<DTYPE_SEQ_LENS, false, false> op(&pipe, &tilingData);
        op.Init(key_cache, key_scale_cache, value_cache, value_scale, block_tables, seq_lens,
                key_in, value_in, seq_offset, key_out, value_out);
        op.Process();
    }
}
