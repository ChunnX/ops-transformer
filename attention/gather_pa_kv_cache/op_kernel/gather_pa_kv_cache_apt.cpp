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
 * \file gather_pa_kv_cache_apt.cpp
 * \brief gather_pa_kv_cache
 */
#include "arch35/gather_pa_kv_cache_nd.h"
#include "arch35/gather_pa_kv_cache_nz.h"

#define TILING_KEY_1111 1111
#define TILING_KEY_1110 1110
#define TILING_KEY_1101 1101
#define TILING_KEY_1100 1100
#define TILING_KEY_1011 1011
#define TILING_KEY_1010 1010
#define TILING_KEY_1001 1001
#define TILING_KEY_1000 1000

extern "C" __global__ __aicore__ void gather_pa_kv_cache(GM_ADDR key_cache, GM_ADDR value_cache,
                                                         GM_ADDR block_tables, GM_ADDR seq_lens,
                                                         GM_ADDR key_in, GM_ADDR value_in, GM_ADDR seq_offset,
                                                         GM_ADDR key_out, GM_ADDR value_out,
                                                         GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(GatherPaKvCacheTilingDataV35, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // ND (SIMT): key_in/value_in 仅用于 Nz 格式, ND 忽略
    (void)key_in;
    (void)value_in;

    // ND 路径 (isCacheModeNorm = true) → SIMT 实现
    if (TILING_KEY_IS(TILING_KEY_1111)) {
        NsGatherPaKvCacheNd::Process<DTYPE_SEQ_LENS, true, true>(
            tilingData,
            reinterpret_cast<__gm__ uint8_t*>(key_cache),
            reinterpret_cast<__gm__ uint8_t*>(value_cache),
            reinterpret_cast<__gm__ const DTYPE_SEQ_LENS*>(block_tables),
            reinterpret_cast<__gm__ const DTYPE_SEQ_LENS*>(seq_lens),
            reinterpret_cast<__gm__ const DTYPE_SEQ_LENS*>(seq_offset),
            reinterpret_cast<__gm__ uint8_t*>(key_out),
            reinterpret_cast<__gm__ uint8_t*>(value_out));
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_1110)) {
        NsGatherPaKvCacheNd::Process<DTYPE_SEQ_LENS, true, false>(
            tilingData,
            reinterpret_cast<__gm__ uint8_t*>(key_cache),
            reinterpret_cast<__gm__ uint8_t*>(value_cache),
            reinterpret_cast<__gm__ const DTYPE_SEQ_LENS*>(block_tables),
            reinterpret_cast<__gm__ const DTYPE_SEQ_LENS*>(seq_lens),
            nullptr,
            reinterpret_cast<__gm__ uint8_t*>(key_out),
            reinterpret_cast<__gm__ uint8_t*>(value_out));
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_1101)) {
        NsGatherPaKvCacheNd::Process<DTYPE_SEQ_LENS, false, true>(
            tilingData,
            reinterpret_cast<__gm__ uint8_t*>(key_cache),
            reinterpret_cast<__gm__ uint8_t*>(value_cache),
            reinterpret_cast<__gm__ const DTYPE_SEQ_LENS*>(block_tables),
            reinterpret_cast<__gm__ const DTYPE_SEQ_LENS*>(seq_lens),
            reinterpret_cast<__gm__ const DTYPE_SEQ_LENS*>(seq_offset),
            reinterpret_cast<__gm__ uint8_t*>(key_out),
            reinterpret_cast<__gm__ uint8_t*>(value_out));
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_1100)) {
        NsGatherPaKvCacheNd::Process<DTYPE_SEQ_LENS, false, false>(
            tilingData,
            reinterpret_cast<__gm__ uint8_t*>(key_cache),
            reinterpret_cast<__gm__ uint8_t*>(value_cache),
            reinterpret_cast<__gm__ const DTYPE_SEQ_LENS*>(block_tables),
            reinterpret_cast<__gm__ const DTYPE_SEQ_LENS*>(seq_lens),
            nullptr,
            reinterpret_cast<__gm__ uint8_t*>(key_out),
            reinterpret_cast<__gm__ uint8_t*>(value_out));
        return;
    }

    // NZ 路径 (isCacheModeNorm = false) → 保留 SIMD 实现
    AscendC::TPipe pipe;

    if (TILING_KEY_IS(TILING_KEY_1011)) {
        GatherPaKvCacheV35::GatherPaKvCacheNz<uint8_t, DTYPE_SEQ_LENS, true, true> op(&pipe, &tilingData);
        op.Init(key_cache, value_cache, block_tables, seq_lens, key_in, value_in, seq_offset, key_out, value_out);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_1010)) {
        GatherPaKvCacheV35::GatherPaKvCacheNz<uint8_t, DTYPE_SEQ_LENS, true, false> op(&pipe, &tilingData);
        op.Init(key_cache, value_cache, block_tables, seq_lens, key_in, value_in, seq_offset, key_out, value_out);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_1001)) {
        GatherPaKvCacheV35::GatherPaKvCacheNz<uint8_t, DTYPE_SEQ_LENS, false, true> op(&pipe, &tilingData);
        op.Init(key_cache, value_cache, block_tables, seq_lens, key_in, value_in, seq_offset, key_out, value_out);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_1000)) {
        GatherPaKvCacheV35::GatherPaKvCacheNz<uint8_t, DTYPE_SEQ_LENS, false, false> op(&pipe, &tilingData);
        op.Init(key_cache, value_cache, block_tables, seq_lens, key_in, value_in, seq_offset, key_out, value_out);
        op.Process();
        return;
    }
}