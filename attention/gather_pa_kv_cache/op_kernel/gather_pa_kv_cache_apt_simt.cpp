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
 * \file gather_pa_kv_cache_apt_simt.cpp
 * \brief SIMT kernel entry for GatherPaKvCache ND format.
 *  Template on schMode for tiling key dispatch.
 *  key_in / value_in are inplace alias of key_out / value_out.
 */

#include "arch35/gather_pa_kv_cache_simt.h"
#include "arch35/gather_pa_kv_cache_tiling_key.h"

template <uint32_t schMode>
__global__ __aicore__ void gather_pa_kv_cache(GM_ADDR key_cache, GM_ADDR value_cache,
                                              GM_ADDR block_tables, GM_ADDR seq_lens,
                                              GM_ADDR key_in, GM_ADDR value_in,
                                              GM_ADDR seq_offset,
                                              GM_ADDR key_out, GM_ADDR value_out,
                                              GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GatherPaKvCacheSimtTilingData);
    GET_TILING_DATA_WITH_STRUCT(GatherPaKvCacheSimtTilingData, tilingData, tiling);

    NsGatherPaKvCache::Process<DTYPE_SEQ_LENS, schMode>(
        key_cache, value_cache,
        block_tables, seq_lens,
        key_out, value_out,
        seq_offset,
        &tilingData);
}

template __global__ __aicore__ void gather_pa_kv_cache<GATHER_PA_KV_CACHE_SCENE_1100>(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
template __global__ __aicore__ void gather_pa_kv_cache<GATHER_PA_KV_CACHE_SCENE_1110>(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
template __global__ __aicore__ void gather_pa_kv_cache<GATHER_PA_KV_CACHE_SCENE_1101>(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
template __global__ __aicore__ void gather_pa_kv_cache<GATHER_PA_KV_CACHE_SCENE_1111>(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
