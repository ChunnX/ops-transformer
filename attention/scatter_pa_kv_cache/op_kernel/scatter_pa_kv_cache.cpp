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
 * \file scatter_pa_kv_cache.cpp
 * \brief scatter_pa_kv_cache
 */

#include "scatter_pa_kv_cache_normal_nz_fully_load.h"
#include "scatter_pa_kv_cache_normal_nz_not_fully_load.h"

#include "scatter_pa_kv_cache_nd.h"
#include "scatter_pa_kv_cache_compress_rope.h"
#include "scatter_pa_kv_cache_compress_alibi.h"
#include "scatter_pa_kv_cache_nd_siso.h"
#include "scatter_pa_kv_cache_compress_omni.h"

using namespace AscendC;

#define NZ_NORMAL_INT32_FULLY_LOAD 1141
#define NZ_NORMAL_INT32_NOT_FULLY_LOAD 1140

#define NZ_NORMAL_INT64_FULLY_LOAD 1181
#define NZ_NORMAL_INT64_NOT_FULLY_LOAD 1180

extern "C" __global__ __aicore__ void scatter_pa_kv_cache(GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping,
                                                          GM_ADDR value, GM_ADDR value_cache_in, GM_ADDR compress_lens,
                                                          GM_ADDR compress_seq_offset, GM_ADDR seq_lens,
                                                          GM_ADDR key_cache_out, GM_ADDR value_cache_out,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
    // comment: KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)
    AscendC::TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);

    if TILING_KEY_IS (NZ_NORMAL_INT32_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNormalNzFullyLoad<DTYPE_KEY, DTYPE_VALUE, int32_t> op(&pipe, &tilingData);
        op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset, seq_lens,
                key_cache_out, value_cache_out);
        op.Process();
    } else if TILING_KEY_IS (NZ_NORMAL_INT32_NOT_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNormalNzNotFullyLoad<DTYPE_KEY, DTYPE_VALUE, int32_t> op(&pipe, &tilingData);
        op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset, seq_lens,
                key_cache_out, value_cache_out);
        op.Process();
    } else if TILING_KEY_IS (NZ_NORMAL_INT64_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNormalNzFullyLoad<DTYPE_KEY, DTYPE_VALUE, int64_t> op(&pipe, &tilingData);
        op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset, seq_lens,
                key_cache_out, value_cache_out);
        op.Process();
    } else if TILING_KEY_IS (NZ_NORMAL_INT64_NOT_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNormalNzNotFullyLoad<DTYPE_KEY, DTYPE_VALUE, int64_t> op(&pipe, &tilingData);
        op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset, seq_lens,
                key_cache_out, value_cache_out);
        op.Process();
    }
#if ORIG_DTYPE_KEY == DT_INT8
    // nd
    else if (TILING_KEY_IS(100000000)) {
        ScatterPaKvCache::ScatterPaKvCacheNd<int8_t> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKVEqual(key, value, key_cache_in, value_cache_in, slot_mapping, key_cache_out, value_cache_out);
    } else if (TILING_KEY_IS(101000000)) {
        ScatterPaKvCache::ScatterPaKvCacheNd<int8_t> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKVUnequalIncrement(key, value, key_cache_in, value_cache_in, slot_mapping, key_cache_out,
                                     value_cache_out);
    } else if (TILING_KEY_IS(102000000)) {
        ScatterPaKvCache::ScatterPaKvCacheNd<int8_t> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKVUnequal(key, value, key_cache_in, value_cache_in, slot_mapping, key_cache_out, value_cache_out);
    }
    // siso
    else if (TILING_KEY_IS(140000000)) {
        ScatterPaKvCache::ScatterPaKvCacheNdSiso<int8_t> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKey(key, key_cache_in, slot_mapping, key_cache_out);
    } else if (TILING_KEY_IS(142000000)) {
        ScatterPaKvCache::ScatterPaKvCacheNdSiso<int8_t> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKeyIncrement(key, key_cache_in, slot_mapping, key_cache_out);
    }
    // alibi
    else if (TILING_KEY_IS(120000000)) {
        ScatterPaKvCache::ScatterPaKvCacheCompressAlibi<int8_t> op;
        op.Init(&pipe, &tilingData);
        op.Process(key, value, key_cache_in, value_cache_in, slot_mapping, compress_lens, seq_lens, key_cache_out,
                   value_cache_out);
    }
#elif (ORIG_DTYPE_KEY == DT_FLOAT16 || ORIG_DTYPE_KEY == DT_BF16)
    // nd
    else if (TILING_KEY_IS(200000000)) {
        ScatterPaKvCache::ScatterPaKvCacheNd<half> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKVEqual(key, value, key_cache_in, value_cache_in, slot_mapping, key_cache_out, value_cache_out);
    } else if (TILING_KEY_IS(201000000)) {
        ScatterPaKvCache::ScatterPaKvCacheNd<half> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKVUnequalIncrement(key, value, key_cache_in, value_cache_in, slot_mapping, key_cache_out,
                                     value_cache_out);
    } else if (TILING_KEY_IS(202000000)) {
        ScatterPaKvCache::ScatterPaKvCacheNd<half> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKVUnequal(key, value, key_cache_in, value_cache_in, slot_mapping, key_cache_out, value_cache_out);
    }
    // nd nct
    else if (TILING_KEY_IS(260000000)) {
        ScatterPaKvCache::ScatterPaKvCacheNd<half> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKVEqual<true>(key, value, key_cache_in, value_cache_in, slot_mapping, key_cache_out, value_cache_out);
    } else if (TILING_KEY_IS(261000000)) {
        ScatterPaKvCache::ScatterPaKvCacheNd<half> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKVUnequalIncrement<true>(key, value, key_cache_in, value_cache_in, slot_mapping, key_cache_out,
                                           value_cache_out);
    } else if (TILING_KEY_IS(262000000)) {
        ScatterPaKvCache::ScatterPaKvCacheNd<half> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKVUnequal<true>(key, value, key_cache_in, value_cache_in, slot_mapping, key_cache_out,
                                  value_cache_out);
    }
    // siso
    else if (TILING_KEY_IS(240000000)) {
        ScatterPaKvCache::ScatterPaKvCacheNdSiso<half> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKey(key, key_cache_in, slot_mapping, key_cache_out);
    } else if (TILING_KEY_IS(242000000)) {
        ScatterPaKvCache::ScatterPaKvCacheNdSiso<half> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKeyIncrement(key, key_cache_in, slot_mapping, key_cache_out);
    }
    // siso nct
    else if (TILING_KEY_IS(270000000)) {
        ScatterPaKvCache::ScatterPaKvCacheNdSiso<half> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKey<true>(key, key_cache_in, slot_mapping, key_cache_out);
    } else if (TILING_KEY_IS(272000000)) {
        ScatterPaKvCache::ScatterPaKvCacheNdSiso<half> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKeyIncrement<true>(key, key_cache_in, slot_mapping, key_cache_out);
    }
    // alibi
    else if (TILING_KEY_IS(220000000)) {
        ScatterPaKvCache::ScatterPaKvCacheCompressAlibi<half> op;
        op.Init(&pipe, &tilingData);
        op.Process(key, value, key_cache_in, value_cache_in, slot_mapping, compress_lens, seq_lens, key_cache_out,
                   value_cache_out);
    }
#if ORIG_DTYPE_KEY == DT_FLOAT16
    // rope
    else if (TILING_KEY_IS(230000000)) {
        ScatterPaKvCache::ScatterPaKvCacheCompressRope<half> op;
        op.Init(&pipe, &tilingData);
        op.Method(key, value, key_cache_in, value_cache_in, slot_mapping, compress_lens, seq_lens, compress_seq_offset,
                  key_cache_out, value_cache_out);
    }
    // omni
    else if (TILING_KEY_IS(250000000)) {
        ScatterPaKvCache::ScatterPaKvCacheCompressOmni<half> op;
        op.Init(&pipe, &tilingData);
        op.Method(key, value, key_cache_in, value_cache_in, slot_mapping, compress_lens, seq_lens, compress_seq_offset,
                  key_cache_out, value_cache_out);
    }
#elif ORIG_DTYPE_KEY == DT_BF16
    // rope
    else if (TILING_KEY_IS(330000000)) {
        ScatterPaKvCache::ScatterPaKvCacheCompressRope<bfloat16_t> op;
        op.Init(&pipe, &tilingData);
        op.Method(key, value, key_cache_in, value_cache_in, slot_mapping, compress_lens, seq_lens, compress_seq_offset,
                  key_cache_out, value_cache_out);
    }
    // omni
    else if (TILING_KEY_IS(350000000)) {
        ScatterPaKvCache::ScatterPaKvCacheCompressOmni<bfloat16_t> op;
        op.Init(&pipe, &tilingData);
        op.Method(key, value, key_cache_in, value_cache_in, slot_mapping, compress_lens, seq_lens, compress_seq_offset,
                  key_cache_out, value_cache_out);
    }
#endif
#endif
}
