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
 * \file gather_pa_kv_cache_tiling_key.h
 * \brief Tiling key declare for GatherPaKvCache SIMT kernel (ND format)
 *
 * schMode encodes (isSeqLensCumsum, hasSeqOffset) as two bits:
 *   bit 0 = isSeqLensCumsum, bit 1 = hasSeqOffset
 */

#ifndef GATHER_PA_KV_CACHE_SIMT_TILING_KEY_H
#define GATHER_PA_KV_CACHE_SIMT_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define GATHER_PA_KV_CACHE_SCENE_1100 0
#define GATHER_PA_KV_CACHE_SCENE_1110 1
#define GATHER_PA_KV_CACHE_SCENE_1101 2
#define GATHER_PA_KV_CACHE_SCENE_1111 3

ASCENDC_TPL_ARGS_DECL(
    GatherPaKvCacheSimt,
    ASCENDC_TPL_UINT_DECL(schMode, 2,
        ASCENDC_TPL_UI_LIST,
        GATHER_PA_KV_CACHE_SCENE_1100,
        GATHER_PA_KV_CACHE_SCENE_1110,
        GATHER_PA_KV_CACHE_SCENE_1101,
        GATHER_PA_KV_CACHE_SCENE_1111)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(schMode,
            ASCENDC_TPL_UI_LIST,
            GATHER_PA_KV_CACHE_SCENE_1100,
            GATHER_PA_KV_CACHE_SCENE_1110,
            GATHER_PA_KV_CACHE_SCENE_1101,
            GATHER_PA_KV_CACHE_SCENE_1111),
        ASCENDC_TPL_TILING_STRUCT_SEL(GatherPaKvCacheSimtTilingData)
    )
);

#endif // GATHER_PA_KV_CACHE_SIMT_TILING_KEY_H
