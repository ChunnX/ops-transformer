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
* \file index_put_with_sort_v2_struct.h
* \brief
*/
#ifndef MHC_PRE_SINKHORN_BACKWARD_STRUCT_H
#define MHC_PRE_SINKHORN_BACKWARD_STRUCT_H
#include "ascendc/host_api/tiling/template_argument.h"
#include "kernel_tiling/kernel_tiling.h"

ASCENDC_TPL_ARGS_DECL(MhcPreSinkhornBackward,
    ASCENDC_TPL_BOOL_DECL(DETERMINISTIC, 0, 1),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(DETERMINISTIC, 0, 1),
        ASCENDC_TPL_TILING_STRUCT_SEL(MhcPreSinkhornBackwardTilingData)
    ),
)

class MhcPreSinkhornBackwardTilingData {
public:
    int64_t batchSize = 0;
    int64_t seqLength = 0;
    int64_t c = 0;
    int64_t n = 0;
    int64_t c0= 0;
    int64_t c1 = 0;
    int64_t cTail = 0;
    int64_t aivNum = 0;
    int64_t tileGradY = 0;
    int64_t tileHHat2 = 0;
    int64_t tileSize = 0;
    int64_t skIterCount = 0;
    int64_t ubSize = 0;
    float eps = 0;
    TCubeTiling mm1TilingData;
    TCubeTiling mm2TilingData;
};

#endif