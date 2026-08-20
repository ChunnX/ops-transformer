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
 * \file matmul_reduce_scatter_v2_tiling_data.h
 * \brief APACE Hcomm tiling data for MatmulReduceScatterV2 MX quantization.
 */

#pragma once

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "../../../tiling/quant_matmul_tiling_data.h"
#include "../../../tiling/comm_tiling_data.h"

// Hcomm MX quantized Matmul + ReduceScatter tiling data.
// CommTilingData describes the per-rank output-M pipeline split, and
// nonSplitAxisSize is the output N dimension. Head/tail Matmul tilings
// describe one pipeline round assembled from rankDim discontiguous M
// fragments. paddedTailM is the logical fragment size used by the tail
// scheduler; FragmentTensor limits physical access to tailM.
struct hcommMatmulReduceScatterV2TilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    CommTilingData commTilingData;
    QuantMatmulTilingData headQbmmTilingData;
    QuantMatmulTilingData tailQbmmTilingData;
    uint32_t paddedTailM{0};
};
