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
// Keep communication slicing and Matmul tiling independent, following the
// AllToAllMatmul/AllGatherMatmul APACE contract:
//   * CommTilingData describes the first/second communication groups.
//   * tileQbmmTilingData describes the complete logical Matmul M
//     (perRankM * rankDim) and is shared by every communication slice.
// The MMRS QBMM kernel creates a scheduler for each communication slice so a
// ReduceScatter can be issued as soon as that slice is complete, while the
// BlockMmad base shape and L1 configuration come from this single Matmul
// tiling. Temporary C uses [slice][rank][sliceM][N] order; inputs stay
// rank-major.
struct hcommMatmulReduceScatterV2TilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    CommTilingData commTilingData;
    QuantMatmulTilingData tileQbmmTilingData;
};
