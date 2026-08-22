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
// The head/tail QuantMatmulTilingData entries describe the assembled logical M
// of one communication slice: sliceM * rankDim. The MMRS QBMM kernel splits a
// scheduler block at rank boundaries and maps each sub-block to the same
// local-M interval in every rank segment. CommTilingData
// describes how the slices cover each rank's output-M range, and
// nonSplitAxisSize is the output N dimension. Keeping a separate tail tiling
// lets the kernel pipeline Matmul and ReduceScatter without running the full
// [rankM, N] Matmul before communication starts. Temporary C uses a
// [slice][rank][sliceM][N] layout to keep each ReduceScatter input contiguous;
// the operator inputs remain rank-major.
struct hcommMatmulReduceScatterV2TilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    CommTilingData commTilingData;
    QuantMatmulTilingData tileQbmmTilingData;
    QuantMatmulTilingData tailQbmmTilingData;
};
