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
 * \file all_gather_quant_matmul_tiling_data.h
 * \brief 定义TilingData
 */

#ifndef ALL_GATHER_QUANT_MATMUL_TILING_DATA_H_
#define ALL_GATHER_QUANT_MATMUL_TILING_DATA_H_

#include "kernel_tiling/kernel_tiling.h"

struct QbmmParams {
    uint32_t ubCalcM;
    uint32_t ubCalcN;
    uint32_t needUbBuffer;
};

struct AllGatherQuantMatmulInfo {
    QbmmParams qbmmParams;
    TCubeTiling matmulTiling;
};

struct AllGatherQuantMatmulTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    AllGatherQuantMatmulInfo tilingInfo;
};

#endif // ALL_GATHER_QUANT_MATMUL_H