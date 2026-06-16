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
 * \file qbmm_reduce_scatter_add_rms_norm_cast_tiling_data.h
 * \brief 定义TilingData
 */

#ifndef QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_H
#define QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_H

#include <kernel_tiling/kernel_tiling.h>
#include "quant_matmul_tiling_data.h"

struct QbmmReduceScatterAddRmsNormCastTilingInfo {
    uint32_t aivNum;
    uint32_t aicNum;
    uint32_t rankSize;
    float epsilon;
    float avgFactor;
    // qbmmParams
    uint32_t ubCalcM;
    uint32_t ubCalcN;
    uint32_t needUbBuffer;
};

struct QbmmReduceScatterAddRmsNormCastTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    QuantMatmulTilingData qbmmTilingdata;
    QbmmReduceScatterAddRmsNormCastTilingInfo qbmmReduceScatterAddRmsNormCastTilingInfo;
};

#endif // QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_H