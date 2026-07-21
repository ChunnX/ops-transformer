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
 * \file quant_matmul_reduce_scatter_tiling_data.h
 * \brief 定义TilingData
 */

#ifndef QUANT_MATMUL_REDUCE_SCATTER_TILING_DATA_H
#define QUANT_MATMUL_REDUCE_SCATTER_TILING_DATA_H

#include <kernel_tiling/kernel_tiling.h>
#if __has_include("quant_matmul/op_host/op_tiling/quant_matmul_tiling_data.h")
#include "quant_matmul/op_host/op_tiling/quant_matmul_tiling_data.h"
#else
#include "../../3rd/quant_matmul/op_host/op_tiling/quant_matmul_tiling_data.h"
#endif
// TilingData 部分的结构体复用、头文件引用与文件放置位置待后续探讨确定

struct QuantMatmulReduceScatterTilingInfo { 
    uint32_t aivNum;
    uint32_t aicNum;
    uint32_t rankSize;
    uint32_t ubCalcM;
    uint32_t ubCalcN;
    uint32_t needUbBuffer;
};

struct QuantMatmulReduceScatterTilingData {
    QuantMatmulTilingData qbmmTilingdata;
    QuantMatmulReduceScatterTilingInfo quantMatmulReduceScatterTilingInfo;
};

#endif // QUANT_MATMUL_REDUCE_SCATTER_TILING_DATA_H