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
 * \file allto_all_matmul_apace_tiling.h
 * \brief Tiling helper functions for AlltoAllMatmulApace operator
 */

#pragma once

#include <cstdint>
#include <string>

// Forward-declarations of apace types (include paths resolved via CMake -I mc2/common/op_kernel/apace)
#include "apace/tiling/comm_tiling_data.h"
#include "apace/tiling/quant_matmul_tiling_data.h"

// Tiling key field constants
#define MX_QUANT_MODE 6
#define DTYPE_BIAS_FP32 1
#define ALL2ALL_COMM_TYPE_UDMA 2

// Default head M size for communication splitting
#define DEFAULT_HEAD_M_SIZE 512

// dtype combo enumeration for kernel dispatch
enum AlltoAllDtypeCombo {
    E4M3_E4M3 = 0,
    E5M2_E5M2 = 1,
    E4M3_E5M2 = 2,
    E5M2_E4M3 = 3,
};

/**
 * @brief Compute the tiling key. Fixed to UDMA comm type.
 */
uint64_t GetMatmulTilingKey();

/**
 * @brief Build communication tiling data using the head-M-size strategy.
 *        M MUST be divisible by headMSize (infershape guarantees this).
 */
void BuildCommTilingData(uint64_t m, uint64_t ka, uint64_t headMSize, CommTilingData &commData);

/**
 * @brief Build matmul tiling data using apace QuantMatmulTilingSwat.
 *        Dispatches to the correct template instantiation based on aType/bType.
 */
void BuildMatmulTilingData(uint64_t m, uint64_t n, uint64_t ka, uint32_t aDtype, uint32_t bDtype,
                           QuantMatmulTilingData &matmulData);

/**
 * @brief Determine dtype combo from ge::DataType values.
 */
AlltoAllDtypeCombo GetDtypeCombo(uint32_t x1Dtype, uint32_t x2Dtype);
