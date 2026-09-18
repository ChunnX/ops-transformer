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
 * \file mc2_matmul_aiv_mode_tiling_common.h
 * \brief Common tiling fields for MC2 matmul AIV mode kernels.
 */

#ifndef MC2_MATMUL_AIV_MODE_TILING_COMMON_H
#define MC2_MATMUL_AIV_MODE_TILING_COMMON_H

#include <cstdint>
#include <type_traits>

struct Mc2MatmulAivModeTiling {
    int32_t m0 = -1;
    int32_t k0 = -1;
    int32_t n0 = -1;
    int32_t mLoop = -1;
    int32_t kLoop = -1;
    int32_t nLoop = -1;
    int32_t swizzlCount = -1;
    int32_t swizzlDirect = -1;
    int32_t pValue = -1;
    int32_t ubMoveNum = -1;
    int32_t commNpuSplit = -1;
    int32_t commDataSplit = -1;
};

static_assert(std::is_standard_layout_v<Mc2MatmulAivModeTiling>);
static_assert(sizeof(Mc2MatmulAivModeTiling) == 12 * sizeof(int32_t));

#endif // MC2_MATMUL_AIV_MODE_TILING_COMMON_H
