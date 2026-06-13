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
 * \file inplace_partial_rotary_mul_grad_tiling_key.h
 * \brief
 */

#ifndef __INPLACE_PARTIAL_ROTARY_MUL_GRAD_TILING_KEY_H__
#define __INPLACE_PARTIAL_ROTARY_MUL_GRAD_TILING_KEY_H__

#include <cstdint>

enum class InplacePartialRotaryMulGradTilingKey : uint32_t {
    TILING_KEY_ABA = 201,
    TILING_KEY_BA = 202,
    TILING_KEY_BAB = 203,
    TILING_KEY_AB = 204,
    TILING_KEY_A = 205,
    TILING_KEY_B = 206,
};

#endif
