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
 * \file grouped_mat_mul_allto_allv_tiling.h
 */
#ifndef GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H_ARCH22__
#define GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H_ARCH22__

#if __has_include("../../quant_grouped_mat_mul_allto_allv/op_kernel/arch35/quant_grouped_mat_mul_allto_allv_tiling.h")
#include "../../quant_grouped_mat_mul_allto_allv/op_kernel/arch35/quant_grouped_mat_mul_allto_allv_tiling.h"
#elif __has_include("../../quant_grouped_mat_mul_allto_allv/arch35/quant_grouped_mat_mul_allto_allv_tiling.h")
#include "../../quant_grouped_mat_mul_allto_allv/arch35/quant_grouped_mat_mul_allto_allv_tiling.h"
#elif __has_include(                                                                                              \
    "../../../quant_grouped_mat_mul_allto_allv/op_kernel/arch35/quant_grouped_mat_mul_allto_allv_tiling.h")
#include "../../../quant_grouped_mat_mul_allto_allv/op_kernel/arch35/quant_grouped_mat_mul_allto_allv_tiling.h"
#else
#include "../../../quant_grouped_mat_mul_allto_allv/arch35/quant_grouped_mat_mul_allto_allv_tiling.h"
#endif

#pragma once

#endif