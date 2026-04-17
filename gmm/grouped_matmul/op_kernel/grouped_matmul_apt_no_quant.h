/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_MATMUL_APT_NO_QUANT_H
#define GROUPED_MATMUL_APT_NO_QUANT_H

#if !defined(V310_GMM_QUANT) && !defined(V310_GMM_ANTI_QUANT)
#include "arch35/non_quant/grouped_matmul_basic_kernel.h"
#include "arch35/non_quant/grouped_matmul_tiling_key.h"

#define GROUPED_MATMUL_NO_QUANT_IMPL_BODY(TDtypeY, kWFormat, NO_QUANT_B_TRANS, NO_QUANT_A_TRANS)                       \
    REGISTER_TILING_DEFAULT(GMMNoQuantTilingData);                                                                     \
    if constexpr (NO_QUANT_B_TRANS == GMM_NO_TRANS && NO_QUANT_A_TRANS == GMM_NO_TRANS) {                             \
        if constexpr (kWFormat == CubeFormat::NZ) {                                                                    \
            GmmNoQuantAswt<layout::RowMajor, layout::Nz>(x, weight, bias, groupList, y, tiling);                      \
        } else {                                                                                                       \
            GmmNoQuantAswt<layout::RowMajor, layout::RowMajor>(x, weight, bias, groupList, y, tiling);                \
        }                                                                                                              \
    } else if constexpr (NO_QUANT_B_TRANS == GMM_NO_TRANS && NO_QUANT_A_TRANS == GMM_TRANS) {                         \
        if ASCEND_IS_AIV {                                                                                             \
            EmptyTensor<TDtypeY>(x, weight, groupList, y, tiling);                                                     \
        }                                                                                                              \
        if ASCEND_IS_AIC {                                                                                             \
            if constexpr (kWFormat == CubeFormat::NZ) {                                                                \
                GmmNoQuantAswt<layout::ColumnMajor, layout::Nz>(x, weight, bias, groupList, y, tiling);                \
            } else {                                                                                                   \
                GmmNoQuantAswt<layout::ColumnMajor, layout::RowMajor>(x, weight, bias, groupList, y, tiling);          \
            }                                                                                                          \
        }                                                                                                              \
    } else if constexpr (NO_QUANT_B_TRANS == GMM_TRANS && NO_QUANT_A_TRANS == GMM_NO_TRANS) {                         \
        if constexpr (kWFormat == CubeFormat::NZ) {                                                                    \
            GmmNoQuantAswt<layout::RowMajor, layout::Zn>(x, weight, bias, groupList, y, tiling);                      \
        } else {                                                                                                       \
            GmmNoQuantAswt<layout::RowMajor, layout::ColumnMajor>(x, weight, bias, groupList, y, tiling);              \
        }                                                                                                              \
    }

#endif
#endif