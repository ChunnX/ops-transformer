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
 * \file moe_distribute_quant_layout.h
 * \brief QuantLayout - 声明量化模式对 UB 输出大小、对齐的需求
 *        无状态组件，仅静态成员函数
 */

#ifndef MOE_DISTRIBUTE_QUANT_LAYOUT_H
#define MOE_DISTRIBUTE_QUANT_LAYOUT_H

#include "moe_distribute_v2_constant.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "quantize_functions.h"
#endif

namespace Mc2Kernel {
namespace MoeDistributeQuant {

using namespace AscendC;

template <typename XIn, typename XOut, int32_t QuantMode, bool IsSmoothScaleExist>
struct QuantLayout {
    __aicore__ static inline uint32_t OutTokenBytes(uint32_t axisH);
    __aicore__ static inline uint32_t OutScaleBytes(uint32_t axisH);
    __aicore__ static inline uint32_t WorkUbBytes(uint32_t axisH);
    static constexpr bool HasScaleOut();
};

template <typename XIn, typename XOut, int32_t QuantMode, bool IsSmoothScaleExist>
__aicore__ inline uint32_t QuantLayout<XIn, XOut, QuantMode, IsSmoothScaleExist>::OutTokenBytes(uint32_t axisH)
{
    if constexpr (QuantMode == UNQUANT || QuantMode == STATIC_QUANT || QuantMode == PERTOKEN_DYNAMIC_QUANT) {
        return axisH * sizeof(XOut);
    }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    else if constexpr (QuantMode == MX_QUANT) {
        if constexpr (Std::IsSame<XOut, fp4x2_e2m1_t>::value || Std::IsSame<XOut, fp4x2_e1m2_t>::value) {
            return Align256(Ceil(axisH, FP4_ELEMS_PER_BYTE));
        } else {
            return Align256(axisH) * sizeof(XOut);
        }
    } else if constexpr (QuantMode == PERGROUP_DYNAMIC_QUANT) {
        return Align128(axisH) * sizeof(XOut);
    }
#endif
    return axisH * sizeof(XOut);
}

template <typename XIn, typename XOut, int32_t QuantMode, bool IsSmoothScaleExist>
__aicore__ inline uint32_t QuantLayout<XIn, XOut, QuantMode, IsSmoothScaleExist>::OutScaleBytes(uint32_t axisH)
{
    if constexpr (QuantMode == UNQUANT && IsSmoothScaleExist) {
        return 0;
    } else if constexpr (QuantMode == PERTOKEN_DYNAMIC_QUANT) {
        return sizeof(float);
    }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    else if constexpr (QuantMode == MX_QUANT) {
        return Align2(Ceil32(axisH)) * sizeof(fp8_e8m0_t);
    } else if constexpr (QuantMode == PERGROUP_DYNAMIC_QUANT) {
        return Ceil128(axisH) * sizeof(float);
    }
#endif
    return 0;
}

template <typename XIn, typename XOut, int32_t QuantMode, bool IsSmoothScaleExist>
__aicore__ inline uint32_t QuantLayout<XIn, XOut, QuantMode, IsSmoothScaleExist>::WorkUbBytes(uint32_t axisH)
{
    if constexpr (QuantMode == UNQUANT && !IsSmoothScaleExist) {
        return 0;
    } else if constexpr (QuantMode == UNQUANT && IsSmoothScaleExist) {
        return axisH * sizeof(float) * 2;
    } else if constexpr (QuantMode == STATIC_QUANT || QuantMode == PERTOKEN_DYNAMIC_QUANT) {
        return axisH * sizeof(float) * 2;
    }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    else if constexpr (QuantMode == MX_QUANT) {
        return Align128(axisH) * sizeof(XIn);
    } else if constexpr (QuantMode == PERGROUP_DYNAMIC_QUANT) {
        return Align128(axisH) * sizeof(float);
    }
#endif
    return 0;
}

template <typename XIn, typename XOut, int32_t QuantMode, bool IsSmoothScaleExist>
constexpr bool QuantLayout<XIn, XOut, QuantMode, IsSmoothScaleExist>::HasScaleOut()
{
    if constexpr (QuantMode == UNQUANT && IsSmoothScaleExist) {
        return true;
    } else if constexpr (QuantMode == PERTOKEN_DYNAMIC_QUANT) {
        return true;
    }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    else if constexpr (QuantMode == MX_QUANT || QuantMode == PERGROUP_DYNAMIC_QUANT) {
        return true;
    }
#endif
    return false;
}

} // namespace MoeDistributeQuant
} // namespace Mc2Kernel

#endif // MOE_DISTRIBUTE_QUANT_LAYOUT_H