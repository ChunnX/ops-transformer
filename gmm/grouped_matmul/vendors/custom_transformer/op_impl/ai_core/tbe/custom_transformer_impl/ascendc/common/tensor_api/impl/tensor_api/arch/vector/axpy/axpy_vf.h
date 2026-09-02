/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#if !defined(ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#warning                                                                                                               \
    "impl/tensor_api/arch/vector/axpy/axpy_vf.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file axpy_vf.h
* \brief
*/
#ifndef IMPL_TENSOR_API_ARCH_VECTOR_AXPY_AXPY_VF_H
#define IMPL_TENSOR_API_ARCH_VECTOR_AXPY_AXPY_VF_H

#include "impl/tensor_api/arch/vector/axpy/instruction.h"
#include "impl/tensor_api/arch/vector/utils/mask_utils.h"

namespace AscendC {
namespace Te {

template<typename CalcFunc, typename TraitType>
class AxpyVF {
public:
    template<typename DstRegType, typename SrcRegType, typename T, typename U, typename S>
    __simd_vf__ inline static void Run(__ubuf__ T* dst, __ubuf__ U* src, S value, uint32_t dataSize)
    {
        constexpr uint32_t oneRepSize = static_cast<uint32_t>(asc_get_vf_len() / sizeof(T));
        uint16_t repeat = static_cast<uint16_t>((dataSize + oneRepSize - 1) / oneRepSize);

        vector_bool vmask;
        DstRegType reg_dst;
        SrcRegType reg_src;

        for (uint16_t i = 0; i < repeat; ++i) {
            vmask = Inst::UpdateMask::template Run<uint32_t>(dataSize);
            asc_loadalign(reg_src, src + i * oneRepSize);
            asc_loadalign(reg_dst, dst + i * oneRepSize);
            CalcFunc::template Run(reg_dst, reg_src, value, vmask);
            asc_storealign(dst + i * oneRepSize, reg_dst, vmask);
        }
    }
};

template<typename CalcFunc, typename TraitType>
class Transform2AxpyVF {
public:
    template<typename T, typename U, typename S>
    __aicore__ inline static void Run(const T& dst, const U& src, const S& value)
    {
  		using srcType = GetAttributeElementType<typename U::elementType*>;
  		using dstType = GetAttributeElementType<typename T::elementType*>;
        using SrcRegType = typename VectorTypeTransform::template Get<srcType>;
        using DstRegType = typename VectorTypeTransform::template Get<dstType>;

        const uint32_t dataSize = dst.Size();

        AxpyVF<CalcFunc, TraitType>::template Run<DstRegType, SrcRegType>(dst.Data().Get(), src.Data().Get(), value, dataSize);
    }
};

}
}

#endif // IMPL_TENSOR_API_ARCH_VECTOR_AXPY_AXPY_VF_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
