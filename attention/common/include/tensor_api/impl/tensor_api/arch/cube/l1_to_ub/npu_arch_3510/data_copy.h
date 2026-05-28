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
    "impl/tensor_api/arch/cube/l1_to_ub/npu_arch_3510/data_copy.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file data_copy.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_L1_TO_UB_NPU_ARCH_3510_DATA_COPY_H
#define IMPL_TENSOR_API_ARCH_CUBE_L1_TO_UB_NPU_ARCH_3510_DATA_COPY_H

#include "../../../../utils/utils_impl.h"
#include "instruction.h"

namespace AscendC {
namespace Te {

struct CopyL12UBTrait {};

class DataCopyL12UB3510 {
public:
    template <const CopyL12UBTrait& trait, typename T, typename U>
    __aicore__ inline static void Run(const T& dst, const U& src)
    { Execute<trait>(dst, src); }

private:
    template <const CopyL12UBTrait& trait, typename T, typename U>
    __aicore__ inline static void Execute(const T& dst, const U& src)
    {
        using SRC_TYPE = typename U::elementType;
        using DST_TYPE = typename T::elementType;

        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        constexpr bool isNdSrcFormat = IsSatisfiedPtnFormatV<U, NDExtLayoutPtn> || IsSatisfiedPtnFormatV<U, NDLayoutPtn>;
        constexpr bool isNdDstFormat = IsSatisfiedPtnFormatV<T, NDExtLayoutPtn> || IsSatisfiedPtnFormatV<T, NDLayoutPtn>;
        constexpr bool isDnSrcFormat = IsSatisfiedPtnFormatV<U, DNExtLayoutPtn> || IsSatisfiedPtnFormatV<U, DNLayoutPtn>;
        constexpr bool isDnDstFormat = IsSatisfiedPtnFormatV<T, DNExtLayoutPtn> || IsSatisfiedPtnFormatV<T, DNLayoutPtn>;

        uint16_t blockCount = 0;
        uint32_t blockLen = 0;
        int64_t srcStride = 0;
        int64_t dstStride = 0;

        if constexpr (isNdSrcFormat && isNdDstFormat) {
            blockCount = GetTotalRowShape(srcLayout);
            // Next three parameters are in unit of 32B
            blockLen = Std::ceil_division(GetTotalColumnShape(srcLayout), C0_ELEMENT<SRC_TYPE>);

            uint32_t srcRowStride;
            uint32_t dstRowStride;
            if constexpr (IsSatisfiedPtnFormatV<U, NDLayoutPtn>) {
                srcRowStride = GetElement<AttrInfo::Stride, AttrInfo::Row>(srcLayout);
            } else {
                srcRowStride = GetElement<AttrInfo::Stride, AttrInfo::Row, 1>(srcLayout);
            }
            if constexpr (IsSatisfiedPtnFormatV<T, NDLayoutPtn>) {
                dstRowStride = GetElement<AttrInfo::Stride, AttrInfo::Row>(dstLayout);
            } else {
                dstRowStride = GetElement<AttrInfo::Stride, AttrInfo::Row, 1>(dstLayout);
            }
            srcStride = Std::ceil_division(srcRowStride - GetTotalColumnShape(srcLayout), C0_ELEMENT<SRC_TYPE>);
            dstStride = Std::ceil_division(dstRowStride - GetTotalColumnShape(srcLayout), C0_ELEMENT<DST_TYPE>);

        } else if constexpr (isDnSrcFormat && isDnDstFormat) {
            blockCount = GetTotalColumnShape(srcLayout);
            // Next three parameters are in unit of 32B
            blockLen = Std::ceil_division(GetTotalRowShape(srcLayout), C0_ELEMENT<SRC_TYPE>);

            uint32_t srcColumnStride;
            uint32_t dstColumnStride;
            if constexpr (IsSatisfiedPtnFormatV<U, DNLayoutPtn>) {
                srcColumnStride = GetElement<AttrInfo::Stride, AttrInfo::Column>(srcLayout);
            } else {
                srcColumnStride = GetElement<AttrInfo::Stride, AttrInfo::Column, 1>(srcLayout);
            }
            if constexpr (IsSatisfiedPtnFormatV<T, DNLayoutPtn>) {
                dstColumnStride = GetElement<AttrInfo::Stride, AttrInfo::Column>(dstLayout);
            } else {
                dstColumnStride = GetElement<AttrInfo::Stride, AttrInfo::Column, 1>(dstLayout);
            }
            srcStride = Std::ceil_division(srcColumnStride - GetTotalRowShape(srcLayout), C0_ELEMENT<SRC_TYPE>);
            dstStride = Std::ceil_division(dstColumnStride - GetTotalRowShape(srcLayout), C0_ELEMENT<DST_TYPE>);

        } else if constexpr (IsSatisfiedPtnFormatV<U, NZLayoutPtn> && IsSatisfiedPtnFormatV<T, NZLayoutPtn>) {
            blockCount = GetElement<AttrInfo::Shape, AttrInfo::Column, 1>(srcLayout);
            // Next three parameters are in unit of 32B
            // note: C0_Byte_Size == 32B
            blockLen = GetTotalRowShape(srcLayout);

            srcStride = GetElement<AttrInfo::Stride, AttrInfo::Column, 1>(srcLayout) / C0_ELEMENT<SRC_TYPE> - blockLen;
            dstStride = GetElement<AttrInfo::Stride, AttrInfo::Column, 1>(dstLayout) / C0_ELEMENT<DST_TYPE> - blockLen;

        } else {
            static_assert((isNdSrcFormat && isNdDstFormat) || (isDnSrcFormat && isDnDstFormat)
                              || (IsSatisfiedPtnFormatV<U, NZLayoutPtn> && IsSatisfiedPtnFormatV<T, NZLayoutPtn>),
                          "Unsupported layout type combination for DataCopyL12UB3510");
        }
        CopyCbufToUbufInstr::DataCopy(dst, src, blockCount, blockLen, srcStride, dstStride);
        // ND和DN场景，需要保证UB和L1上申请的空间和tensor的stride满足32字节对齐，否则CopyCbufToUbuf会有问题，无法正确加载数据，导致数据错误
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_L1_TO_UB_NPU_ARCH_3510_DATA_COPY_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
