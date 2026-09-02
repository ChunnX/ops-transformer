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
 * \file copy_gm_to_l1.h
 * \brief
 */
#pragma once

#include "tensor_api/tensor.h"

namespace Blaze::Gemm::Tile {
using AscendC::Te::C0_ELEMENT;

struct CopySliceGM2L1 {
    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src)
    {
        using srcType = typename U::elementType;
        auto layoutGm = src.Layout(); // shape: [ndNum, [sliceM, curK]], stride: [oriM * k, [k, 1]]
        auto layoutL1 = dst.Layout(); // l1 shape: [mL1, kL1] ==> NZ: ((m0, m1), (k0, k1))

        auto m0 = AscendC::Te::Get<0>(AscendC::Te::Get<0>(layoutL1.Shape()));
        auto m1 = AscendC::Te::Get<1>(AscendC::Te::Get<0>(layoutL1.Shape()));
        uint32_t mL1 = m1 * m0; // curML1
        uint16_t ndNum = static_cast<uint16_t>(AscendC::Te::Get<0>(layoutGm.Shape()));
        uint16_t nValue = static_cast<uint16_t>(AscendC::Te::Get<0>(AscendC::Te::Get<1>(layoutGm.Shape())));
        uint32_t dValue = static_cast<uint32_t>(AscendC::Te::Get<1>(AscendC::Te::Get<1>(layoutGm.Shape())));
        uint64_t srcDValue = AscendC::Te::Get<0>(AscendC::Te::Get<1>(layoutGm.Stride()));
        uint32_t dstNzC0Stride = AscendC::Std::ceil_align(mL1, AscendC::BLOCK_CUBE);
        uint64_t srcNdMatrixStride = AscendC::Te::Get<0>(layoutGm.Stride());
        uint16_t dstNzNStride = 1;
        uint32_t dstNzMatrixStride = nValue * C0_ELEMENT<srcType>;
        uint64_t loop1SrcStride = srcDValue * sizeof(srcType);
        uint64_t loop4SrcStride = srcNdMatrixStride * sizeof(srcType);
        uint16_t loop2DstStride = dstNzNStride;  // loop2_dst_stride = dst_nz_n_stride
        uint16_t loop3DstStride = dstNzC0Stride; // loop3_dst_stride = dst_nz_c0_Stride
        // loop4_dst_stride: dst_nz_matrix_stride / C0_element
        uint16_t loop4DstStride = dstNzMatrixStride / C0_ELEMENT<srcType>;
        uint8_t cacheMode = src.Engine().GetCacheMode();

        if constexpr (sizeof(srcType) == sizeof(half)) {
            CopyGmToCbufMultiNd2nz((__cbuf__ half*)(dst.Data().Get()), (__gm__ half*)(src.Data().Get()), ndNum,
                                   loop2DstStride, loop3DstStride, loop4DstStride, loop1SrcStride, cacheMode, nValue,
                                   dValue, loop4SrcStride, false);
        } else if constexpr (sizeof(srcType) == sizeof(float)) {
            CopyGmToCbufMultiNd2nz((__cbuf__ float*)(dst.Data().Get()), (__gm__ float*)(src.Data().Get()), ndNum,
                                   loop2DstStride, loop3DstStride, loop4DstStride, loop1SrcStride, cacheMode, nValue,
                                   dValue, loop4SrcStride, false);
        }
    }

    template <typename T>
    __aicore__ inline static void CopyGmToCbufMultiNd2nz(__cbuf__ T* dst, __gm__ T* src, uint16_t ndNum,
                                                         uint16_t loop2DstStride, uint16_t loop3DstStride,
                                                         uint16_t loop4DstStride, uint64_t loop1SrcStride,
                                                         uint8_t cacheMode, uint16_t nValue, uint32_t dValue,
                                                         uint64_t loop4SrcStride, bool enableSmallC0)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (AscendC::Te::CURRENT_ARCH_VERSION == AscendC::Te::ArchVersion::V3510) {
            uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
            mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
            mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
            mte2NzPara |= static_cast<uint64_t>(ndNum);                        // MTE2_NZ_PARA[15:0]
            AscendC::Te::SetMTE2NzPara(mte2NzPara); // CCE: store parameters for ND2NZ DMA instructions
            asc_copy_gm2l1_nd2nz(dst, src, loop1SrcStride, cacheMode, nValue, dValue, loop4SrcStride, enableSmallC0);
        }
    }

    template <typename T>
    __aicore__ inline static void CopyGmToCbufMultiDn2nz(uint64_t dstOffset, __gm__ T* src, uint16_t dnNum,
                                                         uint32_t dValue, uint16_t nValue, uint64_t srcDnMatrixStride,
                                                         uint64_t srcDValue, uint16_t dstNzC0Stride,
                                                         uint16_t dstNzNStride, uint32_t dstNzMatrixStride)
    {
        if ASCEND_IS_AIV {
            return;
        }
        AscendC::GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(src);
        AscendC::LocalTensor<uint8_t> l1TensorByte(AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE);
        auto l1Tensor = l1TensorByte[dstOffset].template ReinterpretCast<T>();

        AscendC::Dn2NzParams dn2nzParams;
        dn2nzParams.dnNum = dnNum;
        dn2nzParams.dValue = dValue;
        dn2nzParams.nValue = nValue;
        dn2nzParams.srcDnMatrixStride = srcDnMatrixStride;
        dn2nzParams.srcDValue = srcDValue;
        dn2nzParams.dstNzC0Stride = dstNzC0Stride;
        dn2nzParams.dstNzNStride = dstNzNStride;
        dn2nzParams.dstNzMatrixStride = dstNzMatrixStride;
        AscendC::DataCopy(l1Tensor, gmTensor, dn2nzParams);
    }
};
} // namespace Blaze::Gemm::Tile

namespace AscendC {
namespace Te {

// 特化Traits，绑定自定义GM->L1拷贝实现
template <typename Traits>
struct CopyTraits<Blaze::Gemm::Tile::CopySliceGM2L1, Traits>
    : public CopyTraits<Blaze::Gemm::Tile::CopySliceGM2L1, Traits, Blaze::Gemm::Tile::CopySliceGM2L1, Traits> {};

template <>
struct CopyTraits<Blaze::Gemm::Tile::CopySliceGM2L1>
    : public CopyTraits<Blaze::Gemm::Tile::CopySliceGM2L1, CopyGM2L1TraitDefault> {};

} // namespace Te
} // namespace AscendC
