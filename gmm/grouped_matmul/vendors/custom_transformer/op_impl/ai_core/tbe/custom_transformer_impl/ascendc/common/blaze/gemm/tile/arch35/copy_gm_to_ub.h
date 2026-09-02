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
 * \file copy_gm_to_ub.h
 * \brief Custom copy primitive for packed B4 weight movement from GM to UB.
 */
#pragma once

#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Tile {

// Custom GM-to-UB copy for packed 4-bit weight tensors.
struct CopyGM2UBWeight {
    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src)
    {
        using SrcLayoutPattern = AscendC::Te::GetLayoutPattern<typename U::layoutType>;
        using DstLayoutPattern = AscendC::Te::GetLayoutPattern<typename T::layoutType>;
        constexpr bool IS_ZN_WEIGHT = AscendC::Std::is_same_v<SrcLayoutPattern, AscendC::Te::ZNLayoutPtn>;
        constexpr bool IS_DN_WEIGHT = AscendC::Std::is_same_v<SrcLayoutPattern, AscendC::Te::DNExtLayoutPtn>;
        static_assert(IS_ZN_WEIGHT || IS_DN_WEIGHT, "Packed weight copy only supports ZN and DNExt source layouts");
        static_assert(AscendC::Std::is_same_v<DstLayoutPattern, SrcLayoutPattern>,
            "Packed weight copy source and destination layouts must match");
        static_assert(
            sizeof(typename T::elementType) == sizeof(typename U::elementType) &&
                sizeof(typename T::elementType) == 1,
            "Packed weight copy requires matching packed source and destination elements");

        if constexpr (IS_DN_WEIGHT) {
            CopyDnPackedWeight(dst, src);
        } else {
            CopyZnPackedWeight(dst, src);
        }
    }

private:
    template <typename T, typename U>
    __aicore__ inline static void CopyDnPackedWeight(const T& dst, const U& src)
    {
        const auto& dstLayout = dst.Layout();
        const auto& srcLayout = src.Layout();

        uint8_t cacheMode = src.Engine().GetCacheMode();
        auto srcShape = AscendC::Te::GetShape(srcLayout);
        auto srcStrideTuple = AscendC::Te::GetStride(srcLayout);
        auto dstStrideTuple = AscendC::Te::GetStride(dstLayout);

        uint16_t blockCount = AscendC::Std::get<1>(AscendC::Std::get<1>(srcShape));
        uint32_t kLen = AscendC::Std::get<1>(AscendC::Std::get<0>(srcShape));
        uint32_t srcRowStride = AscendC::Std::get<1>(AscendC::Std::get<1>(srcStrideTuple));
        uint32_t dstRowStride = AscendC::Std::get<1>(AscendC::Std::get<1>(dstStrideTuple));

        // Packed FP4 stores two logical K elements in one byte. The direct
        // intrinsic consumes the byte distance between adjacent N-row starts.
        uint32_t blockLen = kLen >> 1U;
        int64_t srcRowSpanBytes = srcRowStride >> 1U;
        int64_t dstRowSpanBytes = dstRowStride >> 1U;
        asc_copy_gm2ub_align(
            (__ubuf__ uint8_t*)dst.Data().Get(), (__gm__ uint8_t*)src.Data().Get(), blockCount, blockLen, 0, 0, false,
            cacheMode, srcRowSpanBytes, dstRowSpanBytes);
    }

    template <typename T, typename U>
    __aicore__ inline static void CopyZnPackedWeight(const T& dst, const U& src)
    {
        const auto& dstLayout = dst.Layout();
        const auto& srcLayout = src.Layout();
        uint8_t cacheMode = src.Engine().GetCacheMode();

        // Get shape and stride
        auto srcShape = AscendC::Te::GetShape(srcLayout);
        auto srcStrideTuple = AscendC::Te::GetStride(srcLayout);
        auto dstStrideTuple = AscendC::Te::GetStride(dstLayout);

        // Extract k1 from shape: ((c0, k1), (n0, n1)).
        uint16_t blockCount = AscendC::Std::get<1>(AscendC::Std::get<0>(srcShape));

        // For packed B4 (2 elements per byte), convert element strides to byte strides via >> 1.
        uint32_t blockLen = AscendC::Std::get<1>(AscendC::Std::get<0>(dstStrideTuple)) >> 1;
        int64_t srcStride = AscendC::Std::get<1>(AscendC::Std::get<0>(srcStrideTuple)) >> 1;
        int64_t dstStride = AscendC::Std::get<1>(AscendC::Std::get<0>(dstStrideTuple)) >> 1;

        asc_copy_gm2ub_align(
            (__ubuf__ uint8_t*)dst.Data().Get(), (__gm__ uint8_t*)src.Data().Get(), blockCount, blockLen, 0, 0, false,
            cacheMode, srcStride, dstStride);
    }
};

} // namespace Tile
} // namespace Gemm
} // namespace Blaze

// Register CopyTraits for the custom GM-to-UB packed-weight copy.
template <>
struct AscendC::Te::CopyTraits<Blaze::Gemm::Tile::CopyGM2UBWeight>
    : public AscendC::Te::CopyTraits<
          Blaze::Gemm::Tile::CopyGM2UBWeight, AscendC::Te::CopyGM2UBTraitDefault, Blaze::Gemm::Tile::CopyGM2UBWeight,
          AscendC::Te::CopyGM2UBTraitDefault> {};
