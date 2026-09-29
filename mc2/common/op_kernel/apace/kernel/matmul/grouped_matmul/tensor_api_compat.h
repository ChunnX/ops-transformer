/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "tensor_api/tensor.h"

namespace Apace::TensorApiCompat {

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)

using NDExtLayoutPtn = AscendC::Te::NDExtLayoutPtn;
using DNExtLayoutPtn = AscendC::Te::DNExtLayoutPtn;
using NZLayoutPtn = AscendC::Te::NZLayoutPtn;
using ZNLayoutPtn = AscendC::Te::ZNLayoutPtn;
using NNLayoutPtn = AscendC::Te::NNLayoutPtn;
using ScaleANDLayoutPtn = AscendC::Te::ScaleANDLayoutPtn;
using ScaleADNLayoutPtn = AscendC::Te::ScaleADNLayoutPtn;

template <typename... Types>
using Shape = AscendC::Te::Shape<Types...>;

template <typename... Types>
using Coord = AscendC::Te::Coord<Types...>;

template <typename Pattern, typename Trait>
using FrameLayoutFormat = AscendC::Te::FrameLayoutFormat<Pattern, Trait>;

template <typename T>
inline constexpr auto C0Element = AscendC::Te::C0_ELEMENT<T>;

template <int64_t Index, typename T>
__aicore__ inline decltype(auto) Get(const T &value)
{
    return AscendC::Te::Get<Index>(value);
}

template <typename... Args>
__aicore__ inline auto MakeShape(const Args &...args)
{
    return AscendC::Te::MakeShape(args...);
}

template <typename... Args>
__aicore__ inline auto MakeStride(const Args &...args)
{
    return AscendC::Te::MakeStride(args...);
}

template <typename... Args>
__aicore__ inline auto MakeCoord(const Args &...args)
{
    return AscendC::Te::MakeCoord(args...);
}

template <typename ShapeType, typename StrideType>
__aicore__ inline auto MakeLayout(const ShapeType &shape, const StrideType &stride)
{
    return AscendC::Te::MakeLayout(shape, stride);
}

template <typename T>
__aicore__ inline auto MakeGmMemPtr(T *address)
{
    return AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(address);
}

template <typename Iterator, typename Layout>
__aicore__ inline auto MakeTensor(const Iterator &iterator, const Layout &layout)
{
    return AscendC::Te::MakeTensor(iterator, layout);
}

template <typename Tensor, typename CoordType, typename ShapeType>
__aicore__ inline auto Slice(Tensor &tensor, const CoordType &coord, const ShapeType &shape)
{
    return tensor.Slice(coord, shape);
}

template <typename Tensor>
__aicore__ inline void SetL2CacheDisable(Tensor &tensor)
{
    tensor.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
}

template <typename Tensor>
__aicore__ inline void SetL2CacheNormal(Tensor &tensor)
{
    tensor.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
}

__aicore__ inline auto MakeBiasLayout(int64_t m, int64_t n)
{
    return AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(m, n);
}

#else

using NDExtLayoutPtn = asc::te::nd_ext_layout_ptn;
using DNExtLayoutPtn = asc::te::dn_ext_layout_ptn;
using NZLayoutPtn = asc::te::nz_layout_ptn;
using ZNLayoutPtn = asc::te::zn_layout_ptn;
using NNLayoutPtn = asc::te::nn_layout_ptn;
using ScaleANDLayoutPtn = asc::te::scalea_nd_layout_ptn;
using ScaleADNLayoutPtn = asc::te::scalea_dn_layout_ptn;

template <typename... Types>
using Shape = asc::te::shape<Types...>;

template <typename... Types>
using Coord = asc::te::coord<Types...>;

template <typename Pattern, typename Trait>
using FrameLayoutFormat = asc::te::frame_layout_format<Pattern, Trait>;

template <typename T>
inline constexpr auto C0Element = asc::te::c0_element<T>;

template <int64_t Index, typename T>
__aicore__ inline decltype(auto) Get(const T &value)
{
    return asc::te::get<Index>(value);
}

template <typename... Args>
__aicore__ inline auto MakeShape(const Args &...args)
{
    return asc::te::make_shape(args...);
}

template <typename... Args>
__aicore__ inline auto MakeStride(const Args &...args)
{
    return asc::te::make_stride(args...);
}

template <typename... Args>
__aicore__ inline auto MakeCoord(const Args &...args)
{
    return asc::te::make_coord(args...);
}

template <typename ShapeType, typename StrideType>
__aicore__ inline auto MakeLayout(const ShapeType &shape, const StrideType &stride)
{
    return asc::te::make_layout(shape, stride);
}

template <typename T>
__aicore__ inline auto MakeGmMemPtr(T *address)
{
    return asc::te::make_mem_ptr<asc::te::location::gm>(address);
}

template <typename Iterator, typename Layout>
__aicore__ inline auto MakeTensor(const Iterator &iterator, const Layout &layout)
{
    return asc::te::make_tensor(iterator, layout);
}

template <typename Tensor, typename CoordType, typename ShapeType>
__aicore__ inline auto Slice(Tensor &tensor, const CoordType &coord, const ShapeType &shape)
{
    return tensor.slice(coord, shape);
}

template <typename Tensor>
__aicore__ inline void SetL2CacheDisable(Tensor &tensor)
{
    tensor.set_l2_cache_hint(asc::te::cache_mode::disable);
}

template <typename Tensor>
__aicore__ inline void SetL2CacheNormal(Tensor &tensor)
{
    tensor.set_l2_cache_hint(asc::te::cache_mode::normal);
}

__aicore__ inline auto MakeBiasLayout(int64_t m, int64_t n)
{
    return asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(m, n);
}

#endif

template <typename Layout>
inline constexpr bool IsScaleNz = AscendC::Std::is_same_v<Layout, NNLayoutPtn>;

template <typename T>
inline constexpr bool IsFp4 = AscendC::IsSameType<T, fp4x2_e2m1_t>::value;

template <typename T>
inline constexpr bool IsFp8 = AscendC::IsSameType<T, float8_e4m3fn_t>::value ||
                              AscendC::IsSameType<T, float8_e5m2_t>::value;

} // namespace Apace::TensorApiCompat
