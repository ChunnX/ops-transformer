/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_NUMERIC_SIZE_HPP
#define CATLASS_NUMERIC_SIZE_HPP

#include "catlass/catlass.hpp"

namespace Catlass {

#if defined(__NPU_ARCH__)
using AscendC::SizeOfBits;
#else
template <typename T>
struct SizeOfBits {
    static constexpr size_t value = sizeof(T) * 8;
};

#if (defined (CATLASS_ARCH) && CATLASS_ARCH == 3510)
template <>
struct SizeOfBits<float4_e2m1x2_t> {
    static constexpr size_t value = 4;
};

template <>
struct SizeOfBits<float4_e1m2x2_t> {
    static constexpr size_t value = 4;
};
#endif
#endif

/// Returns the number of bytes required to hold a specified number of bits
template <typename ReturnType = size_t, typename T>
CATLASS_HOST_DEVICE
constexpr ReturnType BitsToBytes(T bits)
{
    return (static_cast<ReturnType>(bits) + static_cast<ReturnType>(7)) / static_cast<ReturnType>(8);
}

/// Returns the number of bits required to hold a specified number of bytes
template <typename ReturnType = size_t, typename T>
CATLASS_HOST_DEVICE
constexpr ReturnType BytesToBits(T bytes)
{
    return static_cast<ReturnType>(bytes) * static_cast<ReturnType>(8);
}

} // namespace Catlass

#endif // CATLASS_NUMERIC_SIZE_HPP
