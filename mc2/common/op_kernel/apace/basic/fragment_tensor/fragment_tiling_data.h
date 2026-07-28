/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fragment_tiling_data.h
 * \brief FragmentTensor 的 tiling 元信息，所有 fragment 大小相同。
 */

#pragma once

#include <cstdint>
#include <cassert>

namespace Apace {
namespace Utils {

using FragmentIdx = uint32_t;
using FragmentCount = uint32_t;

constexpr uint32_t MAX_FRAGMENT_COUNT = 32;

/*!
 * \brief FragmentTensor 元信息，所有 fragment 沿 AssembleAxis 均匀切分。
 */
template<uint32_t Dims>
struct FragmentTilingData {
    uint32_t AssembleAxis;
    uint64_t assembledShape[Dims];
    uint64_t fragmentSize;
    uint64_t realFragmentSize;
    uint64_t fragmentStride[Dims];
    uint32_t fragmentCnt;
    
    /*!
     * \brief 验证约束关系（Host侧调用）
     */
    bool Validate() const {
        if (assembledShape[AssembleAxis] != fragmentSize * fragmentCnt) {
            return false;
        }
        
        if (fragmentSize <= 0) {
            return false;
        }
        
        if (realFragmentSize == 0 || realFragmentSize > fragmentSize) {
            return false;
        }
        
        if (fragmentCnt <= 0) {
            return false;
        }
        
        if (AssembleAxis >= Dims) {
            return false;
        }
        
        return true;
    }
};

} // namespace Utils
} // namespace Apace