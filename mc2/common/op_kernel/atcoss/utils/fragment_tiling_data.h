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
 * \brief FragmentTensor元信息结构体定义（简化设计）
 * 
 * 设计原则：
 * - 模板化：支持任意维度（Dims参数）
 * - 简化设计：所有Fragment大小相同（uniformSize）
 * - 简单约束：assembledShape[AssembleAxis] = fragmentSize * fragmentCnt
 * - Slice推导：简单除法和余数（O(1)）
 * - Host侧计算：fragmentCnt由Host侧计算
 * - addrList在Kernel内构建：不在TilingData中存储
 */

#pragma once

#include <cstdint>
#include <cassert>

namespace Dista {
namespace Utils {

using FragmentIdx = uint32_t;
using FragmentCount = uint32_t;

constexpr uint32_t MAX_FRAGMENT_COUNT = 32;

/*!
 * \brief FragmentTensor元信息结构体（简化设计，均匀Fragment）
 * 
 * \tparam Dims 维度数量（编译期确定）
 * 
 * 关键约束：
 * - assembledShape[AssembleAxis] = fragmentSize * fragmentCnt
 * - 所有Fragment大小相同（fragmentSize）
 * 
 * 数据字段说明：
 * - AssembleAxis: 聚合轴索引（0 ~ Dims-1），多个Fragment沿此轴聚合拼接
 * - assembledShape: AssembledShape各维度大小（用户输入）
 * - fragmentSize: 单个Fragment的切分轴大小（所有Fragment相同）
 * - fragmentStride: FragmentStride布局（用户输入，所有Fragment共用）
 * - fragmentCnt: Fragment总数（Host侧计算）
 * 
 * Slice推导公式（均匀Fragment）：
 * - fragmentIdx = coord[AssembleAxis] / fragmentSize
 * - fragmentCoord = coord[AssembleAxis] % fragmentSize
 * 
 * Slice后动态推导（不均匀）：
 * - startFragmentIdx = startCoord / fragmentSize
 * - endFragmentIdx = (startCoord + length) / fragmentSize
 * - headSize = fragmentSize - (startCoord % fragmentSize)  [多Fragment时]
 * - tailSize = (startCoord + length) % fragmentSize         [多Fragment时]
 */
template<uint32_t Dims>
struct FragmentTilingData {
    uint32_t AssembleAxis;
    uint64_t assembledShape[Dims];
    uint64_t fragmentSize;
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
} // namespace Dista