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
 * \file fragment_tensor_api.h
 * \brief FragmentTensor API 总入口（简化设计）
 * 
 * 使用方式：
 * #include "atcoss/basic/fragment_tensor/fragment_tensor_api.h"
 * 
 * 设计原则（简化设计）：
 * - 所有Fragment大小相同（uniformSize）
 * - 简单约束：assembledShape[AssembleAxis] = fragmentSize * fragmentCnt
 * - Slice推导：简单除法和余数（O(1)）
 * - 动态设计：支持运行时shape
 * - FragmentTilingData：Host侧计算，包含元信息
 * - addrList：Kernel内构建，用户自定义传入
 */

#pragma once

#include "../../utils/fragment_tiling_data.h"
#include "fragment_tensor.h"

namespace Dista {
namespace Basic {

using namespace Dista::Utils;

/*!
 * \brief 创建FragmentTensor（简化接口）
 * 
 * \tparam Dims 维度数量（编译期确定）
 * \tparam MaxFragments 最大Fragment数量（默认32）
 * \param tilingData FragmentTilingData元信息（Host侧计算）
 * \param addrList Fragment地址列表（用户传入，大小=fragmentCnt）
 * \return FragmentTensor<Dims, MaxFragments>
 */
template<uint32_t Dims, uint32_t MaxFragments = MAX_FRAGMENT_COUNT>
__aicore__ inline auto MakeFragmentTensor(
    const FragmentTilingData<Dims>& tilingData,
    const GM_ADDR* addrList) {
    return FragmentTensor<Dims, MaxFragments>(tilingData, addrList);
}

} // namespace Basic
} // namespace Dista