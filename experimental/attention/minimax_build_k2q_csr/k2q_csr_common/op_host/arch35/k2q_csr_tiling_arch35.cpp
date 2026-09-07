/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 *
 * \file k2q_csr_tiling_arch35.cpp
 * \brief A5 Host tiling 辅助（供 cmake 在 ARCH_DIRECTORY=arch35 时自动编入）
 *
 * 逻辑以 header-only 形式提供；本文件保证 op_host/arch35 被 obj_func 的
 * GLOB `*_tiling*.cpp` 命中，交叉编译 ascend950 时 Host 侧一并参与链接。
 */
#include "k2q_csr_tiling_arch35.h"

namespace optiling {
namespace k2q_csr_arch35 {

// 显式实例化锚点，避免空翻译单元在部分工具链被丢弃
volatile int g_k2qCsrArch35TilingAnchor = 0;

} // namespace k2q_csr_arch35
} // namespace optiling
