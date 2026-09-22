/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Compatibility alias for the DN block selection.
#ifndef FLASH_ATTN_KERNEL_DN_H_
#define FLASH_ATTN_KERNEL_DN_H_

#include "flash_attn_block_cube_dn.h"
#include "flash_attn_block_vec_dn.h"
#include "flash_attn_kernel.h"

namespace FlashAttnKernel {
template <typename FA_T, typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
using FlashAttentionNoQuantGqaKernelDn =
    FlashAttentionNoQuantGqaKernel<FA_T, CubeBlockType, VecFaBlockType, VecFdBlockType, true>;
} // namespace FlashAttnKernel

#endif // FLASH_ATTN_KERNEL_DN_H_
