/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_DN_UNMERGED_POLICY_H_
#define FLASH_ATTN_DN_UNMERGED_POLICY_H_
#include <cstdint>
namespace FlashAttnPolicy {
constexpr uint32_t DN_NO_MASK_N = 256U;
constexpr uint32_t DN_MASK_D64_N = 208U;
constexpr uint32_t DN_MASK_OTHER_N = 224U;
constexpr uint32_t DnUnmergedS2(uint32_t qkDim, bool hasMask)
{
    return hasMask ? (qkDim == 64U ? DN_MASK_D64_N : DN_MASK_OTHER_N) : DN_NO_MASK_N;
}
} // namespace FlashAttnPolicy
#endif
