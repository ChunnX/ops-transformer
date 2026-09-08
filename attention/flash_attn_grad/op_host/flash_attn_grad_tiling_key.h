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
 * \file flash_attn_grad_tiling_key.h
 * \brief L4 策略层：tilingkey 位域拼装的唯一出口。
 *
 * 位域定义与合法组合的真源在 pypto 侧的 FlashAttnGradTilingKey / is_valid()，
 * host 只能产出 is_valid 为真的组合，否则运行期找不到二进制。
 */

#ifndef FLASH_ATTN_GRAD_TILING_KEY_H_
#define FLASH_ATTN_GRAD_TILING_KEY_H_

#include <cstdint>

#include "flash_attn_grad_tiling_info.h"

namespace optiling {

// TilingKey 位域：
//   bit[1:0] template   0=BN2GS1S2, 1=BN2, 2=未用, 3=BN2S2(保留，勿复用)
//   bit2     layout     0=非TND, 1=TND（BNSD 不占位，运行期靠视图参数区分）
//   bit3     has_atten_mask
//   bit4     swizzle          仅 template=0 有效
//   bit[6:5] d_align    0=64, 1=128, 2=192
//   bit7     dv_align   0=128, 1=192
//   bit8     is_bn2_multiblk  仅 template=1 有效
uint64_t EncodeTilingKey(const FagParsedInfo &info, const FagKernelPlan &kernel, const FagSchedulePlan &schedule);

} // namespace optiling

#endif // FLASH_ATTN_GRAD_TILING_KEY_H_
