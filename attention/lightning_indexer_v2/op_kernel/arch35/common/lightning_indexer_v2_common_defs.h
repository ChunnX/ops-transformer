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
 * \file lightning_indexer_v2_common_defs.h
 * \brief Common definitions shared by LightningIndexerV2 and QuantLightningIndexerV2 on arch35.
 */
#ifndef LIGHTNING_INDEXER_V2_COMMON_DEFS_H
#define LIGHTNING_INDEXER_V2_COMMON_DEFS_H

#include "kernel_operator.h"

namespace LIV2Common {

template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}

} // namespace LIV2Common

// bank冲突优化
// david 256KB bank layout
// shape  (             bank_depth  (            banks  bank_groups  block))  (512  (  2   8  32))
// stride (banks*bank_groups*block  (bank_groups*block        block      1))  (512  (256  32   1))
#define UB_BLOCK 32 // 32B
#define UB_BANK_GROUPS 8
#define UB_BANKS 2
#define UB_BANK_DEPTH 512

#define UB_BANK_GROUP_STRIDE UB_BLOCK                               // 32B
#define UB_BANK_STRIDE (UB_BANK_GROUPS * UB_BLOCK)                  // 256B
#define UB_BANK_DEPTH_STRIDE (UB_BANKS * UB_BANK_GROUPS * UB_BLOCK) // 512B

#endif // LIGHTNING_INDEXER_V2_COMMON_DEFS_H
