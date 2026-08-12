/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPARSE_ATTENTION_SCORE_PREFILL_TILINGKEY_H
#define SPARSE_ATTENTION_SCORE_PREFILL_TILINGKEY_H

#include "kernel_tiling/kernel_tiling.h"

/*
 * 本头文件定义 tiling key 宏常量。tiling key 用于在编译期通过 TILING_KEY_VAR
 * 选择不同的 kernel 代码路径：host 下发的 tilingKey 值与下列宏匹配后，
 * kernel 侧据此进入对应的精度/形状分支实现。
 */

// base key，所有有效 key 均 >= 该值
#define SASA_PREFILL_BASE_TILING 20000

// bf16 输入，D=128，fp32 O_partial 路径（innerPrecise!=1）
#define SASA_PREFILL_BF16_D128_TILING 20001

// bf16 输入，D=128，bf16 O_partial 路径
// innerPrecise==1: O_partial stored as bf16 (PV fixpipe F322BF16 + Phase2 regbase cast).
#define SASA_PREFILL_BF16_D128_INNER_LOW_TILING 20002

#endif  // SPARSE_ATTENTION_SCORE_PREFILL_TILINGKEY_H
