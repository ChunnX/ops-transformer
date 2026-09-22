/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_SANDWICH_NORM_POST_PREONLY_TILING_H_
#define MHC_SANDWICH_NORM_POST_PREONLY_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MhcSandwichNormPostPreonlyTilingData)
// 形状信息，totalTokens为展平后的B*S
TILING_DATA_FIELD_DEF(uint32_t, totalTokens); // M = B*S
TILING_DATA_FIELD_DEF(uint32_t, headCount);   // N固定为4
TILING_DATA_FIELD_DEF(uint32_t, embedDim);    // D按16个元素对齐

// 核映射，沿totalTokens均衡切分
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCoreSp);
TILING_DATA_FIELD_DEF(uint32_t, blockPivot);

// D轴分块
TILING_DATA_FIELD_DEF(uint32_t, dimTile);     // 单块元素数
TILING_DATA_FIELD_DEF(uint32_t, dimLoop);     // 分块数量
TILING_DATA_FIELD_DEF(uint32_t, lastDimTile); // 尾块元素数，0表示整除

// MHC_Pre参数，pre分支只使用N行
TILING_DATA_FIELD_DEF(uint32_t, phiRows); // N固定为4
TILING_DATA_FIELD_DEF(uint32_t, phiCols); // N*D

// 数值稳定项
TILING_DATA_FIELD_DEF(float, normEps);
TILING_DATA_FIELD_DEF(float, mhcPreNormEps);
TILING_DATA_FIELD_DEF(float, hcEps);

// 预先计算的浮点倒数，避免Kernel侧整数转浮点
TILING_DATA_FIELD_DEF(float, invD); // 1.0f / D
TILING_DATA_FIELD_DEF(float, invK); // 1.0f / (N*D)

// 是否在MHC_Post与MHC_Pre之间执行RMSNorm_mid
TILING_DATA_FIELD_DEF(uint32_t, useRmsMid); // 0表示关闭，1表示开启
// A5执行方案。bit2选择双sibling AIV模板，其他组合回退单AIV。
TILING_DATA_FIELD_DEF(uint32_t, coresPerToken); // 取值为1或2

// 是否返回h_in_f32
TILING_DATA_FIELD_DEF(int8_t, returnHInF32); // TILING_DATA_FIELD_DEF接口不支持bool
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MhcSandwichNormPostPreonly, MhcSandwichNormPostPreonlyTilingData)

struct MhcSandwichNormPostPreonlyCompileInfo {};

} // namespace optiling

#endif
