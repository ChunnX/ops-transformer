/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_swiglu_tiling.h
 * \brief MatmulSwiglu TilingData 定义
 *
 * 按输出块分块, 每块 gate/up 两路 matmul 结果在 L0C->UB 驻留并就地做 SwiGLU,
 * 中间 [M, 2N] 不落 GM。mmTiling 按单边 [M, N, K] 生成, 仅需系统 workspace。
 */
#ifndef OPS_OP_HOST_MATMUL_SWIGLU_TILING_H_
#define OPS_OP_HOST_MATMUL_SWIGLU_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

constexpr uint64_t SGM_TILING_KEY_FP16 = 0;
constexpr uint64_t SGM_TILING_KEY_BF16 = 1;
constexpr uint64_t SGM_TILING_KEY_FP32 = 2;

BEGIN_TILING_DATA_DEF(MatmulSwigluTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, m);             // 行数 M
    TILING_DATA_FIELD_DEF(uint64_t, k);             // 收缩维 K
    TILING_DATA_FIELD_DEF(uint64_t, n);             // 单边宽度 N (= 输出列数)
    TILING_DATA_FIELD_DEF(uint64_t, twoN);          // 2N (= weight 的拼接列数 / bias 长度)
    TILING_DATA_FIELD_DEF(uint64_t, baseM);         // 输出块行高 (matmul baseM)
    TILING_DATA_FIELD_DEF(uint64_t, baseN);         // 输出块列宽 (matmul baseN)
    TILING_DATA_FIELD_DEF(uint64_t, mBlockCnt);     // M 方向块数 = ceil(M/baseM)
    TILING_DATA_FIELD_DEF(uint64_t, nBlockCnt);     // N 方向块数 = ceil(N/baseN)
    TILING_DATA_FIELD_DEF(uint64_t, totalBlockCnt); // = mBlockCnt * nBlockCnt
    TILING_DATA_FIELD_DEF(uint32_t, hasBias);       // 是否带 bias
    TILING_DATA_FIELD_DEF(uint32_t, transposeWeight);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);   // 参与计算的 cube 核数
    TILING_DATA_FIELD_DEF(uint32_t, tileN);         // 向量 SwiGLU 行内分块列数 (满足 UB 预算)
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTiling);  // 单边 [M, N, K] 的 cube tiling
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulSwiglu, MatmulSwigluTilingData)

struct MatmulSwigluCompileInfo {
    int64_t coreNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
};

}  // namespace optiling
#endif  // OPS_OP_HOST_MATMUL_SWIGLU_TILING_H_
