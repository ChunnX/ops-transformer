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
 * \file moe_tutel_combine_x_tiling_data.h
 * \brief Tiling data shared by the Ascend950 host tiling and kernel.
 */

#ifndef MOE_TUTEL_COMBINE_X_TILING_DATA_H_
#define MOE_TUTEL_COMBINE_X_TILING_DATA_H_

#include <cstdint>

struct MoeTutelCombineXTilingData {
    int64_t capacity;         // C = y_grad.dim(1)
    int64_t hidden;           // H = y_grad.dim(2)
    int64_t topK;             // K = gates.dim(0)
    int64_t samples;          // S = gates.dim(1)
    int64_t usedCoreNum;      // 实际派活的核数
    int64_t rowOfFormerBlock; // 非尾核各领多少个 token
    int64_t rowOfTailBlock;   // 尾核领多少个 token
    int64_t rowFactor;        // 内层一次处理多少个 token
    int64_t kFactor;          // 内层一次处理多少个 k
    int64_t hFactor;          // H 分块长度，等于 hidden 时为整行全载
    int64_t hChunkAlignT;     // CeilAlign(hFactor, 64)，覆盖 MicroAPI 完整 VL 读
    int64_t hChunkAlignF32;   // CeilAlign(hFactor, 64)，fp32 完整 VL
    int64_t routeStride;      // 路由片在 UB 里的行跨距 = CeilAlign(rowFactor, 16)
    // 新字段只追加在 key=10000 旧布局末尾，保持已有字段偏移不变。
    int64_t expertNum; // E = y_grad.dim(0)
};

#endif // MOE_TUTEL_COMBINE_X_TILING_DATA_H_
