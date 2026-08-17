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
 * \file moe_tutel_combine_gates_tiling_data.h
 * \brief Host 与 Kernel 共用的 TilingData 定义（arch35 / Ascend950）
 */
#ifndef MOE_TUTEL_COMBINE_GATES_TILING_DATA_H_
#define MOE_TUTEL_COMBINE_GATES_TILING_DATA_H_

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

// 轴命名：
//   S = x.shape[0]（token 数）      H = x.shape[1] = y_grad.shape[2]（被收缩的隐藏维）
//   E = y_grad.shape[0]（专家数，同时是路由轴长度）  C = y_grad.shape[1]（每专家 capacity）
//   indices / locations / gates_grad 形状恒为 [E, S]
//
// 分核：blockDim = sCoreNum * kCoreNum；blockIdx b -> (cs, ck) = (b % sCoreNum, b / sCoreNum)。
// s 外层、k 内层，使 x 的一行在本核的整个 k 区间内只搬运一次。
namespace optiling {
struct MoeTutelCombineGatesTilingData {
    uint64_t samples;      // S
    uint64_t hidden;       // H
    uint64_t experts;      // E
    uint64_t capacity;     // C
    uint64_t sCoreNum;     // s 方向核数（k 方向核数 = blockDim / sCoreNum，kernel 无需单独下发）
    uint64_t sPerCoreBase; // S / sCoreNum
    uint64_t sCoreRemain;  // S % sCoreNum，前 sCoreRemain 个 s 分片各多 1
    uint64_t kPerCoreBase; // E / kCoreNum
    uint64_t kCoreRemain;  // E % kCoreNum，前 kCoreRemain 个 k 分片各多 1
    uint64_t sBlock;       // 输出 tile 在 s 方向的宽度（元素数）
    uint64_t sBlockAlign;  // sBlock 按 VL(fp32) 向上对齐后的元素数（tile 行距；VF 整段读写要求）
    uint64_t kBlock;       // 输出 tile 在 k 方向的高度（行数），用于把 tile 的 UB 占用封顶
    uint64_t hFactor;      // 单次载入的 H 长度；NOT_CUT_H 时 == hidden
    uint64_t hFactorAlign; // hFactor 按 VL(fp32) 向上对齐后的元素数，即 UB 行长
    uint64_t numChunks;    // ceil(hidden / hFactor)；NOT_CUT_H 时 == 1（hidden == 0 时为 0）
    uint64_t hTail;        // 最后一块的实际长度；NOT_CUT_H 时 == hidden
};
} // namespace optiling

#endif // MOE_TUTEL_COMBINE_GATES_TILING_DATA_H_
