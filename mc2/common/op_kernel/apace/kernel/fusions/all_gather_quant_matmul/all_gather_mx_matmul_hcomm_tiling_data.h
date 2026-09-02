/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_mx_matmul_hcomm_tiling_data.h
 * \brief Tiling data for AllGather + QuantMatmul fusion kernel (Hcomm/CCU variant)
 *
 * 独立于 URMA 路径的 tiling 结构：Hcomm 路径为 AIC-only 纯 C 核通信（HCCL AllGather），
 * tiling 由 host 侧从算子 tiling 上下文填充。
 */

#pragma once

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "../../../tiling/quant_matmul_tiling_data.h"
#include "../../../tiling/comm_tiling_data.h"

// Hcomm/CCU 路径 tiling 结构体（apace 自有，不依赖外部算子 tiling）
namespace Apace {
struct hcommAllGatherMatmulTilingData {
    Mc2InitTiling mc2InitTiling;  // HCCL 初始化参数
    Mc2CcTiling mc2CcTiling;      // HCCL CC task 配置
    CommTilingData commTile;      // AllGather 切分参数（splitAxis=M_per_rank, nonSplitAxis=K）
    QuantMatmulTilingData mmTile; // Matmul tiling 参数
    uint64_t gatherLen;           // 0: gather_out 输出可用，通信写入 gatherOut; >0: 用 workspace
};
} // namespace Apace
