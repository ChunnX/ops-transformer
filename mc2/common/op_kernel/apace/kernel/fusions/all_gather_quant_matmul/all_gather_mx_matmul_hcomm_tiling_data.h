/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use it except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_mx_matmul_hcomm_tiling_data.h
 * \brief Tiling data for AllGather + QuantMatmul fusion kernel (Hcomm/CCU variant)
 *
 * 与 hcommAllToAllMatmulTilingData 的差异：
 *   - 无 localMatmul 字段：FragmentTensor kernel 的 HEAD 区域天然是 local-first 语义
 * commTile 语义：M 轴每卡 tile 切分（splitAxisTileSize=tileM, splitAxisTileCnt=tileCnt,
 *                 splitAxisTailSize=tailM, splitAxisTailCnt=tailCnt, nonSplitAxisSize=K）
 * scale 通信参数（scaleKLen = CeilDiv(K,64)*2）由 impl 从 mmTile.k 推导，无需独立 tiling 字段。
 */

#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "apace/tiling/quant_matmul_tiling_data.h"
#include "apace/tiling/comm_tiling_data.h"

struct hcommAllGatherMatmulTilingData {
    Mc2InitTiling mc2InitTiling;       // HCCL 初始化参数
    Mc2CcTiling mc2CcTiling;           // HCCL CC task 配置
    CommTilingData commTile;           // AllGather 切分参数（splitAxis=M_per_rank, nonSplitAxis=K）
    QuantMatmulTilingData mmTile;      // Matmul tiling 参数
};
