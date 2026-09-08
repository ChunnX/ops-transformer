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
 * \file cla_gate_mx_quant_with_dual_axis_tilingdata.h
 * \brief Kernel-side TilingData plain struct for ClaGateMxQuantWithDualAxis
 */

#ifndef OPS_NN_CLA_GATE_MX_QUANT_WITH_DUAL_AXIS_TILINGDATA_H
#define OPS_NN_CLA_GATE_MX_QUANT_WITH_DUAL_AXIS_TILINGDATA_H

struct ClaGateMxQuantWithDualAxisTilingData {
    int64_t usedCoreNum;  // 实际使用核数（kernel 侧视为 totalCoreNum）
    int64_t dstType;      // 输出量化类型
    int64_t activateLeft; // 保留字段，当前固定 1
    int64_t dimM;         // 总行数 T
    int64_t dimN;         // 量化列数 K = N * D
    int64_t numGroups;    // 保留字段，当前固定 1
    int64_t blockW;       // 列方向基本块宽度=256
    int64_t splitBlockH;  // 行方向基本块高度=64
    int64_t dimNBlockNum; // K 方向切分基本块数 = ceil(K/blockW)
    int64_t dimNTail;     // K 方向尾块列数
    int64_t headNum;      // head 数量 N
    int64_t headDim;      // 每个 head 的 D（128 或 256）
    int64_t quantMode;    // 0: tail-axis single, 1: dual-axis
    int64_t rowTileNum;   // tail-axis single: 行方向 tile 数
    int64_t colTileNum;   // tail-axis single: 列方向 tile 数
    int64_t rowBlockSize; // tail-axis single: 正常行块行数
    int64_t rowTailSize;  // tail-axis single: 行尾块行数
    int64_t colBlockSize; // tail-axis single: 正常列块列数
    int64_t colTailSize;  // tail-axis single: 列尾块列数
    int64_t batchCount;   // tail-axis single: 每个核的最大 UB batch 数
    int64_t batchSize;    // tail-axis single: 每个 UB batch 包含的 256 列块数
};

#endif // OPS_NN_CLA_GATE_MX_QUANT_WITH_DUAL_AXIS_TILINGDATA_H
