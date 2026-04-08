/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the CANN Open Software License Agreement Version 2.0.
 */

// ============================================================================
// MTE AllToAll 算子 - Tiling 数据定义
// ============================================================================
//
// Tiling 数据分三段：
//   mc2InitTiling          — MC2 通信初始化参数（由 Mc2CcTilingConfig 自动填充）
//   mc2CcTiling            — MC2 通信通道参数（由 Mc2CcTilingConfig 自动填充）
//   mteAllToAllTilingInfo  — AllToAll 业务参数（需手动填充）

#ifndef MTE_ALLTOALL_TILING_DATA_H
#define MTE_ALLTOALL_TILING_DATA_H

#include <kernel_tiling/kernel_tiling.h>

// AllToAll 业务 Tiling 参数
struct MteAllToAllTilingInfo {
    uint64_t sendCount;   // 每个 rank 发送/接收的数据块大小（int8_t 元素数）
    uint64_t rankDim;     // rank 总数（卡数）
    uint64_t aivNum;      // 使用的 AIV 核数
};

// MC2 Tiling 数据结构（三段式）
struct MteAllToAllTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MteAllToAllTilingInfo mteAllToAllTilingInfo;
};

#endif
