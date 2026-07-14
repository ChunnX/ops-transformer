/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/*!
* \file engram_fetch_train_tiling_data.h
* \brief kernel侧tiling data结构 — EngramFetchTrain（训练态前向）
*        commBuffer[] 存 a2a GM 地址（各 rank）
*        localStorageAddr 为 host-pinned 设备映射地址（gather 源）
*/

#ifndef ASCENDC_ENGRAM_FETCH_TRAIN_TILING_H
#define ASCENDC_ENGRAM_FETCH_TRAIN_TILING_H

#include "kernel_tiling/kernel_tiling.h"

struct EngramFetchTrainTilingData {
    int64_t numTokens;             // indices 长度
    int32_t numEntriesPerRank;     // 每 rank embedding 行数
    int64_t hiddenDim;             // hidden 维度
    int64_t hiddenBytes;           // hiddenDim * dtypeSize
    uint32_t aivNum;               // AIV 核数
    uint64_t ubSize;               // UB 空间
    uint32_t rankSize;             // 通信域 rank 数（从 sendCountsOut dim0 获取）
    int64_t numMaxTokensPerRank;   // 每 rank 最大 token 数（workspace 计算用）
    int64_t totalRecv;             // a2a 接收上界 = numMaxTokensPerRank × rankSize
    int64_t localStorageAddr;    // host-pinned 设备映射地址（gather 源, 即 deviceBufPtr_）
};
#endif
