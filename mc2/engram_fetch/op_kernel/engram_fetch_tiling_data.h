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
* \file engram_fetch_tiling_data.h
* \brief kernel侧tiling data结构
*        withGrad 字段由 tiling 函数根据可选输出是否非空设置，kernel 据此选推理/训练分支
*/

#ifndef ASCENDC_ENGRAM_FETCH_TILING_H
#define ASCENDC_ENGRAM_FETCH_TILING_H

#include "kernel_tiling/kernel_tiling.h"

struct EngramFetchTilingData {
    int64_t numTokens;            // indices 长度
    int32_t numEntriesPerRank;    // 每 rank embedding 行数
    int64_t hiddenDim;            // hidden 维度
    int64_t hiddenBytes;          // hiddenDim * dtypeSize
    uint32_t aivNum;              // AIV 核数
    uint64_t ubSize;              // UB 空间
    int32_t withGrad;             // 0=推理(RDMA read), 1=训练(all-to-all); 由可选输出是否非空决定
    int64_t numMaxTokensPerRank;  // 每 rank 最大 token 数（训练态 workspace 计算用）
    uint32_t rankSize;            // 通信域 rank 数（训练态 workspace 计算用）
};
#endif