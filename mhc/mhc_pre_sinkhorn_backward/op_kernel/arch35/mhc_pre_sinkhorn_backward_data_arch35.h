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
 * \file mhc_pre_sinkhorn_backward_data_arch35.h
 * \brief tiling data struct
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_DATA_ARCH35_H
#define MHC_PRE_SINKHORN_BACKWARD_DATA_ARCH35_H

#include <cstdint>

struct MhcPreSinkhornBackwardArch35TilingData {
    int64_t batchSize = 0;        // Batch维度大小
    int64_t seqLength = 0;        // Sequence维度大小
    int64_t c = 0;                // C维度大小（hidden_size相关）
    int64_t n = 0;                // N维度大小（attention相关）
    int64_t c0 = 0;               // C维度切分参数（c0部分）
    int64_t c1 = 0;               // C维度切分参数（c1部分）
    int64_t cTail = 0;            // C维度尾块大小
    int64_t aivNum = 0;           // AIV核数
    int64_t tileGradY = 0;        // grad_y切分tile大小
    int64_t tileHHat2 = 0;        // h_hat2切分tile大小
    int64_t tileSize = 0;         // tile大小
    int64_t skIterCount = 0;      // Sinkhorn迭代次数
    int64_t ubSize = 0;           // UB大小
    float eps = 0;                // eps精度参数
    TCubeTiling mm1TilingData;    // Matmul1 tiling数据
    TCubeTiling mm2TilingData;    // Matmul2 tiling数据
};

struct MhcPreSinkhornBackwardArch35DeterminiticTilingData {
    int64_t batchSize = 0;        // Batch维度大小
    int64_t seqLength = 0;        // Sequence维度大小
    int64_t c = 0;                // C维度大小（hidden_size相关）
    int64_t n = 0;                // N维度大小（attention相关，k=n²+2n）
    int64_t aivNum = 0;           // AIV核数
    int64_t cLoopDataLen = 0;     // C维度循环切分长度（一次处理c的元素数）
    int64_t bsLoopDataLen = 0;    // Batch*Seq循环切分长度（一次处理bs的元素数）
    int64_t skIterCount = 0;      // Sinkhorn迭代次数
    int64_t bsTaskCount = 0;      // 每个核处理的bs任务数（主核）
    int64_t tailBsTaskCount = 0;  // 尾核处理的bs任务数
    float eps = 0;                // eps精度参数
    TCubeTiling mm1TilingData;    // Matmul1 tiling数据
    TCubeTiling mm2TilingData;    // Matmul2 tiling数据
};

#endif // MHC_PRE_SINKHORN_BACKWARD_DATA_ARCH35_H
