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
    int64_t batchSize = 0;
    int64_t seqLength = 0;
    int64_t c = 0;
    int64_t n = 0;
    int64_t c0 = 0;
    int64_t c1 = 0;
    int64_t cTail = 0;
    int64_t aivNum = 0;
    int64_t tileGradY = 0;
    int64_t tileHHat2 = 0;
    int64_t tileSize = 0;
    int64_t skIterCount = 0;
    int64_t ubSize = 0;
    float eps = 0;
    TCubeTiling mm1TilingData;
    TCubeTiling mm2TilingData;
};

struct MhcPreSinkhornBackwardArch35DeterminiticTilingData {
    // === shape: 算子输入张量维度 ===
    int64_t b = 0;                  // batch_size
    int64_t s = 0;                  // seq_length
    int64_t c = 0;                  // C 维度 (hidden_dim)
    int64_t n = 0;                  // N 维度 (head_dim, n ≤ 8)
    // === core allocation: 多核任务分配 ===
    int64_t usedCoreNum = 0;        // AIV 实际使用核数 (含 reset 后修正)
    int64_t aicCoreNum = 0;         // AIC 核数 (用于 MatMul 并行)
    // === UB factors: UB 单次处理元素数 (控制 UB buffer 分配粒度) ===
    int64_t ubFactorBS = 0;         // UB 单次处理 B*S 维元素数
    int64_t ubFactorC = 0;          // UB 单次处理 C 维元素数 (= c 时不切, < c 时切 C)
    int64_t skIterCount = 0;        // Sinkhorn-Knopp 迭代次数 (正向反向共用)
    // === block factors: 每核分配的 BS 工作量 (C 切分由 Kernel 内部处理，不在此处) ===
    int64_t blockFactorBS = 0;      // 普通核处理的 B*S 数 (≈ total_bs / usedCoreNum)
    int64_t blockTailBS = 0;        // 尾核处理的 B*S 数 (≤ blockFactorBS)
    // === 业务参数 ===
    float eps = 0.0f;               // 数值稳定常数 (hc_eps)
    // === MatMul Tiling ===
    TCubeTiling mm1TilingData;      // MatMul1: gradHcBeforeNorm[B*S, k_] × phi[k_, n*c]
    TCubeTiling mm2TilingData;      // MatMul2: gradHcBeforeNorm^T[k_, B*S] × xFp32[B*S, n*c]
};

#endif // MHC_PRE_SINKHORN_BACKWARD_DATA_ARCH35_H
