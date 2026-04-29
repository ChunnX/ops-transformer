/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * 基于 tensor API 的矩阵乘 Tiling 数据结构
 * 简化策略: baseM=baseN=baseK=128 固定切分
 */

#ifndef TENSOR_MATMUL_TILING_DATA_H
#define TENSOR_MATMUL_TILING_DATA_H

#include <cstdint>

struct TensorMatmulTilingData {
    uint32_t tileM;   // 当前 tile 的 M 维度（tileM 或 tailM）
    uint32_t fullK;   // 完整 K 维度 (= rankK * rankDim)
    uint32_t fullN;   // 完整 N 维度 (= rankN)
};

#endif // TENSOR_MATMUL_TILING_DATA_H
