/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * 基于 tensor API 的矩阵乘 Tiling 数据结构
 * 简化策略: baseM=baseN=baseK=128 固定切分
 */

#ifndef TENSOR_MATMUL_TILING_DATA_H
#define TENSOR_MATMUL_TILING_DATA_H

#include <cstdint>

struct TensorMatmulTilingData {
    uint32_t tileM;
    uint32_t fullK;
    uint32_t fullN;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t kL1;
    uint32_t scaleKL1;
    uint32_t mTailTile;
    uint32_t nTailTile;
    uint32_t mBaseTailSplitCnt;
    uint32_t nBaseTailSplitCnt;
    uint32_t mTailMain;
    uint32_t nTailMain;
    uint32_t dbL0c;
};

struct MxTensorMatmulTilingData {
    uint32_t tileM;
    uint32_t fullK;
    uint32_t fullN;
    uint32_t rankK;
    uint32_t rankDim;
    uint32_t rankKScaleGrp;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t kL1;
    uint32_t scaleKL1;
    uint32_t dbL0c;
};

#endif // TENSOR_MATMUL_TILING_DATA_H
