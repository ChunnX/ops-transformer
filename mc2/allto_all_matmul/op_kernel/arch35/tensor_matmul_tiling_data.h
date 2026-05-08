/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * 基于 tensor API 的矩阵乘 Tiling 数据结构
 * 简化策略: baseM=baseN=baseK=128 固定切分
 */

#ifndef TENSOR_MATMUL_TILING_DATA_H
#define TENSOR_MATMUL_TILING_DATA_H

#include <cstdint>

struct TensorMatmulTilingData {
    uint32_t tileM;             // 当前 tile 的 M 维度
    uint32_t fullK;             // 完整 K (= rankK × rankDim)
    uint32_t fullN;             // 完整 N (= rankN)
    uint32_t baseM;             // M 切分块大小
    uint32_t baseN;             // N 切分块大小
    uint32_t baseK;             // K 切分块大小
    uint32_t kL1;               // 单次 L1 K 步长
    uint32_t scaleKL1;          // scale 复用窗口（MX 使用）
    uint32_t mTailTile;         // M 尾块 tile 数
    uint32_t nTailTile;         // N 尾块 tile 数
    uint32_t mBaseTailSplitCnt; // M 尾块合并计数
    uint32_t nBaseTailSplitCnt; // N 尾块合并计数
    uint32_t mTailMain;         // M 尾块主尺寸
    uint32_t nTailMain;         // N 尾块主尺寸
    uint32_t dbL0c;             // L0C 双缓冲：1=关闭，2=开启
};

#endif // TENSOR_MATMUL_TILING_DATA_H
