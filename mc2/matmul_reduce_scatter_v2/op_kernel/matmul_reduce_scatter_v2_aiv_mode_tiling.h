/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_reduce_scatter_v2_aiv_mode_tiling.h
 * \brief MatmulReduceScatterV2算子AIV模式的Tiling数据结构定义
 *
 * 本文件定义了AIV模式所需的Tiling参数，包括：
 *   - DequantType: 反量化类型枚举（PER_CHANNEL / PER_TOKEN）
 *   - MatmulReduceScatterV2AivModeInfo: 算子形状和特性信息（M/K/N、量化标志、对齐标志等）
 *   - CoCTiling: 分块策略参数（m0/n0/k0、Swizzle参数、通信切分策略等）
 *   - MatmulReduceScatterV2AivModeTilingData: 完整的Tiling数据结构（包含MC2通信Tiling）
 *   - MatmulReduceScatterV2AivModeTilingValue: Tiling键值，用于Tiling策略选择
 *   - 默认常量：各参数的默认值
 *   - Tiling Code Mask: Tiling键值的位掩码定义
 */
#ifndef __MATMUL_REDUCE_SCATTER_V2_AIV_MODE_TILING_H__
#define __MATMUL_REDUCE_SCATTER_V2_AIV_MODE_TILING_H__
#include <vector>
#include <map>
#include "kernel_tiling/kernel_tiling.h"
namespace matmulReduceScatterV2_aivmode_tiling{

enum class DequantType : int {
    DEQUANT_TYPE_UNDEFINED = -1,    // 未定义反量化类型
    PER_CHANNEL = 0,                // 按通道反量化（per-channel dequant）：每个输出列一个scale
    PER_TOKEN = 1,                  // 按token反量化（per-token dequant）：每个输入行+每个输出列各一个scale
    DEQUANT_TYPE_MAX = 2,
};

struct MatmulReduceScatterV2AivModeInfo {
    uint32_t M;                     // 矩阵乘法的M维度（token数量/行数）
    uint32_t K;                     // 矩阵乘法的K维度（隐藏层大小/内积维度）
    uint32_t N;                     // 矩阵乘法的N维度（权重列数/输出维度）
    uint32_t aivNum;                // AIV核心总数，用于多核并行
    uint32_t totalUbSize;           // 每个AIV核心的UB（Unified Buffer）总大小
    bool isTransposeA;              // A矩阵是否转置（当前版本仅支持false）
    bool isTransposeB;              // B矩阵是否转置
    uint64_t aAlignSize;            // A矩阵对齐后所需的workspace大小（字节）
    uint64_t bAlignSize;            // B矩阵对齐后所需的workspace大小（字节）
    uint64_t dequantSize;           // 反量化中间结果所需的workspace大小（字节）
    bool hasAAlign;                 // A矩阵是否需要对齐padding
    bool hasBAlign;                 // B矩阵是否需要对齐padding
    bool quantFlag;                 // 是否为量化场景（A和B均为int8时为true）
    bool is910C;                    // 是否为Ascend910_93芯片（910C）
    bool isX2ScaleTypeInt64;        // x2Scale的数据类型是否为int64（仅FP16输出时支持）
    DequantType dequant_type;       // 反量化类型：PER_CHANNEL或PER_TOKEN
};

struct CoCTiling {
    int32_t m0 = -1;                // M维度的L1分块大小（128或256）
    int32_t k0 = -1;                // K维度的L1分块大小（256或512）
    int32_t n0 = -1;                // N维度的L1分块大小（128或256）
    int32_t mLoop = -1;             // M维度的分块循环次数 = Ceil(M, m0)
    int32_t kLoop = -1;             // K维度的分块循环次数 = Ceil(K, k0)
    int32_t nLoop = -1;             // N维度的分块循环次数 = Ceil(N, n0)
    int32_t swizzlCount = -1;       // Swizzle块数量，用于优化L1缓存访问模式
    int32_t swizzlDirect = -1;      // Swizzle方向：0=Zn（按行），1=Nz（按列）
    int32_t pValue = -1;            // 每次通信前AIC计算的块数，控制计算与通信的重叠粒度
    int32_t ubMoveNum = -1;         // UB单次搬运的数据量（以512B为单位）
    int32_t commNpuSplit = -1;      // 通信时NPU维度的切分策略
    int32_t commDataSplit = -1;     // 通信时数据维度的切分策略
    int32_t lenPerLoop = -1;        // 每次循环处理的数据长度
    int32_t commDirect = -1;        // 通信方向：0=正向，1=反向
};

struct MatmulReduceScatterV2AivModeTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MatmulReduceScatterV2AivModeInfo matmulReduceScatterV2AivModeInfo;
    CoCTiling cocTiling;
};

struct MatmulReduceScatterV2AivModeTilingValue {
    int32_t value = -1;
    std::map<int, std::vector<std::vector<int>>> conditionMap = {};
    explicit MatmulReduceScatterV2AivModeTilingValue(int32_t v = -1,
        std::map<int, std::vector<std::vector<int>>> m = {})
        : value(v), conditionMap(std::move(m)) {}
};

constexpr int32_t REDUCESCATTER_M0_DEFAULT = 128;
constexpr int32_t REDUCESCATTER_UBMOVENUM_DEFAULT = 20;
constexpr int32_t REDUCESCATTER_COMMDATASPLIT_DEFAULT = 16;
constexpr int32_t REDUCESCATTER_PVALUE_DEFAULT = 12;
constexpr int32_t RANKSIZE_FOUR = 4;
constexpr int32_t RANKSIZE_EIGHT = 8;
constexpr int32_t SWIZZLE_DIRECT_ONE = 1;
constexpr int32_t SWIZZLE_COUNT_FOUR = 4;
constexpr int32_t COMM_DATA_DIRECT = 0;
constexpr int32_t COMMNPUSPLIT_ONE = 1;
constexpr int32_t HALF_KBYTE = 512;
constexpr int32_t DEFAULT_ROW = 128;
constexpr int32_t DEFAULT_COL = 256;
// Tiling Code Mask
constexpr int32_t COMMDATASPLIT_MASK = 0b11111;
constexpr int32_t COMMDATASPLIT_BNUM = 5;
constexpr int32_t COMMNPUSPLIT_MASK = 0b11111;
constexpr int32_t COMMNPUSPLIT_BNUM = 5;
constexpr int32_t COMMDIRECT_MASK = 0b1;
constexpr int32_t COMMDIRECT_BNUM = 1;
constexpr int32_t UBMOVENUM_MASK = 0b11111111;
constexpr int32_t UBMOVENUM_BNUM = 8;
constexpr int32_t PVALUE_MASK = 0b1111;
constexpr int32_t PVALUE_BNUM = 4;
constexpr int32_t SWIZZLCOUNT_MASK = 0b1111;
constexpr int32_t SWIZZLCOUNT_BNUM = 4;
constexpr int32_t SWIZZLDIRECT_MASK = 0b1;
constexpr int32_t SWIZZLDIRECT_BNUM = 1;
constexpr int32_t M0_MASK = 0b1;
constexpr int32_t M0_BNUM = 1;

constexpr int32_t CONDITION_M_ST = 0;
constexpr int32_t CONDITION_M_END = 1;
constexpr int32_t CONDITION_K_ST = 2;
constexpr int32_t CONDITION_K_END = 3;
constexpr int32_t CONDITION_N_ST = 4;
constexpr int32_t CONDITION_N_END = 5;

constexpr int32_t MATMUL_REDUCESCATTER_A2_FOUR_RANK_INT8_CODE_DEFAULT = 195043376;
constexpr int32_t MATMUL_REDUCESCATTER_A2_FOUR_RANK_FP16_CODE_DEFAULT = 193992752;
constexpr int32_t MATMUL_REDUCESCATTER_A2_EIGHT_RANK_INT8_CODE_DEFAULT = 468734000;
constexpr int32_t MATMUL_REDUCESCATTER_A2_EIGHT_RANK_FP16_CODE_DEFAULT = 463487024;
constexpr int32_t MATMUL_REDUCESCATTER_A3_EIGHT_RANK_INT8_CODE_DEFAULT = 467726384;
constexpr int32_t MATMUL_REDUCESCATTER_A3_EIGHT_RANK_FP16_CODE_DEFAULT = 468754480;
constexpr int32_t MATMUL_REDUCESCATTER_A3_FOUR_RANK_INT8_CODE_DEFAULT = 467726384;
constexpr int32_t MATMUL_REDUCESCATTER_A3_FOUR_RANK_FP16_CODE_DEFAULT = 200339504;
} // namespace
#endif
