/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// fit_balance_tiling 公共类型与常量:
// 枚举(键控维度/标定模式)/数据结构(模型信息/切分结果/引擎入参)/常量。
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

// ============================================================================
// 零、公共类型与常量（两分支共享词汇: CommCutResult/FitBalanceTilingArgs/枚举等）
// ============================================================================
// ============================================================================

constexpr uint64_t BYTES_PER_KB = 1024;                        // 字节数单位词(替代旧 ONE_KBYTE 命名)
constexpr uint64_t BYTES_PER_MB = BYTES_PER_KB * BYTES_PER_KB; // 同上
constexpr uint64_t MIN_RANK_DIM = 2;

enum class TilingSocVersion {
    SOC950 = 4, // 值 4 为键约定: 查表 key 前缀 "{soc}_" 的 soc 段沿用旧枚举序(950=4), 勿改;
                // 引擎仅支持 950; 后续芯片在此扩展
};

enum class MatmulDtypeKind {
    FP16, // default
    QUANT,
    ANTI_QUANT,
};

struct MatmulBaseBlock {
    uint32_t baseM = 256; // 256: init value for baseM
    uint32_t baseN = 256; // 256: init value for baseN
    uint32_t baseK = 128; // 128: init value for baseK
};

// 计算域画像
// 计算域画像
struct MatmulShapeInfo {
    uint64_t mValue;
    uint64_t nValue;
    uint64_t kValue;
    uint64_t baseM; // matmul 最优基本块(对齐粒度/下限, 标定 baseBlock 注入)
    uint64_t baseN;
    uint64_t baseK;
    uint64_t inMatrixADtypeSize; // A 每元素字节数(L2 估算与通信胶水消费)
    uint64_t inMatrixBDtypeSize;
    uint64_t outMatrixCDtypeSize;
    uint64_t coreNum;        // 参与计算的 AIC 核数
    double computesPerCycle; // cube 有效算力(MACs/cycle, 标定查表产出)
    double cyclePerMicroSec; // 频率
};

// 通信域画像
struct CommModelInfo {
    uint64_t rankDim;                        // 通信域卡数(构造期钳位 ≥2)
    uint64_t commDtypeSize;                  // 通信每元素字节数
    uint64_t commMatrixLen;                  // 通信矩阵另一维(K 或 N, CommProfile.axis 决定)
    uint64_t commDtypeSizeExpansionFraction; // 子字节折算分母(fp4=2)
};

// 通信时间线性拟合参数(3 字段): 总量公式 y = k·x + N·b + c——
// k=带宽斜率, b=每轮启动开销(单轮全款), c=一次性固定开销(只进总量口径, 不进单轮/反解)。
// 阶段1 结构先行: 现有标定行填 b=旧截距、c=0 → 与旧 y=k·x+offset 逐位等价(bit 锁保持);
// 阶段2 真实 b/c 标定数据到位后激活行为变化(与开销补丁删除同批, 独立性能验证)。
struct CommFitParams {
    double slopeUsPerMb;    // 带宽斜率 k (us/MB)
    double roundOverheadUs; // 每轮启动开销 b (us)——单轮通信全款, 反解时从目标时间中扣除
    double fixedOverheadUs; // 一次性固定开销 c (us)——整个算子只付一次, 仅进总量口径
};

// 引擎切分结果(M 轴 1短+N长 的切分方案; 算子侧边界处转换为旧全局 CommCutResult)
struct CommCutResult {
    bool shortTileAtBack;
    uint64_t totalTileCnt;
    uint64_t numLongTile;
    uint64_t numShortTile;
    uint64_t longTileLen;
    uint64_t shortTileLen;
};

struct CutParams {
    uint64_t mAlignLen;
    uint64_t maxTileLen;
    uint64_t minTileLen; // threshold size, max(commMinLen, matmulMinLen)
    uint64_t maxTileCnt;
};

// 通信轮次硬上限（通信 API 的 FFTS 队列大小限制; 随 API 上限放宽 16→63）
constexpr uint64_t MAX_COMM_ROUNDS = 63;

constexpr uint32_t DEFAULT_BASE_M = 128;
constexpr uint32_t DEFAULT_BASE_N = 256;
constexpr uint32_t DEFAULT_BASE_K = 64;

// 引擎调度入参( 清理: 旧 33 字段镜像裁至引擎实际消费的 9 个, 按类型聚合;
// 算子侧 ToEngineArgs 从旧 mc2tiling::TilingArgs 挑字段填充)
// 引擎调度入参( 清理: 旧 33 字段镜像裁至引擎实际消费的 9 个, 平铺;
// 算子侧 ToEngineArgs 从旧 mc2tiling::TilingArgs 挑字段填充)
struct FitBalanceTilingArgs {
    uint64_t mValue;
    uint64_t nValue;
    uint64_t kValue;
    uint32_t rankDim;              // 通信域卡数
    uint32_t aicCoreNum;           // 计算 AIC 核数
    matmul_tiling::DataType aType; // mm 域 A dtype(量化类别判定)
    ge::DataType geAType;          // ge 域 A dtype(fp4 识别——DT_FLOAT4_E2M1 无 mm 域表达)
    uint64_t inputDtypeSize;       // A 每元素字节数(glue 真实口径, fp4 时=0)
    uint64_t outputDtypeSize;      // C 每元素字节数(C 侧承载通信时的字节)
};

// ====== 通信引擎枚举与解析（三引擎统一分派的公共词汇）======
enum class TilingCommEngine {
    AICPU,
    CCU,
    AIV_URMA,
};

// comm_mode 属性解析: "aiv_urma"→AIV_URMA, "ccu"→CCU, 其余(含空指针/默认"ai_cpu")→AICPU
inline TilingCommEngine ParseCommEngine(const char *commMode)
{
    if (commMode == nullptr) {
        return TilingCommEngine::AICPU;
    }
    if (strncmp(commMode, "aiv_urma", 8U) == 0) {
        return TilingCommEngine::AIV_URMA;
    }
    if (strncmp(commMode, "ccu", 3U) == 0) {
        return TilingCommEngine::CCU;
    }
    return TilingCommEngine::AICPU;
}

// ====== 通算顺序枚举(键控维度: 同一原语可有相反顺序, 如 matmul_allto_all=mm→A2A /
//       allto_all_matmul=A2A→mm; 顺序不可由原语推导, 必须显式传入) ======
enum class CommComputeOrder {
    COMPUTE_FIRST, // 计算在前通信在后(输出侧通信: RS/AR/matmul_allto_all)
    COMM_FIRST,    // 通信在前计算在后(输入侧通信: AG/allto_all_matmul)
};

// ====== 本地 mm 块位置枚举 ======
// 引擎不区分通信原语——原语语义(份数口径/轴向/流水线结构)全部终结在标定层 CommProfile;
// 引擎唯一的结构键控是"本地 mm 块在流水线的位置"(不经通信的数据块, 前按
// 通信原语隐式表达, 现显式化):
enum class LocalMMPosition {
    FRONT, // 本地 mm 块前置: 流水线首端(输入侧通信场景——本卡 A 块已在手, 聚合前即可计算,
           // 启动端被本地块免费占据, 短块退到尾部专职快速收尾)
    BACK, // 本地 mm 块后置: 流水线尾端(输出侧通信场景——本地保留份无需发出、无通信
          // deadline, 放尾部与末块通信重叠, 收尾端被本地块占据, 短块前置专职启动通信管道)
    NONE, // 无本地 mm 块——短块作流水线垫片, 按 bound 判定放首/尾
};

// ====== 标定层结构参数的模式枚举(CommProfile 消费; 倍数依赖运行时 rankDim, 标定层存模式,
//       引擎一处集中换算——避免按原语的散落取值表) ======
enum class CommAxis { // 通信承载的数据侧(决定通信矩阵另一维与 dtype 字节取值)
    K,                // A 侧(输入): 输入侧通信, 传 A 矩阵行(另一维取 kValue)
    N,                // C 侧(输出): 输出侧通信, 传 C 矩阵行(另一维取 nValue)
};
enum class RankMul { // m 的倍数模式(换算: ONE=1, RANK_DIM=rankDim, RANK_DIM_MINUS_ONE=rankDim-1)
    ONE,
    RANK_DIM,
    RANK_DIM_MINUS_ONE,
};
inline uint64_t RankMulValue(RankMul mode, uint64_t rankDim)
{
    switch (mode) {
        case RankMul::RANK_DIM:
            return rankDim;
        case RankMul::RANK_DIM_MINUS_ONE:
            return rankDim - 1U; // rankDim 已在模型入口钳位 ≥2, 无下溢
        default:
            return 1U;
    }
}

// fp4 子字节建模分母: 1 字节打包 2 个 fp4 元素(D6, dtype 键控)
