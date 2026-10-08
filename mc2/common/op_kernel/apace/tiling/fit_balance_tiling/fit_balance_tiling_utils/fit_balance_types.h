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
// 公共类型与常量(引擎共享词汇: 切分结果/调度入参/枚举等)
// ============================================================================
// ============================================================================

constexpr uint64_t BYTES_PER_KB = 1024;                        // 字节数单位词(替代旧 ONE_KBYTE 命名)
constexpr uint64_t BYTES_PER_MB = BYTES_PER_KB * BYTES_PER_KB; // 同上
constexpr uint64_t MIN_RANK_DIM = 2;

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
    double cyclePerMicroSec; // 频率(cycle/us, 标定注入)
    uint64_t l2CacheBytes;   // 芯片 L2 容量(L2 工作集判定阈值, 标定注入)
};

// 通信域画像
struct CommModelInfo {
    uint64_t rankDim;                        // 通信域卡数(构造期钳位 ≥2)
    uint64_t commDtypeSize;                  // 通信每元素字节数
    uint64_t commMatrixLen;                  // 通信矩阵另一维(K 或 N, dataflow.order 决定)
    uint64_t commDtypeSizeExpansionFraction; // 子字节折算分母(fp4=2)
};

// 通信时间线性拟合参数(3 字段): 总量公式 y = k·x + 轮数·b + c——
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

// 通信承载的数据侧(决定通信矩阵另一维与 dtype 字节取值)
enum class CommAxis {
    K, // A 侧(输入): 输入侧通信, 传 A 矩阵行(另一维取 kValue)
    N, // C 侧(输出): 输出侧通信, 传 C 矩阵行(另一维取 nValue)
};

// 引擎调度入参(纯运行时 shape 事实: 三围/并行度/dtype/通信引擎)
enum class TilingCommEngine {
    AICPU,
    CCU,
    AIV_URMA,
};

inline TilingCommEngine ParseCommEngine(const char* commMode)
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
    TilingCommEngine commEngine;   // 通信引擎(op 侧 ParseCommEngine(comm_mode) 解析后传入——决定
                                   // 拟合表分区与 CCU 的单轮 256MB 整形, 每次调用可变)
};

// ====== 通信引擎枚举与解析（三引擎统一分派的公共词汇）======

// comm_mode 属性解析: "aiv_urma"→AIV_URMA, "ccu"→CCU, 其余(含空指针/默认"ai_cpu")→AICPU

// ====== 通算顺序枚举(键控维度: 同一原语可有相反顺序, 如 matmul_allto_all=mm→A2A /
//       allto_all_matmul=A2A→mm; 顺序不可由原语推导, 必须显式传入) ======
enum class CommComputeOrder {
    COMPUTE_FIRST, // 计算在前通信在后(输出侧通信: RS/AR/matmul_allto_all)
    COMM_FIRST,    // 通信在前计算在后(输入侧通信: AG/allto_all_matmul)
};

// ====== 本地 mm 块位置枚举 ======
// 引擎不区分通信原语——原语语义(份数口径/轴向/流水线结构)由调用侧经 args 结构参数声明;
// 引擎唯一的结构键控是"本地 mm 块在流水线的位置"(不经通信的数据块, 前按
// 通信原语隐式表达, 现显式化):
enum class LocalMMPosition {
    FRONT, // 本地 mm 块前置: 流水线首端(输入侧通信场景——本卡 A 块已在手, 聚合前即可计算,
           // 启动端被本地块免费占据, 短块退到尾部专职快速收尾)
    BACK, // 本地 mm 块后置: 流水线尾端(输出侧通信场景——本地保留份无需发出、无通信
          // deadline, 放尾部与末块通信重叠, 收尾端被本地块占据, 短块前置专职启动通信管道)
    NONE, // 无本地 mm 块——短块作流水线垫片, 按 bound 判定放首/尾
};

// fp4 子字节建模分母: 1 字节打包 2 个 fp4 元素(D6, dtype 键控)

// ============================================================================
// 单一大结构体替代"引擎侧一堆小结构体 + 算子侧一堆具名 constexpr"）
//
// 用法（算子侧标定头, 填一个实例传给 InitCtx 即可）:
//   static const FitBalanceCalibration CALIBRATION = {
//       /* dataflow */ { CommComputeOrder::..., LocalMMPosition::..., bool },
//       /* commRows */ { {TilingCommEngine::..., 卡数, k, b, c}, ... },
//       /* mmFit */    { /* cubeCalc */ { {MatmulDtypeKind, isFp4, aB, bB, 算力} }, ... },
//   };
//
// 填写规则:
//   - 通信行按 (通信引擎, rankDim) 匹配; 未标定形态不写行(留空=miss 走其他切分策略);
//   - matmul 行按 (calcType, A/B 字节数) 匹配, 未填字段取 NSDMI 默认; 行按前缀填, 0 值行=未填;
//   - 通信拟合公式 y = k·x + 轮数·b + c(轮数为通信切分块数, 与 shape 的 N 维无关): 客户标定产出 k/b/c 三个数。
// ============================================================================

struct FitBalanceCalibration {
    static constexpr size_t MAX_COMM_FIT_ROWS = 8;   // 已标定通信形态数上限(多引擎共用一表:
                                                     // 2 引擎 × 2 卡档 + 余量)
    static constexpr size_t MAX_CUBE_CALC_ROWS = 4;  // matmul 算力档位数上限
    static constexpr size_t MAX_BASE_BLOCK_ROWS = 2; // 最优基本块档位数上限

    // —— 数据流形态(算子接入时定死的结构声明; 引擎据此推导 axis 与三个倍数,
    //    推导规则见 InitCtx) ——
    struct DataflowShape {
        CommComputeOrder order = CommComputeOrder::COMM_FIRST; // 通算顺序(并决定通信承载侧:
                                                               // 通信在前=传输入 A 侧/K 轴, 在后=传输出 C 侧/N 轴)
        LocalMMPosition localMMPosition = LocalMMPosition::NONE; // 本地 mm 块位置(决定稳态工作集
                                                                 // 是否剔出本地份: FRONT=rankDim−1)
        bool fullVolume = true; // 全量形态: 切分域为本卡份、实算 mm 总行数与单次通信字节均为
                                // 切分域×rankDim(聚合/分发类); false=单份形态(本卡一份, 两者×1)
    };
    DataflowShape dataflow{};

    // —— 通信拟合（行自带通信引擎维度——新增引擎不改结构体, 加行即可）——
    struct CommFitRow {
        TilingCommEngine commEngine;  // 通信引擎(与运行时 args.commEngine 匹配)
        uint32_t rankDim;             // 卡数(4=标卡4P, 8=8P, ...; 0=未填行哨兵)
        double slopeUsPerMb;          // 带宽斜率 k (us/MB)
        double roundOverheadUs;       // 每轮启动开销 b (us)
        double fixedOverheadUs = 0.0; // 一次性固定开销 c (us)——未标定填 0(兼容旧截距口径)
    };
    CommFitRow commRows[MAX_COMM_FIT_ROWS] = {}; // 已标定行(行按前缀填), rankDim=0 表示未填(哨兵截断)

    // —— matmul 性能拟合（对仗 CommFitRow; 字段序: 用户常改字段在前, 平台默认在后——聚合初始化
    //    可省尾部, 省略的尾部字段自动取 NSDMI 默认值）——
    struct MatmulFit {
        // 键维度裁定: 不含 C 侧字节数——glue 恒 cType=aType,
        // 运行时 cBytes 恒等于 aBytes(零信息量冗余维度, 假设/真实口径曾在其上分叉致算力查表缺陷);
        // 物理上 C dtype 只影响写出带宽不影响 MACs/cycle(老表 _1/_2 行同值 8192 佐证)。
        // 未来 outputDtypeSize 口径独立化(真按输出计)且重标定发现 C 影响算力时, 再按独立 CR 加回。
        struct CubeCalc {
            MatmulDtypeKind calcType;
            bool isFp4;              // FP16 / QUANT / ANTI_QUANT
            uint64_t aBytes;         // 输入 A 每元素字节数
            uint64_t bBytes;         // 输入 B 每元素字节数
            double computesPerCycle; // 有效算力 (MACs/cycle, 含 cube 利用率)
        };
        CubeCalc cubeCalc[MAX_CUBE_CALC_ROWS] = {};
        MatmulBaseBlock baseBlock = {128, 256, 64}; // matmul 最优基本块(单值)
        double defaultComputesPerCycle = 4096.0;    // 算力档回退(通常勿动)
        // —— 平台规格常量(默认 950; 新芯片接入按规格填写, 见 README 标定指南) ——
        double cyclePerMicroSec = 1.65 * BYTES_PER_KB; // AIC 频率(cycle/us)
        uint64_t l2CacheBytes = 128UL * BYTES_PER_MB;  // L2 缓存容量
    };
    MatmulFit mmFit;
};
