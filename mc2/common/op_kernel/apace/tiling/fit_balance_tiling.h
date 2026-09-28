/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fit_balance_tiling.h
 * \brief fit-balance tiling 家族唯一公共文件（三层结构, 两分支共居, 迁移方案 §2.2/§4.9）
 *
 * 当前仅含自由长度分支: HCCL AICPU/CCU 引擎, 每轮开销大 → 反解最优轮长压轮数。
 * 消费者为旧通算融合算子的 fit-balance 路径（950 标卡 4P/8P; 引擎仅支持 950, 后续芯片扩展时更新此注释）。
 *
 * 历史存档: URMA wave-packing 拓展分支（v2 aiv_urma 的 estTime k 搜索方案）曾在此文件,
 * 2026-09-15 评审裁定废弃——与核利用率优先切分大部分结果相同, 少量差异定位为
 * 核利用率优先的实现问题(修复走独立事项)。相关标定方法论与实测数据仍有效,
 * 存档于 tiling-plan/ALLTO_ALL_MATMUL_V2_FIT_BALANCE_*。
 *
 * 布局: 零、公共类型与常量(含客户标定大结构体 FitBalanceCalibration) /
 *       一、模型层(HCCL 通信 + matmul 算力梯度; 内部查表结构勿在算子侧引用) /
 *       二、切分与输出层(CutPlanner + CommCutResult 适配) /
 *       三、求解层(自由长度求解器, ctx+键控分支) / 四、公共逻辑片段
 *
 * 零特有代码架构(§4.24/§4.38 演进): 钩子机制已废除; 引擎不区分通信原语——原语语义
 * (通信份数口径已融进标定斜率 k / 轴向与 mm 倍数在 CommProfile / 流水线结构由 LocalMMPosition
 * 表达)全部终结在标定层; 键控维度仅 (LocalMMPosition 本地块位置, CommComputeOrder 通算顺序,
 * TilingCommEngine 通信引擎, dtype)。算子侧 = CALIBRATION 实例 + 枚举入参 + 一行调用。
 *
 * 统一引擎分派（三引擎接入结构完全一致, 见 TilingCommEngine/ParseCommEngine）:
 *   同时支持多通信模式的算子按同一模式接入——CCU/AICPU 走自由长度分支（各引擎拟合表）;
 *   AIV_URMA 当前无求解分支(原 wave-packing 已废弃), 枚举值保留供未来方案接入;
 *   每引擎一组标定常量 + 一张表 + 一个分派点, 无任何引擎特殊路径。
 *
 * 等价性基线: 自由长度分支 2026-09-14 P0 提取经 162 用例 bit 级验证
 * (与老类逐字段比对), 消费方 agmm_v2 双包构建通过。
 */
#ifndef __APACE_FIT_BALANCE_TILING_H__
#define __APACE_FIT_BALANCE_TILING_H__

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

// ge::DataType 经 register/tilingdata_base.h 与 tiling/tiling_api.h 的既有链路提供
// (不直接 include graph 头: 部分 TU 的 ge 链路来自不同 toolkit 路径, 直接引入会双重定义)
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
// 引擎自包含(2026-09-17 裁定): 不依赖旧 fit-balance/公式化 tiling 头——所需类型/常量
// 在本文件重新实现并改用贴合本项目的名字(与旧全局词汇名不同, 同 TU 共存无冲突)。

// ============================================================================
// 零、公共类型与常量（两分支共享词汇: CommCutResult/FitBalanceTilingArgs/枚举等）
// ============================================================================
// ============================================================================

constexpr uint64_t BYTES_PER_KB = 1024;                        // 字节数单位词(替代旧 ONE_KBYTE 命名)
constexpr uint64_t BYTES_PER_MB = BYTES_PER_KB * BYTES_PER_KB; // 同上
constexpr uint64_t MIN_RANK_DIM = 2;

enum class TilingSocVersion {
    SOC950 = 4, // 值 4 为键约定: 查表 key 前缀 "{soc}_" 的 soc 段沿用旧枚举序(950=4), 勿改;
                // 引擎仅支持 950(旧芯片枚举值随旧芯片逻辑剥离, B 家族原位副本保留完整值域; 后续芯片在此扩展)
};

enum class MatmulDtypeKind {
    FP16, // default
    QUANT,
    ANTI_QUANT,
};

struct MatmulShapeInfo {
    TilingSocVersion socType;
    uint64_t coreNum;
    uint64_t inMatrixADtypeSize;
    uint64_t inMatrixBDtypeSize;
    uint64_t outMatrixCDtypeSize;
    uint64_t batchSize; // E / Ep
    uint64_t mValue;
    uint64_t nValue;
    uint64_t kValue;
    uint64_t baseM;
    uint64_t baseN;
    uint64_t baseK;
    double computesPerCycle; // cube算力
    double cyclePerMicroSec; // frequency
};

enum class CommMeshType {
    FULL_MESH, // 950 仅全互联一种通信形态(旧芯片环网形态随旧芯片逻辑剥离)
};

enum class CommTopology {
    STANDARD_CARD, // 0 - 标卡 4P
    EIGHT_P,       // 1 - 8P场景（POD8P/Server8P）
};

struct CommModelInfo {
    uint64_t rankDim;
    uint64_t commDtypeSize; // data type size in hccl
    uint64_t commMatrixLen; // equal to kValue in gather, equal to nValue in reduce
    CommMeshType commMethod;
    CommTopology topoType;
    uint64_t maxStepSize; // number of steps in a full round of communication
    uint64_t commDtypeSizeExpansionFraction;
};

// 通信时间线性拟合参数(3 字段): 总量公式 y = k·x + N·b + c(§4.37 专家裁定)——
// k=带宽斜率, b=每轮启动开销(单轮全款), c=一次性固定开销(只进总量口径, 不进单轮/反解)。
// 阶段1 结构先行: 现有标定行填 b=旧截距、c=0 → 与旧 y=k·x+offset 逐位等价(bit 锁保持);
// 阶段2 真实 b/c 标定数据到位后激活行为变化(与开销补丁删除同批, 独立性能验证)。
// 字段演进史 12→8→2→3 见 §4.28/§4.29/§4.37。
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

struct MatmulBaseBlock {
    uint32_t baseM = 256; // 256: init value for baseM
    uint32_t baseN = 256; // 256: init value for baseN
    uint32_t baseK = 128; // 128: init value for baseK
};

// 通信轮次硬上限（通信 API 的 FFTS 队列大小限制; 2026-09-17 随 API 上限放宽 16→63）
constexpr uint64_t MAX_COMM_ROUNDS = 63;

// 旧基类头文件里的通算比阈值（E9 减半鼓励门消费）
constexpr double CALC_COMM_BALANCE_RATIO = 2.0;

constexpr uint32_t DEFAULT_BASE_M = 128;
constexpr uint32_t DEFAULT_BASE_N = 256;
constexpr uint32_t DEFAULT_BASE_K = 64;

enum class AicpuComType {
    HCCL_CMD_INVALID = 0,
    HCCL_CMD_BROADCAST = 1,
    HCCL_CMD_ALLREDUCE,
    HCCL_CMD_REDUCE,
    HCCL_CMD_SEND,
    HCCL_CMD_RECEIVE,
    HCCL_CMD_ALLGATHER,
    HCCL_CMD_REDUCE_SCATTER,
    HCCL_CMD_ALLTOALLV,
    HCCL_CMD_ALLTOALLVC,
    HCCL_CMD_ALLTOALL,
    HCCL_CMD_GATHER,
    HCCL_CMD_HALFALLTOALLV = 20,
    HCCL_CMD_MAX
};

// 引擎调度入参(字段镜像旧 mc2tiling::FitBalanceTilingArgs, 算子侧边界处整字段转换)
struct FitBalanceTilingArgs {
    AicpuComType cmdType;
    uint32_t rankDim;
    uint32_t usedCoreNum;
    uint64_t orgMValue;
    uint64_t orgNValue;
    uint64_t orgKValue;
    uint64_t batchValue;
    uint64_t mValue;
    uint64_t kValue;
    uint64_t nValue;
    uint64_t baseMLimit;
    uint64_t inputDtypeSize;
    uint64_t outputDtypeSize;
    uint64_t aicCoreNum;
    uint64_t commTurn;
    uint64_t rankTileNum;
    uint8_t commAlg;
    bool isATrans;
    bool isBTrans;
    bool enablePad;
    bool enableSplitK;
    bool isBias;
    bool isStorageGather;
    bool isLocal = false;
    ge::DataType geAType;
    ge::DataType geBType;
    ge::DataType geCType;
    ge::DataType geBiasType;
    ge::DataType antiquantscaleDType;
    matmul_tiling::DataType aType;
    matmul_tiling::DataType bType;
    matmul_tiling::DataType cType;
    matmul_tiling::DataType biasType;
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

// ====== 本地 mm 块位置枚举(§4.38 核心键控维度, 2026-09-25 裁定) ======
// 引擎不区分通信原语——原语语义(份数口径/轴向/流水线结构)全部终结在标定层 CommProfile;
// 引擎唯一的结构键控是"本地 mm 块在流水线的位置"(不经通信的数据块, 2026-09-25 前按
// 通信原语隐式表达, 现显式化):
enum class LocalMMPosition {
    FRONT, // 本地 mm 块前置: 流水线首端(AGMM——本卡 A 块已在手, 聚合前即可计算, 启动端
           // 被本地块免费占据, 短块退到尾部专职快速收尾)
    BACK,  // 本地 mm 块后置: 流水线尾端(matmul_allto_all——本地保留份无需发出、无通信
          // deadline, 放尾部与末块通信重叠, 收尾端被本地块占据, 短块前置专职启动通信管道)
    NONE, // 无本地 mm 块(MMRS/MMAR/MMA2A——短块作流水线垫片, 按 bound 判定放首/尾)
};

// ====== 标定层结构参数的模式枚举(CommProfile 消费; 倍数依赖运行时 rankDim, 标定层存模式,
//       引擎一处集中换算——避免按原语的散落取值表) ======
enum class CommAxis { // 通信承载的数据侧(决定通信矩阵另一维与 dtype 字节取值)
    K,                // A 侧(输入): 通信沿 K 轴聚散(AGMM/A2A-in——传 A 矩阵行, kValue)
    N,                // C 侧(输出): 沿 N 轴(RS/AR/A2A-out——传 C 矩阵行, nValue)
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
constexpr uint64_t FP4_PACK_RATIO = 2;

// ============================================================================
// 一、模型层
// ============================================================================
// ---- 1.1 HCCL 通信模型与拟合表（自由长度分支, 950 专用）----
// 由旧 HCCLPerformanceModel + HCCLPerformanceArch35 合并精简(2026-09-15, 类名已去 arch35):
// 本引擎按设计仅在 950 标卡 4P/8P 生效, 旧芯片分支(310P 环网/910_93
// 双环/基表两段式查表——其结果恒被 950 查表覆盖)经 grep 证实 5 个消费
// 算子零调用后剥离; 老芯片的真实需求由原位保留的 B 家族(mc2/common/op_host/
// op_tiling)满足。活跃 API 面(键控分支/P2 算子实测): CommTime/InverseCommTime/
// SetCommShapeLen/SetCommDTypeSize/SetCommDtypeSizeExpansionFraction/
// commTimeFactor_/commRankCnt_/commRankCntForMm_(InitCtx 内部口径参数)。

// ====== 拟合表（引擎内部结构——算子侧一律使用 FitBalanceCalibration 大结构体,
//      此处类型由 InitCtx 内部适配构造并按值持有, 勿在算子侧直接引用）======

class HCCLCommModel {
public:
    CommFitParams commEstimatePar_;
    CommModelInfo commTypeInfo_;
    double commTimeFactor_ = 1.0; // 时间折算因子, 恒 1(§4.31: 8-die 折算桥接已随 plain RS 死分支删除)

    // Constructor(soc 恒 950; commFitParams 由 InitCtx 按 rankDim 从标定的行, miss=零值行)
    HCCLCommModel(uint32_t inputRankDim, CommFitParams commFitParams,
                  CommTopology topoType = CommTopology::STANDARD_CARD)
    {
        // 通信域卡数(op 侧来自 GetRankSize(group)), 下限钳位 2, 三层原因:
        //   ① 防 uint64 下溢: AG 远程流转份数 = rankDim-1, rankDim=0 时下溢;
        //   ② 防退化: 单卡无集合通信可建模, 拟合表均按 ≥2 卡标定, 退化输入按最接近的 2 卡建模;
        //   ③ 历史防御: 老基类服务全家族, rankDim 来源多样(attr/group/shape 推导)不可靠, 入口统一兜底
        commTypeInfo_.rankDim = std::max(static_cast<uint64_t>(inputRankDim), MIN_RANK_DIM);
        commTypeInfo_.commDtypeSizeExpansionFraction = 1;
        commTypeInfo_.maxStepSize = 1;
        commTypeInfo_.commMethod = CommMeshType::FULL_MESH; // 全互联(旧 InitPrimitiveParams 残余, 仅完整性保留)
        commTypeInfo_.topoType = topoType;
        commEstimatePar_ = commFitParams; // InitCtx 按 rankDim 选定的标定行(miss=零值行)
    };

    // ---- 通信口径设置(InitCtx 键控胶水调用; fp4 子字节= dtypeSize 1 + fraction 2) ----
    void SetCommShapeLen(uint64_t len)
    {
        commTypeInfo_.commMatrixLen = len;
    }
    void SetCommDTypeSize(uint64_t dTypeSize)
    {
        commTypeInfo_.commDtypeSize = dTypeSize;
    }
    void SetCommDtypeSizeExpansionFraction(uint64_t number)
    {
        commTypeInfo_.commDtypeSizeExpansionFraction = number;
    }
    double GetRealDtypeSizes() const // 有效每元素字节数 = dtypeSize / fraction
    {
        return static_cast<double>(commTypeInfo_.commDtypeSize) / commTypeInfo_.commDtypeSizeExpansionFraction;
    }
    uint64_t rankDim() const
    {
        return commTypeInfo_.rankDim;
    }

    // 单轮通信耗时: y = k·x + b(§4.37)——本轮数据量按带宽线性计, 外加该轮启动开销 b;
    // 一次性开销 c 不进单轮口径。配平(E18/E19 反解)消费的就是这个单轮语义。
    double CommTime(uint64_t mSize) const
    {
        const double commDataSize = static_cast<double>(mSize) * commTypeInfo_.commMatrixLen * GetRealDtypeSizes();
        const double tmpSize = commDataSize / BYTES_PER_MB;
        return (commEstimatePar_.slopeUsPerMb * tmpSize + commEstimatePar_.roundOverheadUs) / commTimeFactor_;
    }

    // 纯线性反解: x = (t·factor − b) / k, 再换算回 M 行数——b 是每轮全款, 从目标时间整额扣除
    uint64_t InverseCommTime(double targetTime) const
    {
        const double tmpSize =
            (targetTime * commTimeFactor_ - commEstimatePar_.roundOverheadUs) / commEstimatePar_.slopeUsPerMb;
        return static_cast<uint64_t>(tmpSize * BYTES_PER_MB / (commTypeInfo_.commMatrixLen * GetRealDtypeSizes()));
    }

    // 总通信耗时(N 轮求和口径): y = k·X + N·b + c(§4.37)——X 为切分域总数据量, 每轮各付
    // 一份启动开销 b, c 整个算子只付一次。切分完成后按实际轮数精算, 与 NoCut 候选(N=1)
    // 之差 = (N−1)·b, 即切分多付的启动税; Estimate 阶段轮数未知, 按 N=1 保守估(§4.37 拍板)。
    double TotalCommTime(uint64_t totalLen, uint64_t numRounds) const
    {
        const double commDataSize = static_cast<double>(totalLen) * commTypeInfo_.commMatrixLen * GetRealDtypeSizes();
        const double tmpSize = commDataSize / BYTES_PER_MB;
        return (commEstimatePar_.slopeUsPerMb * tmpSize +
                commEstimatePar_.roundOverheadUs * static_cast<double>(numRounds) + commEstimatePar_.fixedOverheadUs) /
               commTimeFactor_;
    }
};

// ---- 1.2 matmul 模型（HCCL 算力/梯度模型, 自由长度分支专用, 950）----
// 由旧 MatmulPerformanceModel + MatmulPerformanceArch35 合并精简(2026-09-15, 类名已去 arch35),
// 剥离 310P/910_B4 分支与 4 参 FindCubeUtil(含其独占的 K 对齐修正链)——
// 5 个消费算子经 grep 证实仅用 FindCubeUtil(§4.34 合并为单一参数化函数)。

namespace MatmulPerfConst {
constexpr double COMPUTES_PER_CYCLE = 4096; // 基础算力(查表未命中回退, 标定头回填同值)
// cycle per micro second (950)
constexpr double CYCLE_PER_MICRO_SEC = 1.65 * BYTES_PER_KB;
// end: cycle per micro second
constexpr double MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL = 1.4;
constexpr double MNVALUE_THRESHOLD = 820;
constexpr double KVALUE_THRESHOLD = 1280;
constexpr double LOCAL_MM_UTIL_EXPONENT_COEFFICIENT = 0.6; // 本地窗口(单份数据/流水未满)的 MN 指数
constexpr double CUBE_UTIL_EXPONENT_COEFFICIENT = 0.7;
constexpr double MAX_CUBE_UTIL = 0.95;
constexpr double AVERAGE_CUBE_UTIL = 0.75;
} // namespace MatmulPerfConst

class MatmulPerfModel {
public:
    MatmulShapeInfo mmShapeInfo_;
    MatmulDtypeKind calcType_; // Distinguish the non-quantization, full-quantization and fake-quantization
    bool isFp4_ = false;       // fp4 输入(geAType==DT_FLOAT4_E2M1), 影响 CubeCalc/L2 查表匹配
    double cubeUtil_ = 0.8;
    double matmulGradient_ = 1.0;

    // Constructor(soc 恒 950, 频率 1689.6 cycle/us; 选行参数由 InitCtx 按 dtype 五元组从标定选定)
    MatmulPerfModel(const FitBalanceTilingArgs &args, double computesPerCycle, MatmulBaseBlock baseBlock)
    {
        mmShapeInfo_.socType = TilingSocVersion::SOC950;
        mmShapeInfo_.coreNum = args.aicCoreNum; // 每die核数
        mmShapeInfo_.inMatrixADtypeSize = args.inputDtypeSize;
        mmShapeInfo_.inMatrixBDtypeSize = mmShapeInfo_.inMatrixADtypeSize;
        mmShapeInfo_.outMatrixCDtypeSize = args.outputDtypeSize;
        mmShapeInfo_.mValue = args.mValue;
        mmShapeInfo_.nValue = args.nValue;
        mmShapeInfo_.kValue = args.kValue;
        mmShapeInfo_.baseM = baseBlock.baseM;
        mmShapeInfo_.baseN = baseBlock.baseN;
        mmShapeInfo_.baseK = baseBlock.baseK;
        mmShapeInfo_.batchSize = 1UL; // 初始值
        isFp4_ = (args.geAType == ge::DataType::DT_FLOAT4_E2M1);
        SetCalcType(args);
        mmShapeInfo_.computesPerCycle = computesPerCycle;
        mmShapeInfo_.cyclePerMicroSec = MatmulPerfConst::CYCLE_PER_MICRO_SEC;
    };

    // dtype 分类(选行匹配键的推导; 查表选行已在 InitCtx 完成):
    // 950 量化语义: fp4/fp8/HIF8 或 A8W8 量化, 其余按 FP16 建模;
    // fp4 经 geAType 识别(matmul_tiling::DataType 无 fp4 枚举, aType 对 fp4 为 DT_MAX, §4.17)
    void SetCalcType(const FitBalanceTilingArgs &args)
    {
        if ((args.aType == matmul_tiling::DataType::DT_HIFLOAT8) ||
            (args.aType == matmul_tiling::DataType::DT_FLOAT8_E4M3FN) ||
            (args.aType == matmul_tiling::DataType::DT_FLOAT8_E5M2) ||
            (args.aType == matmul_tiling::DataType::DT_INT8) || (args.geAType == ge::DataType::DT_FLOAT4_E2M1)) {
            calcType_ = MatmulDtypeKind::QUANT;
        }
    }
    // cube 利用率(950 调和平均公式; §4.34 由 FindCubeUtil/FindCubeUtilByL2Usage 两函数合并而来——
    // 老代码 L2 三档+核占用比在 950 上被本公式无条件覆盖、maxTileLen 出参无人请求, 全系死代码,
    // 随 L2 标定链一并删除):
    //   util = min(0.95, 0.75 × (K/1280)^0.7 × (调和平均/820)^mnExp)
    //   调和平均 = (mSize × commRankCntForMm × n) / (mSize × commRankCntForMm + n)
    // mnExponent: MN 因子指数, 两种口径——聚合稳态 0.7(数据齐/流水满, 增长快)、
    //   AG 本地首块(front) 0.6(只有本卡份/流水未满, 增长平缓), 是两个口径的真实物理差别
    double FindCubeUtil(uint64_t mSize, uint64_t commRankCntForMm, double mnExponent)
    {
        double mnharmonicMean = static_cast<double>(mSize * commRankCntForMm * mmShapeInfo_.nValue) /
                                static_cast<double>(mSize * commRankCntForMm + mmShapeInfo_.nValue);
        double kExponentiated = std::min(
            MatmulPerfConst::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
            std::pow(static_cast<double>(mmShapeInfo_.kValue), MatmulPerfConst::CUBE_UTIL_EXPONENT_COEFFICIENT) /
                std::pow(MatmulPerfConst::KVALUE_THRESHOLD, MatmulPerfConst::CUBE_UTIL_EXPONENT_COEFFICIENT));
        double mnExponentiated =
            std::min(MatmulPerfConst::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
                     std::pow(mnharmonicMean, mnExponent) / std::pow(MatmulPerfConst::MNVALUE_THRESHOLD, mnExponent));
        double result = MatmulPerfConst::AVERAGE_CUBE_UTIL * kExponentiated * mnExponentiated;
        return std::min(MatmulPerfConst::MAX_CUBE_UTIL, result);
    }
    // 聚合稳态口径(整 m + 全部流转份数, 设 cubeUtil_ 供 GetMatmulGradient 消费)
    void FindCubeUtil(uint64_t commRankCntForMm)
    {
        cubeUtil_ =
            FindCubeUtil(mmShapeInfo_.mValue, commRankCntForMm, MatmulPerfConst::CUBE_UTIL_EXPONENT_COEFFICIENT);
    }
    void GetMatmulGradient()
    {
        uint64_t tmpN = std::max(mmShapeInfo_.baseN, mmShapeInfo_.nValue);
        uint64_t tmpK = std::max(mmShapeInfo_.baseK, mmShapeInfo_.kValue);
        matmulGradient_ = (tmpN * tmpK) / (mmShapeInfo_.computesPerCycle * mmShapeInfo_.cyclePerMicroSec *
                                           mmShapeInfo_.coreNum * cubeUtil_);
    }
    uint32_t GetBaseM()
    {
        return mmShapeInfo_.baseM;
    };
    uint32_t GetBaseN()
    {
        return mmShapeInfo_.baseN;
    };
    // 性能预测
    double MatmulTime(uint64_t tileSize, uint64_t rowMul)
    {
        tileSize = std::max(mmShapeInfo_.baseM, tileSize);
        return static_cast<double>(tileSize) * static_cast<double>(rowMul) * matmulGradient_;
    }
    uint64_t InverseMatmulTime(double targetTime, uint64_t rowMul)
    {
        uint64_t tileSize = static_cast<uint64_t>(targetTime / (static_cast<double>(rowMul) * matmulGradient_));
        tileSize = std::max(mmShapeInfo_.baseM, tileSize);
        return tileSize;
    }
};

// ============================================================================
// 二、切分与输出层（CutPlanner + CommCutResult 适配）
// ============================================================================
// ============================================================================

// (2026-09-15 精简: 连续切分路径 FitTileLengthContinuous/SmallFrontTileCheck/
//  LargeBackTileCheck/CommBoundShortAlign 为 B 家族(公式化)专用, A 家族 5 算子
//  经 grep 证实零调用, 已随旧芯片逻辑剥离; B 家族原位副本不受影响)
constexpr uint64_t GOOD_LINEARITY_NK_BYTES = 32;
constexpr uint64_t SMALL_CUT_COUNT = 2;
constexpr double BACK_TILE_RATIO = 0.75;
constexpr double ALIGN_TOL_CALC_BOUND = 0.5;
constexpr double ALIGN_TOL_COMM_BOUND = 0.75;

// 切分和对齐
class CutPlanner {
public:
    CommCutResult cutRes;                   // 切分结果
    CutParams tileArgs;                     // 切分参数
    uint64_t totalLen = 1U;                 // 待切分的长度
    double backTileRatio = BACK_TILE_RATIO; // 计算长块个数使用的参数

    // Constructor
    explicit CutPlanner(const FitBalanceTilingArgs &args)
    {
        totalLen = args.mValue;
        tileArgs.mAlignLen = DEFAULT_BASE_M; // 切分长度照顾mm性能，对齐baseM
        tileArgs.minTileLen = tileArgs.mAlignLen;
        tileArgs.maxTileCnt = 1U;
        tileArgs.maxTileLen = totalLen;

        cutRes.shortTileAtBack = false; // 计算Bound时短块后置，通信bound时短块前置
        NoCutTiling();                  // 初始化为不切分
    };

    void SetMinLenByMax(uint64_t newLen);
    void SetMinLenByMin(uint64_t newLen);
    uint64_t GetMinLen()
    {
        return tileArgs.minTileLen;
    };
    void SetAlignLength(uint64_t newLen)
    {
        tileArgs.mAlignLen = std::max(newLen, 1UL);
    };
    uint64_t GetAlignLength()
    {
        return tileArgs.mAlignLen;
    };
    void SetMaxTileCnt(uint64_t newSize)
    {
        tileArgs.maxTileCnt = std::max(newSize, 1UL);
    };

    // TilingMethods
    void GenerateInitialPartition();
    bool SetShortTileLen(bool noCutFlag = false);
    void NoCutTiling();
    void MaxTileCntUniform();

    // 离散切分对齐 (A 家族全部算子的 Adjust 阶段使用)
    void AlignTileSmall();
    void AlignTileLarge();
    void FitTileLengthDiscrete(bool goodLinearityShape, bool localBlockProtected = false);
    void CalcShortTile();
    void CalcBackTile();
    void RedistributeTile();
};

inline void CutPlanner::SetMinLenByMax(uint64_t newLen)
{
    tileArgs.minTileLen = std::max(tileArgs.minTileLen, newLen);
}

inline void CutPlanner::SetMinLenByMin(uint64_t newLen)
{
    tileArgs.minTileLen = std::min(tileArgs.minTileLen, newLen);
}

inline void CutPlanner::AlignTileLarge()
{
    if (cutRes.numLongTile >= tileArgs.mAlignLen) {
        return;
    }
    // Align cutRes.shortTileLen
    cutRes.shortTileLen = (cutRes.shortTileLen + tileArgs.mAlignLen - 1) / tileArgs.mAlignLen * tileArgs.mAlignLen;
    cutRes.longTileLen = (totalLen - cutRes.shortTileLen + cutRes.numLongTile - 1) / cutRes.numLongTile;
    cutRes.shortTileLen = totalLen - cutRes.numLongTile * cutRes.longTileLen;
}

inline void CutPlanner::AlignTileSmall()
{
    // Front tile already aligned, check backTile
    bool smallBackTile = (cutRes.shortTileLen < tileArgs.mAlignLen) || (cutRes.shortTileLen < tileArgs.minTileLen);
    smallBackTile = smallBackTile && cutRes.shortTileLen > 0U;
    bool largeFrontTile =
        (cutRes.longTileLen > 2 * tileArgs.mAlignLen) && (cutRes.longTileLen > 2 * tileArgs.minTileLen);
    if (smallBackTile && largeFrontTile) {
        uint64_t demand = tileArgs.mAlignLen * cutRes.numLongTile;
        bool reBalance = (cutRes.shortTileLen + demand) <= cutRes.longTileLen;
        if (reBalance) {
            cutRes.shortTileLen += demand;
            cutRes.longTileLen -= tileArgs.mAlignLen;
        }
    }

    // 假如对齐后尾块长度大于主块，尝试校正使尾块长度<=主块长度
    // 校正目标：主块长度为totalTileCnt均分MPerRank后向上对齐mAlignLen，尾块长度为MPerRank剩余大小
    // 校正检查：totalTileCnt均分后，所有主块加起来最多向尾块抽取 mAlignLen *
    // numLongTile 大小，尾块剩余大小为 (MPerRank/totalTileCnt) - mAlignLen *
    // numLongTile 校正检查：检查条件为 (MPerRank/totalTileCnt) > mAlignLen *
    // totalTileCnt, 确保尾块至少剩下mAlignLen大小
    bool largeBackTile = (cutRes.shortTileLen > cutRes.longTileLen) &&
                         ((totalLen / cutRes.totalTileCnt) > (tileArgs.mAlignLen * cutRes.totalTileCnt));
    if (largeBackTile) {
        cutRes.longTileLen =
            (totalLen / cutRes.totalTileCnt + tileArgs.mAlignLen - 1) / tileArgs.mAlignLen * tileArgs.mAlignLen;
        cutRes.shortTileLen = totalLen - cutRes.numLongTile * cutRes.longTileLen;
    }
}

// 对齐容差取整: 计算bound(短块后置) 0.5 中性舍入, 通信bound(短块前置) 0.75 消极进位
// (小形态强制对齐开关 smallDimAlignUp 系 B 家族/A2AMM 专属——引擎消费者恒 false, §4.35 删)
inline void CutPlanner::GenerateInitialPartition()
{
    cutRes.longTileLen = std::max(cutRes.longTileLen, cutRes.shortTileLen);
    cutRes.longTileLen = std::min(cutRes.longTileLen, totalLen - cutRes.shortTileLen);
    uint64_t newTileLen = cutRes.longTileLen / tileArgs.mAlignLen * tileArgs.mAlignLen;
    double alignRatio = ALIGN_TOL_CALC_BOUND;
    if (!cutRes.shortTileAtBack) {
        alignRatio = ALIGN_TOL_COMM_BOUND;
    }

    bool alignUpFlag = (cutRes.longTileLen - newTileLen) > (tileArgs.mAlignLen * alignRatio);
    if (alignUpFlag) {
        cutRes.longTileLen = newTileLen + tileArgs.mAlignLen;
    } else {
        cutRes.longTileLen = newTileLen;
    }
    // find cutRes.totalTileCnt
    cutRes.longTileLen = std::min(cutRes.longTileLen, tileArgs.maxTileLen);
    cutRes.numLongTile = (totalLen - static_cast<uint64_t>(static_cast<double>(cutRes.shortTileLen) * backTileRatio)) /
                         cutRes.longTileLen;
    cutRes.numLongTile = std::max(cutRes.numLongTile, static_cast<uint64_t>(1));
    cutRes.totalTileCnt = cutRes.numShortTile + cutRes.numLongTile;
}

inline void CutPlanner::FitTileLengthDiscrete(bool goodLinearityShape, bool localBlockProtected)
{
    // 流水配平，找到初始切分长度和个数
    GenerateInitialPartition();

    // 处理剩余长度
    uint64_t resLen = totalLen - cutRes.numLongTile * cutRes.longTileLen - cutRes.numShortTile * cutRes.shortTileLen;
    bool largeBackTile =
        (resLen >= cutRes.shortTileLen) && ((SMALL_CUT_COUNT * cutRes.shortTileLen) >= cutRes.longTileLen);
    goodLinearityShape =
        (goodLinearityShape && (cutRes.shortTileLen >= tileArgs.mAlignLen)) || (!cutRes.shortTileAtBack);
    bool smallCut = cutRes.numLongTile <= SMALL_CUT_COUNT;
    bool balanceCut = largeBackTile && goodLinearityShape && smallCut;
    if (balanceCut) { // 重新均匀切分
        cutRes.longTileLen = cutRes.shortTileLen;
        cutRes.numLongTile = (totalLen - cutRes.shortTileLen) / cutRes.longTileLen; // floor
        cutRes.numLongTile = std::max(cutRes.numLongTile, static_cast<uint64_t>(1));
        cutRes.totalTileCnt = cutRes.numShortTile + cutRes.numLongTile;
        largeBackTile = false;
    }
    if (cutRes.totalTileCnt == 2U && !localBlockProtected) { // 有本地块的形态跳过两块均摊
                                                             // (保护含本地块的端部结构)
        // 如果前面没有local特殊处理，就把剩余长度均分给长块和短块
        RedistributeTile();
        largeBackTile = false;
    }
    bool moreFrontTile = largeBackTile && (cutRes.longTileLen <= cutRes.shortTileLen + resLen);
    if (moreFrontTile) { // 增加长块个数
        cutRes.numLongTile++;
        CalcBackTile();
    }

    // 检查切分轮次总数符合约束
    tileArgs.maxTileCnt = std::max(tileArgs.maxTileCnt, 1UL); // 确保maxTileCnt > 0
    if (cutRes.totalTileCnt > tileArgs.maxTileCnt) {
        uint64_t tileLen = totalLen / tileArgs.maxTileCnt;
        bool isDivisible = (tileLen * tileArgs.maxTileCnt == totalLen) && (tileLen % tileArgs.mAlignLen == 0U);
        if (isDivisible) {
            cutRes.numLongTile = tileArgs.maxTileCnt;
            cutRes.longTileLen = tileLen;
            cutRes.numShortTile = 0U;
            cutRes.shortTileLen = 0U;
            cutRes.totalTileCnt = tileArgs.maxTileCnt;
        }
    }
    MaxTileCntUniform();

    cutRes.shortTileLen = totalLen - cutRes.numLongTile * cutRes.longTileLen;
    if (cutRes.longTileLen > BYTES_PER_KB) {
        AlignTileLarge();
    } else {
        AlignTileSmall();
    }
}

inline void CutPlanner::NoCutTiling()
{
    cutRes.totalTileCnt = 1U;
    cutRes.numShortTile = 0U;
    cutRes.numLongTile = 1U;
    cutRes.shortTileLen = 0U;
    cutRes.longTileLen = totalLen;
}

inline void CutPlanner::CalcShortTile()
{
    cutRes.numShortTile = 0;
    cutRes.shortTileLen = 0;
    if (totalLen != cutRes.longTileLen * cutRes.numLongTile) {
        cutRes.numShortTile = 1;
        cutRes.shortTileLen = totalLen - cutRes.longTileLen * cutRes.numLongTile;
    }
}

inline void CutPlanner::CalcBackTile()
{
    CalcShortTile();
    bool tinyBackTile = (cutRes.shortTileLen > 0) && (cutRes.shortTileLen < tileArgs.mAlignLen * 0.5);
    if (tinyBackTile && cutRes.numLongTile > 1) {
        cutRes.numLongTile -= 1;
        cutRes.shortTileLen += cutRes.longTileLen;
    }
    cutRes.totalTileCnt = cutRes.numShortTile + cutRes.numLongTile;
}

// 据经验值，切分块数较多且每块长度接近matmul临界值时，ETE性能会劣化
inline void CutPlanner::RedistributeTile()
{
    uint64_t tmpBackTileLen = totalLen - cutRes.numLongTile * cutRes.longTileLen;
    bool redistTileFlag = tmpBackTileLen > 2 * cutRes.shortTileLen;
    if (redistTileFlag) {
        uint64_t redistTileLen = (tmpBackTileLen - cutRes.shortTileLen) / cutRes.totalTileCnt;
        redistTileLen = redistTileLen / tileArgs.mAlignLen * tileArgs.mAlignLen;
        cutRes.longTileLen += redistTileLen;
    }
    CalcBackTile();
}

inline bool CutPlanner::SetShortTileLen(bool noCutFlag)
{
    cutRes.shortTileLen = tileArgs.minTileLen;
    cutRes.numShortTile = 1U;
    // do not cut if mValue is too small
    bool smallMPerRank = (totalLen < cutRes.shortTileLen * 2);
    smallMPerRank = smallMPerRank || noCutFlag;
    if (smallMPerRank) {
        NoCutTiling();
    }
    return smallMPerRank;
}

inline void CutPlanner::MaxTileCntUniform()
{
    if (cutRes.totalTileCnt <= tileArgs.maxTileCnt) {
        return;
    }
    if (tileArgs.maxTileCnt <= 1 || totalLen < tileArgs.maxTileCnt) {
        NoCutTiling();
        return;
    }

    // 尾块应该小于主块
    cutRes.longTileLen = totalLen / (tileArgs.maxTileCnt - 1);
    cutRes.longTileLen = (cutRes.longTileLen + tileArgs.mAlignLen - 1) / tileArgs.mAlignLen * tileArgs.mAlignLen;
    cutRes.longTileLen = std::min(cutRes.longTileLen, totalLen);
    cutRes.numLongTile = totalLen / cutRes.longTileLen;
    CalcBackTile();
}
// ============================================================================
// 三、求解层（两分支求解器）
// ============================================================================
// ---- 3.1 自由长度求解器（ctx + 键控分支, HCCL AICPU/CCU; 钩子机制已于 §4.24 废除）----
// ============================================================================

// 调度上下文: 引擎三件套 + 键控维度 + 过程量(刻意不含静态标定数据, 在 FitBalanceCalibration)
struct FitBasedBalanceCtx {
    MatmulPerfModel matmulPerf;
    HCCLCommModel commPerf;
    CutPlanner tilingM;
    LocalMMPosition localMMPosition; // 键控维度: 本地 mm 块位置(§4.38——引擎唯一的结构键控,
                                     // 取代原通信原语; 原语语义全部终结在标定层 CommProfile)
    CommComputeOrder order;          // 键控维度: 通算顺序
    TilingCommEngine commEngine;     // 键控维度: 通信引擎(CCU 专属的单轮通信量上限整形)
    uint64_t commRankCntForMm = 1;   // 派生: CommProfile.mmWorksetMul 换算(mm 工作集 M 跨度倍数)
    uint64_t mmRankMul = 1;          // 派生: CommProfile.mmTotalMul 换算(mm 总行数倍数)
    uint64_t commVolumeMul = 1; // 派生: CommProfile.commVolumeMul 换算(单次通信调用物理字节倍数, ⑥ 消费)
    double ratioCalcComm = 1.0; // 过程量: Estimate 骨干设置的通算比(键控分支消费)
    double frontMMTime = 0; // 过程量(FRONT 键): 本地首块计算耗时(流水线首端, 启动端被本地块占据)
    double backMMTime = 0; // 过程量(BACK 键): 本地尾块计算耗时(流水线尾端, 与末块通信重叠)
    // 诊断量(§4.37, Schedule 尾部写入): 切分完成后按实际轮数精算的总通信时, 与 NoCut 候单轮
    // 之差 = (N−1)·b 启动税。仅供 op 侧日志消费, 不参与引擎决策(机制与日志分离, §4.24 裁定)
    double commTimeActualUs = 0; // 实际切分(N 轮)的总通信时
    double commTimeNoCutUs = 0;  // 不切候选(1 轮)的总通信时
};

// ============================================================================
// 客户标定数据总集（算子侧唯一需要接触的标定类型, 2026-09-15 评审裁定:
// 单一大结构体替代"引擎侧一堆小结构体 + 算子侧一堆具名 constexpr"）
//
// 用法（算子侧标定头, 填一个实例传给 InitCtx 即可）:
//   static const FitBalanceCalibration CALIBRATION = {
//       /* aicpu */  { /* rows */ { {"4_1_0_4", 斜率, 截距} } },
//       /* ccu */    { /* rows */ { {"4_1_0_4", 斜率, 截距} } },
//       /* matmul */ { /* cubeCalc */ { {MatmulDtypeKind, isFp4, aB, bB, cB, 算力} }, ... },
//   };
//
// 填写规则:
//   - 通信行按 rankDim 匹配(未标定形态由 op 层分派走核利用率优先, §4.29——引擎不设回退);
//   - matmul 行按 (calcType, A/B/C 字节数) 匹配, 未填字段取 NSDMI 默认; 行按前缀填, 0 值行=未填;
//   - 通信拟合公式 y = k·x + N·b + c(§4.37): 客户标定产出 k/b/c 三个数。
// ============================================================================

struct FitBalanceCalibration {
    static constexpr size_t MAX_COMM_FIT_ROWS = 4;   // 已标定通信形态数上限
    static constexpr size_t MAX_CUBE_CALC_ROWS = 4;  // matmul 算力档位数上限
    static constexpr size_t MAX_BASE_BLOCK_ROWS = 2; // 最优基本块档位数上限

    // —— 通信结构参数(§4.38: 原语语义终结在标定层——每算子一份, 引擎不区分通信原语) ——
    // 倍数依赖运行时 rankDim 故存模式枚举, 引擎经 RankMulValue() 一处集中换算。
    struct CommProfile {
        CommAxis axis = CommAxis::K;         // 通信承载侧: K=A 侧输入(传 A 行), N=C 侧输出(传 C 行)
        RankMul mmWorksetMul = RankMul::ONE; // mm 工作集 M 跨度倍数(FindCubeUtil): AG=rankDim−1
                                             // (自己的份不经通信, 本地块单独按 1 建模), RS/A2A=rankDim, AR=1
        RankMul mmTotalMul = RankMul::ONE;    // mm 总行数倍数(MatmulTime): AG/RS=rankDim(聚合全量/
                                              // scatter 除前全量), AR/A2A=1
        RankMul commVolumeMul = RankMul::ONE; // 单次通信调用的物理字节倍数(⑥ 的 256MB 上限判定用,
                                              // 与 k 的标定口径融合无关): AG/RS=rankDim(单次调用搬全量),
                                              // AR/A2A=1(单次调用按单份)
    };
    CommProfile profile{};

    // —— 通信拟合（单引擎）——
    struct CommFit {
        struct Row {
            uint32_t rankDim;             // 卡数(4=标卡4P, 8=8P, ...)
            double slopeUsPerMb;          // 带宽斜率 k (us/MB)
            double roundOverheadUs;       // 每轮启动开销 b (us)
            double fixedOverheadUs = 0.0; // 一次性固定开销 c (us)——未标定填 0(阶段1 兼容旧截距口径)
        };
        Row rows[MAX_COMM_FIT_ROWS] = {}; // 已标定行, rankDim=0 表示未填(哨兵截断)
    };
    CommFit aicpu; // AICPU 引擎(comm_mode != "ccu")
    CommFit ccu;   // CCU 引擎(comm_mode == "ccu")

    // —— matmul 性能拟合（对仗 CommFit; 字段序: 用户常改字段在前, 平台默认在后——聚合初始化
    //    可省尾部, 省略的尾部字段自动取 NSDMI 默认值）——
    struct MatmulFit {
        // 键维度裁定(§4.37 缺陷根治, 2026-09-22): 不含 C 侧字节数——glue 恒 cType=aType,
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
    };
    MatmulFit mmFit;
};

// 统一入口: 算子侧只传大结构体标定数据 + 键控枚举(原语/顺序/引擎; soc 恒 950)
inline FitBasedBalanceCtx InitCtx(const FitBalanceTilingArgs &args, LocalMMPosition localMMPosition,
                                  CommComputeOrder order, TilingCommEngine commEngine,
                                  const FitBalanceCalibration &calib,
                                  CommTopology topoType = CommTopology::STANDARD_CARD)
{
    // ---- 标定选行(所有匹配键此刻已知: rankDim 与 dtype 均来自 args, 就地选定传纯参数) ----
    // 通信行: 按引擎分区(aicpu/ccu) + rankDim 匹配首个命中行; miss=零值行(§4.29 防御 NoCut)
    const auto &commRows = ((commEngine == TilingCommEngine::CCU) ? calib.ccu : calib.aicpu).rows;
    CommFitParams commFitParams = {};
    for (size_t i = 0; i < FitBalanceCalibration::MAX_COMM_FIT_ROWS; ++i) {
        if (commRows[i].rankDim == 0) { // 哨兵截断
            break;
        }
        if (commRows[i].rankDim == args.rankDim) {
            commFitParams = {commRows[i].slopeUsPerMb, commRows[i].roundOverheadUs, commRows[i].fixedOverheadUs};
            break;
        }
    }
    // matmul 行: 按"计算形态"匹配算力行——形态键 = (量化类别, fp4 打包, A/B 字节数) 四维
    // (C 侧字节数不进键, §4.37 根治裁定: glue cType=aType 使其恒等于 aBytes, 零信息量),
    // 键推导与模型 SetCalcType 同源
    const auto isQuant = (args.aType == matmul_tiling::DataType::DT_HIFLOAT8) ||
                         (args.aType == matmul_tiling::DataType::DT_FLOAT8_E4M3FN) ||
                         (args.aType == matmul_tiling::DataType::DT_FLOAT8_E5M2) ||
                         (args.aType == matmul_tiling::DataType::DT_INT8) ||
                         (args.geAType == ge::DataType::DT_FLOAT4_E2M1);
    const MatmulDtypeKind calcType = isQuant ? MatmulDtypeKind::QUANT : MatmulDtypeKind::FP16;
    const bool isFp4 = (args.geAType == ge::DataType::DT_FLOAT4_E2M1);
    const auto matchesCalcProfile = [&](const auto &row) {
        return (row.calcType == calcType) && (row.isFp4 == isFp4) && (row.aBytes == args.inputDtypeSize) &&
               (row.bBytes == args.inputDtypeSize); // C 侧字节数不进键(§4.37 根治裁定, 见 CubeCalc 注释)
    };
    double computesPerCycle = calib.mmFit.defaultComputesPerCycle;
    for (size_t i = 0; i < FitBalanceCalibration::MAX_CUBE_CALC_ROWS; ++i) {
        if (calib.mmFit.cubeCalc[i].computesPerCycle == 0.0) { // 哨兵截断(未填行; 勿用 aBytes——
            break;                                             // glue 真实口径 fp4 行 aBytes=0 合法, 见
        } // AGMM_QUANT_CPC_KEY_DEFECT_AND_FIX.md)
        if (matchesCalcProfile(calib.mmFit.cubeCalc[i])) {
            computesPerCycle = calib.mmFit.cubeCalc[i].computesPerCycle;
            break;
        }
    }
    FitBasedBalanceCtx ctx{MatmulPerfModel(args, computesPerCycle, calib.mmFit.baseBlock),
                           HCCLCommModel(args.rankDim, commFitParams, topoType),
                           CutPlanner(args),
                           localMMPosition,
                           order,
                           commEngine,
                           1,
                           1,
                           1,
                           1.0};
    ctx.tilingM.SetMaxTileCnt(MAX_COMM_ROUNDS); // 通信 API FFTS 队列轮次上限(2026-09-17: 63)
    // §4.38 结构参数入口换算(标定层 CommProfile 声明模式, 引擎一处集中换算)
    ctx.commRankCntForMm = RankMulValue(calib.profile.mmWorksetMul, ctx.commPerf.rankDim());
    ctx.mmRankMul = RankMulValue(calib.profile.mmTotalMul, ctx.commPerf.rankDim());
    ctx.commVolumeMul = RankMulValue(calib.profile.commVolumeMul, ctx.commPerf.rankDim());

    // ---- 通算胶水(键控, 原散在各算子构造函数, §4.24 裁定随引擎下沉) ----
    // D4 通信矩阵另一维(§4.38: 轴向由标定层 CommProfile.axis 声明——A 侧承载沿 K 轴, C 侧沿 N 轴)
    const bool commOnKAxis = (calib.profile.axis == CommAxis::K);
    ctx.commPerf.SetCommShapeLen(commOnKAxis ? ctx.matmulPerf.mmShapeInfo_.kValue : ctx.matmulPerf.mmShapeInfo_.nValue);
    // D5+D6 通信承载 dtype 与 fp4 子字节建模(两件独立的事, 原耦合在 AG 构造函数单分支内, §4.33 解耦):
    // ① 通信 dtype 按 CommProfile.axis 承载侧: K(A 侧输入)取 A 字节, N(C 侧输出)取 C 字节——
    //    fp4 只可能是输入 A 的 dtype, C 侧(计算输出)永远不是 fp4, 故子字节建模仅 K 轴承载时需要;
    // ② fp4(0.5B/元素)无法用 uint64 表达: 通信侧按"1 字节打包 2 元素"建模(dtypeSize=1+
    //    fraction=2 → GetRealDtypeSizes()=0.5, 复用 HCCL 子字节机制)
    const bool isFp4Input = (args.geAType == ge::DataType::DT_FLOAT4_E2M1);
    if (commOnKAxis) {
        if (isFp4Input) {
            ctx.commPerf.SetCommDTypeSize(1);
            ctx.commPerf.SetCommDtypeSizeExpansionFraction(FP4_PACK_RATIO);
        } else {
            ctx.commPerf.SetCommDTypeSize(ctx.matmulPerf.mmShapeInfo_.inMatrixADtypeSize);
        }
    } else {
        ctx.commPerf.SetCommDTypeSize(ctx.matmulPerf.mmShapeInfo_.outMatrixCDtypeSize);
    }
    // ③ mm 侧 fp4 口径(与通信承载哪侧无关, 故独立于上分支): L2 工作集估算按 1B/元素保守取值
    //    (真实 0.5B 无法表达, 2x 高估防 memUsage 退化); 算力查表已由 matchesCalcProfile 的
    //    isFp4 维度接管(§4.17), 不在此处理。op 侧对 fp4 传 inputDtypeSize=1 时此两行为防御冗余
    if (isFp4Input) {
        ctx.matmulPerf.mmShapeInfo_.inMatrixADtypeSize = 1;
        ctx.matmulPerf.mmShapeInfo_.inMatrixBDtypeSize = 1;
    }
    // §4.29 分派后引擎不设回退模型: 标定未命中(零值行 slope=0)时置 minLen=mValue——
    // NoCut 门控即输出单块不切(防御行为, 正常分派下 op 层已引走 COF, 不可达;
    // 同时避免 InverseCommTime 除零斜率的 float→uint UB)
    if (ctx.commPerf.commEstimatePar_.slopeUsPerMb == 0.0) {
        ctx.tilingM.SetMinLenByMax(ctx.matmulPerf.mmShapeInfo_.mValue);
    }
    // D7 离散对齐粒度初始化: baseM
    ctx.tilingM.SetMinLenByMax(ctx.matmulPerf.GetBaseM());
    ctx.tilingM.SetAlignLength(ctx.matmulPerf.GetBaseM());
    return ctx;
}

// ============================================================================
// 四、公共逻辑片段（求解器键控分支与后续算子迁移复用的公共逻辑, P2 迁移时优先对齐;
//     2026-09-15 五算子逐行对比提取, 覆盖率注明各行; §4.24 后由键控分支/P2 迁移消费）
// ============================================================================

// 短长块等长归一（5/5 算子 Adjust 尾部逐字一致）
inline void MergeEqualTiles(CutPlanner &tilingM)
{
    if (tilingM.cutRes.shortTileLen == tilingM.cutRes.longTileLen) {
        tilingM.cutRes.shortTileLen = 0U;
        tilingM.cutRes.numShortTile = 0U;
        tilingM.cutRes.numLongTile++;
    }
}

// 线性度好的 shape（k×n 较大, 通信拟合更可信; 4/5 算子一致, agmm 传字面 true,
// allto_all_matmul 自有离散切分不用）
inline bool IsGoodLinearityShape(const MatmulShapeInfo &mmInfo)
{
    return mmInfo.kValue * mmInfo.nValue >= GOOD_LINEARITY_NK_BYTES * BYTES_PER_MB;
}

// 双向配平长块（3/5 算子一致: mmrs / all_reduce / matmul_allto_all）:
//   短块后置(计算 bound) → 长块 = 反解通信时长(掩盖短块计算);
//   短块前置(通信 bound) → 长块 = 反解计算时长(掩盖短块通信)。
// agmm 不用(其长块掩盖的是 frontMMTime 本地首轮计算); allto_all_matmul 自有模型。
inline void BalanceLongTileLen(FitBasedBalanceCtx &ctx)
{
    if (ctx.tilingM.cutRes.shortTileAtBack) {
        const double targetTime = ctx.matmulPerf.MatmulTime(ctx.tilingM.cutRes.shortTileLen, ctx.mmRankMul);
        ctx.tilingM.cutRes.longTileLen = ctx.commPerf.InverseCommTime(targetTime);
    } else {
        const double targetTime = ctx.commPerf.CommTime(ctx.tilingM.cutRes.shortTileLen);
        ctx.tilingM.cutRes.longTileLen = ctx.matmulPerf.InverseMatmulTime(targetTime, ctx.mmRankMul);
    }
}

// ---- 以下为多算子共有(非全算子)片段: 求解器键控分支与 P2 迁移算子按需调用
//      (调=走公共逻辑, 不调=走各自键控分支), 不设引擎级 bool 开关——片段与键控
//      分支逻辑交错, 引擎级开关无法表达交错顺序且会重演配置汤) ----

// Estimate 骨干（4/5: agmm/mmrs/all_reduce/matmul_allto_all 一致）:
// FindCubeUtil + GetMatmulGradient + 总量估计 + 通算比; shortTileAtBack 判定不在此
// (各算子语义相反: mmrs 系 mm>=comm→后置, allto_all_matmul 是 comm>mm→后置, agmm 恒后置)。
// 总量倍数 ctx.mmRankMul 由 InitCtx 按通信原语单次派生(见 MmRankMul 注释)。
// [已废弃 2026-09-17] mmTimeScale/KC_QUANT×1.43 补偿不再保留——与 A2AMM 独立拟合
// 同根因(老切分尾块引发 matmul 膨胀, 理论失准), 新架构不引入该膨胀, 无需补偿;
// MMA2A 的 E14(80μs 不切门)迁移时同样不带 ×1.43。
struct EstimateTimesResult {
    double totalMatmulTime = 0.0;
    double totalTpTime = 0.0;
};
inline EstimateTimesResult EstimateTotalTimes(FitBasedBalanceCtx &ctx)
{
    ctx.matmulPerf.FindCubeUtil(ctx.commRankCntForMm);
    ctx.matmulPerf.GetMatmulGradient();
    EstimateTimesResult r;
    r.totalMatmulTime = ctx.matmulPerf.MatmulTime(ctx.matmulPerf.mmShapeInfo_.mValue, ctx.mmRankMul);
    // N=1(NoCut 候选口径, §4.37): Estimate 发生在切分前、轮数未知, 按不切保守估通信总量;
    // 切分完成后由 Schedule 尾部按实际轮数精算(ctx.commTimeActualUs)供日志/观察
    r.totalTpTime = ctx.commPerf.TotalCommTime(ctx.matmulPerf.mmShapeInfo_.mValue, 1);
    ctx.ratioCalcComm = (std::max(r.totalTpTime, r.totalMatmulTime) / std::min(r.totalTpTime, r.totalMatmulTime));
    return r;
}

// L2 占用判定（3/3 逐字一致: mmrs/all_reduce/matmul_allto_all, 含 128MB 同值常量）
inline bool ExceedsL2Cache(const MatmulShapeInfo &mmInfo)
{
    constexpr uint64_t L2_CACHE_SIZE = 128UL * BYTES_PER_MB;
    const uint64_t l2UseSize = mmInfo.mValue * mmInfo.kValue * mmInfo.inMatrixADtypeSize +
                               mmInfo.kValue * mmInfo.nValue * mmInfo.inMatrixBDtypeSize;
    return l2UseSize > L2_CACHE_SIZE;
}

// L2 超限鼓励大块: minLen×2（3/5; mmrs 附加 m>minLen×4 守卫 → mGuardMul=4,
// all_reduce/matmul_allto_all 无守卫 → mGuardMul=0; agmm/allto_all_matmul 不用）
inline void ApplyL2MinLenBoost(CutPlanner &tilingM, const MatmulShapeInfo &mmInfo, bool exceedsL2,
                               uint64_t mGuardMul = 0UL)
{
    const bool guardOk = (mGuardMul == 0UL) || (mmInfo.mValue > tilingM.GetMinLen() * mGuardMul);
    if (exceedsL2 && guardOk) {
        tilingM.SetMinLenByMax(tilingM.GetMinLen() * 2U);
    }
}

// 小 shape 整 M 单块装下: minLen 提到 m（2/2 逐字一致: mmrs/matmul_allto_all;
// 判据 m×n ≤ 2×coreNum×baseM×baseN; GetBaseM/N 为非 const 成员故 ctx 取非常量引用）
inline void ApplySmallShapeMinLen(CutPlanner &tilingM, FitBasedBalanceCtx &ctx)
{
    if (ctx.matmulPerf.mmShapeInfo_.mValue * ctx.matmulPerf.mmShapeInfo_.nValue <=
        ctx.matmulPerf.mmShapeInfo_.coreNum * ctx.matmulPerf.GetBaseM() * ctx.matmulPerf.GetBaseN() * 2U) {
        tilingM.SetMinLenByMax(ctx.matmulPerf.mmShapeInfo_.mValue);
    }
}

// 单轮通信量上限整形(CCU 键, §4.37: 自 AGMM 调用点 AdjustCutResultForCCU 四函数链迁入合并):
// HCCL 单次通信调用有 256MB 上限——单块通信量(块长×单行通信字节)超过时按候选表
// {2048,1024,512,256}(大块优先=少轮次)等长重切, 尾块吸收余数, 完全覆盖长短结构
// (超限属极端大 shape, 约束满足优先); 轮次预算 = MAX_COMM_ROUNDS−1(1 轮保留给 scale
// 通道通信, AGMM quant 语义, 与老 HCCL_MAX_TOTAL_TILES=62 一致)。
// 量级安全: 块长×单行字节 ≤ 1e6×5e5 = 5e11 << uint64 上限, 免老代码三段溢出回检;
// 两约束(256MB/轮次预算)冲突时放弃整形保持原切分(老代码 UNSUPPORTED 路径)。
// 流水线总时估计(§4.37.1, 轮数搜索/门控的统一估值器): 切 N 轮的近似总时——不可重叠段在
// 流水线两端(与通算顺序对应), 中段取通信/计算瓶颈侧:
//   通信在前(AG):       T_cut ≈ T_comm(首块) + max(T_comm(其余轮), T_mm − T_mm(末块)) + T_mm(末块)
//   通信在后(RS/AR/A2A): T_cut ≈ T_mm(首块) + max(T_mm − T_mm(首块), T_comm(除末轮)) + T_comm(末轮)
// 单块(含 NoCut 候选)返回 serialTotalTime(全串行基准)。totalMmTime/serialTotalTime 由调用方
// 一次算好传入(搜索循环内不变, 免重复求值)。
inline double EstimateCutTotalTime(FitBasedBalanceCtx &ctx, const CommCutResult &cut, double totalMmTime,
                                   double serialTotalTime)
{
    if (cut.totalTileCnt <= 1U) {
        return serialTotalTime; // 单块即 NoCut 候选
    }
    const uint64_t mValue = ctx.matmulPerf.mmShapeInfo_.mValue;
    const double totalCommTime = ctx.commPerf.TotalCommTime(mValue, cut.totalTileCnt);
    const uint64_t firstLen = (cut.numShortTile > 0U && !cut.shortTileAtBack) ? cut.shortTileLen : cut.longTileLen;
    const uint64_t lastLen = (cut.numShortTile > 0U && cut.shortTileAtBack) ? cut.shortTileLen : cut.longTileLen;
    if (ctx.localMMPosition == LocalMMPosition::FRONT) {
        // 通信在前 + 本地首块: 首块通信不可重叠(远程数据未到), 末段计算收尾
        const double commFirst = ctx.commPerf.CommTime(firstLen);
        const double mmLast = ctx.matmulPerf.MatmulTime(lastLen, ctx.mmRankMul);
        return commFirst + std::max(totalCommTime - commFirst, totalMmTime - mmLast) + mmLast;
    }
    if (ctx.localMMPosition == LocalMMPosition::BACK) {
        // 计算在前 + 本地尾块: 首块计算不可重叠, 末段本地窗口与末块通信并行(取 max)——
        // 本地窗口时长自总计算中抽出归末段(待 MMA2A 迁移实测校准)
        const double mmFirst = ctx.matmulPerf.MatmulTime(firstLen, ctx.mmRankMul);
        const double commLast = ctx.commPerf.CommTime(lastLen);
        const double mmRest = totalMmTime - mmFirst - ctx.backMMTime;
        return mmFirst + std::max(mmRest, totalCommTime - commLast) + std::max(commLast, ctx.backMMTime);
    }
    const double mmFirst = ctx.matmulPerf.MatmulTime(firstLen, ctx.mmRankMul);
    const double commLast = ctx.commPerf.CommTime(lastLen);
    return mmFirst + std::max(totalMmTime - mmFirst, totalCommTime - commLast) + commLast;
}

inline void ApplySingleRoundCommLimit(CutPlanner &tilingM, const HCCLCommModel &commPerf, uint64_t volumeMul,
                                      uint64_t mValue)
{
    constexpr uint64_t MAX_ROUND_COMM_BYTES = 268435456;               // HCCL 单次调用通信量上限 (256MB)
    constexpr uint64_t TILE_LEN_CANDIDATES[] = {2048, 1024, 512, 256}; // 等长重切候选(大值优先=少轮次)
    constexpr uint64_t CANDIDATE_ALIGN = 256;                          // 候选全超限时的兜底对齐粒度
    // 单行通信字节数(物理量, 256MB 是 HCCL 单次调用的真实传输上限——须含份数, 与 k 的标定口径
    // 融合无关): 通信矩阵另一维 × 每元素字节 / 子字节折算 × 单次调用物理倍数(CommProfile.commVolumeMul)
    const uint64_t rowBytes = commPerf.commTypeInfo_.commMatrixLen * commPerf.commTypeInfo_.commDtypeSize * volumeMul /
                              commPerf.commTypeInfo_.commDtypeSizeExpansionFraction;
    if (rowBytes == 0) {
        return; // 防御: 单行为零无法约束
    }
    if (tilingM.cutRes.longTileLen * rowBytes <= MAX_ROUND_COMM_BYTES) {
        return; // 未超限: 保持长短配平结构
    }
    const uint64_t maxTileLen = MAX_ROUND_COMM_BYTES / rowBytes; // 满足上限的最大块长
    if (maxTileLen < CANDIDATE_ALIGN) {
        return; // 单行通信量过大, 任何对齐块都超限——放弃整形
    }
    // 优选候选(降序第一个 ≤ maxTileLen); 全部超限则 floor 对齐兜底
    uint64_t tileLen = 0;
    for (const uint64_t cand : TILE_LEN_CANDIDATES) {
        if (cand <= maxTileLen) {
            tileLen = cand;
            break;
        }
    }
    if (tileLen == 0) {
        tileLen = maxTileLen / CANDIDATE_ALIGN * CANDIDATE_ALIGN;
    }
    // 轮次预算: 候选块太小块数超预算时, 反向取满足预算的最小块(仍优先候选表, 降序第一个 ≥)
    const uint64_t roundsBudget = MAX_COMM_ROUNDS - 1;
    if ((mValue + tileLen - 1) / tileLen > roundsBudget) {
        const uint64_t minLen =
            ((mValue + roundsBudget - 1) / roundsBudget + CANDIDATE_ALIGN - 1) / CANDIDATE_ALIGN * CANDIDATE_ALIGN;
        uint64_t pick = minLen;
        for (const uint64_t cand : TILE_LEN_CANDIDATES) {
            if (cand >= minLen) {
                pick = cand;
                break;
            }
        }
        if (pick > maxTileLen) {
            return; // 256MB 上限与轮次预算冲突——放弃整形
        }
        tileLen = pick;
    }
    // 等长重切(mValue > maxTileLen ≥ tileLen 保证主块数 ≥ 1): 主块等长 + 尾块余数, 短块位置保持
    tilingM.cutRes.longTileLen = tileLen;
    tilingM.cutRes.numLongTile = mValue / tileLen;
    tilingM.cutRes.shortTileLen = mValue - tilingM.cutRes.numLongTile * tileLen;
    tilingM.cutRes.numShortTile = (tilingM.cutRes.shortTileLen > 0) ? 1 : 0;
    tilingM.cutRes.totalTileCnt = tilingM.cutRes.numLongTile + tilingM.cutRes.numShortTile;
}

// 键控维度: LocalMMPosition(§4.38——取代通信原语的结构键控)。
// FRONT: 本地 mm 块在流水线首端(本卡数据已在手, 不等通信)——启动端被本地块免费占据,
//        短块退到尾部专职快速收尾(恒后置), 长块通信掩盖本地首块计算(E5/E19, 原 AG 分支);
// BACK:  本地 mm 块在流水线尾端(本地保留份无需发出、无通信 deadline)——收尾端被本地块
//        占据(与末块通信重叠), 短块前置专职启动通信管道, 长块通信掩盖本地尾块计算(镜像 E5/E19);
// NONE:  无本地块——短块作流水线垫片, 按 bound 判定放首/尾(E4 通用规律)。
inline CommCutResult Schedule(FitBasedBalanceCtx &ctx)
{
    const bool isLocalMMFront = (ctx.localMMPosition == LocalMMPosition::FRONT);
    const bool isLocalMMBack = (ctx.localMMPosition == LocalMMPosition::BACK);

    // ---- ① Estimate: 公共骨干(E1) + E2 总量倍数(ctx.mmRankMul, InitCtx 已派生) ----
    const auto estimate = EstimateTotalTimes(ctx);
    if (isLocalMMFront || isLocalMMBack) {
        // 本地块独立计算的 L2 利用率与稳态不同(单份数据、流水未满, 0.6 指数口径;
        // FRONT=首端 E5, BACK=尾端镜像——同一物理量: 本地窗口的计算时长)
        const double localMMUtil = ctx.matmulPerf.FindCubeUtil(ctx.matmulPerf.mmShapeInfo_.mValue, 1,
                                                               MatmulPerfConst::LOCAL_MM_UTIL_EXPONENT_COEFFICIENT);
        const double localMMTime = (std::abs(localMMUtil) > 1e-10) ?
                                       ctx.matmulPerf.MatmulTime(ctx.matmulPerf.mmShapeInfo_.mValue, 1) / localMMUtil :
                                       ctx.matmulPerf.MatmulTime(ctx.matmulPerf.mmShapeInfo_.mValue, 1);
        if (isLocalMMFront) {
            ctx.frontMMTime = localMMTime;
            ctx.tilingM.cutRes.shortTileAtBack = true; // 本地块占启动端, 短块专职收尾(恒后置)
        } else {
            ctx.backMMTime = localMMTime;
            ctx.tilingM.cutRes.shortTileAtBack = false; // 本地块占收尾端, 短块专职启动(恒前置)
        }
    } else if (ctx.order == CommComputeOrder::COMPUTE_FIRST) {
        // 先行段为瓶颈 → 短块后置(短块放另一端, 由长块掩盖)
        ctx.tilingM.cutRes.shortTileAtBack = (estimate.totalMatmulTime >= estimate.totalTpTime);
    } else {
        ctx.tilingM.cutRes.shortTileAtBack = (estimate.totalTpTime > estimate.totalMatmulTime);
    }

    // ---- ② SetShort: 短块落位(minTileLen 为短块长度种子; NoCut 几何门在落位后) ----
    // [P2 接线] E11-E13(L2/小 shape)随各算子迁移接入; §4.37 废弃条目(E9/E10/E14-E16/E20/E26/E29)不再接线
    ctx.tilingM.cutRes.shortTileLen = ctx.tilingM.GetMinLen();
    ctx.tilingM.cutRes.numShortTile = 1U;

    if (ctx.tilingM.totalLen < ctx.tilingM.cutRes.shortTileLen * 2) { // do not cut if mValue is too small
        ctx.tilingM.NoCutTiling();
        return ctx.tilingM.cutRes;
    }

    // ---- ③ SetLong: 本地窗口配平(FRONT/BACK) / E18 公共双向配平(NONE) ----
    if (isLocalMMFront) {
        // 首长块的通信恰好掩盖本地首块计算(E19)
        ctx.tilingM.cutRes.longTileLen = ctx.commPerf.InverseCommTime(ctx.frontMMTime);
    } else if (isLocalMMBack) {
        // 末长块的通信恰好掩盖本地尾块计算(镜像 E19: 本地保留份在收尾端与末块通信重叠)
        ctx.tilingM.cutRes.longTileLen = ctx.commPerf.InverseCommTime(ctx.backMMTime);
    } else {
        BalanceLongTileLen(ctx); // 乘数 ctx.mmRankMul(老代码 A2A 曾误用管线系数 rankTileNum)
    }
    // ⑤ 轮数搜索的恢复点: ③ 后形态(短块落位 + 长块反解值)。反解与轮数无关——
    // 每档仅在收紧的轮次预算下重跑 ④ 离散化, 无需重算配平
    const CommCutResult afterLong = ctx.tilingM.cutRes;

    // ---- ④ Adjust: E24(goodLinearity: AG 恒 true) + 离散化 + E25 等长归一 ----
    // 有本地块的形态恒信线性段(行为基线, 原 AG 恒 true); 离散化保护含本地块的端部结构
    const bool goodLinearity =
        (isLocalMMFront || isLocalMMBack) ? true : IsGoodLinearityShape(ctx.matmulPerf.mmShapeInfo_);
    ctx.tilingM.FitTileLengthDiscrete(goodLinearity, isLocalMMFront || isLocalMMBack);
    MergeEqualTiles(ctx.tilingM);

    // ---- ⑤ 轮数搜索 + 切分收益门控(§4.37.1): 把每轮启动开销纳入切分决策 ----
    // 切分不改变总通信数据量(k·X 不变), 只多付启动税 (N−1)·b, 买到的是通信-计算重叠。
    // 流水线总时近似(不可重叠段在流水线两端, 按 LocalMMPosition/顺序三形态, 见 EstimateCutTotalTime):
    //   FRONT(本地首块): T_cut ≈ T_comm(首块) + max(T_comm(其余轮), T_mm − T_mm(末块)) + T_mm(末块)
    //   BACK(本地尾块):  T_cut ≈ T_mm(首块) + max(T_mm(其余), T_comm(除末轮)) + max(T_comm(末轮), 本地窗口)
    //   NONE(通信在后):  T_cut ≈ T_mm(首块) + max(T_mm − T_mm(首块), T_comm(除末轮)) + T_comm(末轮)
    // 不切 T_serial = T_comm(1 轮) + T_mm(全串行)。
    // 搜索动机(2026-09-21 用户指出): 配平产出的轮数 N₀ 是"配平最优"而非"全局最优"——
    // 通信重时 T(N) 随轮数单调增(启动税吃收益)、计算重时单调降(至首块通信≈b 触底),
    // 最优轮数常在 [1, N₀] 中间, 二值判定(N₀ vs 1)会错过中间解。
    // 搜索实现: budget 线性枚举 [2, N₀)(每档恢复 ③ 后形态 + 收紧轮次预算 + 重跑 ④,
    // 长短结构随预算重新整形), T_cut 择优; 全部候选 ≥ T_serial → NoCut。
    // 成本: N₀ ≤ MAX_COMM_ROUNDS(63), 每档纯整数重切 O(块数)——host 侧微秒级。
    const uint64_t mValue = ctx.matmulPerf.mmShapeInfo_.mValue;
    const double serialTotalTime = ctx.commPerf.TotalCommTime(mValue, 1U) + estimate.totalMatmulTime;
    CommCutResult bestCut = ctx.tilingM.cutRes; // 初次切分(N₀ 轮, 配平产出)即候选
    double bestTime = EstimateCutTotalTime(ctx, bestCut, estimate.totalMatmulTime, serialTotalTime);
    for (uint64_t budget = 2U; budget < bestCut.totalTileCnt; ++budget) {
        ctx.tilingM.cutRes = afterLong;    // 恢复 ③ 后形态(反解值复用)
        ctx.tilingM.SetMaxTileCnt(budget); // 收紧轮次预算
        ctx.tilingM.FitTileLengthDiscrete(goodLinearity, isLocalMMFront || isLocalMMBack);
        MergeEqualTiles(ctx.tilingM);
        const double cutTime = EstimateCutTotalTime(ctx, ctx.tilingM.cutRes, estimate.totalMatmulTime, serialTotalTime);
        if (cutTime < bestTime) {
            bestTime = cutTime;
            bestCut = ctx.tilingM.cutRes;
        }
    }
    ctx.tilingM.SetMaxTileCnt(MAX_COMM_ROUNDS); // 恢复默认轮次预算
    if (bestTime >= serialTotalTime) {
        ctx.tilingM.NoCutTiling(); // 全部候选不敌串行——启动税买不回重叠收益
    } else {
        ctx.tilingM.cutRes = bestCut;
    }

    // ---- ⑥ 单轮通信量上限整形(CCU 键, §4.37): HCCL 单次调用 256MB 上限, 超限块等长重切 ----
    if (ctx.commEngine == TilingCommEngine::CCU) {
        ApplySingleRoundCommLimit(ctx.tilingM, ctx.commPerf, ctx.commVolumeMul, ctx.matmulPerf.mmShapeInfo_.mValue);
    }

    // §4.37: 按最终切分的实际轮数精算总通信时(N 轮 vs NoCut 候选 1 轮), 差值即切分
    // 多付的启动税 (N−1)·b——诊断输出供 op 侧日志
    ctx.commTimeActualUs =
        ctx.commPerf.TotalCommTime(ctx.matmulPerf.mmShapeInfo_.mValue, ctx.tilingM.cutRes.totalTileCnt);
    ctx.commTimeNoCutUs = ctx.commPerf.TotalCommTime(ctx.matmulPerf.mmShapeInfo_.mValue, 1);
    return ctx.tilingM.cutRes;
}

#endif // __APACE_FIT_BALANCE_TILING_H__
