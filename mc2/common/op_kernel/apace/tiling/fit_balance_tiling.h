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
 * 零特有代码架构(2026-09-17 §4.24 裁定): 钩子机制已废除, 算子差异按
 * (通信原语 CommPrimitive, 通算顺序 CommComputeOrder, 通信引擎 TilingCommEngine, dtype)
 * 键控分支内化于引擎; 阈值经 FitBalanceCalibration.tunables(FitTunables) 注入。
 * 算子侧 = CALIBRATION 实例 + 枚举入参 + 一行调用, 无任何算子特有代码。
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

enum class CommPrimitive {
    ALL_REDUCE, // default
    ALL_GATHER,
    REDUCE_SCATTER,
    ALL_TO_ALL,
    REDUCE_SCATTER_VIA_ALL_TO_ALL,
    ALL_REDUCE_VIA_TWO_SHOT,
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

struct L2UtilTiers {
    uint64_t sizePartL2;
    uint64_t sizeFullL2;
    double utilPartL2;
    double utilFullL2;
    double utilOutL2;
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
    CommPrimitive commPrimitive;
    CommMeshType commMethod;
    CommTopology topoType;
    uint64_t maxStepSize; // number of steps in a full round of communication
    uint64_t commDtypeSizeExpansionFraction;
};

// 通信时间线性拟合参数(2 字段, §4.29 分派落地后 miss 回退路径整体废弃——标定命中是
// 引擎唯一入口, 拟合模型收敛为纯线性 y = slope·x + offset; 12→8→2 两轮裁剪史见 §4.28/§4.29)
struct CommFitParams {
    double linearSlopeUsPerMb; // 斜率 (us/MB), 带宽项
    double linearOffsetUs;     // 截距 (us), 每轮同步固定开销
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

// 通信轮次硬上限（来源: hccl_formulaic_tiling.h 的 MAX_COMM_ROUNDS, 受 FFTS 队列大小限制）
constexpr uint64_t MAX_COMM_ROUNDS = 16;

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

// ====== 键控分支的可调阈值(载体: FitBalanceCalibration.tunables, 缺省=现网值) ======
// 机制随引擎键控分支下沉, 阈值留标定——同键算子将来分叉只改各自 CALIBRATION(§4.22 裁定)
struct FitTunables {
    // E9(AG 键): 短块减半鼓励门——通算平衡且半块双耗时均超阈, 才减半对齐长度(鼓励多切)
    double halveEncourageMmUs = 40;   // 半块 matmul 耗时下限 (us)
    double halveEncourageCommUs = 40; // 半块通信耗时下限 (us)
    // E10(AR 键, P2-MMAR 接线): 切分开销绝对值反解 minLen 的双耗时下限
    double minCutCommUs = 100; // 通信耗时下限 (us)
    double minCutMmUs = 60;    // matmul 耗时下限 (us)
};

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
    // 实际参与通信的数据卡数(旧代码原名 lookUpTileNum_, "数据拟合查表参数"; 语义经开发人员确认):
    // AG/RS(含 via-A2A)=rankDim——聚合/分发时全部卡份数据都过网(AG 自己的份也要广播给别人);
    // AR/A2A=1——两卡口径按单份拟合(A2A 的拟合 x 轴取单卡发送量, 标定口径使然, 并非字面上
    // 只有一卡数据过网)。进 CommTime/InverseCommTime 的字节数换算。
    uint64_t commRankCnt_ = 1;
    // 被 mm 计算依赖的通信数据卡数(旧代码原名 rankTileNum_, 由 GetRankTileNum()/
    // GetFullMeshRankTileNum() 派生; 语义经开发人员确认): 同一批通信数据, 通信完之后(AG,
    // comm→calc)或之前(RS/A2A, calc→comm)参与本卡计算的部分——
    //   AG=rankDim-1: 远程份经通信到达后参与计算, 自己的份不经通信(本地 front 块, E5 单独按 1 建模);
    //   RS/A2A=rankDim: 全部卡份先算后发(收到侧依赖全部卡的计算产物);
    //   AR=1: two-shot 每轮一份。
    // 与 commRankCnt_(通信视角全量, AG=rankDim/A2A=1)、ctx.mmRankMul(mm 总行数倍数,
    // AG·RS=rankDim/A2A·AR=1) 是三个独立标定的口径系数——取值表是标定历史事实而非同一物理量
    // 的推导, 勿混用勿"统一"。
    uint64_t commRankCntForMm_ = 1;

    // Constructor(soc 恒 950; commFitParams 由 InitCtx 按 rankDim 从标定选定的行, miss=零值行)
    HCCLCommModel(uint32_t inputRankDim, CommPrimitive inputKernelType, CommFitParams commFitParams,
                  CommTopology topoType = CommTopology::STANDARD_CARD)
    {
        commTypeInfo_.commPrimitive = inputKernelType; // 区分哪个MC2算子
        // 通信域卡数(op 侧来自 GetRankSize(group)), 下限钳位 2, 三层原因:
        //   ① 防 uint64 下溢: AG 远程流转份数 = rankDim-1, rankDim=0 时下溢;
        //   ② 防退化: 单卡无集合通信可建模, 拟合表均按 ≥2 卡标定, 退化输入按最接近的 2 卡建模;
        //   ③ 历史防御: 老基类服务全家族, rankDim 来源多样(attr/group/shape 推导)不可靠, 入口统一兜底
        commTypeInfo_.rankDim = std::max(static_cast<uint64_t>(inputRankDim), MIN_RANK_DIM);
        commTypeInfo_.commDtypeSizeExpansionFraction = 1;
        commTypeInfo_.maxStepSize = 1;
        InitPrimitiveParams();
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

    // ---- 构造期按通信原语初始化两个建模口径 + 标定查表 ----
    // commRankCnt_/commRankCntForMm_: 两个通信数据卡数口径(见成员定义处取值表);
    // commTimeFactor_: 恒 1(§4.31)——老公式化家族的 8-die AR 折算桥接(plain RS 专用, 无 A 家族
    //   消费者且 950 表无标定行)已随死分支删除, A 家族全部原语的标定行均为直接口径。
    // 标定查表按 rankDim 匹配(首个命中行; soc/commPrimitive 为算子实例常量, 已在构造参数区分)。
    //   §4.29 分派后 miss 由 op 层走 COF——未命中置零值行(CommTime 恒 0 → NoCut), 防御而非回退。
    void InitPrimitiveParams()
    {
        commTypeInfo_.commMethod = CommMeshType::FULL_MESH;
        // AG / RS-via-A2A: 聚合/分发量即拟合口径(一轮线上流动 rankDim 份, 标定行直接可用);
        // 其余(AR/A2A): 每轮一份, 均为默认值 commRankCnt_=1 / commTimeFactor_=1, 无需设置
        if ((commTypeInfo_.commPrimitive == CommPrimitive::ALL_GATHER) ||
            (commTypeInfo_.commPrimitive == CommPrimitive::REDUCE_SCATTER_VIA_ALL_TO_ALL)) {
            commRankCnt_ = commTypeInfo_.rankDim;
        }
        commRankCntForMm_ =
            (commTypeInfo_.commPrimitive == CommPrimitive::ALL_REDUCE) ?
                1 :
                ((commTypeInfo_.commPrimitive == CommPrimitive::ALL_GATHER) ? (commTypeInfo_.rankDim - 1) :
                                                                              commTypeInfo_.rankDim);
    }

    // 性能拟合函数: 纯线性 y = slope·x + offset (x=通信数据量 MB), 再除时间折算因子
    double CommTime(uint64_t mSize) const
    {
        const double commDataSize =
            static_cast<double>(mSize) * commTypeInfo_.commMatrixLen * commRankCnt_ * GetRealDtypeSizes();
        const double tmpSize = commDataSize / BYTES_PER_MB;
        return (commEstimatePar_.linearSlopeUsPerMb * tmpSize + commEstimatePar_.linearOffsetUs) / commTimeFactor_;
    }

    // 纯线性反解: x = (t·factor - offset) / slope, 再换算回 M 行数
    uint64_t InverseCommTime(double targetTime) const
    {
        const double tmpSize =
            (targetTime * commTimeFactor_ - commEstimatePar_.linearOffsetUs) / commEstimatePar_.linearSlopeUsPerMb;
        return static_cast<uint64_t>(tmpSize * BYTES_PER_MB /
                                     (commTypeInfo_.commMatrixLen * commRankCnt_ * GetRealDtypeSizes()));
    }
};

// ---- 1.2 matmul 模型（HCCL 算力/梯度模型, 自由长度分支专用, 950）----
// 由旧 MatmulPerformanceModel + MatmulPerformanceArch35 合并精简(2026-09-15, 类名已去 arch35),
// 剥离 310P/910_B4 分支与 4 参 FindCubeUtil(含其独占的 K 对齐修正链)——
// 5 个消费算子经 grep 证实仅用 1 参 FindCubeUtil/FindCubeUtilByL2Usage。

namespace MatmulPerfConst {
constexpr double COMPUTES_PER_CYCLE = 4096; // 基础算力(查表未命中回退, 标定头回填同值)
// cycle per micro second (950)
constexpr double CYCLE_PER_MICRO_SEC = 1.65 * BYTES_PER_KB;
// end: cycle per micro second
constexpr double MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL = 1.4;
constexpr double MNVALUE_THRESHOLD = 820;
constexpr double KVALUE_THRESHOLD = 1280;
constexpr double FRONT_UTIL_EXPONENT_COEFFICIENT = 0.6;
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
    MatmulPerfModel(const FitBalanceTilingArgs &args, double computesPerCycle, L2UtilTiers l2Params,
                    MatmulBaseBlock baseBlock)
        : l2Params_(l2Params)
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
    // L2 档位利用率 + 950 调和平均公式(950 专用模型, 调和分支无条件执行)
    double FindCubeUtilByL2Usage(uint64_t mSize, uint64_t commRankCntForMm, uint64_t *maxTileLen = nullptr)
    {
        const L2UtilTiers &l2EstimatePar = l2Params_; // InitCtx 构造期已按 dtype 选定
        uint64_t tmpBaseM = std::min(mmShapeInfo_.baseM, mSize);
        uint64_t tmpBaseN = mmShapeInfo_.baseN;
        uint64_t mBlockNum = (mSize + tmpBaseM - 1) / tmpBaseM * commRankCntForMm;
        uint64_t nBlockNum = (mmShapeInfo_.nValue + tmpBaseN - 1) / tmpBaseN;
        uint64_t memUsage =
            std::min(mBlockNum, mmShapeInfo_.coreNum) * tmpBaseM + std::min(nBlockNum, mmShapeInfo_.coreNum) * tmpBaseN;
        memUsage = memUsage * mmShapeInfo_.kValue * mmShapeInfo_.inMatrixADtypeSize / BYTES_PER_MB;
        double tmpUtil;
        double diff = 0;
        if (memUsage <= l2EstimatePar.sizePartL2) {
            tmpUtil = l2EstimatePar.utilPartL2; // 0.85x cube utilization rate
            diff = l2EstimatePar.sizePartL2 - memUsage;
        } else if (memUsage <= l2EstimatePar.sizeFullL2) {
            tmpUtil = l2EstimatePar.utilFullL2; // 0.75x cube utilization rate
            diff = l2EstimatePar.sizeFullL2 - memUsage;
        } else {
            tmpUtil = l2EstimatePar.utilOutL2; // 0.65x cube utilization rate
        }
        tmpUtil *= static_cast<double>(std::min(mBlockNum * nBlockNum, mmShapeInfo_.coreNum) /
                                       static_cast<double>(mmShapeInfo_.coreNum));

        /*950 cube利用率计算公式：两个输入为M、N的调和平均数和K。MN调和平均数的阈值为820，K阈值为1280，
        小于这个阈值认为提供负增益，大于这个阈值认为提供正增益，但正增益最高不超过1.4倍。*/
        double mnharmonicMean = static_cast<double>(mSize * commRankCntForMm * mmShapeInfo_.nValue) /
                                static_cast<double>(mSize * commRankCntForMm + mmShapeInfo_.nValue);
        double kExponentiated = std::min(
            MatmulPerfConst::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
            std::pow(static_cast<double>(mmShapeInfo_.kValue), MatmulPerfConst::CUBE_UTIL_EXPONENT_COEFFICIENT) /
                std::pow(MatmulPerfConst::KVALUE_THRESHOLD, MatmulPerfConst::CUBE_UTIL_EXPONENT_COEFFICIENT));
        double mnExponentiated = std::min(
            MatmulPerfConst::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
            std::pow(mnharmonicMean, MatmulPerfConst::FRONT_UTIL_EXPONENT_COEFFICIENT) /
                std::pow(MatmulPerfConst::MNVALUE_THRESHOLD, MatmulPerfConst::FRONT_UTIL_EXPONENT_COEFFICIENT));
        double result = MatmulPerfConst::AVERAGE_CUBE_UTIL * kExponentiated * mnExponentiated;
        tmpUtil = std::min(MatmulPerfConst::MAX_CUBE_UTIL, result);

        if (maxTileLen != nullptr) {
            *maxTileLen = mmShapeInfo_.mValue;
            uint64_t tmpSize = std::min(mBlockNum + commRankCntForMm, mmShapeInfo_.coreNum) -
                               std::min(mBlockNum, mmShapeInfo_.coreNum);
            tmpSize = tmpSize * mmShapeInfo_.inMatrixADtypeSize * tmpBaseM * mmShapeInfo_.kValue / BYTES_PER_MB;
            if (tmpSize > diff && memUsage <= l2EstimatePar.sizeFullL2 && nBlockNum > mmShapeInfo_.coreNum) {
                *maxTileLen = (mSize + mmShapeInfo_.baseM - 1) / mmShapeInfo_.baseM * mmShapeInfo_.baseM;
            }
        }
        return tmpUtil;
    }
    // 单参版本(消费算子钩子唯一使用的入口)
    void FindCubeUtil(uint64_t commRankCntForMm)
    {
        double mnharmonicMean = static_cast<double>(mmShapeInfo_.mValue * commRankCntForMm * mmShapeInfo_.nValue) /
                                static_cast<double>(mmShapeInfo_.mValue * commRankCntForMm + mmShapeInfo_.nValue);
        double kExponentiated = std::min(
            MatmulPerfConst::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
            std::pow(static_cast<double>(mmShapeInfo_.kValue), MatmulPerfConst::CUBE_UTIL_EXPONENT_COEFFICIENT) /
                std::pow(MatmulPerfConst::KVALUE_THRESHOLD, MatmulPerfConst::CUBE_UTIL_EXPONENT_COEFFICIENT));
        double mnExponentiated =
            std::min(MatmulPerfConst::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
                     std::pow(mnharmonicMean, MatmulPerfConst::CUBE_UTIL_EXPONENT_COEFFICIENT) /
                         std::pow(MatmulPerfConst::MNVALUE_THRESHOLD, MatmulPerfConst::CUBE_UTIL_EXPONENT_COEFFICIENT));
        double result = MatmulPerfConst::AVERAGE_CUBE_UTIL * kExponentiated * mnExponentiated;
        cubeUtil_ = std::min(MatmulPerfConst::MAX_CUBE_UTIL, result);
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

private:
    L2UtilTiers l2Params_; // InitCtx 构造期按 dtype 五元组选定的 L2 档位(实例内恒定, 免逐次查表)
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
    void GenerateInitialPartition(bool smallDimAlignUp = false);
    bool SetShortTileLen(bool noCutFlag = false);
    void NoCutTiling();
    void MaxTileCntUniform();

    // 离散切分对齐 (A 家族全部算子的 Adjust 阶段使用)
    void AlignTileSmall();
    void AlignTileLarge();
    void FitTileLengthDiscrete(bool smallDimAlignUp, bool goodLinearityShape, bool localAtFront = false);
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

inline void CutPlanner::GenerateInitialPartition(bool smallDimAlignUp)
{
    cutRes.longTileLen = std::max(cutRes.longTileLen, cutRes.shortTileLen);
    cutRes.longTileLen = std::min(cutRes.longTileLen, totalLen - cutRes.shortTileLen);
    uint64_t newTileLen = cutRes.longTileLen / tileArgs.mAlignLen * tileArgs.mAlignLen;
    // 0.5 is half of mAlignLen
    double alignRatio = ALIGN_TOL_CALC_BOUND;
    if (!cutRes.shortTileAtBack) {
        alignRatio = ALIGN_TOL_COMM_BOUND;
    }

    bool alignUpFlag = ((cutRes.longTileLen - newTileLen) > (tileArgs.mAlignLen * alignRatio)) || smallDimAlignUp;
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

inline void CutPlanner::FitTileLengthDiscrete(bool smallDimAlignUp, bool goodLinearityShape, bool localAtFront)
{
    // 流水配平，找到初始切分长度和个数
    GenerateInitialPartition(smallDimAlignUp);

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
    if (cutRes.totalTileCnt == 2U && !localAtFront) {
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
    CommPrimitive commPrimitive; // 键控维度: 通信原语
    CommComputeOrder order;      // 键控维度: 通算顺序
    FitTunables tunables;        // 键控分支阈值(自 Calibration.tunables 拷贝)
    uint64_t commRankCntForMm = 1; // 派生: commPerf.commRankCntForMm_(mm 工作集 rank 份数, 取值表见模型成员注释)
    uint64_t mmRankMul = 1;     // 派生: MmRankMul(ctx)(本卡 mm 总行数的 rank 倍数, 见该函数注释)
    double ratioCalcComm = 1.0; // 过程量: Estimate 骨干设置的通算比(键控分支消费)
    double frontMMTime = 0; // 过程量(AG 键): 本地首轮计算耗时(本卡 A 块已在手, 首块无通信依赖)
};

// 本卡 mm 总行数的 rank 倍数(通信原语语义派生; InitCtx 单次赋值进 ctx.mmRankMul, 消费方读字段。
// 旧代码无独立变量——隐式散在各算子钩子对 MatmulTime(m, rankDim_)/MatmulTime(m, 1) 的调用里,
// §4.11 五算子逐行对比时显名提取):
//   AG: mValue 为聚合前本卡份, 聚合后每卡算全量 m×rankDim 行;
//   RS: mValue 已被除为 scatter 保留份, mm 算除之前的全量 orgM = m×rankDim 行;
//   AR/A2A: mm 即本卡 1 份 m 行(规约/分发在原语语义内)。
inline uint64_t MmRankMul(const FitBasedBalanceCtx &ctx)
{
    return (ctx.commPrimitive == CommPrimitive::ALL_GATHER || ctx.commPrimitive == CommPrimitive::REDUCE_SCATTER ||
            ctx.commPrimitive == CommPrimitive::REDUCE_SCATTER_VIA_ALL_TO_ALL) ?
               ctx.commPerf.rankDim() :
               1;
}
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
//   - 通信行为纯线性拟合(客户标定产出即 斜率+截距 两个数)。
// ============================================================================

struct FitBalanceCalibration {
    static constexpr size_t MAX_COMM_FIT_ROWS = 4;   // 已标定通信形态数上限
    static constexpr size_t MAX_CUBE_CALC_ROWS = 4;  // matmul 算力档位数上限
    static constexpr size_t MAX_L2_PARAM_ROWS = 4;   // L2 档位数上限
    static constexpr size_t MAX_BASE_BLOCK_ROWS = 2; // 最优基本块档位数上限

    // —— 通信拟合（单引擎）——
    struct CommFit {
        struct Row {
            uint32_t rankDim;    // 卡数(4=标卡4P, 8=8P, ...)
            double linearSlope;  // 线性段斜率 (us/MB)
            double linearOffset; // 线性段截距 (us), 含每轮同步固定开销
        };
        Row rows[MAX_COMM_FIT_ROWS] = {}; // 已标定行, rankDim=0 表示未填(哨兵截断)
    };
    CommFit aicpu; // AICPU 引擎(comm_mode != "ccu")
    CommFit ccu;   // CCU 引擎(comm_mode == "ccu")

    // —— matmul 性能拟合（对仗 CommFit; 字段序: 用户常改字段在前, 平台默认在后——聚合初始化
    //    可省尾部, 省略的尾部字段自动取 NSDMI 默认值）——
    struct MatmulFit {
        struct CubeCalc {
            MatmulDtypeKind calcType;
            bool isFp4;              // FP16 / QUANT / ANTI_QUANT
            uint64_t aBytes;         // 输入 A 每元素字节数
            uint64_t bBytes;         // 输入 B 每元素字节数
            uint64_t cBytes;         // 输出 C 每元素字节数
            double computesPerCycle; // 有效算力 (MACs/cycle, 含 cube 利用率)
        };
        struct L2Param {
            MatmulDtypeKind calcType;
            bool isFp4;
            uint64_t aBytes;
            uint64_t bBytes;
            uint64_t cBytes;
            L2UtilTiers params;
        };
        CubeCalc cubeCalc[MAX_CUBE_CALC_ROWS] = {};
        L2Param l2Param[MAX_L2_PARAM_ROWS] = {};
        MatmulBaseBlock baseBlock = {128, 256, 64};                // matmul 最优基本块(单值)
        double defaultComputesPerCycle = 4096.0;                   // 算力档回退(通常勿动)
        L2UtilTiers defaultL2Param = {128, 192, 0.85, 0.75, 0.65}; // L2 档回退(通常勿动)
    };
    MatmulFit mmFit;
    FitTunables tunables; // 键控分支阈值(缺省=现网值, 通常勿动; 尾部可整段省略)
};

// 统一入口: 算子侧只传大结构体标定数据 + 键控枚举(原语/顺序/引擎; soc 恒 950)
inline FitBasedBalanceCtx InitCtx(const FitBalanceTilingArgs &args, CommPrimitive commPrimitive, CommComputeOrder order,
                                  TilingCommEngine commEngine, const FitBalanceCalibration &calib,
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
            commFitParams = {commRows[i].linearSlope, commRows[i].linearOffset};
            break;
        }
    }
    // matmul 行: 按"计算形态"匹配算力行与 L2 档行——形态键 = (量化类别, fp4 打包, A/B/C 字节数)
    // 五个维度合起来才确定一种计算形态(只看 dtype 无法区分 fp4/fp8 的 1/1/2 同宽), 键推导与模型
    // SetCalcType 同源
    const auto isQuant = (args.aType == matmul_tiling::DataType::DT_HIFLOAT8) ||
                         (args.aType == matmul_tiling::DataType::DT_FLOAT8_E4M3FN) ||
                         (args.aType == matmul_tiling::DataType::DT_FLOAT8_E5M2) ||
                         (args.aType == matmul_tiling::DataType::DT_INT8) ||
                         (args.geAType == ge::DataType::DT_FLOAT4_E2M1);
    const MatmulDtypeKind calcType = isQuant ? MatmulDtypeKind::QUANT : MatmulDtypeKind::FP16;
    const bool isFp4 = (args.geAType == ge::DataType::DT_FLOAT4_E2M1);
    const auto matchesCalcProfile = [&](const auto &row) {
        return (row.calcType == calcType) && (row.isFp4 == isFp4) && (row.aBytes == args.inputDtypeSize) &&
               (row.bBytes == args.inputDtypeSize) && (row.cBytes == args.outputDtypeSize);
    };
    double computesPerCycle = calib.mmFit.defaultComputesPerCycle;
    for (size_t i = 0; i < FitBalanceCalibration::MAX_CUBE_CALC_ROWS; ++i) {
        if (calib.mmFit.cubeCalc[i].aBytes == 0) { // 哨兵截断
            break;
        }
        if (matchesCalcProfile(calib.mmFit.cubeCalc[i])) {
            computesPerCycle = calib.mmFit.cubeCalc[i].computesPerCycle;
            break;
        }
    }
    L2UtilTiers l2Params = calib.mmFit.defaultL2Param;
    for (size_t i = 0; i < FitBalanceCalibration::MAX_L2_PARAM_ROWS; ++i) {
        if (calib.mmFit.l2Param[i].aBytes == 0) { // 哨兵截断
            break;
        }
        if (matchesCalcProfile(calib.mmFit.l2Param[i])) {
            l2Params = calib.mmFit.l2Param[i].params;
            break;
        }
    }
    FitBasedBalanceCtx ctx{MatmulPerfModel(args, computesPerCycle, l2Params, calib.mmFit.baseBlock),
                           HCCLCommModel(args.rankDim, commPrimitive, commFitParams, topoType),
                           CutPlanner(args),
                           commPrimitive,
                           order,
                           calib.tunables,
                           1,
                           1,
                           1.0};
    ctx.tilingM.SetMaxTileCnt(
        MAX_COMM_ROUNDS); // Due to the limited FFTS queue size, a maximum of 16 rounds can be cut.
    ctx.commRankCntForMm = ctx.commPerf.commRankCntForMm_;
    ctx.mmRankMul = MmRankMul(ctx);

    // ---- 通算胶水(键控, 原散在各算子构造函数, §4.24 裁定随引擎下沉) ----
    // D4 通信矩阵另一维: AG/A2A-in 沿 K 轴聚散 → kValue; RS/AR/A2A-out 沿 N 轴 → nValue
    const bool commOnKAxis = (commPrimitive == CommPrimitive::ALL_GATHER) ||
                             ((commPrimitive == CommPrimitive::ALL_TO_ALL) && (order == CommComputeOrder::COMM_FIRST));
    ctx.commPerf.SetCommShapeLen(commOnKAxis ? ctx.matmulPerf.mmShapeInfo_.kValue : ctx.matmulPerf.mmShapeInfo_.nValue);
    // D5+D6 通信承载 dtype 与 fp4 子字节建模(两件独立的事, 原耦合在 AG 构造函数单分支内, §4.33 解耦):
    // ① 通信 dtype 按承载侧: AG/A2A-in 传 A 侧数据, 其余传 C 侧——fp4 只可能是输入 A 的 dtype,
    //    C 侧(计算输出)永远不是 fp4, 故子字节建模仅 A 侧承载时需要;
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
    if (ctx.commPerf.commEstimatePar_.linearSlopeUsPerMb == 0.0) {
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

// 双向配平长块（3/5 算子一致: rs_v2 / all_reduce / matmul_allto_all）:
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

// Estimate 骨干（4/5: agmm/rs_v2/all_reduce/matmul_allto_all 一致）:
// FindCubeUtil + GetMatmulGradient + 总量估计 + 通算比; shortTileAtBack 判定不在此
// (各算子语义相反: rs_v2 系 mm>=comm→后置, allto_all_matmul 是 comm>mm→后置, agmm 恒后置)。
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
    r.totalTpTime = ctx.commPerf.CommTime(ctx.matmulPerf.mmShapeInfo_.mValue);
    ctx.ratioCalcComm = (std::max(r.totalTpTime, r.totalMatmulTime) / std::min(r.totalTpTime, r.totalMatmulTime));
    return r;
}

// L2 占用判定（3/3 逐字一致: rs_v2/all_reduce/matmul_allto_all, 含 128MB 同值常量）
inline bool ExceedsL2Cache(const MatmulShapeInfo &mmInfo)
{
    constexpr uint64_t L2_CACHE_SIZE = 128UL * BYTES_PER_MB;
    const uint64_t l2UseSize = mmInfo.mValue * mmInfo.kValue * mmInfo.inMatrixADtypeSize +
                               mmInfo.kValue * mmInfo.nValue * mmInfo.inMatrixBDtypeSize;
    return l2UseSize > L2_CACHE_SIZE;
}

// L2 超限鼓励大块: minLen×2（3/5; rs_v2 附加 m>minLen×4 守卫 → mGuardMul=4,
// all_reduce/matmul_allto_all 无守卫 → mGuardMul=0; agmm/allto_all_matmul 不用）
inline void ApplyL2MinLenBoost(CutPlanner &tilingM, const MatmulShapeInfo &mmInfo, bool exceedsL2,
                               uint64_t mGuardMul = 0UL)
{
    const bool guardOk = (mGuardMul == 0UL) || (mmInfo.mValue > tilingM.GetMinLen() * mGuardMul);
    if (exceedsL2 && guardOk) {
        tilingM.SetMinLenByMax(tilingM.GetMinLen() * 2U);
    }
}

// 小 shape 整 M 单块装下: minLen 提到 m（2/2 逐字一致: rs_v2/matmul_allto_all;
// 判据 m×n ≤ 2×coreNum×baseM×baseN; GetBaseM/N 为非 const 成员故 ctx 取非常量引用）
inline void ApplySmallShapeMinLen(CutPlanner &tilingM, FitBasedBalanceCtx &ctx)
{
    if (ctx.matmulPerf.mmShapeInfo_.mValue * ctx.matmulPerf.mmShapeInfo_.nValue <=
        ctx.matmulPerf.mmShapeInfo_.coreNum * ctx.matmulPerf.GetBaseM() * ctx.matmulPerf.GetBaseN() * 2U) {
        tilingM.SetMinLenByMax(ctx.matmulPerf.mmShapeInfo_.mValue);
    }
}

// 键控维度: isAg = (通信原语 == ALL_GATHER)。AG 语义——本卡 A 块已在手:
// 首块计算无通信依赖(E5), 长块通信掩盖的是本地首轮计算而非短块(E19), 短块恒后置(E4)
inline CommCutResult Schedule(FitBasedBalanceCtx &ctx)
{
    const bool isAg = (ctx.commPrimitive == CommPrimitive::ALL_GATHER);

    // ---- ① Estimate: 公共骨干(E1) + E2 总量倍数(ctx.mmRankMul, InitCtx 已派生) ----
    const auto est = EstimateTotalTimes(ctx);
    if (isAg) {
        // E5: 首轮独立计算的 L2 利用率与聚合后不同(frontUtil 修正)
        const double frontUtil = ctx.matmulPerf.FindCubeUtilByL2Usage(ctx.matmulPerf.mmShapeInfo_.mValue, 1);
        if (std::abs(frontUtil) > 1e-10) {
            ctx.frontMMTime = ctx.matmulPerf.MatmulTime(ctx.matmulPerf.mmShapeInfo_.mValue, 1) / frontUtil;
        } else {
            ctx.frontMMTime = ctx.matmulPerf.MatmulTime(ctx.matmulPerf.mmShapeInfo_.mValue, 1);
        }
        ctx.tilingM.cutRes.shortTileAtBack = true; // E4: AG 恒后置
    } else if (ctx.order == CommComputeOrder::COMPUTE_FIRST) {
        // 先行段为瓶颈 → 短块后置(短块放另一端, 由长块掩盖)
        ctx.tilingM.cutRes.shortTileAtBack = (est.totalMatmulTime >= est.totalTpTime);
    } else {
        ctx.tilingM.cutRes.shortTileAtBack = (est.totalTpTime > est.totalMatmulTime);
    }

    // ---- ② SetShort: E9(AG 键) 减半鼓励门 + 短块落位 ----
    if (isAg) {
        // 通算平衡且半块双耗时均超阈(切了不亏) → 对齐长度减半鼓励多切(阈值见 FitTunables)
        const bool isCalcCommBalance = ctx.ratioCalcComm < CALC_COMM_BALANCE_RATIO;
        const uint64_t cutLen = ctx.tilingM.GetAlignLength() / 2U;
        const double mmCost = ctx.matmulPerf.MatmulTime(cutLen, ctx.commPerf.rankDim());
        const double commCost = ctx.commPerf.CommTime(cutLen);
        if (isCalcCommBalance && (mmCost > ctx.tunables.halveEncourageMmUs) &&
            (commCost > ctx.tunables.halveEncourageCommUs)) {
            ctx.tilingM.SetAlignLength(cutLen);
            ctx.tilingM.SetMinLenByMin(cutLen);
        }
    }
    // [P2 接线] E10(AR 键 阈值反解)/E11-E13(L2/小shape)/E14-E16(E27 等)随各算子迁移接入
    ctx.tilingM.cutRes.shortTileLen = ctx.tilingM.GetMinLen();
    ctx.tilingM.cutRes.numShortTile = 1U;

    if (ctx.tilingM.totalLen < ctx.tilingM.cutRes.shortTileLen * 2) { // do not cut if mValue is too small
        ctx.tilingM.NoCutTiling();
        return ctx.tilingM.cutRes;
    }

    // ---- ③ SetLong: E19(AG 键) 长块通信掩盖本地首轮计算 / E18 公共双向配平 ----
    if (isAg) {
        ctx.tilingM.cutRes.longTileLen = ctx.commPerf.InverseCommTime(ctx.frontMMTime);
    } else {
        BalanceLongTileLen(ctx); // 乘数 ctx.mmRankMul(老代码 A2A 曾误用管线系数 rankTileNum)
    }

    // ---- ④ Adjust: E24(goodLinearity: AG 恒 true) + 离散化 + E25 等长归一 ----
    const bool goodLin = isAg ? true : IsGoodLinearityShape(ctx.matmulPerf.mmShapeInfo_);
    ctx.tilingM.FitTileLengthDiscrete(false, goodLin, isAg); // localAtFront: AG 前端有本地块
    MergeEqualTiles(ctx.tilingM);
    return ctx.tilingM.cutRes;
}

#endif // __APACE_FIT_BALANCE_TILING_H__
