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
 * \file all_gather_fit_balance_tiling_apace.h
 * \brief AllGatherMMFitBalanceTiling 的 apace 孪生类（迁移过渡形态, 见迁移方案
 *        MC2_LEGACY_TILING_APACE_MIGRATION_PLAN.md §4.6 孪生类切换模式）
 *
 * 与老类（all_gather_fit_balance_tiling.h, 原样保留）逐钩子等价:
 *   - 公开接口完全一致（构造签名/GetTiling 返回旧 CutResult）——调用点切换只改类名;
 *   - 4 个钩子实现为老 .cpp 的逐行拷贝（仅常量改 apace::legacy 限定）;
 *   - 拟合数据来自本算子标定头 all_gather_matmul_v2_fit_balance_calibration.h（客户编辑面）。
 * 出问题回退: 把调用点类名改回 AllGatherMMFitBalanceTiling 即可, 老类全程未被修改。
 * 全部算子迁移完成后, 孪生类转正、老类删除（P4 收尾项）。
 */

#ifndef __ALL_GATHER_FIT_BALANCE_TILING_APACE_H__
#define __ALL_GATHER_FIT_BALANCE_TILING_APACE_H__

#pragma once

#include "all_gather_matmul_v2_fit_balance_calibration.h"
#include "apace/tiling/legacy/fit_based_balance_tiling.h"
#include "op_host/op_tiling/formulaic_tiling_datatype.h" // 旧 CutResult/KernelType/TopoType/SocVersion(孪生类公开接口沿用旧类型)
#include "op_host/op_tiling/matmul_formulaic_tiling.h" // mc2tiling::TilingArgs

constexpr uint64_t FP4_EXPANSION_RATIO_AP = 2;
constexpr double FP4_COMPUTE_PER_CYCLE_RATIO_AP = 4;

// ====== 新旧边界转换（迁移胶水, 随孪生类一并转正后删除） ======
inline apace::legacy::TilingArgs ToApaceTilingArgs(const mc2tiling::TilingArgs &args)
{
    apace::legacy::TilingArgs out{};
    out.cmdType = static_cast<apace::legacy::AicpuComType>(args.cmdType);
    out.rankDim = args.rankDim;
    out.usedCoreNum = args.usedCoreNum;
    out.orgMValue = args.orgMValue;
    out.orgNValue = args.orgNValue;
    out.orgKValue = args.orgKValue;
    out.batchValue = args.batchValue;
    out.mValue = args.mValue;
    out.kValue = args.kValue;
    out.nValue = args.nValue;
    out.baseMLimit = args.baseMLimit;
    out.inputDtypeSize = args.inputDtypeSize;
    out.outputDtypeSize = args.outputDtypeSize;
    out.aicCoreNum = args.aicCoreNum;
    out.commTurn = args.commTurn;
    out.rankTileNum = args.rankTileNum;
    out.commAlg = args.commAlg;
    out.isATrans = args.isATrans;
    out.isBTrans = args.isBTrans;
    out.enablePad = args.enablePad;
    out.enableSplitK = args.enableSplitK;
    out.isBias = args.isBias;
    out.isStorageGather = args.isStorageGather;
    out.isLocal = args.isLocal;
    out.geAType = args.geAType;
    out.geBType = args.geBType;
    out.geCType = args.geCType;
    out.geBiasType = args.geBiasType;
    out.antiquantscaleDType = args.antiquantscaleDType;
    out.aType = args.aType;
    out.bType = args.bType;
    out.cType = args.cType;
    out.biasType = args.biasType;
    return out;
}

inline CutResult ToLegacyCut(const apace::legacy::CutResult &cut)
{
    CutResult out{};
    out.shortTileAtBack = cut.shortTileAtBack;
    out.totalTileCnt = cut.totalTileCnt;
    out.numLongTile = cut.numLongTile;
    out.numShortTile = cut.numShortTile;
    out.longTileLen = cut.longTileLen;
    out.shortTileLen = cut.shortTileLen;
    return out;
}

class AllGatherMMFitBalanceTilingApace : public apace::legacy::Mc2FitBasedBalanceTiling {
public:
    double frontMMTime_ = 0;

    explicit AllGatherMMFitBalanceTilingApace(const mc2tiling::TilingArgs &args, KernelType kernelType,
                                              TopoType topoType = TopoType::STANDARD_CARD,
                                              SocVersion socVersion = SocVersion::SOC950)
        : Mc2FitBasedBalanceTiling(ToApaceTilingArgs(args), static_cast<apace::legacy::KernelType>(kernelType),
                                   apace::legacy::all_gather_matmul_v2_calibration::ARCH35_COMM_FIT_TABLE,
                                   apace::legacy::all_gather_matmul_v2_calibration::MATMUL_PERF_TABLE,
                                   static_cast<apace::legacy::TopoType>(topoType),
                                   static_cast<apace::legacy::SocVersion>(socVersion))
    {
        commPerf_.SetCommShapeLen(mmInfo_.kValue);
        // mxfp4(DT_FLOAT4_E2M1)每元素占0.5字节，uint64_t无法表达0.5
        // 此处按"1字节打包2个fp4元素"建模：通信侧用SetCommDTypeSize(1)+ExpansionFraction(2)表达0.5字节/元素
        // (复用HCCL现成子字节机制，CommTime/InverseCommTime/GetLinearThresholdLen均经GetRealDtypeSizes()取0.5)；
        // matmul侧inMatrixADtypeSize置1(L2占用按2x高估，避免0导致memUsage/kLength退化为0)。
        if (args.geAType == ge::DataType::DT_FLOAT4_E2M1) {
            commPerf_.SetCommDTypeSize(1);
            commPerf_.SetCommDtypeSizeExpansionFraction(FP4_EXPANSION_RATIO_AP);
            matmulPerf_.mmShapeInfo_.inMatrixADtypeSize = 1;
            matmulPerf_.mmShapeInfo_.inMatrixBDtypeSize = 1;
            mmInfo_.inMatrixADtypeSize = 1;
            mmInfo_.inMatrixBDtypeSize = 1;
            // fp4属量化dtype, 但公共层 SetCalcType 不识别fp4(ConvertGeTypeToMmType无fp4映射→DT_MAX),
            // 致 calcType_=FP16、查表落空→computesPerCycle 退回默认4096
            // (fp8 拿到 8192), totalMatmulTime 被高估4x。此处把fp4归类为QUANT并覆盖 computesPerCycle
            // =4*COMPUTES_PER_CYCLE(=16384);。GetMachineParameters 已在基类构造时按FP16跑过, 此处覆盖。
            matmulPerf_.calcType_ = apace::legacy::MatmulCalcType::QUANT;
            matmulPerf_.mmShapeInfo_.computesPerCycle =
                FP4_COMPUTE_PER_CYCLE_RATIO_AP * apace::legacy::MatmulPerformance::COMPUTES_PER_CYCLE;
            mmInfo_.computesPerCycle = matmulPerf_.mmShapeInfo_.computesPerCycle;
        } else {
            commPerf_.SetCommDTypeSize(mmInfo_.inMatrixADtypeSize);
        }
        tilingM_.SetMinLenByMax(matmulPerf_.GetBaseM());
        tilingM_.SetAlignLength(matmulPerf_.GetBaseM());
    }

    // 旧签名入口（命名区别于基类虚函数 GetTiling, 内部调用基类模板流并完成 apace→旧 CutResult
    // 边界转换）, 使调用点与老类调用同构（切换 = 类名+方法名两处 token）。
    CutResult GetTilingResult()
    {
        return ToLegacyCut(Mc2FitBasedBalanceTiling::GetTiling());
    }

    void EstimateMMCommTime() override;
    void SetShortTileLen() override;
    void SetLongTileLen() override;
    void AdjustLongShortTileLen() override;
};

#endif // __ALL_GATHER_FIT_BALANCE_TILING_APACE_H__
