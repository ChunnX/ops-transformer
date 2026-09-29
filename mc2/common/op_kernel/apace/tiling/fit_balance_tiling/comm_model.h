/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// fit_balance_tiling 通信模型(HCCLCommModel, §4.39 拆分——原一.1 区):
// 线性拟合 y = k·x + N·b + c 的前向/反解/总量三口径(§4.37 公式改革;
// §4.38 口径消融后通信公式零份数乘子——份数已融进标定斜率 k)。
#pragma once

#include "fit_balance_types.h"

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
