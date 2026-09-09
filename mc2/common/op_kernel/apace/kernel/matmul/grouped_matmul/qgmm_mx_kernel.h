/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qgmm_mx_kernel.h
 * \brief Apace MX 量化分组矩阵乘 kernel（AIC 侧）。
 *
 * 输入约定（AllToAllv 融合场景）：
 *   - A / scaleA 已按 expert-major 连续排布（分组列表为累计行数）；
 *   - B / scaleB 为本 rank 的专家权重，按组连续存放；
 *   - C 输出与 A 行序一致（expert-major）。
 *
 * 调度参考 Blaze GemmUniversal(KernelGroupedMmadWithScaleMx) 的分组迭代：
 * 逐组刷新 ProblemShape，按 BlockSchedulerGmmSwatWithTailSplit 切块，
 * 末组触发尾块二次切分，保证多核负载均衡。
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "include/tensor_api/tensor.h"

namespace Apace {
namespace Kernel {

using namespace AscendC;
using namespace Blaze::Gemm;

#define QGMM_MX_KERNEL_CLASS_TEM_PARAMS template <class ProblemShape, class BlockMmad, class BlockScheduler>
#define QGMM_MX_KERNEL_FUNC_TEM_PARAMS ProblemShape, BlockMmad, BlockScheduler

QGMM_MX_KERNEL_CLASS_TEM_PARAMS
class QuantGroupedMatmulKernel {
public:
    __aicore__ inline QuantGroupedMatmulKernel() {}
    __aicore__ inline ~QuantGroupedMatmulKernel() {}

    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    static constexpr bool TRANS_A = Blaze::Gemm::IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = Blaze::Gemm::IsTrans<LayoutB>::value;
    static constexpr bool WEIGHT_NZ = Blaze::Gemm::IsWeightNz<LayoutB>::value;
    static constexpr int64_t C0_SIZE = Blaze::Gemm::IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr int32_t SCALE_C0 = 2;

    using BlockShape = typename BlockMmad::BlockShape;
    using BlockCoord = typename BlockScheduler::BlockCoord;
    using SchedulerBlockShape = typename BlockScheduler::BlockShape;
    using SchedulerProblemShape = typename BlockScheduler::ProblemShape;

    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<C0_SIZE>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<C0_SIZE>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<Te::C0_ELEMENT<CType>>>;
    using MakeLayoutBias = Te::FrameLayoutFormat<Te::NDExtLayoutPtn>;
    using MakeLayoutScaleA =
        Std::conditional_t<TRANS_A, Te::FrameLayoutFormat<Te::ScaleADNLayoutPtn, Std::Int<SCALE_C0>>,
                           Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB =
        Std::conditional_t<TRANS_B, Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<SCALE_C0>>,
                           Te::FrameLayoutFormat<Te::ScaleBNDLayoutPtn, Std::Int<SCALE_C0>>>;

    /*!
     * \brief GMM 切分参数（host tiling 下发）
     */
    struct GMMTiling {
        uint32_t groupNum;      // 分组数（本 rank 专家数）
        int64_t n;              // N
        int64_t k;              // K
        uint32_t baseM;         // M 方向基础块
        uint32_t baseN;         // N 方向基础块
        uint32_t baseK;         // K 方向基础块
        uint32_t kAL1;          // A 矩阵 L1 K 轴切分
        uint32_t kBL1;          // B 矩阵 L1 K 轴切分
        uint32_t scaleKAL1;     // MX 场景 ScaleA/ScaleB 共享 K 窗口（scaleKAL1 == scaleKBL1）
        uint8_t isBias;         // 预留，当前恒 0
        uint8_t dbL0C{2};       // L0C 双缓冲
        uint8_t l1BufferStage{2}; // L1 缓冲级数
    };

    /*!
     * \brief kernel 顶层参数
     */
    struct Params {
        __gm__ AType *aBasePtr;              // [totalM, K] 量化 A（expert-major）
        __gm__ fp8_e8m0_t *scaleABasePtr;    // [totalM, scaleKLen] A 的 MX scale
        __gm__ BType *bBasePtr;              // [groupNum, K, N] / [groupNum, N, K] 权重
        __gm__ fp8_e8m0_t *scaleBBasePtr;    // 权重 MX scale（布局随 TransB）
        __gm__ CType *cBasePtr;              // [totalM, N] 输出
        __gm__ const uint32_t *groupList;    // 累计行数分组列表，长度 groupNum
        GMMTiling gmm;
    };

    __aicore__ inline void operator()(const Params &params)
    {
        Run(params);
    }

private:
    __aicore__ inline int64_t GetScaleK(int64_t k) const
    {
        return ((k + MXFP_DIVISOR_SIZE - 1) >> MXFP_DIVISOR_SHIFT) << MXFP_MULTI_BASE_SHIFT;
    }

    __aicore__ inline void SetSchedulerTailAlign(BlockScheduler &scheduler)
    {
        if constexpr (!TRANS_A) {
            constexpr uint32_t mTailAlign = 1;
            constexpr uint32_t nTailAlign = TRANS_B ? static_cast<uint32_t>(Blaze::Gemm::BLOCK_CUBE) :
                                                     static_cast<uint32_t>(C0_SIZE);
            scheduler.SetTailAlign(mTailAlign, nTailAlign);
        } else {
            constexpr uint32_t mTailAlign = static_cast<uint32_t>(Blaze::Gemm::Block::INNER_AXIS_MIN_SPLIT_VAL);
            constexpr uint32_t nTailAlign = TRANS_B ? static_cast<uint32_t>(Blaze::Gemm::BLOCK_CUBE) :
                                                     static_cast<uint32_t>(Blaze::Gemm::Block::INNER_AXIS_MIN_SPLIT_VAL);
            scheduler.SetTailAlign(mTailAlign, nTailAlign);
        }
    }

    __aicore__ inline void BaseMBalance(BlockScheduler &scheduler, int64_t m, int64_t baseM)
    {
        if constexpr (!TRANS_A) {
            if (m <= 0) {
                return;
            }
            constexpr int64_t blockCubeMask = Blaze::Gemm::BLOCK_CUBE - 1;
            const int64_t safeBaseM = baseM > 0 ? baseM : static_cast<int64_t>(Blaze::Gemm::BLOCK_CUBE);
            const int64_t mCnt = (m + safeBaseM - 1) / safeBaseM;
            const int64_t balancedBaseM = (m + mCnt - 1) / mCnt;
            curBaseM_ = static_cast<uint32_t>((balancedBaseM + blockCubeMask) & ~blockCubeMask);
            scheduler.UpdateBaseM(curBaseM_);
        }
    }

    __aicore__ inline bool IsLastGroupAndNeedSplit(const BlockScheduler &scheduler)
    {
        return (scheduler.GetEndBlockIdx() + 1) <= (AscendC::GetBlockNum() >> 1);
    }

    __aicore__ inline int64_t GetGroupRows(uint32_t groupIdx)
    {
        int64_t end = static_cast<int64_t>(groupListGm_[groupIdx]);
        return end - preGroupOffset_;
    }

    __aicore__ inline void Run(const Params &params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        Init(params);
        if (groupNum_ == 0) {
            return;
        }
        const auto &gmm = params.gmm;
        BlockScheduler scheduler(gmm.baseM, gmm.baseN, gmm.baseK);
        SetSchedulerTailAlign(scheduler);

        const uint32_t lastGroupIdx = groupNum_ - 1;
        for (uint32_t loopIdx = 0; loopIdx < lastGroupIdx; ++loopIdx) {
            const int64_t groupM = GetGroupRows(loopIdx);
            const int64_t groupStart = preGroupOffset_;
            preGroupOffset_ += groupM;
            if (groupM <= 0) {
                continue;
            }
            BaseMBalance(scheduler, groupM, gmm.baseM);
            scheduler.UpdateNextProblem(SchedulerProblemShape{groupM, gmm.n, gmm.k, 0});
            ProcessSingleGroup<false>(params, scheduler, loopIdx, groupM, groupStart);
        }

        const int64_t groupM = GetGroupRows(lastGroupIdx);
        const int64_t groupStart = preGroupOffset_;
        preGroupOffset_ += groupM;
        if (groupM > 0) {
            BaseMBalance(scheduler, groupM, gmm.baseM);
            scheduler.UpdateNextProblem(SchedulerProblemShape{groupM, gmm.n, gmm.k, 0});
            if (IsLastGroupAndNeedSplit(scheduler)) {
                scheduler.UpdateTailTile();
                ProcessSingleGroup<true>(params, scheduler, lastGroupIdx, groupM, groupStart);
            } else {
                ProcessSingleGroup<false>(params, scheduler, lastGroupIdx, groupM, groupStart);
            }
        }
    }

    __aicore__ inline void Init(const Params &params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        groupListGm_ = params.groupList;
        groupNum_ = params.gmm.groupNum;
        n_ = params.gmm.n;
        k_ = params.gmm.k;
        curBaseM_ = params.gmm.baseM;
        baseN_ = params.gmm.baseN;
        preGroupOffset_ = 0;

        const BlockShape l0Shape{static_cast<int64_t>(params.gmm.baseM), static_cast<int64_t>(params.gmm.baseN),
                                 static_cast<int64_t>(params.gmm.baseK), 0};
        const typename BlockMmad::L1Params l1Params{params.gmm.kAL1, params.gmm.kBL1, params.gmm.scaleKAL1};
        const typename BlockMmad::MmadParams mmadParams{l0Shape, l1Params, params.gmm.isBias == 1,
                                                        params.gmm.dbL0C == DOUBLE_BUFFER_COUNT,
                                                        params.gmm.l1BufferStage};
        mmadOp_.Init(ProblemShape{0, n_, k_, 0}, mmadParams);
    }

    template <bool isLastGroupAndNeedSplit>
    __aicore__ inline void ProcessSingleGroup(const Params &params, BlockScheduler &scheduler, uint32_t groupIdx,
                                              int64_t groupM, int64_t groupStart)
    {
        BlockCoord blockCoord;
        if (!scheduler.GetNextBlockCoord(blockCoord)) {
            return;
        }
        const int64_t problemM = groupM;
        const int64_t scaleK = GetScaleK(k_);
        const int64_t baseN = static_cast<int64_t>(baseN_);

        mmadOp_.UpdateParamsForNextProblem(ProblemShape{problemM, n_, k_, 0});

        // 组内偏移：A/scaleA/C 按累计行数推进；B/scaleB 按组 stride 推进
        const int64_t aOffset = groupStart * k_;
        const int64_t scaleAOffset = groupStart * scaleK;
        const int64_t bOffset = static_cast<int64_t>(groupIdx) * k_ * n_;
        const int64_t scaleBOffset = static_cast<int64_t>(groupIdx) * n_ * scaleK;
        const int64_t cOffset = groupStart * n_;

        auto layoutA = MakeLayoutA{}(problemM, k_);
        auto layoutScaleA = MakeLayoutScaleA{}(problemM, scaleK);
        auto layoutB = MakeLayoutB{}(k_, n_);
        auto layoutScaleB = MakeLayoutScaleB{}(scaleK, n_);
        auto layoutC = MakeLayoutC{}(problemM, n_);
        auto layoutBias = MakeLayoutBias{}(1, n_);
        auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(params.aBasePtr + aOffset), layoutA);
        auto gmScaleA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(params.scaleABasePtr + scaleAOffset),
                                       layoutScaleA);
        auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(params.bBasePtr + bOffset), layoutB);
        auto gmScaleB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(params.scaleBBasePtr + scaleBOffset),
                                       layoutScaleB);
        auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(params.cBasePtr + cOffset), layoutC);
        auto gmBias =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(static_cast<__gm__ BiasType *>(nullptr)), layoutBias);

        do {
            const SchedulerBlockShape schedulerBlockShape = scheduler.GetBlockShape(blockCoord);
            const int64_t blockM = Te::Get<MNK_M>(schedulerBlockShape);
            const int64_t blockN = Te::Get<MNK_N>(schedulerBlockShape);
            if (blockM <= 0 || blockN <= 0) {
                continue;
            }
            const int64_t mSplitOffset = Te::Get<MNK_K>(schedulerBlockShape);
            const int64_t nSplitOffset = Te::Get<MNK_B>(schedulerBlockShape);
            const int64_t mBlockIdx = Te::Get<MNK_M>(blockCoord);
            const int64_t nBlockIdx = Te::Get<MNK_N>(blockCoord);
            const int64_t blockK = k_;
            const BlockShape blockShape{blockM, blockN, blockK, 0};
            const int64_t mPos = mBlockIdx * curBaseM_ + mSplitOffset;
            const int64_t nPos = nBlockIdx * baseN + nSplitOffset;

            auto gmBlockA = gmA.Slice(Te::MakeCoord(mPos, 0), Te::MakeShape(blockM, blockK));
            auto gmBlockScaleA = gmScaleA.Slice(Te::MakeCoord(mPos, 0), Te::MakeShape(blockM, scaleK));
            auto gmBlockB = gmB.Slice(Te::MakeCoord(0, nPos), Te::MakeShape(blockK, blockN));
            auto gmBlockScaleB = gmScaleB.Slice(Te::MakeCoord(0, nPos), Te::MakeShape(scaleK, blockN));
            auto gmBlockBias = gmBias.Slice(Te::MakeCoord(0, nPos), Te::MakeShape(1, blockN));
            auto gmBlockC = gmC.Slice(Te::MakeCoord(mPos, nPos), Te::MakeShape(blockM, blockN));
            mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, blockShape);
        } while (scheduler.GetNextBlockCoord(blockCoord));
    }

private:
    BlockMmad mmadOp_;
    __gm__ const uint32_t *groupListGm_{nullptr};
    int64_t preGroupOffset_{0};
    int64_t n_{0};
    int64_t k_{0};
    uint32_t groupNum_{0};
    uint32_t curBaseM_{0};
    uint32_t baseN_{0};
};

} // namespace Kernel
} // namespace Apace
