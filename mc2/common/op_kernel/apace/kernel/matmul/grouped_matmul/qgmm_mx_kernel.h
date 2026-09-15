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
 * \brief Apace MX grouped-matmul kernel based on Blaze kernel_qgmm_mx.h.
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

constexpr uint8_t GROUP_LIST_TYPE_OFFSET = 0U;
constexpr uint8_t GROUP_LIST_TYPE_LENGTH = 1U;
constexpr uint8_t GROUP_LIST_TYPE_SPARSE = 2U;
constexpr uint32_t SPARSE_GROUP_LIST_ITEM_STRIDE = 2U;
constexpr uint32_t SPARSE_GROUP_LIST_TOKEN_OFFSET = 1U;

template <class ProblemShape, class BlockMmad, class BlockScheduler>
class GroupedMatmulMxKernel {
public:
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutScaleB = typename BlockMmad::LayoutScaleB;
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockShape = typename BlockMmad::BlockShape;
    using BlockCoord = typename BlockScheduler::BlockCoord;
    using SchedulerBlockShape = typename BlockScheduler::BlockShape;
    using SchedulerProblemShape = typename BlockScheduler::ProblemShape;

    static constexpr bool TRANS_A = Blaze::Gemm::IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = Blaze::Gemm::IsTrans<LayoutB>::value;
    static constexpr bool WEIGHT_NZ = Blaze::Gemm::IsWeightNz<LayoutB>::value;
    static constexpr bool SCALE_NZ = Blaze::Gemm::IsScaleNz<LayoutScaleB>::value;
    static constexpr int64_t C0_SIZE = Blaze::Gemm::IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr int32_t SCALE_C0 = 2;
    static constexpr int64_t BLOCK_CUBE_MASK = Blaze::Gemm::BLOCK_CUBE - 1;
    static constexpr int64_t SCALE_CACHE_MASK = 0xff;

    static_assert(Blaze::Gemm::IsFp4<AType>() == Blaze::Gemm::IsFp4<BType>(),
                  "GroupedMatmulMxKernel requires A and B to have the same MX bit width.");
    static_assert(Std::is_same_v<LayoutA, Te::NDExtLayoutPtn>,
                  "GroupedMatmulMxKernel only supports non-transposed ND A.");

    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<C0_SIZE>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<C0_SIZE>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<Te::C0_ELEMENT<CType>>>;
    using MakeLayoutBias = Te::FrameLayoutFormat<Te::NDExtLayoutPtn>;
    using MakeLayoutScaleA =
        Std::conditional_t<TRANS_A, Te::FrameLayoutFormat<Te::ScaleADNLayoutPtn, Std::Int<SCALE_C0>>,
                           Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = Te::FrameLayoutFormat<LayoutScaleB, Std::Int<SCALE_C0>>;

    struct GroupOffsetTables {
        // Logical element offsets. FP4 packing is converted exactly once when
        // the typed GM view is built. Tables are indexed by group-list slot,
        // not by sparse expert ID. A null table means derive a contiguous
        // offset from groupList; scaleA follows A's row offset automatically.
        const __gm__ uint64_t *aOffsetTable{nullptr};
        const __gm__ uint64_t *cOffsetTable{nullptr};
    };

    struct GMMTiling {
        uint32_t groupNum;
        int64_t m;
        int64_t n;
        int64_t k;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t kAL1;
        uint32_t kBL1;
        uint32_t scaleKAL1;
        uint8_t isBias;
        uint8_t dbL0C{DOUBLE_BUFFER_COUNT};
        uint8_t l1BufferStage{DOUBLE_BUFFER_COUNT};
        // OFFSET: cumulative token end; LENGTH: token count;
        // SPARSE: repeated [expertId, tokenCount].
        uint8_t groupListType{GROUP_LIST_TYPE_OFFSET};
    };

    struct Params {
        // A/B/C/scaleA/scaleB/bias base addresses consumed by BlockMmad.
        BlockMmadParams mmadParams;
        GM_ADDR groupListGmAddr{nullptr};
        GMMTiling gmmParams;
        GroupOffsetTables groupOffsetTables;
    };

    __aicore__ inline GroupedMatmulMxKernel() {}
    __aicore__ inline ~GroupedMatmulMxKernel() {}

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
            constexpr uint32_t nTailAlign =
                TRANS_B ? static_cast<uint32_t>(Blaze::Gemm::BLOCK_CUBE) :
                          static_cast<uint32_t>(Blaze::Gemm::Block::INNER_AXIS_MIN_SPLIT_VAL);
            scheduler.SetTailAlign(mTailAlign, nTailAlign);
        }
    }

    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2CacheHint(TensorB &gmB, TensorScaleB &gmScaleB, int64_t mSize)
    {
        if constexpr (WEIGHT_NZ) {
            if (curBaseM_ >= mSize) {
                gmB.SetL2CacheHint(Te::CacheMode::CACHE_MODE_DISABLE);
                gmScaleB.SetL2CacheHint(Te::CacheMode::CACHE_MODE_DISABLE);
            } else {
                gmB.SetL2CacheHint(Te::CacheMode::CACHE_MODE_NORMAL);
                gmScaleB.SetL2CacheHint(Te::CacheMode::CACHE_MODE_NORMAL);
            }
        } else if constexpr (TRANS_B) {
            if (curBaseM_ >= mSize && (k_ & SCALE_CACHE_MASK) == 0) {
                gmB.SetL2CacheHint(Te::CacheMode::CACHE_MODE_DISABLE);
                gmScaleB.SetL2CacheHint(Te::CacheMode::CACHE_MODE_DISABLE);
            } else {
                gmB.SetL2CacheHint(Te::CacheMode::CACHE_MODE_NORMAL);
                gmScaleB.SetL2CacheHint(Te::CacheMode::CACHE_MODE_NORMAL);
            }
        } else {
            if (curBaseM_ >= mSize && (n_ & SCALE_CACHE_MASK) == 0 &&
                (static_cast<int64_t>(baseN_) & SCALE_CACHE_MASK) == 0) {
                gmB.SetL2CacheHint(Te::CacheMode::CACHE_MODE_DISABLE);
                gmScaleB.SetL2CacheHint(Te::CacheMode::CACHE_MODE_DISABLE);
            } else {
                gmB.SetL2CacheHint(Te::CacheMode::CACHE_MODE_NORMAL);
                gmScaleB.SetL2CacheHint(Te::CacheMode::CACHE_MODE_NORMAL);
            }
        }
    }

    __aicore__ inline void BaseMBalance(BlockScheduler &scheduler, int64_t m)
    {
        if constexpr (!TRANS_A) {
            const int64_t safeBaseM = baseM_ > 0 ? baseM_ : static_cast<int64_t>(Blaze::Gemm::BLOCK_CUBE);
            const int64_t mCnt = (m + safeBaseM - 1) / safeBaseM;
            const int64_t balancedBaseM = (m + mCnt - 1) / mCnt;
            curBaseM_ = static_cast<uint32_t>((balancedBaseM + BLOCK_CUBE_MASK) & ~BLOCK_CUBE_MASK);
            scheduler.UpdateBaseM(curBaseM_);
        }
    }

    __aicore__ inline bool IsLastGroupAndNeedSplit(const BlockScheduler &scheduler) const
    {
        return (scheduler.GetEndBlockIdx() + 1) <= (AscendC::GetBlockNum() >> 1);
    }

    template <typename GroupListTensor>
    __aicore__ inline int64_t GetGroupRows(const GroupListTensor &groupList, uint32_t listIdx)
    {
        int64_t rows = 0;
        if (groupListType_ == GROUP_LIST_TYPE_OFFSET) {
            const int64_t end = groupList[listIdx];
            rows = end - preGroupOffset_;
            preGroupOffset_ = end;
        } else if (groupListType_ == GROUP_LIST_TYPE_LENGTH) {
            rows = groupList[listIdx];
            preGroupOffset_ += rows;
        } else {
            rows = groupList[listIdx * SPARSE_GROUP_LIST_ITEM_STRIDE + SPARSE_GROUP_LIST_TOKEN_OFFSET];
            preGroupOffset_ += rows;
        }
        return rows;
    }

    template <typename GroupListTensor>
    __aicore__ inline uint32_t GetExpertIdx(const GroupListTensor &groupList, uint32_t listIdx) const
    {
        if (groupListType_ == GROUP_LIST_TYPE_SPARSE) {
            return static_cast<uint32_t>(groupList[listIdx * SPARSE_GROUP_LIST_ITEM_STRIDE]);
        }
        return listIdx;
    }

    __aicore__ inline void Init(const Params &params)
    {
        const auto &gmm = params.gmmParams;
        aBasePtr_ = reinterpret_cast<__gm__ AType *>(params.mmadParams.aGmAddr);
        bBasePtr_ = reinterpret_cast<__gm__ BType *>(params.mmadParams.bGmAddr);
        cBasePtr_ = reinterpret_cast<__gm__ CType *>(params.mmadParams.cGmAddr);
        scaleABasePtr_ = reinterpret_cast<__gm__ fp8_e8m0_t *>(params.mmadParams.scaleAGmAddr);
        scaleBBasePtr_ = reinterpret_cast<__gm__ fp8_e8m0_t *>(params.mmadParams.scaleBGmAddr);
        biasBasePtr_ = reinterpret_cast<__gm__ BiasType *>(params.mmadParams.biasGmAddr);
        groupListGmAddr_ = params.groupListGmAddr;
        offsetTables_ = params.groupOffsetTables;
        groupNum_ = gmm.groupNum;
        groupListType_ = gmm.groupListType;
        n_ = gmm.n;
        k_ = gmm.k;
        baseM_ = gmm.baseM;
        baseN_ = gmm.baseN;
        curBaseM_ = gmm.baseM;
        isBias_ = gmm.isBias == 1U;
        preGroupOffset_ = 0;
        aOffset_ = 0;
        scaleAOffset_ = 0;
        bOffset_ = 0;
        scaleBOffset_ = 0;
        biasOffset_ = 0;
        cOffset_ = 0;

        if constexpr (!WEIGHT_NZ) {
            perGroupBOffset_ = n_ * k_;
        } else if constexpr (TRANS_B) {
            perGroupBOffset_ = static_cast<int64_t>(
                Align16(n_) * (Blaze::Gemm::IsFp4<AType>() ? Align64(k_) : Align32(k_)));
        } else {
            perGroupBOffset_ = static_cast<int64_t>(
                (Blaze::Gemm::IsFp4<AType>() ? Align64(n_) : Align32(n_)) * Align16(k_));
        }

        const BlockShape l0Shape{static_cast<int64_t>(gmm.baseM), static_cast<int64_t>(gmm.baseN),
                                 static_cast<int64_t>(gmm.baseK), 0};
        const typename BlockMmad::L1Params l1Params{gmm.kAL1, gmm.kBL1, gmm.scaleKAL1};
        const typename BlockMmad::MmadParams blockParams{l0Shape, l1Params, isBias_,
                                                         gmm.dbL0C == DOUBLE_BUFFER_COUNT, gmm.l1BufferStage};
        const ProblemShape initProblemShape{gmm.m, gmm.n, gmm.k, 0};
        problemShape_ = initProblemShape;
        mmadOp_.Init(initProblemShape, blockParams);
    }

    __aicore__ inline void UpdateBaseOffsets(uint32_t listIdx, uint32_t expertIdx, int64_t groupM)
    {
        const int64_t groupStart = preGroupOffset_ - groupM;
        const int64_t logicalAOffset = offsetTables_.aOffsetTable == nullptr ? groupStart * k_ :
                                       static_cast<int64_t>(offsetTables_.aOffsetTable[listIdx]);
        aOffset_ = Blaze::Gemm::IsFp4<AType>() ? (logicalAOffset >> 1) : logicalAOffset;
        scaleAOffset_ = (logicalAOffset / k_) * GetScaleK(k_);

        bOffset_ = perGroupBOffset_ * static_cast<int64_t>(expertIdx);
        if constexpr (Blaze::Gemm::IsFp4<BType>()) {
            bOffset_ >>= 1;
        }
        const int64_t scaleK = GetScaleK(k_);
        if constexpr (SCALE_NZ) {
            scaleBOffset_ = static_cast<int64_t>(CalScaleNZGmAddrOffset(TRANS_B, expertIdx, n_, scaleK));
        } else {
            scaleBOffset_ = static_cast<int64_t>(expertIdx) * n_ * scaleK;
        }
        biasOffset_ = static_cast<int64_t>(expertIdx) * n_;
        cOffset_ = offsetTables_.cOffsetTable == nullptr ? groupStart * n_ :
                   static_cast<int64_t>(offsetTables_.cOffsetTable[listIdx]);
    }

    template <bool isLastGroupAndNeedSplit>
    __aicore__ inline void ProcessSingleGroup(BlockScheduler &scheduler, uint32_t listIdx, uint32_t expertIdx,
                                               int64_t groupM)
    {
        BlockCoord blockCoord;
        if (!scheduler.GetNextBlockCoord(blockCoord)) {
            return;
        }
        UpdateBaseOffsets(listIdx, expertIdx, groupM);
        problemShape_ = ProblemShape{groupM, n_, k_, 0};
        mmadOp_.UpdateParamsForNextProblem(problemShape_);

        const int64_t scaleK = GetScaleK(k_);
        auto layoutA = MakeLayoutA{}(groupM, k_);
        auto layoutScaleA = MakeLayoutScaleA{}(groupM, scaleK);
        auto layoutB = MakeLayoutB{}(k_, n_);
        auto layoutScaleB = MakeLayoutScaleB{}(scaleK, n_);
        auto layoutC = MakeLayoutC{}(groupM, n_);
        auto layoutBias = MakeLayoutBias{}(1, n_);
        auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(aBasePtr_ + aOffset_), layoutA);
        auto gmScaleA =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(scaleABasePtr_ + scaleAOffset_), layoutScaleA);
        auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(bBasePtr_ + bOffset_), layoutB);
        auto gmScaleB =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(scaleBBasePtr_ + scaleBOffset_), layoutScaleB);
        auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(cBasePtr_ + cOffset_), layoutC);
        auto gmBias = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(isBias_ ? biasBasePtr_ + biasOffset_ : nullptr), layoutBias);

        if constexpr (!isLastGroupAndNeedSplit) {
            SetL2CacheHint(gmB, gmScaleB, groupM);
        }

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
            const BlockShape blockShape{blockM, blockN, k_, 0};
            const int64_t mPos = mBlockIdx * curBaseM_ + mSplitOffset;
            const int64_t nPos = nBlockIdx * static_cast<int64_t>(baseN_) + nSplitOffset;

            auto gmBlockA = gmA.Slice(Te::MakeCoord(mPos, 0), Te::MakeShape(blockM, k_));
            auto gmBlockScaleA = gmScaleA.Slice(Te::MakeCoord(mPos, 0), Te::MakeShape(blockM, scaleK));
            auto gmBlockB = gmB.Slice(Te::MakeCoord(0, nPos), Te::MakeShape(k_, blockN));
            auto gmBlockScaleB = gmScaleB.Slice(Te::MakeCoord(0, nPos), Te::MakeShape(scaleK, blockN));
            auto gmBlockBias = gmBias.Slice(Te::MakeCoord(0, nPos), Te::MakeShape(1, blockN));
            auto gmBlockC = gmC.Slice(Te::MakeCoord(mPos, nPos), Te::MakeShape(blockM, blockN));
            mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, blockShape);
        } while (scheduler.GetNextBlockCoord(blockCoord));
    }

    __aicore__ inline void Run(const Params &params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        Init(params);
        if (groupNum_ == 0U) {
            return;
        }

        const int64_t groupListSize = static_cast<int64_t>(groupNum_) *
            (groupListType_ == GROUP_LIST_TYPE_SPARSE ? SPARSE_GROUP_LIST_ITEM_STRIDE : 1U);
        const auto groupListLayout = Te::MakeLayout(Te::MakeShape(groupListSize), Te::MakeStride(1L));
        const auto groupList = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ int64_t *>(groupListGmAddr_)),
            groupListLayout);
        const auto &gmm = params.gmmParams;
        BlockScheduler scheduler(gmm.baseM, gmm.baseN, gmm.baseK);
        SetSchedulerTailAlign(scheduler);

        const uint32_t lastGroupIdx = groupNum_ - 1U;
        for (uint32_t listIdx = 0; listIdx < lastGroupIdx; ++listIdx) {
            const int64_t groupM = GetGroupRows(groupList, listIdx);
            if (groupM <= 0) {
                continue;
            }
            BaseMBalance(scheduler, groupM);
            scheduler.UpdateNextProblem(SchedulerProblemShape{groupM, n_, k_, 0});
            ProcessSingleGroup<false>(scheduler, listIdx, GetExpertIdx(groupList, listIdx), groupM);
        }

        const int64_t groupM = GetGroupRows(groupList, lastGroupIdx);
        if (groupM > 0) {
            BaseMBalance(scheduler, groupM);
            scheduler.UpdateNextProblem(SchedulerProblemShape{groupM, n_, k_, 0});
            const uint32_t expertIdx = GetExpertIdx(groupList, lastGroupIdx);
            if (IsLastGroupAndNeedSplit(scheduler)) {
                scheduler.UpdateTailTile();
                ProcessSingleGroup<true>(scheduler, lastGroupIdx, expertIdx, groupM);
            } else {
                ProcessSingleGroup<false>(scheduler, lastGroupIdx, expertIdx, groupM);
            }
        }
    }

private:
    BlockMmad mmadOp_;
    ProblemShape problemShape_{};
    __gm__ AType *aBasePtr_{nullptr};
    __gm__ BType *bBasePtr_{nullptr};
    __gm__ CType *cBasePtr_{nullptr};
    __gm__ BiasType *biasBasePtr_{nullptr};
    __gm__ fp8_e8m0_t *scaleABasePtr_{nullptr};
    __gm__ fp8_e8m0_t *scaleBBasePtr_{nullptr};
    GM_ADDR groupListGmAddr_{nullptr};
    GroupOffsetTables offsetTables_{};

    int64_t preGroupOffset_{0};
    int64_t perGroupBOffset_{0};
    int64_t aOffset_{0};
    int64_t scaleAOffset_{0};
    int64_t bOffset_{0};
    int64_t scaleBOffset_{0};
    int64_t biasOffset_{0};
    int64_t cOffset_{0};
    int64_t n_{0};
    int64_t k_{0};
    uint32_t groupNum_{0};
    uint32_t baseM_{0};
    uint32_t baseN_{0};
    uint32_t curBaseM_{0};
    uint8_t groupListType_{GROUP_LIST_TYPE_OFFSET};
    bool isBias_{false};
};

} // namespace Kernel
} // namespace Apace
