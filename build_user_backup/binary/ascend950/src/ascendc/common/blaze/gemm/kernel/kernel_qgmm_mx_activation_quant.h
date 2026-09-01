// clang-format off
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_qgmm_mx_activation_quant.h
 * \brief Grouped MX matmul kernel with activation and dynamic MX quantization.
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {

namespace Kernel {
namespace {
constexpr uint64_t GROUP_LIST_TYPE_OFFSET = 0UL;
constexpr uint64_t GROUP_LIST_TYPE_LENGTH = 1UL;
constexpr uint64_t GROUP_LIST_TYPE_SPARSE = 2UL;
constexpr uint64_t SPARSE_GROUP_LIST_ITEM_STRIDE = 2UL;
constexpr uint64_t SPARSE_GROUP_LIST_SPLIT_VALUE_OFFSET = 1UL;
constexpr int64_t BLOCK_CUBE_MASK = BLOCK_CUBE - 1;
constexpr int64_t SCALE_CACHE_MASK = 0xff;

template <typename T_>
__aicore__ inline __gm__ T_* GetTensorAddrFromTensorList(uint32_t index, __gm__ T_* tensorPtr)
{
    auto tensorList = AscendC::GlobalTensor<uint64_t>();
    tensorList.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(tensorPtr));
    const int64_t tensorPtrOffset = static_cast<int64_t>(tensorList.GetValue(0));
    // Tensor-list offsets are byte offsets in the ACL descriptor.  Keep the
    // unit conversion explicit so the uint64_t table layout is not coupled to
    // a magic shift literal.
    const uint64_t tensorAddr = tensorList.GetValue(tensorPtrOffset / static_cast<int64_t>(sizeof(uint64_t)) +
                                                    static_cast<int64_t>(index));
    return reinterpret_cast<__gm__ T_*>(tensorAddr);
}
} // namespace

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<
    ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<KernelGroupedMmadWithScaleMxActivationQuant,
                                                      typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    using AType = typename BlockMmad_::AType;
    using BType = typename BlockMmad_::BType;
    using CType = typename BlockMmad_::CType;
    using BiasType = typename BlockMmad_::BiasType;
    using LayoutA = typename BlockMmad_::LayoutA;
    using LayoutB = typename BlockMmad_::LayoutB;
    using LayoutC = typename BlockMmad_::LayoutC;
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
    static constexpr bool HAS_ACTIVATION_QUANT = true;
    static_assert(!HAS_ACTIVATION_QUANT || !TRANS_A,
                  "Grouped MX activation quantization only supports non-transposed A.");

    using BlockMmadParams = typename BlockMmad_::Params;
    using BlockEpilogueParams = typename BlockEpilogue_::Params;
    using L1Params = typename BlockMmad_::L1Params;

    using BlockShape = typename BlockMmad_::BlockShape;
    using SchedulerProblemShape = typename BlockScheduler_::ProblemShape;

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
        uint32_t scaleKAL1; // ScaleA L1 K-axis split size; must equal scaleKBL1 for MX.
        uint32_t scaleKBL1; // ScaleB L1 K-axis split size; kept for tiling compatibility.
        uint8_t isBias;
        uint8_t dbL0C{DOUBLE_BUFFER_COUNT};
        uint8_t l1BufferStage{DOUBLE_BUFFER_COUNT};
        // Reserved for GMM tiling compatibility. Current kernel does not read this field;
        // The split axis is selected by LayoutA: !TRANS_A means split-M, TRANS_A means split-K.
        int8_t groupType;
        uint8_t groupListType;
        uint8_t singleW;
    };

    struct Params {
        ProblemShape_ problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        GM_ADDR groupListGmAddr;
        GMMTiling gmmParams;
    };

    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    __aicore__ inline void operator()(const Params& params) { Run(params); }

private:
    static constexpr uint32_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<LayoutC, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<CType>>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        TRANS_A, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleADNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        TRANS_B, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;

    __aicore__ inline void SetSchedulerTailAlign(BlockScheduler_& scheduler)
    {
        if constexpr (!TRANS_A) {
            constexpr uint32_t mTailAlign = 1;
            constexpr uint32_t nTailAlign = TRANS_B ? static_cast<uint32_t>(BLOCK_CUBE) :
                                                      static_cast<uint32_t>(C0_SIZE);
            scheduler.SetTailAlign(mTailAlign, nTailAlign);
        } else {
            constexpr uint32_t mTailAlign = static_cast<uint32_t>(Block::INNER_AXIS_MIN_SPLIT_VAL);
            constexpr uint32_t nTailAlign = TRANS_B ? static_cast<uint32_t>(BLOCK_CUBE) :
                                                      static_cast<uint32_t>(Block::INNER_AXIS_MIN_SPLIT_VAL);
            scheduler.SetTailAlign(mTailAlign, nTailAlign);
        }
    }

    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2CacheHint(TensorB& gmB, TensorScaleB& gmScaleB, int64_t mSize, int64_t curBaseM,
                                          int64_t baseN)
    {
        const int64_t problemN = AscendC::Te::Get<MNK_N>(problemShape_);
        const int64_t problemK = AscendC::Te::Get<MNK_K>(problemShape_);
        if constexpr (WEIGHT_NZ) {
            if (curBaseM >= mSize) {
                gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
            } else {
                gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
            }
        } else {
            if constexpr (TRANS_B) {
                if (curBaseM >= mSize && (problemK & SCALE_CACHE_MASK) == 0) {
                    gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                    gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                } else {
                    gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                    gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                }
            } else {
                if (curBaseM >= mSize && (problemN & SCALE_CACHE_MASK) == 0 && (baseN & SCALE_CACHE_MASK) == 0) {
                    gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                    gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                } else {
                    gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                    gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                }
            }
        }
    }

    __aicore__ inline void Run(const Params& params)
    {
        if constexpr (!HAS_ACTIVATION_QUANT) {
            if ASCEND_IS_AIV {
                return;
            }
        }
        Init(params);
        if (groupNum_ == 0) {
            return;
        }
        const auto& gmmParams = params.gmmParams;
        BlockScheduler_ scheduler(gmmParams.baseM, gmmParams.baseN, gmmParams.baseK);
        SetSchedulerTailAlign(scheduler);
        const uint32_t lastGroupIdx = groupNum_ - 1;
        for (uint32_t loopIdx = 0; loopIdx < lastGroupIdx; ++loopIdx) {
            uint32_t groupIdx = loopIdx;
            if (groupListType_ == GROUP_LIST_TYPE_SPARSE) {
                groupIdx = static_cast<uint32_t>(groupListGlobal_.GetValue(loopIdx * SPARSE_GROUP_LIST_ITEM_STRIDE));
            }
            SetMNK(loopIdx);
            const int64_t problemM = AscendC::Te::Get<MNK_M>(problemShape_);
            const int64_t problemN = AscendC::Te::Get<MNK_N>(problemShape_);
            const int64_t problemK = AscendC::Te::Get<MNK_K>(problemShape_);
            if (problemM <= 0 || problemN <= 0 || problemK <= 0) {
                if (groupListType_ == GROUP_LIST_TYPE_SPARSE && problemM <= 0) {
                    break;
                }
                continue;
            }
            BaseMBalance(scheduler, problemM, gmmParams.baseM);
            scheduler.UpdateNextProblem(SchedulerProblemShape{problemM, problemN, problemK, 0});
            ProcessSingleGroup<false>(scheduler, groupIdx);
        }

        uint32_t groupIdx = lastGroupIdx;
        if (groupListType_ == GROUP_LIST_TYPE_SPARSE) {
            groupIdx = static_cast<uint32_t>(groupListGlobal_.GetValue(lastGroupIdx * SPARSE_GROUP_LIST_ITEM_STRIDE));
        }
        SetMNK(lastGroupIdx);
        const int64_t problemM = AscendC::Te::Get<MNK_M>(problemShape_);
        const int64_t problemN = AscendC::Te::Get<MNK_N>(problemShape_);
        const int64_t problemK = AscendC::Te::Get<MNK_K>(problemShape_);
        if (problemM > 0 && problemN > 0 && problemK > 0) {
            BaseMBalance(scheduler, problemM, gmmParams.baseM);
            scheduler.UpdateNextProblem(SchedulerProblemShape{problemM, problemN, problemK, 0});
            if constexpr (HAS_ACTIVATION_QUANT) {
                // MX yScale is emitted per 64 N elements. Splitting the last N tile below that granularity can make
                // adjacent sub-tiles alias the same scale group or step past its row. Preserve the original fused
                // kernel's no-tail-split behavior while keeping the existing QGMM path unchanged.
                ProcessSingleGroup<false>(scheduler, groupIdx);
            } else {
                if (IsLastGroupAndNeedSplit(scheduler)) {
                    scheduler.UpdateTailTile();
                    ProcessSingleGroup<true>(scheduler, groupIdx);
                } else {
                    ProcessSingleGroup<false>(scheduler, groupIdx);
                }
            }
        }
        if constexpr (HAS_ACTIVATION_QUANT) {
            if ASCEND_IS_AIC {
                if (isVecSetSyncCom_) {
                    WaitForVector();
                }
            }
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        const auto& gmmParams = params.gmmParams;
        aBasePtr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        bBasePtr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
        cBasePtr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
        scaleABasePtr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr);
        scaleBBasePtr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.mmadParams.scaleBGmAddr);
        singleW_ = params.gmmParams.singleW == 1;
        if (gmmParams.isBias == 1) {
            biasBasePtr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
            isBias_ = true;
        }
        if (params.groupListGmAddr != nullptr) {
            groupListGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.groupListGmAddr));
        }
        const ProblemShape_ initProblemShape{gmmParams.m, gmmParams.n, gmmParams.k, 0};
        problemShape_ = initProblemShape;
        groupNum_ = gmmParams.groupNum;
        groupListType_ = gmmParams.groupListType;
        curBaseM_ = gmmParams.baseM;
        baseN_ = gmmParams.baseN;
        if constexpr (!TRANS_A) {
            if constexpr (!WEIGHT_NZ) {
                perGroupBOffset_ = gmmParams.n * gmmParams.k;
            } else {
                if constexpr (TRANS_B) {
                    perGroupBOffset_ = static_cast<int64_t>(
                        Align16(gmmParams.n) * (IsFp4<AType>() ? Align64(gmmParams.k) : Align32(gmmParams.k)));
                } else {
                    perGroupBOffset_ = static_cast<int64_t>(
                        (IsFp4<AType>() ? Align64(gmmParams.n) : Align32(gmmParams.n)) * Align16(gmmParams.k));
                }
            }
        }

        if ASCEND_IS_AIC {
            const BlockShape l0Shape{static_cast<int64_t>(gmmParams.baseM), static_cast<int64_t>(gmmParams.baseN),
                                     static_cast<int64_t>(gmmParams.baseK), 0};
            // MX uses one shared scale K window for ScaleA/ScaleB. Tiling must provide scaleKAL1 == scaleKBL1.
            const L1Params l1Params{static_cast<uint64_t>(gmmParams.kAL1), static_cast<uint64_t>(gmmParams.kBL1),
                                    static_cast<uint64_t>(gmmParams.scaleKAL1)};
            const typename BlockMmad_::MmadParams mmadParams{
                l0Shape, l1Params, isBias_, gmmParams.dbL0C == DOUBLE_BUFFER_COUNT, gmmParams.l1BufferStage};
            blockMmad_.Init(initProblemShape, mmadParams);
        }
        if constexpr (HAS_ACTIVATION_QUANT) {
            if ASCEND_IS_AIV {
                epilogueOp_.Init(params.epilogueParams);
            }
        }
    }

    __aicore__ inline void BaseMBalance(BlockScheduler_& scheduler, int64_t m, int64_t baseM)
    {
        if constexpr (!TRANS_A) {
            if (m <= 0) {
                return;
            }
            const int64_t safeBaseM = baseM > 0 ? baseM : static_cast<int64_t>(BLOCK_CUBE);
            const int64_t mCnt = (m + safeBaseM - 1) / safeBaseM;
            const int64_t balancedBaseM = (m + mCnt - 1) / mCnt;
            curBaseM_ = static_cast<uint32_t>((balancedBaseM + BLOCK_CUBE_MASK) & ~BLOCK_CUBE_MASK);
            scheduler.UpdateBaseM(curBaseM_);
        }
    }

    __aicore__ inline bool IsLastGroupAndNeedSplit(const BlockScheduler_& scheduler)
    {
        return (scheduler.GetEndBlockIdx() + 1) <= (AscendC::GetBlockNum() >> 1);
    }

    __aicore__ inline void SetMNK(uint32_t groupIdx)
    {
        const int64_t splitValue = GetSplitValueFromGroupList(groupIdx);
        const int64_t n = AscendC::Te::Get<MNK_N>(problemShape_);
        // Current MX scalar path selects the split axis from LayoutA, not from GMMTiling::groupType.
        if constexpr (!TRANS_A) {
            problemShape_ = ProblemShape_{splitValue, n, AscendC::Te::Get<MNK_K>(problemShape_), 0};
        } else {
            problemShape_ = ProblemShape_{AscendC::Te::Get<MNK_M>(problemShape_), n, splitValue, 0};
        }
    }

    __aicore__ inline int64_t GetSplitValueFromGroupList(uint32_t groupIdx)
    {
        int64_t splitValue = 0;
        if (groupListType_ == GROUP_LIST_TYPE_OFFSET) {
            const int64_t offset = groupListGlobal_.GetValue(groupIdx);
            splitValue = offset - preOffset_;
            preOffset_ = offset;
        } else if (groupListType_ == GROUP_LIST_TYPE_LENGTH) {
            splitValue = groupListGlobal_.GetValue(groupIdx);
            preOffset_ += splitValue;
        } else {
            const uint32_t splitValueIdx = groupIdx * SPARSE_GROUP_LIST_ITEM_STRIDE +
                                           SPARSE_GROUP_LIST_SPLIT_VALUE_OFFSET;
            splitValue = groupListGlobal_.GetValue(splitValueIdx);
            preOffset_ += splitValue;
        }
        return splitValue;
    }

    template <bool isLastGroupAndNeedSplit>
    __aicore__ inline void ProcessSingleGroup(BlockScheduler_& scheduler, uint32_t groupIdx)
    {
        const typename BlockScheduler_::MxGroupParams groupParams{
            groupIdx, preOffset_, perGroupBOffset_, problemShape_, singleW_, TRANS_A, IsFp4<AType>(), IsFp4<BType>()};
        scheduler.UpdateMxGroup(groupParams);
        typename BlockScheduler_::MxBlockInfo blockInfo;
        if (!scheduler.GetNextMxBlock(blockInfo)) {
            return;
        }
        const typename BlockScheduler_::MxGroupInfo& groupInfo = scheduler.GetMxGroupInfo();

        if ASCEND_IS_AIC {
            // Different groups may carry different split-M / split-K sizes. Keep the double-buffer phase continuous
            // across groups and only refresh the problem shape.
            blockMmad_.UpdateParamsForNextProblem(problemShape_);
        }
        if constexpr (HAS_ACTIVATION_QUANT) {
            if ASCEND_IS_AIV {
                epilogueOp_.UpdateNextProblem(problemShape_);
                epilogueOp_.UpdateGlobalAddr({groupInfo.outputOffset, groupInfo.outputScaleOffset});
            }
        }

        const int64_t problemM = AscendC::Te::Get<MNK_M>(problemShape_);
        const int64_t problemN = AscendC::Te::Get<MNK_N>(problemShape_);
        const int64_t problemK = AscendC::Te::Get<MNK_K>(problemShape_);
        const int64_t scaleK = groupInfo.inputScaleK;
        const int64_t baseN = static_cast<int64_t>(baseN_);
        auto layoutA = MakeLayoutA{}(problemM, problemK);
        auto layoutScaleA = MakeLayoutScaleA{}(problemM, scaleK);
        auto layoutB = MakeLayoutB{}(problemK, problemN);
        auto layoutScaleB = MakeLayoutScaleB{}(scaleK, problemN);
        auto layoutC = MakeLayoutC{}(problemM, problemN);
        auto gmA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aBasePtr_ + groupInfo.aOffset), layoutA);
        auto gmScaleA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleABasePtr_ + groupInfo.scaleAOffset), layoutScaleA);
        auto gmB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                (singleW_ ? bBasePtr_ : GetTensorAddrFromTensorList(groupIdx, bBasePtr_)) + groupInfo.bOffset),
            layoutB);
        auto gmScaleB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                (singleW_ ? scaleBBasePtr_ : GetTensorAddrFromTensorList(groupIdx, scaleBBasePtr_)) +
                groupInfo.scaleBOffset),
            layoutScaleB);
        auto gmC = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cBasePtr_ + groupInfo.outputOffset), layoutC);
        auto biasPtr = isBias_ ? (biasBasePtr_ + groupInfo.biasOffset) : nullptr;
        auto layoutBias = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(static_cast<int64_t>(1), problemN);
        auto gmBias = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasPtr), layoutBias);

        if constexpr (!isLastGroupAndNeedSplit) {
            if ASCEND_IS_AIC {
                SetL2CacheHint(gmB, gmScaleB, problemM, static_cast<int64_t>(curBaseM_), static_cast<int64_t>(baseN_));
            }
        }

        do {
            const int64_t blockM = AscendC::Te::Get<MNK_M>(blockInfo.blockShape);
            const int64_t blockN = AscendC::Te::Get<MNK_N>(blockInfo.blockShape);
            if (blockM <= 0 || blockN <= 0) {
                continue;
            }
            const int64_t blockK = problemK;
            BlockShape blockShape{blockM, blockN, blockK, 0};
            const int64_t mPos = blockInfo.mOffset;
            const int64_t nPos = blockInfo.nOffset;

            auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(mPos, static_cast<int64_t>(0)),
                                      AscendC::Te::MakeShape(blockM, blockK));
            auto gmBlockScaleA = gmScaleA.Slice(AscendC::Te::MakeCoord(mPos, static_cast<int64_t>(0)),
                                                AscendC::Te::MakeShape(blockM, scaleK));
            auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(static_cast<int64_t>(0), nPos),
                                      AscendC::Te::MakeShape(blockK, blockN));
            auto gmBlockScaleB = gmScaleB.Slice(AscendC::Te::MakeCoord(static_cast<int64_t>(0), nPos),
                                                AscendC::Te::MakeShape(scaleK, blockN));
            auto gmBlockBias = gmBias.Slice(AscendC::Te::MakeCoord(static_cast<int64_t>(0), nPos),
                                            AscendC::Te::MakeShape(static_cast<int64_t>(1), blockN));
            if constexpr (HAS_ACTIVATION_QUANT) {
                if ASCEND_IS_AIC {
                    if (isVecSetSyncCom_) {
                        WaitForVector();
                    }
                    auto ubC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(0),
                                                       AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(
                                                           (blockM + 1) & ~static_cast<int64_t>(1), Align32(blockN)));
                    blockMmad_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, ubC, blockShape);
                    NotifyVector();
                }
                isVecSetSyncCom_ = true;
                if ASCEND_IS_AIV {
                    WaitForCube();
                    epilogueOp_({blockM, blockN, static_cast<int64_t>(0), static_cast<int64_t>(0)},
                                {blockInfo.outputOffsets.outputOffset, blockInfo.outputOffsets.outputScaleOffset});
                    NotifyCube();
                }
            } else {
                auto gmBlockC = gmC.Slice(AscendC::Te::MakeCoord(mPos, nPos), AscendC::Te::MakeShape(blockM, blockN));
                blockMmad_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, blockShape);
            }
        } while (scheduler.GetNextMxBlock(blockInfo));
    }

private:
    BlockMmad_ blockMmad_;
    BlockEpilogue_ epilogueOp_;
    ProblemShape_ problemShape_{};
    AscendC::GlobalTensor<int64_t> groupListGlobal_;

    __gm__ AType* aBasePtr_{nullptr};
    __gm__ BType* bBasePtr_{nullptr};
    __gm__ CType* cBasePtr_{nullptr};
    __gm__ BiasType* biasBasePtr_{nullptr};
    __gm__ fp8_e8m0_t* scaleABasePtr_{nullptr};
    __gm__ fp8_e8m0_t* scaleBBasePtr_{nullptr};

    int64_t preOffset_{0};
    int64_t perGroupBOffset_{0};
    uint32_t groupNum_{0};
    uint32_t curBaseM_{0};
    uint32_t baseN_{0};
    uint8_t groupListType_{GROUP_LIST_TYPE_OFFSET};
    bool isBias_{false};
    bool singleW_{true};
    bool isVecSetSyncCom_{false};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
// clang-format on
