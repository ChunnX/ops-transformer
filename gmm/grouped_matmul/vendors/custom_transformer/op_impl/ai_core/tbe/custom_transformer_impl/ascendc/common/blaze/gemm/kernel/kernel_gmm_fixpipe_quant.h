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
 * \file kernel_gmm_fixpipe_quant.h
 * \brief Grouped int8 Cube-input per-channel/per-group kernel using FixPipe quant and an AIV epilogue.
 *
 * The operator repository supplies int8 Cube inputs and optional asymmetric-quantization workspaces.
 * AIC writes the FixPipe result to a per-core workspace slot. AIV applies per-token scaling and,
 * when withOffset is enabled, the row-sum/offset correction before storing the final output.
 */
#pragma once

#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/epilogue/block/block_epilogue_per_token_scale.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {
namespace {
constexpr uint8_t GMM_GROUP_LIST_OFFSET = 0U;
constexpr uint8_t GMM_GROUP_LIST_LENGTH = 1U;
constexpr uint8_t GMM_GROUP_LIST_SPARSE = 2U;
constexpr uint32_t GMM_SPARSE_STRIDE = 2U;
constexpr uint32_t GMM_SPARSE_VALUE_OFFSET = 1U;
constexpr int64_t GMM_BLOCK_MASK = BLOCK_CUBE - 1;
} // namespace

#define GMM_FIXPIPE_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define GMM_FIXPIPE_TEM_PARAMS                                                                    \
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,                                       \
        AscendC::Std::enable_if_t<AscendC::Std::is_same_v<KernelGroupedMmadWithScaleFixpipeQuant, \
                                                          typename BlockMmad::DispatchPolicy::ScheduleType>>
#define GMM_FIXPIPE_TEMPLATE_DEF \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>

GMM_FIXPIPE_CLASS_TEM_PARAMS
class GemmUniversal<GMM_FIXPIPE_TEM_PARAMS> {
public:
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using ScaleType = typename BlockMmad::X2ScaleType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using SchedulerProblemShape = typename BlockScheduler::ProblemShape;
    using SchedulerBlockInfo = typename BlockScheduler::BlockInfo;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    static_assert(AscendC::IsSameType<AType, int8_t>::value);
    static_assert(AscendC::IsSameType<BType, int8_t>::value);
    static_assert(AscendC::IsSameType<ScaleType, uint64_t>::value);
    static_assert(AscendC::IsSameType<CType, half>::value);
    static_assert(AscendC::IsSameType<typename BlockEpilogue::FixpipeType, CType>::value);

    struct GMMTiling {
        uint32_t groupNum{0};
        int64_t m{0};
        int64_t n{0};
        int64_t k{0};
        uint32_t baseM{0};
        uint32_t baseN{0};
        uint32_t baseK{0};
        // K elements sharing one [N] scale vector; 0 falls back to baseK.
        uint32_t quantGroupSize{0};
        uint32_t quantMode{static_cast<uint32_t>(QuantMode::PERCHANNEL_MODE)};
        uint32_t kAL1{0};
        uint32_t kBL1{0};
        uint32_t nBufferNum{0};
        uint8_t dbL0C{0};
        uint8_t groupListType{GMM_GROUP_LIST_OFFSET};
    };

    struct Params {
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        GM_ADDR groupListGmAddr{nullptr};
        GMMTiling gmmParams;
    };

    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}
    __aicore__ inline void operator()(const Params& params) { Run(params); }

private:
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
    static constexpr int64_t C0_SIZE = AscendC::Te::C0_ELEMENT<AType>;
    // This kernel uses direct Cube/Vector pairing (sync mode 2). In this mode one
    // flag operation synchronizes the paired Cube and Vector task; unlike mode
    // 4 it must not explicitly signal the second Vector sub-block with +16.
    static constexpr uint16_t GMM_SYNC_MODE = 2;
    // Keep the flag allocation consistent with the A5 grouped_matmul mode-2
    // implementation in grouped_matmul_utils.h.
    static constexpr uint16_t AIC_TO_AIV_FLAG = 5;
    static constexpr uint16_t AIV_TO_AIC_FLAG = 3;
    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<LayoutC, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<CType>>>;

    __aicore__ inline void Run(const Params& params);
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline int64_t GetPerGroupBOffset(int64_t n, int64_t k) const;
    __aicore__ inline uint32_t GetGroupIdx(uint32_t loopIdx);
    __aicore__ inline int64_t GetGroupM(uint32_t loopIdx);
    __aicore__ inline void SetGroupShape(uint32_t loopIdx);
    __aicore__ inline bool IsValidProblem() const;
    __aicore__ inline void PrepareScheduler(BlockScheduler& scheduler);
    __aicore__ inline void SetSchedulerTailAlign(BlockScheduler& scheduler);
    __aicore__ inline bool IsLastGroupAndNeedSplit(const BlockScheduler& scheduler) const;
    __aicore__ inline void ProcessSingleGroup(BlockScheduler& scheduler, uint32_t groupIdx);

    template <class TensorA, class TensorB>
    __aicore__ inline void ProcessOneBlock(const TensorA& gmA, const TensorB& gmB, const BlockShape& blockShape,
                                           uint32_t groupIdx, int64_t groupMOffset, int64_t mPos, int64_t nPos,
                                           int64_t curM, int64_t curN, int64_t k, int64_t n);
    __aicore__ inline void NotifyVector() { AscendC::CrossCoreSetFlag<GMM_SYNC_MODE, PIPE_FIX>(AIC_TO_AIV_FLAG); }
    __aicore__ inline void WaitForVector() { AscendC::CrossCoreWaitFlag(AIV_TO_AIC_FLAG); }
    __aicore__ inline void NotifyCube() { AscendC::CrossCoreSetFlag<GMM_SYNC_MODE, PIPE_MTE2>(AIV_TO_AIC_FLAG); }
    __aicore__ inline void WaitForCube() { AscendC::CrossCoreWaitFlag(AIC_TO_AIV_FLAG); }

    BlockMmad mmOp_;
    BlockEpilogue epilogueOp_;
    ProblemShape problemShape_{};
    AscendC::GlobalTensor<int64_t> groupListGlobal_;
    __gm__ AType* aBasePtr_{nullptr};
    __gm__ BType* bBasePtr_{nullptr};
    __gm__ CType* workspaceBasePtr_{nullptr};
    __gm__ ScaleType* scaleBasePtr_{nullptr};
    int64_t preOffset_{0};
    int64_t perGroupBOffset_{0};
    uint32_t groupNum_{0};
    uint32_t curBaseM_{0};
    uint32_t baseM_{0};
    uint32_t baseN_{0};
    uint32_t quantGroupSize_{0};
    uint32_t quantGroupNum_{1};
    QuantMode quantMode_{QuantMode::PERCHANNEL_MODE};
    uint8_t groupListType_{GMM_GROUP_LIST_OFFSET};
    bool isPerGroup_{false};
    bool isFirstBlock_{true};
};

GMM_FIXPIPE_TEMPLATE_DEF
__aicore__ inline void GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::Run(const Params& params)
{
    if ASCEND_IS_AIC {
        AscendC::SetMMLayoutTransform(true);
    }

    Init(params);
    if (groupNum_ == 0) {
        return;
    }

    const auto& tiling = params.gmmParams;
    BlockScheduler scheduler(baseM_, baseN_, tiling.baseK);
    SetSchedulerTailAlign(scheduler);
    const uint32_t lastLoopIdx = groupNum_ - 1;
    for (uint32_t loopIdx = 0; loopIdx < lastLoopIdx; ++loopIdx) {
        const uint32_t groupIdx = GetGroupIdx(loopIdx);
        SetGroupShape(loopIdx);
        if (!IsValidProblem()) {
            if (groupListType_ == GMM_GROUP_LIST_SPARSE && AscendC::Te::Get<MNK_M>(problemShape_) <= 0) {
                break;
            }
            continue;
        }
        PrepareScheduler(scheduler);
        ProcessSingleGroup(scheduler, groupIdx);
    }

    const uint32_t groupIdx = GetGroupIdx(lastLoopIdx);
    SetGroupShape(lastLoopIdx);
    if (IsValidProblem()) {
        PrepareScheduler(scheduler);
        if (IsLastGroupAndNeedSplit(scheduler)) {
            scheduler.UpdateTailTile();
        }
        ProcessSingleGroup(scheduler, groupIdx);
    }
    if ASCEND_IS_AIC {
        if (!isFirstBlock_) {
            WaitForVector();
        }
    }
}

GMM_FIXPIPE_TEMPLATE_DEF
__aicore__ inline void GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::Init(const Params& params)
{
    const auto& tiling = params.gmmParams;
    problemShape_ = ProblemShape{tiling.m, tiling.n, tiling.k, 0};
    groupNum_ = tiling.groupNum;
    groupListType_ = tiling.groupListType;
    baseM_ = tiling.baseM;
    baseN_ = tiling.baseN;
    curBaseM_ = baseM_;
    isPerGroup_ = static_cast<QuantMode>(tiling.quantMode) == QuantMode::PERGROUP_MODE;
    quantMode_ = isPerGroup_ ? QuantMode::PERGROUP_MODE : QuantMode::PERCHANNEL_MODE;
    const int64_t problemK = tiling.k > 0 ? tiling.k : 0;
    quantGroupSize_ = isPerGroup_ ? (tiling.quantGroupSize > 0 ?
                                         tiling.quantGroupSize :
                                         (tiling.baseK > 0 ? tiling.baseK : static_cast<uint32_t>(problemK))) :
                                    static_cast<uint32_t>(problemK);
    quantGroupNum_ = isPerGroup_ && quantGroupSize_ > 0 && problemK > 0 ?
                         static_cast<uint32_t>(CeilDiv(problemK, static_cast<int64_t>(quantGroupSize_))) :
                         1;
    if (params.groupListGmAddr != nullptr) {
        groupListGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.groupListGmAddr));
    }
    workspaceBasePtr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    if ASCEND_IS_AIC {
        aBasePtr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        bBasePtr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
        scaleBasePtr_ = reinterpret_cast<__gm__ ScaleType*>(params.mmadParams.scaleBGmAddr);
        perGroupBOffset_ = GetPerGroupBOffset(tiling.n, tiling.k);
        const int64_t mmK = isPerGroup_ ? Min(problemK, static_cast<int64_t>(quantGroupSize_)) : problemK;
        const BlockShape mmProblemShape{tiling.m, tiling.n, mmK, 0};
        const BlockShape l0Shape{static_cast<int64_t>(baseM_), static_cast<int64_t>(baseN_),
                                 Min(static_cast<int64_t>(tiling.baseK), mmK), 0};
        mmOp_.Init(mmProblemShape, l0Shape, tiling.kAL1, tiling.kBL1, tiling.nBufferNum, quantMode_, false,
                   tiling.dbL0C > 1);
    }
    if ASCEND_IS_AIV {
        epilogueOp_.Init(params.epilogueParams);
    }
}

GMM_FIXPIPE_TEMPLATE_DEF
__aicore__ inline int64_t GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::GetPerGroupBOffset(int64_t n, int64_t k) const
{
    if constexpr (!WEIGHT_NZ) {
        return n * k;
    } else if constexpr (TRANS_B) {
        return Align16(n) * Align32(k);
    } else {
        return Align32(n) * Align16(k);
    }
}

GMM_FIXPIPE_TEMPLATE_DEF
__aicore__ inline uint32_t GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::GetGroupIdx(uint32_t loopIdx)
{
    if (groupListType_ == GMM_GROUP_LIST_SPARSE) {
        return static_cast<uint32_t>(groupListGlobal_.GetValue(loopIdx * GMM_SPARSE_STRIDE));
    }
    return loopIdx;
}

GMM_FIXPIPE_TEMPLATE_DEF
__aicore__ inline int64_t GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::GetGroupM(uint32_t loopIdx)
{
    int64_t groupM = 0;
    if (groupListType_ == GMM_GROUP_LIST_OFFSET) {
        const int64_t offset = groupListGlobal_.GetValue(loopIdx);
        groupM = offset - preOffset_;
        preOffset_ = offset;
    } else if (groupListType_ == GMM_GROUP_LIST_LENGTH) {
        groupM = groupListGlobal_.GetValue(loopIdx);
        preOffset_ += groupM;
    } else {
        groupM = groupListGlobal_.GetValue(loopIdx * GMM_SPARSE_STRIDE + GMM_SPARSE_VALUE_OFFSET);
        preOffset_ += groupM;
    }
    return groupM;
}

GMM_FIXPIPE_TEMPLATE_DEF
__aicore__ inline void GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::SetGroupShape(uint32_t loopIdx)
{
    const int64_t groupM = GetGroupM(loopIdx);
    problemShape_ = ProblemShape{groupM, AscendC::Te::Get<MNK_N>(problemShape_), AscendC::Te::Get<MNK_K>(problemShape_),
                                 0};
}

GMM_FIXPIPE_TEMPLATE_DEF
__aicore__ inline bool GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::IsValidProblem() const
{
    return AscendC::Te::Get<MNK_M>(problemShape_) > 0 && AscendC::Te::Get<MNK_N>(problemShape_) > 0 &&
           AscendC::Te::Get<MNK_K>(problemShape_) > 0;
}

GMM_FIXPIPE_TEMPLATE_DEF
__aicore__ inline void GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::PrepareScheduler(BlockScheduler& scheduler)
{
    const int64_t m = AscendC::Te::Get<MNK_M>(problemShape_);
    const int64_t n = AscendC::Te::Get<MNK_N>(problemShape_);
    const int64_t k = AscendC::Te::Get<MNK_K>(problemShape_);
    const int64_t safeBaseM = baseM_ > 0 ? baseM_ : static_cast<int64_t>(BLOCK_CUBE);
    const int64_t blockCount = CeilDiv(m, safeBaseM);
    const int64_t balancedBaseM = CeilDiv(m, blockCount);
    curBaseM_ = static_cast<uint32_t>((balancedBaseM + GMM_BLOCK_MASK) & ~GMM_BLOCK_MASK);
    scheduler.UpdateBaseM(curBaseM_);
    scheduler.UpdateNextProblem(SchedulerProblemShape{m, n, k, 0});
}

GMM_FIXPIPE_TEMPLATE_DEF
__aicore__ inline void GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::SetSchedulerTailAlign(BlockScheduler& scheduler)
{
    constexpr uint32_t mTailAlign = 1;
    constexpr uint32_t nTailAlign = TRANS_B ? static_cast<uint32_t>(BLOCK_CUBE) : static_cast<uint32_t>(C0_SIZE);
    scheduler.SetTailAlign(mTailAlign, nTailAlign);
}

GMM_FIXPIPE_TEMPLATE_DEF
__aicore__ inline bool GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::IsLastGroupAndNeedSplit(
    const BlockScheduler& scheduler) const
{
    return (scheduler.GetEndBlockIdx() + 1) <= (AscendC::GetBlockNum() >> 1);
}

GMM_FIXPIPE_TEMPLATE_DEF
__aicore__ inline void GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::ProcessSingleGroup(BlockScheduler& scheduler,
                                                                                 uint32_t groupIdx)
{
    const int64_t m = AscendC::Te::Get<MNK_M>(problemShape_);
    const int64_t n = AscendC::Te::Get<MNK_N>(problemShape_);
    const int64_t k = AscendC::Te::Get<MNK_K>(problemShape_);
    const int64_t groupMOffset = preOffset_ - m;

    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutB = MakeLayoutB{}(k, n);
    auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aBasePtr_ + groupMOffset * k),
                                       layoutA);
    auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                                           bBasePtr_ + static_cast<int64_t>(groupIdx) * perGroupBOffset_),
                                       layoutB);
    SchedulerBlockInfo blockInfo;
    while (scheduler.GetNextBlock(blockInfo)) {
        const int64_t curM = AscendC::Te::Get<MNK_M>(blockInfo.blockShape);
        const int64_t curN = AscendC::Te::Get<MNK_N>(blockInfo.blockShape);
        if (curM <= 0 || curN <= 0) {
            continue;
        }

        const int64_t mPos = AscendC::Te::Get<MNK_M>(blockInfo.blockCoord);
        const int64_t nPos = AscendC::Te::Get<MNK_N>(blockInfo.blockCoord);
        const BlockShape blockShape{curM, curN, k, 0};
        ProcessOneBlock(gmA, gmB, blockShape, groupIdx, groupMOffset, mPos, nPos, curM, curN, k, n);
    }
}

GMM_FIXPIPE_TEMPLATE_DEF
template <class TensorA, class TensorB>
__aicore__ inline void GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::ProcessOneBlock(const TensorA& gmA, const TensorB& gmB,
                                                                              const BlockShape& blockShape,
                                                                              uint32_t groupIdx, int64_t groupMOffset,
                                                                              int64_t mPos, int64_t nPos, int64_t curM,
                                                                              int64_t curN, int64_t k, int64_t n)
{
    const uint64_t workspaceOffset = static_cast<uint64_t>(AscendC::GetBlockIdx() / AscendC::GetTaskRation()) *
                                     static_cast<uint64_t>(baseN_) * baseM_;
    if ASCEND_IS_AIC {
        if (!isFirstBlock_) {
            WaitForVector();
        }
        auto workspaceLayout = MakeLayoutC{}(curM, curN);
        auto gmWorkspace = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(workspaceBasePtr_ + workspaceOffset), workspaceLayout);
        // BlockMmad keeps a bias tensor in its call signature. Init uses isBias=false,
        // so this placeholder is never loaded or involved in the calculation.
        auto unusedBiasLayout = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1L, 1L);
        auto gmUnusedBias = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(reinterpret_cast<__gm__ BiasType*>(workspaceBasePtr_)),
            unusedBiasLayout);

        auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(mPos, static_cast<int64_t>(0)),
                                  AscendC::Te::MakeShape(curM, k));
        auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(static_cast<int64_t>(0), nPos),
                                  AscendC::Te::MakeShape(k, curN));
        const int64_t scaleExpertOffset = static_cast<int64_t>(groupIdx) * quantGroupNum_ * n;
        auto scaleLayout = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(
            static_cast<int64_t>(quantGroupNum_), n);
        auto gmScale = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleBasePtr_ + scaleExpertOffset), scaleLayout);
        auto gmBlockScale = gmScale.Slice(AscendC::Te::MakeCoord(static_cast<int64_t>(0), nPos),
                                          AscendC::Te::MakeShape(static_cast<int64_t>(quantGroupNum_), curN));
        if (isPerGroup_) {
            mmOp_(gmBlockA, gmBlockB, gmBlockScale, gmUnusedBias, gmWorkspace, blockShape, quantGroupSize_,
                  quantGroupNum_);
        } else {
            mmOp_(gmBlockA, gmBlockB, gmBlockScale, gmUnusedBias, gmWorkspace, blockShape);
        }
        NotifyVector();
        isFirstBlock_ = false;
    }

    if ASCEND_IS_AIV {
        WaitForCube();
        const int64_t offsetPerTokenScale = groupMOffset + mPos;
        const int64_t offsetOffset = static_cast<int64_t>(groupIdx) * n + nPos;
        const int64_t offsetRowSum = offsetPerTokenScale;
        const int64_t offsetC = (groupMOffset + mPos) * n + nPos;
        epilogueOp_(curM, curN, offsetPerTokenScale, offsetOffset, offsetRowSum, offsetC, workspaceOffset);
        NotifyCube();
    }
}

#undef GMM_FIXPIPE_TEMPLATE_DEF
#undef GMM_FIXPIPE_TEM_PARAMS
#undef GMM_FIXPIPE_CLASS_TEM_PARAMS

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
