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
 * \file kernel_gmm_finalize_routing_tensor_api.h
 * \brief Tensor API MIX kernel for grouped matmul finalize routing MX quantization.
 */

#pragma once

#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {

struct KernelGmmFinalizeRoutingTensorApiMixMx {};

namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelGmmFinalizeRoutingTensorApiMixMx, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    static_assert(AscendC::Std::is_same_v<CType, typename BlockEpilogue::DataTypeIn>,
                  "BlockMmad UB output type must match BlockEpilogue input type.");

    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;

    using BlockMmadAddressParams = typename BlockMmad::Params;
    using BlockMmadInitParams = typename BlockMmad::MmadParams;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using L1Params = typename BlockMmad::L1Params;
    using BlockShape = typename BlockMmad::BlockShape;
    using SchedulerProblemShape = typename BlockScheduler::ProblemShape;
    using SchedulerBlockInfo = typename BlockScheduler::BlockInfo;
    using EpilogueProblemShape = typename BlockEpilogue::ProblemShape;
    using EpilogueBlockShape = typename BlockEpilogue::BlockShape;
    using EpilogueOutputOffsets = typename BlockEpilogue::OutputOffsets;

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
        uint32_t scaleKL1;
        uint8_t dbL0C;
        uint8_t groupListType;
        uint8_t hasBias;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadAddressParams blockMmadAddressParams;
        BlockEpilogueParams epilogueParams;
        GM_ADDR groupListGmAddr;
        GMMTiling gmmParams;
    };

    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    __aicore__ inline void operator()(const Params &params)
    {
        Run(params);
    }

private:
    static constexpr uint16_t FLAG_ID_MAX = 16;
    static constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
    static constexpr int64_t SCALE_CACHE_MASK = 0xff;
    static constexpr uint64_t INPUT_C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;

    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<INPUT_C0_SIZE>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<INPUT_C0_SIZE>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        TRANS_A, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleADNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        TRANS_B, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;

    __aicore__ inline void NotifyVector()
    {
        if ASCEND_IS_AIC {
            AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
            AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
        }
    }

    __aicore__ inline void NotifyCube()
    {
        if ASCEND_IS_AIV {
            AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG);
        }
    }

    __aicore__ inline void WaitForVector()
    {
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
            AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX);
        }
    }

    __aicore__ inline void WaitForCube()
    {
        if ASCEND_IS_AIV {
            AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG);
        }
    }

    __aicore__ inline void WaitForPrologue()
    {
        if ASCEND_IS_AIV {
            AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
        }
        if ASCEND_IS_AIC {
            WaitForVector();
        }
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIC {
            if (isVecSetSyncCom_) {
                WaitForVector();
            }
        }
    }

    __aicore__ inline void SetSchedulerTailAlign(BlockScheduler &scheduler)
    {
        constexpr uint32_t mTailAlign = 1;
        constexpr uint32_t nTailAlign =
            TRANS_B ? static_cast<uint32_t>(BLOCK_CUBE) : static_cast<uint32_t>(INPUT_C0_SIZE);
        scheduler.SetTailAlign(mTailAlign, nTailAlign);
    }

    __aicore__ inline void BaseMBalance(BlockScheduler &scheduler, int64_t m, int64_t baseM)
    {
        const int64_t safeBaseM = baseM > 0 ? baseM : static_cast<int64_t>(BLOCK_CUBE);
        curBaseM_ = static_cast<uint32_t>(safeBaseM);
        scheduler.UpdateBaseM(curBaseM_);
        if (m <= 0) {
            return;
        }
        const int64_t mCnt = CeilDiv(m, safeBaseM);
        const int64_t balancedBaseM = CeilDiv(m, mCnt);
        curBaseM_ = static_cast<uint32_t>(CeilAlign(balancedBaseM, static_cast<int64_t>(BLOCK_CUBE)));
        scheduler.UpdateBaseM(curBaseM_);
    }

    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2CacheHint(TensorB &gmB, TensorScaleB &gmScaleB, int64_t mSize)
    {
        const int64_t problemN = AscendC::Te::Get<MNK_N>(problemShape_);
        const int64_t problemK = AscendC::Te::Get<MNK_K>(problemShape_);
        if constexpr (TRANS_B) {
            if (curBaseM_ >= mSize && (problemK & SCALE_CACHE_MASK) == 0) {
                gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
            } else {
                gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
            }
        } else {
            if (curBaseM_ >= mSize && (problemN & SCALE_CACHE_MASK) == 0 &&
                (static_cast<int64_t>(baseN_) & SCALE_CACHE_MASK) == 0) {
                gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
            } else {
                gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
            }
        }
    }

    __aicore__ inline int64_t GetScaleK(int64_t k) const
    {
        return CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    }

    __aicore__ inline int64_t GetPackedOffset(int64_t elementOffset) const
    {
        if constexpr (IsFp4<AType>()) {
            return elementOffset >> 1;
        }
        return elementOffset;
    }

    __aicore__ inline int64_t GetPackedWeightOffset(int64_t elementOffset) const
    {
        if constexpr (IsFp4<BType>()) {
            return elementOffset >> 1;
        }
        return elementOffset;
    }

    __aicore__ inline int64_t GetGroupM(uint32_t groupIdx, int64_t &groupMOffset)
    {
        groupMOffset = preOffset_;
        const int64_t groupValue = groupListGlobal_.GetValue(groupIdx);
        if (groupListType_ == 0) {
            preOffset_ = groupValue;
            return groupValue - groupMOffset;
        }
        preOffset_ += groupValue;
        return groupValue;
    }

    __aicore__ inline bool IsLastGroupAndNeedSplit(const BlockScheduler &scheduler) const
    {
        return (scheduler.GetEndBlockIdx() + 1) <= (AscendC::GetBlockNum() >> 1);
    }

    __aicore__ inline void Init(const Params &params)
    {
        const auto &gmmParams = params.gmmParams;
        aBasePtr_ = reinterpret_cast<__gm__ AType *>(params.blockMmadAddressParams.aGmAddr);
        bBasePtr_ = reinterpret_cast<__gm__ BType *>(params.blockMmadAddressParams.bGmAddr);
        biasBasePtr_ = reinterpret_cast<__gm__ BiasType *>(params.blockMmadAddressParams.biasGmAddr);
        scaleABasePtr_ = reinterpret_cast<__gm__ fp8_e8m0_t *>(params.blockMmadAddressParams.scaleAGmAddr);
        scaleBBasePtr_ = reinterpret_cast<__gm__ fp8_e8m0_t *>(params.blockMmadAddressParams.scaleBGmAddr);
        groupListGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(params.groupListGmAddr));

        problemShape_ = ProblemShape{gmmParams.m, gmmParams.n, gmmParams.k, 0};
        groupNum_ = gmmParams.groupNum;
        groupListType_ = gmmParams.groupListType;
        curBaseM_ = gmmParams.baseM > 0 ? gmmParams.baseM : static_cast<uint32_t>(BLOCK_CUBE);
        baseN_ = gmmParams.baseN;
        hasBias_ = gmmParams.hasBias == 1;

        if ASCEND_IS_AIC {
            const BlockShape l0TileShape{static_cast<int64_t>(curBaseM_), static_cast<int64_t>(gmmParams.baseN),
                                         static_cast<int64_t>(gmmParams.baseK), 0};
            const L1Params l1Params{static_cast<uint64_t>(gmmParams.kAL1), static_cast<uint64_t>(gmmParams.kBL1),
                                    static_cast<uint64_t>(gmmParams.scaleKL1)};
            const BlockMmadInitParams blockMmadInitParams{l0TileShape, l1Params, hasBias_,
                                                          gmmParams.dbL0C == DOUBLE_BUFFER_COUNT};
            blockMmad_.Init(problemShape_, blockMmadInitParams);
        }
        if ASCEND_IS_AIV {
            blockEpilogue_.Init(params.epilogueParams);
        }
    }

    __aicore__ inline void Run(const Params &params)
    {
        Init(params);
        WaitForPrologue();
        if (groupNum_ == 0) {
            return;
        }

        const auto &gmmParams = params.gmmParams;
        BlockScheduler scheduler(gmmParams.baseM, gmmParams.baseN, gmmParams.baseK);
        SetSchedulerTailAlign(scheduler);
        for (uint32_t groupIdx = 0; groupIdx < groupNum_; ++groupIdx) {
            int64_t groupMOffset = 0;
            const int64_t groupM = GetGroupM(groupIdx, groupMOffset);
            if (groupM <= 0) {
                continue;
            }
            problemShape_ = ProblemShape{groupM, gmmParams.n, gmmParams.k, 0};
            BaseMBalance(scheduler, groupM, gmmParams.baseM);
            scheduler.UpdateNextProblem(SchedulerProblemShape{groupM, gmmParams.n, gmmParams.k, 0});
            if (groupIdx == groupNum_ - 1 && IsLastGroupAndNeedSplit(scheduler)) {
                scheduler.UpdateTailTile();
            }
            ProcessSingleGroup(scheduler, groupIdx, groupMOffset);
        }
        End();
    }

    __aicore__ inline void ProcessSingleGroup(BlockScheduler &scheduler, uint32_t groupIdx, int64_t groupMOffset)
    {
        const int64_t problemM = AscendC::Te::Get<MNK_M>(problemShape_);
        const int64_t problemN = AscendC::Te::Get<MNK_N>(problemShape_);
        const int64_t problemK = AscendC::Te::Get<MNK_K>(problemShape_);
        const int64_t scaleK = GetScaleK(problemK);

        if ASCEND_IS_AIC {
            blockMmad_.UpdateParamsForNextProblem(problemShape_);
        }
        if ASCEND_IS_AIV {
            blockEpilogue_.UpdateNextProblem(EpilogueProblemShape{problemM, problemN, problemK});
            blockEpilogue_.UpdateGlobalAddr(EpilogueOutputOffsets{0, groupMOffset});
        }

        auto layoutA = MakeLayoutA{}(problemM, problemK);
        auto layoutScaleA = MakeLayoutScaleA{}(problemM, scaleK);
        auto layoutB = MakeLayoutB{}(problemK, problemN);
        auto layoutScaleB = MakeLayoutScaleB{}(scaleK, problemN);
        auto layoutBias = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(static_cast<int64_t>(1), problemN);

        const int64_t aOffset = GetPackedOffset(groupMOffset * problemK);
        const int64_t bOffset = GetPackedWeightOffset(static_cast<int64_t>(groupIdx) * problemN * problemK);
        const int64_t scaleAOffset = groupMOffset * scaleK;
        const int64_t scaleBOffset = static_cast<int64_t>(groupIdx) * problemN * scaleK;
        const int64_t biasOffset = static_cast<int64_t>(groupIdx) * problemN;
        auto gmA =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aBasePtr_ + aOffset), layoutA);
        auto gmScaleA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleABasePtr_ + scaleAOffset), layoutScaleA);
        auto gmB =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bBasePtr_ + bOffset), layoutB);
        auto gmScaleB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleBBasePtr_ + scaleBOffset), layoutScaleB);
        auto biasPtr = hasBias_ ? biasBasePtr_ + biasOffset : nullptr;
        auto gmBias = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasPtr), layoutBias);
        if ASCEND_IS_AIC {
            SetL2CacheHint(gmB, gmScaleB, problemM);
        }

        SchedulerBlockInfo blockInfo;
        while (scheduler.GetNextBlock(blockInfo)) {
            const int64_t blockM = AscendC::Te::Get<MNK_M>(blockInfo.blockShape);
            const int64_t blockN = AscendC::Te::Get<MNK_N>(blockInfo.blockShape);
            if (blockM <= 0 || blockN <= 0) {
                continue;
            }
            const int64_t mOffset = AscendC::Te::Get<MNK_M>(blockInfo.blockCoord);
            const int64_t nOffset = AscendC::Te::Get<MNK_N>(blockInfo.blockCoord);
            const BlockShape blockShape{blockM, blockN, problemK, 0};

            auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(mOffset, static_cast<int64_t>(0)),
                                      AscendC::Te::MakeShape(blockM, problemK));
            auto gmBlockScaleA = gmScaleA.Slice(AscendC::Te::MakeCoord(mOffset, static_cast<int64_t>(0)),
                                                AscendC::Te::MakeShape(blockM, scaleK));
            auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(static_cast<int64_t>(0), nOffset),
                                      AscendC::Te::MakeShape(problemK, blockN));
            auto gmBlockScaleB = gmScaleB.Slice(AscendC::Te::MakeCoord(static_cast<int64_t>(0), nOffset),
                                                AscendC::Te::MakeShape(scaleK, blockN));
            auto gmBlockBias = gmBias.Slice(AscendC::Te::MakeCoord(static_cast<int64_t>(0), nOffset),
                                            AscendC::Te::MakeShape(static_cast<int64_t>(1), blockN));

            if ASCEND_IS_AIC {
                if (isVecSetSyncCom_) {
                    WaitForVector();
                }
                auto ubOut = blockEpilogue_.GetL0c2UbTensor(blockM, blockN);
                blockMmad_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, ubOut, blockShape);
                NotifyVector();
            }
            isVecSetSyncCom_ = true;

            if ASCEND_IS_AIV {
                WaitForCube();
                blockEpilogue_(EpilogueBlockShape{blockM, blockN, 0, 0}, EpilogueOutputOffsets{nOffset, mOffset});
                NotifyCube();
            }
        }
    }

    BlockMmad blockMmad_;
    BlockEpilogue blockEpilogue_;
    ProblemShape problemShape_{};
    AscendC::GlobalTensor<int64_t> groupListGlobal_;

    __gm__ AType *aBasePtr_{nullptr};
    __gm__ BType *bBasePtr_{nullptr};
    __gm__ BiasType *biasBasePtr_{nullptr};
    __gm__ fp8_e8m0_t *scaleABasePtr_{nullptr};
    __gm__ fp8_e8m0_t *scaleBBasePtr_{nullptr};

    int64_t preOffset_{0};
    uint32_t groupNum_{0};
    uint32_t curBaseM_{0};
    uint32_t baseN_{0};
    uint8_t groupListType_{0};
    bool hasBias_{false};
    bool isVecSetSyncCom_{false};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
