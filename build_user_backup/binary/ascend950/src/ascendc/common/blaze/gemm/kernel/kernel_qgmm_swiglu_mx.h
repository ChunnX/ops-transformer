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
 * \file kernel_qgmm_swiglu_mx.h
 * \brief
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/epilogue/block/block_epilogue_swiglu_mx_quant.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

namespace {
constexpr int64_t SCALE_CACHE_MASK = 0xff;
constexpr uint16_t FLAG_ID_MAX = 16;
constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
} // namespace

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelGmmSwiGluMixMx, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
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
    static_assert(AscendC::IsSameType<CType, typename BlockEpilogue::DataTypeIn>::value,
                  "BlockMmad UB output type must match BlockEpilogue input type.");
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;

    using BlockMmadAddressParams = typename BlockMmad::Params;
    using BlockMmadInitParams = typename BlockMmad::MmadParams;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using L1Params = typename BlockMmad::L1Params;
    using BlockShape = typename BlockMmad::BlockShape;
    using SchedulerShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

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
        uint32_t scaleKAL1;
        uint32_t scaleKBL1;
        uint8_t dbL0C;
        int8_t groupType;
        uint8_t groupListType;
        uint8_t singleW;
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

    __aicore__ inline void operator()(const Params& params) { Run(params); }

private:
    static constexpr uint64_t INPUT_C0_SIZE = Blaze::Gemm::C0_SIZE_B8;
    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<INPUT_C0_SIZE>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<INPUT_C0_SIZE>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        TRANS_A, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleADNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        TRANS_B, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;
    __aicore__ inline void SyncAicToAiv()
    {
        if ASCEND_IS_AIC {
            AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
            AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
        }
    }

    __aicore__ inline void SyncAivToAic()
    {
        if ASCEND_IS_AIV {
            AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG);
        }
    }

    __aicore__ inline void WaitForAiv()
    {
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
            AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX);
        }
    }

    __aicore__ inline void WaitForAic()
    {
        if ASCEND_IS_AIV {
            AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG);
        }
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIC {
            if (isVecSetSyncCom_) {
                WaitForAiv();
            }
        }
    }

    __aicore__ inline void SetSchedulerTailAlign(BlockScheduler& bs)
    {
        uint32_t mTailAlign = 1;
        uint32_t nTailAlign = TRANS_B ? static_cast<uint32_t>(BLOCK_CUBE) : static_cast<uint32_t>(INPUT_C0_SIZE);
        bs.SetTailAlign(mTailAlign, nTailAlign);
    }

    __aicore__ inline void BaseMBalance(BlockScheduler& scheduler, int64_t m, int64_t baseM)
    {
        if (m <= 0) {
            return;
        }
        const int64_t safeBaseM = baseM > 0 ? baseM : static_cast<int64_t>(BLOCK_CUBE);
        const int64_t mCnt = CeilDiv(m, safeBaseM);
        const int64_t balancedBaseM = CeilDiv(m, mCnt);
        curBaseM_ = static_cast<uint32_t>(CeilAlign(balancedBaseM, static_cast<int64_t>(BLOCK_CUBE)));
        scheduler.UpdateBaseM(curBaseM_);
    }

    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2CacheHint(TensorB& gmB, TensorScaleB& gmScaleB, int64_t mSize, int64_t curBaseM,
                                          int64_t baseN)
    {
        const int64_t problemN = AscendC::Te::Get<MNK_N>(problemShape_);
        const int64_t problemK = AscendC::Te::Get<MNK_K>(problemShape_);
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

    __aicore__ inline void Run(const Params& params)
    {
        Init(params);
        if (groupNum_ == 0) {
            return;
        }
        const auto groupListLayout = AscendC::Te::MakeLayout(AscendC::Te::MakeShape(static_cast<int64_t>(groupNum_)),
                                                             AscendC::Te::MakeStride(1L));
        const auto gmGroupList = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                                                             reinterpret_cast<__gm__ int64_t*>(params.groupListGmAddr)),
                                                         groupListLayout);
        const auto& gmmParams = params.gmmParams;
        BlockScheduler bs(gmmParams.baseM, gmmParams.baseN, gmmParams.baseK);
        SetSchedulerTailAlign(bs);

        for (uint32_t groupIdx = 0; groupIdx < groupNum_; ++groupIdx) {
            if (!SetMNK(gmGroupList, groupIdx)) {
                continue;
            }
            const int64_t problemM = AscendC::Te::Get<MNK_M>(problemShape_);
            const int64_t problemN = AscendC::Te::Get<MNK_N>(problemShape_);
            const int64_t problemK = AscendC::Te::Get<MNK_K>(problemShape_);
            BaseMBalance(bs, problemM, gmmParams.baseM);
            if ASCEND_IS_AIC {
                blockMmad.UpdateParamsForNextProblem(problemShape_);
            }
            bs.UpdateNextProblem(SchedulerShape{problemM, problemN >> 1, problemK, 0});
            ProcessSingleGroup(bs, groupIdx);
        }
        End();
    }

    __aicore__ inline void Init(const Params& params)
    {
        const auto& gmmParams = params.gmmParams;
        aBasePtr_ = reinterpret_cast<__gm__ AType*>(params.blockMmadAddressParams.aGmAddr);
        bBasePtr_ = reinterpret_cast<__gm__ BType*>(params.blockMmadAddressParams.bGmAddr);
        scaleABasePtr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.blockMmadAddressParams.scaleAGmAddr);
        scaleBBasePtr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.blockMmadAddressParams.scaleBGmAddr);
        singleW_ = (gmmParams.singleW == 1);

        const ProblemShape initProblemShape{gmmParams.m, gmmParams.n, gmmParams.k, 0};
        problemShape_ = initProblemShape;
        groupNum_ = gmmParams.groupNum;
        groupListType_ = gmmParams.groupListType;
        curBaseM_ = gmmParams.baseM;
        baseN_ = gmmParams.baseN;

        const BlockShape l0TileShape{static_cast<int64_t>(gmmParams.baseM), static_cast<int64_t>(gmmParams.baseN),
                                     static_cast<int64_t>(gmmParams.baseK), 0};
        const L1Params l1Params{static_cast<uint64_t>(gmmParams.kAL1), static_cast<uint64_t>(gmmParams.kBL1),
                                static_cast<uint64_t>(gmmParams.scaleKAL1)};
        const BlockMmadInitParams blockMmadInitParams{l0TileShape, l1Params, false,
                                                      gmmParams.dbL0C == DOUBLE_BUFFER_COUNT};
        if ASCEND_IS_AIC {
            blockMmad.Init(initProblemShape, blockMmadInitParams);
        }

        blockEpilogue.Init(params.epilogueParams);

        const EpilogueProblemShape epilogueProblemShape{AscendC::Te::Get<MNK_M>(problemShape_),
                                                        AscendC::Te::Get<MNK_N>(problemShape_) >> 1,
                                                        AscendC::Te::Get<MNK_K>(problemShape_)};
        blockEpilogue.UpdateNextProblem(epilogueProblemShape);
    }

    template <typename GroupListTensor>
    __aicore__ inline bool SetMNK(const GroupListTensor& gmGroupList, uint32_t groupIdx)
    {
        const int64_t splitValue = GetSplitValueFromGroupList(gmGroupList, groupIdx);
        const ProblemShape initProblemShape{splitValue, AscendC::Te::Get<MNK_N>(problemShape_),
                                            AscendC::Te::Get<MNK_K>(problemShape_), 0};
        problemShape_ = initProblemShape;
        if (splitValue <= 0) {
            return false;
        }
        return true;
    }

    template <typename GroupListTensor>
    __aicore__ inline int64_t GetSplitValueFromGroupList(const GroupListTensor& gmGroupList, uint32_t groupIdx)
    {
        int64_t splitValue = 0;
        const int64_t groupValue = gmGroupList[groupIdx];
        // groupListType supports only 0 (cumulative offsets/cumsum) and 1 (per-group counts).
        if (groupListType_ == 0) {
            splitValue = groupValue - preOffset_;
            preOffset_ = groupValue;
        } else {
            splitValue = groupValue;
            preOffset_ += splitValue;
        }
        return splitValue;
    }

    __aicore__ inline void ProcessSingleGroup(BlockScheduler& bs, uint32_t groupIdx)
    {
        const int64_t problemM = AscendC::Te::Get<MNK_M>(problemShape_);
        const int64_t problemN = AscendC::Te::Get<MNK_N>(problemShape_);
        const int64_t problemK = AscendC::Te::Get<MNK_K>(problemShape_);
        const int64_t baseN = static_cast<int64_t>(baseN_);

        const typename BlockScheduler::SwigluGroupParams groupParams{groupIdx, preOffset_, problemShape_, singleW_};
        bs.UpdateSwigluGroup(groupParams);
        typename BlockScheduler::SwigluBlockInfo blockInfo;
        if (!bs.GetNextSwigluBlock(blockInfo)) {
            return;
        }
        const typename BlockScheduler::SwigluGroupInfo& groupInfo = bs.GetSwigluGroupInfo();

        EpilogueOutputOffsets baseOutputOffsets{groupInfo.outputOffset, groupInfo.outputScaleOffset};
        blockEpilogue.UpdateGlobalAddr(baseOutputOffsets);

        auto layoutA = MakeLayoutA{}(problemM, problemK);
        auto layoutScaleA = MakeLayoutScaleA{}(problemM, groupInfo.inputScaleK);
        auto layoutB = MakeLayoutB{}(problemK, problemN);
        auto layoutScaleB = MakeLayoutScaleB{}(groupInfo.inputScaleK, problemN);
        auto layoutBias = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(static_cast<int64_t>(1), problemN);

        auto gmA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aBasePtr_ + groupInfo.aOffset), layoutA);
        auto gmScaleA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleABasePtr_ + groupInfo.scaleAOffset), layoutScaleA);
        auto gmB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                (singleW_ ? bBasePtr_ : GetTensorAddrFromList(groupIdx, bBasePtr_)) + groupInfo.bOffset),
            layoutB);
        auto gmScaleB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                (singleW_ ? scaleBBasePtr_ : GetTensorAddrFromList(groupIdx, scaleBBasePtr_)) + groupInfo.scaleBOffset),
            layoutScaleB);
        auto gmBias = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasPtr_), layoutBias);

        SetL2CacheHint(gmB, gmScaleB, problemM, static_cast<int64_t>(curBaseM_), baseN);

        do {
            const int64_t blockM = AscendC::Te::Get<MNK_M>(blockInfo.blockShape);
            const int64_t blockN = AscendC::Te::Get<MNK_N>(blockInfo.blockShape);
            if (blockM <= 0 || blockN <= 0) {
                continue;
            }
            const BlockShape& singleShape = blockInfo.blockShape;

            auto gmBlockA = gmA.Slice(AscendC::MakeCoord(blockInfo.aMOffset, static_cast<int64_t>(0)),
                                      AscendC::MakeShape(blockM, problemK));
            auto gmBlockScaleA = gmScaleA.Slice(AscendC::MakeCoord(blockInfo.aMOffset, static_cast<int64_t>(0)),
                                                AscendC::MakeShape(blockM, groupInfo.inputScaleK));
            auto gmBlockBLeft = gmB.Slice(AscendC::MakeCoord(static_cast<int64_t>(0), blockInfo.bLeftNOffset),
                                          AscendC::MakeShape(problemK, blockN));
            auto gmBlockBRight = gmB.Slice(AscendC::MakeCoord(static_cast<int64_t>(0), blockInfo.bRightNOffset),
                                           AscendC::MakeShape(problemK, blockN));
            auto gmBlockScaleBLeft = gmScaleB.Slice(AscendC::MakeCoord(static_cast<int64_t>(0), blockInfo.bLeftNOffset),
                                                    AscendC::MakeShape(groupInfo.inputScaleK, blockN));
            auto gmBlockScaleBRight = gmScaleB.Slice(
                AscendC::MakeCoord(static_cast<int64_t>(0), blockInfo.bRightNOffset),
                AscendC::MakeShape(groupInfo.inputScaleK, blockN));

            if ASCEND_IS_AIC {
                if (isVecSetSyncCom_) {
                    WaitForAiv();
                }
                auto ubOut = blockEpilogue.GetConcatL0c2UbTensor(blockM, blockN);
                blockMmad(gmBlockA, gmBlockBLeft, gmBlockScaleA, gmBlockScaleBLeft, ubOut, singleShape);
                SyncAicToAiv();
            }
            isVecSetSyncCom_ = true;

            if ASCEND_IS_AIV {
                WaitForAic();
                const EpilogueBlockShape epiBlockShape = blockInfo.epilogueBlockShape;
                const EpilogueOutputOffsets outputOffsets{blockInfo.outputOffsets.outputOffset,
                                                          blockInfo.outputOffsets.outputScaleOffset};
                blockEpilogue(epiBlockShape, outputOffsets);
                SyncAivToAic();
            }
        } while (bs.GetNextSwigluBlock(blockInfo));
    }

    template <typename T>
    __aicore__ inline __gm__ T* GetTensorAddrFromList(uint16_t index, __gm__ T* tensorPtr)
    {
        __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
        uint64_t tensorPtrOffset = *dataAddr;
        __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
        return reinterpret_cast<__gm__ T*>(*(retPtr + index));
    }

    BlockMmad blockMmad;
    BlockEpilogue blockEpilogue;
    ProblemShape problemShape_{};

    __gm__ AType* aBasePtr_{nullptr};
    __gm__ BType* bBasePtr_{nullptr};
    __gm__ BiasType* biasPtr_{nullptr};
    __gm__ fp8_e8m0_t* scaleABasePtr_{nullptr};
    __gm__ fp8_e8m0_t* scaleBBasePtr_{nullptr};

    int64_t preOffset_{0};

    uint32_t groupNum_{0};
    uint32_t curBaseM_{0};
    uint32_t baseN_{0};
    uint8_t groupListType_{0};
    bool isVecSetSyncCom_{false};
    bool singleW_{true};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
