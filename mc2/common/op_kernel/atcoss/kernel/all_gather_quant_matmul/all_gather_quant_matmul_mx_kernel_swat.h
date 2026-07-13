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
 * \file all_gather_quant_matmul_mx_kernel_swat.h
 * \brief Custom GemmUniversal specialization for AllGather+QuantMatmul.
 */

#pragma once

#include "blaze/gemm/kernel/kernel_universal.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "tensor_api/tensor.h"
#include "block_prologue_all_gather.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {
#define AGQMM_MX_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define AGQMM_MX_KERNEL_TEM_PARAMS                           \
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler, \
        AscendC::Std::enable_if_t<                          \
            AscendC::Std::is_same_v<KernelMmadWithScaleMx, typename BlockMmad::DispatchPolicy::ScheduleType>>

AGQMM_MX_KERNEL_CLASS_TEM_PARAMS
class GemmUniversal<AGQMM_MX_KERNEL_TEM_PARAMS> {
public:
    __aicore__ inline GemmUniversal()
    {}
    __aicore__ inline ~GemmUniversal()
    {}

    static constexpr bool weightNz = BlockMmad::weightNz;
    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;
    static constexpr bool isAtomicAdd = BlockMmad::DispatchPolicy::isAtomicAdd;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr int32_t SCALE_C0 = 2;

    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;

    using BlockSchedulerParams = typename BlockScheduler::Params;

    using PrologueOp = BlockEpilogue;
    static constexpr uint32_t kSyncMode = PrologueOp::kSyncMode;

    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<LayoutC, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<CType>>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        transA, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleADNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        transB, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;

    struct QBMMTiling {
        uint32_t batchA1;
        uint32_t batchA2;
        uint32_t batchA3;
        uint32_t batchA4;
        uint32_t batchB1;
        uint32_t batchB2;
        uint32_t batchB3;
        uint32_t batchB4;
        uint32_t batchC1;
        uint32_t batchC2;
        uint32_t batchC3;
        uint32_t batchC4;
        uint32_t biasThreeDim;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t isBias;
        uint32_t dbL0C;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
        typename BlockMmad::SyncParams syncParams;
        GM_ADDR tilingGM;
        GM_ADDR hcclContext;
        GM_ADDR x1Addr;
        GM_ADDR x1ScaleAddr;
        GM_ADDR allGatherDataOutAddr;
        GM_ADDR allGatherScalesOutAddr;
    };

public:
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Run(Params params);
    __aicore__ inline void operator()(const Params& params)
    {
        Run(params);
    }

private:
    __aicore__ inline void ComputeSelfAddrs(const Params& params, GM_ADDR& selfDataAddr, GM_ADDR& selfScaleAddr);
    __aicore__ inline void ResetGmAddr(const Params& params);
    __aicore__ inline void ProcessSingleBatch(
        const Params& params, BlockScheduler& bs, uint64_t restBatch, bool isTailRound);

    __aicore__ inline void ProcessWithBatch(const Params& params, BlockScheduler& bs);
    __aicore__ inline void AddBatchOffset(const Params& params);

    template <typename TensorB, typename TensorC>
    __aicore__ inline void SetL2Cache(
        const ProblemShape& problemShape, uint64_t baseM, uint64_t baseN, TensorB& gmB, TensorC& gmC);

private:
    BlockMmad mmadOp_;
    __gm__ allGatherMatmulTilingData* tilingData_;
    GM_ADDR hcclContext_;

    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr;
    __gm__ AscendC::fp8_e8m0_t* scaleAGmAddr_;
    __gm__ AscendC::fp8_e8m0_t* scaleBGmAddr_;

    uint64_t batchCOffset_{0};
    uint64_t batchAOffset_{0};
    uint64_t batchBOffset_{0};
    bool isBiasThreeDim_{false};
    bool isBias_{false};
    bool isSameBatch_{false};
    bool needUpdateTail_{false};
};

AGQMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<AGQMM_MX_KERNEL_TEM_PARAMS>::Run(Params params)
{
    tilingData_ = (__gm__ allGatherMatmulTilingData*)params.tilingGM;
    hcclContext_ = params.hcclContext;

    if ASCEND_IS_AIV {
        typename PrologueOp::Params prologueParams;
        prologueParams.inputXAddr = params.x1Addr;
        prologueParams.inputXScaleAddr = params.x1ScaleAddr;
        prologueParams.allGatherDataOutAddr = params.allGatherDataOutAddr;
        prologueParams.allGatherScalesOutAddr = params.allGatherScalesOutAddr;
        prologueParams.tilingData = &tilingData_->allGatherTiling;
        prologueParams.hcclContext = hcclContext_;
        PrologueOp prologueOp;
        prologueOp.Init(prologueParams);
        prologueOp.Run();
        return;
    }

    if constexpr (isAtomicAdd) {
        AscendC::SetAtomicAdd<float>();
    }

    GM_ADDR selfDataAddr = nullptr;
    GM_ADDR selfScaleAddr = nullptr;
    ComputeSelfAddrs(params, selfDataAddr, selfScaleAddr);

    params.mmadParams.aGmAddr = selfDataAddr;
    params.mmadParams.scaleAGmAddr = selfScaleAddr;

    const auto& problemShape = params.problemShape;
    const auto& qbmmParams = params.qbmmParams;

    const BlockShape l0TileShape{qbmmParams.baseM, qbmmParams.baseN, qbmmParams.baseK, 0};
    mmadOp_.Init(problemShape, l0TileShape, params.l1Params, isBias_, qbmmParams.dbL0C > 1, params.syncParams);

    ResetGmAddr(params);

    isBias_ = (qbmmParams.isBias == 1);
    isSameBatch_ = true;

    BlockScheduler bs(problemShape, params.schParams);

    if (AscendC::Te::Get<MNK_B>(problemShape) == 1) {
        ProcessSingleBatch(params, bs, 0, true);
        if constexpr (isAtomicAdd) {
            AscendC::SetAtomicNone();
        }
        return;
    }

    ProcessWithBatch(params, bs);
}

AGQMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<AGQMM_MX_KERNEL_TEM_PARAMS>::ComputeSelfAddrs(
    const Params& params, GM_ADDR& selfDataAddr, GM_ADDR& selfScaleAddr)
{
    auto* hcclCtx = (__gm__ Mc2Kernel::HcclOpParam*)params.hcclContext;
    uint32_t curRankId = Mc2Kernel::GetRankId(hcclCtx);
    uint32_t rankSize = Mc2Kernel::GetRankDim(hcclCtx);

    uint64_t M = tilingData_->allGatherTiling.M;
    uint64_t K = tilingData_->allGatherTiling.Ka;

    uint64_t xTotalBytes = M * K * sizeof(AType);
    using PrologueType = Block::BlockPrologueAllGather<int8_t, int8_t, kSyncMode>;
    uint64_t winScaleOffset = Blaze::Gemm::CeilDiv(rankSize * xTotalBytes,
        PrologueType::WIN_ADDR_ALIGN) * PrologueType::WIN_ADDR_ALIGN;

    // Producer (BlockPrologueAllGather) always gathers into windowsIn + DATA_START_OFFSET.
    selfDataAddr = (GM_ADDR)(hcclCtx->windowsIn[curRankId] + PrologueType::DATA_START_OFFSET);
    selfScaleAddr = (GM_ADDR)((uint64_t)selfDataAddr + winScaleOffset);
}

AGQMM_MX_KERNEL_CLASS_TEM_PARAMS
template <typename TensorB, typename TensorC>
__aicore__ inline void GemmUniversal<AGQMM_MX_KERNEL_TEM_PARAMS>::SetL2Cache(
    const ProblemShape& problemShape, uint64_t baseM, uint64_t baseN, TensorB& gmB, TensorC& gmC)
{
    if constexpr (isAtomicAdd) {
        gmC.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
    }

    const bool fullMTile = baseM >= AscendC::Te::Get<MNK_M>(problemShape);
    if (!(isSameBatch_ && fullMTile)) {
        return;
    }

    if constexpr (weightNz) {
        gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
    } else {
        constexpr int64_t cacheLineAlignMask = IsFp4<AType>() ? 0xff : 0x7f;
        if constexpr (transB) {
            bool bAlignForL2Stream = (AscendC::Te::Get<MNK_K>(problemShape) & cacheLineAlignMask) == 0;
            gmB.SetL2CacheHint(
                bAlignForL2Stream ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE : AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
        } else {
            bool bAlignForL2Stream =
                (AscendC::Te::Get<MNK_N>(problemShape) & cacheLineAlignMask) == 0 &&
                (baseN & cacheLineAlignMask) == 0;
            gmB.SetL2CacheHint(
                bAlignForL2Stream ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE : AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
        }
    }
}

AGQMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<AGQMM_MX_KERNEL_TEM_PARAMS>::ProcessWithBatch(const Params& params, BlockScheduler& bs)
{
    const auto& qbmmParams = params.qbmmParams;
    const uint64_t batchC3C4 = static_cast<uint64_t>(qbmmParams.batchC3) * qbmmParams.batchC4;
    const uint64_t batchC2C3C4 = qbmmParams.batchC2 * batchC3C4;
    const uint64_t batchB3B4 = static_cast<uint64_t>(qbmmParams.batchB3) * qbmmParams.batchB4;
    const uint64_t batchB2B3B4 = qbmmParams.batchB2 * batchB3B4;
    const uint64_t batchA3A4 = static_cast<uint64_t>(qbmmParams.batchA3) * qbmmParams.batchA4;
    const uint64_t batchA2A3A4 = qbmmParams.batchA2 * batchA3A4;
    const uint32_t multiA1C1 = qbmmParams.batchA1 / qbmmParams.batchC1;
    const uint32_t multiA2C2 = qbmmParams.batchA2 / qbmmParams.batchC2;
    const uint32_t multiA3C3 = qbmmParams.batchA3 / qbmmParams.batchC3;
    const uint32_t multiA4C4 = qbmmParams.batchA4 / qbmmParams.batchC4;
    const uint32_t multiB1C1 = qbmmParams.batchB1 / qbmmParams.batchC1;
    const uint32_t multiB2C2 = qbmmParams.batchB2 / qbmmParams.batchC2;
    const uint32_t multiB3C3 = qbmmParams.batchB3 / qbmmParams.batchC3;
    const uint32_t multiB4C4 = qbmmParams.batchB4 / qbmmParams.batchC4;

    uint64_t batchC1Offset = 0;
    uint64_t batchA1Offset = 0;
    uint64_t batchB1Offset = 0;
    uint64_t curBatchC = 1UL;
    const uint64_t singleBatchTileCnt = bs.GetTotalCnt();
    const uint64_t batchCount = 1;
    const uint64_t tailRoundStart =
        (singleBatchTileCnt * batchCount / AscendC::GetBlockNum()) * AscendC::GetBlockNum();
    for (uint64_t b1Index = 0; b1Index < qbmmParams.batchC1; ++b1Index) {
        uint64_t batchC2Offset = batchC1Offset;
        uint64_t batchA2Offset = batchA1Offset;
        uint64_t batchB2Offset = batchB1Offset;
        for (uint64_t b2Index = 0; b2Index < qbmmParams.batchC2; ++b2Index) {
            uint64_t batchC3Offset = batchC2Offset;
            uint64_t batchA3Offset = batchA2Offset;
            uint64_t batchB3Offset = batchB2Offset;
            for (uint64_t b3Index = 0; b3Index < qbmmParams.batchC3; ++b3Index) {
                batchCOffset_ = batchC3Offset;
                batchAOffset_ = batchA3Offset;
                batchBOffset_ = batchB3Offset;
                for (uint64_t b4Index = 0; b4Index < qbmmParams.batchC4; ++b4Index) {
                    const bool isTailRound = curBatchC * singleBatchTileCnt > tailRoundStart;
                    AddBatchOffset(params);
                    ProcessSingleBatch(params, bs, batchCount - curBatchC, isTailRound);
                    curBatchC++;
                    batchCOffset_ += 1;
                    batchAOffset_ += multiA4C4;
                    batchBOffset_ += multiB4C4;
                }
                batchC3Offset += qbmmParams.batchC4;
                batchA3Offset += qbmmParams.batchA4 * static_cast<uint64_t>(multiA3C3);
                batchB3Offset += qbmmParams.batchB4 * static_cast<uint64_t>(multiB3C3);
            }
            batchC2Offset += batchC3C4;
            batchA2Offset += batchA3A4 * multiA2C2;
            batchB2Offset += batchB3B4 * multiB2C2;
        }
        batchC1Offset += batchC2C3C4;
        batchA1Offset += batchA2A3A4 * multiA1C1;
        batchB1Offset += batchB2B3B4 * multiB1C1;
    }
}

AGQMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<AGQMM_MX_KERNEL_TEM_PARAMS>::AddBatchOffset(const Params& params)
{
    ResetGmAddr(params);
    const auto& problemShape = params.problemShape;
    constexpr uint64_t sizeShift = IsFp4<AType>() ? 1 : 0;
    const auto m = AscendC::Te::Get<MNK_M>(problemShape);
    const auto n = AscendC::Te::Get<MNK_N>(problemShape);
    const auto k = AscendC::Te::Get<MNK_K>(problemShape);
    aGmAddr_ += (batchAOffset_ * m * k) >> sizeShift;
    if constexpr (weightNz) {
        if constexpr (transB) {
            const auto kBlockCnt = Blaze::Gemm::CeilDiv(k, C0_SIZE);
            const auto nBlockCnt = Blaze::Gemm::CeilDiv(n, static_cast<int64_t>(BLOCK_CUBE));
            bGmAddr_ += (batchBOffset_ * kBlockCnt * nBlockCnt * BLOCK_CUBE * C0_SIZE) >> sizeShift;
        } else {
            const auto nBlockCnt = Blaze::Gemm::CeilDiv(n, C0_SIZE);
            const auto kBlockCnt = Blaze::Gemm::CeilDiv(k, static_cast<int64_t>(BLOCK_CUBE));
            bGmAddr_ += (batchBOffset_ * nBlockCnt * kBlockCnt * BLOCK_CUBE * C0_SIZE) >> sizeShift;
        }
    } else {
        bGmAddr_ += (batchBOffset_ * n * k) >> sizeShift;
    }
    cGmAddr_ += batchCOffset_ * m * n;
    if (isBiasThreeDim_) {
        biasGmAddr_ += batchCOffset_ * n;
    }
}

AGQMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<AGQMM_MX_KERNEL_TEM_PARAMS>::ResetGmAddr(const Params& params)
{
    aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    scaleAGmAddr_ = reinterpret_cast<__gm__ AscendC::fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr);
    scaleBGmAddr_ = reinterpret_cast<__gm__ AscendC::fp8_e8m0_t*>(params.mmadParams.scaleBGmAddr);
    if (isBias_) {
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
    }
}

AGQMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<AGQMM_MX_KERNEL_TEM_PARAMS>::ProcessSingleBatch(
    const Params& params, BlockScheduler& bs, uint64_t restBatch, bool isTailRound)
{
    const auto& problemShape = params.problemShape;
    const auto& qbmmParams = params.qbmmParams;
    const auto m = AscendC::Te::Get<MNK_M>(problemShape);
    const auto n = AscendC::Te::Get<MNK_N>(problemShape);
    const auto k = AscendC::Te::Get<MNK_K>(problemShape);
    const auto scaleKLen = Blaze::Gemm::CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutScaleA = MakeLayoutScaleA{}(m, scaleKLen);
    auto layoutB = MakeLayoutB{}(k, n);
    auto layoutScaleB = MakeLayoutScaleB{}(scaleKLen, n);
    auto layoutBias = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1L, n);
    auto layoutC = MakeLayoutC{}(m, n);

    auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGmAddr_), layoutA);
    auto gmScaleA =
        AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleAGmAddr_), layoutScaleA);
    auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGmAddr_), layoutB);
    auto gmScaleB =
        AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleBGmAddr_), layoutScaleB);
    auto gmBias = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasGmAddr_), layoutBias);
    auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGmAddr_), layoutC);

    const auto mTailTile = params.schParams.mTailTile;
    const auto nTailTile = params.schParams.nTailTile;
    if (needUpdateTail_ ||
        (isTailRound && ((bs.GetEndBlockIdx() + 1) + (restBatch * bs.GetTotalCnt())) * mTailTile * nTailTile <=
                            AscendC::GetBlockNum())) {
        needUpdateTail_ = true;
        bs.UpdateTailTile(mTailTile, nTailTile);
    }
    SetL2Cache(problemShape, qbmmParams.baseM, qbmmParams.baseN, gmB, gmC);

    BlockCoord blockIdx;
    int64_t mPos = 0L;
    int64_t nPos = 0L;
    constexpr int64_t kPos = 0L;
    while (bs.GetTileIdx(blockIdx)) {
        BlockShape singleShape =
            bs.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE, weightNz>(blockIdx);
        const auto baseM = AscendC::Te::Get<IDX_M_TILEIDX>(singleShape);
        const auto baseN = AscendC::Te::Get<IDX_N_TILEIDX>(singleShape);
        if (baseM <= 0 || baseN <= 0) {
            return;
        }

        bs.GetTileCoord(blockIdx, mPos, nPos);
        auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(mPos, kPos), AscendC::Te::MakeShape(baseM, k));
        auto gmBlockScaleA =
            gmScaleA.Slice(AscendC::Te::MakeCoord(mPos, kPos), AscendC::Te::MakeShape(baseM, scaleKLen));
        auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(kPos, nPos), AscendC::Te::MakeShape(k, baseN));
        auto gmBlockScaleB =
            gmScaleB.Slice(AscendC::Te::MakeCoord(kPos, nPos), AscendC::Te::MakeShape(scaleKLen, baseN));
        auto gmBlockBias =
            gmBias.Slice(AscendC::Te::MakeCoord(0L, nPos), AscendC::Te::MakeShape(1L, baseN));
        auto gmBlockC = gmC.Slice(AscendC::Te::MakeCoord(mPos, nPos), AscendC::Te::MakeShape(baseM, baseN));
        mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape);
    }
    bs.UpdateNextBatchBlockRoundParams();
}
} // namespace Kernel
} // namespace Gemm
} // namespace Blaze

#undef AGQMM_MX_KERNEL_CLASS_TEM_PARAMS
#undef AGQMM_MX_KERNEL_TEM_PARAMS
