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
 * \file all_gather_qbmm_mx_kernel.h
 * \brief AllGather Prefill QMM 内核 — 基于 FragmentTensor (refactored copy)
 *
 * Refactoring vs original (kernel/matmul/quant_batch_matmul/all_gather_qbmm_mx_kernel.h):
 *   - Eliminated Params / QBMMTiling / FragmentParams structs (31 fields → 20)
 *   - New Run(tilingData, derivedParams) signature: tilingData passed as pointer,
 *     kernel reads mmTile/commTile/isBias directly — no re-packaging
 *   - Private methods read from stored td_/addrs_ pointers — no per-call param passing
 *   - Win buffer reuse: MAIN region reads A/scaleA as continuous GM tensors (remote compact + slot rewind)
 *   - AIC release: after consuming a round, signals AIV via CrossCoreSetFlag<PIPE_FIX>
 *   - Eliminated: curMainA_, curMainScaleA_, winDataRankBase_, winScaleRankBase_, EnsureWinRankBasesReady
 */

#pragma once

#include "kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#define WINDOW_LEN 1L
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#undef WINDOW_LEN
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "include/tensor_api/tensor.h"
#include "apace/basic/buffer_channel.h"
#include "apace/basic/fragment_tensor/fragment_tensor.h"
#include "apace/basic/fragment_tensor/fragment_tensor_api.h"
#include "apace/tiling/quant_matmul_tiling_data.h"
#include "apace/tiling/comm_tiling_data.h"

namespace Apace {

using namespace AscendC;

using LayoutA = AscendC::Te::NDExtLayoutPtn;
using LayoutB = AscendC::Te::DNExtLayoutPtn;
using LayoutC = AscendC::Te::NDExtLayoutPtn;
using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

enum RegionTag {
    HEAD,
    MAIN,
    TAIL
};

// SyncAll<true> occupies flagId 14, release direction modulo [0,13] to avoid collision.
constexpr uint32_t SYNCALL_FLAG_ID_MAX = 14;

template <typename AType, typename BType, typename CType, typename TilingDataType>
class AllGatherQbmmMxKernel {
public:
    __aicore__ inline AllGatherQbmmMxKernel() {}
    __aicore__ inline ~AllGatherQbmmMxKernel() {}

    // ---- Component types ----
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, 0>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using BlockScheduler =
        Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB, AType>;
    using BlockMmad =
        Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, float, LayoutC>;
    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;

    // ---- Compile-time constants ----
    static constexpr int64_t kC0Size = 32;
    static constexpr int64_t kCacheLineAlignMask = 0x7f;
    static constexpr int32_t kScaleC0 = 2;

    // ---- Layout factories ----
    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<kC0Size>>;
    using MakeLayoutScaleA =
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<kScaleC0>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<kC0Size>>;
    using MakeLayoutScaleB =
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<kScaleC0>>;
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<LayoutC, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<CType>>>;

    // ---- Fragment MMAD types ----
    using BlockMmadFragC = Blaze::Gemm::Block::QmmMxBlockMmadFragment<0, false, AType, LayoutA, BType, LayoutB, CType,
                                                                      LayoutC, float, LayoutC>;
    static constexpr bool weightNz = BlockMmadFragC::weightNz;
    using FragL1Params = typename BlockMmadFragC::L1Params;
    using FragBlockShape = typename BlockMmadFragC::BlockShape;

    // ---- FragmentTensor types ----
    using FragTensorA = Apace::Basic::FragmentTensor<2, Apace::Basic::MAX_FRAGMENT_COUNT, MakeLayoutA, AType>;
    using FragScaleA =
        Apace::Basic::FragmentTensor<2, Apace::Basic::MAX_FRAGMENT_COUNT, MakeLayoutScaleA, AscendC::fp8_e8m0_t>;
    using FragTensorC = Apace::Basic::FragmentTensor<2, Apace::Basic::MAX_FRAGMENT_COUNT, MakeLayoutC, CType>;

    /*!
     * \brief Cat 1: Tensor GM addresses, change every kernel launch.
     */
    struct TensorAddrs {
        GM_ADDR aGM{nullptr};
        GM_ADDR aScaleGM{nullptr};
        GM_ADDR bGM{nullptr};
        GM_ADDR bScaleGM{nullptr};
        GM_ADDR cGM{nullptr};
        GM_ADDR biasGM{nullptr};
    };

    /*!
     * \brief Cat 2: Tiling-derived params — only fields needed by both impl and kernel.
     * Matmul-only derived values are computed inside kernel Init(), not here.
     */
    struct TilingDerivedParams {
        uint32_t commTurn{0};
        uint64_t effBufferCount{0};
        uint32_t rankId{0};
        uint32_t rankSize{0};
    };

    __aicore__ inline void Run(const TilingDataType *td, const TensorAddrs &addrs, const TilingDerivedParams &tp,
                               const Apace::Basic::BufferChannel &dataChannel,
                               const Apace::Basic::BufferChannel &scaleChannel)
    {
        td_ = td;
        addrs_ = &addrs;
        tp_ = &tp;
        dataChannel_ = &dataChannel;
        scaleChannel_ = &scaleChannel;
        Init();
        Process();
    }

private:
    __aicore__ inline void Init();
    __aicore__ inline void Process();

    // ---- L2 cache optimization ----
    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2Cache(const ProblemShape &problemShape, int64_t baseM, int64_t baseN, TensorB &gmB,
                                      TensorScaleB &gmScaleB);

    // ---- FragmentTensor builders ----
    __aicore__ inline void BuildFragmentTensors();
    __aicore__ inline void BuildMainFragment(uint32_t roundIdx);
    __aicore__ inline void BuildTailFragment();
    __aicore__ inline void UpdateMainRoundAddrs(uint32_t roundIdx);
    __aicore__ inline Apace::Basic::FragmentParam<2> MakeFragParam(uint64_t fragSize, uint64_t realFragSize,
                                                                   uint32_t fragCnt, uint64_t shape1) const;

    // ---- Tile context resolution ----
    struct TileCtx {
        uint32_t dependTileIdx;
        uint32_t roundIdx;
        RegionTag region;
        int64_t regionMPos;
        const FragTensorA *fragA;       // HEAD/TAIL only; nullptr for MAIN
        const FragScaleA *fragScaleA;   // HEAD/TAIL only; nullptr for MAIN
        const FragTensorC *fragC;
        uint64_t rankCnt;
    };
    __aicore__ inline TileCtx ResolveTileCtx(int64_t mPos, int64_t headMainRows, int64_t mainRoundRows,
                                             int64_t mainSectionRows, uint32_t rankSize, uint32_t commTurn) const;

    // ---- Stored params (pointers, no copy) ----
    const TilingDataType *td_{nullptr};
    const TensorAddrs *addrs_{nullptr};
    const TilingDerivedParams *tp_{nullptr};
    const Apace::Basic::BufferChannel *dataChannel_{nullptr};
    const Apace::Basic::BufferChannel *scaleChannel_{nullptr};

    // ---- Derived in Init() from tp_ + td_ ----
    uint32_t rankId_{0};
    uint32_t rankSize_{0};

    // ---- Matmul derived values (computed in Init() from td_) ----
    static constexpr uint32_t kPaddingLength = 16;
    uint32_t mPerRank_{0};
    uint32_t paddedTailM_{0};
    uint64_t headRows_{0};
    uint64_t dataBytesPerMRow_{0};
    uint64_t scaleBytesPerMRow_{0};
    uint64_t cBytesPerM_{0};

    // ---- GM addresses ----
    __gm__ AType *aGmAddr_{nullptr};
    __gm__ AscendC::fp8_e8m0_t *scaleAGmAddr_{nullptr};
    __gm__ BType *bGmAddr_{nullptr};
    __gm__ AscendC::fp8_e8m0_t *scaleBGmAddr_{nullptr};
    __gm__ CType *cGmAddr_{nullptr};
    __gm__ float *biasGmAddr_{nullptr};

    // ---- FragmentTensor state ----
    FragTensorA headFragA_{};
    FragScaleA headFragScaleA_{};
    FragTensorC headFragC_{};
    FragTensorC curMainC_{};          // MAIN 仅 C 保留 frag scatter；A/scaleA 连续读
    FragTensorA tailFragA_{};
    FragScaleA tailFragScaleA_{};
    FragTensorC tailFragC_{};

    static constexpr uint32_t MAX_RANK_SIZE = 16U;
    GM_ADDR cFragAddrs_[MAX_RANK_SIZE]{};

    static constexpr uint32_t MAX_FRAG = Apace::Basic::MAX_FRAGMENT_COUNT;
    GM_ADDR headAddrListA_[MAX_FRAG]{};
    GM_ADDR headAddrListScale_[MAX_FRAG]{};
    GM_ADDR headAddrListC_[MAX_FRAG]{};
    GM_ADDR mainAddrListC_[MAX_FRAG]{};
    GM_ADDR tailAddrListA_[MAX_FRAG]{};
    GM_ADDR tailAddrListScale_[MAX_FRAG]{};
    GM_ADDR tailAddrListC_[MAX_FRAG]{};

    uint32_t curMainRoundIdx_{0xFFFFFFFF};
    bool tailBuilt_{false};
    uint64_t curRoundCOff_{};
    int32_t releasedRoundIdx_{-1};

    // ---- Tile strides ----
    uint64_t tileMDataStride_{};
    uint64_t tileMScaleStride_{};
    uint64_t tileMCStride_{};
};

// =================================================================================
// Init
// =================================================================================

template <typename AType, typename BType, typename CType, typename TilingDataType>
__aicore__ inline void AllGatherQbmmMxKernel<AType, BType, CType, TilingDataType>::Init()
{
    if ASCEND_IS_AIV {
        return;
    }
    aGmAddr_ = reinterpret_cast<__gm__ AType *>(addrs_->aGM);
    bGmAddr_ = reinterpret_cast<__gm__ BType *>(addrs_->bGM);
    cGmAddr_ = reinterpret_cast<__gm__ CType *>(addrs_->cGM);
    scaleAGmAddr_ = reinterpret_cast<__gm__ AscendC::fp8_e8m0_t *>(addrs_->aScaleGM);
    scaleBGmAddr_ = reinterpret_cast<__gm__ AscendC::fp8_e8m0_t *>(addrs_->bScaleGM);
    biasGmAddr_ = reinterpret_cast<__gm__ float *>(addrs_->biasGM);

    rankId_ = tp_->rankId;
    rankSize_ = tp_->rankSize;

    const auto &ct = td_->commTile;
    const auto &mmT = td_->mmTile;
    uint32_t tileM = static_cast<uint32_t>(ct.splitAxisTileSize);
    uint32_t tileCnt = static_cast<uint32_t>(ct.splitAxisTileCnt);
    uint32_t tailM = static_cast<uint32_t>(ct.splitAxisTailSize);
    uint32_t tailCnt = static_cast<uint32_t>(ct.splitAxisTailCnt);

    mPerRank_ = tileM * tileCnt + tailM * tailCnt;
    paddedTailM_ = (tailM > 0) ? ((tailM + kPaddingLength - 1) / kPaddingLength * kPaddingLength) : 0U;
    headRows_ = static_cast<uint64_t>(tileCnt) * tileM;

    uint64_t scaleKGroups = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(mmT.k), Blaze::Gemm::MXFP_DIVISOR_SIZE);
    dataBytesPerMRow_ = Blaze::Gemm::IsFp4<AType>() ? (static_cast<uint64_t>(mmT.k) >> 1) :
                                                       static_cast<uint64_t>(mmT.k) * sizeof(AType);
    scaleBytesPerMRow_ = scaleKGroups * static_cast<uint64_t>(Blaze::Gemm::MXFP_MULTI_BASE_SIZE) *
                         sizeof(AscendC::fp8_e8m0_t);
    cBytesPerM_ = static_cast<uint64_t>(mmT.n) * sizeof(CType);
}

// =================================================================================
// Process
// =================================================================================

template <typename AType, typename BType, typename CType, typename TilingDataType>
__aicore__ inline void AllGatherQbmmMxKernel<AType, BType, CType, TilingDataType>::Process()
{
    const auto &mmT = td_->mmTile;
    const auto &ct = td_->commTile;
    const auto &tp = *tp_;

    const int64_t Ki = static_cast<int64_t>(mmT.k);
    const int64_t Ni = static_cast<int64_t>(mmT.n);
    const int64_t scaleKLen = static_cast<int64_t>(ct.scaleNonSplitAxisSize);

    const uint32_t tileCnt = static_cast<uint32_t>(ct.splitAxisTileCnt);
    const uint32_t tileM = static_cast<uint32_t>(ct.splitAxisTileSize);
    const uint32_t tailCnt = static_cast<uint32_t>(ct.splitAxisTailCnt);
    const uint32_t tailM = static_cast<uint32_t>(ct.splitAxisTailSize);

    bool hasTail = (tailCnt > 0 && tailM > 0);
    const int64_t headMainRows = static_cast<int64_t>(headRows_);
    const int64_t mainRoundRows = static_cast<int64_t>(rankSize_ - 1) * static_cast<int64_t>(tileM);
    const int64_t mainSectionRows = static_cast<int64_t>(tileCnt) * mainRoundRows;
    const int64_t tailRows = hasTail ? static_cast<int64_t>(rankSize_) * static_cast<int64_t>(paddedTailM_) : 0;
    const int64_t totalLogicalM = headMainRows + mainSectionRows + tailRows;

    const int64_t rankMinus1 = static_cast<int64_t>(rankSize_ - 1);

    // MAIN continuous GM tensors (remote compact + buffer reuse):
    const int64_t mainDataM = static_cast<int64_t>(tp.effBufferCount) * rankMinus1 * static_cast<int64_t>(tileM);
    const int64_t mainScaleM = static_cast<int64_t>(tp.commTurn) * rankMinus1 * static_cast<int64_t>(tileM);
    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ AType *>(dataChannel_->GetBase())),
        MakeLayoutA{}(mainDataM, Ki));
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ AscendC::fp8_e8m0_t *>(scaleChannel_->GetBase())),
        MakeLayoutScaleA{}(mainScaleM, scaleKLen));

    const auto &schParams = reinterpret_cast<const BlockSchedulerParams &>(mmT.schedulerParams);
    static_assert(sizeof(mmT.schedulerParams) == sizeof(BlockSchedulerParams), "SchedulerParams layout mismatch");

    FragBlockShape l0TileShape{static_cast<int64_t>(mmT.schedulerParams.baseM), static_cast<int64_t>(mmT.schedulerParams.baseN),
                               static_cast<int64_t>(mmT.baseK), 0L};
    FragL1Params fragL1{static_cast<uint64_t>(mmT.stepK) * mmT.baseK, mmT.scaleKL1, mmT.nBufferNum};

    ProblemShape problemShape{totalLogicalM, Ni, Ki, 1};
    BlockScheduler sch(problemShape, schParams);
    BlockMmadFragC mmadFrag;
    const bool isBias = (td_->isBias != 0);
    mmadFrag.Init(problemShape, l0TileShape, fragL1, isBias, mmT.dbL0c > 1);

    BuildFragmentTensors();

    // ---- Tile loop ----
    auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(bGmAddr_), MakeLayoutB{}(Ki, Ni));
    auto gmScaleB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(scaleBGmAddr_), MakeLayoutScaleB{}(scaleKLen, Ni));
    __gm__ float *biasNull = nullptr;
    __gm__ float *biasPtr = isBias ? biasGmAddr_ : biasNull;
    auto gmBias =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(biasPtr), Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, Ni));

    const auto &mTailTile = mmT.schedulerParams.mTailTile;
    const auto &nTailTile = mmT.schedulerParams.nTailTile;
    if ((sch.GetEndBlockIdx() + 1) * mTailTile * nTailTile <= AscendC::GetBlockNum()) {
        sch.UpdateTailTile(mTailTile, nTailTile);
    }
    uint32_t readyTileIdx = 0;
    CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);

    const int32_t commTurnI = static_cast<int32_t>(tp.commTurn);
    const int32_t effBufferCountI = static_cast<int32_t>(tp.effBufferCount);

    Te::Coord<int64_t, int64_t, int64_t, int64_t> blockIdx;
    int64_t mPos = 0L, nPos = 0L;

    while (sch.GetTileIdx(blockIdx)) {
        auto singleShape =
            sch.template GetBlockShape<Blaze::Gemm::QuantMode::MX_PERGROUP_MODE,
                                       Blaze::Gemm::QuantMode::MX_PERGROUP_MODE, BlockMmadFragC::weightNz>(blockIdx);
        int64_t curMtile = Te::Get<0>(singleShape);
        int64_t curNtile = Te::Get<1>(singleShape);
        if (curMtile <= 0 || curNtile <= 0) {
            break;
        }
        sch.GetTileCoord(blockIdx, mPos, nPos);

        auto ctx = ResolveTileCtx(mPos, headMainRows, mainRoundRows, mainSectionRows, rankSize_, tp.commTurn);
        while (readyTileIdx < ctx.dependTileIdx) {
            readyTileIdx++;
            CrossCoreWaitFlag<0x2, PIPE_MTE2>(readyTileIdx);
        }

        // win buffer reuse: release consumed round to AIV (deduplicated, monotonically increasing).
        {
            int32_t dependRound = static_cast<int32_t>(ctx.dependTileIdx) - 1;
            while (releasedRoundIdx_ < dependRound - 1) {
                releasedRoundIdx_++;
                if (releasedRoundIdx_ < commTurnI - effBufferCountI) {
                    CrossCoreSetFlag<0x2, PIPE_FIX>(
                        static_cast<uint32_t>(releasedRoundIdx_) % SYNCALL_FLAG_ID_MAX);
                }
            }
        }

        SetL2Cache(problemShape, curMtile, curNtile, gmB, gmScaleB);

        auto gmBlockB = gmB.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(Ki, curNtile));
        auto gmBlockScaleB = gmScaleB.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(scaleKLen, curNtile));
        auto gmBlockBias = gmBias.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(1L, curNtile));

        auto coordC = Te::MakeCoord(ctx.regionMPos, nPos);
        auto shapeC = Te::MakeShape(curMtile, curNtile);

        if (ctx.region == MAIN) {
            // C addrList updated per round (must update before Slice).
            if (curMainRoundIdx_ == 0xFFFFFFFF) {
                BuildMainFragment(ctx.roundIdx);
            } else if (curMainRoundIdx_ != ctx.roundIdx) {
                UpdateMainRoundAddrs(ctx.roundIdx);
            }
            auto blockC = ctx.fragC->Slice(coordC, shapeC);
            // A/scaleA continuous read: data slot rewind, scale full (no rewind).
            int64_t slotIdx = static_cast<int64_t>(static_cast<uint64_t>(ctx.roundIdx) % tp.effBufferCount);
            int64_t actualMPos = slotIdx * rankMinus1 * static_cast<int64_t>(tileM) + ctx.regionMPos;
            int64_t scaleActualMPos = static_cast<int64_t>(ctx.roundIdx) * rankMinus1 * static_cast<int64_t>(tileM) + ctx.regionMPos;
            auto blockA = gmA.Slice(Te::MakeCoord(actualMPos, 0L), Te::MakeShape(curMtile, Ki));
            auto blockScaleA = gmScaleA.Slice(Te::MakeCoord(scaleActualMPos, 0L), Te::MakeShape(curMtile, scaleKLen));
            mmadFrag(blockA, gmBlockB, blockScaleA, gmBlockScaleB, gmBlockBias,
                     cFragAddrs_, static_cast<uint64_t>(mPerRank_), static_cast<uint64_t>(tileM),
                     static_cast<uint64_t>(tileCnt), static_cast<uint64_t>(tailM),
                     ctx.rankCnt, Ni, singleShape, ctx.regionMPos, nPos, 0, blockC);
        } else {
            // HEAD/TAIL: frag read (HEAD single fragment, TAIL padded multi-fragment).
            if (ctx.region == TAIL && !tailBuilt_) {
                BuildTailFragment();
                tailBuilt_ = true;
            }
            auto blockC = ctx.fragC->Slice(coordC, shapeC);
            auto coordA = Te::MakeCoord(ctx.regionMPos, 0L);
            auto blockA = ctx.fragA->Slice(coordA, Te::MakeShape(curMtile, Ki));
            auto blockScaleA = ctx.fragScaleA->Slice(coordA, Te::MakeShape(curMtile, scaleKLen));
            mmadFrag(blockA, gmBlockB, blockScaleA, gmBlockScaleB, gmBlockBias,
                     cFragAddrs_, static_cast<uint64_t>(mPerRank_), static_cast<uint64_t>(tileM),
                     static_cast<uint64_t>(tileCnt), static_cast<uint64_t>(tailM),
                     ctx.rankCnt, Ni, singleShape, ctx.regionMPos, nPos, 0, blockC);
        }
    }

    // Drain all remaining dependTileIdx.
    while (readyTileIdx < tp.commTurn) {
        readyTileIdx++;
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(readyTileIdx);
    }
    // Release fallback: every AIC core (including idle ones) must fill flags,
    // AIV reuse wait needs up to commTurn-bufferCount-1 releases.
    while (releasedRoundIdx_ < commTurnI - effBufferCountI - 1) {
        releasedRoundIdx_++;
        CrossCoreSetFlag<0x2, PIPE_FIX>(
            static_cast<uint32_t>(releasedRoundIdx_) % SYNCALL_FLAG_ID_MAX);
    }
}

// =================================================================================
// L2 cache
// =================================================================================

template <typename AType, typename BType, typename CType, typename TilingDataType>
template <typename TensorB, typename TensorScaleB>
__aicore__ inline void
AllGatherQbmmMxKernel<AType, BType, CType, TilingDataType>::SetL2Cache(const ProblemShape &problemShape,
                                                                        int64_t baseM, int64_t baseN, TensorB &gmB,
                                                                        TensorScaleB &gmScaleB)
{
    const bool fullMBlock = (baseM >= AscendC::Te::Get<Blaze::Gemm::MNK_M>(problemShape));

    if constexpr (weightNz) {
        gmB.SetL2CacheHint(fullMBlock ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE :
                                        AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
    } else {
        const bool bAlignForL2Stream = (AscendC::Te::Get<Blaze::Gemm::MNK_K>(problemShape) & kCacheLineAlignMask) == 0;
        gmB.SetL2CacheHint((fullMBlock && bAlignForL2Stream) ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE :
                                                               AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
    }

    const int64_t scaleNStrideBytes = AscendC::Te::Get<Blaze::Gemm::MNK_N>(problemShape) * kScaleC0;
    const int64_t scaleBaseNStrideBytes = baseN * kScaleC0;
    const bool scaleAlignForL2Stream =
        (scaleNStrideBytes & kCacheLineAlignMask) == 0 && (scaleBaseNStrideBytes & kCacheLineAlignMask) == 0;
    gmScaleB.SetL2CacheHint((fullMBlock && scaleAlignForL2Stream) ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE :
                                                                    AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
}

// =================================================================================
// FragmentTensor builders
// =================================================================================

template <typename AType, typename BType, typename CType, typename TilingDataType>
__aicore__ inline Apace::Basic::FragmentParam<2>
AllGatherQbmmMxKernel<AType, BType, CType, TilingDataType>::MakeFragParam(uint64_t fragSize, uint64_t realFragSize,
                                                                           uint32_t fragCnt, uint64_t shape1) const
{
    Apace::Basic::FragmentParam<2> param{};
    param.assembleAxis = 0;
    param.assembledShape[0] = fragSize * fragCnt;
    param.assembledShape[1] = shape1;
    param.fragmentSize = fragSize;
    param.realFragmentSize = realFragSize;
    param.fragmentCnt = fragCnt;
    return param;
}

template <typename AType, typename BType, typename CType, typename TilingDataType>
__aicore__ inline void AllGatherQbmmMxKernel<AType, BType, CType, TilingDataType>::BuildFragmentTensors()
{
    const auto &ct = td_->commTile;
    const auto &mmT = td_->mmTile;
    const auto &addrs = *addrs_;
    const uint32_t tileM = static_cast<uint32_t>(ct.splitAxisTileSize);
    const uint32_t tileCnt = static_cast<uint32_t>(ct.splitAxisTileCnt);
    const uint64_t scaleKLen = ct.scaleNonSplitAxisSize;

    for (uint32_t r = 0; r < rankSize_; ++r) {
        cFragAddrs_[r] = addrs.cGM + static_cast<uint64_t>(r) * mPerRank_ * cBytesPerM_;
    }

    tileMDataStride_ = static_cast<uint64_t>(tileM) * dataBytesPerMRow_;
    tileMScaleStride_ = static_cast<uint64_t>(tileM) * scaleBytesPerMRow_;
    tileMCStride_ = static_cast<uint64_t>(tileM) * cBytesPerM_;

    // HEAD: self data, single fragment tensor.
    headAddrListA_[0] = addrs.aGM;
    headAddrListScale_[0] = addrs.aScaleGM;
    headAddrListC_[0] = cFragAddrs_[rankId_];
    headFragA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutA, AType>(
        MakeFragParam(headRows_, headRows_, 1, mmT.k), headAddrListA_);
    headFragScaleA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutScaleA, AscendC::fp8_e8m0_t>(
        MakeFragParam(headRows_, headRows_, 1, scaleKLen), headAddrListScale_);
    headFragC_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutC, CType>(
        MakeFragParam(headRows_, headRows_, 1, mmT.n), headAddrListC_);
}

template <typename AType, typename BType, typename CType, typename TilingDataType>
__aicore__ inline void
AllGatherQbmmMxKernel<AType, BType, CType, TilingDataType>::BuildMainFragment(uint32_t roundIdx)
{
    const auto &ct = td_->commTile;
    const auto &mmT = td_->mmTile;
    const uint32_t tileM = static_cast<uint32_t>(ct.splitAxisTileSize);

    curRoundCOff_ = static_cast<uint64_t>(roundIdx) * tileMCStride_;

    uint32_t fragIdx = 0;
    for (uint32_t r = 0; r < rankSize_; ++r) {
        if (r == rankId_) {
            continue;
        }
        mainAddrListC_[fragIdx] = cFragAddrs_[r] + curRoundCOff_;
        fragIdx++;
    }
    // MAIN: only C fragment; A/scaleA read as continuous GM tensors.
    curMainC_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutC, CType>(
        MakeFragParam(static_cast<uint64_t>(tileM),
                      static_cast<uint64_t>(tileM),
                      rankSize_ - 1, mmT.n), mainAddrListC_);
    curMainRoundIdx_ = roundIdx;
}

template <typename AType, typename BType, typename CType, typename TilingDataType>
__aicore__ inline void AllGatherQbmmMxKernel<AType, BType, CType, TilingDataType>::BuildTailFragment()
{
    const auto &ct = td_->commTile;
    const auto &mmT = td_->mmTile;
    const auto &addrs = *addrs_;
    const auto &tp = *tp_;
    const uint32_t tileCnt = static_cast<uint32_t>(ct.splitAxisTileCnt);
    const uint32_t tailM = static_cast<uint32_t>(ct.splitAxisTailSize);
    const uint64_t scaleKLen = ct.scaleNonSplitAxisSize;

    uint64_t tailRowOff = headRows_;
    uint64_t tailDataOff = tailRowOff * dataBytesPerMRow_;
    uint64_t tailScaleOff = tailRowOff * scaleBytesPerMRow_;
    uint64_t tailCOff = tailRowOff * cBytesPerM_;

    // Remote compact: data reuse (slot = tileCnt % effBufferCount rewind), scale full (slot = tileCnt no rewind).
    GM_ADDR dataSlotBase = dataChannel_->GetSlotBase(static_cast<uint32_t>(tileCnt) % static_cast<uint32_t>(tp.effBufferCount));
    GM_ADDR scaleSlotBase = scaleChannel_->GetSlotBase(static_cast<uint32_t>(tileCnt));

    for (uint32_t r = 0; r < rankSize_; ++r) {
        if (r == rankId_) {
            tailAddrListA_[r] = addrs.aGM + tailDataOff;
            tailAddrListScale_[r] = addrs.aScaleGM + tailScaleOff;
        } else {
            uint32_t remoteIdx = (r > rankId_) ? (r - 1) : r;
            tailAddrListA_[r] = dataSlotBase + static_cast<uint64_t>(remoteIdx) * tileMDataStride_;
            tailAddrListScale_[r] = scaleSlotBase + static_cast<uint64_t>(remoteIdx) * tileMScaleStride_;
        }
        tailAddrListC_[r] = cFragAddrs_[r] + tailCOff;
    }
    tailFragA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutA, AType>(
        MakeFragParam(paddedTailM_, tailM, rankSize_, mmT.k), tailAddrListA_);
    tailFragScaleA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutScaleA, AscendC::fp8_e8m0_t>(
        MakeFragParam(paddedTailM_, tailM, rankSize_, scaleKLen), tailAddrListScale_);
    tailFragC_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutC, CType>(
        MakeFragParam(paddedTailM_, tailM, rankSize_, mmT.n), tailAddrListC_);
}

template <typename AType, typename BType, typename CType, typename TilingDataType>
__aicore__ inline void
AllGatherQbmmMxKernel<AType, BType, CType, TilingDataType>::UpdateMainRoundAddrs(uint32_t roundIdx)
{
    uint32_t delta = roundIdx - curMainRoundIdx_;
    curRoundCOff_ += static_cast<uint64_t>(delta) * tileMCStride_;

    uint32_t fragIdx = 0;
    for (uint32_t r = 0; r < rankSize_; ++r) {
        if (r == rankId_) {
            continue;
        }
        mainAddrListC_[fragIdx] = cFragAddrs_[r] + curRoundCOff_;
        fragIdx++;
    }
    curMainC_.UpdateAddrList(mainAddrListC_);
    curMainRoundIdx_ = roundIdx;
}

template <typename AType, typename BType, typename CType, typename TilingDataType>
__aicore__ inline typename AllGatherQbmmMxKernel<AType, BType, CType, TilingDataType>::TileCtx
AllGatherQbmmMxKernel<AType, BType, CType, TilingDataType>::ResolveTileCtx(int64_t mPos, int64_t headMainRows,
                                                                            int64_t mainRoundRows,
                                                                            int64_t mainSectionRows, uint32_t rankSize,
                                                                            uint32_t commTurn) const
{
    TileCtx ctx{};
    if (mPos < headMainRows) {
        ctx.dependTileIdx = 0;
        ctx.region = HEAD;
        ctx.regionMPos = mPos;
        ctx.fragA = &headFragA_;
        ctx.fragScaleA = &headFragScaleA_;
        ctx.fragC = &headFragC_;
        ctx.rankCnt = static_cast<uint64_t>(rankSize - 1);
    } else if (mPos < headMainRows + mainSectionRows) {
        int64_t relPos = mPos - headMainRows;
        ctx.roundIdx = static_cast<uint32_t>(relPos / mainRoundRows);
        ctx.dependTileIdx = ctx.roundIdx + 1;
        ctx.region = MAIN;
        ctx.regionMPos = relPos % mainRoundRows;
        ctx.fragA = nullptr;        // MAIN reads A/scaleA as continuous GM tensor
        ctx.fragScaleA = nullptr;
        ctx.fragC = &curMainC_;
        ctx.rankCnt = static_cast<uint64_t>(rankSize - 1);
    } else {
        ctx.dependTileIdx = commTurn;
        ctx.region = TAIL;
        ctx.regionMPos = mPos - headMainRows - mainSectionRows;
        ctx.fragA = &tailFragA_;
        ctx.fragScaleA = &tailFragScaleA_;
        ctx.fragC = &tailFragC_;
        ctx.rankCnt = static_cast<uint64_t>(rankSize);
    }
    return ctx;
}

} // namespace Apace
