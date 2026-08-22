/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qmm_mx_kernel_rs_frag.h
 * \brief ReduceScatter 专用 QMM 内核 — 基于 FragmentTensor，消除 R 循环
 *
 * 与 allgather 的 QmmMxKernelAgUdma 的关键区别：
 *   - allgather: 通信在前(AIV 先 AllGather A), 计算在后(AIC 从 win 区读 A)
 *     → FragmentTensor 打包 win 区不同 rank 的地址
 *   - ReduceScatter: 计算在前(AIC 先算 C), 通信在后(AIV PUT C)
 *     → A 数据全在本卡 GM 连续 [m,k], 各 rank 段天然连续
 *     → FragmentTensor 打包本卡 GM 中各 rank 段的地址 (虽然连续, fragment 边界保持 L1 隔离)
 *     → C 输出到 workspace 连续 [m,n], FragmentTensor 打包 workspace 中各 rank 段地址
 *     → 无需 CrossCoreWaitFlag (计算不依赖通信)
 *
 * FragmentTensor 消除 R 循环的核心：
 *   - totalLogicalM = rankSize * chunkM = m
 *   - BlockScheduler 遍历 totalLogicalM / baseM 个 tile
 *   - 当 tile 跨越 rank 边界时, FragmentSliceCopy 自动从不同 fragment 地址加载
 *   - 每个 fragment 保持独立的 L1 缓存上下文 (与 per-tile 子调用一致)
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
#include "apace/block/blaze_ext/gemm/block/qmm_mx_block_mmad_fragment.h"
#include "apace/basic/fragment_tensor/fragment_tensor.h"
#include "apace/basic/fragment_tensor/fragment_tensor_api.h"
#include "apace/tiling/quant_matmul_tiling_data.h"

namespace ReduceScatterFragImpl {

using namespace AscendC;

using LayoutA = Te::NDExtLayoutPtn;
using LayoutB = Te::DNExtLayoutPtn;
using LayoutC = Te::NDExtLayoutPtn;
using ProblemShape = Te::Shape<int64_t, int64_t, int64_t, int64_t>;

enum RegionTag { HEAD, MAIN, TAIL };

template <typename AType, typename BType, typename CType>
class QmmMxKernelRsFrag {
public:
    __aicore__ inline QmmMxKernelRsFrag() {}
    __aicore__ inline ~QmmMxKernelRsFrag() {}

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, 0>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, 0, LayoutA, LayoutB, AType>;
    using BlockMmadParams = typename Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, float, LayoutC>::Params;
    using L1Params = typename Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, float, LayoutC>::L1Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;

    static constexpr int64_t kC0Size = 32;
    static constexpr int64_t kCacheLineAlignMask = 0x7f;
    static constexpr int32_t kScaleC0 = 2;

    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<kC0Size>>;
    using MakeLayoutScaleA = Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<kScaleC0>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<kC0Size>>;
    using MakeLayoutScaleB = Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<kScaleC0>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<Te::C0_ELEMENT<CType>>>;

    using BlockMmadFragC = Blaze::Gemm::Block::QmmMxBlockMmadFragment<
        0, false, AType, LayoutA, BType, LayoutB, CType, LayoutC, float, LayoutC>;
    static constexpr bool weightNz = BlockMmadFragC::weightNz;
    using FragL1Params = typename BlockMmadFragC::L1Params;
    using FragBlockShape = typename BlockMmadFragC::BlockShape;

    using FragTensorA = Apace::Basic::FragmentTensor<2, Apace::Basic::MAX_FRAGMENT_COUNT,
                              MakeLayoutA, AType>;
    using FragScaleA = Apace::Basic::FragmentTensor<2, Apace::Basic::MAX_FRAGMENT_COUNT,
                              MakeLayoutScaleA, AscendC::fp8_e8m0_t>;
    using FragTensorC = Apace::Basic::FragmentTensor<2, Apace::Basic::MAX_FRAGMENT_COUNT,
                              MakeLayoutC, CType>;

    struct QBMMTiling {
        uint32_t baseM{0};
        uint32_t baseN{0};
        uint32_t baseK{0};
        uint32_t dbL0C{0};
    };

    struct FragmentParams {
        uint32_t tileCnt{0};
        uint32_t tileM{0};
        uint32_t tailCnt{0};
        uint32_t tailM{0};
        uint32_t paddedTailM{0};
        uint32_t commTurn{0};
        uint64_t headRows{0};
        uint32_t rankId{0};
        uint32_t rankSize{0};
        uint32_t mPerRank{0};
        uint64_t k{0};
        uint64_t n{0};
        uint64_t scaleKLen{0};
    };

    struct Params {
        const QuantMatmulTilingData* mmTile{nullptr};
        QBMMTiling qbmmParams;
        FragmentParams fragParams;

        GM_ADDR aGM{nullptr};
        GM_ADDR aScaleGM{nullptr};
        GM_ADDR bGM{nullptr};
        GM_ADDR bScaleGM{nullptr};
        GM_ADDR cGM{nullptr};

        uint64_t dataBytesPerMRow{0};
        uint64_t scaleBytesPerMRow{0};
        uint64_t cBytesPerM{0};
    };

    __aicore__ inline void Run(const Params& params);
    __aicore__ inline void operator()(const Params& params) { Run(params); }

private:
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Process(const Params& params, const ProblemShape& problemShape,
                                   BlockScheduler& bs, BlockMmadFragC& mmadFrag);

    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2Cache(
        const ProblemShape& problemShape, int64_t baseM, int64_t baseN,
        TensorB& gmB, TensorScaleB& gmScaleB);

    __aicore__ inline void BuildFragmentTensors(const Params& params);
    __aicore__ inline void BuildMainFragment(const Params& params, uint32_t roundIdx);
    __aicore__ inline void BuildTailFragment(const Params& params);
    __aicore__ inline void UpdateMainRoundAddrs(const Params& params, uint32_t roundIdx);
    __aicore__ inline Apace::Basic::FragmentParam<2> MakeFragParam(
        uint64_t fragSize, uint64_t realFragSize, uint32_t fragCnt, uint64_t shape1) const;

    struct TileCtx {
        uint32_t dependId;
        uint32_t roundIdx;
        RegionTag region;
        int64_t regionMPos;
        const FragTensorA* fragA;
        const FragScaleA* fragScaleA;
        const FragTensorC* fragC;
        uint64_t rankCnt;
    };
    __aicore__ inline TileCtx ResolveTileCtx(int64_t mPos,
        int64_t headMainRows, int64_t mainRoundRows, int64_t mainSectionRows,
        uint32_t rankSize, uint32_t commTurn) const;

    __gm__ AType* aGmAddr_{nullptr};
    __gm__ fp8_e8m0_t* scaleAGmAddr_{nullptr};
    __gm__ BType* bGmAddr_{nullptr};
    __gm__ fp8_e8m0_t* scaleBGmAddr_{nullptr};
    __gm__ CType* cGmAddr_{nullptr};

    FragTensorA headFragA_{};
    FragScaleA headFragScaleA_{};
    FragTensorC headFragC_{};
    FragTensorA curMainA_{};
    FragScaleA curMainScaleA_{};
    FragTensorC curMainC_{};
    FragTensorA tailFragA_{};
    FragScaleA tailFragScaleA_{};
    FragTensorC tailFragC_{};

    GM_ADDR cFragAddrs_[8]{};
    uint32_t curMainRoundIdx_{0xFFFFFFFF};
    bool tailBuilt_{false};
    uint64_t curRoundDataOff_{};
    uint64_t curRoundScaleOff_{};
    uint64_t curRoundCOff_{};

    uint64_t tileMDataStride_{};
    uint64_t tileMScaleStride_{};
    uint64_t tileMCStride_{};
};

template <typename AType, typename BType, typename CType>
__aicore__ inline void QmmMxKernelRsFrag<AType, BType, CType>::Init(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }
    aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.aGM);
    bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.bGM);
    cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.cGM);
    scaleAGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.aScaleGM);
    scaleBGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.bScaleGM);
}

template <typename AType, typename BType, typename CType>
template <typename TensorB, typename TensorScaleB>
__aicore__ inline void QmmMxKernelRsFrag<AType, BType, CType>::SetL2Cache(
    const ProblemShape& problemShape, int64_t baseM, int64_t baseN,
    TensorB& gmB, TensorScaleB& gmScaleB)
{
    const bool fullMBlock = (baseM >= Te::Get<Blaze::Gemm::MNK_M>(problemShape));
    if constexpr (weightNz) {
        gmB.SetL2CacheHint(
            fullMBlock ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
    } else {
        const bool bAlignForL2Stream =
            (Te::Get<Blaze::Gemm::MNK_K>(problemShape) & kCacheLineAlignMask) == 0;
        gmB.SetL2CacheHint(
            (fullMBlock && bAlignForL2Stream) ? Te::CacheMode::CACHE_MODE_DISABLE
                                              : Te::CacheMode::CACHE_MODE_NORMAL);
    }
    const int64_t scaleNStrideBytes = Te::Get<Blaze::Gemm::MNK_N>(problemShape) * kScaleC0;
    const int64_t scaleBaseNStrideBytes = baseN * kScaleC0;
    const bool scaleAlignForL2Stream =
        (scaleNStrideBytes & kCacheLineAlignMask) == 0 && (scaleBaseNStrideBytes & kCacheLineAlignMask) == 0;
    gmScaleB.SetL2CacheHint(
        (fullMBlock && scaleAlignForL2Stream) ? Te::CacheMode::CACHE_MODE_DISABLE
                                              : Te::CacheMode::CACHE_MODE_NORMAL);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void QmmMxKernelRsFrag<AType, BType, CType>::Run(const Params& params)
{
    Init(params);

    const auto& mmT = *params.mmTile;
    const auto& fp = params.fragParams;
    const int64_t Ki = static_cast<int64_t>(fp.k);
    const int64_t Ni = static_cast<int64_t>(fp.n);
    const int64_t scaleKLen = static_cast<int64_t>(fp.scaleKLen);

    bool hasTail = (fp.tailCnt > 0 && fp.tailM > 0);
    const int64_t headMainRows = static_cast<int64_t>(fp.headRows);
    const int64_t mainRoundRows = static_cast<int64_t>(fp.rankSize) * static_cast<int64_t>(fp.tileM);
    const int64_t mainSectionRows = static_cast<int64_t>(fp.tileCnt) * mainRoundRows;
    const int64_t tailRows = hasTail ? static_cast<int64_t>(fp.rankSize) * static_cast<int64_t>(fp.paddedTailM) : 0;
    const int64_t totalLogicalM = headMainRows + mainSectionRows + tailRows;

    BlockSchedulerParams schParams;
    schParams.baseM = mmT.baseM;
    schParams.baseN = mmT.baseN;
    schParams.mTailTile = mmT.mTailTile;
    schParams.nTailTile = mmT.nTailTile;
    schParams.mBaseTailSplitCnt = mmT.mBaseTailSplitCnt;
    schParams.nBaseTailSplitCnt = mmT.nBaseTailSplitCnt;
    schParams.mTailMain = mmT.mTailMain;
    schParams.nTailMain = mmT.nTailMain;

    FragBlockShape l0TileShape{static_cast<int64_t>(mmT.baseM), static_cast<int64_t>(mmT.baseN),
                              static_cast<int64_t>(mmT.baseK), 0L};
    FragL1Params fragL1{static_cast<uint64_t>(mmT.stepK) * mmT.baseK, mmT.scaleKL1, mmT.nBufferNum};

    ProblemShape problemShape{totalLogicalM, Ni, Ki, 1};
    BlockScheduler sch(problemShape, schParams);
    BlockMmadFragC mmadFrag;
    mmadFrag.Init(problemShape, l0TileShape, fragL1, false, mmT.dbL0c > 1);

    BuildFragmentTensors(params);

    Process(params, problemShape, sch, mmadFrag);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void QmmMxKernelRsFrag<AType, BType, CType>::Process(
    const Params& params, const ProblemShape& problemShape, BlockScheduler& sch, BlockMmadFragC& mmadFrag)
{
    const auto& fp = params.fragParams;
    const int64_t Ki = static_cast<int64_t>(fp.k);
    const int64_t Ni = static_cast<int64_t>(fp.n);
    const int64_t scaleKLen = static_cast<int64_t>(fp.scaleKLen);

    const int64_t headMainRows = static_cast<int64_t>(fp.headRows);
    const int64_t mainRoundRows = static_cast<int64_t>(fp.rankSize) * static_cast<int64_t>(fp.tileM);
    const int64_t mainSectionRows = static_cast<int64_t>(fp.tileCnt) * mainRoundRows;

    auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(bGmAddr_),
                MakeLayoutB{}(Ki, Ni));
    auto gmScaleB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(scaleBGmAddr_),
                      MakeLayoutScaleB{}(scaleKLen, Ni));
    __gm__ float *biasNull = nullptr;
    auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(biasNull),
                    Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, Ni));

    Te::Coord<int64_t, int64_t, int64_t, int64_t> blockIdx;
    int64_t mPos = 0L, nPos = 0L;

    while (sch.GetTileIdx(blockIdx)) {
        auto singleShape = sch.template GetBlockShape<
          Blaze::Gemm::QuantMode::MX_PERGROUP_MODE, Blaze::Gemm::QuantMode::MX_PERGROUP_MODE,
          BlockMmadFragC::weightNz>(blockIdx);
        int64_t curMtile = Te::Get<0>(singleShape);
        int64_t curNtile = Te::Get<1>(singleShape);
        if (curMtile <= 0 || curNtile <= 0) {
            break;
        }
        sch.GetTileCoord(blockIdx, mPos, nPos);

        auto ctx = ResolveTileCtx(mPos, headMainRows, mainRoundRows, mainSectionRows,
                                  fp.rankSize, fp.commTurn);

        SetL2Cache(problemShape, curMtile, curNtile, gmB, gmScaleB);

        auto gmBlockB = gmB.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(Ki, curNtile));
        auto gmBlockScaleB = gmScaleB.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(scaleKLen, curNtile));
        auto gmBlockBias = gmBias.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(1L, curNtile));

        auto coordA = Te::MakeCoord(ctx.regionMPos, 0L);
        auto coordC = Te::MakeCoord(ctx.regionMPos, nPos);
        auto shapeA = Te::MakeShape(curMtile, Ki);
        auto shapeScaleA = Te::MakeShape(curMtile, scaleKLen);
        auto shapeC = Te::MakeShape(curMtile, curNtile);

        if (ctx.region == MAIN) {
            if (curMainRoundIdx_ == 0xFFFFFFFF) {
                BuildMainFragment(params, ctx.roundIdx);
            } else if (curMainRoundIdx_ != ctx.roundIdx) {
                UpdateMainRoundAddrs(params, ctx.roundIdx);
            }
        } else if (ctx.region == TAIL && !tailBuilt_) {
            BuildTailFragment(params);
            tailBuilt_ = true;
        }

        auto blockA = ctx.fragA->Slice(coordA, shapeA);
        auto blockScaleA = ctx.fragScaleA->Slice(coordA, shapeScaleA);
        auto blockC = ctx.fragC->Slice(coordC, shapeC);
        mmadFrag(blockA, gmBlockB, blockScaleA, gmBlockScaleB, gmBlockBias,
                cFragAddrs_, static_cast<uint64_t>(fp.mPerRank), static_cast<uint64_t>(fp.tileM),
                static_cast<uint64_t>(fp.tileCnt), static_cast<uint64_t>(fp.tailM),
                ctx.rankCnt, Ni, singleShape, ctx.regionMPos, nPos, 0, blockC);
    }
}

template <typename AType, typename BType, typename CType>
__aicore__ inline Apace::Basic::FragmentParam<2>
QmmMxKernelRsFrag<AType, BType, CType>::MakeFragParam(
    uint64_t fragSize, uint64_t realFragSize, uint32_t fragCnt, uint64_t shape1) const
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

template <typename AType, typename BType, typename CType>
__aicore__ inline void QmmMxKernelRsFrag<AType, BType, CType>::BuildFragmentTensors(const Params& params)
{
    // ReduceScatter 场景: A 数据全在本卡 GM 连续 [m, k]
    // 各 rank 段在 GM 中连续排列: rank 0 的 A[0:chunkM], rank 1 的 A[chunkM:2*chunkM], ...
    // FragmentTensor 打包各 rank 段地址, 保持 per-fragment L1 缓存隔离
    // C 输出到 workspace 连续 [m, n], 同样按 rank 段打包

    const auto& fp = params.fragParams;
    constexpr uint32_t MAX_FRAG = Apace::Basic::MAX_FRAGMENT_COUNT;

    // C fragment 地址: workspace 中各 rank 段的起始位置
    for (uint32_t r = 0; r < fp.rankSize; ++r) {
        cFragAddrs_[r] = params.cGM + static_cast<uint64_t>(r) * fp.mPerRank * params.cBytesPerM;
    }

    tileMDataStride_ = static_cast<uint64_t>(fp.tileM) * params.dataBytesPerMRow;
    tileMScaleStride_ = static_cast<uint64_t>(fp.tileM) * params.scaleBytesPerMRow;
    tileMCStride_ = static_cast<uint64_t>(fp.tileM) * params.cBytesPerM;

    // HEAD: 当前 comm tile 的所有 rank 段
    // 每个 fragment 指向 aGm + r * chunkM * K + tileOffset (当前 tile 在 chunkM 内的偏移)
    // tileOffset 由调用方通过 params.aGM 的偏移传入 (impl.h 设置 aGM = aGm + tileMOffset * K)
    GM_ADDR addrListA[MAX_FRAG];
    GM_ADDR addrListScale[MAX_FRAG];
    GM_ADDR addrListC[MAX_FRAG];
    for (uint32_t r = 0; r < fp.rankSize; ++r) {
        addrListA[r] = params.aGM + static_cast<uint64_t>(r) * fp.mPerRank * params.dataBytesPerMRow;
        addrListScale[r] = params.aScaleGM + static_cast<uint64_t>(r) * fp.mPerRank * params.scaleBytesPerMRow;
        addrListC[r] = cFragAddrs_[r];
    }
    // fragmentSize = headRows / rankSize = curTileM (每个 rank 段在当前 tile 的行数)
    uint64_t fragM = fp.headRows / fp.rankSize;
    headFragA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutA, AType>(
        MakeFragParam(fragM, fragM, fp.rankSize, fp.k), addrListA);
    headFragScaleA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutScaleA, fp8_e8m0_t>(
        MakeFragParam(fragM, fragM, fp.rankSize, fp.scaleKLen), addrListScale);
    headFragC_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutC, CType>(
        MakeFragParam(fragM, fragM, fp.rankSize, fp.n), addrListC);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void QmmMxKernelRsFrag<AType, BType, CType>::BuildMainFragment(
    const Params& params, uint32_t roundIdx)
{
    const auto& fp = params.fragParams;
    constexpr uint32_t MAX_FRAG = Apace::Basic::MAX_FRAGMENT_COUNT;

    curRoundDataOff_ = static_cast<uint64_t>(roundIdx) * tileMDataStride_;
    curRoundScaleOff_ = static_cast<uint64_t>(roundIdx) * tileMScaleStride_;
    curRoundCOff_ = static_cast<uint64_t>(roundIdx) * tileMCStride_;

    GM_ADDR addrListA[MAX_FRAG];
    GM_ADDR addrListScale[MAX_FRAG];
    GM_ADDR addrListC[MAX_FRAG];
    for (uint32_t r = 0; r < fp.rankSize; ++r) {
        addrListA[r] = params.aGM + static_cast<uint64_t>(r) * fp.mPerRank * params.dataBytesPerMRow
                     + curRoundDataOff_;
        addrListScale[r] = params.aScaleGM + static_cast<uint64_t>(r) * fp.mPerRank * params.scaleBytesPerMRow
                         + curRoundScaleOff_;
        addrListC[r] = cFragAddrs_[r] + curRoundCOff_;
    }
    curMainA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutA, AType>(
        MakeFragParam(fp.tileM, fp.tileM, fp.rankSize, fp.k), addrListA);
    curMainScaleA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutScaleA, fp8_e8m0_t>(
        MakeFragParam(fp.tileM, fp.tileM, fp.rankSize, fp.scaleKLen), addrListScale);
    curMainC_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutC, CType>(
        MakeFragParam(fp.tileM, fp.tileM, fp.rankSize, fp.n), addrListC);
    curMainRoundIdx_ = roundIdx;
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void QmmMxKernelRsFrag<AType, BType, CType>::BuildTailFragment(const Params& params)
{
    const auto& fp = params.fragParams;
    constexpr uint32_t MAX_FRAG = Apace::Basic::MAX_FRAGMENT_COUNT;

    uint64_t tailRowOff = fp.headRows + static_cast<uint64_t>(fp.tileCnt) * fp.tileM * fp.rankSize;
    uint64_t tailDataOff = tailRowOff * params.dataBytesPerMRow;
    uint64_t tailScaleOff = tailRowOff * params.scaleBytesPerMRow;
    uint64_t tailCOff = tailRowOff * params.cBytesPerM;

    GM_ADDR addrListA[MAX_FRAG];
    GM_ADDR addrListScale[MAX_FRAG];
    GM_ADDR addrListC[MAX_FRAG];
    for (uint32_t r = 0; r < fp.rankSize; ++r) {
        addrListA[r] = params.aGM + static_cast<uint64_t>(r) * fp.mPerRank * params.dataBytesPerMRow
                     + tailDataOff;
        addrListScale[r] = params.aScaleGM + static_cast<uint64_t>(r) * fp.mPerRank * params.scaleBytesPerMRow
                         + tailScaleOff;
        addrListC[r] = cFragAddrs_[r] + tailCOff;
    }
    tailFragA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutA, AType>(
        MakeFragParam(fp.paddedTailM, fp.tailM, fp.rankSize, fp.k), addrListA);
    tailFragScaleA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutScaleA, fp8_e8m0_t>(
        MakeFragParam(fp.paddedTailM, fp.tailM, fp.rankSize, fp.scaleKLen), addrListScale);
    tailFragC_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutC, CType>(
        MakeFragParam(fp.paddedTailM, fp.tailM, fp.rankSize, fp.n), addrListC);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void QmmMxKernelRsFrag<AType, BType, CType>::UpdateMainRoundAddrs(
    const Params& params, uint32_t roundIdx)
{
    const auto& fp = params.fragParams;
    uint32_t delta = roundIdx - curMainRoundIdx_;
    curRoundDataOff_ += static_cast<uint64_t>(delta) * tileMDataStride_;
    curRoundScaleOff_ += static_cast<uint64_t>(delta) * tileMScaleStride_;
    curRoundCOff_ += static_cast<uint64_t>(delta) * tileMCStride_;

    constexpr uint32_t MAX_FRAG = Apace::Basic::MAX_FRAGMENT_COUNT;
    GM_ADDR addrListA[MAX_FRAG];
    GM_ADDR addrListScale[MAX_FRAG];
    GM_ADDR addrListC[MAX_FRAG];
    for (uint32_t r = 0; r < fp.rankSize; ++r) {
        addrListA[r] = params.aGM + static_cast<uint64_t>(r) * fp.mPerRank * params.dataBytesPerMRow
                     + curRoundDataOff_;
        addrListScale[r] = params.aScaleGM + static_cast<uint64_t>(r) * fp.mPerRank * params.scaleBytesPerMRow
                         + curRoundScaleOff_;
        addrListC[r] = cFragAddrs_[r] + curRoundCOff_;
    }
    curMainA_.UpdateAddrList(addrListA);
    curMainScaleA_.UpdateAddrList(addrListScale);
    curMainC_.UpdateAddrList(addrListC);
    curMainRoundIdx_ = roundIdx;
}

template <typename AType, typename BType, typename CType>
__aicore__ inline typename QmmMxKernelRsFrag<AType, BType, CType>::TileCtx
QmmMxKernelRsFrag<AType, BType, CType>::ResolveTileCtx(
    int64_t mPos, int64_t headMainRows, int64_t mainRoundRows, int64_t mainSectionRows,
    uint32_t rankSize, uint32_t commTurn) const
{
    TileCtx ctx{};
    if (mPos < headMainRows) {
        ctx.dependId = 0;
        ctx.region = HEAD;
        ctx.regionMPos = mPos;
        ctx.fragA = &headFragA_;
        ctx.fragScaleA = &headFragScaleA_;
        ctx.fragC = &headFragC_;
        ctx.rankCnt = static_cast<uint64_t>(rankSize);
    } else if (mPos < headMainRows + mainSectionRows) {
        int64_t relPos = mPos - headMainRows;
        ctx.roundIdx = static_cast<uint32_t>(relPos / mainRoundRows);
        ctx.dependId = ctx.roundIdx + 1;
        ctx.region = MAIN;
        ctx.regionMPos = relPos % mainRoundRows;
        ctx.fragA = &curMainA_;
        ctx.fragScaleA = &curMainScaleA_;
        ctx.fragC = &curMainC_;
        ctx.rankCnt = static_cast<uint64_t>(rankSize);
    } else {
        ctx.dependId = commTurn;
        ctx.region = TAIL;
        ctx.regionMPos = mPos - headMainRows - mainSectionRows;
        ctx.fragA = &tailFragA_;
        ctx.fragScaleA = &tailFragScaleA_;
        ctx.fragC = &tailFragC_;
        ctx.rankCnt = static_cast<uint64_t>(rankSize);
    }
    return ctx;
}

} // namespace ReduceScatterFragImpl
