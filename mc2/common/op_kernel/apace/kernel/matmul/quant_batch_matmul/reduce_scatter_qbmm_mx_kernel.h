/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_scatter_qbmm_mx_kernel.h
 * \brief ReduceScatter 专用 QMM 内核 — 基于 FragmentTensor，消除 R 循环
 *
 * 与 allgather 的 QmmMxKernelAgUdma 的关键区别：
 *   - allgather: 通信在前(AIV 先 AllGather A), 计算在后(AIC 从 win 区读 A)
 *     → FragmentTensor 打包 win 区不同 rank 的地址
 *   - ReduceScatter: 计算在前(AIC 先算 C), 通信在后(AIV PUT C)
 *     → A 数据全在本卡 GM 连续 [m,k], 各 rank 段天然连续
 *     → FragmentTensor 打包本卡 GM 中各 rank 段的地址 (虽然连续, fragment 边界保持 L1 隔离)
 *     → C 输出到 workspace 连续 [m,n], FragmentTensor 打包 workspace 中各 rank 段地址
 *     → 无需 CrossCoreWaitFlag (计算不依赖通信), 仅 SetFlag(1) 通知 AIV remote 段完成
 *
 * FragmentTensor 消除 R 循环的核心：
 *   - 完整逻辑 M 划分为 HEAD / MAIN(按轮) / TAIL 区域 (ResolveTileCtx):
 *     totalLogicalM = headMainRows + tileCnt * effRankCnt * tileM + tailRows
 *   - 当前调用方 (impl.h) 逐 tile 调用 (tileCnt=0, tailCnt=0), 仅触发 HEAD 区域,
 *     totalLogicalM 退化为 rankSize * curTileM
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
#include "apace/block/mmad/qmm_mx_block_mmad_fragment.h"
#include "apace/basic/fragment_tensor/fragment_tensor.h"
#include "apace/basic/fragment_tensor/fragment_tensor_api.h"
#include "apace/tiling/quant_matmul_tiling_data.h"

namespace ReduceScatterFragImpl {

using namespace AscendC;

using LayoutA = Te::NDExtLayoutPtn;
using LayoutB = Te::DNExtLayoutPtn;
using LayoutC = Te::NDExtLayoutPtn;
using ProblemShape = Te::Shape<int64_t, int64_t, int64_t, int64_t>;

enum RegionTag {
    HEAD,
    MAIN,
    TAIL
};

template <typename AType, typename BType, typename CType>
class QmmMxKernelRsFrag {
public:
    __aicore__ inline QmmMxKernelRsFrag() {}
    __aicore__ inline ~QmmMxKernelRsFrag() {}

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, 0>;
    using BlockScheduler =
        Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB, AType>;
    using BlockMmadParams = typename Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB,
                                                                   CType, LayoutC, float, LayoutC>::Params;
    using L1Params = typename Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType,
                                                            LayoutC, float, LayoutC>::L1Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;

    static constexpr int64_t kC0Size = 32;
    static constexpr int64_t kCacheLineAlignMask = 0x7f;
    static constexpr int32_t kScaleC0 = 2;

    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<kC0Size>>;
    using MakeLayoutScaleA = Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<kScaleC0>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<kC0Size>>;
    using MakeLayoutScaleB = Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<kScaleC0>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<Te::C0_ELEMENT<CType>>>;

    using BlockMmadFragC = Blaze::Gemm::Block::QmmMxBlockMmadFragment<0, false, AType, LayoutA, BType, LayoutB, CType,
                                                                      LayoutC, float, LayoutC>;
    static constexpr bool weightNz = BlockMmadFragC::weightNz;
    using FragL1Params = typename BlockMmadFragC::L1Params;
    using FragBlockShape = typename BlockMmadFragC::BlockShape;

    using FragTensorA = Apace::Basic::FragmentTensor<2, Apace::Basic::MAX_FRAGMENT_COUNT, MakeLayoutA, AType>;
    using FragScaleA =
        Apace::Basic::FragmentTensor<2, Apace::Basic::MAX_FRAGMENT_COUNT, MakeLayoutScaleA, AscendC::fp8_e8m0_t>;
    using FragTensorC = Apace::Basic::FragmentTensor<2, Apace::Basic::MAX_FRAGMENT_COUNT, MakeLayoutC, CType>;

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
        bool localLast{false};
        uint32_t fragCount{0};
    };

    struct Params {
        const QuantMatmulTilingData *mmTile{nullptr};
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

    __aicore__ inline void Run(const Params &params);
    __aicore__ inline void operator()(const Params &params)
    {
        Run(params);
    }

private:
    __aicore__ inline void Init(const Params &params);
    __aicore__ inline void Process(const Params &params, const ProblemShape &problemShape, BlockScheduler &bs,
                                   BlockMmadFragC &mmadFrag);

    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2Cache(const ProblemShape &problemShape, int64_t baseM, int64_t baseN, TensorB &gmB,
                                      TensorScaleB &gmScaleB);

    __aicore__ inline void BuildFragmentTensors(const Params &params);
    __aicore__ inline void BuildMainFragment(const Params &params, uint32_t roundIdx);
    __aicore__ inline void BuildTailFragment(const Params &params);
    __aicore__ inline void UpdateMainRoundAddrs(const Params &params, uint32_t roundIdx);
    __aicore__ inline Apace::Basic::FragmentParam<2> MakeFragParam(uint64_t fragSize, uint64_t realFragSize,
                                                                   uint32_t fragCnt, uint64_t shape1) const;

    struct TileCtx {
        uint32_t dependId;
        uint32_t roundIdx;
        RegionTag region;
        int64_t regionMPos;
        const FragTensorA *fragA;
        const FragScaleA *fragScaleA;
        const FragTensorC *fragC;
        uint64_t rankCnt;
    };
    __aicore__ inline TileCtx ResolveTileCtx(int64_t mPos, int64_t headMainRows, int64_t mainRoundRows,
                                             int64_t mainSectionRows, uint32_t rankSize, uint32_t commTurn) const;

    __gm__ AType *aGmAddr_{nullptr};
    __gm__ fp8_e8m0_t *scaleAGmAddr_{nullptr};
    __gm__ BType *bGmAddr_{nullptr};
    __gm__ fp8_e8m0_t *scaleBGmAddr_{nullptr};
    __gm__ CType *cGmAddr_{nullptr};

    FragTensorA headFragA_{};
    FragScaleA headFragScaleA_{};
    FragTensorC headFragC_{};
    FragTensorA curMainA_{};
    FragScaleA curMainScaleA_{};
    FragTensorC curMainC_{};
    FragTensorA tailFragA_{};
    FragScaleA tailFragScaleA_{};
    FragTensorC tailFragC_{};

    // FragmentTensor 仅保存 addrList 指针（不拷贝），故地址数组须为成员变量，
    // 与 FragmentTensor 生命周期一致，避免栈上数组悬空
    static constexpr uint32_t MAX_FRAG = Apace::Basic::MAX_FRAGMENT_COUNT;
    GM_ADDR headAddrListA_[MAX_FRAG]{};
    GM_ADDR headAddrListScale_[MAX_FRAG]{};
    GM_ADDR headAddrListC_[MAX_FRAG]{};
    GM_ADDR mainAddrListA_[MAX_FRAG]{};
    GM_ADDR mainAddrListScale_[MAX_FRAG]{};
    GM_ADDR mainAddrListC_[MAX_FRAG]{};
    GM_ADDR tailAddrListA_[MAX_FRAG]{};
    GM_ADDR tailAddrListScale_[MAX_FRAG]{};
    GM_ADDR tailAddrListC_[MAX_FRAG]{};

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
__aicore__ inline void QmmMxKernelRsFrag<AType, BType, CType>::Init(const Params &params)
{
    if ASCEND_IS_AIV {
        return;
    }
    aGmAddr_ = reinterpret_cast<__gm__ AType *>(params.aGM);
    bGmAddr_ = reinterpret_cast<__gm__ BType *>(params.bGM);
    cGmAddr_ = reinterpret_cast<__gm__ CType *>(params.cGM);
    scaleAGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t *>(params.aScaleGM);
    scaleBGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t *>(params.bScaleGM);
}

template <typename AType, typename BType, typename CType>
template <typename TensorB, typename TensorScaleB>
__aicore__ inline void QmmMxKernelRsFrag<AType, BType, CType>::SetL2Cache(const ProblemShape &problemShape,
                                                                          int64_t baseM, int64_t baseN, TensorB &gmB,
                                                                          TensorScaleB &gmScaleB)
{
    const bool fullMBlock = (baseM >= Te::Get<Blaze::Gemm::MNK_M>(problemShape));
    if constexpr (weightNz) {
        gmB.SetL2CacheHint(fullMBlock ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
    } else {
        const bool bAlignForL2Stream = (Te::Get<Blaze::Gemm::MNK_K>(problemShape) & kCacheLineAlignMask) == 0;
        gmB.SetL2CacheHint((fullMBlock && bAlignForL2Stream) ? Te::CacheMode::CACHE_MODE_DISABLE :
                                                               Te::CacheMode::CACHE_MODE_NORMAL);
    }
    const int64_t scaleNStrideBytes = Te::Get<Blaze::Gemm::MNK_N>(problemShape) * kScaleC0;
    const int64_t scaleBaseNStrideBytes = baseN * kScaleC0;
    const bool scaleAlignForL2Stream =
        (scaleNStrideBytes & kCacheLineAlignMask) == 0 && (scaleBaseNStrideBytes & kCacheLineAlignMask) == 0;
    gmScaleB.SetL2CacheHint((fullMBlock && scaleAlignForL2Stream) ? Te::CacheMode::CACHE_MODE_DISABLE :
                                                                    Te::CacheMode::CACHE_MODE_NORMAL);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void QmmMxKernelRsFrag<AType, BType, CType>::Run(const Params &params)
{
    Init(params);

    const auto &mmT = *params.mmTile;
    const auto &fp = params.fragParams;
    const int64_t Ki = static_cast<int64_t>(fp.k);
    const int64_t Ni = static_cast<int64_t>(fp.n);
    const int64_t scaleKLen = static_cast<int64_t>(fp.scaleKLen);

    bool hasTail = (fp.tailCnt > 0 && fp.tailM > 0);
    uint32_t effRankCnt = (fp.fragCount > 0) ? fp.fragCount : fp.rankSize;
    const int64_t headMainRows = static_cast<int64_t>(fp.headRows) * effRankCnt / fp.rankSize;
    const int64_t mainRoundRows = static_cast<int64_t>(effRankCnt) * static_cast<int64_t>(fp.tileM);
    const int64_t mainSectionRows = static_cast<int64_t>(fp.tileCnt) * mainRoundRows;
    const int64_t tailRows = hasTail ? static_cast<int64_t>(effRankCnt) * static_cast<int64_t>(fp.paddedTailM) : 0;
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
__aicore__ inline void QmmMxKernelRsFrag<AType, BType, CType>::Process(const Params &params,
                                                                       const ProblemShape &problemShape,
                                                                       BlockScheduler &sch, BlockMmadFragC &mmadFrag)
{
    const auto &fp = params.fragParams;
    const int64_t Ki = static_cast<int64_t>(fp.k);
    const int64_t Ni = static_cast<int64_t>(fp.n);
    const int64_t scaleKLen = static_cast<int64_t>(fp.scaleKLen);

    uint32_t effRankCnt = (fp.fragCount > 0) ? fp.fragCount : fp.rankSize;
    const int64_t headMainRows = static_cast<int64_t>(fp.headRows) * effRankCnt / fp.rankSize;
    const int64_t mainRoundRows = static_cast<int64_t>(effRankCnt) * static_cast<int64_t>(fp.tileM);
    const int64_t mainSectionRows = static_cast<int64_t>(fp.tileCnt) * mainRoundRows;

    auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(bGmAddr_), MakeLayoutB{}(Ki, Ni));
    auto gmScaleB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(scaleBGmAddr_), MakeLayoutScaleB{}(scaleKLen, Ni));
    __gm__ float *biasNull = nullptr;
    auto gmBias =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(biasNull), Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, Ni));

    Te::Coord<int64_t, int64_t, int64_t, int64_t> blockIdx;
    int64_t mPos = 0L, nPos = 0L;

    // localLast 模式: 检测 remote→local fragment 边界, 提前 SetFlag(1)
    int64_t fragM = headMainRows / effRankCnt;
    int64_t localFragBoundary = fp.localLast ? (headMainRows - fragM) : -1;
    bool remoteFlagSet = false;

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

        // localLast: mPos 跨越 remote→local 边界时, 提前 SetFlag(1) 通知 AIV 启动 AllToAll
        if (fp.localLast && !remoteFlagSet && mPos >= localFragBoundary) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(1);
            remoteFlagSet = true;
        }

        auto ctx = ResolveTileCtx(mPos, headMainRows, mainRoundRows, mainSectionRows, effRankCnt, fp.commTurn);

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
        mmadFrag(blockA, gmBlockB, blockScaleA, gmBlockScaleB, gmBlockBias, cFragAddrs_,
                 static_cast<uint64_t>(fp.mPerRank), static_cast<uint64_t>(fp.tileM), static_cast<uint64_t>(fp.tileCnt),
                 static_cast<uint64_t>(fp.tailM), ctx.rankCnt, Ni, singleShape, ctx.regionMPos, nPos, 0, blockC);
    }

    // localLast: 确保所有 core 都设置 flag(1) (某些 core 可能未跨越边界)
    if (fp.localLast && !remoteFlagSet) {
        CrossCoreSetFlag<0x2, PIPE_FIX>(1);
    }
}

template <typename AType, typename BType, typename CType>
__aicore__ inline Apace::Basic::FragmentParam<2> QmmMxKernelRsFrag<AType, BType, CType>::MakeFragParam(
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
__aicore__ inline void QmmMxKernelRsFrag<AType, BType, CType>::BuildFragmentTensors(const Params &params)
{
    // ReduceScatter 场景: A 数据全在本卡 GM 连续 [m, k]
    // 各 rank 段在 GM 中连续排列: rank 0 的 A[0:chunkM], rank 1 的 A[chunkM:2*chunkM], ...
    // FragmentTensor 打包各 rank 段地址, 保持 per-fragment L1 缓存隔离
    // C 输出到 workspace 连续 [m, n], 同样按 rank 段打包

    const auto &fp = params.fragParams;

    // C fragment 地址: workspace 中各 rank 段的起始位置 (保持原始 rank 顺序)
    // localLast 模式下 addrListC 会按 [remote..., local] 重排, 但 cFragAddrs_ 保持原始顺序
    // mmadFrag 内部用 cFragAddrs_ 做 L1 cache 管理, 用 blockC (从 FragmentTensor) 做实际输出
    for (uint32_t r = 0; r < fp.rankSize; ++r) {
        cFragAddrs_[r] = params.cGM + static_cast<uint64_t>(r) * fp.mPerRank * params.cBytesPerM;
    }

    tileMDataStride_ = static_cast<uint64_t>(fp.tileM) * params.dataBytesPerMRow;
    tileMScaleStride_ = static_cast<uint64_t>(fp.tileM) * params.scaleBytesPerMRow;
    tileMCStride_ = static_cast<uint64_t>(fp.tileM) * params.cBytesPerM;

    // HEAD: 当前 comm tile 的所有 rank 段
    // 每个 fragment 指向 aGm + r * chunkM * K + tileOffset (当前 tile 在 chunkM 内的偏移)
    // tileOffset 由调用方通过 params.aGM 的偏移传入 (impl.h 设置 aGM = aGm + tileMOffset * K)
    if (fp.localLast) {
        uint32_t idx = 0;
        for (uint32_t r = 0; r < fp.rankSize; ++r) {
            if (r == fp.rankId) {
                continue;
            }
            headAddrListA_[idx] = params.aGM + static_cast<uint64_t>(r) * fp.mPerRank * params.dataBytesPerMRow;
            headAddrListScale_[idx] =
                params.aScaleGM + static_cast<uint64_t>(r) * fp.mPerRank * params.scaleBytesPerMRow;
            headAddrListC_[idx] = cFragAddrs_[r];
            ++idx;
        }
        headAddrListA_[idx] = params.aGM + static_cast<uint64_t>(fp.rankId) * fp.mPerRank * params.dataBytesPerMRow;
        headAddrListScale_[idx] =
            params.aScaleGM + static_cast<uint64_t>(fp.rankId) * fp.mPerRank * params.scaleBytesPerMRow;
        headAddrListC_[idx] = cFragAddrs_[fp.rankId];
    } else {
        for (uint32_t r = 0; r < fp.rankSize; ++r) {
            headAddrListA_[r] = params.aGM + static_cast<uint64_t>(r) * fp.mPerRank * params.dataBytesPerMRow;
            headAddrListScale_[r] = params.aScaleGM + static_cast<uint64_t>(r) * fp.mPerRank * params.scaleBytesPerMRow;
            headAddrListC_[r] = cFragAddrs_[r];
        }
    }
    uint64_t fragM = fp.headRows / fp.rankSize;
    uint32_t fragCnt = (fp.fragCount > 0) ? fp.fragCount : fp.rankSize;
    headFragA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutA, AType>(
        MakeFragParam(fragM, fragM, fragCnt, fp.k), headAddrListA_);
    headFragScaleA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutScaleA, fp8_e8m0_t>(
        MakeFragParam(fragM, fragM, fragCnt, fp.scaleKLen), headAddrListScale_);
    headFragC_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutC, CType>(
        MakeFragParam(fragM, fragM, fragCnt, fp.n), headAddrListC_);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void QmmMxKernelRsFrag<AType, BType, CType>::BuildMainFragment(const Params &params,
                                                                                 uint32_t roundIdx)
{
    const auto &fp = params.fragParams;

    curRoundDataOff_ = static_cast<uint64_t>(roundIdx) * tileMDataStride_;
    curRoundScaleOff_ = static_cast<uint64_t>(roundIdx) * tileMScaleStride_;
    curRoundCOff_ = static_cast<uint64_t>(roundIdx) * tileMCStride_;

    for (uint32_t r = 0; r < fp.rankSize; ++r) {
        mainAddrListA_[r] =
            params.aGM + static_cast<uint64_t>(r) * fp.mPerRank * params.dataBytesPerMRow + curRoundDataOff_;
        mainAddrListScale_[r] =
            params.aScaleGM + static_cast<uint64_t>(r) * fp.mPerRank * params.scaleBytesPerMRow + curRoundScaleOff_;
        mainAddrListC_[r] = cFragAddrs_[r] + curRoundCOff_;
    }
    curMainA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutA, AType>(
        MakeFragParam(fp.tileM, fp.tileM, fp.rankSize, fp.k), mainAddrListA_);
    curMainScaleA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutScaleA, fp8_e8m0_t>(
        MakeFragParam(fp.tileM, fp.tileM, fp.rankSize, fp.scaleKLen), mainAddrListScale_);
    curMainC_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutC, CType>(
        MakeFragParam(fp.tileM, fp.tileM, fp.rankSize, fp.n), mainAddrListC_);
    curMainRoundIdx_ = roundIdx;
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void QmmMxKernelRsFrag<AType, BType, CType>::BuildTailFragment(const Params &params)
{
    const auto &fp = params.fragParams;

    uint64_t tailRowOff = fp.headRows + static_cast<uint64_t>(fp.tileCnt) * fp.tileM * fp.rankSize;
    uint64_t tailDataOff = tailRowOff * params.dataBytesPerMRow;
    uint64_t tailScaleOff = tailRowOff * params.scaleBytesPerMRow;
    uint64_t tailCOff = tailRowOff * params.cBytesPerM;

    for (uint32_t r = 0; r < fp.rankSize; ++r) {
        tailAddrListA_[r] = params.aGM + static_cast<uint64_t>(r) * fp.mPerRank * params.dataBytesPerMRow + tailDataOff;
        tailAddrListScale_[r] =
            params.aScaleGM + static_cast<uint64_t>(r) * fp.mPerRank * params.scaleBytesPerMRow + tailScaleOff;
        tailAddrListC_[r] = cFragAddrs_[r] + tailCOff;
    }
    tailFragA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutA, AType>(
        MakeFragParam(fp.paddedTailM, fp.tailM, fp.rankSize, fp.k), tailAddrListA_);
    tailFragScaleA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutScaleA, fp8_e8m0_t>(
        MakeFragParam(fp.paddedTailM, fp.tailM, fp.rankSize, fp.scaleKLen), tailAddrListScale_);
    tailFragC_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutC, CType>(
        MakeFragParam(fp.paddedTailM, fp.tailM, fp.rankSize, fp.n), tailAddrListC_);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void QmmMxKernelRsFrag<AType, BType, CType>::UpdateMainRoundAddrs(const Params &params,
                                                                                    uint32_t roundIdx)
{
    const auto &fp = params.fragParams;
    uint32_t delta = roundIdx - curMainRoundIdx_;
    curRoundDataOff_ += static_cast<uint64_t>(delta) * tileMDataStride_;
    curRoundScaleOff_ += static_cast<uint64_t>(delta) * tileMScaleStride_;
    curRoundCOff_ += static_cast<uint64_t>(delta) * tileMCStride_;

    for (uint32_t r = 0; r < fp.rankSize; ++r) {
        mainAddrListA_[r] =
            params.aGM + static_cast<uint64_t>(r) * fp.mPerRank * params.dataBytesPerMRow + curRoundDataOff_;
        mainAddrListScale_[r] =
            params.aScaleGM + static_cast<uint64_t>(r) * fp.mPerRank * params.scaleBytesPerMRow + curRoundScaleOff_;
        mainAddrListC_[r] = cFragAddrs_[r] + curRoundCOff_;
    }
    curMainA_.UpdateAddrList(mainAddrListA_);
    curMainScaleA_.UpdateAddrList(mainAddrListScale_);
    curMainC_.UpdateAddrList(mainAddrListC_);
    curMainRoundIdx_ = roundIdx;
}

template <typename AType, typename BType, typename CType>
__aicore__ inline typename QmmMxKernelRsFrag<AType, BType, CType>::TileCtx
QmmMxKernelRsFrag<AType, BType, CType>::ResolveTileCtx(int64_t mPos, int64_t headMainRows, int64_t mainRoundRows,
                                                       int64_t mainSectionRows, uint32_t rankSize,
                                                       uint32_t commTurn) const
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
