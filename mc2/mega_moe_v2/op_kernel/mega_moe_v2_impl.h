/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* V2 GMM1+epilogue / GMM2+combine block primitives, driven by WaveScheduler::ForEachBlock.
   Reuses BlockMmad / BlockEpilogue from v1 -- we only restructure the outer loop. */

#ifndef MEGA_MOE_V2_IMPL_H
#define MEGA_MOE_V2_IMPL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"

// tensor_api / blaze MUST come before V2 own headers. V2 headers do
// file-scope `using namespace AscendC`, which leaks AscendC::SYNC_MODE_SHIFT_VALUE
// into global; tensor_api transitively pulls c_api/sync/sync.h which has its
// own global SYNC_MODE_SHIFT_VALUE — both visible at the same scope =
// "ambiguous" compile error in --pkg builds. Including tensor_api before V2
// headers means c_api/sync's lookup happens before AscendC is brought in.
#include "include/tensor_api/tensor.h"
#include "blaze/block/block_mmad_mx.h"
#include "blaze/block/block_scheduler_swizzle.h"

#include "mega_moe_v2_block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_v2_base.h"
#include "mega_moe_v2_wave_scheduler.h"
#include "mega_moe_v2_combine_send.h"

using namespace AscendC;

namespace MegaMoeV2Impl {

constexpr uint32_t L1_TILE_M = 128;
constexpr uint32_t L1_TILE_N = 256;
constexpr uint32_t L1_TILE_K = 256;
constexpr uint32_t L0_TILE_K = 128;
constexpr uint32_t SCALE_K_L1_RATE = 2;
constexpr uint32_t MAX_SINGLE_MN = L1_TILE_M * L1_TILE_N;
constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM = (MAX_SINGLE_MN + 31U) / 32U * 32U;

// ---------- shared layouts/types factored out (same as v1 impl) ----------
template <typename ElementA, typename ElementB, typename ElementC>
struct GmmTypes {
    static constexpr bool transA = false;
    static constexpr bool transB = true;
    static constexpr uint32_t c0SizeA = AuxGetC0Size<ElementA>();
    static constexpr uint32_t c0SizeB = AuxGetC0Size<ElementB>();
    static constexpr uint32_t c0SizeC = AuxGetC0Size<ElementC>();
    static constexpr uint32_t c0SizeScale = 2U;

    using LayoutA = Std::conditional_t<transA, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutB = Std::conditional_t<transB, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutBias = Te::NDExtLayoutPtn;
    using LayoutC = Te::NDExtLayoutPtn;

    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<c0SizeA>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<c0SizeB>>;
    using MakeLayoutScaleA = Std::conditional_t<
        transA, Te::FrameLayoutFormat<Te::ScaleADNLayoutPtn, Std::Int<c0SizeScale>>,
        Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<c0SizeScale>>>;
    using MakeLayoutScaleB = Std::conditional_t<
        transB, Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<c0SizeScale>>,
        Te::FrameLayoutFormat<Te::ScaleBNDLayoutPtn, Std::Int<c0SizeScale>>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<c0SizeC>>;

    using BiasType = float;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, BiasType, LayoutBias, void>;
};

// ---------- AIC: spin until dispatchArrival[chunkId] >= targetVal ----------
__aicore__ inline void WaitDispatchChunk(__gm__ int32_t* dispatchArrivalGm, uint32_t chunkId, int32_t targetVal)
{
    __gm__ int32_t* flagAddr = dispatchArrivalGm + chunkId;
    while (ReadGmByPassDCache(flagAddr) < targetVal) {
        int64_t st = GetSystemCycle();
        while (GetSystemCycle() - st < 100) {}
    }
}

// ---------- AIC: spin until l2ArrivalMask[mBlockGlobal] has all numNBlocks bits set ----------
__aicore__ inline void WaitL2Mask(__gm__ uint64_t* l2MaskGm, uint32_t mBlockGlobal, uint32_t numNBlocks)
{
    uint64_t expected = (numNBlocks >= 64u) ? ~uint64_t(0) : ((uint64_t(1) << numNBlocks) - 1u);
    __gm__ uint64_t* flagAddr = l2MaskGm + mBlockGlobal;
    while ((ReadGmByPassDCache(reinterpret_cast<__gm__ int64_t*>(flagAddr)) & (int64_t)expected) != (int64_t)expected) {
        int64_t st = GetSystemCycle();
        while (GetSystemCycle() - st < 100) {}
    }
}

// ---------- One block of GMM1 (Linear1 phase) ----------
// AIC path: BlockMmad call + fixpipe to vec0 UB; AIV path: SwiGLU+MX-quant via epilogueOp.
template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB>
__aicore__ inline void DoLinear1Block(
    BlockEpilogueSwigluMxQuant<ElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true>& epilogueOp,
    const Params& params,
    const GMMAddrInfo& gmmAddrInfo,            // points at this expert's ring chunk (a ring + b weights)
    uint32_t expertM, uint32_t expertN, uint32_t expertK,
    uint32_t mLoc, uint32_t nLoc,              // current block coord (token offset, n offset within intermediate)
    uint32_t actualM, uint32_t actualN, uint32_t actualK,
    uint32_t chunkId, uint32_t mBlockGlobal,
    __gm__ int32_t* dispatchArrivalGm,
    __gm__ uint64_t* l2ArrivalMaskGm,
    int32_t targetDispatchCount,
    uint16_t pingpongSlot)                     // sync slot (0..WAVE_PINGPONG_DEPTH-1)
{
    using Types = GmmTypes<ElementA, ElementB, ElementC>;
    auto layoutA = typename Types::MakeLayoutA{}(expertM, expertK);
    auto layoutB = typename Types::MakeLayoutB{}(expertK, expertN);
    auto scaleK = Ops::Base::CeilDiv(expertK,
        static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    auto layoutScaleA = typename Types::MakeLayoutScaleA{}(expertM, scaleK);
    auto layoutScaleB = typename Types::MakeLayoutScaleB{}(scaleK, expertN);
    auto layoutBias = Te::MakeFrameLayout<typename Types::LayoutBias>(1u, expertN);
    auto layoutC = typename Types::MakeLayoutC{}(L1_TILE_M, L1_TILE_N);

    typename Types::BlockMmad blockMmad;
    typename Types::BlockMmad::L1Params l1Params{
        .kL1 = L1_TILE_K, .scaleKL1 = L1_TILE_K * SCALE_K_L1_RATE, .l1BufNum = 2};
    typename Types::BlockMmad::BlockShape l0TileShape{L1_TILE_M, L1_TILE_N, L0_TILE_K, 0};
    typename Types::BlockMmad::TupleShape matmulShape{expertM, expertN, expertK};
    blockMmad.Init(matmulShape, l0TileShape, l1Params, false, false);

    int64_t ubOffset = 0;
    auto l0cOutUbFirst = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutC);
    ubOffset += MAX_SINGLE_MN_ALIGN32_NUM * sizeof(ElementC);
    auto l0cOutUbSecond = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutC);

    auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
        reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal)), layoutA);
    auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
        reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal)), layoutB);
    auto gmScaleA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
        reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal)), layoutScaleA);
    auto gmScaleB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
        reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal)), layoutScaleB);
    auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
        reinterpret_cast<__gm__ typename Types::BiasType*>(0UL)), layoutBias);

    if constexpr (g_coreType == AscendC::AIC) {
        // 1. wait dispatch chunk arrived
        WaitDispatchChunk(dispatchArrivalGm, chunkId, targetDispatchCount);

        // 2. wait vec0 finished prev epilogue on this slot
        WaitForVector(pingpongSlot);

        auto gmBlockA = gmA.Slice(
            Te::MakeCoord(mLoc, 0u),
            Te::MakeShape(actualM, actualK));
        auto gmBlockScaleA = gmScaleA.Slice(
            Te::MakeCoord(mLoc, 0u / MXFP_SCALE_GROUP_NUM),
            Te::MakeShape(actualM, Ops::Base::CeilDiv(actualK, (uint32_t)MXFP_SCALE_GROUP_NUM)));

        auto tensorBlockUbFirst = l0cOutUbFirst.Slice(
            Te::MakeCoord(0u, 0u), Te::MakeShape(actualM, actualN));
        auto tensorBlockUbSecond = l0cOutUbSecond.Slice(
            Te::MakeCoord(0u, 0u), Te::MakeShape(actualM, actualN));

        typename Types::BlockMmad::BlockShape singleShape{actualM, actualN, actualK, 0};
        // SwiGLU has gate/up halves: weight tensor is laid out [intermediate] || [intermediate]
        for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
            auto gmBlockB = gmB.Slice(
                Te::MakeCoord(0u, nLoc + weightBlock * expertN),
                Te::MakeShape(actualK, actualN));
            auto gmBlockScaleB = gmScaleB.Slice(
                Te::MakeCoord(0u / MXFP_SCALE_GROUP_NUM, nLoc + weightBlock * expertN),
                Te::MakeShape(Ops::Base::CeilDiv(actualK, (uint32_t)MXFP_SCALE_GROUP_NUM), actualN));
            blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias,
                weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond, singleShape);
        }
        NotifyVector(pingpongSlot);
    }

    if constexpr (g_coreType == AscendC::AIV) {
        Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{actualM, actualN, 0, 0};
        Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
            mLoc * expertN + nLoc,
            mLoc * Ops::Base::CeilDiv(expertN, (uint32_t)MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                Ops::Base::CeilDiv(nLoc, (uint32_t)MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
            0, 0, 0, 0};
        WaitForCube(pingpongSlot);
        AscendC::SetCtrlSpr<60, 60>(0);
        epilogueOp(epilogueShape, epilogueOffset);
        NotifyCube(pingpongSlot);

        // signal Linear2: set bit nBlock in l2ArrivalMask[mBlockGlobal]
        // (n_loc is the absolute n offset within intermediate; convert to nBlock id)
        uint32_t nBlock = nLoc / L1_TILE_N;
        uint64_t bit = uint64_t(1) << nBlock;
        AtomicAdd(reinterpret_cast<__gm__ int64_t*>(l2ArrivalMaskGm + mBlockGlobal), (int64_t)bit);
    }
}

// ---------- One block of GMM2 (Linear2 phase) + combine push fanout ----------
template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB>
__aicore__ inline void DoLinear2Block(
    const Params& params,
    const GMMAddrInfo& gmmAddrInfo,
    uint32_t expertM, uint32_t expertN, uint32_t expertK,
    uint32_t mLoc, uint32_t nLoc,
    uint32_t actualM, uint32_t actualN, uint32_t actualK,
    uint32_t chunkId, uint32_t mBlockGlobal, uint32_t numNBlocksPrev,
    __gm__ uint64_t* l2ArrivalMaskGm,
    __gm__ int32_t* gmm2ToCombineGm,
    uint32_t expertIdx, uint32_t rankId,
    uint16_t pingpongSlot)
{
    using Types = GmmTypes<ElementA, ElementB, ElementC>;
    constexpr uint32_t GMM2_BLOCK_SIZE = L1_TILE_M * L1_TILE_N;

    auto layoutA = typename Types::MakeLayoutA{}(expertM, expertK);
    auto layoutB = typename Types::MakeLayoutB{}(expertK, expertN);
    auto scaleK = Ops::Base::CeilDiv(expertK,
        static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    auto layoutScaleA = typename Types::MakeLayoutScaleA{}(expertM, scaleK);
    auto layoutScaleB = typename Types::MakeLayoutScaleB{}(scaleK, expertN);
    auto layoutBias = Te::MakeFrameLayout<typename Types::LayoutBias>(1u, expertN);
    auto layoutC = typename Types::MakeLayoutC{}(L1_TILE_M, L1_TILE_N);

    typename Types::BlockMmad blockMmad;
    typename Types::BlockMmad::L1Params l1Params{
        .kL1 = L1_TILE_K, .scaleKL1 = L1_TILE_K * SCALE_K_L1_RATE, .l1BufNum = 2};
    typename Types::BlockMmad::BlockShape l0TileShape{L1_TILE_M, L1_TILE_N, L0_TILE_K, 0};
    typename Types::BlockMmad::TupleShape matmulShape{expertM, expertN, expertK};
    blockMmad.Init(matmulShape, l0TileShape, l1Params, false, false);

    int64_t ubOffset = 0;
    LocalTensor<ElementC> l0cOutUbGMM2First =
        LocalTensor<ElementC>(TPosition::VECIN, ubOffset, L1_TILE_M * L1_TILE_N);
    auto l0cOutUbFirst = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutC);
    ubOffset += MAX_SINGLE_MN_ALIGN32_NUM * sizeof(ElementC);
    LocalTensor<ElementC> l0cOutUbGMM2Second =
        LocalTensor<ElementC>(TPosition::VECIN, ubOffset, L1_TILE_M * L1_TILE_N);
    auto l0cOutUbSecond = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutC);

    auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
        reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal)), layoutA);
    auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
        reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal)), layoutB);
    auto gmScaleA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
        reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal)), layoutScaleA);
    auto gmScaleB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
        reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal)), layoutScaleB);
    auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
        reinterpret_cast<__gm__ typename Types::BiasType*>(0UL)), layoutBias);

    bool slotEven = ((pingpongSlot & 0x1u) == 0u);
    auto tensorUb = slotEven ? l0cOutUbFirst : l0cOutUbSecond;
    auto l0cOutUbGMM2 = slotEven ? l0cOutUbGMM2First : l0cOutUbGMM2Second;
    uint16_t cubeFlag = (uint16_t)GMM2_WAVE_FLAG_OFFSET + pingpongSlot;

    if constexpr (g_coreType == AscendC::AIC) {
        // 1. wait Linear1 for this m_block fully done (all n-blocks bit-set)
        WaitL2Mask(l2ArrivalMaskGm, mBlockGlobal, numNBlocksPrev);

        // 2. wait vec0 finished prev combine on this slot
        WaitForVector(cubeFlag);

        auto gmBlockA = gmA.Slice(
            Te::MakeCoord(mLoc, 0u), Te::MakeShape(actualM, actualK));
        auto gmBlockScaleA = gmScaleA.Slice(
            Te::MakeCoord(mLoc, 0u / MXFP_SCALE_GROUP_NUM),
            Te::MakeShape(actualM, Ops::Base::CeilDiv(actualK, (uint32_t)MXFP_SCALE_GROUP_NUM)));
        auto tensorBlockUb = tensorUb.Slice(
            Te::MakeCoord(0u, 0u), Te::MakeShape(actualM, actualN));
        auto gmBlockB = gmB.Slice(
            Te::MakeCoord(0u, nLoc), Te::MakeShape(actualK, actualN));
        auto gmBlockScaleB = gmScaleB.Slice(
            Te::MakeCoord(0u / MXFP_SCALE_GROUP_NUM, nLoc),
            Te::MakeShape(Ops::Base::CeilDiv(actualK, (uint32_t)MXFP_SCALE_GROUP_NUM), actualN));

        typename Types::BlockMmad::BlockShape singleShape{actualM, actualN, actualK, 0};
        blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, tensorBlockUb, singleShape);
        NotifyVector(cubeFlag);
    }

    if constexpr (g_coreType == AscendC::AIV) {
        WaitForCube(cubeFlag);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::Shape<int64_t, int64_t, int64_t, int64_t> actualShape{actualM, actualN, 0, 0};
        MegaMoeCombineImpl::CombineTokens<ElementC, decltype(actualShape)>(
            mLoc, nLoc, expertN, expertIdx, rankId, l0cOutUbGMM2, actualShape, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        NotifyCube(cubeFlag);

        // signal vec1 to push combine for this chunk
        AtomicAdd(gmm2ToCombineGm + chunkId, int32_t(1));
    }
}

} // namespace MegaMoeV2Impl

#endif // MEGA_MOE_V2_IMPL_H
