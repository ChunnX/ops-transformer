/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef SLIDING_TILE_ATTENTION_BLOCK_MMAD_FAI_PV_DYNAMIC_HPP
#define SLIDING_TILE_ATTENTION_BLOCK_MMAD_FAI_PV_DYNAMIC_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "../../../resource_ledger.hpp"

namespace SlidingTileAttentionCatlass {

template <class DispatchPolicy, class L1TileShape, class L0TileShape, class AType, class BType, class CType,
          class TileCopy = Catlass::Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, void>,
          class TileMmad = Catlass::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, void>>
struct BlockMmadFaiPvDynamic {
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementA = typename AType::Element;
    using LayoutA = typename AType::Layout;
    using ElementB = typename BType::Element;
    using LayoutB = typename BType::Layout;
    using ElementC = typename CType::Element;
    using LayoutC = typename CType::Layout;
    using CopyGmToL1A = typename TileCopy::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy::CopyL0CToGm;
    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = Catlass::layout::zN;
    using L1AAlignHelper = Catlass::Gemm::helper::L1AlignHelper<ElementA, LayoutA>;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / STAGES;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = ArchTag::L0C_SIZE / STAGES;
    static constexpr uint32_t EMBED_SPLIT_SIZE = 128;
    static constexpr uint32_t UNIT_BLOCK_STACK_NUM = 4;
    static constexpr uint32_t KV_SPLIT_SIZE = 128;
    static constexpr uint32_t LOAB_BLOCK = 1;
    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;

    static_assert(std::is_same_v<LayoutC, Catlass::layout::RowMajor>, "PV output must be row-major");
    static_assert(opforge::ascend_c::resource_ledger::ValidateL0ReuseEventPair(
        opforge::ascend_c::resource_ledger::L0Event::MTE1_M, opforge::ascend_c::resource_ledger::L0Event::M_MTE1));
    static_assert(opforge::ascend_c::resource_ledger::ValidateLayoutHandoff<LayoutBInL1, LayoutBInL1>(),
                  "packed V GM-to-L1 and L1-to-L0 layouts must match");

    CATLASS_DEVICE explicit BlockMmadFaiPvDynamic(Catlass::Arch::Resource<ArchTag> &resource,
                                                  uint32_t l1BufAddrStart = 0)
    {
        l1ATensor = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart);
        for (uint32_t i = 0; i < STAGES; ++i) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart +
                                                                             L1A_SIZE * STAGES * STAGES + L1B_SIZE * i);
        }
    }

    CATLASS_DEVICE void GetBlockShape(Catlass::GemmCoord &shape, uint32_t nIdx, uint32_t kIdx, uint32_t nLoop,
                                      uint32_t kLoop, uint32_t kvSeqlen, uint32_t embed) const
    {
        uint32_t nSplit = KV_SPLIT_SIZE;
        uint32_t embedSplit = EMBED_SPLIT_SIZE;
        if (nIdx == nLoop - 1) {
            nSplit = kvSeqlen - nIdx * KV_SPLIT_SIZE;
        }
        if (kIdx == kLoop - 1) {
            embedSplit = embed - kIdx * EMBED_SPLIT_SIZE;
        }
        shape[1] = embedSplit;
        shape[2] = nSplit;
    }

    CATLASS_DEVICE void GetKvOffset(uint32_t &offset, uint32_t nIdx, uint32_t kIdx, uint32_t nLoop, uint32_t kLoop,
                                    uint32_t strideKV) const
    {
        offset = nIdx >= nLoop || kIdx >= kLoop ? 0 : nIdx * KV_SPLIT_SIZE * strideKV + kIdx * EMBED_SPLIT_SIZE;
    }

    CATLASS_DEVICE void operator()(AscendC::GlobalTensor<ElementA> gA, AscendC::GlobalTensor<ElementB> gB,
                                   AscendC::GlobalTensor<ElementC> gC, LayoutA layoutA, LayoutB layoutB,
                                   Catlass::GemmCoord actualShape, uint32_t nIdx, uint32_t nLoop, uint32_t kvSeqlen,
                                   uint32_t strideKV, uint32_t outputStride, Catlass::Arch::CrossCoreFlag softmaxFlag)
    {
        uint32_t embed = actualShape.n();
        uint32_t kLoop = CeilDiv<L1TileShape::K>(embed);
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        Catlass::Arch::CrossCoreWaitFlag(softmaxFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(5);
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(mRound, 512U);
        auto aTile = layoutA.GetTileLayout(Catlass::MakeCoord(actualShape.m(), 512U));
        copyGmToL1A(l1ATensor, gA, layoutAInL1, aTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(4);
        uint32_t remainingN = nIdx < nLoop ? nLoop - nIdx : 0;
        uint32_t nBlockCount = remainingN < UNIT_BLOCK_STACK_NUM ? remainingN : UNIT_BLOCK_STACK_NUM;
        uint32_t nkBlockLoop = nBlockCount * kLoop;
        uint32_t nkBlockNextIdx = 1;
        Catlass::GemmCoord current{layoutA.shape(0), 0, 0};
        Catlass::GemmCoord next{layoutA.shape(0), 0, 0};
        for (uint32_t kIdx = 0; kIdx < kLoop; ++kIdx) {
            for (uint32_t blockStackIdx = 0; blockStackIdx < UNIT_BLOCK_STACK_NUM && nIdx + blockStackIdx < nLoop;
                 blockStackIdx += LOAB_BLOCK) {
                uint32_t nowNIdx = nIdx + blockStackIdx;
                bool hasNextN = blockStackIdx + LOAB_BLOCK < UNIT_BLOCK_STACK_NUM && nowNIdx + LOAB_BLOCK < nLoop;
                uint32_t nLoopNextIdx = hasNextN ? nowNIdx + LOAB_BLOCK : nIdx;
                uint32_t kLoopNextIdx = hasNextN ? kIdx : kIdx + 1;
                if (kLoopNextIdx >= kLoop) {
                    nLoopNextIdx = nLoop;
                }
                GetBlockShape(current, nowNIdx, kIdx, nLoop, kLoop, kvSeqlen, embed);
                GetBlockShape(next, nLoopNextIdx, kLoopNextIdx, nLoop, kLoop, kvSeqlen, embed);
                uint32_t bOffset = 0;
                uint32_t bNextOffset = 0;
                GetKvOffset(bOffset, nowNIdx, kIdx, nLoop, kLoop, strideKV);
                GetKvOffset(bNextOffset, nLoopNextIdx, kLoopNextIdx, nLoop, kLoop, strideKV);
                bool firstItr = blockStackIdx == 0;
                bool endItr = blockStackIdx + LOAB_BLOCK >= UNIT_BLOCK_STACK_NUM || nowNIdx + LOAB_BLOCK >= nLoop;
                Compute(gA, gB[bOffset], gC[kIdx * EMBED_SPLIT_SIZE], gB[bNextOffset], layoutA, layoutB, current, next,
                        blockStackIdx, nkBlockNextIdx, nkBlockLoop, firstItr, endItr, firstItr, outputStride);
                ++nkBlockNextIdx;
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(5);
    }

    CATLASS_DEVICE void ComputePackedD256(AscendC::GlobalTensor<ElementA> gA, AscendC::GlobalTensor<ElementB> gB,
                                          AscendC::GlobalTensor<ElementC> gC, LayoutA layoutA, LayoutB layoutB,
                                          Catlass::GemmCoord actualShape, uint32_t nLoop, uint32_t kvSeqlen,
                                          uint32_t outputStride, Catlass::Arch::CrossCoreFlag softmaxFlag,
                                          const uint64_t *blockOffsets)
    {
        uint32_t embed = actualShape.n();
        uint32_t kLoop = CeilDiv<L1TileShape::K>(embed);
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        Catlass::Arch::CrossCoreWaitFlag(softmaxFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(5);
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(mRound, 512U);
        auto aTile = layoutA.GetTileLayout(Catlass::MakeCoord(actualShape.m(), 512U));
        copyGmToL1A(l1ATensor, gA, layoutAInL1, aTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(4);
        uint32_t nBlockCount = nLoop < UNIT_BLOCK_STACK_NUM ? nLoop : UNIT_BLOCK_STACK_NUM;
        uint32_t nkBlockLoop = nBlockCount * kLoop;
        uint32_t nkBlockNextIdx = 1;
        Catlass::GemmCoord current{layoutA.shape(0), 0, 0};
        Catlass::GemmCoord next{layoutA.shape(0), 0, 0};
        for (uint32_t kIdx = 0; kIdx < kLoop; ++kIdx) {
            for (uint32_t blockStackIdx = 0; blockStackIdx < UNIT_BLOCK_STACK_NUM && blockStackIdx < nLoop;
                 blockStackIdx += LOAB_BLOCK) {
                uint32_t nowNIdx = blockStackIdx;
                bool hasNextN = blockStackIdx + LOAB_BLOCK < UNIT_BLOCK_STACK_NUM && nowNIdx + LOAB_BLOCK < nLoop;
                uint32_t nLoopNextIdx = hasNextN ? nowNIdx + LOAB_BLOCK : 0;
                uint32_t kLoopNextIdx = hasNextN ? kIdx : kIdx + 1;
                if (kLoopNextIdx >= kLoop) {
                    nLoopNextIdx = nLoop;
                }
                GetBlockShape(current, nowNIdx, kIdx, nLoop, kLoop, kvSeqlen, embed);
                GetBlockShape(next, nLoopNextIdx, kLoopNextIdx, nLoop, kLoop, kvSeqlen, embed);
                uint64_t bOffset = blockOffsets[nowNIdx] + static_cast<uint64_t>(kIdx) * EMBED_SPLIT_SIZE;
                uint64_t bNextOffset =
                    nLoopNextIdx >= nLoop ?
                        0 :
                        blockOffsets[nLoopNextIdx] + static_cast<uint64_t>(kLoopNextIdx) * EMBED_SPLIT_SIZE;
                bool firstItr = blockStackIdx == 0;
                bool endItr = blockStackIdx + LOAB_BLOCK >= UNIT_BLOCK_STACK_NUM || nowNIdx + LOAB_BLOCK >= nLoop;
                Compute(gA, gB[bOffset], gC[kIdx * EMBED_SPLIT_SIZE], gB[bNextOffset], layoutA, layoutB, current, next,
                        blockStackIdx, nkBlockNextIdx, nkBlockLoop, firstItr, endItr, firstItr, outputStride);
                ++nkBlockNextIdx;
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(5);
    }

    CATLASS_DEVICE void FullD128(AscendC::GlobalTensor<ElementA> gA, AscendC::GlobalTensor<ElementB> gB,
                                 AscendC::GlobalTensor<ElementC> gC, LayoutA layoutA, LayoutB layoutB, uint32_t nIdx,
                                 uint32_t strideKV, uint32_t outputStride, Catlass::Arch::CrossCoreFlag softmaxFlag)
    {
        constexpr uint32_t FULL_BLOCKS = 4;
        constexpr uint32_t FIXED_DIM = 128;
        static_assert(L1TileShape::K == FIXED_DIM, "fixed D128 PV requires one 128-column output slice");
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(layoutA.shape(0));
        Catlass::Arch::CrossCoreWaitFlag(softmaxFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(5);
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(mRound, 512U);
        auto aTile = layoutA.GetTileLayout(Catlass::MakeCoord(layoutA.shape(0), 512U));
        copyGmToL1A(l1ATensor, gA, layoutAInL1, aTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(4);

        ComputeFullD128Block<0, FULL_BLOCKS>(gB, gC, layoutA, layoutB, nIdx, strideKV, outputStride);
        ComputeFullD128Block<1, FULL_BLOCKS>(gB, gC, layoutA, layoutB, nIdx, strideKV, outputStride);
        ComputeFullD128Block<2, FULL_BLOCKS>(gB, gC, layoutA, layoutB, nIdx, strideKV, outputStride);
        ComputeFullD128Block<3, FULL_BLOCKS>(gB, gC, layoutA, layoutB, nIdx, strideKV, outputStride);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(5);
    }

    template <uint32_t BLOCK_IDX, uint32_t FULL_BLOCKS>
    CATLASS_DEVICE void ComputeFullD128Block(AscendC::GlobalTensor<ElementB> const &gB,
                                             AscendC::GlobalTensor<ElementC> const &gC, LayoutA layoutA,
                                             LayoutB layoutB, uint32_t nIdx, uint32_t strideKV, uint32_t outputStride)
    {
        constexpr uint32_t FIXED_DIM = 128;
        constexpr uint32_t NK_BLOCK_IDX = BLOCK_IDX + 1;
        constexpr uint32_t L1_FLAG = NK_BLOCK_IDX % 2;
        constexpr uint32_t L0_FLAG = NK_BLOCK_IDX % 2;
        constexpr bool FIRST = BLOCK_IDX == 0;
        constexpr bool LAST = BLOCK_IDX + 1 == FULL_BLOCKS;
        uint32_t mActual = layoutA.shape(0);
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(mActual);
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(mRound, 512U);
        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mRound, FIXED_DIM);
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(FIXED_DIM, FIXED_DIM);
        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(FIXED_DIM, FIXED_DIM);
        uint32_t bOffset = (nIdx + BLOCK_IDX) * KV_SPLIT_SIZE * strideKV;

        if constexpr (FIRST) {
            auto bTile = layoutB.GetTileLayout(Catlass::MakeCoord(FIXED_DIM, FIXED_DIM));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_FLAG + 2);
            copyGmToL1B(l1BTensor[L1_FLAG], gB[bOffset], layoutBInL1, bTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1_FLAG + 2);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1_FLAG + 2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_FLAG + 2);
        copyL1ToL0B(l0BTensor[L0_FLAG], l1BTensor[L1_FLAG], layoutBInL0, layoutBInL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_FLAG + 2);
        if constexpr (!LAST) {
            constexpr uint32_t NEXT_L1_FLAG = 1 - L1_FLAG;
            uint32_t bNextOffset = (nIdx + BLOCK_IDX + 1) * KV_SPLIT_SIZE * strideKV;
            auto nextBTile = layoutB.GetTileLayout(Catlass::MakeCoord(FIXED_DIM, FIXED_DIM));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(NEXT_L1_FLAG + 2);
            copyGmToL1B(l1BTensor[NEXT_L1_FLAG], gB[bNextOffset], layoutBInL1, nextBTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(NEXT_L1_FLAG + 2);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_FLAG);
        copyL1ToL0A(l0ATensor[L0_FLAG], l1ATensor[mRound * FIXED_DIM * BLOCK_IDX], layoutAInL0, layoutAInL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0_FLAG);
        if constexpr (!ENABLE_UNIT_FLAG && FIRST) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
        }
        uint8_t unitFlag = 0;
        if constexpr (ENABLE_UNIT_FLAG) {
            unitFlag = LAST ? 0b11 : 0b10;
        }
        tileMmad(l0CTensor[0], l0ATensor[L0_FLAG], l0BTensor[L0_FLAG], mRound, FIXED_DIM, FIXED_DIM, FIRST, unitFlag);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_FLAG + 2);
        if constexpr (LAST) {
            auto l0Layout = LayoutCInL0::MakeLayoutInL0C(Catlass::MakeCoord(mActual, FIXED_DIM));
            LayoutC outputLayout(mActual, FIXED_DIM, outputStride);
            if constexpr (ENABLE_UNIT_FLAG) {
                copyL0CToGm(gC, l0CTensor[0], outputLayout, l0Layout, 0b11);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
                copyL0CToGm(gC, l0CTensor[0], outputLayout, l0Layout);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
            }
        }
    }

    CATLASS_DEVICE void Compute(AscendC::GlobalTensor<ElementA> const &gA, AscendC::GlobalTensor<ElementB> const &gB,
                                AscendC::GlobalTensor<ElementC> const &gC,
                                AscendC::GlobalTensor<ElementB> const &gNextB, LayoutA layoutA, LayoutB layoutB,
                                Catlass::GemmCoord actualShape, Catlass::GemmCoord nextShape, uint32_t nowIdx,
                                uint32_t &nkBlockIdx, uint32_t nkBlockLoop, bool firstItr, bool endItr, bool initMmad,
                                uint32_t outputStride)
    {
        uint32_t mActual = actualShape.m();
        uint32_t kActual = actualShape.k();
        uint32_t nActual = actualShape.n();
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(mActual);
        uint32_t kRound = RoundUp<L1AAlignHelper::M_ALIGNED>(kActual);
        uint32_t nRound = RoundUp<L1AAlignHelper::M_ALIGNED>(nActual);
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(mRound, 512U);
        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mRound, kActual);
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(kActual, nActual);
        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kActual, nActual);
        uint32_t l1Flag = nkBlockIdx % 2;
        uint32_t l0Flag = nkBlockIdx % 2;
        if (nkBlockIdx == 1) {
            auto bTile = layoutB.GetTileLayout(Catlass::MakeCoord(kActual, nActual));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1Flag + 2);
            copyGmToL1B(l1BTensor[l1Flag], gB, layoutBInL1, bTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1Flag + 2);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1Flag + 2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0Flag + 2);
        copyL1ToL0B(l0BTensor[l0Flag], l1BTensor[l1Flag], layoutBInL0, layoutBInL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1Flag + 2);
        if (nkBlockIdx != nkBlockLoop) {
            uint32_t nNext = nextShape.n();
            uint32_t kNext = nextShape.k();
            LayoutBInL1 nextBInL1 = LayoutBInL1::template MakeLayout<ElementB>(kNext, nNext);
            auto nextBTile = layoutB.GetTileLayout(Catlass::MakeCoord(kNext, nNext));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1 - l1Flag + 2);
            copyGmToL1B(l1BTensor[1 - l1Flag], gNextB, nextBInL1, nextBTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(1 - l1Flag + 2);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0Flag);
        copyL1ToL0A(l0ATensor[l0Flag], l1ATensor[mRound * 128 * nowIdx], layoutAInL0, layoutAInL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0Flag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0Flag);
        if constexpr (!ENABLE_UNIT_FLAG) {
            if (firstItr) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
            }
        }
        uint8_t unitFlag = 0;
        if constexpr (ENABLE_UNIT_FLAG) {
            unitFlag = endItr ? 0b11 : 0b10;
        }
        tileMmad(l0CTensor[0], l0ATensor[l0Flag], l0BTensor[l0Flag], mRound, nActual, kActual, initMmad, unitFlag);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0Flag);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0Flag + 2);
        if (endItr) {
            auto l0Layout = LayoutCInL0::MakeLayoutInL0C(Catlass::MakeCoord(mActual, nActual));
            uint32_t fixpipeN = RoundUp<16>(nActual);
            LayoutC outputLayout(mActual, fixpipeN, outputStride);
            if constexpr (ENABLE_UNIT_FLAG) {
                copyL0CToGm(gC, l0CTensor[0], outputLayout, l0Layout, 0b11);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
                copyL0CToGm(gC, l0CTensor[0], outputLayout, l0Layout);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
            }
        }
    }

    AscendC::LocalTensor<ElementA> l1ATensor;
    AscendC::LocalTensor<ElementB> l1BTensor[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[STAGES];
    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};

} // namespace SlidingTileAttentionCatlass

#endif
