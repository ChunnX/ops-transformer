/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_BLOCK_MMAD_BSAG_2_HPP
#define GEMM_BLOCK_MMAD_BSAG_2_HPP

#include "../../../attn_infra/bsag_base_defs.hpp"
#include "../../../attn_infra/arch/bsag_resource.hpp"
#include "../../../attn_infra/bsag_coord.hpp"
#include "../../../attn_infra/gemm/bsag_gemm_dispatch_policy.hpp"
#include "../../../attn_infra/gemm/bsag_helper.hpp"
#include "../../../attn_infra/bsag_gemm_coord.hpp"
#include "../../../attn_infra/gemm/tile_common/bsag_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/bsag_tile_mmad.hpp"

namespace NpuArch::Gemm::Block {

template <class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_, class BiasType_,
          class TileCopy_, class TileMmad_>
struct BlockMmad<MmadAtlasA2SBSAG2, L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_, TileCopy_,
                 TileMmad_> {
public:
    using DispatchPolicy = MmadAtlasA2SBSAG2;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;

    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;

    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using CopyL0CToGmNz = Gemm::Tile::CopyL0CToGm<
        ArchTag, typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator,
        Gemm::GemmType<ElementC, layout::zN>>;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    // Keep this empty copy helper in the verified device-object layout.  Some
    // AscendC helper types are empty, but removing the member can still move
    // the following helper and change generated device code.
    using CopyL0CToGmInput =
        Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, Gemm::GemmType<ElementA, layout::RowMajor>>;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t EXTENDED_CACHE_SLOTS = 8;
    static constexpr uint32_t EXTENDED_B_SLOTS = 4;

    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);

    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;

    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_SIZE / STAGES;

    static constexpr uint32_t BLOCK_SIZE = 16;

    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    __aicore__ inline BlockMmad(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0,
                                uint32_t pingpongFlagOffset = 0, bool isAtomicAdd_ = false, bool fullL1Cache_ = false,
                                bool splitB8Cache_ = false, bool pairedA4Cache_ = false, uint32_t residentCacheMode = 0)
    {
        isAtomicAdd = isAtomicAdd_;
        PINGPONG_FLAG_OFFSET = pingpongFlagOffset;
        for (uint32_t i = 0; i < STAGES; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
        }
        if (residentCacheMode != 0) {
            for (uint32_t i = 0; i < STAGES; ++i) {
                l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart + L1A_SIZE * i);
            }
            if (residentCacheMode == 1) {
                // Low 64 KiB holds P/dS. Eight K slots occupy [64,320) KiB.
                for (uint32_t i = 0; i < EXTENDED_CACHE_SLOTS; ++i) {
                    l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(L1A_SIZE * 2 + L1B_SIZE * i);
                }
            } else {
                // Three persistent (dO,Q) pairs occupy [320,512) KiB.
                for (uint32_t i = 0; i < 3; ++i) {
                    l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(L1B_SIZE * (10 + i));
                    l1BTensor[i + 4] = resource.l1Buf.template GetBufferByByte<ElementB>(L1B_SIZE * (13 + i));
                }
            }
        } else if (pairedA4Cache_) {
            for (uint32_t i = 0; i < 4; ++i) {
                l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart + L1A_SIZE * i);
            }
            for (uint32_t i = 0; i < STAGES; ++i) {
                l1BTensor[i] =
                    resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1A_SIZE * 4 + L1B_SIZE * i);
            }
        } else if (fullL1Cache_) {
            for (uint32_t i = 0; i < EXTENDED_CACHE_SLOTS; ++i) {
                l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(L1A_SIZE * i);
                l1BTensor[i] =
                    resource.l1Buf.template GetBufferByByte<ElementB>(L1A_SIZE * EXTENDED_CACHE_SLOTS + L1B_SIZE * i);
            }
        } else if (splitB8Cache_) {
            for (uint32_t i = 0; i < STAGES; ++i) {
                l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart + L1A_SIZE * i);
            }
            // The grouped long-sequence path compacts Cube1 and dQ into the
            // low 192 KiB.  Reserve the following eight blocks for persistent
            // dOut/Q rows; the regular A ping-pong uses the final 64 KiB.
            const uint32_t persistentBBase = L1A_SIZE * 6;
            for (uint32_t i = 0; i < EXTENDED_CACHE_SLOTS; ++i) {
                l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(persistentBBase + L1B_SIZE * i);
            }
        } else {
            for (uint32_t i = 0; i < STAGES; ++i) {
                l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart + L1A_SIZE * i);
            }
            for (uint32_t i = 0; i < EXTENDED_B_SLOTS; ++i) {
                l1BTensor[i] =
                    resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1A_SIZE * 2 + L1B_SIZE * i);
            }
        }
    }

    __aicore__ inline ~BlockMmad() {}

    // Cache an operand that is invariant across all KV tiles of one Q task.
    // The cached tensor stays in its dedicated L1 A/B region; the regular
    // ping-pong slot is still used for L0 and the tile-varying operand.
    __aicore__ inline void PreloadA(AscendC::GlobalTensor<ElementA> gA, LayoutA layoutA, GemmCoord actualShape,
                                    uint32_t cacheSlot)
    {
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(cacheSlot + PINGPONG_FLAG_OFFSET);
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), actualShape.k()));
        copyGmToL1A(l1ATensor[cacheSlot], gA, layoutAInL1, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(cacheSlot + PINGPONG_FLAG_OFFSET);
    }

    __aicore__ inline void PreloadB(AscendC::GlobalTensor<ElementB> gB, LayoutB layoutB, GemmCoord actualShape,
                                    uint32_t cacheSlot, bool holdForConsumer = false)
    {
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(cacheSlot + PINGPONG_FLAG_OFFSET);
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(actualShape.k(), actualShape.n()));
        copyGmToL1B(l1BTensor[cacheSlot], gB, layoutBInL1, layoutTileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        if (!holdForConsumer) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(cacheSlot + PINGPONG_FLAG_OFFSET);
        }
    }

    // Assemble non-contiguous sparse fragments into one full-width B tile.
    // AlongN is used by Q*K^T (fragments are output columns); AlongK is used
    // by dS*K (fragments are reduction rows).  Both retain the native source
    // stride while using the combined L1 layout for destination addressing.
    __aicore__ inline void PreloadGatheredBAlongN(AscendC::GlobalTensor<ElementB> first, LayoutB firstLayout,
                                                  AscendC::GlobalTensor<ElementB> second, LayoutB secondLayout,
                                                  AscendC::GlobalTensor<ElementB> padding, LayoutB paddingLayout,
                                                  uint32_t firstSize, uint32_t secondSize, GemmCoord actualShape,
                                                  uint32_t cacheSlot)
    {
        LayoutBInL1 combinedLayout = LayoutBInL1::template MakeLayout<ElementB>(actualShape.k(), actualShape.n());
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(cacheSlot + PINGPONG_FLAG_OFFSET);
        copyGmToL1B(l1BTensor[cacheSlot], first, combinedLayout, firstLayout);
        if (secondSize != 0) {
            MatrixCoord secondCoord{0, firstSize};
            auto secondDst = l1BTensor[cacheSlot][combinedLayout.GetOffset(secondCoord)];
            copyGmToL1B(secondDst, second, combinedLayout, secondLayout);
        }
        if (firstSize + secondSize < actualShape.n()) {
            MatrixCoord paddingCoord{0, firstSize + secondSize};
            auto paddingDst = l1BTensor[cacheSlot][combinedLayout.GetOffset(paddingCoord)];
            copyGmToL1B(paddingDst, padding, combinedLayout, paddingLayout);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(cacheSlot + PINGPONG_FLAG_OFFSET);
    }

    __aicore__ inline void PreloadGatheredBAlongK(AscendC::GlobalTensor<ElementB> first, LayoutB firstLayout,
                                                  AscendC::GlobalTensor<ElementB> second, LayoutB secondLayout,
                                                  AscendC::GlobalTensor<ElementB> padding, LayoutB paddingLayout,
                                                  uint32_t firstSize, uint32_t secondSize, GemmCoord actualShape,
                                                  uint32_t cacheSlot)
    {
        LayoutBInL1 combinedLayout = LayoutBInL1::template MakeLayout<ElementB>(actualShape.k(), actualShape.n());
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(cacheSlot + PINGPONG_FLAG_OFFSET);
        copyGmToL1B(l1BTensor[cacheSlot], first, combinedLayout, firstLayout);
        if (secondSize != 0) {
            MatrixCoord secondCoord{firstSize, 0};
            auto secondDst = l1BTensor[cacheSlot][combinedLayout.GetOffset(secondCoord)];
            copyGmToL1B(secondDst, second, combinedLayout, secondLayout);
        }
        if (firstSize + secondSize < actualShape.k()) {
            MatrixCoord paddingCoord{firstSize + secondSize, 0};
            auto paddingDst = l1BTensor[cacheSlot][combinedLayout.GetOffset(paddingCoord)];
            copyGmToL1B(paddingDst, padding, combinedLayout, paddingLayout);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(cacheSlot + PINGPONG_FLAG_OFFSET);
    }

    __aicore__ inline void PreloadFullB(AscendC::GlobalTensor<ElementB> gB, LayoutB layoutB, GemmCoord actualShape,
                                        uint32_t cacheSlot)
    {
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(actualShape.k(), actualShape.n()));
        copyGmToL1B(l1BTensor[cacheSlot], gB, layoutBInL1, layoutTileB);
    }

    __aicore__ inline void FinishFullPreload()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    __aicore__ inline AscendC::LocalTensor<ElementB> GetCachedB(uint32_t slot)
    {
        return l1BTensor[slot];
    }

    // External-A users do not call AccumulatePreloadedA, so they do not
    // consume/reseed this object's regular A/B reuse events automatically.
    // Acquire the two tokens before replacing a persistent B cache; the last
    // external consumer in the batch publishes the corresponding tokens.
    __aicore__ inline void AcquireFullPreloadReuse()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(PINGPONG_FLAG_OFFSET);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(PINGPONG_FLAG_OFFSET + 1);
    }

    // Publish ownership of a persistent-B region after an external phase has
    // finished reading it.  Both flags are issued by MTE1, so a following
    // MTE2 acquire cannot overwrite the region before all preceding L1->L0B
    // transfers have retired.
    __aicore__ inline void ReleaseFullPreloadReuse()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(PINGPONG_FLAG_OFFSET);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(PINGPONG_FLAG_OFFSET + 1);
    }

    __aicore__ inline void WithCachedA(uint32_t cacheSlot, AscendC::GlobalTensor<ElementB> gB,
                                       AscendC::GlobalTensor<ElementC> gC, LayoutA layoutA, LayoutB layoutB,
                                       LayoutC layoutC, GemmCoord actualShape, uint32_t &pingpongFlag)
    {
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);

        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());
        uint32_t kRound = RoundUp<L1BAlignHelper::K_ALIGNED>(actualShape.k());

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlag + PINGPONG_FLAG_OFFSET);
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(actualShape.k(), actualShape.n()));
        copyGmToL1B(l1BTensor[pingpongFlag], gB, layoutBInL1, layoutTileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(pingpongFlag);
        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kRound, nRound);
        copyL1ToL0B(l0BTensor[pingpongFlag], l1BTensor[pingpongFlag], layoutBInL0, layoutBInL1);
        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mRound, kRound);
        copyL1ToL0A(l0ATensor[pingpongFlag], l1ATensor[cacheSlot], layoutAInL0, layoutAInL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlag + PINGPONG_FLAG_OFFSET);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        const uint32_t scratchCSlot = pingpongFlag;
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(scratchCSlot);
        tileMmad(l0CTensor[scratchCSlot], l0ATensor[pingpongFlag], l0BTensor[pingpongFlag], mRound, nRound,
                 actualShape.k());
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(pingpongFlag);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);

        auto blockShape = MakeCoord(actualShape.m(), actualShape.n());
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(blockShape);
        if (isAtomicAdd) {
            AscendC::SetAtomicAdd<ElementC>();
            copyL0CToGm(gC, l0CTensor[scratchCSlot], layoutC, layoutInL0C);
            AscendC::SetAtomicNone();
        } else {
            copyL0CToGm(gC, l0CTensor[scratchCSlot], layoutC, layoutInL0C);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(scratchCSlot);
        pingpongFlag = 1 - pingpongFlag;
    }

    __aicore__ inline void WithCachedB(uint32_t cacheSlot, AscendC::GlobalTensor<ElementA> gA,
                                       AscendC::GlobalTensor<ElementC> gC, LayoutA layoutA, LayoutB layoutB,
                                       LayoutC layoutC, GemmCoord actualShape, uint32_t &pingpongFlag)
    {
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);

        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());
        uint32_t kRound = RoundUp<L1BAlignHelper::K_ALIGNED>(actualShape.k());

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlag + PINGPONG_FLAG_OFFSET);
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), actualShape.k()));
        copyGmToL1A(l1ATensor[pingpongFlag], gA, layoutAInL1, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(pingpongFlag);
        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kRound, nRound);
        copyL1ToL0B(l0BTensor[pingpongFlag], l1BTensor[cacheSlot], layoutBInL0, layoutBInL1);
        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mRound, kRound);
        copyL1ToL0A(l0ATensor[pingpongFlag], l1ATensor[pingpongFlag], layoutAInL0, layoutAInL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlag + PINGPONG_FLAG_OFFSET);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        const uint32_t scratchCSlot = pingpongFlag;
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(scratchCSlot);
        tileMmad(l0CTensor[scratchCSlot], l0ATensor[pingpongFlag], l0BTensor[pingpongFlag], mRound, nRound,
                 actualShape.k(), true, 3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(pingpongFlag);

        auto blockShape = MakeCoord(actualShape.m(), actualShape.n());
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(blockShape);
        if (isAtomicAdd) {
            AscendC::SetAtomicAdd<ElementC>();
            copyL0CToGm(gC, l0CTensor[scratchCSlot], layoutC, layoutInL0C, 3);
            AscendC::SetAtomicNone();
        } else {
            copyL0CToGm(gC, l0CTensor[scratchCSlot], layoutC, layoutInL0C, 3);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(scratchCSlot);
        pingpongFlag = 1 - pingpongFlag;
    }

    // Keep Cube1's native NZ order for the wide vector epilogue, avoiding
    // a Fixpipe row-major reorder. Round M/N to the hardware fractal while
    // preserving the logical extents in actualShape.
    __aicore__ inline void WithCachedBNz(uint32_t cacheSlot, AscendC::GlobalTensor<ElementA> gA,
                                         AscendC::GlobalTensor<ElementC> gC, LayoutA layoutA, LayoutB layoutB,
                                         GemmCoord actualShape, uint32_t &pingpongFlag,
                                         const AscendC::LocalTensor<ElementA> *externalA = nullptr,
                                         bool releaseB = false)
    {
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);

        const uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        const uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());
        const uint32_t kRound = RoundUp<L1BAlignHelper::K_ALIGNED>(actualShape.k());

        const bool resident = externalA != nullptr && fastCachedInputs;
        if (!resident) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlag + PINGPONG_FLAG_OFFSET);
            auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), actualShape.k()));
            if (externalA == nullptr) {
                copyGmToL1A(l1ATensor[pingpongFlag], gA, layoutAInL1, layoutTileA);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        }

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(pingpongFlag);
        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kRound, nRound);
        copyL1ToL0B(l0BTensor[pingpongFlag], l1BTensor[cacheSlot], layoutBInL0, layoutBInL1);
        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mRound, kRound);
        copyL1ToL0A(l0ATensor[pingpongFlag], externalA != nullptr ? *externalA : l1ATensor[pingpongFlag], layoutAInL0,
                    layoutAInL1);
        if (!resident || releaseB) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((resident ? cacheSlot : pingpongFlag) +
                                                            PINGPONG_FLAG_OFFSET);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        const uint32_t scratchCSlot = pingpongFlag;
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(scratchCSlot);
        tileMmad(l0CTensor[scratchCSlot], l0ATensor[pingpongFlag], l0BTensor[pingpongFlag], mRound, nRound,
                 actualShape.k(), true, 3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(pingpongFlag);

        const auto blockShape = MakeCoord(actualShape.m(), actualShape.n());
        const auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(blockShape);
        // Fixpipe's FP32 NZ ABI keeps a 16-element C0 (64 bytes), whereas
        // the generic layout factory derives an 8-element C0 from the
        // 32-byte DMA block.  Describe the hardware ABI explicitly here.
        // This also makes dstStride == mRound * 2 in 32-byte units.
        const auto layoutNz =
            layout::zN(actualShape.m(), actualShape.n(), 16, mRound / 16, 16, RoundUp<16>(actualShape.n()) / 16, 16,
                       256, 1, static_cast<int64_t>(mRound) * 16);
        copyL0CToGmNz(gC, l0CTensor[scratchCSlot], layoutNz, layoutInL0C, 3);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(scratchCSlot);
        pingpongFlag = 1 - pingpongFlag;
    }

    // Accumulate all KV contributions for one dQ tile in L0C slot 0.  Each
    // Q task has one AIC owner, so the final GM store does not need atomics.
    __aicore__ inline void Accumulate(AscendC::GlobalTensor<ElementA> gA, AscendC::GlobalTensor<ElementB> gB,
                                      LayoutA layoutA, LayoutB layoutB, GemmCoord actualShape, uint32_t &pingpongFlag,
                                      bool initC, bool finalC, uint32_t accumulatorSlot = 0)
    {
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());
        uint32_t kRound = RoundUp<L1BAlignHelper::K_ALIGNED>(actualShape.k());

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlag + PINGPONG_FLAG_OFFSET);
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(actualShape.k(), actualShape.n()));
        copyGmToL1B(l1BTensor[pingpongFlag], gB, layoutBInL1, layoutTileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(pingpongFlag);
        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kRound, nRound);
        copyL1ToL0B(l0BTensor[pingpongFlag], l1BTensor[pingpongFlag], layoutBInL0, layoutBInL1);

        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), actualShape.k()));
        copyGmToL1A(l1ATensor[pingpongFlag], gA, layoutAInL1, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mRound, kRound);
        copyL1ToL0A(l0ATensor[pingpongFlag], l1ATensor[pingpongFlag], layoutAInL0, layoutAInL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlag + PINGPONG_FLAG_OFFSET);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        if (initC) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(accumulatorSlot);
        }
        tileMmad(l0CTensor[accumulatorSlot], l0ATensor[pingpongFlag], l0BTensor[pingpongFlag], mRound, nRound,
                 actualShape.k(), initC, finalC ? 3 : 2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(pingpongFlag);
        pingpongFlag = 1 - pingpongFlag;
    }

    // Split the accumulation input path into producer/consumer steps so the
    // next GM->L1 transfer can run while Cube consumes the current tile.
    __aicore__ inline void PreloadAccumA(AscendC::GlobalTensor<ElementA> gA, LayoutA layoutA, GemmCoord actualShape,
                                         uint32_t slot)
    {
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(slot + PINGPONG_FLAG_OFFSET);
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), actualShape.k()));
        copyGmToL1A(l1ATensor[slot], gA, layoutAInL1, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
    }

    // Directly copy a compact GM zN tile into this class' fixed-stride L1 zN
    // slot.  This is the native workspace path for wide P/dS and avoids the
    // ND2NZ conversion performed by the generic RowMajor loader.  The source
    // and destination layouts may have different rounded Q strides, so the
    // existing zN->zN copy handles arbitrary Q tails slab by slab.
    __aicore__ inline void PreloadAccumANz(AscendC::GlobalTensor<ElementA> gA, GemmCoord actualShape, uint32_t slot)
    {
        using NzType = Gemm::GemmType<ElementA, layout::zN>;
        Gemm::Tile::CopyGmToL1<ArchTag, NzType> copyGmNzToL1;
        const auto sourceLayout = layout::zN::template MakeLayout<ElementA>(actualShape.m(), actualShape.k());
        const auto destinationLayout = layout::zN::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(slot + PINGPONG_FLAG_OFFSET);
        copyGmNzToL1(l1ATensor[slot], gA, destinationLayout, sourceLayout);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
    }

    // Bridge a direct GM->L1 preload to AccumulateTransposeFromExternalZNL1.
    // The latter owns the actual L1 read and returns the reuse token only
    // after that read has reached L0.
    __aicore__ inline void FinishAccumPreloadForExternal(uint32_t slot)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(slot + PINGPONG_FLAG_OFFSET);
    }

    // Expose the transient A1 tile to a serialized companion MMAD phase.
    // The caller must consume the MTE1->MTE2 reuse token before reading this
    // tensor and restore it after the L1 read has completed.
    __aicore__ inline AscendC::LocalTensor<ElementA> GetAccumL1A(uint32_t slot) const
    {
        return l1ATensor[slot];
    }

    // Cube23-style transpose update from a zN [Q,KV] tile.  dS/P is
    // transposed only on L1->L0A, avoiding a second layout conversion.  The
    // accumulator slot is chosen by the caller, so dV and dK can stay in the
    // two L0C banks.  This works for any real q/kv tail within the 128x128
    // internal compute tile and does not specialize a user block shape.
    __aicore__ inline void AccumulateTransposeFromExternalZNL1(AscendC::LocalTensor<ElementA> externalA,
                                                               uint32_t externalReuseEvent, uint32_t cacheBSlot,
                                                               LayoutB layoutB, GemmCoord actualShape,
                                                               uint32_t accumulatorSlot, bool initC = true,
                                                               bool finalC = true, bool releaseLocalCache = false)
    {
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        const uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        const uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());
        const uint32_t kRound = RoundUp<L1AAlignHelper::K_ALIGNED>(actualShape.k());

        // AccumulatePreloadedA publishes this token after its zN->L0A read.
        // Consume it here so the next GM preload cannot overwrite dS until
        // the transposed L1 read below has retired.
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(externalReuseEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(accumulatorSlot);

        // dQ stores zN with the class' fixed 128-row L1 stride, whereas the
        // dK zZ destination is compact in the actual rounded q size.  Convert
        // one KV C0 fractal at a time so arbitrary q tails skip the unused L1
        // row fractals instead of interpreting them as valid dS.
        const uint32_t qC0 = kRound / C0_NUM_PER_FRACTAL;
        const uint32_t sourceFractalStride = L1TileShape::M * C0_NUM_PER_FRACTAL;
        const uint32_t destinationFractalStride = kRound * C0_NUM_PER_FRACTAL;
        const uint32_t mC0 = mRound / C0_NUM_PER_FRACTAL;
        if (fastCachedInputs && kRound == L1TileShape::M) {
            // Full Q tiles have identical source/destination slab strides;
            // one repeated-fractal transpose replaces eight load commands.
            AscendC::LoadData2dParams transposeParams{0, static_cast<uint8_t>(qC0 * mC0), 1, 0, 0, true, 0};
            AscendC::LoadData(l0ATensor[accumulatorSlot], externalA, transposeParams);
        } else {
            for (uint32_t mFractal = 0; mFractal < mC0; ++mFractal) {
                AscendC::LoadData2dParams transposeParams{0, static_cast<uint8_t>(qC0), 1, 0, 0, true, 0};
                AscendC::LoadData(l0ATensor[accumulatorSlot][mFractal * destinationFractalStride],
                                  externalA[mFractal * sourceFractalStride], transposeParams);
            }
        }

        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kRound, nRound);
        copyL1ToL0B(l0BTensor[accumulatorSlot], l1BTensor[cacheBSlot], layoutBInL0, layoutBInL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(externalReuseEvent);
        if (releaseLocalCache) {
            // Pair traversal has no local streamed-A consumer to publish the
            // regular event.  Return it after the final persistent-B read so
            // the next batch may safely overwrite this cache bank.
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(accumulatorSlot + PINGPONG_FLAG_OFFSET);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

        if (initC) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(accumulatorSlot);
        }
        tileMmad(l0CTensor[accumulatorSlot], l0ATensor[accumulatorSlot], l0BTensor[accumulatorSlot], mRound, nRound,
                 actualShape.k(), initC, finalC ? 3 : 2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(accumulatorSlot);
    }

    __aicore__ inline void AccumulatePreloadedA(uint32_t slot, uint32_t cacheBSlot, LayoutA layoutA, LayoutB layoutB,
                                                GemmCoord actualShape, bool initC, bool finalC,
                                                uint32_t accumulatorSlot)
    {
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        const uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        const uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());
        const uint32_t kRound = RoundUp<L1BAlignHelper::K_ALIGNED>(actualShape.k());

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(slot);
        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mRound, kRound);
        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kRound, nRound);
        copyL1ToL0A(l0ATensor[slot], l1ATensor[slot], layoutAInL0, layoutAInL1);
        copyL1ToL0B(l0BTensor[slot], l1BTensor[cacheBSlot], layoutBInL0, layoutBInL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(slot + PINGPONG_FLAG_OFFSET);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        if (initC) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(accumulatorSlot);
        }
        tileMmad(l0CTensor[accumulatorSlot], l0ATensor[slot], l0BTensor[slot], mRound, nRound, actualShape.k(), initC,
                 finalC ? 3 : 2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(slot);
    }

    __aicore__ inline void FlushAccumulator(AscendC::GlobalTensor<ElementC> gC, LayoutC layoutC, GemmCoord actualShape)
    {
        FlushAccumulator(gC, layoutC, actualShape, isAtomicAdd);
    }

    __aicore__ inline void FlushAccumulator(AscendC::GlobalTensor<ElementC> gC, LayoutC layoutC, GemmCoord actualShape,
                                            bool atomicWrite, uint32_t accumulatorSlot = 0)
    {
        auto blockShape = MakeCoord(actualShape.m(), actualShape.n());
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(blockShape);
        if (atomicWrite) {
            AscendC::SetAtomicAdd<ElementC>();
            copyL0CToGm(gC, l0CTensor[accumulatorSlot], layoutC, layoutInL0C, 3);
            AscendC::SetAtomicNone();
        } else {
            copyL0CToGm(gC, l0CTensor[accumulatorSlot], layoutC, layoutInL0C, 3);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(accumulatorSlot);
    }

    __aicore__ inline void SetFastCachedInputs(bool enabled)
    {
        fastCachedInputs = enabled;
    }

    // Scatter two row ranges from one completed L0C accumulator.  Sparse KV
    // fragments may be non-contiguous in GM while remaining contiguous in
    // the gathered compute tile.  Keeping the full tile in L0C avoids a
    // second MMAD; only the two final Fixpipe stores are split.
    __aicore__ inline void FlushAccumulatorSplitRows(AscendC::GlobalTensor<ElementC> first, LayoutC firstLayout,
                                                     AscendC::GlobalTensor<ElementC> second, LayoutC secondLayout,
                                                     GemmCoord actualShape, uint32_t firstRows, bool atomicWrite,
                                                     uint32_t accumulatorSlot = 0)
    {
        const uint32_t secondRows = actualShape.m() - firstRows;
        auto fullL0CLayout = LayoutCInL0::MakeLayoutInL0C(MakeCoord(actualShape.m(), actualShape.n()));
        auto firstL0CLayout = fullL0CLayout.GetTileLayout(MakeCoord(firstRows, actualShape.n()));
        auto secondL0CLayout = fullL0CLayout.GetTileLayout(MakeCoord(secondRows, actualShape.n()));
        const uint64_t secondOffset = fullL0CLayout.GetOffset(MatrixCoord(firstRows, 0));
        if (atomicWrite) {
            AscendC::SetAtomicAdd<ElementC>();
        }
        copyL0CToGm(first, l0CTensor[accumulatorSlot], firstLayout, firstL0CLayout, 3);
        copyL0CToGm(second, l0CTensor[accumulatorSlot][secondOffset], secondLayout, secondL0CLayout, 3);
        if (atomicWrite) {
            AscendC::SetAtomicNone();
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(accumulatorSlot);
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<ElementA> gA, AscendC::GlobalTensor<ElementB> gB,
                                      AscendC::GlobalTensor<ElementC> gC, LayoutA layoutA, LayoutB layoutB,
                                      LayoutC layoutC, GemmCoord actualShape, uint32_t &pingpongFlag)
    {
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);

        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());
        uint32_t kRound = RoundUp<L1BAlignHelper::K_ALIGNED>(actualShape.k());

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlag + PINGPONG_FLAG_OFFSET);
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(actualShape.k(), actualShape.n()));
        copyGmToL1B(l1BTensor[pingpongFlag], gB, layoutBInL1, layoutTileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(pingpongFlag);
        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kRound, nRound);
        copyL1ToL0B(l0BTensor[pingpongFlag], l1BTensor[pingpongFlag], layoutBInL0, layoutBInL1);

        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), actualShape.k()));
        copyGmToL1A(l1ATensor[pingpongFlag], gA, layoutAInL1, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mRound, kRound);
        copyL1ToL0A(l0ATensor[pingpongFlag], l1ATensor[pingpongFlag], layoutAInL0, layoutAInL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlag + PINGPONG_FLAG_OFFSET);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(pingpongFlag);
        tileMmad(l0CTensor[pingpongFlag], l0ATensor[pingpongFlag], l0BTensor[pingpongFlag], mRound, nRound,
                 actualShape.k());
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(pingpongFlag);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);

        // copy block out
        auto blockShape = MakeCoord(actualShape.m(), actualShape.n());
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(blockShape);
        if (isAtomicAdd) {
            AscendC::SetAtomicAdd<ElementC>();
            copyL0CToGm(gC, l0CTensor[pingpongFlag], layoutC, layoutInL0C);
            AscendC::SetAtomicNone();
        } else {
            copyL0CToGm(gC, l0CTensor[pingpongFlag], layoutC, layoutInL0C);
        }

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(pingpongFlag);

        pingpongFlag = 1 - pingpongFlag;
    }

protected:
    AscendC::LocalTensor<ElementA> l1ATensor[EXTENDED_CACHE_SLOTS];
    AscendC::LocalTensor<ElementB> l1BTensor[EXTENDED_CACHE_SLOTS];

    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[STAGES];

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
    CopyL0CToGmInput copyL0CToGmInput;
    CopyL0CToGmNz copyL0CToGmNz;
    bool fastCachedInputs = false;

    uint32_t PINGPONG_FLAG_OFFSET = 0;
    bool isAtomicAdd = false;
};

} // namespace NpuArch::Gemm::Block

#endif // GEMM_BLOCK_MMAD_BSAG_2_HPP
