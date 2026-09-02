/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SLIDING_TILE_ATTENTION_BLOCK_MMAD_FAI_PV_GROUPED_D72_HPP
#define SLIDING_TILE_ATTENTION_BLOCK_MMAD_FAI_PV_GROUPED_D72_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "../../../resource_ledger.hpp"

namespace SlidingTileAttentionCatlass {

// Three 128x80 FP32 accumulators occupy 120KiB of AtlasA2's 128KiB L0C.
template <class DispatchPolicy, class L1TileShape, class L0TileShape, class AType, class BType, class CType,
          uint32_t QueryCount = 3, uint32_t OutputWidth = 80,
          class TileCopy = Catlass::Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, void>,
          class TileMmad = Catlass::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, void>>
struct BlockMmadFaiPvGroupedD72 {
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
    static constexpr uint32_t QUERY_COUNT = QueryCount;
    static constexpr uint32_t BLOCK_ROWS = 128;
    static constexpr uint32_t MAX_STACK_BLOCKS = 4;
    static constexpr uint32_t OUTPUT_WIDTH = OutputWidth;
    static constexpr uint32_t L1_A_BYTES = L1TileShape::M * MAX_STACK_BLOCKS * BLOCK_ROWS * sizeof(ElementA);
    static constexpr uint32_t L1_B_BLOCK_BYTES = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
    static constexpr uint32_t L0C_QUERY_ELEMENTS = BLOCK_ROWS * OUTPUT_WIDTH;
    static_assert(std::is_same_v<LayoutC, Catlass::layout::RowMajor>, "PV output must be row-major");
    static_assert(DispatchPolicy::ENABLE_UNIT_FLAG, "D72 grouped PV requires unit-owned completion");
    static_assert(QUERY_COUNT * L0C_QUERY_ELEMENTS * sizeof(ElementAccumulator) <= ArchTag::L0C_SIZE,
                  "D72 output regions must fit in L0C");
    static_assert(opforge::ascend_c::resource_ledger::ValidateL0ReuseEventPair(
                      opforge::ascend_c::resource_ledger::L0Event::MTE1_M,
                      opforge::ascend_c::resource_ledger::L0Event::M_MTE1),
                  "packed D128 PV requires matching L0 operand publish/release events");
    static_assert(opforge::ascend_c::resource_ledger::ValidateLayoutHandoff<LayoutAInL1, LayoutAInL1>(),
                  "packed D128 P GM-to-L1 and L1-to-L0 layouts must match");
    static_assert(opforge::ascend_c::resource_ledger::ValidateLayoutHandoff<LayoutBInL1, LayoutBInL1>(),
                  "packed D128 V GM-to-L1 and L1-to-L0 layouts must match");

    CATLASS_DEVICE explicit BlockMmadFaiPvGroupedD72(Catlass::Arch::Resource<ArchTag> &resource,
                                                     uint32_t l1BufAddrStart)
    {
        pL1 = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart);
        for (uint32_t block = 0; block < MAX_STACK_BLOCKS; ++block) {
            vL1[block] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1_A_BYTES +
                                                                           block * L1_B_BLOCK_BYTES);
        }
        for (uint32_t stage = 0; stage < 2; ++stage) {
            pL0[stage] = resource.l0ABuf.template GetBufferByByte<ElementA>(stage * ArchTag::L0A_SIZE / 2);
            vL0[stage] = resource.l0BBuf.template GetBufferByByte<ElementB>(stage * ArchTag::L0B_SIZE / 2);
        }
        l0CBase = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
    }

    CATLASS_DEVICE void LoadV(AscendC::GlobalTensor<ElementB> v, LayoutB layoutV, uint32_t stackRows, uint32_t kvIdx,
                              uint32_t kvBlocks, uint32_t segmentRows, uint32_t strideKV,
                              const uint64_t *blockOffsets = nullptr)
    {
        uint32_t embed = layoutV.shape(1);
        uint32_t stackBlocks = (stackRows + BLOCK_ROWS - 1) / BLOCK_ROWS;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        for (uint32_t block = 0; block < stackBlocks; ++block) {
            uint32_t globalBlock = kvIdx + block;
            uint32_t rows = globalBlock + 1 == kvBlocks ? segmentRows - globalBlock * BLOCK_ROWS : BLOCK_ROWS;
            LayoutBInL1 layout = LayoutBInL1::template MakeLayout<ElementB>(rows, embed);
            auto tile = layoutV.GetTileLayout(Catlass::MakeCoord(rows, embed));
            uint64_t vOffset = blockOffsets == nullptr ? static_cast<uint64_t>(globalBlock) * BLOCK_ROWS * strideKV :
                                                         blockOffsets[block];
            copyGmToL1B(vL1[block], v[vOffset], layout, tile);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
    }

    CATLASS_DEVICE void ComputeQuery(AscendC::GlobalTensor<ElementA> p, AscendC::GlobalTensor<ElementC> output,
                                     LayoutA layoutP, uint32_t queryRows, uint32_t embed, uint32_t stackRows,
                                     uint32_t outputStride, Catlass::Arch::CrossCoreFlag softmaxReady, uint32_t query)
    {
        uint32_t rowsRound = RoundUp<L1AAlignHelper::M_ALIGNED>(queryRows);
        uint32_t stackBlocks = (stackRows + BLOCK_ROWS - 1) / BLOCK_ROWS;
        Catlass::Arch::CrossCoreWaitFlag(softmaxReady);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        LayoutAInL1 pLayout = LayoutAInL1::template MakeLayout<ElementA>(rowsRound, MAX_STACK_BLOCKS * BLOCK_ROWS);
        auto pTile = layoutP.GetTileLayout(Catlass::MakeCoord(queryRows, stackRows));
        copyGmToL1A(pL1, p, pLayout, pTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID4);
        for (uint32_t block = 0; block < stackBlocks; ++block) {
            uint32_t stage = block & 1U;
            uint32_t rows = block + 1 == stackBlocks ? stackRows - block * BLOCK_ROWS : BLOCK_ROWS;
            LayoutAInL0 pL0Layout = LayoutAInL0::template MakeLayout<ElementA>(rowsRound, rows);
            LayoutBInL0 vL0Layout = LayoutBInL0::template MakeLayout<ElementB>(rows, embed);
            LayoutBInL1 vL1Layout = LayoutBInL1::template MakeLayout<ElementB>(rows, embed);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(stage);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(stage + 2);
            copyL1ToL0A(pL0[stage], pL1[rowsRound * BLOCK_ROWS * block], pL0Layout, pLayout);
            copyL1ToL0B(vL0[stage], vL1[block], vL0Layout, vL1Layout);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(stage);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(stage + 2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(stage);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(stage + 2);
            bool last = block + 1 == stackBlocks;
            tileMmad(l0CBase[query * L0C_QUERY_ELEMENTS], pL0[stage], vL0[stage], rowsRound, embed, rows, block == 0,
                     last ? 0b11 : 0b10);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(stage);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(stage + 2);
        }
        auto l0CLayout = LayoutCInL0::MakeLayoutInL0C(Catlass::MakeCoord(queryRows, embed));
        LayoutC outputLayout(queryRows, embed, outputStride);
        copyL0CToGm(output, l0CBase[query * L0C_QUERY_ELEMENTS], outputLayout, l0CLayout, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
    }

    CATLASS_DEVICE void ComputeQuerySingleD128(AscendC::GlobalTensor<ElementA> p,
                                               AscendC::GlobalTensor<ElementC> output, LayoutA layoutP,
                                               uint32_t queryRows, uint32_t stackRows, uint32_t outputStride,
                                               Catlass::Arch::CrossCoreFlag softmaxReady)
    {
        constexpr uint32_t FIXED_DIM = 128;
        static_assert(L1TileShape::K == FIXED_DIM, "packed D128 PV requires a 128-column output tile");
        static_assert(BLOCK_ROWS * FIXED_DIM * sizeof(ElementAccumulator) <= ArchTag::L0C_SIZE,
                      "one packed D128 PV accumulator must fit in L0C");
        uint32_t rowsRound = RoundUp<L1AAlignHelper::M_ALIGNED>(queryRows);
        uint32_t stackBlocks = (stackRows + BLOCK_ROWS - 1) / BLOCK_ROWS;
        Catlass::Arch::CrossCoreWaitFlag(softmaxReady);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        LayoutAInL1 pLayout = LayoutAInL1::template MakeLayout<ElementA>(rowsRound, MAX_STACK_BLOCKS * BLOCK_ROWS);
        auto pTile = layoutP.GetTileLayout(Catlass::MakeCoord(queryRows, stackRows));
        copyGmToL1A(pL1, p, pLayout, pTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID4);
        for (uint32_t block = 0; block < stackBlocks; ++block) {
            uint32_t stage = block & 1U;
            uint32_t rows = block + 1 == stackBlocks ? stackRows - block * BLOCK_ROWS : BLOCK_ROWS;
            LayoutAInL0 pL0Layout = LayoutAInL0::template MakeLayout<ElementA>(rowsRound, rows);
            LayoutBInL0 vL0Layout = LayoutBInL0::template MakeLayout<ElementB>(rows, FIXED_DIM);
            LayoutBInL1 vL1Layout = LayoutBInL1::template MakeLayout<ElementB>(rows, FIXED_DIM);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(stage);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(stage + 2);
            copyL1ToL0A(pL0[stage], pL1[rowsRound * BLOCK_ROWS * block], pL0Layout, pLayout);
            copyL1ToL0B(vL0[stage], vL1[block], vL0Layout, vL1Layout);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(stage);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(stage + 2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(stage);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(stage + 2);
            bool first = block == 0;
            bool last = block + 1 == stackBlocks;
            tileMmad(l0CBase, pL0[stage], vL0[stage], rowsRound, FIXED_DIM, rows, first, last ? 0b11 : 0b10);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(stage);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(stage + 2);
        }
        auto l0CLayout = LayoutCInL0::MakeLayoutInL0C(Catlass::MakeCoord(queryRows, FIXED_DIM));
        LayoutC outputLayout(queryRows, FIXED_DIM, outputStride);
        copyL0CToGm(output, l0CBase, outputLayout, l0CLayout, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
    }

    CATLASS_DEVICE void ReleaseV()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
    }

    AscendC::LocalTensor<ElementA> pL1;
    AscendC::LocalTensor<ElementB> vL1[MAX_STACK_BLOCKS];
    AscendC::LocalTensor<ElementA> pL0[2];
    AscendC::LocalTensor<ElementB> vL0[2];
    AscendC::LocalTensor<ElementAccumulator> l0CBase;
    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};
} // namespace SlidingTileAttentionCatlass
#endif
