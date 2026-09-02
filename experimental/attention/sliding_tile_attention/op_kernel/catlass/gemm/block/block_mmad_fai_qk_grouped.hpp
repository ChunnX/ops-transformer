/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef SLIDING_TILE_ATTENTION_BLOCK_MMAD_FAI_QK_GROUPED_HPP
#define SLIDING_TILE_ATTENTION_BLOCK_MMAD_FAI_QK_GROUPED_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

namespace SlidingTileAttentionCatlass {

template <class DispatchPolicy, class L1TileShape, class L0TileShape, class AType, class BType, class CType,
          uint32_t QueryCount = 3, uint32_t ResidentKStages = 4,
          class TileCopy = Catlass::Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, void>,
          class TileMmad = Catlass::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, void>>
struct BlockMmadFaiQkGrouped {
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
    static constexpr uint32_t RESIDENT_K_STAGES = ResidentKStages;
    static constexpr uint32_t L1_A_BYTES = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1_B_BYTES = L1TileShape::N * L1TileShape::K * sizeof(ElementB);

    static_assert(std::is_same_v<LayoutC, Catlass::layout::RowMajor>, "QK output must be row-major");
    static_assert(DispatchPolicy::ENABLE_UNIT_FLAG, "grouped QK requires unit-owned MMAD/Fixpipe completion");
    static_assert(RESIDENT_K_STAGES == 2 || RESIDENT_K_STAGES == 4,
                  "grouped QK supports two-stage or four-stage resident K ownership");

    CATLASS_DEVICE explicit BlockMmadFaiQkGrouped(Catlass::Arch::Resource<ArchTag> &resource)
    {
        for (uint32_t query = 0; query < QUERY_COUNT; ++query) {
            qL1[query] = resource.l1Buf.template GetBufferByByte<ElementA>(query * L1_A_BYTES);
        }
        for (uint32_t stage = 0; stage < RESIDENT_K_STAGES; ++stage) {
            kL1[stage] =
                resource.l1Buf.template GetBufferByByte<ElementB>(QUERY_COUNT * L1_A_BYTES + stage * L1_B_BYTES);
        }
        for (uint32_t stage = 0; stage < 2; ++stage) {
            qL0[stage] = resource.l0ABuf.template GetBufferByByte<ElementA>(stage * ArchTag::L0A_SIZE / 2);
            kL0[stage] = resource.l0BBuf.template GetBufferByByte<ElementB>(stage * ArchTag::L0B_SIZE / 2);
        }
        cL0 = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
    }

    CATLASS_DEVICE void ComputeStackQueryMajor(AscendC::GlobalTensor<ElementB> k, AscendC::GlobalTensor<ElementC> score,
                                               LayoutB layoutK, uint32_t queryRows, uint32_t stackRows, uint32_t kvIdx,
                                               uint32_t kvBlocks, uint32_t segmentRows, uint32_t strideKV,
                                               uint64_t scoreChunkStride, Catlass::Arch::CrossCoreFlag qkReady,
                                               const uint64_t *blockOffsets = nullptr)
    {
        static_assert(RESIDENT_K_STAGES == MAX_STACK_BLOCKS, "query-major grouped QK requires four resident K stages");
        uint32_t embed = layoutK.shape(0);
        uint32_t rowsRound = RoundUp<L1AAlignHelper::M_ALIGNED>(queryRows);
        LayoutAInL0 qL0Layout = LayoutAInL0::template MakeLayout<ElementA>(rowsRound, embed);
        LayoutAInL1 qL1Layout = LayoutAInL1::template MakeLayout<ElementA>(rowsRound, embed);
        uint32_t stackBlocks = (stackRows + BLOCK_ROWS - 1) / BLOCK_ROWS;
        if (stackBlocks != MAX_STACK_BLOCKS) {
            ComputeStack(k, score, layoutK, queryRows, stackRows, kvIdx, kvBlocks, segmentRows, strideKV,
                         scoreChunkStride, qkReady, blockOffsets);
            return;
        }

#pragma unroll
        for (uint32_t block = 0; block < MAX_STACK_BLOCKS; ++block) {
            uint32_t globalBlock = kvIdx + block;
            LayoutBInL1 kL1Layout = LayoutBInL1::template MakeLayout<ElementB>(embed, BLOCK_ROWS);
            auto kTile = layoutK.GetTileLayout(Catlass::MakeCoord(embed, BLOCK_ROWS));
            uint64_t kOffset = blockOffsets == nullptr ? static_cast<uint64_t>(globalBlock) * BLOCK_ROWS * strideKV :
                                                         blockOffsets[block];
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(block);
            copyGmToL1B(kL1[block], k[kOffset], kL1Layout, kTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(block);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(block);
        }

#pragma unroll
        for (uint32_t query = 0; query < QUERY_COUNT; ++query) {
            uint32_t qStage = query & 1U;
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(qStage);
            copyL1ToL0A(qL0[qStage], qL1[query], qL0Layout, qL1Layout);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(qStage);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(qStage);

#pragma unroll
            for (uint32_t pairBase = 0; pairBase < MAX_STACK_BLOCKS; pairBase += 2) {
#pragma unroll
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    uint32_t block = pairBase + pair;
                    LayoutBInL1 kL1Layout = LayoutBInL1::template MakeLayout<ElementB>(embed, BLOCK_ROWS);
                    LayoutBInL0 kL0Layout = LayoutBInL0::template MakeLayout<ElementB>(embed, BLOCK_ROWS);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(pair + 2);
                    copyL1ToL0B(kL0[pair], kL1[block], kL0Layout, kL1Layout);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(pair + 2);
                }
#pragma unroll
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(pair + 2);
                    tileMmad(cL0[pair * rowsRound * BLOCK_ROWS], qL0[qStage], kL0[pair], rowsRound, BLOCK_ROWS, embed,
                             true, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(pair + 2);
                }
                auto cL0Layout = LayoutCInL0::MakeLayoutInL0C(Catlass::MakeCoord(queryRows, 2 * BLOCK_ROWS));
                LayoutC scoreLayout(queryRows, 2 * BLOCK_ROWS, MAX_STACK_BLOCKS * BLOCK_ROWS);
                uint64_t scoreOffset =
                    static_cast<uint64_t>(query) * scoreChunkStride + static_cast<uint64_t>(pairBase) * BLOCK_ROWS;
                copyL0CToGm(score[scoreOffset], cL0, scoreLayout, cL0Layout, 0b11);
            }
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(qStage);
            Catlass::Arch::CrossCoreSetFlag<2, PIPE_FIX>(qkReady);
        }

#pragma unroll
        for (uint32_t block = 0; block < MAX_STACK_BLOCKS; ++block) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(block);
        }
    }

    CATLASS_DEVICE void LoadQueries(AscendC::GlobalTensor<ElementA> q, LayoutA layoutQ, uint32_t queryRows,
                                    uint32_t queryStrideRows, uint32_t queryCount = QUERY_COUNT)
    {
        uint32_t embed = layoutQ.shape(1);
        uint32_t rowsRound = RoundUp<L1AAlignHelper::M_ALIGNED>(queryRows);
        LayoutAInL1 qL1Layout = LayoutAInL1::template MakeLayout<ElementA>(rowsRound, embed);
        auto qTile = layoutQ.GetTileLayout(Catlass::MakeCoord(queryRows, embed));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        for (uint32_t query = 0; query < queryCount; ++query) {
            copyGmToL1A(qL1[query], q[static_cast<uint64_t>(query) * queryStrideRows * embed], qL1Layout, qTile);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
    }

    CATLASS_DEVICE void LoadQueriesOffsets(AscendC::GlobalTensor<ElementA> q, LayoutA layoutQ, uint32_t queryRows,
                                           const uint64_t *queryOffsets, uint32_t queryCount)
    {
        uint32_t embed = layoutQ.shape(1);
        uint32_t rowsRound = RoundUp<L1AAlignHelper::M_ALIGNED>(queryRows);
        LayoutAInL1 qL1Layout = LayoutAInL1::template MakeLayout<ElementA>(rowsRound, embed);
        auto qTile = layoutQ.GetTileLayout(Catlass::MakeCoord(queryRows, embed));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        for (uint32_t query = 0; query < queryCount; ++query) {
            copyGmToL1A(qL1[query], q[queryOffsets[query]], qL1Layout, qTile);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
    }

    CATLASS_DEVICE void ComputeStack(AscendC::GlobalTensor<ElementB> k, AscendC::GlobalTensor<ElementC> score,
                                     LayoutB layoutK, uint32_t queryRows, uint32_t stackRows, uint32_t kvIdx,
                                     uint32_t kvBlocks, uint32_t segmentRows, uint32_t strideKV,
                                     uint64_t scoreChunkStride, Catlass::Arch::CrossCoreFlag qkReady,
                                     const uint64_t *blockOffsets = nullptr, uint32_t queryCount = QUERY_COUNT)
    {
        uint32_t embed = layoutK.shape(0);
        uint32_t rowsRound = RoundUp<L1AAlignHelper::M_ALIGNED>(queryRows);
        LayoutAInL0 qL0Layout = LayoutAInL0::template MakeLayout<ElementA>(rowsRound, embed);
        LayoutAInL1 qL1Layout = LayoutAInL1::template MakeLayout<ElementA>(rowsRound, embed);
        uint32_t stackBlocks = (stackRows + BLOCK_ROWS - 1) / BLOCK_ROWS;

        for (uint32_t pairBase = 0; pairBase < stackBlocks; pairBase += 2) {
            for (uint32_t pair = 0; pair < 2 && pairBase + pair < stackBlocks; ++pair) {
                uint32_t block = pairBase + pair;
                uint32_t globalBlock = kvIdx + block;
                uint32_t columns = globalBlock + 1 == kvBlocks ? segmentRows - globalBlock * BLOCK_ROWS : BLOCK_ROWS;
                LayoutBInL1 kL1Layout = LayoutBInL1::template MakeLayout<ElementB>(embed, columns);
                LayoutBInL0 kL0Layout = LayoutBInL0::template MakeLayout<ElementB>(embed, columns);
                auto kTile = layoutK.GetTileLayout(Catlass::MakeCoord(embed, columns));
                uint64_t kOffset = blockOffsets == nullptr ?
                                       static_cast<uint64_t>(globalBlock) * BLOCK_ROWS * strideKV :
                                       blockOffsets[block];
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(pair);
                copyGmToL1B(kL1[pair], k[kOffset], kL1Layout, kTile);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(pair);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(pair);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(pair + 2);
                copyL1ToL0B(kL0[pair], kL1[pair], kL0Layout, kL1Layout);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(pair);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(pair + 2);
            }

            for (uint32_t query = 0; query < queryCount; ++query) {
                uint32_t qStage = query & 1U;
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(qStage);
                copyL1ToL0A(qL0[qStage], qL1[query], qL0Layout, qL1Layout);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(qStage);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(qStage);
                for (uint32_t pair = 0; pair < 2 && pairBase + pair < stackBlocks; ++pair) {
                    uint32_t globalBlock = kvIdx + pairBase + pair;
                    uint32_t columns =
                        globalBlock + 1 == kvBlocks ? segmentRows - globalBlock * BLOCK_ROWS : BLOCK_ROWS;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(pair + 2);
                    tileMmad(cL0[pair * rowsRound * BLOCK_ROWS], qL0[qStage], kL0[pair], rowsRound, columns, embed,
                             true, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(pair + 2);
                    if (query + 1 < queryCount) {
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(pair + 2);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(pair + 2);
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(qStage);
                uint32_t groupColumns = 0;
                for (uint32_t pair = 0; pair < 2 && pairBase + pair < stackBlocks; ++pair) {
                    uint32_t globalBlock = kvIdx + pairBase + pair;
                    uint32_t columns =
                        globalBlock + 1 == kvBlocks ? segmentRows - globalBlock * BLOCK_ROWS : BLOCK_ROWS;
                    groupColumns += columns;
                }
                auto cL0Layout = LayoutCInL0::MakeLayoutInL0C(Catlass::MakeCoord(queryRows, 2 * BLOCK_ROWS));
                LayoutC scoreLayout(queryRows, groupColumns, MAX_STACK_BLOCKS * BLOCK_ROWS);
                uint64_t scoreOffset =
                    static_cast<uint64_t>(query) * scoreChunkStride + static_cast<uint64_t>(pairBase) * BLOCK_ROWS;
                copyL0CToGm(score[scoreOffset], cL0, scoreLayout, cL0Layout, 0b11);
            }
        }
        for (uint32_t query = 0; query < queryCount; ++query) {
            Catlass::Arch::CrossCoreSetFlag<2, PIPE_FIX>(qkReady);
        }
    }

    AscendC::LocalTensor<ElementA> qL1[QUERY_COUNT];
    AscendC::LocalTensor<ElementB> kL1[RESIDENT_K_STAGES];
    AscendC::LocalTensor<ElementA> qL0[2];
    AscendC::LocalTensor<ElementB> kL0[2];
    AscendC::LocalTensor<ElementAccumulator> cL0;
    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};

} // namespace SlidingTileAttentionCatlass

#endif
