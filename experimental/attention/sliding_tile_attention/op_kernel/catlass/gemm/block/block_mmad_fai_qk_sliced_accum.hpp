/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SLIDING_TILE_ATTENTION_BLOCK_MMAD_FAI_QK_SLICED_ACCUM_HPP
#define SLIDING_TILE_ATTENTION_BLOCK_MMAD_FAI_QK_SLICED_ACCUM_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

namespace SlidingTileAttentionCatlass {

template <class DispatchPolicy, class L1TileShape, class L0TileShape, class AType, class BType, class CType,
          class TileCopy = Catlass::Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, void>,
          class TileMmad = Catlass::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, void>>
struct BlockMmadFaiQkSlicedAccum {
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

    static constexpr uint32_t SLICE_DIM = 128;
    static constexpr uint32_t N_GROUP_DIM = 256;
    static constexpr uint32_t EVENT_Q = 6;
    static constexpr uint32_t EVENT_K = 7;
    static constexpr uint32_t L1_A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1_B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);

    static_assert(std::is_same_v<LayoutC, Catlass::layout::RowMajor>, "QK output must be row-major");
    static_assert(!DispatchPolicy::ENABLE_UNIT_FLAG, "sliced accumulation owns one terminal Fixpipe");

    CATLASS_DEVICE explicit BlockMmadFaiQkSlicedAccum(Catlass::Arch::Resource<ArchTag> &resource,
                                                      uint32_t l1BufAddrStart = 0)
    {
        l1ATensor = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart);
        l1BTensor = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1_A_SIZE);
        l0ATensor = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
        l0BTensor = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
    }

    CATLASS_DEVICE void operator()(AscendC::GlobalTensor<ElementA> q, AscendC::GlobalTensor<ElementB> k,
                                   AscendC::GlobalTensor<ElementC> score, uint32_t rowNum, uint32_t columnNum,
                                   uint32_t sliceCount, uint32_t qSliceStride, uint32_t kSliceStride,
                                   uint32_t scoreStride, uint32_t logicalDim)
    {
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(rowNum);

        for (uint32_t nBase = 0; nBase < columnNum; nBase += N_GROUP_DIM) {
            uint32_t groupColumns = columnNum - nBase < N_GROUP_DIM ? columnNum - nBase : N_GROUP_DIM;
            uint32_t nBlockCount = (groupColumns + SLICE_DIM - 1) / SLICE_DIM;
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);

            for (uint32_t sliceIdx = 0; sliceIdx < sliceCount; ++sliceIdx) {
                uint32_t sliceBase = sliceIdx * SLICE_DIM;
                uint32_t kActual = logicalDim - sliceBase < SLICE_DIM ? logicalDim - sliceBase : SLICE_DIM;
                LayoutA qLayout(rowNum, kActual, SLICE_DIM);
                LayoutAInL1 qL1Layout = LayoutAInL1::template MakeLayout<ElementA>(mRound, kActual);
                LayoutAInL0 qL0Layout = LayoutAInL0::template MakeLayout<ElementA>(mRound, kActual);
                uint64_t qOffset = static_cast<uint64_t>(sliceIdx) * qSliceStride;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_Q);
                copyGmToL1A(l1ATensor, q[qOffset], qL1Layout, qLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_Q);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_Q);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_Q);
                copyL1ToL0A(l0ATensor, l1ATensor, qL0Layout, qL1Layout);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_Q);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_Q);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_Q);

                for (uint32_t nBlock = 0; nBlock < nBlockCount; ++nBlock) {
                    uint32_t blockColumn = nBase + nBlock * SLICE_DIM;
                    uint32_t nActual = columnNum - blockColumn < SLICE_DIM ? columnNum - blockColumn : SLICE_DIM;
                    uint64_t kOffset = static_cast<uint64_t>(sliceIdx) * kSliceStride +
                                       static_cast<uint64_t>(blockColumn) * SLICE_DIM;
                    LayoutB kLayout(kActual, nActual, SLICE_DIM);
                    LayoutBInL1 kL1Layout = LayoutBInL1::template MakeLayout<ElementB>(kActual, nActual);
                    LayoutBInL0 kL0Layout = LayoutBInL0::template MakeLayout<ElementB>(kActual, nActual);

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K);
                    copyGmToL1B(l1BTensor, k[kOffset], kL1Layout, kLayout);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_K);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_K);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_K);
                    copyL1ToL0B(l0BTensor, l1BTensor, kL0Layout, kL1Layout);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_K);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_K);
                    tileMmad(l0CTensor[nBlock * mRound * SLICE_DIM], l0ATensor, l0BTensor, mRound, nActual,
                             kActual, sliceIdx == 0, 0);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_K);
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_Q);
            }

            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            auto l0Layout = LayoutCInL0::MakeLayoutInL0C(Catlass::MakeCoord(rowNum, N_GROUP_DIM));
            LayoutC outputLayout(rowNum, groupColumns, scoreStride);
            copyL0CToGm(score[nBase], l0CTensor, outputLayout, l0Layout);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        }
    }

    AscendC::LocalTensor<ElementA> l1ATensor;
    AscendC::LocalTensor<ElementB> l1BTensor;
    AscendC::LocalTensor<ElementA> l0ATensor;
    AscendC::LocalTensor<ElementB> l0BTensor;
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;
    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};

} // namespace SlidingTileAttentionCatlass

#endif
