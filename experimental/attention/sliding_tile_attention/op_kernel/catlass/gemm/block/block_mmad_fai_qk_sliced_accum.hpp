/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef SLIDING_TILE_ATTENTION_BLOCK_MMAD_FAI_QK_SLICED_ACCUM_HPP
#define SLIDING_TILE_ATTENTION_BLOCK_MMAD_FAI_QK_SLICED_ACCUM_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "../../../resource_ledger.hpp"

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
    // Keep private L1 staging channels disjoint from the physical L0 ownership
    // channels used by CATLASS: L0A parity starts at 0 and L0B at 2.
    static constexpr uint32_t EVENT_Q = 6;
    static constexpr uint32_t EVENT_K = 7;
    static constexpr uint32_t EVENT_L0A = 0;
    static constexpr uint32_t EVENT_L0B = 2;
    static constexpr uint32_t L1_A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1_B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
    static constexpr uint32_t ELE_NUM_PER_C0 =
        Catlass::BytesToBits(Catlass::BYTE_PER_C0) / Catlass::SizeOfBits<ElementB>::value;

    static_assert(std::is_same_v<LayoutC, Catlass::layout::RowMajor>, "QK output must be row-major");
    static_assert(!DispatchPolicy::ENABLE_UNIT_FLAG, "sliced accumulation owns one terminal Fixpipe");
    static_assert(opforge::ascend_c::resource_ledger::ValidateL0ReuseEventPair(
        opforge::ascend_c::resource_ledger::L0Event::MTE1_M, opforge::ascend_c::resource_ledger::L0Event::M_MTE1));
    static_assert(opforge::ascend_c::resource_ledger::ValidateLayoutHandoff<LayoutBInL1, LayoutBInL1>(),
                  "packed raw-K GM-to-L1 and L1-to-L0 layouts must match");

    CATLASS_DEVICE explicit BlockMmadFaiQkSlicedAccum(Catlass::Arch::Resource<ArchTag> &resource,
                                                      uint32_t l1BufAddrStart = 0)
    {
        l1ATensor = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart);
        l1BTensor = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1_A_SIZE);
        l0ATensor = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
        l0BTensor = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
    }

    CATLASS_DEVICE void LoadPackedD256Queries(AscendC::GlobalTensor<ElementA> q, uint32_t rowNum,
                                              uint32_t sourceRowStride)
    {
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(rowNum);
        AscendC::LocalTensor<ElementA> packedQSecond = l1ATensor[L1_A_SIZE / sizeof(ElementA)];
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_Q);
        for (uint32_t sliceIdx = 0; sliceIdx < 2; ++sliceIdx) {
            LayoutA qLayout(rowNum, SLICE_DIM, sourceRowStride);
            LayoutAInL1 qL1Layout = LayoutAInL1::template MakeLayout<ElementA>(mRound, SLICE_DIM);
            AscendC::LocalTensor<ElementA> dst = sliceIdx == 0 ? l1ATensor : packedQSecond;
            copyGmToL1A(dst, q[sliceIdx * SLICE_DIM], qL1Layout, qLayout);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_Q);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_Q);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_Q);
    }

    CATLASS_DEVICE void ComputePackedD256Stack(AscendC::GlobalTensor<ElementB> k, AscendC::GlobalTensor<ElementC> score,
                                               uint32_t rowNum, uint32_t columnNum, uint32_t scoreStride,
                                               const uint64_t *blockOffsets)
    {
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(rowNum);
        AscendC::LocalTensor<ElementA> packedQSecond = l1ATensor[L1_A_SIZE / sizeof(ElementA)];
        AscendC::LocalTensor<ElementB> packedKTensor =
            l1ATensor[2 * L1_A_SIZE / sizeof(ElementA)].template ReinterpretCast<ElementB>();
        for (uint32_t nBase = 0; nBase < columnNum; nBase += N_GROUP_DIM) {
            uint32_t groupColumns = columnNum - nBase < N_GROUP_DIM ? columnNum - nBase : N_GROUP_DIM;
            uint32_t nBlockCount = (groupColumns + SLICE_DIM - 1) / SLICE_DIM;
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            for (uint32_t sliceIdx = 0; sliceIdx < 2; ++sliceIdx) {
                uint32_t sliceBase = sliceIdx * SLICE_DIM;
                LayoutAInL1 qL1Layout = LayoutAInL1::template MakeLayout<ElementA>(mRound, SLICE_DIM);
                LayoutAInL0 qL0Layout = LayoutAInL0::template MakeLayout<ElementA>(mRound, SLICE_DIM);
                AscendC::LocalTensor<ElementA> qL1 = sliceIdx == 0 ? l1ATensor : packedQSecond;
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
                copyL1ToL0A(l0ATensor, qL1, qL0Layout, qL1Layout);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0A);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0A);
                for (uint32_t nBlock = 0; nBlock < nBlockCount; ++nBlock) {
                    uint32_t blockColumn = nBase + nBlock * SLICE_DIM;
                    uint32_t nActual = columnNum - blockColumn < SLICE_DIM ? columnNum - blockColumn : SLICE_DIM;
                    uint32_t physicalBlock = blockColumn / SLICE_DIM;
                    uint64_t kOffset = blockOffsets[physicalBlock] + sliceBase;
                    LayoutBInL1 kL1Layout = LayoutBInL1::template MakeLayout<ElementB>(SLICE_DIM, nActual);
                    LayoutBInL0 kL0Layout = LayoutBInL0::template MakeLayout<ElementB>(SLICE_DIM, nActual);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K);
                    CopyRawKNdToNz(packedKTensor, k[kOffset], kL1Layout, SLICE_DIM, nActual, 256);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_K);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_K);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
                    copyL1ToL0B(l0BTensor, packedKTensor, kL0Layout, kL1Layout);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0B);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0B);
                    tileMmad(l0CTensor[nBlock * mRound * SLICE_DIM], l0ATensor, l0BTensor, mRound, nActual, SLICE_DIM,
                             sliceIdx == 0, 0);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
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

    CATLASS_DEVICE void LoadPackedD3SliceQueries(AscendC::GlobalTensor<ElementA> q, uint32_t rowNum,
                                                 uint32_t sourceRowStride)
    {
        constexpr uint32_t SLICE_COUNT = 3;
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(rowNum);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_Q);
        for (uint32_t sliceIdx = 0; sliceIdx < SLICE_COUNT; ++sliceIdx) {
            uint32_t sliceBase = sliceIdx * SLICE_DIM;
            uint32_t kActual = sourceRowStride - sliceBase < SLICE_DIM ? sourceRowStride - sliceBase : SLICE_DIM;
            LayoutA qLayout(rowNum, kActual, sourceRowStride);
            LayoutAInL1 qL1Layout = LayoutAInL1::template MakeLayout<ElementA>(mRound, kActual);
            AscendC::LocalTensor<ElementA> dst = l1ATensor[sliceIdx * L1_A_SIZE / sizeof(ElementA)];
            copyGmToL1A(dst, q[sliceBase], qL1Layout, qLayout);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_Q);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_Q);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_Q);
    }

    CATLASS_DEVICE void ComputePackedD3SliceStack(AscendC::GlobalTensor<ElementB> k,
                                                  AscendC::GlobalTensor<ElementC> score, uint32_t rowNum,
                                                  uint32_t columnNum, uint32_t scoreStride, uint32_t sourceRowStride,
                                                  const uint64_t *blockOffsets)
    {
        constexpr uint32_t SLICE_COUNT = 3;
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(rowNum);
        AscendC::LocalTensor<ElementB> packedKTensor =
            l1ATensor[SLICE_COUNT * L1_A_SIZE / sizeof(ElementA)].template ReinterpretCast<ElementB>();
        for (uint32_t nBase = 0; nBase < columnNum; nBase += N_GROUP_DIM) {
            uint32_t groupColumns = columnNum - nBase < N_GROUP_DIM ? columnNum - nBase : N_GROUP_DIM;
            uint32_t nBlockCount = (groupColumns + SLICE_DIM - 1) / SLICE_DIM;
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            for (uint32_t sliceIdx = 0; sliceIdx < SLICE_COUNT; ++sliceIdx) {
                uint32_t sliceBase = sliceIdx * SLICE_DIM;
                uint32_t kActual = sourceRowStride - sliceBase < SLICE_DIM ? sourceRowStride - sliceBase : SLICE_DIM;
                LayoutAInL1 qL1Layout = LayoutAInL1::template MakeLayout<ElementA>(mRound, kActual);
                LayoutAInL0 qL0Layout = LayoutAInL0::template MakeLayout<ElementA>(mRound, kActual);
                AscendC::LocalTensor<ElementA> qL1 = l1ATensor[sliceIdx * L1_A_SIZE / sizeof(ElementA)];
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
                copyL1ToL0A(l0ATensor, qL1, qL0Layout, qL1Layout);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0A);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0A);
                for (uint32_t nBlock = 0; nBlock < nBlockCount; ++nBlock) {
                    uint32_t blockColumn = nBase + nBlock * SLICE_DIM;
                    uint32_t nActual = columnNum - blockColumn < SLICE_DIM ? columnNum - blockColumn : SLICE_DIM;
                    uint32_t physicalBlock = blockColumn / SLICE_DIM;
                    uint64_t kOffset = blockOffsets[physicalBlock] + sliceBase;
                    LayoutBInL1 kL1Layout = LayoutBInL1::template MakeLayout<ElementB>(kActual, nActual);
                    LayoutBInL0 kL0Layout = LayoutBInL0::template MakeLayout<ElementB>(kActual, nActual);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K);
                    CopyRawKNdToNz(packedKTensor, k[kOffset], kL1Layout, kActual, nActual, sourceRowStride);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_K);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_K);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
                    copyL1ToL0B(l0BTensor, packedKTensor, kL0Layout, kL1Layout);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0B);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0B);
                    tileMmad(l0CTensor[nBlock * mRound * SLICE_DIM], l0ATensor, l0BTensor, mRound, nActual, kActual,
                             sliceIdx == 0, 0);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
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

    CATLASS_DEVICE void operator()(AscendC::GlobalTensor<ElementA> q, AscendC::GlobalTensor<ElementB> k,
                                   AscendC::GlobalTensor<ElementC> score, uint32_t rowNum, uint32_t columnNum,
                                   uint32_t sliceCount, uint32_t qSliceStride, uint32_t kSliceStride,
                                   uint32_t scoreStride, uint32_t logicalDim, uint32_t rawKRowStride = 0)
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
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
                copyL1ToL0A(l0ATensor, l1ATensor, qL0Layout, qL1Layout);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_Q);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0A);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0A);

                for (uint32_t nBlock = 0; nBlock < nBlockCount; ++nBlock) {
                    uint32_t blockColumn = nBase + nBlock * SLICE_DIM;
                    uint32_t nActual = columnNum - blockColumn < SLICE_DIM ? columnNum - blockColumn : SLICE_DIM;
                    uint64_t kOffset = rawKRowStride == 0 ?
                                           static_cast<uint64_t>(sliceIdx) * kSliceStride +
                                               static_cast<uint64_t>(blockColumn) * SLICE_DIM :
                                           static_cast<uint64_t>(blockColumn) * rawKRowStride + sliceBase;
                    LayoutB kLayout(kActual, nActual, SLICE_DIM);
                    LayoutBInL1 kL1Layout = LayoutBInL1::template MakeLayout<ElementB>(kActual, nActual);
                    LayoutBInL0 kL0Layout = LayoutBInL0::template MakeLayout<ElementB>(kActual, nActual);

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K);
                    if (rawKRowStride == 0) {
                        copyGmToL1B(l1BTensor, k[kOffset], kL1Layout, kLayout);
                    } else {
                        CopyRawKNdToNz(l1BTensor, k[kOffset], kL1Layout, kActual, nActual, rawKRowStride);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_K);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_K);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
                    copyL1ToL0B(l0BTensor, l1BTensor, kL0Layout, kL1Layout);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0B);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0B);
                    tileMmad(l0CTensor[nBlock * mRound * SLICE_DIM], l0ATensor, l0BTensor, mRound, nActual, kActual,
                             sliceIdx == 0, 0);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
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

private:
    CATLASS_DEVICE void CopyRawKNdToNz(AscendC::LocalTensor<ElementB> const &dst,
                                       AscendC::GlobalTensor<ElementB> const &src, LayoutBInL1 const &layoutDst,
                                       uint32_t kActual, uint32_t nActual, uint32_t sourceRowStride)
    {
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = nActual;
        params.dValue = kActual;
        params.srcNdMatrixStride = 0;
        params.srcDValue = sourceRowStride;
        params.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        params.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
        params.dstNzMatrixStride = 0;
        AscendC::DataCopy(dst, src, params);
    }
};

} // namespace SlidingTileAttentionCatlass

#endif
