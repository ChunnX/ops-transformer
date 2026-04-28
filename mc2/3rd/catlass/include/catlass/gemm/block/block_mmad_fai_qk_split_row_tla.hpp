/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_MMAD_QK_SPLIT_ROW_TLA_HPP
#define CATLASS_GEMM_BLOCK_MMAD_QK_SPLIT_ROW_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

////////////////////////////////////////////////////////////////////

namespace Catlass::Gemm::Block {
////////////////////////////////////////////////////////////////////

template <
    class ArchTag_,
    bool PAGED_CACHE_FLAG_,
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class ElementA_,
    class ElementB_,
    class ElementC_,
    class ElementBias_,
    class TileCopy_,
    class TileMmad_>
struct BlockMmadTla<
    MmadFAIQKSplitRow<ArchTag_, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L0TileShape_,
    ElementA_,
    ElementB_,
    ElementC_,
    ElementBias_,
    TileCopy_,
    TileMmad_> {
public:
    // Type Aliases
    using DispatchPolicy = MmadFAIQKSplitRow<ArchTag_, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using LayoutA = typename TileCopy::LayoutA;
    using ElementB = ElementB_;
    using LayoutB = typename TileCopy::LayoutB;
    using ElementC = ElementC_;
    using LayoutC = typename TileCopy::LayoutC;
    using ElementBias = ElementBias_;

    using TileMmad = TileMmad_;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;

    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static_assert(tla::is_tuple<L1TileShape>::value && tla::is_static<L1TileShape>::value,
        "L1TileShape must be tla::tuple and static!");
    static_assert(tla::is_tuple<L0TileShape>::value && tla::is_static<L0TileShape>::value,
        "L0TileShape must be tla::tuple and static!");

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    // L1 tile size
    static constexpr uint32_t L1A_TILE_SIZE = L1_TILE_M * L1_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L1_TILE_N * L1_TILE_K * sizeof(ElementB);
    // L0 tile size
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L0_TILE_M * L0_TILE_N * sizeof(ElementAccumulator);
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / STAGES;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = ArchTag::L0C_SIZE / STAGES;

    // Check LayoutC
    static_assert(tla::detail::isRowMajor<LayoutC>::value ||
                      ((std::is_same_v<ElementC, half> || std::is_same_v<ElementC, bfloat16_t> ||
                          std::is_same_v<ElementC, float>) && tla::detail::iszN<ElementC, LayoutC>::value),
        "LayoutC only supports zN in half or bfloat16 or float, RowMajor in all dtype yet!");

    // Check L1TileShape
    static_assert(L1A_TILE_SIZE + L1B_TILE_SIZE * STAGES <= ArchTag::L1_SIZE,
        "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static_assert(L0A_TILE_SIZE * STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE * STAGES <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");

    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {
        // Allocate L1 memory space
        l1ATensor = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart);
        for (uint32_t i = 0; i < STAGES; i++) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1A_TILE_SIZE + L1B_TILE_SIZE * i);
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
        }
    }

    CATLASS_DEVICE
    ~BlockMmadTla() {}

    template <class TensorA>
    CATLASS_DEVICE
    void loadQGM(TensorA &tensorA)
    {
        using CopyGmToL1A = Gemm::Tile::TileCopyFAQTla<ArchTag, TensorA, typename TileCopy_::TensorL1A>;
        CopyGmToL1A copyGmToL1A;

        uint32_t rowNum = tla::get<0>(tensorA.shape()) * tla::get<1>(tensorA.shape());
        uint32_t embed = tla::get<2>(tensorA.shape());
        auto L1A_LAYOUT = tla::MakeLayout<ElementA, LayoutTagL1A>(rowNum, embed);
        auto tensorL1A = tla::MakeTensor(l1ATensor, L1A_LAYOUT, Arch::PositionL1{});
        copyGmToL1A(tensorL1A, tensorA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
    }

    CATLASS_DEVICE
    void getBlockShape(GemmCoord &actualShape, uint32_t nL1Idx, uint32_t nL1Loop, uint32_t stackSeqTile)
    {
        uint32_t nSplitSize = L1_TILE_N;
        if (nL1Idx == nL1Loop - 1) {
            nSplitSize = stackSeqTile - nL1Idx * L1_TILE_N;
        }
        actualShape[1] = nSplitSize;
    }

    CATLASS_DEVICE
    void getKVOffset(AscendC::GlobalTensor<int32_t> &gBlockTable, uint32_t &kOffset, uint32_t nowNIdx, uint32_t nL1Idx,
        uint32_t strideKV, uint32_t blockSize)
    {
        if constexpr (PAGED_CACHE_FLAG_) {
            uint32_t blockTableId = gBlockTable.GetValue(nowNIdx);
            kOffset = blockTableId * blockSize * strideKV + nL1Idx * L1_TILE_N * strideKV;
        } else {
            kOffset = nowNIdx * blockSize * strideKV + nL1Idx * L1_TILE_N * strideKV;
        }
    }

    template <class TensorA, class TensorB, class TensorC>
    CATLASS_DEVICE
    void operator()(TensorA &tensorA, TensorB &tensorB, TensorC &tensorC,
                    AscendC::GlobalTensor<int32_t> gBlockTable, GemmCoord actualOriShape,
                    uint32_t &nIdx, uint32_t &nLoop, uint32_t &blockSize, uint32_t strideKV)
    {
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<TensorB>;
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorC>;
        CopyGmToL1B copyGmToL1B;
        CopyL0CToDst copyL0CToDst;

        uint32_t rowNum = actualOriShape[0];
        uint32_t stackSeqTile = actualOriShape[1];
        uint32_t embed = actualOriShape[2];

        GemmCoord actualShape{rowNum, 0, embed};
        GemmCoord actualNextShape{rowNum, 0, embed};
        uint32_t gBOffset = 0;
        uint32_t gBNextOffset = 0;

        auto L1A_LAYOUT = tla::MakeLayout<ElementA, LayoutTagL1A>(rowNum, embed);

        uint32_t tileNNumPerPaged = blockSize / L1_TILE_N;
        uint32_t nL1Loop = CeilDiv<L1_TILE_N>(stackSeqTile);
        for (uint32_t nL1Idx = 0; nL1Idx < nL1Loop; ++nL1Idx) {
            uint32_t nowNIdx = nIdx + nL1Idx / tileNNumPerPaged;
            uint32_t nextNL1Idx = nL1Idx + 1;
            uint32_t nextNIdx = nIdx + nextNL1Idx / tileNNumPerPaged;
            getBlockShape(actualShape, nL1Idx, nL1Loop, stackSeqTile);
            getBlockShape(actualNextShape, nextNL1Idx, nL1Loop, stackSeqTile);
            getKVOffset(gBlockTable, gBOffset, nowNIdx, nL1Idx % tileNNumPerPaged, strideKV, blockSize);
            getKVOffset(gBlockTable, gBNextOffset, nextNIdx, nextNL1Idx % tileNNumPerPaged, strideKV, blockSize);

            uint32_t mActual = actualShape.m();
            uint32_t kActual = actualShape.k();
            uint32_t nActual = actualShape.n();

            auto L1B_LAYOUT = tla::MakeLayout<ElementB, LayoutTagL1B>(kActual, nActual);
            auto tensorL1B = tla::MakeTensor(l1BTensor[l1KvPingPongFlag], L1B_LAYOUT, Arch::PositionL1{});
            auto tensorTileB = GetTile(tensorB, tla::MakeCoord(gBOffset % strideKV, gBOffset / strideKV), tla::MakeShape(kActual, nActual));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);
            copyGmToL1B(tensorL1B, tensorTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);

            uint32_t mL0Loop = CeilDiv<L0_TILE_M>(mActual);
            uint32_t kL0Loop = CeilDiv<L0_TILE_K>(kActual);
            for (uint32_t mL0Idx = 0; mL0Idx < mL0Loop; mL0Idx++) {
                uint32_t mL0Actual = (mL0Idx < mL0Loop - 1) ? L0_TILE_M : (mActual - mL0Idx * L0_TILE_M);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
                auto L0C_LAYOUT = tla::MakeLayoutL0C(mL0Actual, nActual);
                auto tensorL0C = tla::MakeTensor(l0CTensor[l0CPingPongFlag], L0C_LAYOUT, Arch::PositionL0C{});
                for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                    uint32_t kL0Actual = (kL0Idx < kL0Loop - 1) ? L0_TILE_K : (kActual - kL0Idx * L0_TILE_K);

                    auto L0A_LAYOUT = tla::MakeLayout<ElementA, LayoutTagL0A>(mL0Actual, kL0Actual);
                    auto tensorL0A = tla::MakeTensor(l0ATensor[l0ABPingPongFlag], L0A_LAYOUT, Arch::PositionL0A{});
                    auto tensorTileL1A = tla::MakeTensor(l1ATensor, L1A_LAYOUT, tla::MakeCoord(mL0Idx * L0_TILE_M, kL0Idx * L0_TILE_K),
                        Arch::PositionL1{});
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                    copyL1ToL0A(tensorL0A, tensorTileL1A);

                    auto L0B_LAYOUT = tla::MakeLayout<ElementB, LayoutTagL0B>(kL0Actual, nActual);
                    auto tensorL0B = tla::MakeTensor(l0BTensor[l0ABPingPongFlag], L0B_LAYOUT, Arch::PositionL0B{});
                    auto tensorTileL1B = tla::MakeTensor(l1BTensor[l1KvPingPongFlag], L1B_LAYOUT, tla::MakeCoord(kL0Idx * L0_TILE_K, 0),
                        Arch::PositionL1{});
                    if ((mL0Idx == 0) && (kL0Idx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);
                    }
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2);
                    copyL1ToL0B(tensorL0B, tensorTileL1B);

                    if ((mL0Idx == mL0Loop - 1) && (kL0Idx == kL0Loop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    bool initMmad = kL0Idx == 0;
                    tileMmad(tensorL0C,
                        tensorL0A,
                        tensorL0B,
                        mL0Actual,
                        nActual,
                        kL0Actual,
                        initMmad);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2);
                    l0ABPingPongFlag = 1 - l0ABPingPongFlag;
                }
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                auto tensorTileC = GetTile(tensorC, tla::MakeCoord(mL0Idx * L0_TILE_M, nL1Idx * L1_TILE_N),
                    tla::MakeShape(mL0Actual, nActual));
                copyL0CToDst(tensorTileC, tensorL0C);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
                l0CPingPongFlag = 1 - l0CPingPongFlag;
            }
            l1KvPingPongFlag = 1 - l1KvPingPongFlag;
        }
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensor;
    AscendC::LocalTensor<ElementB> l1BTensor[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[STAGES];

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;

    uint32_t l1KvPingPongFlag = 0;
    uint32_t l0CPingPongFlag = 0;
    uint32_t l0ABPingPongFlag = 0;
};

////////////////////////////////////////////////////////////////////

}  // namespace Catlass::Gemm::Block

#endif  // CATLASS_GEMM_BLOCK_MMAD_QK_SPLIT_ROW_TLA_HPP