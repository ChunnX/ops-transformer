/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_MMAD_HSTU_PV_SPLIT_ROW_TLA_HPP
#define CATLASS_GEMM_BLOCK_MMAD_HSTU_PV_SPLIT_ROW_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
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
    MmadHSTUPVSplitRow<ArchTag_, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>,
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
    using DispatchPolicy = MmadHSTUPVSplitRow<ArchTag_, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>;
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
    static_assert(L1A_TILE_SIZE * STAGES + L1B_TILE_SIZE * STAGES <= ArchTag::L1_SIZE,
        "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static_assert(L0A_TILE_SIZE * STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE * STAGES <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");

    /// Construct
    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {
        // Allocate L1 memory space
        for (uint32_t i = 0; i < STAGES; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart + L1A_TILE_SIZE * i);
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1A_TILE_SIZE * 2 +
                                                                             L1B_TILE_SIZE * i);
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmadTla() {}

    template <class TensorA, class TensorB, class TensorC>
    CATLASS_DEVICE
    void operator()(TensorA &tensorA, TensorB &tensorB, TensorC &tensorC, GemmCoord actualOriShape,
        uint32_t kvHeads, bool isNTD)
    {
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorA>;
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<TensorB>;
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorC>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
        CopyL0CToDst copyL0CToDst;
        
        uint32_t mL1Actual = actualOriShape[0];
        uint32_t embed = actualOriShape[1];
        uint32_t actualKvSeqlen = actualOriShape[2];
        uint32_t kL1Loop = CeilDiv<L1_TILE_K>(actualKvSeqlen);

        for (uint32_t kL1Idx = 0; kL1Idx < kL1Loop; kL1Idx++) {
            uint32_t kL1Actual = kL1Idx < kL1Loop - 1 ? L1_TILE_K : actualKvSeqlen - kL1Idx * L1_TILE_K;
            auto L1A_LAYOUT = tla::MakeLayout<ElementA, LayoutTagL1A>(mL1Actual, kL1Actual);
            auto tensorL1A = tla::MakeTensor(l1ATensor[l1PPingPongFlag], L1A_LAYOUT, Arch::PositionL1{});
            auto tensorTileA = GetTile(tensorA, tla::MakeCoord(0, kL1Idx * L1_TILE_K),
                tla::MakeShape(mL1Actual, kL1Actual));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1PPingPongFlag + 2);
            copyGmToL1A(tensorL1A, tensorTileA);

            auto L1B_LAYOUT = tla::MakeLayout<ElementB, LayoutTagL1B>(kL1Actual, embed);
            auto tensorL1B = tla::MakeTensor(l1BTensor[l1PPingPongFlag], L1B_LAYOUT, Arch::PositionL1{});
            auto tensorTileB = GetTile(tensorB, tla::MakeCoord(kL1Idx * L1_TILE_K * (isNTD ? 1 : kvHeads), 0),
                tla::MakeShape(kL1Actual, embed));
            if (isNTD) {
                copyGmToL1B(tensorL1B, tensorTileB);
            } else {
                CopyGmToL1(tensorL1B, tensorTileB, embed * kvHeads);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1PPingPongFlag + 2);
            
            uint32_t mL0Loop = CeilDiv<L0_TILE_M>(mL1Actual);
            uint32_t kL0Loop = CeilDiv<L0_TILE_K>(kL1Actual);
            for (uint32_t mL0Idx = 0; mL0Idx < mL0Loop; mL0Idx++) {
                uint32_t mL0Actual = (mL0Idx < mL0Loop - 1) ? L0_TILE_M : (mL1Actual - mL0Idx * L0_TILE_M);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag + 2);
                auto L0C_LAYOUT = tla::MakeLayoutL0C(mL0Actual, embed);
                auto tensorL0C = tla::MakeTensor(l0CTensor[l0CPingPongFlag], L0C_LAYOUT, Arch::PositionL0C{});
                for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                    uint32_t kL0Actual =
                        (kL0Idx < kL0Loop - 1) ? L0_TILE_K : (kL1Actual - kL0Idx * L0_TILE_K);
                    auto L0A_LAYOUT = tla::MakeLayout<ElementA, LayoutTagL0A>(mL0Actual, kL0Actual);
                    auto tensorL0A = tla::MakeTensor(l0ATensor[l0ABPingPongFlag], L0A_LAYOUT, Arch::PositionL0A{});
                    auto tensorTileL1A = GetTile(tensorL1A,
                        tla::MakeCoord(mL0Idx * L0_TILE_M, kL0Idx * L0_TILE_K), tla::MakeShape(mL0Actual, kL0Actual));

                    if ((mL0Idx == 0) && (kL0Idx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1PPingPongFlag + 2);
                    }
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                    copyL1ToL0A(tensorL0A, tensorTileL1A);

                    auto L0B_LAYOUT = tla::MakeLayout<ElementB, LayoutTagL0B>(kL0Actual, embed);
                    auto tensorL0B = tla::MakeTensor(l0BTensor[l0ABPingPongFlag], L0B_LAYOUT, Arch::PositionL0B{});
                    auto tensorTileL1B = GetTile(tensorL1B,
                        tla::MakeCoord(kL1Idx * L1_TILE_K + kL0Idx * L0_TILE_K, 0), tla::MakeShape(kL0Actual, embed));
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2);
                    CopyL1ToL0b(tensorL0B, tensorTileL1B, true);
                    if ((mL0Idx == mL0Loop - 1) && (kL0Idx == kL0Loop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1PPingPongFlag + 2);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ABPingPongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ABPingPongFlag);
                    bool initMmad = (kL0Idx == 0);
                    tileMmad(tensorL0C,
                        tensorL0A,
                        tensorL0B,
                        mL0Actual,
                        embed,
                        kL0Actual,
                        initMmad);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2);
                    l0ABPingPongFlag = 1 - l0ABPingPongFlag;
                }
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CPingPongFlag + 2);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CPingPongFlag + 2);
                // gm上累加设置
                AscendC::SetAtomicAdd<ElementC>();
                auto tensorTileC = GetTile(tensorC, tla::MakeCoord(mL0Idx * L0_TILE_M, 0),
                                           tla::MakeShape(mL0Actual, embed));
                copyL0CToDst(tensorTileC, tensorL0C);
                AscendC::SetAtomicNone();
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag + 2);
                l0CPingPongFlag = 1 - l0CPingPongFlag;
            }
            l1PPingPongFlag = 1 - l1PPingPongFlag;
        }
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[STAGES];

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;

    uint32_t l1PPingPongFlag = 0;
    uint32_t l0CPingPongFlag = 0;
    uint32_t l0ABPingPongFlag = 0;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE
    void CopyGmToL1(
        TensorDst const &dstTensor,
        TensorSrc const &srcTensor,
        uint32_t srcDValue,
        uint32_t ndNum = 1,
        uint32_t srcNdMatrixStride = 0,
        uint32_t dstNzMatrixStride = 0
    )
    {
        const uint32_t ELE_NUM_PER_C0 = 16;

        const uint32_t nValue = tla::get<0>(srcTensor.shape());
        const uint32_t dValue = tla::get<1>(srcTensor.shape());
        const uint32_t dstInnerStrideRow = tla::get<0, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = ndNum;
        intriParams.nValue = nValue;
        intriParams.dValue = dValue;
        intriParams.srcNdMatrixStride = srcNdMatrixStride;
        intriParams.srcDValue = srcDValue;
        intriParams.dstNzC0Stride = dstOuterStrideCol / ELE_NUM_PER_C0;
        intriParams.dstNzNStride = dstInnerStrideRow / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = dstNzMatrixStride;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE
    void CopyL1ToL0b(
        TensorDst const &dstTensor,
        TensorSrc const &srcTensor,
        bool isTranspose
    )
    {
        const uint32_t ELE_NUM_PER_C0 = 16;
        const uint32_t C0_NUM_PER_FRACTAL = 256;
        const uint32_t dstOuterShapeRow = tla::get<0, 1>(dstTensor.shape());
        const uint32_t dstOuterShapeCol = tla::get<1, 1>(dstTensor.shape());
        auto dstOffset = dstTensor.layout()(dstTensor.coord());

        if (isTranspose) {
            for(uint32_t i = 0; i < dstOuterShapeRow; i++) {
                AscendC::LoadData2DParams loadDataParams;
                loadDataParams.repeatTimes = dstOuterShapeCol;
                loadDataParams.srcStride = dstOuterShapeRow;
                loadDataParams.dstGap = 0;
                loadDataParams.ifTranspose = isTranspose;
                AscendC::LoadData(dstTensor.data()[dstOffset +
                    i * dstOuterShapeCol * ELE_NUM_PER_C0 * ELE_NUM_PER_C0],
                    srcTensor.data()[i * C0_NUM_PER_FRACTAL], loadDataParams);
            }
        } else {
                AscendC::LoadData2DParams loadDataParams;
                loadDataParams.repeatTimes = dstOuterShapeRow * dstOuterShapeCol;
                loadDataParams.srcStride = 1;
                loadDataParams.dstGap = 0;
                loadDataParams.ifTranspose = isTranspose;
                AscendC::LoadData(dstTensor.data()[dstOffset], srcTensor.data(), loadDataParams);
        }
    }
};

////////////////////////////////////////////////////////////////////

}  // namespace Catlass::Gemm::Block

#endif  // CATLASS_GEMM_BLOCK_MMAD_HSTU_PV_SPLIT_ROW_TLA_HPP