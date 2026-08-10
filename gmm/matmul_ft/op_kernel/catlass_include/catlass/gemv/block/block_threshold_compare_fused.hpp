/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_BLOCK_BLOCK_THRESHOLD_COMPARE_FUSED_AIV_HPP
#define CATLASS_GEMV_BLOCK_BLOCK_THRESHOLD_COMPARE_FUSED_AIV_HPP

#include "../../catlass.hpp"
#include "../../arch/resource.hpp"
#include "../../coord.hpp"
#include "../../gemv_coord.hpp"
#include "../helper.hpp"
#include "../../gemm/helper.hpp"
#include "../../layout/layout.hpp"
#include "../../detail/alignment.hpp"
#include "../../gemm/dispatch_policy.hpp"
#include "../tile/tile_fault_compare.hpp"

namespace Vecself::Gemv::Block {

using namespace Catlass;

// class TileVmuls_

template <
    class UBTileShape_,
    class UBTileShapeTotal_,
    class AType_,
    class XType_,
    class YType_,
    class ZType_,
    class BiasType_,
    class TileCopy_,
    class TileThreCalc_,
    class TileVmuls_
>
struct BlockThresholdCalcFused <
    Gemm::GemvAtlasA2,
    Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM::AABFT,
    Vecself::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::NO_FUSED,
    Vecself::Gemv::helper::FT_ENC_TYPE::RCE,
    Vecself::Gemv::helper::FT_COMP_TYPE::RSUB,
    UBTileShape_,
    UBTileShapeTotal_,
    AType_,
    XType_,
    YType_,
    ZType_,
    BiasType_,
    TileCopy_,
    TileThreCalc_,
    TileVmuls_
> {
public:
    // Type Aliases
    using DispatchPolicy = Gemm::GemvAtlasA2;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using FT_COMP_TYPE = Vecself::Gemv::helper::FT_COMP_TYPE;

    using FT_THRESHOLD_ALGORITHM = Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM;
    using FT_AIV_PIPE_FUSE_TYPE = Vecself::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE;
    using UBTileShape = UBTileShape_;
    using UBTileShapeTotal = UBTileShapeTotal_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;
    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;

    using ElementZ = typename ZType_::Element;
    using LayoutZ = typename ZType_::Layout;

    using TileThreCalc = TileThreCalc_;
    using TileVmuls = TileVmuls_;
    using VecCopyGmToUb = typename TileCopy_::VecCopyGmToUb;
    using VecCopyUbToGm = typename TileCopy_::VecCopyUbToGm;
    using MatrixCopyGmToUb = typename TileCopy_::MatrixCopyGmToUb;
    using VecCopyGmToUbInX = typename TileCopy_::VecCopyGmToUbInX;
    using VecCopyGmToUbInY = typename TileCopy_::VecCopyGmToUbInY;
    using VecCopyUbToGmZ = typename TileCopy_::VecCopyUbToGmZ;


    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    using UBAlignHelper = Vecself::Gemv::helper::UBAlignHelper<ElementA>;
    using TensorCoord = layout::VectorLayout::TensorCoord;

    using ElementXoR = uint16_t;
    using ElementComp = int32_t;
    using ElementSub =  ElementY;

    using ElementWork = ElementY;

    using TileCompare = Vecself::Gemv::Tile::TileFaultVcompare<FT_COMP_TYPE::RSUB, ArchTag,
                                        ZType_, YType_, YType_>;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t Abuf_SIZE_ = 128 * 1024;
    static constexpr FT_COMP_TYPE COMP_TYPE = FT_COMP_TYPE::RSUB;
    // static constexpr uint32_t Xbuf_SIZE_ = 16 * 1024;
    static constexpr uint32_t Ybuf_SIZE_ = 12 * 1024;
    static constexpr uint32_t Zbuf_SIZE = 4 *1024;
    static constexpr uint32_t InXbuf_SIZE = 8 * 1024;
    static constexpr uint32_t InCbuf_SIZE = 8 * 1024;
    static constexpr uint32_t workspace_SIZE_ = 32 * 1024;

    static constexpr uint32_t ELE_NUM_PER_REPEAT = BYTE_PER_C0 * 8 / sizeof(ElementY);
    static constexpr FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE = FT_AIV_PIPE_FUSE_TYPE::NO_FUSED;
    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FT_THRESHOLD_ALGORITHM::AABFT;


    CATLASS_DEVICE
    BlockThresholdCalcFused() {}

    /// Construct
    CATLASS_DEVICE
    BlockThresholdCalcFused(Arch::Resource<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        // uint32_t UbXOffset = UBufAddrStart + Abuf_SIZE_;
        // + Xbuf_SIZE_ + Xbuf_SIZE_
        uint32_t UbYOffset = UBufAddrStart + Abuf_SIZE_;
        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_;
        uint32_t UbInXOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_ + workspace_SIZE_;
        uint32_t UbInYOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_ + workspace_SIZE_ + InXbuf_SIZE;
        uint32_t UbZOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_ + workspace_SIZE_ + InXbuf_SIZE + InCbuf_SIZE;

        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            // UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset + i * (Xbuf_SIZE_ / 2));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));
            UbWTensorList[i] =
                resource.ubBuf.template GetBufferByByte<ElementX>(UbWOffset + i * (workspace_SIZE_ / 2));

            UbInXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbInXOffset + i * (InXbuf_SIZE / 2));
            UbInCTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbInYOffset + i * (InCbuf_SIZE / 2));
            UbZTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementZ>(UbZOffset + i * (Zbuf_SIZE / 2));
            // Assign event ID for each stages
            UbInAEventList[i] = i;
            UbInXEventList[i] = i + STAGES;
            UbOutEventList[i] = i;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
        }
    }

    /// Construct
    CATLASS_DEVICE
    BlockThresholdCalcFused(Arch::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        // uint32_t UbXOffset = UBufAddrStart + Abuf_SIZE_;
        // + Xbuf_SIZE_ + Xbuf_SIZE_
        uint32_t UbYOffset = UBufAddrStart + Abuf_SIZE_;
        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_;
        uint32_t UbInXOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_ + workspace_SIZE_;
        uint32_t UbInYOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_ + workspace_SIZE_ + InXbuf_SIZE;
        uint32_t UbZOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_ + workspace_SIZE_ + InXbuf_SIZE + InCbuf_SIZE;

        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            // UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset + i * (Xbuf_SIZE_ / 2));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));
            UbWTensorList[i] =
                resource.ubBuf.template GetBufferByByte<ElementX>(UbWOffset + i * (workspace_SIZE_ / 2));

            UbInXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbInXOffset + i * (InXbuf_SIZE / 2));
            UbInCTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbInYOffset + i * (InCbuf_SIZE / 2));
            UbZTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementZ>(UbZOffset + i * (Zbuf_SIZE / 2));
            // Assign event ID for each stages
            UbInAEventList[i] = i;
            UbInXEventList[i] = i + STAGES;
            UbOutEventList[i] = i;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockThresholdCalcFused()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
        }
    }

    // AscendC::GlobalTensor<ElementX> const &gmX, LayoutX const &layoutX,

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmInX, LayoutY const &layoutInX,
        AscendC::GlobalTensor<ElementY> const &gmInC, LayoutY const &layoutInC,
        AscendC::GlobalTensor<ElementY> const &gmThreZ, LayoutY const &layoutY,
        AscendC::GlobalTensor<ElementZ> const &gmOutZ, LayoutZ const &layoutZ,
        GemvCoord const &actualShape, GemvCoord const &actualShapeTotal,
        ElementY alpha,bool outputThre)
    {
        TileMRound = RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);

        TileMTotalRound = RoundUp(UBTileShapeTotal::M, UBAlignHelper::ALIGN);
        TileNTotalRound = RoundUp(UBTileShapeTotal::N, UBAlignHelper::ALIGN);

        strideA = layoutA.stride(1) * TileNRound;
        strideARow = layoutA.stride(0) * TileMRound;
        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;

        m_actual_total = (actualShapeTotal.m() < TileMTotalRound) ? actualShapeTotal.m() : TileMTotalRound;
        n_actual_total = (actualShapeTotal.n() < TileNTotalRound) ? actualShapeTotal.n() : TileNTotalRound;

        uint32_t MLoop = (m_actual_total + TileMRound - 1) / TileMRound;

        dst_offset_ratio = MLoop;

        out_z_actual = (m_actual_total + 8 - 1)/ 8;
        // ElementY aim_weight = alpha * beta;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));

        AscendC::Duplicate<ElementY>(UbYTensorList[UbOutListId], (ElementY)0.0, m_actual_total);

        AscendC::PipeBarrier<PIPE_V>();

        // AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
        // AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));

        // AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListId]));
        // vecCopyGmToUb(UbXTensorList[UbInListId], gmX, n_actual);
        // AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListId]));

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
        auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUb(UbATensorList[UbInListId], gmA, layoutAInUb, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbOutListId]));
        vecCopyGmToUbInX(UbInXTensorList[UbOutListId], gmInX, m_actual_total);
        vecCopyGmToUbInY(UbInCTensorList[UbOutListId], gmInC, m_actual_total);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbOutListId]));

        // main loop
        uint32_t Nloop = CeilDiv(actualShape.n(), TileNRound);
        for(uint32_t mLoopIdx = 0; mLoopIdx < MLoop; mLoopIdx++) {
            m_actual = (mLoopIdx == MLoop - 1) ? (actualShapeTotal.m() - mLoopIdx * TileMRound) : TileMRound;
            auto OutYTile = UbYTensorList[UbOutListId][mLoopIdx * TileMRound];
            for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {
                n_actual = (LoopIdx == Nloop - 1) ? (actualShape.n() - LoopIdx * TileNRound) : TileNRound;
                y_actual = m_actual;
                x_actual = n_actual;

                uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
                if (LoopIdx < Nloop - 1) {
                    uint32_t LoopIdxNext = LoopIdx + 1;
                    uint32_t m_actual_next = m_actual;
                    uint32_t n_actual_next =
                        (LoopIdxNext == Nloop - 1) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;
                    uint32_t y_actual_next = m_actual_next;
                    uint32_t x_actual_next = n_actual_next;
                    // uint32_t row_start_next = mLoopIdx * TileMRound;
                    // Get L1 tensor for next stage
                    auto matrixTensor = UbATensorList[UbInListIdNext];

                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(matrixTensor, gmA[mLoopIdx * strideARow + LoopIdxNext * strideA], layoutAInUb, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }else if((LoopIdx == Nloop - 1) && (mLoopIdx < MLoop - 1)){
                    uint32_t LoopIdxNext = 0;
                    uint32_t mLoopIdxNext = mLoopIdx + 1;
                    uint32_t m_actual_next = (mLoopIdxNext == MLoop - 1) ? (actualShapeTotal.m() - mLoopIdxNext * TileMRound) : TileMRound;
                    uint32_t n_actual_next =
                        (LoopIdxNext == Nloop - 1) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;
                    uint32_t y_actual_next = m_actual_next;
                    uint32_t x_actual_next = n_actual_next;
                    // uint32_t row_start_next = mLoopIdx * TileMRound;
                    // Get L1 tensor for next stage
                    auto matrixTensor = UbATensorList[UbInListIdNext];

                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(matrixTensor, gmA[mLoopIdxNext * strideARow + LoopIdxNext * strideA], layoutAInUb, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
                auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));

                tileThreCalc(
                    OutYTile,
                    UbATensorList[UbInListId],
                    UbWTensorList[UbInListId],
                    layoutComputeInUb,
                    layoutTileCompute,dst_offset_ratio);

                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListId]));
                UbInListId = UbInListIdNext;
            }

            AscendC::PipeBarrier<PIPE_V>();
        }

        // UbYTensorList[UbOutListId]
        tileVmuls(UbYTensorList[UbOutListId], UbYTensorList[UbOutListId], (ElementY)alpha, m_actual_total);

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbOutListId]));

        /*
        // AscendC::LocalTensor<ElementWIn> workSpaceTensor,
        CATLASS_DEVICE
        void operator()(
            AscendC::LocalTensor<ElementZ> dstTensor,
            AscendC::LocalTensor<ElementX> srcTensor_x,
            AscendC::LocalTensor<ElementY> srcTensor_y,
            AscendC::LocalTensor<ElementX> srcTensor_thre,
            LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
            ElementX threshold
        )
        */
        // UbWTensorList[UbListId],

        auto layoutCompareInUb = layoutY.GetTileLayout(MakeCoord(TileMTotalRound));
        auto layoutTileCompare = layoutY.GetTileLayout(MakeCoord(m_actual_total));

        tileCompare(
            UbZTensorList[UbOutListId],
            UbInXTensorList[UbOutListId],
            UbInCTensorList[UbOutListId],
            UbYTensorList[UbOutListId],
            layoutCompareInUb, layoutTileCompare, (ElementY)0.002f);

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));

        // AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventList[UbOutListId]));
        // AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventList[UbOutListId]));

        if(outputThre){
            auto layoutDstYThre = layoutY.GetTileLayout(TensorCoord(m_actual_total));
            auto layoutComputeThreInUb = layoutY.GetTileLayout(TensorCoord(m_actual_total));
            vecCopyUbToGm(gmThreZ, UbYTensorList[UbOutListId], layoutDstYThre, layoutComputeThreInUb);
        }

        auto layoutDstZ = layoutZ.GetTileLayout(TensorCoord(out_z_actual));
        auto layoutComputeZInUb = layoutZ.GetTileLayout(TensorCoord(out_z_actual));
        vecCopyUbToGmZ(gmOutZ, UbZTensorList[UbOutListId], layoutDstZ, layoutComputeZInUb);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbOutListId]));

        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> UbATensorList[STAGES];
    // AscendC::LocalTensor<ElementX> UbXTensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorList[STAGES];
    AscendC::LocalTensor<ElementX> UbWTensorList[STAGES];

    AscendC::LocalTensor<ElementY> UbInXTensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbInCTensorList[STAGES];
    AscendC::LocalTensor<ElementZ> UbZTensorList[STAGES];

    // Multi-stage event id list
    int32_t UbInAEventList[STAGES];
    int32_t UbInXEventList[STAGES];
    int32_t UbOutEventList[STAGES];

    // The id of current stage
    uint32_t UbOutListId{0};
    uint32_t UbInListId{0};

    uint32_t m_actual, n_actual, x_actual, y_actual, out_z_actual, m_actual_total, n_actual_total;
    uint32_t TileMRound, TileNRound;
    uint32_t TileMTotalRound, TileNTotalRound;
    uint32_t strideA, strideARow;
    uint32_t dst_offset_ratio;

    // TileVmad tileVmad;
    TileThreCalc tileThreCalc;
    TileVmuls tileVmuls;
    MatrixCopyGmToUb matrixCopyGmToUb;
    VecCopyGmToUb vecCopyGmToUb;
    VecCopyUbToGm vecCopyUbToGm;
    VecCopyGmToUbInX vecCopyGmToUbInX;
    VecCopyGmToUbInY vecCopyGmToUbInY;
    VecCopyUbToGmZ vecCopyUbToGmZ;

    // Tile Compare
    TileCompare tileCompare;
};


/*
template <
    class DispatchPolicy,
    Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
    Vecself::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    Vecself::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    Vecself::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    class... Args
>
struct BlockThresholdCalcFused
*/
template <
    class UBTileShape_,
    class UBTileShapeTotal_,
    class L1TileShape_,
    class AType_,
    class XType_,
    class YType_,
    class ZType_,
    class BiasType_,
    class TileCopy_,
    class TileThreCalc_,
    class TileVmuls_
>
struct BlockThresholdCalcFused <
    Gemm::GemvAtlasA2,
    Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM::AABFT,
    Vecself::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::THRE_FUSED,
    Vecself::Gemv::helper::FT_ENC_TYPE::RCE,
    Vecself::Gemv::helper::FT_COMP_TYPE::RSUB,
    UBTileShape_,
    UBTileShapeTotal_,
    L1TileShape_,
    AType_,
    XType_,
    YType_,
    ZType_,
    BiasType_,
    TileCopy_,
    TileThreCalc_,
    TileVmuls_
> {
public:
    // Type Aliases
    using DispatchPolicy = Gemm::GemvAtlasA2;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using FT_COMP_TYPE = Vecself::Gemv::helper::FT_COMP_TYPE;
    using FT_AIV_PIPE_FUSE_TYPE = Vecself::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE;
    using FT_THRESHOLD_ALGORITHM = Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    using UBTileShape = UBTileShape_;
    using UBTileShapeTotal = UBTileShapeTotal_;
    using L1TileShape = L1TileShape_;


    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;
    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;

    using ElementZ = typename ZType_::Element;
    using LayoutZ = typename ZType_::Layout;

    using TileThreCalc = TileThreCalc_;
    using TileVmuls = TileVmuls_;
    using VecCopyGmToUb = typename TileCopy_::VecCopyGmToUb;
    using VecCopyUbToGm = typename TileCopy_::VecCopyUbToGm;
    using MatrixCopyGmToUb = typename TileCopy_::MatrixCopyGmToUb;
    using VecCopyGmToUbInX = typename TileCopy_::VecCopyGmToUbInX;
    using VecCopyGmToUbInY = typename TileCopy_::VecCopyGmToUbInY;
    using VecCopyUbToGmZ = typename TileCopy_::VecCopyUbToGmZ;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    using UBAlignHelper = Vecself::Gemv::helper::UBAlignHelper<ElementA>;
    using TensorCoord = layout::VectorLayout::TensorCoord;

    using ElementXoR = uint16_t;
    using ElementComp = int32_t;
    using ElementSub =  ElementY;

    using ElementWork = ElementY;

    using TileCompare = Vecself::Gemv::Tile::TileFaultVcompare<FT_COMP_TYPE::RSUB, ArchTag,
                                        ZType_, YType_, YType_>;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t Abuf_SIZE_ = 128 * 1024;
    static constexpr FT_COMP_TYPE COMP_TYPE = FT_COMP_TYPE::RSUB;
    static constexpr FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE = FT_AIV_PIPE_FUSE_TYPE::THRE_FUSED;
    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FT_THRESHOLD_ALGORITHM::AABFT;
    // static constexpr uint32_t Xbuf_SIZE_ = 16 * 1024;
    static constexpr uint32_t Ybuf_SIZE_ = 12 * 1024;
    static constexpr uint32_t Zbuf_SIZE = 4 *1024;
    static constexpr uint32_t InXbuf_SIZE = 8 * 1024;
    static constexpr uint32_t InCbuf_SIZE = 8 * 1024;
    static constexpr uint32_t workspace_SIZE_ = 32 * 1024;

    static constexpr uint32_t ELE_NUM_PER_REPEAT = BYTE_PER_C0 * 8 / sizeof(ElementY);

    static_assert(L1TileShape::M == UBTileShapeTotal::M,
        "The situation where the basic Tile of UB and L1 for MMA differ on the m axes is not supported yet");

    static_assert(L1TileShape::N == UBTileShapeTotal::N,
        "The situation where the basic Tile of UB and L1 for MMA differ on the n axes is not supported yet");

    static_assert(UBTileShapeTotal::N == UBTileShape::N,
        "The situation where the basic Tile of UB in tile and block differ on the n axes is not supported yet");

    CATLASS_DEVICE
    BlockThresholdCalcFused() {}

    /// Construct
    CATLASS_DEVICE
    BlockThresholdCalcFused(Arch::Resource<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        // uint32_t UbXOffset = UBufAddrStart + Abuf_SIZE_;
        // + Xbuf_SIZE_ + Xbuf_SIZE_
        uint32_t UbYOffset = UBufAddrStart + Abuf_SIZE_;
        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_;
        uint32_t UbInXOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_ + workspace_SIZE_;
        uint32_t UbInYOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_ + workspace_SIZE_ + InXbuf_SIZE;
        uint32_t UbZOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_ + workspace_SIZE_ + InXbuf_SIZE + InCbuf_SIZE;

        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            // UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset + i * (Xbuf_SIZE_ / 2));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));
            UbWTensorList[i] =
                resource.ubBuf.template GetBufferByByte<ElementX>(UbWOffset + i * (workspace_SIZE_ / 2));

            UbInXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbInXOffset + i * (InXbuf_SIZE / 2));
            UbInCTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbInYOffset + i * (InCbuf_SIZE / 2));
            UbZTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementZ>(UbZOffset + i * (Zbuf_SIZE / 2));
            // Assign event ID for each stages
            UbInAEventList[i] = i;
            UbInXEventList[i] = i + STAGES;
            UbOutEventList[i] = i;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
        }
    }

    /// Construct
    CATLASS_DEVICE
    BlockThresholdCalcFused(Arch::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        // uint32_t UbXOffset = UBufAddrStart + Abuf_SIZE_;
        // + Xbuf_SIZE_ + Xbuf_SIZE_
        uint32_t UbYOffset = UBufAddrStart + Abuf_SIZE_;
        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_;
        uint32_t UbInXOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_ + workspace_SIZE_;
        uint32_t UbInYOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_ + workspace_SIZE_ + InXbuf_SIZE;
        uint32_t UbZOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_ + workspace_SIZE_ + InXbuf_SIZE + InCbuf_SIZE;

        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            // UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset + i * (Xbuf_SIZE_ / 2));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));
            UbWTensorList[i] =
                resource.ubBuf.template GetBufferByByte<ElementX>(UbWOffset + i * (workspace_SIZE_ / 2));

            UbInXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbInXOffset + i * (InXbuf_SIZE / 2));
            UbInCTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbInYOffset + i * (InCbuf_SIZE / 2));
            UbZTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementZ>(UbZOffset + i * (Zbuf_SIZE / 2));
            // Assign event ID for each stages
            UbInAEventList[i] = i;
            UbInXEventList[i] = i + STAGES;
            UbOutEventList[i] = i;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockThresholdCalcFused()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
        }
    }

    // AscendC::GlobalTensor<ElementX> const &gmX, LayoutX const &layoutX,

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmInX, LayoutY const &layoutInX,
        AscendC::GlobalTensor<ElementY> const &gmInC, LayoutY const &layoutInC,
        AscendC::GlobalTensor<ElementY> const &gmThreZ, LayoutY const &layoutY,
        AscendC::GlobalTensor<ElementZ> const &gmOutZ, LayoutZ const &layoutZ,
        GemvCoord const &actualShape, GemvCoord const &actualShapeTotal,
        ElementY alpha,bool outputThre, uint32_t aiv_part_num=2)
    {
        TileMRound = RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);

        BlockMRound = RoundUp(UBTileShapeTotal::M, UBAlignHelper::ALIGN);
        BlockNRound = RoundUp(UBTileShapeTotal::N, UBAlignHelper::ALIGN);

        strideACol = layoutA.stride(1) * TileNRound;
        strideARow = layoutA.stride(0) * TileMRound;
        // m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        // n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;

        m_actual_total = (actualShapeTotal.m() < BlockMRound) ? actualShapeTotal.m() : BlockMRound;
        n_actual_total = (actualShapeTotal.n() < BlockNRound) ? actualShapeTotal.n() : BlockNRound;

        if(actualShape.n() < n_actual_total){
            n_actual_total = actualShape.n();
        }

        m_actual_part = m_actual_total / aiv_part_num;

        uint32_t M_start_offset = AscendC::GetSubBlockIdx() * m_actual_part;
        uint32_t Z_start_offset = (M_start_offset + 8 - 1) / 8;

        if(AscendC::GetSubBlockIdx() == (aiv_part_num -1)) {
            m_actual_part = m_actual_total - M_start_offset;
        }

        uint32_t MLoop = (m_actual_part + TileMRound - 1) / TileMRound;
        uint32_t Nloop = CeilDiv(n_actual_total, TileNRound);

        dst_offset_ratio = MLoop;

        out_z_actual_total = (m_actual_part + 8 - 1) / 8;

        for(uint32_t mLoopIdx = 0; mLoopIdx < MLoop; mLoopIdx++){
            m_actual = (mLoopIdx < (MLoop - 1)) ? TileMRound : m_actual_part - mLoopIdx * TileMRound;
            n_actual = (n_actual_total < TileNRound) ? n_actual_total : TileNRound;

            out_z_actual = (m_actual + 8 - 1) / 8;

            uint32_t mLoopOffset = mLoopIdx * TileMRound + M_start_offset;
            uint32_t mLoopOffset_for_z = (mLoopOffset + 8 - 1) / 8;
            uint32_t A_row_offset = mLoopOffset;
            uint32_t A_col_offset = 0;
            uint32_t A_block_offset = A_row_offset * layoutA.stride(0) + A_col_offset * layoutA.stride(1);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));
            AscendC::Duplicate<ElementY>(UbYTensorList[UbOutListId], (ElementY)0.0, m_actual);
            AscendC::PipeBarrier<PIPE_V>();

            auto OutYTile = UbYTensorList[UbOutListId];

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbOutListId]));
            vecCopyGmToUbInX(UbInXTensorList[UbOutListId], gmInX[mLoopOffset], m_actual);
            vecCopyGmToUbInY(UbInCTensorList[UbOutListId], gmInC[mLoopOffset], m_actual);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbOutListId]));

            if(mLoopIdx == 0){
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
                matrixCopyGmToUb(UbATensorList[UbInListId], gmA[A_block_offset], layoutAInUb, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            }

            for (uint32_t nLoopIdx = 0; nLoopIdx < Nloop; nLoopIdx++) {
                n_actual = (nLoopIdx == Nloop - 1) ? (n_actual_total - nLoopIdx * TileNRound) : TileNRound;
                y_actual = m_actual;
                x_actual = n_actual;

                uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
                if (nLoopIdx < Nloop - 1) {
                    uint32_t nLoopIdxNext = nLoopIdx + 1;
                    uint32_t m_actual_next = m_actual;
                    uint32_t n_actual_next =
                        (nLoopIdxNext == Nloop - 1) ? (n_actual_total - nLoopIdxNext * TileNRound) : TileNRound;
                    uint32_t y_actual_next = m_actual_next;
                    uint32_t x_actual_next = n_actual_next;
                    // uint32_t row_start_next = mLoopIdx * TileMRound;
                    // Get L1 tensor for next stage
                    auto matrixTensor = UbATensorList[UbInListIdNext];

                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(matrixTensor, gmA[A_block_offset + nLoopIdxNext * strideACol], layoutAInUb, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }else if((nLoopIdx == Nloop - 1) && (mLoopIdx < MLoop - 1)){
                    uint32_t nLoopIdxNext = 0;
                    uint32_t mLoopIdxNext = mLoopIdx + 1;
                    uint32_t m_actual_next = (mLoopIdxNext == MLoop - 1) ? (m_actual_part - mLoopIdxNext * TileMRound) : TileMRound;
                    uint32_t n_actual_next =
                        (nLoopIdxNext == Nloop - 1) ? (n_actual_total - nLoopIdxNext * TileNRound) : TileNRound;
                    uint32_t y_actual_next = m_actual_next;
                    uint32_t x_actual_next = n_actual_next;
                    uint32_t A_block_offset_next = A_block_offset + strideARow;
                    // uint32_t row_start_next = mLoopIdx * TileMRound;
                    // Get L1 tensor for next stage
                    auto matrixTensor = UbATensorList[UbInListIdNext];

                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(matrixTensor, gmA[A_block_offset_next + nLoopIdxNext * strideACol], layoutAInUb, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
                auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));

                tileThreCalc(
                    OutYTile,
                    UbATensorList[UbInListId],
                    UbWTensorList[UbInListId],
                    layoutComputeInUb,
                    layoutTileCompute,dst_offset_ratio);

                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListId]));
                UbInListId = UbInListIdNext;
            }
            AscendC::PipeBarrier<PIPE_V>();

            // UbYTensorList[UbOutListId]
            tileVmuls(UbYTensorList[UbOutListId], UbYTensorList[UbOutListId], (ElementY)alpha, m_actual);

            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbOutListId]));

            auto layoutCompareInUb = layoutY.GetTileLayout(MakeCoord(TileMRound));
            auto layoutTileCompare = layoutY.GetTileLayout(MakeCoord(m_actual));

            tileCompare(
                UbZTensorList[UbOutListId],
                UbInXTensorList[UbOutListId],
                UbInCTensorList[UbOutListId],
                UbYTensorList[UbOutListId],
                layoutCompareInUb,
                layoutTileCompare,
                (ElementY)0.002f
            );

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));

            if(outputThre){
                auto layoutDstYThre = layoutY.GetTileLayout(TensorCoord(m_actual));
                auto layoutComputeThreInUb = layoutY.GetTileLayout(TensorCoord(m_actual));
                vecCopyUbToGm(gmThreZ[mLoopOffset],
                    UbYTensorList[UbOutListId],
                    layoutDstYThre,
                    layoutComputeThreInUb);
            }

            auto layoutDstZ = layoutZ.GetTileLayout(TensorCoord(out_z_actual));
            auto layoutComputeZInUb = layoutZ.GetTileLayout(TensorCoord(out_z_actual));
            vecCopyUbToGmZ(gmOutZ[mLoopOffset_for_z], UbZTensorList[UbOutListId], layoutDstZ, layoutComputeZInUb);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbOutListId]));

            UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
        }
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> UbATensorList[STAGES];
    // AscendC::LocalTensor<ElementX> UbXTensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorList[STAGES];
    AscendC::LocalTensor<ElementX> UbWTensorList[STAGES];

    AscendC::LocalTensor<ElementY> UbInXTensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbInCTensorList[STAGES];
    AscendC::LocalTensor<ElementZ> UbZTensorList[STAGES];

    // Multi-stage event id list
    int32_t UbInAEventList[STAGES];
    int32_t UbInXEventList[STAGES];
    int32_t UbOutEventList[STAGES];

    // The id of current stage
    uint32_t UbOutListId{0};
    uint32_t UbInListId{0};

    uint32_t m_actual, n_actual, x_actual, y_actual, out_z_actual, m_actual_total, n_actual_total;
    uint32_t m_actual_part;
    uint32_t out_z_actual_total;
    uint32_t TileMRound, TileNRound;
    uint32_t BlockMRound, BlockNRound;
    uint32_t strideACol, strideARow;
    uint32_t dst_offset_ratio;

    // TileVmad tileVmad;
    TileThreCalc tileThreCalc;
    TileVmuls tileVmuls;
    MatrixCopyGmToUb matrixCopyGmToUb;
    VecCopyGmToUb vecCopyGmToUb;
    VecCopyUbToGm vecCopyUbToGm;
    VecCopyGmToUbInX vecCopyGmToUbInX;
    VecCopyGmToUbInY vecCopyGmToUbInY;
    VecCopyUbToGmZ vecCopyUbToGmZ;

    // Tile Compare
    TileCompare tileCompare;
};




} // namespace Vecself::Gemv::Block

#endif // CATLASS_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
