/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_BLOCK_BLOCK_SLICEKMN_RED_SUM_FUSED_HPP
#define CATLASS_GEMV_BLOCK_BLOCK_SLICEKMN_RED_SUM_FUSED_HPP

#include "../../catlass.hpp"
#include "catlass/arch/resource_aiv.hpp"
#include "../../coord.hpp"
#include "../../gemv_coord.hpp"
#include "../helper.hpp"
#include "../../gemm/helper.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/alignment.hpp"
#include "../../gemm/dispatch_policy.hpp"
#include "../../matrix_coord.hpp"

namespace Vecself::Gemv::Block {

using namespace Catlass;

template <
    class UBTileShapeforA_,
    class UBTileShapeforB_,
    class AType_,
    class BType_,
    class XType_,
    class YType_,
    class BiasType_,
    class TileCopyforA_,
    class TileCopyforB_,
    class TileMatrixAdd_,
    class TileFaultSum_,
    class TileVmuls_
>
struct BlockSliceKMNSum <
    Gemm::GemvAtlasA2,
    Vecself::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_MIXED,
    UBTileShapeforA_,
    UBTileShapeforB_,
    AType_,
    BType_,
    XType_,
    YType_,
    BiasType_,
    TileCopyforA_,
    TileCopyforB_,
    TileMatrixAdd_,
    TileFaultSum_,
    TileVmuls_
> {
public:
    // Type Aliases
    using DispatchPolicy = Gemm::GemvAtlasA2;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using UBTileShapeforA = UBTileShapeforA_;
    using UBTileShapeforB = UBTileShapeforB_;

    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;

    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;

    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;

    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;
    using TileMatrixAdd = TileMatrixAdd_;

    using TileFaultSum = TileFaultSum_;
    using TileVmuls = TileVmuls_;

    using FT_THRESHOLD_ALGORITHM = Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM;
    using FT_REDUCE_TYPE = Vecself::Gemv::helper::FT_REDUCE_TYPE;

    using TileFaultMeanforB = Vecself::Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::SUM, BType_, XType_>;
    using TileFaultMaxforB = Vecself::Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::MAX, BType_, XType_>;

    using MatrixCopyGmToUbforA = typename TileCopyforA_::MatrixCopyGmToUb;
    using MatrixCopyUbToGmforA = typename TileCopyforA_::MatrixCopyUbToGm;

    using VecCopyGmToUbforB = typename TileCopyforB_::VecCopyGmToUb;
    using VecCopyUbToGmforB = typename TileCopyforB_::VecCopyUbToGm;
    using MatrixCopyGmToUbforB = typename TileCopyforB_::MatrixCopyGmToUb;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using UBAlignHelper = Vecself::Gemv::helper::UBAlignHelper<ElementA>;
    using TensorCoord = layout::VectorLayout::TensorCoord;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;

    static constexpr uint32_t Abuf_SIZE_ = 64 * 1024;
    // static constexpr uint32_t Xbuf_SIZE_ = 16 * 1024;
    static constexpr uint32_t workspace_SIZE_ = 32 * 1024;
    static constexpr uint32_t Ybuf_SIZE_for_B_ = 16 * 1024;
    static constexpr uint32_t Ybuf_SIZE_for_A_ = 64 * 1024;

    // static constexpr uint32_t Abuf_SIZE_ = 128 * 1024;
    // static constexpr uint32_t Xbuf_SIZE_ = 16 * 1024;


    CATLASS_DEVICE
    BlockSliceKMNSum() {}

    /// Construct
    CATLASS_DEVICE
    BlockSliceKMNSum(Arch::Resource<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbYOffsetforA = UBufAddrStart + Abuf_SIZE_;
        uint32_t UbYOffsetforB = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_for_A_;
        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_for_A_ + Ybuf_SIZE_for_B_;

        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbBTensorList[i] = UbATensorList[i].template ReinterpretCast<ElementB>();

            UbYTensorforAList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforA + i * (Ybuf_SIZE_for_A_ / 2));
            UbYTensorforBList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforB + i * (Ybuf_SIZE_for_B_ / 2));
            UbWTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementB>(UbWOffset + i * (workspace_SIZE_ / 2));

            // Assign event ID for each stages
            UbInAEventList[i] = i;
            UbOutEventforAList[i] = i;
            UbOutEventforBList[i] = i + STAGES;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventforAList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventforAList[i]);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventforBList[i]);
        }
    }

    /// Construct
    CATLASS_DEVICE
    BlockSliceKMNSum(FTSelf::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbYOffsetforA = UBufAddrStart + Abuf_SIZE_;
        uint32_t UbYOffsetforB = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_for_A_;
        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_for_A_ + Ybuf_SIZE_for_B_;

        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbBTensorList[i] = UbATensorList[i].template ReinterpretCast<ElementB>();

            UbYTensorforAList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforA + i * (Ybuf_SIZE_for_A_ / 2));
            UbYTensorforBList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforB + i * (Ybuf_SIZE_for_B_ / 2));
            UbWTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementB>(UbWOffset + i * (workspace_SIZE_ / 2));

            // Assign event ID for each stages
            UbInAEventList[i] = i;
            UbOutEventforAList[i] = i;
            UbOutEventforBList[i] = i + STAGES;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventforAList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventforAList[i]);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventforBList[i]);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockSliceKMNSum()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutEventforAList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventforAList[i]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutEventforBList[i]);

        }
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        GemvCoord const &actualShape, uint32_t NRealRound, uint32_t KFTRoundCount)
    {

        TileMRound = RoundUp(UBTileShapeforA::M, UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShapeforA::N, UBAlignHelper::ALIGN);
        // RoundUp(NRealRound, UBAlignHelper::ALIGN);
        strideA = layoutA.stride(1) * TileNRound;
        strideY = layoutY.stride(1) * TileNRound;
        strideSlice = layoutA.shape(0) * layoutA.shape(1);

        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;

        // main loop
        uint32_t Nloop = CeilDiv(actualShape.n(), TileNRound);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(UbOutEventforAList[UbOutListId]));
        auto UbYTensor = UbYTensorforAList[UbOutListId];

        auto layoutYInUb = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(TileMRound, TileNRound));
        auto layoutTileY = layoutY.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUbforA(UbYTensor, gmY, layoutYInUb, layoutTileY);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventforAList[UbOutListId]));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>((event_t)(UbOutEventforAList[UbOutListId]));
        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {

            m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
            n_actual = (LoopIdx == Nloop - 1) ? (actualShape.n() - LoopIdx * TileNRound) : TileNRound;
            y_actual = m_actual;
            x_actual = n_actual;

            uint32_t Y_block_offset = LoopIdx * strideY;
            uint32_t A_block_offset = LoopIdx * strideA;

            uint32_t UbOutListIdNext = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
            if (LoopIdx < Nloop - 1) {
                uint32_t LoopIdxNext = LoopIdx + 1;
                uint32_t m_actual_next = m_actual;
                uint32_t n_actual_next =
                    (LoopIdxNext == Nloop - 1) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;
                uint32_t y_actual_next = m_actual_next;
                uint32_t x_actual_next = n_actual_next;
                // Get L1 tensor for next stage
                auto UbYTensorNext = UbYTensorforAList[UbOutListIdNext];
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(UbOutEventforAList[UbOutListIdNext]));
                auto layoutYInUbNext = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(TileMRound, TileNRound));
                auto layoutTileYNext = layoutY.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                matrixCopyGmToUbforA(UbYTensorNext, gmY[LoopIdxNext * strideY], layoutYInUbNext, layoutTileYNext);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventforAList[UbOutListIdNext]));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>((event_t)(UbOutEventforAList[UbOutListIdNext]));
            }

            if(LoopIdx == 0){
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
                matrixCopyGmToUbforA(UbATensorList[UbInListId], gmA[LoopIdx * strideA], layoutAInUb, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventforAList[UbOutListId]));
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutEventforAList[UbOutListId]);

            auto UbYIterTensor = UbYTensorforAList[UbOutListId];

            for(uint32_t kFTRoundIdx = 0; kFTRoundIdx < KFTRoundCount; kFTRoundIdx++){
                uint32_t kOffset = kFTRoundIdx * strideSlice;
                uint32_t kFTRoundIdxNext = kFTRoundIdx + 1;
                uint32_t kOffsetNext = kFTRoundIdxNext * strideSlice;
                uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
                if(kFTRoundIdx < KFTRoundCount - 1){
                    // Preload next K FT round data
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    auto layoutAInUbNext = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
                    matrixCopyGmToUbforA(UbATensorList[UbInListIdNext], gmA[LoopIdx * strideA + kOffsetNext], layoutAInUbNext, layoutTileANext);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }else if(LoopIdx < (Nloop - 1)){
                    uint32_t LoopIdxNext = LoopIdx + 1;
                    uint32_t kFTRoundIdxNext = 0;
                    uint32_t kOffsetNext = kFTRoundIdxNext * strideSlice;
                    // Preload next N loop first K FT round data
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    auto layoutAInUbNext = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    uint32_t m_actual_next = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
                    uint32_t n_actual_next = (LoopIdxNext == Nloop - 1) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;
                    // auto layoutAInUbNext = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUbforA(UbATensorList[UbInListIdNext], gmA[LoopIdxNext * strideA + kOffsetNext], layoutAInUbNext, layoutTileANext);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
                auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));

                auto UbATensor = UbATensorList[UbInListId];
                tileMatrixAdd(UbYIterTensor,UbATensor,UbYIterTensor,layoutComputeInUb,layoutTileCompute);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                UbInListId = UbInListIdNext;
            }

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventforAList[UbOutListId]));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventforAList[UbOutListId]));

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(UbOutEventforAList[UbOutListId]);

            auto layoutYInUb = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(m_actual, TileNRound));
            auto layoutDstY = layoutY.GetTileLayout(MakeCoord(m_actual, n_actual));

            matrixCopyUbToGmforA(gmY[Y_block_offset], UbYTensorforAList[UbOutListId], layoutDstY, layoutYInUb);
            // AscendC::PipeBarrier<PIPE_MTE3>();

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventforAList[UbOutListId]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventforAList[UbOutListId]);

            UbOutListId = UbOutListIdNext;
        }
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    CATLASS_DEVICE
    void RowMean(
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementY> const &gmZ, LayoutX const &layoutX,
        GemvCoord const &actualShape,
        uint32_t NRealRound,
        float kn_scale_ratio)
    {

        TileMRound = RoundUp(UBTileShapeforB::M, UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShapeforB::N, UBAlignHelper::ALIGN);
        // RoundUp(NRealRound, UBAlignHelper::ALIGN);
        strideB = layoutB.stride(1) * TileNRound;
        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventforBList[UbOutListId]));

        auto UbYTensor = UbYTensorforBList[UbOutListId];

        AscendC::Duplicate<ElementY>(UbYTensor, (ElementY)0.0, m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventforBList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventforBList[UbOutListId]));

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
        auto layoutBInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUbforB(UbBTensorList[UbInListId], gmB, layoutBInUb, layoutTileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
        // main loop
        uint32_t Nloop = CeilDiv(actualShape.n(), TileNRound);
        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {
            m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
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
                // Get L1 tensor for next stage
                auto matrixTensor = UbBTensorList[UbInListIdNext];

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                auto layoutBInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                matrixCopyGmToUbforB(matrixTensor, gmB[LoopIdxNext * strideB], layoutBInUb, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            auto layoutComputeInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            auto layoutTileCompute = layoutB.GetTileLayout(MakeCoord(m_actual, n_actual));

            /*
            void operator()(
            AscendC::LocalTensor<ElementY> dstTensor,
            AscendC::LocalTensor<ElementA> srcTensor_m,
            AscendC::LocalTensor<ElementAccumulator> temp,
            LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
            )
            */
            tileFaultMeanforB(UbYTensorforBList[UbOutListId],
                UbBTensorList[UbInListId],
                UbWTensorList[UbInListId],
                layoutComputeInUb,
                layoutTileCompute);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            UbInListId = UbInListIdNext;
        }
        AscendC::PipeBarrier<PIPE_V>();

        tileVmuls(UbYTensorforBList[UbOutListId], UbYTensorforBList[UbOutListId], (ElementY)kn_scale_ratio, m_actual);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventforBList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventforBList[UbOutListId]));
        auto layoutDstY = layoutX.GetTileLayout(TensorCoord(y_actual));
        auto layoutComputeInUb = layoutX.GetTileLayout(TensorCoord(y_actual));
        vecCopyUbToGmforB(gmZ, UbYTensorforBList[UbOutListId], layoutDstY, layoutComputeInUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventforBList[UbOutListId]));
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    CATLASS_DEVICE
    void RowMax(
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementY> const &gmZ, LayoutX const &layoutX,
        GemvCoord const &actualShape,
        uint32_t NRealRound,
        float kn_scale_ratio)
    {

        TileMRound = RoundUp(UBTileShapeforB::M, UBAlignHelper::ALIGN);
        TileNRound = NRealRound;
        // RoundUp(NRealRound, UBAlignHelper::ALIGN);
        strideB = layoutB.stride(1) * TileNRound;
        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventforBList[UbOutListId]));

        auto UbYTensor = UbYTensorforBList[UbOutListId];

        AscendC::Duplicate<ElementY>(UbYTensor, (ElementY)0.0, m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventforBList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventforBList[UbOutListId]));

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
        auto layoutBInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUbforB(UbBTensorList[UbInListId], gmB, layoutBInUb, layoutTileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
        // main loop
        uint32_t Nloop = CeilDiv(actualShape.n(), TileNRound);
        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {
            m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
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
                // Get L1 tensor for next stage
                auto matrixTensor = UbBTensorList[UbInListIdNext];

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                auto layoutBInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                matrixCopyGmToUbforB(matrixTensor, gmB[LoopIdxNext * strideB], layoutBInUb, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            auto layoutComputeInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            auto layoutTileCompute = layoutB.GetTileLayout(MakeCoord(m_actual, n_actual));

            /*
            void operator()(
            AscendC::LocalTensor<ElementY> dstTensor,
            AscendC::LocalTensor<ElementA> srcTensor_m,
            AscendC::LocalTensor<ElementAccumulator> temp,
            LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
            )
            */
            tileFaultMaxforB(UbYTensorforBList[UbOutListId],
                UbBTensorList[UbInListId],
                UbWTensorList[UbInListId],
                layoutComputeInUb,
                layoutTileCompute);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            UbInListId = UbInListIdNext;
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventforBList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventforBList[UbOutListId]));
        auto layoutDstY = layoutX.GetTileLayout(TensorCoord(y_actual));
        auto layoutComputeInUb = layoutX.GetTileLayout(TensorCoord(y_actual));
        vecCopyUbToGmforB(gmZ, UbYTensorforBList[UbOutListId], layoutDstY, layoutComputeInUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventforBList[UbOutListId]));
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> UbATensorList[STAGES];
    AscendC::LocalTensor<ElementB> UbBTensorList[STAGES];
    // AscendC::LocalTensor<ElementX> UbXTensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorforAList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorforBList[STAGES];
    AscendC::LocalTensor<ElementB> UbWTensorList[STAGES];
    // AscendC::LocalTensor<ElementA> UbWTensorList[STAGES];

    // Multi-stage event id list
    int32_t UbInAEventList[STAGES];
    // int32_t UbInXEventList[STAGES];
    int32_t UbOutEventforAList[STAGES];
    int32_t UbOutEventforBList[STAGES];

    // The id of current stage
    uint32_t UbOutListId{0};
    uint32_t UbInListId{0};

    uint32_t m_actual, n_actual, x_actual, y_actual;
    uint32_t TileMRound, TileNRound;
    uint32_t strideA, strideY, strideB;
    uint32_t strideSlice;

    TileMatrixAdd tileMatrixAdd;

    TileFaultMeanforB tileFaultMeanforB;
    TileFaultMaxforB tileFaultMaxforB;
    TileVmuls tileVmuls;

    MatrixCopyGmToUbforB matrixCopyGmToUbforB;
    VecCopyGmToUbforB vecCopyGmToUbforB;
    VecCopyUbToGmforB vecCopyUbToGmforB;

    MatrixCopyGmToUbforA matrixCopyGmToUbforA;
    MatrixCopyUbToGmforA matrixCopyUbToGmforA;

};

} // namespace Vecself::Gemv::Block

#endif // CATLASS_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
