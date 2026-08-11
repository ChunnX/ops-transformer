/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_BLOCK_BLOCK_SLICEKMN_RED_SUM_HPP
#define CATLASS_GEMV_BLOCK_BLOCK_SLICEKMN_RED_SUM_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource_aiv.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/matrix_coord.hpp"

namespace FTSelf::Gemv::Block {

using namespace Catlass;

template <
    class UBTileShape_,
    class AType_,
    class YType_,
    class BiasType_,
    class TileCopy_,
    class TileMatrixAdd_
>
struct BlockSliceKMNSum <
    Catlass::Gemm::GemvAtlasA2,
    FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::ABE_FUSED_THRE,
    UBTileShape_,
    AType_,
    YType_,
    BiasType_,
    TileCopy_,
    TileMatrixAdd_
> {
public:
    // Type Aliases
    using DispatchPolicy = Catlass::Gemm::GemvAtlasA2;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using UBTileShape = UBTileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;

    using ElementX = typename AType_::Element;
    using LayoutX = Catlass::layout::VectorLayout;

    using TileMatrixAdd = TileMatrixAdd_;

    using MatrixCopyGmToUb = typename TileCopy_::MatrixCopyGmToUb;
    using MatrixCopyUbToGm = typename TileCopy_::MatrixCopyUbToGm;
    using VecCopyGmToUbforX = typename TileCopy_::VecCopyGmToUb;
    using VecCopyUbToGmforY = typename TileCopy_::VecCopyUbToGm;

    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using UBAlignHelper = FTSelf::Gemv::helper::UBAlignHelper<ElementA>;
    using TensorCoord = layout::VectorLayout::TensorCoord;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t Abuf_SIZE_ = 64 * 1024;
    // static constexpr uint32_t Xbuf_SIZE_ = 16 * 1024;
    static constexpr uint32_t Ybuf_SIZE_ = 64 * 1024;
    static constexpr uint32_t Xbuf_SIZE_for_Ae_ = 8 * 1024;
    static constexpr uint32_t Ybuf_SIZE_for_Ae_ = 8 * 1024;

    CATLASS_DEVICE
    BlockSliceKMNSum() {}

    /// Construct
    CATLASS_DEVICE
    BlockSliceKMNSum(Arch::Resource<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbYOffset = UBufAddrStart + Abuf_SIZE_;
        uint32_t UbXOffsetforAe = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_;
        uint32_t UbYOffsetforAe = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_ + Xbuf_SIZE_for_Ae_;

        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));
            UbXTensorforAeList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbXOffsetforAe + i * (Xbuf_SIZE_for_Ae_ / 2));
            UbYTensorforAeList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforAe + i * (Ybuf_SIZE_for_Ae_ / 2));

            // Assign event ID for each stages
            UbInAEventList[i] = i;
            // UbInXEventList[i] = i + STAGES;
            UbOutEventList[i] = i;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
        }
    }

    /// Construct
    CATLASS_DEVICE
    BlockSliceKMNSum(FTSelf::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbYOffset = UBufAddrStart + Abuf_SIZE_;
        uint32_t UbXOffsetforAe = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_;
        uint32_t UbYOffsetforAe = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_ + Xbuf_SIZE_for_Ae_;

        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));
            UbXTensorforAeList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbXOffsetforAe + i * (Xbuf_SIZE_for_Ae_ / 2));
            UbYTensorforAeList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforAe + i * (Ybuf_SIZE_for_Ae_ / 2));

            // Assign event ID for each stages
            UbInAEventList[i] = i;
            // UbInXEventList[i] = i + STAGES;
            UbOutEventList[i] = i;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockSliceKMNSum()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            // AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
        }
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        GemvCoord const &actualShape, uint32_t NRealRound, uint32_t KFTRoundCount)
    {

        TileMRound = RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);
        // RoundUp(NRealRound, UBAlignHelper::ALIGN);
        strideA = layoutA.stride(1) * TileNRound;
        strideY = layoutY.stride(1) * TileNRound;
        strideSlice = layoutA.shape(0) * layoutA.shape(1);

        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;

        // main loop
        uint32_t Nloop = CeilDiv(actualShape.n(), TileNRound);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(UbOutEventList[UbOutListId]));
        auto UbYTensor = UbYTensorList[UbOutListId];
        // auto layoutXInL1 = LayoutXInL1::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, L1TileShape::N);
        auto layoutYInUb = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(TileMRound, TileNRound));
        auto layoutTileY = layoutY.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUb(UbYTensor, gmY, layoutYInUb, layoutTileY);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventList[UbOutListId]));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>((event_t)(UbOutEventList[UbOutListId]));
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
                auto UbYTensorNext = UbYTensorList[UbOutListIdNext];
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(UbOutEventList[UbOutListIdNext]));
                auto layoutYInUbNext = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(TileMRound, TileNRound));
                auto layoutTileYNext = layoutY.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                matrixCopyGmToUb(UbYTensorNext, gmY[LoopIdxNext * strideY], layoutYInUbNext, layoutTileYNext);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventList[UbOutListIdNext]));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>((event_t)(UbOutEventList[UbOutListIdNext]));
            }

            if(LoopIdx == 0){
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
                matrixCopyGmToUb(UbATensorList[UbInListId], gmA[LoopIdx * strideA], layoutAInUb, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventList[UbOutListId]));
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[UbOutListId]);

            auto UbYIterTensor = UbYTensorList[UbOutListId];

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
                    matrixCopyGmToUb(UbATensorList[UbInListIdNext], gmA[LoopIdx * strideA + kOffsetNext], layoutAInUbNext, layoutTileANext);
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
                    matrixCopyGmToUb(UbATensorList[UbInListIdNext], gmA[LoopIdxNext * strideA + kOffsetNext], layoutAInUbNext, layoutTileANext);
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

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(UbOutEventList[UbOutListId]);

            // auto layoutXInL1 = LayoutXInL1::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, L1TileShape::N);
            // MakeLayoutInUb(MatrixCoord const &shape)

            auto layoutYInUb = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(m_actual, TileNRound));
            auto layoutDstY = layoutY.GetTileLayout(MakeCoord(m_actual, n_actual));

            // vecCopyUbToGm(gmZ[TileY_Row_offset], UbYTensorList[UbOutListId], layoutDstY, layoutComputeInUb);

            matrixCopyUbToGm(gmY[Y_block_offset], UbYTensorList[UbOutListId], layoutDstY, layoutYInUb);
            // AscendC::PipeBarrier<PIPE_MTE3>();

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[UbOutListId]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[UbOutListId]);

            UbOutListId = UbOutListIdNext;
        }
    }

    CATLASS_DEVICE
    void add_ae_op(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementA> const &gmXforAe, LayoutX const &layoutXforAe,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        AscendC::GlobalTensor<ElementY> const &gmYforAe, LayoutX const &layoutYforAe,
        GemvCoord const &actualShape, uint32_t NRealRound, uint32_t KFTRoundCount)
    {

        TileMRound = RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);
        // RoundUp(NRealRound, UBAlignHelper::ALIGN);
        strideA = layoutA.stride(1) * TileNRound;
        strideY = layoutY.stride(1) * TileNRound;
        strideSlice = layoutA.shape(0) * layoutA.shape(1);

        strideXforAe = TileNRound;
        strideYforAe = TileNRound;
        strideSliceforAe = layoutA.shape(1);

        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;

        // main loop
        uint32_t Nloop = CeilDiv(actualShape.n(), TileNRound);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(UbOutEventList[UbOutListId]));
        auto UbYTensor = UbYTensorList[UbOutListId];
        auto UbYTensorforAe = UbYTensorforAeList[UbOutListId];
        // auto layoutXInL1 = LayoutXInL1::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, L1TileShape::N);
        auto layoutYInUb = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(TileMRound, TileNRound));
        auto layoutTileY = layoutY.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUb(UbYTensor, gmY, layoutYInUb, layoutTileY);
        vecCopyGmToUbforX(UbYTensorforAe, gmYforAe, n_actual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventList[UbOutListId]));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>((event_t)(UbOutEventList[UbOutListId]));
        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {

            m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
            n_actual = (LoopIdx == Nloop - 1) ? (actualShape.n() - LoopIdx * TileNRound) : TileNRound;
            y_actual = m_actual;
            x_actual = n_actual;

            uint32_t Y_block_offset = LoopIdx * strideY;
            uint32_t Y_block_offset_for_Ae = LoopIdx * strideYforAe;
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
                auto UbYTensorNext = UbYTensorList[UbOutListIdNext];
                auto UbYTensorforAeNext = UbYTensorforAeList[UbOutListIdNext];
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(UbOutEventList[UbOutListIdNext]));
                auto layoutYInUbNext = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(TileMRound, TileNRound));
                auto layoutTileYNext = layoutY.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                matrixCopyGmToUb(UbYTensorNext, gmY[LoopIdxNext * strideY], layoutYInUbNext, layoutTileYNext);
                vecCopyGmToUbforX(UbYTensorforAeNext, gmYforAe[LoopIdxNext * strideYforAe], n_actual_next);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventList[UbOutListIdNext]));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>((event_t)(UbOutEventList[UbOutListIdNext]));
            }

            if(LoopIdx == 0){
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
                matrixCopyGmToUb(UbATensorList[UbInListId], gmA[LoopIdx * strideA], layoutAInUb, layoutTileA);
                vecCopyGmToUbforX(UbXTensorforAeList[UbInListId], gmXforAe[LoopIdx * strideXforAe], n_actual);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventList[UbOutListId]));
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[UbOutListId]);

            auto UbYIterTensor = UbYTensorList[UbOutListId];
            auto UbYIterTensorforAe = UbYTensorforAeList[UbOutListId];

            for(uint32_t kFTRoundIdx = 0; kFTRoundIdx < KFTRoundCount; kFTRoundIdx++){
                uint32_t kOffset = kFTRoundIdx * strideSlice;
                uint32_t kFTRoundIdxNext = kFTRoundIdx + 1;
                uint32_t kOffsetNext = kFTRoundIdxNext * strideSlice;

                uint32_t kOffsetforAe = kFTRoundIdx * strideSliceforAe;
                uint32_t kOffsetforAeNext = kFTRoundIdxNext * strideSliceforAe;

                uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
                if(kFTRoundIdx < KFTRoundCount - 1){
                    // Preload next K FT round data
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    auto layoutAInUbNext = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
                    matrixCopyGmToUb(UbATensorList[UbInListIdNext], gmA[LoopIdx * strideA + kOffsetNext], layoutAInUbNext, layoutTileANext);
                    vecCopyGmToUbforX(UbXTensorforAeList[UbInListIdNext], gmXforAe[LoopIdx * strideXforAe + kOffsetforAeNext], n_actual);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }else if(LoopIdx < (Nloop - 1)){
                    uint32_t LoopIdxNext = LoopIdx + 1;
                    uint32_t kFTRoundIdxNext = 0;

                    uint32_t kOffsetNext = kFTRoundIdxNext * strideSlice;
                    uint32_t kOffsetforAeNext = kFTRoundIdxNext * strideSliceforAe;
                    // Preload next N loop first K FT round data
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    auto layoutAInUbNext = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    uint32_t m_actual_next = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
                    uint32_t n_actual_next = (LoopIdxNext == Nloop - 1) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;
                    // auto layoutAInUbNext = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(UbATensorList[UbInListIdNext], gmA[LoopIdxNext * strideA + kOffsetNext], layoutAInUbNext, layoutTileANext);
                    vecCopyGmToUbforX(UbXTensorforAeList[UbInListIdNext], gmXforAe[LoopIdxNext * strideXforAe + kOffsetforAeNext], n_actual_next);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
                auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));

                auto UbATensor = UbATensorList[UbInListId];
                auto UbXTensorforAe = UbXTensorforAeList[UbInListId];

                tileMatrixAdd(UbYIterTensor,UbATensor,UbYIterTensor,layoutComputeInUb,layoutTileCompute);
                /*
                template <typename T>
                __aicore__ inline void Add(const LocalTensor<T>& dstLocal,
                    const LocalTensor<T>& src0Local,
                    const LocalTensor<T>& src1Local,
                    const int32_t& calCount)
                */
                AscendC::Add(UbYIterTensorforAe, UbXTensorforAe, UbYIterTensorforAe, n_actual);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                UbInListId = UbInListIdNext;
            }

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(UbOutEventList[UbOutListId]);

            // auto layoutXInL1 = LayoutXInL1::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, L1TileShape::N);
            // MakeLayoutInUb(MatrixCoord const &shape)

            auto layoutYInUb = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(m_actual, TileNRound));
            auto layoutDstY = layoutY.GetTileLayout(MakeCoord(m_actual, n_actual));

            // vecCopyUbToGm(gmZ[TileY_Row_offset], UbYTensorList[UbOutListId], layoutDstY, layoutComputeInUb);

            matrixCopyUbToGm(gmY[Y_block_offset], UbYTensorList[UbOutListId], layoutDstY, layoutYInUb);
            // AscendC::PipeBarrier<PIPE_MTE3>();
            /*
            auto layoutDstY = layoutX.GetTileLayout(TensorCoord(y_actual));
            auto layoutComputeInUb = layoutX.GetTileLayout(TensorCoord(y_actual));
            vecCopyUbToGmforB(gmZ, UbYTensorforBList[UbOutListId], layoutDstY, layoutComputeInUb);
            */
            auto layoutDstYforAe = layoutYforAe.GetTileLayout(TensorCoord(n_actual));
            auto layoutYforAeInUb = layoutYforAe.GetTileLayout(TensorCoord(n_actual));
            vecCopyUbToGmforY(gmYforAe[Y_block_offset_for_Ae],
                UbYTensorforAeList[UbOutListId], layoutDstYforAe, layoutYforAeInUb);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[UbOutListId]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[UbOutListId]);

            UbOutListId = UbOutListIdNext;
        }
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> UbATensorList[STAGES];
    // AscendC::LocalTensor<ElementX> UbXTensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorList[STAGES];
    // AscendC::LocalTensor<ElementA> UbWTensorList[STAGES];

    AscendC::LocalTensor<ElementA> UbXTensorforAeList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorforAeList[STAGES];

    // Multi-stage event id list
    int32_t UbInAEventList[STAGES];
    // int32_t UbInXEventList[STAGES];
    int32_t UbOutEventList[STAGES];

    // The id of current stage
    uint32_t UbOutListId{0};
    uint32_t UbInListId{0};

    uint32_t m_actual, n_actual, x_actual, y_actual;
    uint32_t TileMRound, TileNRound;
    uint32_t strideA, strideY;
    uint32_t strideSlice;

    uint32_t strideXforAe, strideYforAe;
    uint32_t strideSliceforAe;

    TileMatrixAdd tileMatrixAdd;

    MatrixCopyGmToUb matrixCopyGmToUb;
    MatrixCopyUbToGm matrixCopyUbToGm;

    VecCopyGmToUbforX vecCopyGmToUbforX;
    VecCopyUbToGmforY vecCopyUbToGmforY;
};

} // namespace FTSelf::Gemv::Block

#endif // CATLASS_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
