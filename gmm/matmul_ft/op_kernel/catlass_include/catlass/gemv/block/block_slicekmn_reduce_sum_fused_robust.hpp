/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_BLOCK_BLOCK_SLICEKMN_RED_SUM_FUSED_HPP_ROBUST
#define CATLASS_GEMV_BLOCK_BLOCK_SLICEKMN_RED_SUM_FUSED_HPP_ROBUST

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
#include "catlass/gemv/tile/tile_reduce_mean_var.hpp"

namespace FTSelf::Gemv::Block {

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
    Catlass::Gemm::GemvAtlasA2,
    FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_ROBUST,
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
    using DispatchPolicy = Catlass::Gemm::GemvAtlasA2;
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

    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;
    using FT_REDUCE_TYPE = FTSelf::Gemv::helper::FT_REDUCE_TYPE;

    using TileFaultMeanforB = FTSelf::Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::SUM, BType_, XType_>;
    using TileFaultMaxforB = FTSelf::Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::MAX, BType_, XType_>;

    /*
    template <
        /// Tag indicating architecture
        class ArchTag,
        helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
        helper::FT_REDUCE_TYPE REDUCE_TYPE_,
        class AType,
        class XType,
        class YType,
        class BiasType = void
    >
    */
    using TileReduceMeanforB = FTSelf::Gemv::Tile::TileReduce<
        ArchTag,
        FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        FT_REDUCE_TYPE::MEAN_SQUARE,
        BType_,
        XType_,
        XType_,
        void>;

    using TileReduceVarforB = FTSelf::Gemv::Tile::TileReduce<
        ArchTag,
        FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        FT_REDUCE_TYPE::VAR,
        BType_,
        XType_,
        XType_,
        void>;

    using MatrixCopyGmToUbforA = typename TileCopyforA_::MatrixCopyGmToUb;
    using MatrixCopyUbToGmforA = typename TileCopyforA_::MatrixCopyUbToGm;
    using VecCopyGmToUbforX = typename TileCopyforA_::VecCopyGmToUb;
    using VecCopyUbToGmforY = typename TileCopyforA_::VecCopyUbToGm;

    using VecCopyGmToUbforB = typename TileCopyforB_::VecCopyGmToUb;
    using VecCopyUbToGmforB = typename TileCopyforB_::VecCopyUbToGm;
    using MatrixCopyGmToUbforB = typename TileCopyforB_::MatrixCopyGmToUb;
    using MatrixCopyGmToUbSimplingContinueforB = typename TileCopyforB_::MatrixCopyGmToUbSimplingContinue;
    using MatrixCopyGmToUbSimplingStridedforB = typename TileCopyforB_::MatrixCopyGmToUbSimplingStrided;

    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using UBAlignHelper = FTSelf::Gemv::helper::UBAlignHelper<ElementA>;
    using TensorCoord = layout::VectorLayout::TensorCoord;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;

    static constexpr uint32_t Abuf_SIZE_ = 84 * 1024;
    static constexpr uint32_t Xbuf_SIZE_for_Ae_ = 8 * 1024;
    static constexpr uint32_t Ybuf_SIZE_for_Ae_ = 8 * 1024;
    // static constexpr uint32_t Xbuf_SIZE_ = 16 * 1024;
    static constexpr uint32_t workspace_SIZE_ = 16 * 1024;
    static constexpr uint32_t Ybuf_SIZE_for_BVar_ = 6 * 1024;
    static constexpr uint32_t Ybuf_SIZE_for_BMean_ = 6 * 1024;
    static constexpr uint32_t Ybuf_SIZE_for_A_ = 64 * 1024;

    static constexpr uint32_t Abuf_SIZE_for_BMean_ = 28 * 1024;
    static constexpr uint32_t Abuf_SIZE_for_BMin_ = 28 * 1024;
    static constexpr uint32_t Abuf_SIZE_for_BMax_ = 28 * 1024;
    // static constexpr uint32_t Abuf_SIZE_ = 128 * 1024;
    // static constexpr uint32_t Xbuf_SIZE_ = 16 * 1024;


    CATLASS_DEVICE
    BlockSliceKMNSum() {}

    /// Construct
    CATLASS_DEVICE
    BlockSliceKMNSum(Arch::Resource<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbAOffsetforBMean = UBufAddrStart;
        uint32_t UbAOffsetforBMax = UBufAddrStart + Abuf_SIZE_for_BMean_;
        uint32_t UbAOffsetforBMin = UBufAddrStart + Abuf_SIZE_for_BMean_ + Abuf_SIZE_for_BMax_;

        uint32_t UbXOffsetforAe = UBufAddrStart + Abuf_SIZE_;
        uint32_t UbYOffsetforA = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_for_Ae_;

        uint32_t UbYOffsetforBMean = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_for_Ae_ + Ybuf_SIZE_for_A_;
        uint32_t UbYOffsetforBVar = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_for_Ae_ + Ybuf_SIZE_for_A_ + Ybuf_SIZE_for_BMean_;
        uint32_t UbYOffsetforAe = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_for_Ae_ + Ybuf_SIZE_for_A_ + Ybuf_SIZE_for_BMean_ + Ybuf_SIZE_for_BVar_;

        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_for_Ae_ + Ybuf_SIZE_for_A_ + Ybuf_SIZE_for_BMean_ + Ybuf_SIZE_for_BVar_ + Ybuf_SIZE_for_Ae_;

        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbBMeanTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementB>(UbAOffsetforBMean + i * (Abuf_SIZE_for_BMean_/ 2));
            UbBMaxTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementB>(UbAOffsetforBMax + i * (Abuf_SIZE_for_BMax_ / 2));
            UbBMinTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementB>(UbAOffsetforBMin + i * (Abuf_SIZE_for_BMin_ / 2));


            UbXTensorforAeList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbXOffsetforAe + i * (Xbuf_SIZE_for_Ae_ / 2));

            UbYTensorforAList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforA + i * (Ybuf_SIZE_for_A_ / 2));

            UbYTensorforBMeanList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforBMean + i * (Ybuf_SIZE_for_BMean_ / 2));
            UbYTensorforBVarList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforBVar + i * (Ybuf_SIZE_for_BVar_ / 2));

            UbYTensorforAeList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforAe + i * (Ybuf_SIZE_for_Ae_ / 2));

            UbWTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbWOffset + i * (workspace_SIZE_ / 2));
            UbWTensorforRawList[i] = UbWTensorList[i].template ReinterpretCast<ElementB>();

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
        uint32_t UbAOffsetforBMean = UBufAddrStart;
        uint32_t UbAOffsetforBMax = UBufAddrStart + Abuf_SIZE_for_BMean_;
        uint32_t UbAOffsetforBMin = UBufAddrStart + Abuf_SIZE_for_BMean_ + Abuf_SIZE_for_BMax_;

        uint32_t UbXOffsetforAe = UBufAddrStart + Abuf_SIZE_;
        uint32_t UbYOffsetforA = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_for_Ae_;

        uint32_t UbYOffsetforBMean = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_for_Ae_ + Ybuf_SIZE_for_A_;
        uint32_t UbYOffsetforBVar = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_for_Ae_ + Ybuf_SIZE_for_A_ + Ybuf_SIZE_for_BMean_;
        uint32_t UbYOffsetforAe = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_for_Ae_ + Ybuf_SIZE_for_A_ + Ybuf_SIZE_for_BMean_ + Ybuf_SIZE_for_BVar_;

        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_for_Ae_ + Ybuf_SIZE_for_A_ + Ybuf_SIZE_for_BMean_ + Ybuf_SIZE_for_BVar_ + Ybuf_SIZE_for_Ae_;

        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbBMeanTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementB>(UbAOffsetforBMean + i * (Abuf_SIZE_for_BMean_/ 2));
            UbBMaxTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementB>(UbAOffsetforBMax + i * (Abuf_SIZE_for_BMax_ / 2));
            UbBMinTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementB>(UbAOffsetforBMin + i * (Abuf_SIZE_for_BMin_ / 2));


            UbXTensorforAeList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbXOffsetforAe + i * (Xbuf_SIZE_for_Ae_ / 2));

            UbYTensorforAList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforA + i * (Ybuf_SIZE_for_A_ / 2));

            UbYTensorforBMeanList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforBMean + i * (Ybuf_SIZE_for_BMean_ / 2));
            UbYTensorforBVarList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforBVar + i * (Ybuf_SIZE_for_BVar_ / 2));

            UbYTensorforAeList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforAe + i * (Ybuf_SIZE_for_Ae_ / 2));

            UbWTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementB>(UbWOffset + i * (workspace_SIZE_ / 2));
            UbWTensorforRawList[i] = UbWTensorList[i].template ReinterpretCast<ElementB>();

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
    void add_ae_op(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementA> const &gmXforAe, LayoutX const &layoutXforAe,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        AscendC::GlobalTensor<ElementY> const &gmYforAe, LayoutX const &layoutYforAe,
        GemvCoord const &actualShape, uint32_t NRealRound, uint32_t KFTRoundCount)
    {

        TileMRound = RoundUp(UBTileShapeforA::M, UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShapeforA::N, UBAlignHelper::ALIGN);
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

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(UbOutEventforAList[UbOutListId]));
        auto UbYTensor = UbYTensorforAList[UbOutListId];
        auto UbYTensorforAe = UbYTensorforAeList[UbOutListId];

        auto layoutYInUb = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(TileMRound, TileNRound));
        auto layoutTileY = layoutY.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUbforA(UbYTensor, gmY, layoutYInUb, layoutTileY);
        vecCopyGmToUbforX(UbYTensorforAe, gmYforAe, n_actual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventforAList[UbOutListId]));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>((event_t)(UbOutEventforAList[UbOutListId]));
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
                auto UbYTensorNext = UbYTensorforAList[UbOutListIdNext];
                auto UbYTensorforAeNext = UbYTensorforAeList[UbOutListIdNext];
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(UbOutEventforAList[UbOutListIdNext]));
                auto layoutYInUbNext = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(TileMRound, TileNRound));
                auto layoutTileYNext = layoutY.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                matrixCopyGmToUbforA(UbYTensorNext, gmY[LoopIdxNext * strideY], layoutYInUbNext, layoutTileYNext);
                vecCopyGmToUbforX(UbYTensorforAeNext, gmYforAe[LoopIdxNext * strideYforAe], n_actual_next);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventforAList[UbOutListIdNext]));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>((event_t)(UbOutEventforAList[UbOutListIdNext]));
            }

            if(LoopIdx == 0){
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
                matrixCopyGmToUbforA(UbATensorList[UbInListId], gmA[LoopIdx * strideA], layoutAInUb, layoutTileA);
                vecCopyGmToUbforX(UbXTensorforAeList[UbInListId], gmXforAe[LoopIdx * strideXforAe], n_actual);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventforAList[UbOutListId]));
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutEventforAList[UbOutListId]);

            auto UbYIterTensor = UbYTensorforAList[UbOutListId];
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
                    matrixCopyGmToUbforA(UbATensorList[UbInListIdNext], gmA[LoopIdx * strideA + kOffsetNext], layoutAInUbNext, layoutTileANext);
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
                    matrixCopyGmToUbforA(UbATensorList[UbInListIdNext], gmA[LoopIdxNext * strideA + kOffsetNext], layoutAInUbNext, layoutTileANext);
                    vecCopyGmToUbforX(UbXTensorforAeList[UbInListIdNext], gmXforAe[LoopIdxNext * strideXforAe + kOffsetforAeNext], n_actual_next);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
                auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));

                auto UbATensor = UbATensorList[UbInListId];
                auto UbXTensorforAe = UbXTensorforAeList[UbInListId];
                tileMatrixAdd(UbYIterTensor, UbATensor, UbYIterTensor, layoutComputeInUb, layoutTileCompute);
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

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventforAList[UbOutListId]));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventforAList[UbOutListId]));

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(UbOutEventforAList[UbOutListId]);

            auto layoutYInUb = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(m_actual, TileNRound));
            auto layoutDstY = layoutY.GetTileLayout(MakeCoord(m_actual, n_actual));

            matrixCopyUbToGmforA(gmY[Y_block_offset], UbYTensorforAList[UbOutListId], layoutDstY, layoutYInUb);
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

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventforAList[UbOutListId]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventforAList[UbOutListId]);

            UbOutListId = UbOutListIdNext;
        }
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    CATLASS_DEVICE
    void RowMeanAbsSquare(
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementY> const &gmZMeanAbs,
        AscendC::GlobalTensor<ElementY> const &gmZMeanSquare,
        LayoutX const &layoutX,
        GemvCoord const &actualShape,
        uint32_t NRealRound,
        float n_scale_ratio)
    {
        TileMRound = RoundUp(UBTileShapeforB::M, UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShapeforB::N, UBAlignHelper::ALIGN);
        // RoundUp(NRealRound, UBAlignHelper::ALIGN);
        strideB = layoutB.stride(1) * TileNRound;
        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventforBList[UbOutListId]));

        auto UbYTensorforAbs = UbYTensorforBMeanList[UbOutListId];
        auto UbYTensorforSquare = UbYTensorforBVarList[UbOutListId];

        AscendC::Duplicate<ElementY>(UbYTensorforAbs, (ElementY)0.0, m_actual);
        AscendC::Duplicate<ElementY>(UbYTensorforSquare, (ElementY)0.0, m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventforBList[UbOutListId]));
        // AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventforBList[UbOutListId]));

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
        auto layoutBInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUbforB(UbBMeanTensorList[UbInListId], gmB, layoutBInUb, layoutTileB);
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
                auto matrixTensor = UbBMeanTensorList[UbInListIdNext];

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
            CATLASS_DEVICE
            void operator()(
                AscendC::LocalTensor<ElementY> dstTensorMeanAbs,
                AscendC::LocalTensor<ElementY> dstTensorMeanSquare,
                AscendC::LocalTensor<ElementA> srcMeanTensor,
                AscendC::LocalTensor<ElementY> red_workspace,
                LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
                ElementA n_ratio_factor)
            */
            tileReduceMeanforB(UbYTensorforBMeanList[UbOutListId],
                UbYTensorforBVarList[UbOutListId],
                UbBMeanTensorList[UbInListId],
                UbWTensorList[UbInListId],
                layoutComputeInUb,
                layoutTileCompute,
                (ElementB)n_scale_ratio);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            UbInListId = UbInListIdNext;
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventforBList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventforBList[UbOutListId]));

        auto layoutDstY = layoutX.GetTileLayout(TensorCoord(y_actual));
        auto layoutComputeInUb = layoutX.GetTileLayout(TensorCoord(y_actual));

        vecCopyUbToGmforB(gmZMeanAbs, UbYTensorforBMeanList[UbOutListId], layoutDstY, layoutComputeInUb);
        vecCopyUbToGmforB(gmZMeanSquare, UbYTensorforBVarList[UbOutListId], layoutDstY, layoutComputeInUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventforBList[UbOutListId]));
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    CATLASS_DEVICE
    void RowMeanAbsSquareWithSimpleCon(
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementY> const &gmZMeanAbs,
        AscendC::GlobalTensor<ElementY> const &gmZMeanSquare,
        LayoutX const &layoutX,
        GemvCoord const &actualShape,
        uint32_t NRealRound,
        float n_scale_ratio, uint32_t simpling_stride)
    {

        TileMRound = RoundUp(UBTileShapeforB::M, UBAlignHelper::ALIGN);
        TileMRoundSimpling = TileMRound;

        TileNRound = RoundUp(UBTileShapeforB::N, UBAlignHelper::ALIGN);
        TileNRoundSimpling = (simpling_stride < 2) ? TileNRound : (TileNRound / simpling_stride);
        TileNRoundSimpling = RoundUp(TileNRoundSimpling, UBAlignHelper::ALIGN);

        uint32_t simpling_stride_round = (simpling_stride < 2) ? 1 : simpling_stride;
        // TileNRound / TileNRoundSimpling;
        TileNRound = TileNRoundSimpling * simpling_stride_round;

        // RoundUp(NRealRound, UBAlignHelper::ALIGN);
        strideB = layoutB.stride(1) * TileNRound;
        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;




        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventforBList[UbOutListId]));

        auto UbYTensorforAbs = UbYTensorforBMeanList[UbOutListId];
        auto UbYTensorforSquare = UbYTensorforBVarList[UbOutListId];

        AscendC::Duplicate<ElementY>(UbYTensorforAbs, (ElementY)0.0, m_actual);
        AscendC::Duplicate<ElementY>(UbYTensorforSquare, (ElementY)0.0, m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventforBList[UbOutListId]));
        // AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventforBList[UbOutListId]));

        /*
        CATLASS_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> dstTensor,
            AscendC::GlobalTensor<Element> srcTensor,
            LayoutDst const &layoutDst,
            LayoutSrc const &layoutSrc, uint32_t simpling_stride)
        */

        bool using_simpling = (n_actual > TileNRoundSimpling) ? true : false;
        auto layoutBInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRoundSimpling));
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(m_actual, n_actual));

        if(using_simpling && (simpling_stride_round > 1)){
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            matrixCopyGmToUbSimplingCforB(UbBMeanTensorList[UbInListId], gmB,
                layoutBInUb, layoutTileB, simpling_stride_round);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
        }else{
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            matrixCopyGmToUbforB(UbBMeanTensorList[UbInListId], gmB, layoutBInUb, layoutTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
        }

        // main loop
        uint32_t Nloop = CeilDiv(actualShape.n(), TileNRound);
        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {
            m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
            n_actual = (LoopIdx == Nloop - 1) ? (actualShape.n() - LoopIdx * TileNRound) : TileNRound;
            n_actual_local = (n_actual > TileNRoundSimpling) ? (n_actual / simpling_stride_round) : n_actual;

            y_actual = m_actual;
            x_actual = n_actual;

            uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
            if (LoopIdx < Nloop - 1) {
                uint32_t LoopIdxNext = LoopIdx + 1;
                uint32_t m_actual_next = m_actual;
                uint32_t n_actual_next =
                    (LoopIdxNext == Nloop - 1) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;

                bool using_simpling = (n_actual_next > TileNRoundSimpling) ? true : false;

                uint32_t y_actual_next = m_actual_next;
                uint32_t x_actual_next = n_actual_next;

                auto layoutBInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRoundSimpling));
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));

                // Get L1 tensor for next stage
                auto matrixTensor = UbBMeanTensorList[UbInListIdNext];
                if(using_simpling && (simpling_stride_round > 1)){
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    matrixCopyGmToUbSimplingCforB(matrixTensor,
                        gmB[LoopIdxNext * strideB],
                        layoutBInUb, layoutTileB, simpling_stride_round);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }else{
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    matrixCopyGmToUbforB(matrixTensor, gmB[LoopIdxNext * strideB], layoutBInUb, layoutTileB);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            auto layoutComputeInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRoundSimpling));
            auto layoutTileCompute = layoutB.GetTileLayout(MakeCoord(m_actual, n_actual_local));

            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::LocalTensor<ElementY> dstTensorMeanAbs,
                AscendC::LocalTensor<ElementY> dstTensorMeanSquare,
                AscendC::LocalTensor<ElementA> srcMeanTensor,
                AscendC::LocalTensor<ElementY> red_workspace,
                LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
                ElementA n_ratio_factor)
            */
            tileReduceMeanforB(UbYTensorforBMeanList[UbOutListId],
                UbYTensorforBVarList[UbOutListId],
                UbBMeanTensorList[UbInListId],
                UbWTensorList[UbInListId],
                layoutComputeInUb,
                layoutTileCompute,
                (ElementB)n_scale_ratio);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            UbInListId = UbInListIdNext;
        }

        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Muls(const LocalTensor<T>& dstLocal,
            const LocalTensor<T>& srcLocal,
            const T& scalarValue,
            const int32_t& calCount)
        */
        AscendC::Muls(UbYTensorforBMeanList[UbOutListId],
            UbYTensorforBMeanList[UbOutListId],
            (ElementY)(1.0f*simpling_stride_round), y_actual);

        AscendC::Muls(UbYTensorforBVarList[UbOutListId],
            UbYTensorforBVarList[UbOutListId],
            (ElementY)(1.0f*simpling_stride_round), y_actual);

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventforBList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventforBList[UbOutListId]));

        auto layoutDstY = layoutX.GetTileLayout(TensorCoord(y_actual));
        auto layoutComputeInUb = layoutX.GetTileLayout(TensorCoord(y_actual));

        vecCopyUbToGmforB(gmZMeanAbs, UbYTensorforBMeanList[UbOutListId], layoutDstY, layoutComputeInUb);
        vecCopyUbToGmforB(gmZMeanSquare, UbYTensorforBVarList[UbOutListId], layoutDstY, layoutComputeInUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventforBList[UbOutListId]));
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    CATLASS_DEVICE
    void RowMeanAbsSquareWithSimpleStride(
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementY> const &gmZMeanAbs,
        AscendC::GlobalTensor<ElementY> const &gmZMeanSquare,
        LayoutX const &layoutX,
        GemvCoord const &actualShape,
        uint32_t NRealRound,
        float n_scale_ratio,
        uint32_t simpling_stride, uint32_t stride_unit)
    {
        /*
        CATLASS_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> dstTensor,
            AscendC::GlobalTensor<Element> srcTensor,
            LayoutDst const &layoutDst,
            LayoutSrc const &layoutSrc, uint32_t simpling_stride, uint32_t stride_unit)
        */

        TileMRound = RoundUp(UBTileShapeforB::M, UBAlignHelper::ALIGN);
        TileMRoundSimpling = TileMRound;

        TileNRound = RoundUp(UBTileShapeforB::N, UBAlignHelper::ALIGN);
        uint32_t stride_unit_aligned = RoundUp(stride_unit, UBAlignHelper::ALIGN);
        uint32_t stride_chunk_aligned = stride_unit_aligned * simpling_stride;
        uint32_t stride_chunk_num_raw = TileNRound / stride_chunk_aligned;
        uint32_t stride_chunk_num = stride_chunk_num_raw;

        if(stride_chunk_num_raw < 1){
            stride_unit_aligned = TileNRound / simpling_stride;
            stride_unit_aligned = RoundUp(stride_unit_aligned, UBAlignHelper::ALIGN);
            stride_chunk_num = 1;
        }

        TileNRound = stride_chunk_num * stride_unit_aligned * simpling_stride;
        TileNRoundSimpling = (simpling_stride < 2) ? TileNRound : stride_chunk_num * stride_unit_aligned;

        uint32_t simpling_stride_round = (simpling_stride < 2) ? 1 : simpling_stride;

        // RoundUp(NRealRound, UBAlignHelper::ALIGN);
        strideB = layoutB.stride(1) * TileNRound;
        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;


        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventforBList[UbOutListId]));

        auto UbYTensorforAbs = UbYTensorforBMeanList[UbOutListId];
        auto UbYTensorforSquare = UbYTensorforBVarList[UbOutListId];

        AscendC::Duplicate<ElementY>(UbYTensorforAbs, (ElementY)0.0, m_actual);
        AscendC::Duplicate<ElementY>(UbYTensorforSquare, (ElementY)0.0, m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventforBList[UbOutListId]));
        // AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventforBList[UbOutListId]));

        /*
        CATLASS_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> dstTensor,
            AscendC::GlobalTensor<Element> srcTensor,
            LayoutDst const &layoutDst,
            LayoutSrc const &layoutSrc, uint32_t simpling_stride)
        */

        bool using_simpling = (n_actual > TileNRoundSimpling) ? true : false;
        auto layoutBInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRoundSimpling));
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(m_actual, n_actual));

        if(using_simpling && (simpling_stride_round > 1)){
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            matrixCopyGmToUbSimplingSforB(UbBMeanTensorList[UbInListId], gmB,
                layoutBInUb, layoutTileB,
                simpling_stride_round, stride_unit_aligned);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
        }else{
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            matrixCopyGmToUbforB(UbBMeanTensorList[UbInListId], gmB,
                layoutBInUb, layoutTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
        }

        // main loop
        uint32_t Nloop = CeilDiv(actualShape.n(), TileNRound);
        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {
            m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
            n_actual = (LoopIdx == Nloop - 1) ? (actualShape.n() - LoopIdx * TileNRound) : TileNRound;
            if(LoopIdx < (Nloop - 1)){
                n_actual_local = TileNRoundSimpling;
            }else if(n_actual <= TileNRoundSimpling){
                n_actual_local = n_actual;
            }else{
                uint32_t last_chunk_num = n_actual / (stride_unit_aligned * simpling_stride);
                uint32_t last_remain_actual = n_actual % (stride_unit_aligned * simpling_stride);

                n_actual_local = last_chunk_num * stride_unit_aligned;
                uint32_t n_actual_tile = (last_remain_actual < stride_unit_aligned) ? last_remain_actual : stride_unit_aligned;

                n_actual_local = n_actual_local + n_actual_tile;
            }

            y_actual = m_actual;
            x_actual = n_actual;

            uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
            if (LoopIdx < Nloop - 1) {
                uint32_t LoopIdxNext = LoopIdx + 1;
                uint32_t m_actual_next = m_actual;
                uint32_t n_actual_next =
                    (LoopIdxNext == Nloop - 1) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;

                bool using_simpling = (n_actual_next > TileNRoundSimpling) ? true : false;

                uint32_t y_actual_next = m_actual_next;
                uint32_t x_actual_next = n_actual_next;

                auto layoutBInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRoundSimpling));
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));

                // Get L1 tensor for next stage
                auto matrixTensor = UbBMeanTensorList[UbInListIdNext];
                if(using_simpling && (simpling_stride_round > 1)){
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    matrixCopyGmToUbSimplingSforB(matrixTensor,
                        gmB[LoopIdxNext * strideB],
                        layoutBInUb, layoutTileB,
                        simpling_stride_round, stride_unit_aligned);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }else{
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    matrixCopyGmToUbforB(matrixTensor, gmB[LoopIdxNext * strideB],
                        layoutBInUb, layoutTileB);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            auto layoutComputeInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRoundSimpling));
            auto layoutTileCompute = layoutB.GetTileLayout(MakeCoord(m_actual, n_actual_local));

            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::LocalTensor<ElementY> dstTensorMeanAbs,
                AscendC::LocalTensor<ElementY> dstTensorMeanSquare,
                AscendC::LocalTensor<ElementA> srcMeanTensor,
                AscendC::LocalTensor<ElementY> red_workspace,
                LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
                ElementA n_ratio_factor)
            */
            tileReduceMeanforB(UbYTensorforBMeanList[UbOutListId],
                UbYTensorforBVarList[UbOutListId],
                UbBMeanTensorList[UbInListId],
                UbWTensorList[UbInListId],
                layoutComputeInUb,
                layoutTileCompute,
                (ElementB)n_scale_ratio);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            UbInListId = UbInListIdNext;
        }

        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Muls(const LocalTensor<T>& dstLocal,
            const LocalTensor<T>& srcLocal,
            const T& scalarValue,
            const int32_t& calCount)
        */
        AscendC::Muls(UbYTensorforBMeanList[UbOutListId],
            UbYTensorforBMeanList[UbOutListId],
            (ElementY)(1.0f*simpling_stride_round), y_actual);

        AscendC::Muls(UbYTensorforBVarList[UbOutListId],
            UbYTensorforBVarList[UbOutListId],
            (ElementY)(1.0f*simpling_stride_round), y_actual);

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventforBList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventforBList[UbOutListId]));

        auto layoutDstY = layoutX.GetTileLayout(TensorCoord(y_actual));
        auto layoutComputeInUb = layoutX.GetTileLayout(TensorCoord(y_actual));

        vecCopyUbToGmforB(gmZMeanAbs, UbYTensorforBMeanList[UbOutListId], layoutDstY, layoutComputeInUb);
        vecCopyUbToGmforB(gmZMeanSquare, UbYTensorforBVarList[UbOutListId], layoutDstY, layoutComputeInUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventforBList[UbOutListId]));
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    CATLASS_DEVICE
    void RowVariance(
        AscendC::GlobalTensor<ElementB> const &gmBMean,
        AscendC::GlobalTensor<ElementB> const &gmBMax,
        AscendC::GlobalTensor<ElementB> const &gmBMin,
        LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementY> const &gmZVar,
        LayoutX const &layoutX,
        GemvCoord const &actualShape,
        uint32_t NRealRound,
        float n_scale_ratio)
    {

        TileMRound = RoundUp(UBTileShapeforB::M, UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShapeforB::N, UBAlignHelper::ALIGN);
        // RoundUp(NRealRound, UBAlignHelper::ALIGN);
        strideB = layoutB.stride(1) * TileNRound;
        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventforBList[UbOutListId]));

        auto UbYTensorforVar = UbYTensorforBVarList[UbOutListId];

        AscendC::Duplicate<ElementY>(UbYTensorforVar, (ElementY)0.0, m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventforBList[UbOutListId]));
        // AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventforBList[UbOutListId]));

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
        auto layoutBInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(m_actual, n_actual));

        matrixCopyGmToUbforB(UbBMeanTensorList[UbInListId], gmBMean, layoutBInUb, layoutTileB);
        matrixCopyGmToUbforB(UbBMaxTensorList[UbInListId], gmBMax, layoutBInUb, layoutTileB);
        matrixCopyGmToUbforB(UbBMinTensorList[UbInListId], gmBMin, layoutBInUb, layoutTileB);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
        // main loop
        uint32_t Nloop = CeilDiv(actualShape.n(), TileNRound);
        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {
            m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
            n_actual = (LoopIdx == (Nloop - 1)) ? (actualShape.n() - LoopIdx * TileNRound) : TileNRound;
            y_actual = m_actual;
            x_actual = n_actual;

            uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
            if (LoopIdx < Nloop - 1) {
                uint32_t LoopIdxNext = LoopIdx + 1;
                uint32_t m_actual_next = m_actual;
                uint32_t n_actual_next =
                    (LoopIdxNext == (Nloop - 1)) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;
                uint32_t y_actual_next = m_actual_next;
                uint32_t x_actual_next = n_actual_next;
                // Get L1 tensor for next stage
                auto matrixTensorforMean = UbBMeanTensorList[UbInListIdNext];
                auto matrixTensorforMax = UbBMaxTensorList[UbInListIdNext];
                auto matrixTensorforMin = UbBMinTensorList[UbInListIdNext];

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                auto layoutBInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                matrixCopyGmToUbforB(matrixTensorforMean, gmBMean[LoopIdxNext * strideB], layoutBInUb, layoutTileB);
                matrixCopyGmToUbforB(matrixTensorforMax, gmBMax[LoopIdxNext * strideB], layoutBInUb, layoutTileB);
                matrixCopyGmToUbforB(matrixTensorforMin, gmBMin[LoopIdxNext * strideB], layoutBInUb, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            auto layoutComputeInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            auto layoutTileCompute = layoutB.GetTileLayout(MakeCoord(m_actual, n_actual));

            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::LocalTensor<ElementY> dstTensorVar,
                AscendC::LocalTensor<ElementA> srcMeanTensor,
                AscendC::LocalTensor<ElementA> srcMaxTensor,
                AscendC::LocalTensor<ElementA> srcMinTensor,
                AscendC::LocalTensor<ElementY> red_workspace,
                LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
                ElementA n_ratio_factor)
            */
            tileReduceVarforB(UbYTensorforBVarList[UbOutListId],
                UbBMeanTensorList[UbInListId],
                UbBMaxTensorList[UbInListId],
                UbBMinTensorList[UbInListId],
                UbWTensorList[UbInListId],
                layoutComputeInUb,
                layoutTileCompute,
                (ElementB)n_scale_ratio);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            UbInListId = UbInListIdNext;
        }

        AscendC::PipeBarrier<PIPE_V>();
        /*
        template <typename T>
        __aicore__ inline void Sqrt(const LocalTensor<T>& dstLocal,
            const LocalTensor<T>& srcLocal, const int32_t& calCount)
        */
        // AscendC::Sqrt(UbYTensorforBVarList[UbOutListId], UbYTensorforBVarList[UbOutListId], m_actual);
        AscendC::Abs(UbYTensorforBVarList[UbOutListId], UbYTensorforBVarList[UbOutListId], m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sqrt(UbYTensorforBVarList[UbOutListId], UbYTensorforBVarList[UbOutListId], m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventforBList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventforBList[UbOutListId]));

        auto layoutDstY = layoutX.GetTileLayout(TensorCoord(y_actual));
        auto layoutComputeInUb = layoutX.GetTileLayout(TensorCoord(y_actual));

        vecCopyUbToGmforB(gmZVar, UbYTensorforBVarList[UbOutListId], layoutDstY, layoutComputeInUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventforBList[UbOutListId]));
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
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

        auto UbYTensor = UbYTensorforBMeanList[UbOutListId];

        AscendC::Duplicate<ElementY>(UbYTensor, (ElementY)0.0, m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventforBList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventforBList[UbOutListId]));

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
        auto layoutBInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUbforB(UbBMeanTensorList[UbInListId], gmB, layoutBInUb, layoutTileB);
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
                auto matrixTensor = UbBMeanTensorList[UbInListIdNext];

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
            tileFaultMeanforB(UbYTensorforBMeanList[UbOutListId],
                UbBMeanTensorList[UbInListId],
                UbWTensorforRawList[UbInListId],
                layoutComputeInUb,
                layoutTileCompute);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            UbInListId = UbInListIdNext;
        }
        AscendC::PipeBarrier<PIPE_V>();

        tileVmuls(UbYTensorforBMeanList[UbOutListId], UbYTensorforBMeanList[UbOutListId], (ElementY)kn_scale_ratio, m_actual);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventforBList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventforBList[UbOutListId]));
        auto layoutDstY = layoutX.GetTileLayout(TensorCoord(y_actual));
        auto layoutComputeInUb = layoutX.GetTileLayout(TensorCoord(y_actual));
        vecCopyUbToGmforB(gmZ, UbYTensorforBMeanList[UbOutListId], layoutDstY, layoutComputeInUb);
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

        auto UbYTensor = UbYTensorforBMeanList[UbOutListId];

        AscendC::Duplicate<ElementY>(UbYTensor, (ElementY)0.0, m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventforBList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventforBList[UbOutListId]));

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
        auto layoutBInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUbforB(UbBMaxTensorList[UbInListId], gmB, layoutBInUb, layoutTileB);
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
                auto matrixTensor = UbBMaxTensorList[UbInListIdNext];

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
            tileFaultMaxforB(UbYTensorforBMeanList[UbOutListId],
                UbBMaxTensorList[UbInListId],
                UbWTensorforRawList[UbInListId],
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
        vecCopyUbToGmforB(gmZ, UbYTensorforBMeanList[UbOutListId], layoutDstY, layoutComputeInUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventforBList[UbOutListId]));
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> UbATensorList[STAGES];

    AscendC::LocalTensor<ElementB> UbBMeanTensorList[STAGES];
    AscendC::LocalTensor<ElementB> UbBMaxTensorList[STAGES];
    AscendC::LocalTensor<ElementB> UbBMinTensorList[STAGES];

    AscendC::LocalTensor<ElementA> UbXTensorforAeList[STAGES];

    AscendC::LocalTensor<ElementY> UbYTensorforAList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorforBVarList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorforBMeanList[STAGES];

    AscendC::LocalTensor<ElementY> UbYTensorforAeList[STAGES];
    AscendC::LocalTensor<ElementY> UbWTensorList[STAGES];
    AscendC::LocalTensor<ElementB> UbWTensorforRawList[STAGES];
    // AscendC::LocalTensor<ElementA> UbWTensorList[STAGES];

    // Multi-stage event id list
    int32_t UbInAEventList[STAGES];
    // int32_t UbInXEventList[STAGES];
    int32_t UbOutEventforAList[STAGES];
    int32_t UbOutEventforBList[STAGES];

    // The id of current stage
    uint32_t UbOutListId{0};
    uint32_t UbInListId{0};

    uint32_t m_actual, n_actual, x_actual, y_actual, n_actual_local;
    uint32_t TileMRound, TileNRound;
    uint32_t TileMRoundSimpling, TileNRoundSimpling;
    uint32_t strideA, strideY, strideB;
    uint32_t strideSlice;
    uint32_t strideXforAe, strideYforAe;
    uint32_t strideSliceforAe;

    TileMatrixAdd tileMatrixAdd;

    TileFaultMeanforB tileFaultMeanforB;
    TileFaultMaxforB tileFaultMaxforB;

    TileReduceMeanforB tileReduceMeanforB;
    TileReduceVarforB tileReduceVarforB;

    TileVmuls tileVmuls;

    MatrixCopyGmToUbforB matrixCopyGmToUbforB;

    MatrixCopyGmToUbSimplingContinueforB matrixCopyGmToUbSimplingCforB;
    MatrixCopyGmToUbSimplingStridedforB matrixCopyGmToUbSimplingSforB;

    VecCopyGmToUbforB vecCopyGmToUbforB;
    VecCopyUbToGmforB vecCopyUbToGmforB;

    VecCopyGmToUbforX vecCopyGmToUbforX;
    VecCopyUbToGmforY vecCopyUbToGmforY;

    MatrixCopyGmToUbforA matrixCopyGmToUbforA;

    MatrixCopyUbToGmforA matrixCopyUbToGmforA;

};

} // namespace FTSelf::Gemv::Block

#endif // CATLASS_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
