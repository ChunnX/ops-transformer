/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_BLOCK_BLOCK_SLICE_REDUCE_SUM_AIV_HPP
#define CATLASS_GEMV_BLOCK_BLOCK_SLICE_REDUCE_SUM_AIV_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource_aiv.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm/dispatch_policy.hpp"

namespace FTSelf::Gemv::Block {

using namespace Catlass;

// class TileVmuls_

template <
    class UBTileShape_,
    class AType_,
    class YType_,
    class BiasType_,
    class TileCopy_,
    class TileSliceSum_
>
struct BlockSliceSum <
    Catlass::Gemm::GemvAtlasA2,
    UBTileShape_,
    AType_,
    YType_,
    BiasType_,
    TileCopy_,
    TileSliceSum_
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
    using TileSliceSum = TileSliceSum_;
    using VecCopyGmToUb = typename TileCopy_::VecCopyGmToUb;
    using VecCopyUbToGm = typename TileCopy_::VecCopyUbToGm;
    using MatrixCopyGmToUb = typename TileCopy_::MatrixCopyGmToUb;
    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;

    using UBAlignHelper = FTSelf::Gemv::helper::UBAlignHelper<ElementA>;
    using TensorCoord = layout::VectorLayout::TensorCoord;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t Abuf_SIZE_ = 160 * 1024;
    // static constexpr uint32_t Xbuf_SIZE_ = 16 * 1024;
    static constexpr uint32_t Ybuf_SIZE_ = 32 * 1024;
    // static constexpr uint32_t workspace_SIZE_ = 48 * 1024;

    static constexpr uint32_t ELE_NUM_PER_REPEAT = BYTE_PER_C0 * 8 / sizeof(ElementY);

    CATLASS_DEVICE
    BlockSliceSum() {}

    /// Construct
    CATLASS_DEVICE
    BlockSliceSum(Arch::Resource<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        // uint32_t UbXOffset = UBufAddrStart + Abuf_SIZE_;
        // + Xbuf_SIZE_ + Xbuf_SIZE_
        uint32_t UbYOffset = UBufAddrStart + Abuf_SIZE_;
        // uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            // UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset + i * (Xbuf_SIZE_ / 2));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));
            // UbWTensorList[i] =
            //     resource.ubBuf.template GetBufferByByte<ElementX>(UbWOffset + i * (workspace_SIZE_ / 2));

            // Assign event ID for each stages
            UbInAEventList[i] = i;
            // UbInXEventList[i] = i + STAGES;
            UbOutEventList[i] = i;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
        }
    }

    /// Construct
    CATLASS_DEVICE
    BlockSliceSum(FTSelf::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        // uint32_t UbXOffset = UBufAddrStart + Abuf_SIZE_;
        // + Xbuf_SIZE_ + Xbuf_SIZE_
        uint32_t UbYOffset = UBufAddrStart + Abuf_SIZE_;
        // uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            // UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset + i * (Xbuf_SIZE_ / 2));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));
            // UbWTensorList[i] =
            //     resource.ubBuf.template GetBufferByByte<ElementX>(UbWOffset + i * (workspace_SIZE_ / 2));

            // Assign event ID for each stages
            UbInAEventList[i] = i;
            // UbInXEventList[i] = i + STAGES;
            UbOutEventList[i] = i;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockSliceSum()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            // AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
        }
    }

    // AscendC::GlobalTensor<ElementX> const &gmX, LayoutX const &layoutX,

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmZ, LayoutY const &layoutY,
        GemvCoord const &actualShape)
    {
        TileMRound = RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);

        strideA = layoutA.stride(0) * TileMRound;
        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;

        // ElementY aim_weight = alpha * beta;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));

        /*
        void Duplicate(const LocalTensor<T>& dstLocal, const T& scalarValue, const int32_t& calCount)
        */
        AscendC::Duplicate<ElementY>(UbYTensorList[UbOutListId], (ElementY)0.0, actualShape.n());

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
        // main loop
        uint32_t Mloop = CeilDiv(actualShape.m(), TileMRound);
        for (uint32_t LoopIdx = 0; LoopIdx < Mloop; LoopIdx++) {
            n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;
            m_actual = (LoopIdx == Mloop - 1) ? (actualShape.m() - LoopIdx * TileMRound) : TileMRound;
            y_actual = n_actual;
            x_actual = m_actual;

            uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
            if (LoopIdx < Mloop - 1) {
                uint32_t LoopIdxNext = LoopIdx + 1;
                uint32_t n_actual_next = n_actual;
                uint32_t m_actual_next =
                    (LoopIdxNext == Mloop - 1) ? (actualShape.m() - LoopIdxNext * TileMRound) : TileMRound;
                uint32_t y_actual_next = n_actual_next;
                uint32_t x_actual_next = m_actual_next;
                // Get L1 tensor for next stage
                auto matrixTensor = UbATensorList[UbInListIdNext];
                // auto vecTensor = UbXTensorList[UbInListIdNext];

                // AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListIdNext]));
                // vecCopyGmToUb(vecTensor, gmX[LoopIdxNext * TileNRound], x_actual_next);
                // AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListIdNext]));

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                matrixCopyGmToUb(matrixTensor, gmA[LoopIdxNext * strideA], layoutAInUb, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
            }
            // AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListId]));
            // tileVmuls(UbXTensorList[UbInListId], UbXTensorList[UbInListId], (ElementA)alpha, x_actual);

            // AscendC::PipeBarrier<PIPE_V>();

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));

            // tileVmad(UbYTensorList[UbOutListId],
            //     UbXTensorList[UbInListId],
            //     UbATensorList[UbInListId],
            //     UbWTensorList[UbInListId],
            //     layoutComputeInUb,
            //     layoutTileCompute);

            /*
            (AscendC::LocalTensor<ElementY> dstTensor,
            AscendC::LocalTensor<ElementA> srcTensor_m,
            LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
            */

            tileSliceSum(
                UbYTensorList[UbOutListId],
                UbATensorList[UbInListId],
                layoutComputeInUb,
                layoutTileCompute);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListId]));
            UbInListId = UbInListIdNext;
        }

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));

        auto layoutDstY = layoutY.GetTileLayout(TensorCoord(n_actual));
        auto layoutComputeInUb = layoutY.GetTileLayout(TensorCoord(n_actual));
        vecCopyUbToGm(gmZ, UbYTensorList[UbOutListId], layoutDstY, layoutComputeInUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> UbATensorList[STAGES];
    // AscendC::LocalTensor<ElementX> UbXTensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorList[STAGES];
    // AscendC::LocalTensor<ElementX> UbWTensorList[STAGES];

    // Multi-stage event id list
    int32_t UbInAEventList[STAGES];
    // int32_t UbInXEventList[STAGES];
    int32_t UbOutEventList[STAGES];

    // The id of current stage
    uint32_t UbOutListId{0};
    uint32_t UbInListId{0};

    uint32_t m_actual, n_actual, x_actual, y_actual;
    uint32_t TileMRound, TileNRound;
    uint32_t strideA;

    // TileVmad tileVmad;
    TileSliceSum tileSliceSum;
    MatrixCopyGmToUb matrixCopyGmToUb;
    VecCopyGmToUb vecCopyGmToUb;
    VecCopyUbToGm vecCopyUbToGm;
};




} // namespace FTSelf::Gemv::Block

#endif // CATLASS_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
