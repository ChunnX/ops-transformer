
/*
一共开发两个功能，即1）进行作差；2）进行异或比较
在这里因为是简单的比较两个变量，所以不需要区分矩阵和向量了
直接加载一块连续的GM空间，使用AIV 核，
对于每个AIV核，结果直接存放到GM的相应位置上即可。

在这里我定义Z为输出的位置，X 为操作数向量1，Y为操作数向量2，即比较两个操作数
我这里先给出
*/

#ifndef CATLASS_GEMV_BLOCK_LARGE_LOCAL_COPARE_AIV_HPP
#define CATLASS_GEMV_BLOCK_LARGE_LOCAL_COPARE_AIV_HPP

#include "../../catlass.hpp"
#include "catlass/arch/resource_aiv.hpp"
#include "../../coord.hpp"
#include "../../gemv_coord.hpp"
#include "../helper.hpp"
#include "../../gemm/helper.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/alignment.hpp"
#include "../../gemm/dispatch_policy.hpp"
#include "../tile/tile_fault_compare.hpp"
#include "block_gemv.hpp"

namespace Vecself::Gemv::Block {

using namespace Catlass;

template <
    Vecself::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    class UBTileShape_,
    class ZType_,
    class XType_,
    class YType_,
    class TileCopy_
>
struct BlockCompare <
    Gemm::GemvAtlasA2,
    COMP_TYPE_,
    UBTileShape_,
    ZType_,
    XType_,
    YType_,
    TileCopy_
> {
public:
    // Type Aliases
    using DispatchPolicy = Gemm::GemvAtlasA2;
    using FT_COMP_TYPE = Vecself::Gemv::helper::FT_COMP_TYPE;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using UBTileShape = UBTileShape_;

    using ElementZ = typename ZType_::Element;
    using LayoutZ = typename ZType_::Layout;

    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;

    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;

    // using LayoutW = Catlass::layout::VectorLayout;

    using ElementXoR = uint16_t;
    using ElementComp = int32_t;
    using ElementSub =  ElementX;

    using ElementWork = typename std::conditional<
        (COMP_TYPE_ == FT_COMP_TYPE::XOR),
        uint16_t,
        typename std::conditional<(COMP_TYPE_ == FT_COMP_TYPE::COMPARE), int32_t, ElementX>::type
    >::type;

    using VecCopyGmToUbX = typename TileCopy_::VecCopyGmToUbX;
    using VecCopyGmToUbY = typename TileCopy_::VecCopyGmToUbY;

    using VecCopyUbToGmZ = typename TileCopy_::VecCopyUbToGmZ;
    using VecCopyUbToGmW = typename TileCopy_::VecCopyUbToGmW;

    /*
    <
    /// Tag indicating architecture
    Vecself::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    class ArchTag,
    class ZType,
    class XType,
    class YType
    >
    */

    using TileCompare = Vecself::Gemv::Tile::TileFaultVcompare<COMP_TYPE_, ArchTag,
                                        ZType_, XType_, YType_>;

    using UBAlignHelper = Vecself::Gemv::helper::UBAlignHelper<ElementX>;
    using TensorCoord = layout::VectorLayout::TensorCoord;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t Xbuf_SIZE_ = 64 * 1024;
    static constexpr uint32_t Ybuf_SIZE_ = 64 * 1024;
    // static constexpr uint32_t workspace_SIZE_ = 16 * 1024;
    static constexpr uint32_t Zbuf_SIZE_ = 64 * 1024;
    static constexpr FT_COMP_TYPE COMP_TYPE = COMP_TYPE_;
    // static constexpr uint32_t Abuf_SIZE_ = 128 * 1024;
    // static constexpr uint32_t Xbuf_SIZE_ = 16 * 1024;
    // static constexpr uint32_t Ybuf_SIZE_ = 16 * 1024;
    // static constexpr uint32_t workspace_SIZE_ = 32 * 1024;



    CATLASS_DEVICE
    BlockCompare() {}

    /// Construct
    CATLASS_DEVICE
    BlockCompare(Arch::Resource<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbXOffset = UBufAddrStart;
        uint32_t UbYOffset = UBufAddrStart + Xbuf_SIZE_;
        // uint32_t UbWOffset = UBufAddrStart + Xbuf_SIZE_ + Ybuf_SIZE_;
        // + workspace_SIZE_
        uint32_t UbZOffset = UBufAddrStart + Xbuf_SIZE_ + Ybuf_SIZE_;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages

            // AscendC::LocalTensor<ElementZ> UbZTensorList[STAGES];
            // AscendC::LocalTensor<ElementX> UbXTensorList[STAGES];
            // AscendC::LocalTensor<ElementY> UbYTensorList[STAGES];
            // AscendC::LocalTensor<ElementWork> UbWTensorList[STAGES];

            UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset + i * (Xbuf_SIZE_ / 2));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));

            // UbWTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementWork>(UbWOffset + i * (workspace_SIZE_ / 2));

            UbZTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementZ>(UbZOffset + i * (Zbuf_SIZE_ / 2));

            // Assign event ID for each stages
            // int32_t UbInYEventList[STAGES];
            // int32_t UbInXEventList[STAGES];
            // int32_t UbOutEventList[STAGES];

            UbInXEventList[i] = i;
            UbInYEventList[i] = i + STAGES;
            UbOutEventList[i] = i + STAGES * 2;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInYEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
        }
    }

    /// Construct
    CATLASS_DEVICE
    BlockCompare(FTSelf::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbXOffset = UBufAddrStart;
        uint32_t UbYOffset = UBufAddrStart + Xbuf_SIZE_;
        // uint32_t UbWOffset = UBufAddrStart + Xbuf_SIZE_ + Ybuf_SIZE_;
        // + workspace_SIZE_
        uint32_t UbZOffset = UBufAddrStart + Xbuf_SIZE_ + Ybuf_SIZE_;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages

            // AscendC::LocalTensor<ElementZ> UbZTensorList[STAGES];
            // AscendC::LocalTensor<ElementX> UbXTensorList[STAGES];
            // AscendC::LocalTensor<ElementY> UbYTensorList[STAGES];
            // AscendC::LocalTensor<ElementWork> UbWTensorList[STAGES];

            UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset + i * (Xbuf_SIZE_ / 2));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));

            // UbWTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementWork>(UbWOffset + i * (workspace_SIZE_ / 2));

            UbZTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementZ>(UbZOffset + i * (Zbuf_SIZE_ / 2));

            // Assign event ID for each stages
            // int32_t UbInYEventList[STAGES];
            // int32_t UbInXEventList[STAGES];
            // int32_t UbOutEventList[STAGES];

            UbInXEventList[i] = i;
            UbInYEventList[i] = i + STAGES;
            UbOutEventList[i] = i + STAGES * 2;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInYEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockCompare()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInYEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
        }
    }

    // AscendC::GlobalTensor<ElementWork> const &gmWork, LayoutW const &layoutW,
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementX> const &gmX, LayoutX const &layoutX,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        AscendC::GlobalTensor<ElementX> const &gmXNext,
        AscendC::GlobalTensor<ElementY> const &gmYNext,
        AscendC::GlobalTensor<ElementZ> const &gmZ, LayoutZ const &layoutZ,
        GemvCoord const &actualShape, GemvCoord const &actualShapeNext,
        bool isFirstBlock, bool hasNextBlock, bool OutputWorkspace,
        ElementX threshold)
    {
        uint32_t actuallength = actualShape.m() * actualShape.n();
        uint32_t actualNextlength = actualShapeNext.m() * actualShapeNext.n();

        TileRound = RoundUp(UBTileShape::M*UBTileShape::N, UBAlignHelper::ALIGN);
        // TileNRound = RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);
        // strideA = layoutA.stride(1) * TileNRound;
        n_actual = (actuallength < TileRound) ? actuallength : TileRound;

        // main loop
        uint32_t Nloop = CeilDiv(actuallength, TileRound);
        uint32_t NloopNext = CeilDiv(actualNextlength, TileRound);

        uint32_t tile_output_actual = TileRound / 8;
        uint32_t tile_work_actual = TileRound * sizeof(ElementX) / sizeof(ElementWork);

        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(UbOutEventList[UbListId]));

            n_actual = (LoopIdx == Nloop - 1) ? (actuallength - LoopIdx * TileRound) : TileRound;
            y_actual = n_actual;
            x_actual = n_actual;

            uint32_t UbListIdNext = (UbListId + 1 < STAGES) ? (UbListId + 1) : 0;
            uint32_t LoopIdxNext{0};
            uint32_t n_actual_next{0};
            uint32_t y_actual_next{0};
            uint32_t x_actual_next{0};

            if (LoopIdx == 0 && isFirstBlock){
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbListId]));
                vecCopyGmToUbX(UbXTensorList[UbListId], gmX[LoopIdx * TileRound], n_actual);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbListId]));

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInYEventList[UbListId]));
                vecCopyGmToUbY(UbYTensorList[UbListId], gmY[LoopIdx * TileRound], n_actual);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInYEventList[UbListId]));
            }

            if (LoopIdx < Nloop - 1) {
                LoopIdxNext = LoopIdx + 1;
                n_actual_next =
                    (LoopIdxNext == Nloop - 1) ? (actuallength - LoopIdxNext * TileRound) : TileRound;

                y_actual_next = n_actual_next;
                x_actual_next = n_actual_next;

                // Get L1 tensor for next stage
                auto vecXTensor = UbXTensorList[UbListIdNext];
                auto vecYTensor = UbYTensorList[UbListIdNext];

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbListIdNext]));
                vecCopyGmToUbX(vecXTensor, gmX[LoopIdxNext * TileRound], x_actual_next);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbListIdNext]));

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInYEventList[UbListIdNext]));
                vecCopyGmToUbY(vecYTensor, gmY[LoopIdxNext * TileRound], y_actual_next);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInYEventList[UbListIdNext]));
            }

            if ((LoopIdx == Nloop - 1) && hasNextBlock){

                LoopIdxNext = 0;
                n_actual_next =
                    (LoopIdxNext == NloopNext - 1) ? (actualNextlength - LoopIdxNext * TileRound) : TileRound;

                y_actual_next = n_actual_next;
                x_actual_next = n_actual_next;
                // Get L1 tensor for next stage
                auto vecXTensor = UbXTensorList[UbListIdNext];
                auto vecYTensor = UbYTensorList[UbListIdNext];

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbListIdNext]));
                vecCopyGmToUbX(vecXTensor, gmXNext[LoopIdxNext * TileRound], x_actual_next);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbListIdNext]));

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInYEventList[UbListIdNext]));
                vecCopyGmToUbY(vecYTensor, gmYNext[LoopIdxNext * TileRound], y_actual_next);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInYEventList[UbListIdNext]));
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbListId]));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInYEventList[UbListId]));

            auto layoutComputeInUb = layoutX.GetTileLayout(MakeCoord(TileRound));
            auto layoutTileCompute = layoutX.GetTileLayout(MakeCoord(n_actual));

            AscendC::PipeBarrier<PIPE_MTE2>();

            // UbWTensorList[UbListId],
            tileCompare(
                UbZTensorList[UbListId],
                UbXTensorList[UbListId],
                UbYTensorList[UbListId],
                layoutComputeInUb, layoutTileCompute, threshold);

            /*
            (
            AscendC::LocalTensor<ElementZ> dstTensor,
            AscendC::LocalTensor<ElementX> srcTensor_x,
            AscendC::LocalTensor<ElementY> srcTensor_y,
            AscendC::LocalTensor<ElementWIn> workSpaceTensor,
            LayoutDst const &layoutDst, LayoutSrc const &layoutSrc, ElementX threshold
            )
            */

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInYEventList[UbListId]));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbListId]));

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)UbOutEventList[UbListId]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)UbOutEventList[UbListId]);


            uint32_t n_output_actual = n_actual / 8;
            // auto layoutDstZ = layoutZ.GetTileLayout(TensorCoord(n_actual));
            // auto layoutComputeInUb = layoutZ.GetTileLayout(TensorCoord(y_actual));
            // vecCopyUbToGmZ(gmZ, UbZTensorList[UbListId], layoutDstY, layoutComputeInUb);
            auto layoutDstZ = layoutZ.GetTileLayout(TensorCoord(n_output_actual));
            auto layoutComputeInUbZ = layoutZ.GetTileLayout(TensorCoord(n_output_actual));
            // AscendC::printf("hhu",UbZTensorList[UbListId].GetValue(0));
            // AscendC::printf("%f\n",static_cast<float>(1.0));
            vecCopyUbToGmZ(gmZ[tile_output_actual * LoopIdx], UbZTensorList[UbListId], layoutDstZ, layoutComputeInUbZ);


            // if(OutputWorkspace){

            //     uint32_t n_work_actual = n_actual * sizeof(ElementX) / sizeof(ElementWork);

            //     auto layoutDstW = layoutW.GetTileLayout(TensorCoord(n_work_actual));
            //     auto layoutWInUb = layoutW.GetTileLayout(TensorCoord(n_work_actual));

            //     vecCopyUbToGmW(gmWork[tile_work_actual * LoopIdx], UbWTensorList[UbListId], layoutDstW, layoutWInUb);
            // }

            AscendC::PipeBarrier<PIPE_MTE3>();

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(UbOutEventList[UbListId]));
            UbListId = UbListIdNext;
            // n_actual = n_actual_next;
        }
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementZ> UbZTensorList[STAGES];
    AscendC::LocalTensor<ElementX> UbXTensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorList[STAGES];
    // AscendC::LocalTensor<ElementWork> UbWTensorList[STAGES];

    // Multi-stage event id list
    int32_t UbInYEventList[STAGES];
    int32_t UbInXEventList[STAGES];
    int32_t UbOutEventList[STAGES];

    // The id of current stage
    uint32_t UbListId{0};
    // uint32_t UbListId{0};

    uint32_t n_actual, x_actual, y_actual;
    uint32_t TileRound;

    VecCopyGmToUbX vecCopyGmToUbX;
    VecCopyGmToUbY vecCopyGmToUbY;

    VecCopyUbToGmZ vecCopyUbToGmZ;
    // VecCopyUbToGmW vecCopyUbToGmW;

    // Tile Compare
    TileCompare tileCompare;
};


// Vecself::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
template <
    class UBTileShape_,
    class ZType_,
    class XType_,
    class YType_,
    class TileCopy_
>
struct BlockCompare <
    Gemm::GemvAtlasA2,
    Vecself::Gemv::helper::FT_COMP_TYPE::RSUB,
    UBTileShape_,
    ZType_,
    XType_,
    YType_,
    TileCopy_
> {
public:
    // Type Aliases
    using DispatchPolicy = Gemm::GemvAtlasA2;
    using FT_COMP_TYPE = Vecself::Gemv::helper::FT_COMP_TYPE;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using UBTileShape = UBTileShape_;

    using ElementZ = typename ZType_::Element;
    using LayoutZ = typename ZType_::Layout;

    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;

    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;

    // using LayoutW = Catlass::layout::VectorLayout;


    using ElementWork = ElementX;

    using VecCopyGmToUbX = typename TileCopy_::VecCopyGmToUbX;
    using VecCopyGmToUbY = typename TileCopy_::VecCopyGmToUbY;
    using VecCopyGmToUbW = typename TileCopy_::VecCopyGmToUbW;

    using VecCopyUbToGmZ = typename TileCopy_::VecCopyUbToGmZ;
    using VecCopyUbToGmW = typename TileCopy_::VecCopyUbToGmW;

    /*
    <
    /// Tag indicating architecture
    Vecself::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    class ArchTag,
    class ZType,
    class XType,
    class YType
    >
    */

    using TileCompare = Vecself::Gemv::Tile::TileFaultVcompare<FT_COMP_TYPE::RSUB, ArchTag,
                                        ZType_, XType_, YType_>;


    using UBAlignHelper = Vecself::Gemv::helper::UBAlignHelper<ElementX>;
    using TensorCoord = layout::VectorLayout::TensorCoord;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t Xbuf_SIZE_ = 60 * 1024;
    static constexpr uint32_t Ybuf_SIZE_ = 60 * 1024;
    static constexpr uint32_t Wbuf_SIZE_ = 60 * 1024;
    // static constexpr uint32_t workspace_SIZE_ = 16 * 1024;
    static constexpr uint32_t Zbuf_SIZE_ = 12 * 1024;
    static constexpr FT_COMP_TYPE COMP_TYPE = FT_COMP_TYPE::RSUB;
    // static constexpr uint32_t Abuf_SIZE_ = 128 * 1024;
    // static constexpr uint32_t Xbuf_SIZE_ = 16 * 1024;
    // static constexpr uint32_t Ybuf_SIZE_ = 16 * 1024;
    // static constexpr uint32_t workspace_SIZE_ = 32 * 1024;



    CATLASS_DEVICE
    BlockCompare() {}

    /// Construct
    CATLASS_DEVICE
    BlockCompare(Arch::Resource<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbXOffset = UBufAddrStart;
        uint32_t UbYOffset = UBufAddrStart + Xbuf_SIZE_;
        uint32_t UbWOffset = UBufAddrStart + Xbuf_SIZE_ + Ybuf_SIZE_;
        // + workspace_SIZE_
        uint32_t UbZOffset = UBufAddrStart + Xbuf_SIZE_ + Ybuf_SIZE_ + Wbuf_SIZE_;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages

            // AscendC::LocalTensor<ElementZ> UbZTensorList[STAGES];
            // AscendC::LocalTensor<ElementX> UbXTensorList[STAGES];
            // AscendC::LocalTensor<ElementY> UbYTensorList[STAGES];
            // AscendC::LocalTensor<ElementWork> UbWTensorList[STAGES];

            UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset + i * (Xbuf_SIZE_ / 2));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));

            UbWTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementWork>(UbWOffset + i * (Wbuf_SIZE_ / 2));

            UbZTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementZ>(UbZOffset + i * (Zbuf_SIZE_ / 2));

            // Assign event ID for each stages
            // int32_t UbInYEventList[STAGES];
            // int32_t UbInXEventList[STAGES];
            // int32_t UbOutEventList[STAGES];

            UbInXEventList[i] = i;
            UbInYEventList[i] = i + STAGES;
            UbInWEventList[i] = i + STAGES * 2;
            UbOutEventList[i] = i; //  + STAGES * 2

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInYEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInWEventList[i]);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
        }
    }

    /// Construct
    CATLASS_DEVICE
    BlockCompare(FTSelf::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbXOffset = UBufAddrStart;
        uint32_t UbYOffset = UBufAddrStart + Xbuf_SIZE_;
        uint32_t UbWOffset = UBufAddrStart + Xbuf_SIZE_ + Ybuf_SIZE_;
        // uint32_t UbWOffset = UBufAddrStart + Xbuf_SIZE_ + Ybuf_SIZE_;
        // + workspace_SIZE_
        uint32_t UbZOffset = UBufAddrStart + Xbuf_SIZE_ + Ybuf_SIZE_ + Wbuf_SIZE_;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages

            // AscendC::LocalTensor<ElementZ> UbZTensorList[STAGES];
            // AscendC::LocalTensor<ElementX> UbXTensorList[STAGES];
            // AscendC::LocalTensor<ElementY> UbYTensorList[STAGES];
            // AscendC::LocalTensor<ElementWork> UbWTensorList[STAGES];

            UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset + i * (Xbuf_SIZE_ / 2));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));

            UbWTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementWork>(UbWOffset + i * (Wbuf_SIZE_ / 2));
            // UbWTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementWork>(UbWOffset + i * (workspace_SIZE_ / 2));

            UbZTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementZ>(UbZOffset + i * (Zbuf_SIZE_ / 2));

            // Assign event ID for each stages
            // int32_t UbInYEventList[STAGES];
            // int32_t UbInXEventList[STAGES];
            // int32_t UbOutEventList[STAGES];

            UbInXEventList[i] = i;
            UbInYEventList[i] = i + STAGES;
            UbInWEventList[i] = i + STAGES * 2;
            UbOutEventList[i] = i;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInYEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInWEventList[i]);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockCompare()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInYEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInWEventList[i]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
        }
    }

    // AscendC::GlobalTensor<ElementWork> const &gmWork, LayoutW const &layoutW,
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementX> const &gmX, LayoutX const &layoutX,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        AscendC::GlobalTensor<ElementWork> const &gmW, LayoutX const &layoutW,
        AscendC::GlobalTensor<ElementX> const &gmXNext,
        AscendC::GlobalTensor<ElementY> const &gmYNext,
        AscendC::GlobalTensor<ElementWork> const &gmWNext,
        AscendC::GlobalTensor<ElementZ> const &gmZ, LayoutZ const &layoutZ,
        GemvCoord const &actualShape, GemvCoord const &actualShapeNext,
        bool isFirstBlock, bool hasNextBlock, bool OutputWorkspace,
        ElementX threshold)
    {
        uint32_t actuallength = actualShape.m() * actualShape.n();
        uint32_t actualNextlength = actualShapeNext.m() * actualShapeNext.n();

        TileRound = RoundUp(UBTileShape::M*UBTileShape::N, UBAlignHelper::ALIGN);
        // TileNRound = RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);
        // strideA = layoutA.stride(1) * TileNRound;
        n_actual = (actuallength < TileRound) ? actuallength : TileRound;

        // main loop
        uint32_t Nloop = CeilDiv(actuallength, TileRound);
        uint32_t NloopNext = CeilDiv(actualNextlength, TileRound);

        uint32_t tile_output_actual = TileRound / 8;
        uint32_t tile_work_actual = TileRound * sizeof(ElementX) / sizeof(ElementWork);



        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(UbOutEventList[UbListId]));

            n_actual = (LoopIdx == Nloop - 1) ? (actuallength - LoopIdx * TileRound) : TileRound;
            y_actual = n_actual;
            x_actual = n_actual;

            uint32_t UbListIdNext = (UbListId + 1 < STAGES) ? (UbListId + 1) : 0;
            uint32_t LoopIdxNext{0};
            uint32_t n_actual_next{0};
            uint32_t y_actual_next{0};
            uint32_t x_actual_next{0};
            uint32_t w_actual_next{0};

            if (LoopIdx == 0 && isFirstBlock){
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbListId]));
                vecCopyGmToUbX(UbXTensorList[UbListId], gmX[LoopIdx * TileRound], n_actual);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbListId]));

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInYEventList[UbListId]));
                vecCopyGmToUbY(UbYTensorList[UbListId], gmY[LoopIdx * TileRound], n_actual);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInYEventList[UbListId]));

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInWEventList[UbListId]));
                vecCopyGmToUbW(UbWTensorList[UbListId], gmW[LoopIdx * TileRound], n_actual);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInWEventList[UbListId]));
            }

            if (LoopIdx < Nloop - 1) {
                LoopIdxNext = LoopIdx + 1;
                n_actual_next =
                    (LoopIdxNext == Nloop - 1) ? (actuallength - LoopIdxNext * TileRound) : TileRound;

                y_actual_next = n_actual_next;
                x_actual_next = n_actual_next;
                w_actual_next = n_actual_next;

                // Get L1 tensor for next stage
                auto vecXTensor = UbXTensorList[UbListIdNext];
                auto vecYTensor = UbYTensorList[UbListIdNext];
                auto vecWTensor = UbWTensorList[UbListIdNext];

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbListIdNext]));
                vecCopyGmToUbX(vecXTensor, gmX[LoopIdxNext * TileRound], x_actual_next);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbListIdNext]));

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInYEventList[UbListIdNext]));
                vecCopyGmToUbY(vecYTensor, gmY[LoopIdxNext * TileRound], y_actual_next);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInYEventList[UbListIdNext]));

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInWEventList[UbListIdNext]));
                vecCopyGmToUbW(vecWTensor, gmW[LoopIdxNext * TileRound], w_actual_next);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInWEventList[UbListIdNext]));
            }

            if ((LoopIdx == Nloop - 1) && hasNextBlock){

                LoopIdxNext = 0;
                n_actual_next =
                    (LoopIdxNext == NloopNext - 1) ? (actualNextlength - LoopIdxNext * TileRound) : TileRound;

                y_actual_next = n_actual_next;
                x_actual_next = n_actual_next;
                w_actual_next = n_actual_next;

                // Get L1 tensor for next stage
                auto vecXTensor = UbXTensorList[UbListIdNext];
                auto vecYTensor = UbYTensorList[UbListIdNext];
                auto vecWTensor = UbWTensorList[UbListIdNext];

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbListIdNext]));
                vecCopyGmToUbX(vecXTensor, gmXNext[LoopIdxNext * TileRound], x_actual_next);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbListIdNext]));

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInYEventList[UbListIdNext]));
                vecCopyGmToUbY(vecYTensor, gmYNext[LoopIdxNext * TileRound], y_actual_next);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInYEventList[UbListIdNext]));

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInWEventList[UbListIdNext]));
                vecCopyGmToUbW(vecWTensor, gmWNext[LoopIdxNext * TileRound], w_actual_next);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInWEventList[UbListIdNext]));
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbListId]));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInYEventList[UbListId]));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInWEventList[UbListId]));

            auto layoutComputeInUb = layoutX.GetTileLayout(MakeCoord(TileRound));
            auto layoutTileCompute = layoutX.GetTileLayout(MakeCoord(n_actual));

            AscendC::PipeBarrier<PIPE_MTE2>();

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
            tileCompare(
                UbZTensorList[UbListId],
                UbXTensorList[UbListId],
                UbYTensorList[UbListId],
                UbWTensorList[UbListId],
                layoutComputeInUb, layoutTileCompute, threshold);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInYEventList[UbListId]));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbListId]));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInWEventList[UbListId]));


            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)UbOutEventList[UbListId]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)UbOutEventList[UbListId]);

            uint32_t n_output_actual = n_actual / 8;
            // auto layoutDstZ = layoutZ.GetTileLayout(TensorCoord(n_actual));
            // auto layoutComputeInUb = layoutZ.GetTileLayout(TensorCoord(y_actual));
            // vecCopyUbToGmZ(gmZ, UbZTensorList[UbListId], layoutDstY, layoutComputeInUb);
            auto layoutDstZ = layoutZ.GetTileLayout(TensorCoord(n_output_actual));
            auto layoutComputeInUbZ = layoutZ.GetTileLayout(TensorCoord(n_output_actual));
            // AscendC::printf("hhu",UbZTensorList[UbListId].GetValue(0));
            // AscendC::printf("%f\n",static_cast<float>(1.0));
            vecCopyUbToGmZ(gmZ[tile_output_actual * LoopIdx], UbZTensorList[UbListId], layoutDstZ, layoutComputeInUbZ);


            // if(OutputWorkspace){

            //     uint32_t n_work_actual = n_actual * sizeof(ElementX) / sizeof(ElementWork);

            //     auto layoutDstW = layoutW.GetTileLayout(TensorCoord(n_work_actual));
            //     auto layoutWInUb = layoutW.GetTileLayout(TensorCoord(n_work_actual));

            //     vecCopyUbToGmW(gmWork[tile_work_actual * LoopIdx], UbWTensorList[UbListId], layoutDstW, layoutWInUb);
            // }

            AscendC::PipeBarrier<PIPE_MTE3>();

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(UbOutEventList[UbListId]));
            UbListId = UbListIdNext;
            // n_actual = n_actual_next;
        }
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementZ> UbZTensorList[STAGES];

    AscendC::LocalTensor<ElementX> UbXTensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorList[STAGES];
    AscendC::LocalTensor<ElementWork> UbWTensorList[STAGES];

    // Multi-stage event id list
    int32_t UbInYEventList[STAGES];
    int32_t UbInXEventList[STAGES];
    int32_t UbInWEventList[STAGES];

    int32_t UbOutEventList[STAGES];

    // The id of current stage
    uint32_t UbListId{0};
    // uint32_t UbListId{0};

    uint32_t n_actual, x_actual, y_actual;
    uint32_t TileRound;

    VecCopyGmToUbX vecCopyGmToUbX;
    VecCopyGmToUbY vecCopyGmToUbY;
    VecCopyGmToUbW vecCopyGmToUbW;

    VecCopyUbToGmZ vecCopyUbToGmZ;
    // VecCopyUbToGmW vecCopyUbToGmW;

    // Tile Compare
    TileCompare tileCompare;
};


} // namespace Vecself::Gemv::Block

#endif // CATLASS_GEMV_BLOCK_LARGE_LOCAL_COPARE_AIV_HPP
