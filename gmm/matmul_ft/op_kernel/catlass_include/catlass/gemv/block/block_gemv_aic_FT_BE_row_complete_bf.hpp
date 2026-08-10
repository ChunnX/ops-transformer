#ifndef CATLASS_GEMV_BLOCK_BLOCK_AIC_FT_BE_ROWCOMPLETE_HPP_BF
#define CATLASS_GEMV_BLOCK_BLOCK_AIC_FT_BE_ROWCOMPLETE_HPP_BF

#include "../../catlass.hpp"
#include "../../arch/resource.hpp"
#include "../../coord.hpp"
#include "../../gemm/dispatch_policy.hpp"
#include "../../gemm/tile/tile_copy.hpp"
#include "../../gemm/tile/tile_mmad.hpp"
#include "../../gemv_coord.hpp"

#include "../helper.hpp"
#include "../../gemm/helper.hpp"
#include "../tile/tile_copy.hpp"

// Vecself::Gemv::helper::FT_AIC_BE_SCHEME  BeScheme_,
namespace Vecself::Gemv::Block {

using namespace Catlass;
template <
    bool ENABLE_UNIT_FLAG_,
    bool ENABLE_SHUFFLE_K_,
    class UBBlockShape_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class XType_,
    class YType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockFTGemvBe<
    Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>,
    Vecself::Gemv::helper::FT_AIC_BE_SCHEME::ROWCOMPLETE_BF,
    UBBlockShape_,
    L1TileShape_,
    L0TileShape_,
    AType_,
    XType_,
    YType_,
    BiasType_,
    TileCopy_,
    TileMmad_
> {
public:
    // Type Aliases
    using DispatchPolicy = Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using UBBlockShape = UBBlockShape_;

    using FT_AIC_BE_SCHEME = Vecself::Gemv::helper::FT_AIC_BE_SCHEME;

    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;

    using ElementYforAIV = float;

    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;

    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;

    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;

    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;

    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using CopyL0CToGmforBFAIV = typename TileCopy_::CopyL0CToGmforBFAIV;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;
    using LayoutXInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutAInL1 = typename CopyL1ToL0B::LayoutSrc;

    using LayoutXInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutAInL0 = typename CopyL1ToL0B::LayoutDst;

    using LayoutYInL0 = layout::zN;
    using FT_ENC_TYPE = Vecself::Gemv::helper::FT_ENC_TYPE;

    using L1AAlignHelper = Vecself::Gemv::helper::L1AlignHelper<ElementA, LayoutA>;
    // using L1AColAlignHelper = Vecself::Gemv::helper::L1AlignHelper<ElementA, LayoutACol>;
    using L1XAlignHelper = Vecself::Gemv::helper::L1AlignHelper<ElementX, LayoutX>;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool ENABLE_SHUFFLE_K = DispatchPolicy::ENABLE_SHUFFLE_K;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr FT_AIC_BE_SCHEME BE_SCHEME = Vecself::Gemv::helper::FT_AIC_BE_SCHEME::ROWCOMPLETE_BF;

    static constexpr uint32_t L1B_SIZE = L1TileShape::M * L1TileShape::N * sizeof(ElementA);
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;


    static constexpr uint32_t L0C_TILE_NUM = (L1TileShape::M + L0TileShape::M - 1) / L0TileShape::M;
    // L0C_SIZE / L0C_TILE_SIZE / sizeof(ElementAccumulator);
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_SIZE / STAGES;

    // static constexpr FT_ENC_TYPE ENC_TYPE = ENC_TYPE_;

    static constexpr uint32_t MAX_L1TILE_SIZE = (L1TileShape::M > L1TileShape::N) ? L1TileShape::M : L1TileShape::N;
    static constexpr uint32_t MAX_L0TILE_SIZE = (L0TileShape::M > L0TileShape::N) ? L0TileShape::M : L0TileShape::N;


    static constexpr uint32_t L1A_SIZE = 16 * L1TileShape::N * sizeof(ElementX);


    static constexpr uint32_t L0A_TILE_SIZE = L1XAlignHelper::M_ALIGNED * L0TileShape::N * sizeof(ElementX);

    static constexpr uint32_t L0C_TILE_SIZE = L1XAlignHelper::M_ALIGNED * L0TileShape::M * sizeof(ElementAccumulator);
    // Check L1TileShape
    static_assert((L1A_SIZE * STAGES + L1B_SIZE * STAGES) <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");

    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::M * L0TileShape::N * sizeof(ElementA);
    static_assert((L0A_TILE_SIZE * STAGES) <= L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert((L0B_TILE_SIZE * STAGES) <= L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert((L0C_TILE_SIZE * STAGES) <= L0C_SIZE, "L0TileShape exceeding the L0C space!");

    static_assert(L0TileShape::M == L1TileShape::M,
        "The situation where the basic blocks of L0 for FT and L0 for MMA differ on the m axes is not supported yet");

    static_assert((UBBlockShape::N % L1TileShape::N) == 0,
        "The situation where the basic Tile of UB and L1 for MMA disalign at n axes is not supported yet");



    /// Construct
    CATLASS_DEVICE
    BlockFTGemvBe(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart = 0)
    {
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1A_SIZE * STAGES;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementX>(l1AOffset + L1A_SIZE * i);
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BOffset + L1B_SIZE * i);
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementX>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementA>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensorList[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);

            l1AEventList[i] = i;
            l1BEventList[i] = i + STAGES;
            l0AEventList[i] = i;
            l0BEventList[i] = i + STAGES;
            l0CEventList[i] = i + STAGES * 2;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
        }

    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockFTGemvBe()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
        }
    }

    /// Perform a block-scoped vector-matrix multiply-accumulate
    CATLASS_DEVICE
    void op_with_addition_copy(
        AscendC::GlobalTensor<ElementX> const& gmBlockX, LayoutX const& layoutX,
        AscendC::GlobalTensor<ElementA> const& gmBlockA, LayoutA const& layoutA,
        AscendC::GlobalTensor<ElementY> const& gmBlockY,
        AscendC::GlobalTensor<ElementYforAIV> const& gmBlockYforAIV,
        LayoutY const& layoutY,
        AscendC::GlobalTensor<ElementX> const& gmNextBlockX,
        AscendC::GlobalTensor<ElementA> const& gmNextBlockA,
        GemvCoord const& actualShape, GemvCoord const& actualShapeNext,
        bool isFirstBlock, bool hasNextBlock)
    {
        TileMRound = L1TileShape::M;
        TileNRound = L1TileShape::N;

        BlockMRound = UBBlockShape::M;
        BlockNRound = UBBlockShape::N;

        strideACol = layoutA.stride(1) * TileNRound;
        strideARow = layoutA.stride(0) * TileMRound;

        // strideOut = layoutA.stride(0);
        strideOut = layoutA.shape(0);

        uint32_t NloopBlock = 1;
        uint32_t MBLoopIdx = 0;

        uint32_t MloopBlock = 1;
        uint32_t NBLoopIdx = 0;

        m_actual_total = (actualShape.m() < BlockMRound) ? actualShape.m() : BlockMRound;
        n_actual_total = (actualShape.n() < BlockNRound) ? actualShape.n() : BlockNRound;

        uint32_t m_actual_total_next = (actualShapeNext.m() < BlockMRound) ? actualShapeNext.m() : BlockMRound;
        uint32_t n_actual_total_next = (actualShapeNext.n() < BlockNRound) ? actualShapeNext.n() : BlockNRound;

        splitNnum = (n_actual_total + TileNRound - 1) / TileNRound;
        tileMnum = (m_actual_total + TileMRound - 1) / TileMRound;

        uint32_t splitNnumNext = (n_actual_total_next + TileNRound - 1) / TileNRound;
        uint32_t tileMnumNext = (m_actual_total_next + TileMRound - 1) / TileMRound;

        // actualShape.m()
        auto layoutXInL1 = LayoutXInL1::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, L1TileShape::N);
        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::N);
        auto layoutInL0C = LayoutYInL0::MakeLayoutInL0C(MatrixCoord(L1XAlignHelper::M_ALIGNED, TileMRound));

        uint32_t Nloop = splitNnum;
        uint32_t Mloop = tileMnum;

        y_actual_total = m_actual_total;
        x_actual_total = n_actual_total;

        uint32_t nTileCount = splitNnum;
        uint32_t mTileCount = tileMnum;
        // CeilDiv<L1TileShape::N>(actualShape.n());
        uint32_t nTileCountNext = splitNnumNext;
        uint32_t mTileCountNext = tileMnumNext;

        // Optimize points：ShuffleK
        uint32_t startTileIdx = 0;
        if constexpr (ENABLE_SHUFFLE_K_) {
            startTileIdx = AscendC::GetBlockIdx();
        }
        uint32_t firstTileIdx = startTileIdx % nTileCount;
        uint32_t lastTileIdx = (startTileIdx + nTileCount - 1) % nTileCount;
        uint32_t firstTileIdxNext = startTileIdx % nTileCountNext;

        uint32_t A_row_offset = MBLoopIdx * BlockMRound;
        uint32_t A_col_offset = NBLoopIdx * BlockNRound;
        uint32_t A_block_offset = A_row_offset * layoutA.stride(0) + A_col_offset * layoutA.stride(1);

        m_actual = (m_actual_total < TileMRound) ? m_actual_total : TileMRound;
        n_actual =
                (firstTileIdx < nTileCount - 1) ? TileNRound : (n_actual_total - firstTileIdx * TileNRound);

        // main loop
        for (uint32_t nLoopIdx = 0; nLoopIdx < nTileCount; nLoopIdx++) {
            uint32_t shuffleKIdx = (startTileIdx + nLoopIdx) % nTileCount;
            n_actual =
                (shuffleKIdx < nTileCount - 1) ? TileNRound : (n_actual_total - shuffleKIdx * TileNRound);
            m_actual = (m_actual_total < TileMRound) ? m_actual_total : TileMRound;

            uint32_t nRound = RoundUp<L1AAlignHelper::N_ALIGNED>(n_actual);
            uint32_t mRound = RoundUp<L1XAlignHelper::M_ALIGNED>(m_actual);

            uint32_t TileY_Slice_offset = shuffleKIdx;

            if (shuffleKIdx == firstTileIdx && isFirstBlock) {
                MatrixCoord gmTileAOffset{0, shuffleKIdx * L1TileShape::N};
                uint32_t gmTilexOffset{shuffleKIdx * L1TileShape::N};

                auto gmTileA = gmBlockA[layoutA.GetOffset(gmTileAOffset)];
                auto gmTilex = gmBlockX[gmTilexOffset];

                // load first vector x tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                auto layoutTilex = layoutX.GetTileLayout(MakeCoord(nRound));
                copyGmToL1A(l1ATensorList[l1ListId], gmTilex, layoutXInL1, layoutTilex);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

                // load first matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, nRound));
                copyGmToL1B(l1BTensorList[l1ListId], gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

                // AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                // AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

                // AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                // AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
            }

            for(uint32_t mLoopIdx=0; mLoopIdx < mTileCount; mLoopIdx++){

                uint32_t TileY_Row_offset = mLoopIdx * TileMRound;
                MatrixCoord gmTileYOffset{TileY_Slice_offset, TileY_Row_offset};

                uint32_t l1ListIdNext = (l1ListId + 1) % STAGES;
                uint32_t n_actual_next{0};
                uint32_t nRoundNext{0};
                uint32_t m_actual_next{0};
                uint32_t mRoundNext{0};
                uint32_t m_actual_now = (mLoopIdx < (mTileCount - 1)) ? TileMRound : m_actual_total - mLoopIdx * TileMRound;

                if(mLoopIdx < (mTileCount - 1)){
                    uint32_t mLoopIdxNext = mLoopIdx + 1;
                    m_actual_next = (mLoopIdxNext < (mTileCount - 1)) ? TileMRound : m_actual_total - mLoopIdxNext * TileMRound;
                    n_actual_next = (shuffleKIdx < nTileCount - 1) ? TileNRound : (n_actual_total - shuffleKIdx * TileNRound);

                    nRoundNext = RoundUp<L1AAlignHelper::N_ALIGNED>(n_actual_next);
                    mRoundNext = RoundUp<L1XAlignHelper::M_ALIGNED>(m_actual_next);

                    uint32_t ATileMOffsetNext = mLoopIdxNext * TileMRound;
                    MatrixCoord gmTileAOffsetNext{ATileMOffsetNext, shuffleKIdx * L1TileShape::N};
                    uint32_t gmTilexOffsetNext{shuffleKIdx * L1TileShape::N};

                    auto gmTileANext = gmBlockA[layoutA.GetOffset(gmTileAOffsetNext)];
                    auto gmTilexNext = gmBlockX[gmTilexOffsetNext];

                    // load first vector x tile from GM to L1
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                    auto layoutTilexNext = layoutX.GetTileLayout(MakeCoord(nRoundNext));
                    copyGmToL1A(l1ATensorList[l1ListIdNext], gmTilexNext, layoutXInL1, layoutTilexNext);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                    // load first matrix A tile from GM to L1
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                    auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(m_actual_next, nRoundNext));
                    copyGmToL1B(l1BTensorList[l1ListIdNext], gmTileANext, layoutAInL1, layoutTileANext);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                }else{
                    // preload next tile from GM to L1
                    if (shuffleKIdx != lastTileIdx) {
                        uint32_t mLoopIdxNext = 0;
                        uint32_t shuffleKIdxNext = (startTileIdx + nLoopIdx + 1) % nTileCount;
                        n_actual_next = (shuffleKIdxNext < nTileCount - 1) ? L1TileShape::N
                                                                 : (n_actual_total - shuffleKIdxNext * L1TileShape::N);

                        m_actual_next = (mLoopIdxNext < (mTileCount - 1)) ? TileMRound : m_actual_total - mLoopIdxNext * TileMRound;

                        nRoundNext = RoundUp<L1AAlignHelper::N_ALIGNED>(n_actual_next);
                        mRoundNext = RoundUp<L1XAlignHelper::M_ALIGNED>(m_actual_next);

                        // Get L1 tensor
                        auto l1ATensorNext = l1ATensorList[l1ListIdNext];
                        auto l1BTensorNext = l1BTensorList[l1ListIdNext];

                        // Get GM tile
                        uint32_t ATileMOffsetNext = 0;
                        MatrixCoord gmTileAOffsetNext{ATileMOffsetNext, shuffleKIdxNext * L1TileShape::N};
                        uint32_t gmTilexOffsetNext{shuffleKIdxNext * L1TileShape::N};

                        auto gmTileANext = gmBlockA[layoutA.GetOffset(gmTileAOffsetNext)];
                        //
                        auto gmTilexNext = gmBlockX[gmTilexOffsetNext];

                        // load vector x tile from GM to L1
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                        auto layoutTilexNext = layoutX.GetTileLayout(MakeCoord(nRoundNext));

                        copyGmToL1A(l1ATensorNext, gmTilexNext, layoutXInL1, layoutTilexNext);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                        // load Matrix A tile from GM to L1
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                        auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(m_actual_next, nRoundNext));

                        copyGmToL1B(l1BTensorNext, gmTileANext, layoutAInL1, layoutTileANext);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                    }
                    if (shuffleKIdx == lastTileIdx && hasNextBlock) {
                        // Get L1 tensor
                        auto l1ATensorNext = l1ATensorList[l1ListIdNext];
                        auto l1BTensorNext = l1BTensorList[l1ListIdNext];
                        uint32_t mLoopIdxNext = 0;
                        m_actual_next = (mLoopIdxNext < (mTileCountNext - 1)) ? TileMRound : m_actual_total_next - mLoopIdxNext * TileMRound;

                        // Get GM tensor for next stage
                        n_actual_next= (firstTileIdxNext < nTileCountNext - 1)
                            ? L1TileShape::N : (n_actual_total_next - firstTileIdxNext * L1TileShape::N);

                        nRoundNext = RoundUp<L1AAlignHelper::N_ALIGNED>(n_actual_next);
                        mRoundNext = RoundUp<L1XAlignHelper::M_ALIGNED>(m_actual_next);

                        uint32_t ATileMOffsetNext = 0;
                        // Get GM tile
                        MatrixCoord gmTileAOffsetNext{ATileMOffsetNext, firstTileIdxNext * L1TileShape::N};
                        uint32_t gmTilexOffsetNext{firstTileIdxNext * L1TileShape::N};

                        auto gmTileANext = gmNextBlockA[layoutA.GetOffset(gmTileAOffsetNext)];
                        auto gmTilexNext = gmNextBlockX[gmTilexOffsetNext];

                        // load vector x tile from GM to L1
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);

                        auto layoutTilexNext = layoutX.GetTileLayout(MakeCoord(nRoundNext));

                        copyGmToL1A(l1ATensorNext, gmTilexNext, layoutXInL1, layoutTilexNext);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                        // load Matrix A tile from GM to L1
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                        auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(m_actual_next, nRoundNext));

                        copyGmToL1B(l1BTensorNext, gmTileANext, layoutAInL1, layoutTileANext);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                    }
                }

                // get L1 Tensor for current stage
                auto l1ATensor = l1ATensorList[l1ListId];
                auto l1BTensor = l1BTensorList[l1ListId];

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

                uint32_t nRound = RoundUp<L1AAlignHelper::N_ALIGNED>(n_actual);
                uint32_t nPartLoop = CeilDiv<L0TileShape::N>(n_actual);

                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l1ListId]);
                auto l0CTile = l0CTensorList[l1ListId];

                for (uint32_t nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                    uint32_t nPartActual =
                        (nPartIdx < nPartLoop - 1) ? L0TileShape::N : (n_actual - nPartIdx * L0TileShape::N);

                    // Locate the current tile on L0A
                    auto l0ATile = l0ATensorList[l0AListId];
                    LayoutXInL0 layoutxInL0 =
                        LayoutXInL0::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, nPartActual);

                    MatrixCoord l1xOffset{0, nPartIdx * L0TileShape::N};
                    auto l1ATile = l1ATensor[layoutXInL1.GetOffset(l1xOffset)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    // Load current tile from L1 to L0A
                    copyL1ToL0A(l0ATile, l1ATile, layoutxInL0, layoutXInL1);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);

                    // Locate the current tile on L0B
                    auto l0BTile = l0BTensorList[l0BListId];
                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(L0TileShape::M, nPartActual);

                    MatrixCoord l1AOffset{0, nPartIdx * L0TileShape::N};
                    auto l1BTile = l1BTensor[layoutAInL1.GetOffset(l1AOffset)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    // Load current tile from L1 to L0B
                    copyL1ToL0B(l0BTile, l1BTile, layoutAInL0, layoutAInL1);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l1ListId]);
                    // l0BEventList[l0BListId]

                    // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                    // (nLoopIdx == 0) &&
                    bool initC = ((nPartIdx == 0));
                    // L0TileShape::M
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l1ListId]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);
                    tileMmad(l0CTile, l0ATile, l0BTile, L1XAlignHelper::M_ALIGNED, m_actual_now, nPartActual, initC);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);

                    l0AListId = (l0AListId + 1) % STAGES;
                    l0BListId = (l0BListId + 1) % STAGES;
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);

                l1ListId = l1ListIdNext;

                LayoutY layoutBlock = layoutY.GetTileLayout(MakeCoord(uint32_t(1), m_actual_now));

                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l1ListId]);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l1ListId]);

                // uint32_t gmTileYOffsetInBlock = layoutY.GetOffset(gmTileYOffset);

                copyL0CToGm(gmBlockY[layoutY.GetOffset(gmTileYOffset)], l0CTile, layoutBlock, layoutInL0C);
                copyL0CToGmforBFAIV(gmBlockYforAIV[layoutY.GetOffset(gmTileYOffset)], l0CTile, layoutBlock, layoutInL0C);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l1ListId]);
            }
        }
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementX> const& gmBlockX, LayoutX const& layoutX,
        AscendC::GlobalTensor<ElementA> const& gmBlockA, LayoutA const& layoutA,
        AscendC::GlobalTensor<ElementY> const& gmBlockY,
        LayoutY const& layoutY,
        AscendC::GlobalTensor<ElementX> const& gmNextBlockX,
        AscendC::GlobalTensor<ElementA> const& gmNextBlockA,
        GemvCoord const& actualShape, GemvCoord const& actualShapeNext,
        bool isFirstBlock, bool hasNextBlock)
    {
        TileMRound = L1TileShape::M;
        TileNRound = L1TileShape::N;

        BlockMRound = UBBlockShape::M;
        BlockNRound = UBBlockShape::N;

        strideACol = layoutA.stride(1) * TileNRound;
        strideARow = layoutA.stride(0) * TileMRound;

        // strideOut = layoutA.stride(0);
        strideOut = layoutA.shape(0);

        uint32_t NloopBlock = 1;
        uint32_t MBLoopIdx = 0;

        uint32_t MloopBlock = 1;
        uint32_t NBLoopIdx = 0;

        m_actual_total = (actualShape.m() < BlockMRound) ? actualShape.m() : BlockMRound;
        n_actual_total = (actualShape.n() < BlockNRound) ? actualShape.n() : BlockNRound;

        uint32_t m_actual_total_next = (actualShapeNext.m() < BlockMRound) ? actualShapeNext.m() : BlockMRound;
        uint32_t n_actual_total_next = (actualShapeNext.n() < BlockNRound) ? actualShapeNext.n() : BlockNRound;

        splitNnum = (n_actual_total + TileNRound - 1) / TileNRound;
        tileMnum = (m_actual_total + TileMRound - 1) / TileMRound;

        uint32_t splitNnumNext = (n_actual_total_next + TileNRound - 1) / TileNRound;
        uint32_t tileMnumNext = (m_actual_total_next + TileMRound - 1) / TileMRound;

        // actualShape.m()
        auto layoutXInL1 = LayoutXInL1::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, L1TileShape::N);
        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::N);
        auto layoutInL0C = LayoutYInL0::MakeLayoutInL0C(MatrixCoord(L1XAlignHelper::M_ALIGNED, TileMRound));

        uint32_t Nloop = splitNnum;
        uint32_t Mloop = tileMnum;

        y_actual_total = m_actual_total;
        x_actual_total = n_actual_total;

        uint32_t nTileCount = splitNnum;
        uint32_t mTileCount = tileMnum;
        // CeilDiv<L1TileShape::N>(actualShape.n());
        uint32_t nTileCountNext = splitNnumNext;
        uint32_t mTileCountNext = tileMnumNext;

        // Optimize points：ShuffleK
        uint32_t startTileIdx = 0;
        if constexpr (ENABLE_SHUFFLE_K_) {
            startTileIdx = AscendC::GetBlockIdx();
        }
        uint32_t firstTileIdx = startTileIdx % nTileCount;
        uint32_t lastTileIdx = (startTileIdx + nTileCount - 1) % nTileCount;
        uint32_t firstTileIdxNext = startTileIdx % nTileCountNext;

        uint32_t A_row_offset = MBLoopIdx * BlockMRound;
        uint32_t A_col_offset = NBLoopIdx * BlockNRound;
        uint32_t A_block_offset = A_row_offset * layoutA.stride(0) + A_col_offset * layoutA.stride(1);

        m_actual = (m_actual_total < TileMRound) ? m_actual_total : TileMRound;
        n_actual =
                (firstTileIdx < nTileCount - 1) ? TileNRound : (n_actual_total - firstTileIdx * TileNRound);

        // main loop
        for (uint32_t nLoopIdx = 0; nLoopIdx < nTileCount; nLoopIdx++) {
            uint32_t shuffleKIdx = (startTileIdx + nLoopIdx) % nTileCount;
            n_actual =
                (shuffleKIdx < nTileCount - 1) ? TileNRound : (n_actual_total - shuffleKIdx * TileNRound);
            m_actual = (m_actual_total < TileMRound) ? m_actual_total : TileMRound;

            uint32_t nRound = RoundUp<L1AAlignHelper::N_ALIGNED>(n_actual);
            uint32_t mRound = RoundUp<L1XAlignHelper::M_ALIGNED>(m_actual);

            uint32_t TileY_Slice_offset = shuffleKIdx;

            if (shuffleKIdx == firstTileIdx && isFirstBlock) {
                MatrixCoord gmTileAOffset{0, shuffleKIdx * L1TileShape::N};
                uint32_t gmTilexOffset{shuffleKIdx * L1TileShape::N};

                auto gmTileA = gmBlockA[layoutA.GetOffset(gmTileAOffset)];
                auto gmTilex = gmBlockX[gmTilexOffset];

                // load first vector x tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                auto layoutTilex = layoutX.GetTileLayout(MakeCoord(nRound));
                copyGmToL1A(l1ATensorList[l1ListId], gmTilex, layoutXInL1, layoutTilex);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

                // load first matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, nRound));
                copyGmToL1B(l1BTensorList[l1ListId], gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

                // AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                // AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

                // AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                // AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
            }

            for(uint32_t mLoopIdx=0; mLoopIdx < mTileCount; mLoopIdx++){

                uint32_t TileY_Row_offset = mLoopIdx * TileMRound;
                MatrixCoord gmTileYOffset{TileY_Slice_offset, TileY_Row_offset};

                uint32_t l1ListIdNext = (l1ListId + 1) % STAGES;
                uint32_t n_actual_next{0};
                uint32_t nRoundNext{0};
                uint32_t m_actual_next{0};
                uint32_t mRoundNext{0};
                uint32_t m_actual_now = (mLoopIdx < (mTileCount - 1)) ? TileMRound : m_actual_total - mLoopIdx * TileMRound;

                if(mLoopIdx < (mTileCount - 1)){
                    uint32_t mLoopIdxNext = mLoopIdx + 1;
                    m_actual_next = (mLoopIdxNext < (mTileCount - 1)) ? TileMRound : m_actual_total - mLoopIdxNext * TileMRound;
                    n_actual_next = (shuffleKIdx < nTileCount - 1) ? TileNRound : (n_actual_total - shuffleKIdx * TileNRound);

                    nRoundNext = RoundUp<L1AAlignHelper::N_ALIGNED>(n_actual_next);
                    mRoundNext = RoundUp<L1XAlignHelper::M_ALIGNED>(m_actual_next);

                    uint32_t ATileMOffsetNext = mLoopIdxNext * TileMRound;
                    MatrixCoord gmTileAOffsetNext{ATileMOffsetNext, shuffleKIdx * L1TileShape::N};
                    uint32_t gmTilexOffsetNext{shuffleKIdx * L1TileShape::N};

                    auto gmTileANext = gmBlockA[layoutA.GetOffset(gmTileAOffsetNext)];
                    auto gmTilexNext = gmBlockX[gmTilexOffsetNext];

                    // load first vector x tile from GM to L1
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                    auto layoutTilexNext = layoutX.GetTileLayout(MakeCoord(nRoundNext));
                    copyGmToL1A(l1ATensorList[l1ListIdNext], gmTilexNext, layoutXInL1, layoutTilexNext);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                    // load first matrix A tile from GM to L1
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                    auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(m_actual_next, nRoundNext));
                    copyGmToL1B(l1BTensorList[l1ListIdNext], gmTileANext, layoutAInL1, layoutTileANext);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                }else{
                    // preload next tile from GM to L1
                    if (shuffleKIdx != lastTileIdx) {
                        uint32_t mLoopIdxNext = 0;
                        uint32_t shuffleKIdxNext = (startTileIdx + nLoopIdx + 1) % nTileCount;
                        n_actual_next = (shuffleKIdxNext < nTileCount - 1) ? L1TileShape::N
                                                                 : (n_actual_total - shuffleKIdxNext * L1TileShape::N);

                        m_actual_next = (mLoopIdxNext < (mTileCount - 1)) ? TileMRound : m_actual_total - mLoopIdxNext * TileMRound;

                        nRoundNext = RoundUp<L1AAlignHelper::N_ALIGNED>(n_actual_next);
                        mRoundNext = RoundUp<L1XAlignHelper::M_ALIGNED>(m_actual_next);

                        // Get L1 tensor
                        auto l1ATensorNext = l1ATensorList[l1ListIdNext];
                        auto l1BTensorNext = l1BTensorList[l1ListIdNext];

                        // Get GM tile
                        uint32_t ATileMOffsetNext = 0;
                        MatrixCoord gmTileAOffsetNext{ATileMOffsetNext, shuffleKIdxNext * L1TileShape::N};
                        uint32_t gmTilexOffsetNext{shuffleKIdxNext * L1TileShape::N};

                        auto gmTileANext = gmBlockA[layoutA.GetOffset(gmTileAOffsetNext)];
                        //
                        auto gmTilexNext = gmBlockX[gmTilexOffsetNext];

                        // load vector x tile from GM to L1
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                        auto layoutTilexNext = layoutX.GetTileLayout(MakeCoord(nRoundNext));

                        copyGmToL1A(l1ATensorNext, gmTilexNext, layoutXInL1, layoutTilexNext);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                        // load Matrix A tile from GM to L1
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                        auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(m_actual_next, nRoundNext));

                        copyGmToL1B(l1BTensorNext, gmTileANext, layoutAInL1, layoutTileANext);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                    }
                    if (shuffleKIdx == lastTileIdx && hasNextBlock) {
                        // Get L1 tensor
                        auto l1ATensorNext = l1ATensorList[l1ListIdNext];
                        auto l1BTensorNext = l1BTensorList[l1ListIdNext];
                        uint32_t mLoopIdxNext = 0;
                        m_actual_next = (mLoopIdxNext < (mTileCountNext - 1)) ? TileMRound : m_actual_total_next - mLoopIdxNext * TileMRound;

                        // Get GM tensor for next stage
                        n_actual_next= (firstTileIdxNext < nTileCountNext - 1)
                            ? L1TileShape::N : (n_actual_total_next - firstTileIdxNext * L1TileShape::N);

                        nRoundNext = RoundUp<L1AAlignHelper::N_ALIGNED>(n_actual_next);
                        mRoundNext = RoundUp<L1XAlignHelper::M_ALIGNED>(m_actual_next);

                        uint32_t ATileMOffsetNext = 0;
                        // Get GM tile
                        MatrixCoord gmTileAOffsetNext{ATileMOffsetNext, firstTileIdxNext * L1TileShape::N};
                        uint32_t gmTilexOffsetNext{firstTileIdxNext * L1TileShape::N};

                        auto gmTileANext = gmNextBlockA[layoutA.GetOffset(gmTileAOffsetNext)];
                        auto gmTilexNext = gmNextBlockX[gmTilexOffsetNext];

                        // load vector x tile from GM to L1
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);

                        auto layoutTilexNext = layoutX.GetTileLayout(MakeCoord(nRoundNext));

                        copyGmToL1A(l1ATensorNext, gmTilexNext, layoutXInL1, layoutTilexNext);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                        // load Matrix A tile from GM to L1
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                        auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(m_actual_next, nRoundNext));

                        copyGmToL1B(l1BTensorNext, gmTileANext, layoutAInL1, layoutTileANext);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                    }
                }

                // get L1 Tensor for current stage
                auto l1ATensor = l1ATensorList[l1ListId];
                auto l1BTensor = l1BTensorList[l1ListId];

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

                uint32_t nRound = RoundUp<L1AAlignHelper::N_ALIGNED>(n_actual);
                uint32_t nPartLoop = CeilDiv<L0TileShape::N>(n_actual);

                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l1ListId]);
                auto l0CTile = l0CTensorList[l1ListId];

                for (uint32_t nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                    uint32_t nPartActual =
                        (nPartIdx < nPartLoop - 1) ? L0TileShape::N : (n_actual - nPartIdx * L0TileShape::N);

                    // Locate the current tile on L0A
                    auto l0ATile = l0ATensorList[l0AListId];
                    LayoutXInL0 layoutxInL0 =
                        LayoutXInL0::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, nPartActual);

                    MatrixCoord l1xOffset{0, nPartIdx * L0TileShape::N};
                    auto l1ATile = l1ATensor[layoutXInL1.GetOffset(l1xOffset)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    // Load current tile from L1 to L0A
                    copyL1ToL0A(l0ATile, l1ATile, layoutxInL0, layoutXInL1);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);

                    // Locate the current tile on L0B
                    auto l0BTile = l0BTensorList[l0BListId];
                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(L0TileShape::M, nPartActual);

                    MatrixCoord l1AOffset{0, nPartIdx * L0TileShape::N};
                    auto l1BTile = l1BTensor[layoutAInL1.GetOffset(l1AOffset)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    // Load current tile from L1 to L0B
                    copyL1ToL0B(l0BTile, l1BTile, layoutAInL0, layoutAInL1);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l1ListId]);
                    // l0BEventList[l0BListId]

                    // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                    // (nLoopIdx == 0) &&
                    bool initC = ((nPartIdx == 0));
                    // L0TileShape::M
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l1ListId]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);
                    tileMmad(l0CTile, l0ATile, l0BTile, L1XAlignHelper::M_ALIGNED, m_actual_now, nPartActual, initC);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);

                    l0AListId = (l0AListId + 1) % STAGES;
                    l0BListId = (l0BListId + 1) % STAGES;
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);

                l1ListId = l1ListIdNext;

                LayoutY layoutBlock = layoutY.GetTileLayout(MakeCoord(uint32_t(1), m_actual_now));

                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l1ListId]);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l1ListId]);

                // uint32_t gmTileYOffsetInBlock = layoutY.GetOffset(gmTileYOffset);

                copyL0CToGm(gmBlockY[layoutY.GetOffset(gmTileYOffset)], l0CTile, layoutBlock, layoutInL0C);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l1ListId]);
            }
        }
    }


    /// Perform a block-scoped vector-matrix multiply-accumulate
    // AscendC::GlobalTensor<ElementX> const& gmBlockX, LayoutX const& layoutX,
    // AscendC::GlobalTensor<ElementX> const& gmNextBlockX,



    /// Perform a block-scoped vector-matrix multiply-accumulate


protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementX> l1ATensorList[STAGES];
    AscendC::LocalTensor<ElementA> l1BTensorList[STAGES];
    AscendC::LocalTensor<ElementX> l0ATensorList[STAGES];
    AscendC::LocalTensor<ElementA> l0BTensorList[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[STAGES];

    // Multi-stage event id list
    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];
    int32_t l0AEventList[STAGES];
    int32_t l0BEventList[STAGES];
    int32_t l0CEventList[STAGES];

    // The id of current stage
    uint32_t l1ListId{0};

    uint32_t l0AListId{0};
    uint32_t l0BListId{0};

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;


    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;


    CopyL0CToGm copyL0CToGm;
    CopyL0CToGmforBFAIV copyL0CToGmforBFAIV;

    uint32_t m_actual, n_actual, x_actual, y_actual;
    uint32_t m_actual_total, n_actual_total, x_actual_total, y_actual_total;
    uint32_t TileMRound, TileNRound;
    uint32_t BlockMRound, BlockNRound;
    uint32_t strideACol, strideARow;
    uint32_t strideOut;
    uint32_t splitNnum;
    uint32_t tileMnum;
};

} // namespace Vecself::Gemv::Block

#endif // CATLASS_GEMV_BLOCK_BLOCK_AIC_FT_HPP
