#ifndef FTSELF_CATLASS_GEMM_BLOCK_BLOCK_MMAD_HPP
#define FTSELF_CATLASS_GEMM_BLOCK_BLOCK_MMAD_HPP

#include_next "catlass/gemm/block/block_mmad.hpp"

#include "FT_self/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "FT_self/gemv/helper.hpp"

namespace FTSelf::Gemm::Block {
/*
template<
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockMmadPreload<
    Catlass::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_
>
*/

template<
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class AType,
    class BType,
    class CType,
    class BiasType = void,
    class TileCopy = Catlass::Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
    class TileMmad = Catlass::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmadPreload {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadPreload is not implemented for this DispatchPolicy");
};


/*
template <
    // Tag indicating architecture
    class ArchTag,
    // GemmType for A matrix operand;
    class AType,
    // GemmType for B matrix operand;
    class BType,
    // GemmType for C matrix operand;
    class CType,
    // GemvType for X vector operand;
    class XType,
    // GemvType for Y vector operand;
    class YType,
    // GemmType for Bias operand;
    class BiasType = void
>
*/
// using TileMmadAIC = Gemm::Tile::TileMmad<typename GEMVAICDispatchPolicy::ArchTag, XType, CType, BiasType>;
// class TileMmadforFT = Catlass::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag,>
//  = FTSelf::Gemv::helper::FT_L02L1_TYPE::FIX_PIPE

/*
 template <
    class ArchTag,
    class AType,
    class BType,
    class CType,
    class XType,
    class YType,
    class BiasType = void,
    FTSelf::Gemv::helper::FT_L02L1_TYPE COPY_TYPE_ = FTSelf::Gemv::helper::FT_L02L1_TYPE::FIX_PIPE
>
struct TileCopyFT
*/

template<
    class DispatchPolicy,
    FTSelf::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    FTSelf::Gemv::helper::FT_L02L1_TYPE COPY_TYPE_,
    class L1TileShape,
    class L0TileShape,
    class L0TileShapeforFT,
    class AType,
    class BType,
    class CType,
    class XType,
    class YType,
    class BiasType = void,
    class TileCopyFT = FTSelf::Gemm::Tile::TileCopyFT<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, YType, BiasType, COPY_TYPE_>,
    class TileMmad = Catlass::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmadFTNOSPLIT {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

/*
struct BlockMmadFTABeNoSplitK<
    Catlass::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L1TileShapeforFT_,
    L0TileShape_,
    L0TileShapeforFT_,
    AType_,
    BType_,
    CType_,
    XType_,
    YType_,
    BiasType_,
    TileCopyFTABonAic_,
    TileMmad_
>
*/
template<
    class DispatchPolicy,
    class L1TileShape,
    class L1TileShapeforFT,
    class L0TileShape,
    class L0TileShapeforFT,
    class AType,
    class BType,
    class CType,
    class XType,
    class YType,
    class BiasType = void,
    class TileCopyFTABonAic = FTSelf::Gemm::Tile::TileCopyFTABonAic<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, YType, BiasType>,
    class TileMmad = Catlass::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmadFTABeNoSplitK {
    /*
    L1TileShape_,
    L1TileShapeforFT_,
    L0TileShape_,
    L0TileShapeforFT_,
    */
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

template<
    class DispatchPolicy,
    class L1TileShape,
    class L1TileShapeforFT,
    class L0TileShape,
    class L0TileShapeforFT,
    class AType,
    class BType,
    class CType,
    class XType,
    class XColType,
    class YType,
    class BiasType = void,
    class TileCopyFTABonAicAuged = FTSelf::Gemm::Tile::TileCopyFTABonAicAuged<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, XColType, YType, BiasType>,
    class TileMmad = Catlass::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmadFTABeAugedNoSplitK{
    /*
    template <
    // Tag indicating architecture
    class ArchTag,
    // GemmType for A matrix operand;
    class AType,
    // GemmType for B matrix operand;
    class BType,
    // GemmType for C matrix operand;
    class CType,
    // GemvType for X vector operand;
    class XType,
    class XTypeCol,
    // GemvType for Y vector operand;
    class YType,
    // GemmType for Bias operand;
    class BiasType = void
    >
    struct TileCopyFTABonAicAuged {
    */
    /*
    Catlass::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L1TileShapeforFT_,
    L0TileShape_,
    L0TileShapeforFT_,
    AType_,
    BType_,
    CType_,
    XType_,
    XColType_,
    YType_,
    BiasType_,
    TileCopyFTABonAicAuged_,
    TileMmad_
    */
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadFTABeAugedNoSplitK is not implemented for this DispatchPolicy");
};


template<
    class DispatchPolicy,
    class L1TileShape,
    class L1TileShapeforFT,
    class L0TileShape,
    class L0TileShapeforFT,
    class AType,
    class BType,
    class CType,
    class XType,
    class XColType,
    class YType,
    class BiasType = void,
    class TileCopyFTABonAicAuged = FTSelf::Gemm::Tile::TileCopyFTABonAicAuged<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, XColType, YType, BiasType>,
    class TileMmad = Catlass::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmadFTABeAugedNoSplitKGemv{
    /*
    template <
    // Tag indicating architecture
    class ArchTag,
    // GemmType for A matrix operand;
    class AType,
    // GemmType for B matrix operand;
    class BType,
    // GemmType for C matrix operand;
    class CType,
    // GemvType for X vector operand;
    class XType,
    class XTypeCol,
    // GemvType for Y vector operand;
    class YType,
    // GemmType for Bias operand;
    class BiasType = void
    >
    struct TileCopyFTABonAicAuged {
    */
    /*
    Catlass::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L1TileShapeforFT_,
    L0TileShape_,
    L0TileShapeforFT_,
    AType_,
    BType_,
    CType_,
    XType_,
    XColType_,
    YType_,
    BiasType_,
    TileCopyFTABonAicAuged_,
    TileMmad_
    */
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadFTABeAugedNoSplitKGemv is not implemented for this DispatchPolicy");
};

template<
    class DispatchPolicy,
    class L1TileShape,
    class L1TileShapeforFT,
    class L0TileShape,
    class L0TileShapeforFT,
    class AType,
    class BType,
    class CType,
    class XType,
    class XColType,
    class YType,
    class BiasType = void,
    class TileCopyFTABonAicAuged = FTSelf::Gemm::Tile::TileCopyFTABonAicAuged<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, XColType, YType, BiasType>,
    class TileMmad = Catlass::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmadFTABeAugedNoSplitKRobust{
    /*
    template <
    // Tag indicating architecture
    class ArchTag,
    // GemmType for A matrix operand;
    class AType,
    // GemmType for B matrix operand;
    class BType,
    // GemmType for C matrix operand;
    class CType,
    // GemvType for X vector operand;
    class XType,
    class XTypeCol,
    // GemvType for Y vector operand;
    class YType,
    // GemmType for Bias operand;
    class BiasType = void
    >
    struct TileCopyFTABonAicAuged {
    */
    /*
    Catlass::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L1TileShapeforFT_,
    L0TileShape_,
    L0TileShapeforFT_,
    AType_,
    BType_,
    CType_,
    XType_,
    XColType_,
    YType_,
    BiasType_,
    TileCopyFTABonAicAuged_,
    TileMmad_
    */
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadFTABeAugedNoSplitKRobust is not implemented for this DispatchPolicy");
};

template<
    class DispatchPolicy,
    FTSelf::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    FTSelf::Gemv::helper::FT_L02L1_TYPE COPY_TYPE_,
    class L1TileShape,
    class L0TileShape,
    class L0TileShapeforFT,
    class AType,
    class BType,
    class CType,
    class XType,
    class YType,
    class BiasType = void,
    class TileCopyFT = FTSelf::Gemm::Tile::TileCopyFT<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, YType, BiasType, COPY_TYPE_>,
    class TileMmad = Catlass::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmadFTSpiltK {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadFTSpiltK is not implemented for this DispatchPolicy");
};

template<
    class DispatchPolicy,
    class L1TileShapeforFT,
    class L0TileShapeforFT,
    class AType,
    class BType,
    class CType,
    class XType,
    class YType,
    class BiasType,
    class TileCopyFTABonAic = FTSelf::Gemm::Tile::TileCopyFTABonAic<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, YType, BiasType>,
    class TileMmad = Catlass::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, XType, BiasType>
>
struct BlockMmadSpecABeNoSplitK {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadSpecABeNoSplitK is not implemented for this DispatchPolicy");
};

template<
    class DispatchPolicy,
    class L1TileShapeforFT,
    class L0TileShapeforFT,
    class AType,
    class BType,
    class CType,
    class XType,
    class YType,
    class BiasType,
    class TileCopyFTABonAic = FTSelf::Gemm::Tile::TileCopyFTABonAic<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, YType, BiasType>,
    class TileMmad = Catlass::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, XType, BiasType>
>
struct BlockMmadSpecABeNoSplitKRobust {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadSpecABeNoSplitK is not implemented for this DispatchPolicy");
};

/*
struct BlockMmadFTABeNoSplitKRobust<
    Catlass::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L1TileShapeforFT_,
    L0TileShape_,
    L0TileShapeforFT_,
    AType_,
    BType_,
    CType_,
    XType_,
    YType_,
    BiasType_,
    TileCopyFTABonAic_,
    TileMmad_
>
*/

template<
    class DispatchPolicy,
    class L1TileShape,
    class L1TileShapeforFT,
    class L0TileShape,
    class L0TileShapeforFT,
    class AType,
    class BType,
    class CType,
    class XType,
    class YType,
    class BiasType = void,
    class TileCopyFTABonAic = FTSelf::Gemm::Tile::TileCopyFTABonAic<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, YType, BiasType>,
    class TileMmad = Catlass::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmadFTABeNoSplitKRobust {
    /*
    L1TileShape_,
    L1TileShapeforFT_,
    L0TileShape_,
    L0TileShapeforFT_,
    */
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class AType,
    class BType,
    class CType,
    class BiasType = void,
    class TileCopy = Catlass::Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
    class TileMmad = Catlass::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmadFault {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadFault is not implemented for this DispatchPolicy");
};


} // namespace FTSelf::Gemm::Block

#include "FT_self/gemm/block/block_mmad_pingpong_fault_abe_spec_no_splitk_robust.hpp"
#include "FT_self/gemm/block/block_mmad_pingpong_preload.hpp"


#endif // FTSELF_CATLASS_GEMM_BLOCK_BLOCK_MMAD_HPP
