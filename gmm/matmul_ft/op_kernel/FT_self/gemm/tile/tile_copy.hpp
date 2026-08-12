#ifndef FTSELF_CATLASS_GEMM_TILE_TILE_COPY_HPP
#define FTSELF_CATLASS_GEMM_TILE_TILE_COPY_HPP

#include_next "catlass/gemm/tile/tile_copy.hpp"

#include "FT_self/gemm/tile/atlasa2/copy_l1_to_l0b.hpp"
#include "FT_self/gemv/helper.hpp"
#include "FT_self/gemm/tile/copy_l0c_to_gm.hpp"

namespace FTSelf::Gemm::Tile {
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
    class BiasType = void,
    FTSelf::Gemv::helper::FT_L02L1_TYPE COPY_TYPE_ = FTSelf::Gemv::helper::FT_L02L1_TYPE::FIX_PIPE
>
struct TileCopyFT {
    using FT_L02L1_TYPE = FTSelf::Gemv::helper::FT_L02L1_TYPE;
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;

    using ElementCInL1 = typename XType::Element;

    using CTypeForL1 = Catlass::Gemm::GemmType<ElementCInL1,Catlass::layout::zN>;
    using L1XType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, CTypeForL1>::L1AType;
    using L1CTypeforFT = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, CTypeForL1>::L1BType;

    using L0XType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, CTypeForL1>::L0AType;
    using L0CTypeforFT = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, CTypeForL1>::L0BType;

    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, AType>;
    using CopyGmToL1B = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, BType>;

    using CopyGmToL1X = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, XType, L1XType>;

    using CopyL1ToL0A = Catlass::Gemm::Tile::CopyL1ToL0A<
        ArchTag, typename Catlass::Gemm::helper::L1ATypeSelector<AType>::L1AType>;

    using CopyL1ToL0B = FTSelf::Gemm::Tile::CopyL1ToL0B<
        ArchTag, typename Catlass::Gemm::helper::L1BTypeSelector<BType>::L1BType>;

    using CopyL1ToL0X = Catlass::Gemm::Tile::CopyL1ToL0A<ArchTag, L1XType, L0XType>;

    using CopyL1ToL0CforFT = FTSelf::Gemm::Tile::CopyL1ToL0B<ArchTag, L1CTypeforFT, L0CTypeforFT>;

    using CopyL0CToGm = Catlass::Gemm::Tile::CopyL0CToGm<ArchTag,ElementAccumulator,CType>;

    using CopyL0CToGmforFT = Catlass::Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, YType>;

    /*
    template <
        class ElementAccumulator_,
        class ElementDst_,
        bool ReluEnable_
    >
    struct CopyL0CToL1<Catlass::Arch::AtlasA2,
                   ElementAccumulator_,
                   Gemm::GemmType<ElementDst_, layout::zN>,
                   ScaleGranularity::NO_QUANT,
                   ReluEnable_>

    template <
    class ArchTag,
    class ElementAccumulator,
    class L1Type,
    FTSelf::Gemv::helper::FT_L02L1_TYPE CopyType,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false
    >
    struct CopyL0CToL1
    */

    using CopyL0CToL1forFT =  FTSelf::Gemm::Tile::CopyL0CToL1<ArchTag,
        ElementAccumulator, CTypeForL1, COPY_TYPE_>;

    using BiasTypeSelector = Catlass::Gemm::helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;

    using CopyGmToL1Bias = std::conditional_t<std::is_same_v<BiasType,void>,void,
                            Catlass::Gemm::Tile::CopyGmToL1<
                                ArchTag,
                                typename BiasTypeSelector::GMBiasType,
                                typename BiasTypeSelector::L1BiasType>>;

    using CopyL1ToBT = std::conditional_t<std::is_same_v<BiasType,void>,void,
                            Catlass::Gemm::Tile::CopyL1ToBT<ArchTag,
                                typename BiasTypeSelector::L1BiasType,
                                typename BiasTypeSelector::L0BiasType>>;


};

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
struct TileCopyFTABonAic {
    using FT_L02L1_TYPE = FTSelf::Gemv::helper::FT_L02L1_TYPE;
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;

    using ElementX = typename XType::Element;
    using LayoutX = typename XType::Layout;

    using CopyGmToL1B = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, BType>;


    // using LayoutXInL1 = typename std::conditional<
    //     std::is_same<LayoutX, Catlass::layout::ColumnMajor>::value,
    //     Catlass::layout::nZ,
    //     Catlass::layout::zN>::type;

    // using ElementCInL1 = typename XType::Element;
    using LayoutVX = Catlass::layout::VectorLayout;
    using VXType = Catlass::Gemm::GemmType<ElementX, LayoutVX>;
    using L1XType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<VXType, AType>::L1AType;

    // using LayoutXInL1 = typename CopyGmToL1B::LayoutDst;

    // using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
    // using L1XType = Catlass::Gemm::GemmType<ElementX,LayoutXInL1, AscendC::TPosition::A1>;
    // using L1XType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, CTypeForL1>::L1AType;

    // using L0XType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, CTypeForL1>::L0AType;
    // using LayoutXInL0 = Catlass::layout::nZ;
    // using L0XType = Catlass::Gemm::GemmType<ElementX, LayoutXInL0, AscendC::TPosition::B2>;
    using L0XType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<VXType, AType>::L0AType;

    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, AType>;

    using LayoutAInL1 = typename CopyGmToL1A::LayoutDst;

    /*
    template <class ArchTag, class Element>
        struct CopyGmToL1<
        ArchTag, Gemm::GemmType<Element, layout::ColumnMajor>,
        Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>> {
    */
    using CopyGmToL1VX = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, VXType, L1XType>;
    using CopyGmToL1X = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, XType, L1XType>;

    using CopyL1ToL0A = Catlass::Gemm::Tile::CopyL1ToL0A<
        ArchTag, typename Catlass::Gemm::helper::L1ATypeSelector<AType>::L1AType>;

    using CopyL1ToL0B = FTSelf::Gemm::Tile::CopyL1ToL0B<
        ArchTag, typename Catlass::Gemm::helper::L1BTypeSelector<BType>::L1BType>;

    using CopyL1ToL0X = Catlass::Gemm::Tile::CopyL1ToL0A<ArchTag, L1XType, L0XType>;

    using L1ATypeforFT = typename Catlass::Gemm::helper::L1ATypeSelector<AType>::L1AType;
    using L0ATypeforFT = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<VXType, AType>::L0BType;

    using CopyL1ToL0AforFT = FTSelf::Gemm::Tile::CopyL1ToL0B<ArchTag, L1ATypeforFT, L0ATypeforFT>;

    using CopyL0CToGm = Catlass::Gemm::Tile::CopyL0CToGm<ArchTag,ElementAccumulator, CType>;

    using CopyL0CToGmforABE = Catlass::Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, YType>;

    using BiasTypeSelector = Catlass::Gemm::helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;

    using CopyGmToL1Bias = std::conditional_t<std::is_same_v<BiasType,void>,void,
                            Catlass::Gemm::Tile::CopyGmToL1<
                                ArchTag,
                                typename BiasTypeSelector::GMBiasType,
                                typename BiasTypeSelector::L1BiasType>>;

    using CopyL1ToBT = std::conditional_t<std::is_same_v<BiasType,void>,void,
                            Catlass::Gemm::Tile::CopyL1ToBT<ArchTag,
                                typename BiasTypeSelector::L1BiasType,
                                typename BiasTypeSelector::L0BiasType>>;


};

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
    using FT_L02L1_TYPE = FTSelf::Gemv::helper::FT_L02L1_TYPE;
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;

    using ElementX = typename XType::Element;

    using LayoutX = typename XType::Layout;
    using LayoutXCol = typename XTypeCol::Layout;

    using CopyGmToL1B = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, BType>;


    // using LayoutXInL1 = typename std::conditional<
    //     std::is_same<LayoutX, Catlass::layout::ColumnMajor>::value,
    //     Catlass::layout::nZ,
    //     Catlass::layout::zN>::type;

    // using ElementCInL1 = typename XType::Element;
    using LayoutVX = Catlass::layout::VectorLayout;
    using LayoutVXCol = Catlass::layout::ColumnMajor;

    using VXType = Catlass::Gemm::GemmType<ElementX, LayoutVX>;
    using VXTypeCol = Catlass::Gemm::GemmType<ElementX, LayoutVXCol>;

    using L1XType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<VXType, AType>::L1AType;
    /*
    template<class Element>
    struct L1BTypeSelector<Gemm::GemmType<Element, layout::ColumnMajor>> {
        using L1BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
    };
    */
    using L1XTypeCol = typename Catlass::Gemm::helper::L1BTypeSelector<VXTypeCol>::L1BType;

    // using LayoutXInL1 = typename CopyGmToL1B::LayoutDst;

    // using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
    // using L1XType = Catlass::Gemm::GemmType<ElementX,LayoutXInL1, AscendC::TPosition::A1>;
    // using L1XType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, CTypeForL1>::L1AType;

    // using L0XType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, CTypeForL1>::L0AType;

    using LayoutXInL0Col = Catlass::layout::nZ;

    using L0XType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<VXType, AType>::L0AType;
    using L0XTypeCol = Catlass::Gemm::GemmType<ElementX, LayoutXInL0Col, AscendC::TPosition::B2>;

    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, AType>;

    using LayoutAInL1 = typename CopyGmToL1A::LayoutDst;

    /*
    template <class ArchTag, class Element>
        struct CopyGmToL1<
        ArchTag, Gemm::GemmType<Element, layout::ColumnMajor>,
        Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>> {
    */
    using CopyGmToL1VX = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, VXType, L1XType>;
    using CopyGmToL1VXCol = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, VXTypeCol, L1XTypeCol>;

    using CopyGmToL1X = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, XType, L1XType>;
    using CopyGmToL1XCol = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, XTypeCol, L1XTypeCol>;

    using CopyL1ToL0A = Catlass::Gemm::Tile::CopyL1ToL0A<
        ArchTag, typename Catlass::Gemm::helper::L1ATypeSelector<AType>::L1AType>;

    using CopyL1ToL0B = FTSelf::Gemm::Tile::CopyL1ToL0B<
        ArchTag, typename Catlass::Gemm::helper::L1BTypeSelector<BType>::L1BType>;

    using CopyL1ToL0X = Catlass::Gemm::Tile::CopyL1ToL0A<ArchTag, L1XType, L0XType>;
    using CopyL1ToL0XCol = FTSelf::Gemm::Tile::CopyL1ToL0B<ArchTag, L1XTypeCol, L0XTypeCol>;

    using L1ATypeforFT = typename Catlass::Gemm::helper::L1ATypeSelector<AType>::L1AType;
    using L0ATypeforFT = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<VXType, AType>::L0BType;

    using CopyL1ToL0AforFT = FTSelf::Gemm::Tile::CopyL1ToL0B<ArchTag, L1ATypeforFT, L0ATypeforFT>;

    using CopyL0CToGm = Catlass::Gemm::Tile::CopyL0CToGm<ArchTag,ElementAccumulator, CType>;

    using CopyL0CToGmforABE = Catlass::Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, YType>;

    using BiasTypeSelector = Catlass::Gemm::helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;

    using CopyGmToL1Bias = std::conditional_t<std::is_same_v<BiasType,void>,void,
                            Catlass::Gemm::Tile::CopyGmToL1<
                                ArchTag,
                                typename BiasTypeSelector::GMBiasType,
                                typename BiasTypeSelector::L1BiasType>>;

    using CopyL1ToBT = std::conditional_t<std::is_same_v<BiasType,void>,void,
                            Catlass::Gemm::Tile::CopyL1ToBT<ArchTag,
                                typename BiasTypeSelector::L1BiasType,
                                typename BiasTypeSelector::L0BiasType>>;

    //(cSize * 2 + 1) * 16 * 8 * 4B
    /*
    auto h0 = 16; // 当数据类型的位宽为8时，h0 = 32；其他情况下，h0 = 16
    auto w0 = 32 / sizeof(type);  // type代表数据类型
    auto tmpBufferSize = (cSize  * 2 + 1)  * h0 * w0 * sizeof(type);
    */
    // (cSize  * 2 + 1)  * h0 * w0 * sizeof(type);
    // TRANSPOSE_NHWC2NCHW


};

} // namespace FTSelf::Gemm::Tile

#endif // FTSELF_CATLASS_GEMM_TILE_TILE_COPY_HPP
