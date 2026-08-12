#ifndef FTSELF_CATLASS_GEMV_TILE_TILE_COPY_HPP
#define FTSELF_CATLASS_GEMV_TILE_TILE_COPY_HPP

#include_next "catlass/gemv/tile/tile_copy.hpp"
#include "FT_self/gemv/tile/matrix_copy_gm_to_ub_simpling.hpp"
#include "FT_self/gemv/tile/matrix_copy_ub_to_gm.hpp"
#include "FT_self/gemv/tile/vec_copy_gm_to_ub_pad.hpp"
#include "FT_self/gemv/tile/vec_copy_ub_to_gm_pad.hpp"

namespace FTSelf::Gemv::Tile {

using namespace Catlass;
using Catlass::Gemv::Tile::MatrixCopyGmToUB;
using Catlass::Gemv::Tile::VecCopyGmToUB;
using Catlass::Gemv::Tile::VecCopyUBToGm;

template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyGemvAiv {
    using MATRIX_SIMPLING_TYPE = FTSelf::Gemv::helper::MATRIX_SIMPLING_TYPE;

    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;
    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = FTSelf::Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    static constexpr bool is_atoadd = FTSelf::Gemv::helper::AtomicAddSelector<AType>::value;
    using VecCopyUbToGm = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = FTSelf::Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;

    using MatrixCopyGmToUbSimplingContinue = FTSelf::Gemv::Tile::MatrixCopyGmToUBSimpling<
        ArchTag, AType, MATRIX_SIMPLING_TYPE::CONTINUOUS_SIMPLING>;

    using MatrixCopyGmToUbSimplingStrided = FTSelf::Gemv::Tile::MatrixCopyGmToUBSimpling<
        ArchTag, AType, MATRIX_SIMPLING_TYPE::STRIDED_SIMPLING>;
};


template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    class BType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyFTRedAiv {
    using MATRIX_SIMPLING_TYPE = FTSelf::Gemv::helper::MATRIX_SIMPLING_TYPE;
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;
    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    // the function of aiv
    // using ElementCInL1 = typename XType::Element;
    using VecCopyGmToUb = FTSelf::Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;

    static constexpr bool is_atoadd = FTSelf::Gemv::helper::AtomicAddSelector<AType>::value;

    using LayoutVX = Catlass::layout::VectorLayout;
    using VXType = Catlass::Gemm::GemmType<ElementX, LayoutVX>;
    using VecCopyUbToGmforBMax = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, XType, is_atoadd>;
    using VecCopyUbToGmforAMax = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, VXType, is_atoadd>;
    using VecCopyUbToGmforAMean = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, YType, is_atoadd>;

    using MatrixCopyGmToUb = FTSelf::Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;

    /*
    template <class Element>
    struct MatrixCopyGmToUBSimpling<
        Arch::AtlasA2,
        Catlass::Gemm::GemmType<Element, layout::RowMajor>,
        FTSelf::Gemv::helper::MATRIX_SIMPLING_TYPE::CONTINUOUS_SIMPLING>
    */
    using MatrixCopyGmToUbSimplingContinue = FTSelf::Gemv::Tile::MatrixCopyGmToUBSimpling<
        ArchTag, AType, MATRIX_SIMPLING_TYPE::CONTINUOUS_SIMPLING>;

    using MatrixCopyGmToUbSimplingStrided = FTSelf::Gemv::Tile::MatrixCopyGmToUBSimpling<
        ArchTag, AType, MATRIX_SIMPLING_TYPE::STRIDED_SIMPLING>;
};




template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyGemvAic {
    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    using ElementYforBFAIV = float;
    using LayoutY = typename YType::Layout;
    using YTypeforBFAIV = Catlass::Gemm::GemmType<ElementYforBFAIV, LayoutY>;

    // the function of aic
    using L1XType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L1AType;
    using L1AType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L1BType;
    using L1AColType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L1BColType;

    using L0AType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L0AType;
    using L0BType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L0BType;
    using L0BColType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L0BColType;


    using CopyGmToL1A = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, XType, L1XType>;
    using CopyGmToL1B = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, AType, L1AType>;
    using CopyGmToL1BCol = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, AType, L1AColType>;


    using CopyL1ToL0A = Catlass::Gemm::Tile::CopyL1ToL0A<ArchTag, L1XType, L0AType>;
    using CopyL1ToL0B = FTSelf::Gemm::Tile::CopyL1ToL0B<ArchTag, L1AType, L0BType>;
    using CopyL1ToL0BCol = FTSelf::Gemm::Tile::CopyL1ToL0B<ArchTag, L1AColType, L0BColType>;
    using CopyL1ToL0BBoth = FTSelf::Gemm::Tile::CopyL1ToL0B<ArchTag, L1AType, L0BColType>;


    using CopyL0CToGm = Catlass::Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, YType>;
    using CopyL0CToGmforBFAIV = Catlass::Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, YTypeforBFAIV>;


};

template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyMatrixReduceAiv {
    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;
    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = FTSelf::Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    // FTSelf::Gemv::helper::AtomicAddSelector<AType>::value
    static constexpr bool is_atoadd = false;
    using VecCopyUbToGm = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = FTSelf::Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
};

template <
    FTSelf::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    /// Tag indicating architecture
    class ArchTag,
    /// Output Result operand namely vector Z
    class ZType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyCompareAiv {
    using ElementZ = typename ZType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;
    using FT_COMP_TYPE = FTSelf::Gemv::helper::FT_COMP_TYPE;

    using ElementWork = typename std::conditional<
        (COMP_TYPE_ == FT_COMP_TYPE::XOR),
        uint16_t,
        typename std::conditional<(COMP_TYPE_ == FT_COMP_TYPE::COMPARE), int32_t, ElementX>::type>::type;


    // using ElementAccumulator =
    //     typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aiv

    using VecCopyGmToUbX = FTSelf::Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    using VecCopyGmToUbY = FTSelf::Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    using VecCopyGmToUbW = FTSelf::Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;

    using VecCopyUbToGmZ = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, ZType>;

    using WType = Catlass::Gemm::GemmType<ElementWork, Catlass::layout::VectorLayout>;
    using VecCopyUbToGmW = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, WType>;



    // static constexpr bool is_atoadd = FTSelf::Gemv::helper::AtomicAddSelector<AType>::value;
    // using VecCopyUbToGm = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    // using MatrixCopyGmToUb = FTSelf::Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
};

template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// Output Result operand namely vector Z
    class ZType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyMatrixReduceAivFused {
    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;
    using ElementZ = typename ZType::Element;

    using FT_COMP_TYPE = FTSelf::Gemv::helper::FT_COMP_TYPE;

    using ElementWork = ElementY;

    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = FTSelf::Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    // FTSelf::Gemv::helper::AtomicAddSelector<AType>::value
    static constexpr bool is_atoadd = false;
    using VecCopyUbToGm = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = FTSelf::Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;

    using VecCopyGmToUbInX = FTSelf::Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    using VecCopyGmToUbInY = FTSelf::Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    using VecCopyGmToUbW = FTSelf::Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;

    using VecCopyUbToGmZ = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, ZType>;

    using WType = Catlass::Gemm::GemmType<ElementWork, Catlass::layout::VectorLayout>;
    using VecCopyUbToGmW = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, WType>;
};

template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType for C matric operand
    class CType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// Output Result operand namely vector Z
    class ZType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyGemvThreCompFusedAiv {
    using ElementA = typename AType::Element;
    using ElementC = typename CType::Element;

    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;

    using FT_COMP_TYPE = FTSelf::Gemv::helper::FT_COMP_TYPE;

    using ElementWork = ElementY;

    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = FTSelf::Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    static constexpr bool is_atoadd = FTSelf::Gemv::helper::AtomicAddSelector<AType>::value;
    using VecCopyUbToGm = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = FTSelf::Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;

    using MatrixCopyGmToUbforThre = FTSelf::Gemv::Tile::MatrixCopyGmToUB<ArchTag, CType>;

    using VecCopyGmToUbInY = FTSelf::Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    using VecCopyGmToUbW = FTSelf::Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;

    using VecCopyUbToGmZ = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, ZType>;
    using VecCopyUbToGmforThre = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, YType>;

    using WType = Catlass::Gemm::GemmType<ElementWork, Catlass::layout::VectorLayout>;
    using VecCopyUbToGmW = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, WType>;

};

template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType for C matric operand
    class CType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// Output Result operand namely vector Z
    class ZType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyGemvThreCompFusedAivFI {
    using ElementA = typename AType::Element;
    using ElementC = typename CType::Element;

    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;

    using FT_COMP_TYPE = FTSelf::Gemv::helper::FT_COMP_TYPE;

    using ElementWork = ElementY;

    using ElementFIID = int32_t;
    using ElementFIData = ElementA;

    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = FTSelf::Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    static constexpr bool is_atoadd = FTSelf::Gemv::helper::AtomicAddSelector<AType>::value;
    using VecCopyUbToGm = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = FTSelf::Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;

    using MatrixCopyGmToUbforThre = FTSelf::Gemv::Tile::MatrixCopyGmToUB<ArchTag, CType>;

    using VecCopyGmToUbInY = FTSelf::Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    using VecCopyGmToUbW = FTSelf::Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;

    using VecCopyUbToGmZ = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, ZType>;
    using VecCopyUbToGmforThre = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, YType>;

    using WType = Catlass::Gemm::GemmType<ElementWork, Catlass::layout::VectorLayout>;
    using VecCopyUbToGmW = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, WType>;

    using FIIndexType = Catlass::Gemm::GemmType<ElementFIID, Catlass::layout::VectorLayout>;
    using VecCopyUbToGmforFIIndex = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, FIIndexType, false>;

    using FIDataType = Catlass::Gemm::GemmType<ElementFIData, Catlass::layout::VectorLayout>;
    using VecCopyUbToGmforFIData = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, FIDataType, false>;


};

template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyMatrixSliceSumAiv {
    using ElementA = typename AType::Element;
    using ElementY = typename YType::Element;
    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = FTSelf::Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    // FTSelf::Gemv::helper::AtomicAddSelector<AType>::value
    static constexpr bool is_atoadd = false;
    using VecCopyUbToGm = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = FTSelf::Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
};

template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType for Y matrix operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyMatrixAddAiv {
    using ElementA = typename AType::Element;
    using ElementY = typename YType::Element;

    using LayoutVX = Catlass::layout::VectorLayout;
    using VXType = Catlass::Gemm::GemmType<ElementA, LayoutVX>;

    using LayoutVY = Catlass::layout::VectorLayout;
    using VYType = Catlass::Gemm::GemmType<ElementY, LayoutVY>;

    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;

    using VecCopyGmToUb = FTSelf::Gemv::Tile::VecCopyGmToUB<ArchTag, VXType>;
    static constexpr bool is_atoadd = false;
    using VecCopyUbToGm = FTSelf::Gemv::Tile::VecCopyUBToGm<ArchTag, VYType, is_atoadd>;

    // the function of aiv
    using MatrixCopyGmToUb = FTSelf::Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
    // MatrixCopyUbtoGm
    using MatrixCopyUbToGm = FTSelf::Gemv::Tile::MatrixCopyUBToGm<ArchTag, YType>;
};


template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType for Y matrix operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyMatrixAddVectorizedAiv {
    using ElementA = typename AType::Element;
    using ElementY = typename YType::Element;

    using LayoutVX = Catlass::layout::VectorLayout;
    using VXType = Catlass::Gemm::GemmType<ElementA, LayoutVX>;

    using LayoutVY = Catlass::layout::VectorLayout;
    using VYType = Catlass::Gemm::GemmType<ElementY, LayoutVY>;

    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;

    /*
    template <
        class ArchTag_,
        class VType_,
        FTSelf::Gemv::helper::VEC_PADDING_TYPE::ALIGNED
    >
    struct VecCopyGmToUBPadding
    */
    using VecCopyGmToUbCommon = FTSelf::Gemv::Tile::VecCopyGmToUBPadding<ArchTag, VXType, FTSelf::Gemv::helper::VEC_PADDING_TYPE::ALIGNED>;
    /*
    template <
        class ArchTag_,
        class VType_,
        FTSelf::Gemv::helper::VEC_PADDING_TYPE::PADDING
    >
    struct VecCopyGmToUBPadding
    */
    using VecCopyGmToUbTail = FTSelf::Gemv::Tile::VecCopyGmToUBPadding<ArchTag, VXType, FTSelf::Gemv::helper::VEC_PADDING_TYPE::PADDING>;
    static constexpr bool is_atoadd = false;
    /*
    template <class Element>
    struct VecCopyUBToGmPadding<Arch::AtlasA2,
        Catlass::Gemm::GemmType<Element, layout::VectorLayout>,
        FTSelf::Gemv::helper::VEC_PADDING_TYPE::ALIGNED,
        false>
    */
    using VecCopyUbToGmCommon = FTSelf::Gemv::Tile::VecCopyUBToGmPadding<ArchTag, VYType, FTSelf::Gemv::helper::VEC_PADDING_TYPE::ALIGNED, is_atoadd>;

    /*
    template <class Element>
    struct VecCopyUBToGmPadding<Arch::AtlasA2,
        Catlass::Gemm::GemmType<Element, layout::VectorLayout>,
        FTSelf::Gemv::helper::VEC_PADDING_TYPE::PADDING,
        false>
    */
    using VecCopyUbToGmTail = FTSelf::Gemv::Tile::VecCopyUBToGmPadding<ArchTag, VYType, FTSelf::Gemv::helper::VEC_PADDING_TYPE::PADDING, is_atoadd>;

    // the function of aiv
    using MatrixCopyGmToUb = FTSelf::Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
    using MatrixCopyUbToGm = FTSelf::Gemv::Tile::MatrixCopyUBToGm<ArchTag, YType>;
};

} // namespace FTSelf::Gemv::Tile

#endif // FTSELF_CATLASS_GEMV_TILE_TILE_COPY_HPP
