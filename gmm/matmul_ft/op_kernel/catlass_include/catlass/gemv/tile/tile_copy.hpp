/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_TILE_TILE_COPY_HPP
#define CATLASS_GEMV_TILE_TILE_COPY_HPP

#include "../../catlass.hpp"
#include "../../detail/tag_to_layout.hpp"

#include "vec_copy_gm_to_ub.hpp"
#include "vec_copy_ub_to_gm.hpp"
#include "matrix_copy_gm_to_ub.hpp"
#include "matrix_copy_ub_to_gm.hpp"
// include/catlass/gemv/tile/matrix_copy_gm_to_ub_simpling.hpp
#include "matrix_copy_gm_to_ub_simpling.hpp"
// include/catlass/gemv/tile/vec_copy_gm_to_ub_pad.hpp
#include "vec_copy_gm_to_ub_pad.hpp"
// include/catlass/gemv/tile/vec_copy_ub_to_gm_pad.hpp
#include "vec_copy_ub_to_gm_pad.hpp"

#include "../../gemm/tile/copy_gm_to_l1.hpp"
#include "../../gemm/tile/copy_l0c_to_gm.hpp"
#include "../../gemm/tile/copy_l1_to_l0a.hpp"
#include "../../gemm/tile/copy_l1_to_l0b.hpp"

#include "../../gemm/helper.hpp"
#include "../helper.hpp"
#include "../../gemm/gemm_type.hpp"

namespace Catlass::Gemv::Tile {

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
    class BiasType = void>
struct TileCopyGemvAiv {
    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    static constexpr bool is_atoadd = Vecself::Gemv::helper::AtomicAddSelector<AType>::value;
    using VecCopyUbToGm = Gemv::Tile::VecCopyUBToGm<ArchTag, YType, is_atoadd>;
    using MatrixCopyGmToUb = Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
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
    class BiasType = void>
struct TileCopyGemvAic {
    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aic
    using L1XType = typename Vecself::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L1AType;
    using L1AType = typename Vecself::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L1BType;
    using L0AType = typename Vecself::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L0AType;
    using L0BType = typename Vecself::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L0BType;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, XType, L1XType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, AType, L1AType>;

    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<ArchTag, L1XType, L0AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<ArchTag, L1AType, L0BType>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, YType>;
};

} // namespace Catlass::Gemv::Tile

// self implemented
namespace Vecself::Gemv::Tile {

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
    using MATRIX_SIMPLING_TYPE = Vecself::Gemv::helper::MATRIX_SIMPLING_TYPE;

    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = Vecself::Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    static constexpr bool is_atoadd = Vecself::Gemv::helper::AtomicAddSelector<AType>::value;
    using VecCopyUbToGm = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = Vecself::Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;

    using MatrixCopyGmToUbSimplingContinue = Vecself::Gemv::Tile::MatrixCopyGmToUBSimpling<
        ArchTag, AType, MATRIX_SIMPLING_TYPE::CONTINUOUS_SIMPLING>;

    using MatrixCopyGmToUbSimplingStrided = Vecself::Gemv::Tile::MatrixCopyGmToUBSimpling<
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
    using MATRIX_SIMPLING_TYPE = Vecself::Gemv::helper::MATRIX_SIMPLING_TYPE;
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    // the function of aiv
    // using ElementCInL1 = typename XType::Element;
    using VecCopyGmToUb = Vecself::Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;

    static constexpr bool is_atoadd = Vecself::Gemv::helper::AtomicAddSelector<AType>::value;
    
    using LayoutVX = Catlass::layout::VectorLayout;
    using VXType = Catlass::Gemm::GemmType<ElementX, LayoutVX>;
    using VecCopyUbToGmforBMax = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, XType, is_atoadd>;
    using VecCopyUbToGmforAMax = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, VXType, is_atoadd>;
    using VecCopyUbToGmforAMean = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, YType, is_atoadd>;

    using MatrixCopyGmToUb = Vecself::Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;

    /*
    template <class Element>
    struct MatrixCopyGmToUBSimpling<
        Arch::AtlasA2, 
        Gemm::GemmType<Element, layout::RowMajor>, 
        Vecself::Gemv::helper::MATRIX_SIMPLING_TYPE::CONTINUOUS_SIMPLING>
    */
    using MatrixCopyGmToUbSimplingContinue = Vecself::Gemv::Tile::MatrixCopyGmToUBSimpling<
        ArchTag, AType, MATRIX_SIMPLING_TYPE::CONTINUOUS_SIMPLING>;

    using MatrixCopyGmToUbSimplingStrided = Vecself::Gemv::Tile::MatrixCopyGmToUBSimpling<
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
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;
    
    using ElementYforBFAIV = float;
    using LayoutY = typename YType::Layout;
    using YTypeforBFAIV = Gemm::GemmType<ElementYforBFAIV, LayoutY>;

    // the function of aic
    using L1XType = typename Vecself::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L1AType;
    using L1AType = typename Vecself::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L1BType;
    using L1AColType = typename Vecself::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L1BColType;

    using L0AType = typename Vecself::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L0AType;
    using L0BType = typename Vecself::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L0BType;
    using L0BColType = typename Vecself::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L0BColType;


    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, XType, L1XType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, AType, L1AType>;
    using CopyGmToL1BCol = Gemm::Tile::CopyGmToL1<ArchTag, AType, L1AColType>;


    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<ArchTag, L1XType, L0AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<ArchTag, L1AType, L0BType>;
    using CopyL1ToL0BCol = Gemm::Tile::CopyL1ToL0B<ArchTag, L1AColType, L0BColType>;
    using CopyL1ToL0BBoth = Gemm::Tile::CopyL1ToL0B<ArchTag, L1AType, L0BColType>;


    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, YType>;
    using CopyL0CToGmforBFAIV = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, YTypeforBFAIV>;

    
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
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = Vecself::Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    // Vecself::Gemv::helper::AtomicAddSelector<AType>::value
    static constexpr bool is_atoadd = false;
    using VecCopyUbToGm = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = Vecself::Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
};

template <
    Vecself::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
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
    using FT_COMP_TYPE = Vecself::Gemv::helper::FT_COMP_TYPE;

    using ElementWork = typename std::conditional<
        (COMP_TYPE_ == FT_COMP_TYPE::XOR),
        uint16_t,
        typename std::conditional<(COMP_TYPE_ == FT_COMP_TYPE::COMPARE), int32_t, ElementX>::type>::type;
    
    
    // using ElementAccumulator =
    //     typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aiv
    
    using VecCopyGmToUbX = Vecself::Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    using VecCopyGmToUbY = Vecself::Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    using VecCopyGmToUbW = Vecself::Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    
    using VecCopyUbToGmZ = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, ZType>;

    using WType = Gemm::GemmType<ElementWork, Catlass::layout::VectorLayout>;
    using VecCopyUbToGmW = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, WType>;



    // static constexpr bool is_atoadd = Vecself::Gemv::helper::AtomicAddSelector<AType>::value;
    // using VecCopyUbToGm = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    // using MatrixCopyGmToUb = Vecself::Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
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

    using FT_COMP_TYPE = Vecself::Gemv::helper::FT_COMP_TYPE;

    using ElementWork = ElementY;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = Vecself::Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    // Vecself::Gemv::helper::AtomicAddSelector<AType>::value
    static constexpr bool is_atoadd = false;
    using VecCopyUbToGm = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = Vecself::Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;

    using VecCopyGmToUbInX = Vecself::Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    using VecCopyGmToUbInY = Vecself::Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    using VecCopyGmToUbW = Vecself::Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    
    using VecCopyUbToGmZ = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, ZType>;

    using WType = Gemm::GemmType<ElementWork, Catlass::layout::VectorLayout>;
    using VecCopyUbToGmW = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, WType>;
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

    using FT_COMP_TYPE = Vecself::Gemv::helper::FT_COMP_TYPE;

    using ElementWork = ElementY;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = Vecself::Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    static constexpr bool is_atoadd = Vecself::Gemv::helper::AtomicAddSelector<AType>::value;
    using VecCopyUbToGm = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = Vecself::Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;

    using MatrixCopyGmToUbforThre = Vecself::Gemv::Tile::MatrixCopyGmToUB<ArchTag, CType>;

    using VecCopyGmToUbInY = Vecself::Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    using VecCopyGmToUbW = Vecself::Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    
    using VecCopyUbToGmZ = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, ZType>;
    using VecCopyUbToGmforThre = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, YType>;

    using WType = Gemm::GemmType<ElementWork, Catlass::layout::VectorLayout>;
    using VecCopyUbToGmW = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, WType>;

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

    using FT_COMP_TYPE = Vecself::Gemv::helper::FT_COMP_TYPE;

    using ElementWork = ElementY;

    using ElementFIID = int32_t;
    using ElementFIData = ElementA;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = Vecself::Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    static constexpr bool is_atoadd = Vecself::Gemv::helper::AtomicAddSelector<AType>::value;
    using VecCopyUbToGm = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = Vecself::Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;

    using MatrixCopyGmToUbforThre = Vecself::Gemv::Tile::MatrixCopyGmToUB<ArchTag, CType>;

    using VecCopyGmToUbInY = Vecself::Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    using VecCopyGmToUbW = Vecself::Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    
    using VecCopyUbToGmZ = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, ZType>;
    using VecCopyUbToGmforThre = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, YType>;

    using WType = Gemm::GemmType<ElementWork, Catlass::layout::VectorLayout>;
    using VecCopyUbToGmW = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, WType>;

    using FIIndexType = Gemm::GemmType<ElementFIID, Catlass::layout::VectorLayout>;
    using VecCopyUbToGmforFIIndex = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, FIIndexType, false>;

    using FIDataType = Gemm::GemmType<ElementFIData, Catlass::layout::VectorLayout>;
    using VecCopyUbToGmforFIData = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, FIDataType, false>;


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
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = Vecself::Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    // Vecself::Gemv::helper::AtomicAddSelector<AType>::value
    static constexpr bool is_atoadd = false;
    using VecCopyUbToGm = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = Vecself::Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
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
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;

    using VecCopyGmToUb = Vecself::Gemv::Tile::VecCopyGmToUB<ArchTag, VXType>;
    static constexpr bool is_atoadd = false;
    using VecCopyUbToGm = Vecself::Gemv::Tile::VecCopyUBToGm<ArchTag, VYType, is_atoadd>;

    // the function of aiv
    using MatrixCopyGmToUb = Vecself::Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
    // MatrixCopyUbtoGm
    using MatrixCopyUbToGm = Vecself::Gemv::Tile::MatrixCopyUBToGm<ArchTag, YType>;
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
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;

    /*
    template <
        class ArchTag_,
        class VType_,
        Vecself::Gemv::helper::VEC_PADDING_TYPE::ALIGNED
    >
    struct VecCopyGmToUBPadding
    */
    using VecCopyGmToUbCommon = Vecself::Gemv::Tile::VecCopyGmToUBPadding<ArchTag, VXType, Vecself::Gemv::helper::VEC_PADDING_TYPE::ALIGNED>;
    /*
    template <
        class ArchTag_,
        class VType_,
        Vecself::Gemv::helper::VEC_PADDING_TYPE::PADDING
    >
    struct VecCopyGmToUBPadding
    */
    using VecCopyGmToUbTail = Vecself::Gemv::Tile::VecCopyGmToUBPadding<ArchTag, VXType, Vecself::Gemv::helper::VEC_PADDING_TYPE::PADDING>;
    static constexpr bool is_atoadd = false;
    /*
    template <class Element>
    struct VecCopyUBToGmPadding<Arch::AtlasA2, 
        Gemm::GemmType<Element, layout::VectorLayout>, 
        Vecself::Gemv::helper::VEC_PADDING_TYPE::ALIGNED,
        false>
    */
    using VecCopyUbToGmCommon = Vecself::Gemv::Tile::VecCopyUBToGmPadding<ArchTag, VYType, Vecself::Gemv::helper::VEC_PADDING_TYPE::ALIGNED, is_atoadd>;
    
    /*
    template <class Element>
    struct VecCopyUBToGmPadding<Arch::AtlasA2, 
        Gemm::GemmType<Element, layout::VectorLayout>, 
        Vecself::Gemv::helper::VEC_PADDING_TYPE::PADDING,
        false>
    */
    using VecCopyUbToGmTail = Vecself::Gemv::Tile::VecCopyUBToGmPadding<ArchTag, VYType, Vecself::Gemv::helper::VEC_PADDING_TYPE::PADDING, is_atoadd>;
    
    // the function of aiv
    using MatrixCopyGmToUb = Vecself::Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
    using MatrixCopyUbToGm = Vecself::Gemv::Tile::MatrixCopyUBToGm<ArchTag, YType>;
};

} // namespace Vecself::Gemv::Tile

#endif // CATLASS_GEMV_TILE_TILE_COPY_HPP
