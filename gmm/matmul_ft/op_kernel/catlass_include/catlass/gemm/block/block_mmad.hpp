/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_HPP

#include "../../catlass.hpp"
#include "../tile/tile_copy.hpp"
#include "../tile/tile_mmad.hpp"

namespace Catlass::Gemm::Tile {

template <
    class ArchTag,
    class ElementA,
    class LayoutTagA,
    class ElementB,
    class LayoutTagB,
    class ElementC,
    class LayoutTagC
>
struct SparseTileCopyTla;

} // namespace Catlass::Gemm::Tile

namespace Catlass::Gemm::Block {

template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class AType,
    class BType,
    class CType,
    class BiasType = void,
    class TileCopy = Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
    class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>,
    class Enable = void
>
struct BlockMmad {
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
    class TileCopy = Gemm::Tile::TileCopyGemm<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,  // change the name
    class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockGemm {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class AType,
    class BType,
    class CType,
    class BiasType,
    class TileCopy,
    class TileMmad
>
struct BlockMmadAiv {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadAiv is not implemented for this DispatchPolicy");
};


template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class ElementA,
    class ElementB,
    class ElementC,
    class ElementBias = void,
    class TileCopy = Gemm::Tile::SparseTileCopyTla<typename DispatchPolicy::ArchTag, ElementA, layout::RowMajor,
        ElementB, layout::ColumnMajor, ElementC, layout::RowMajor>
>
struct BlockMmadSparseTla {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadSparseTla is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class ElementA,
    class ElementB,
    class ElementC,
    class ElementBias = void,
    class TileCopy = Gemm::Tile::PackedTileCopyTla<typename DispatchPolicy::ArchTag, ElementA, layout::RowMajor,
        ElementB, layout::RowMajor, ElementC, layout::RowMajor, ElementBias>,
    class TileMmad =
        Gemm::Tile::TileMmadTla<typename DispatchPolicy::ArchTag, ElementA, typename TileCopy::LayoutTagL1A>
>
struct BlockMmadTla {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadTla is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class ElementA,
    class ElementB,
    class ElementC,
    class ElementPrologueB,
    class ElementBias = void,
    class TileCopy = Gemm::Tile::PackedTileCopyTla<typename DispatchPolicy::ArchTag, ElementA, layout::zN,
        ElementB, layout::zN, ElementC, layout::zN, ElementBias>,
    class TileMmad =
        Gemm::Tile::TileMmadTla<typename DispatchPolicy::ArchTag, ElementA, typename TileCopy::LayoutTagL1A>
>
struct BlockMmadA8W4Mx {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadA8W4Mx is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class PrologueSrcType,
    class PrologueDstType,
    class L1TileShape,
    class TileCopy
>
struct BlockPrologue {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockPrologue is not implemented for this DispatchPolicy");
};

} // namespace Catlass::Gemm::Block

// The standalone matmul_ft package carries only the Catlass headers used by its
// self-implemented block specializations. Keep Catlass' aggregation list below
// as part of the upstream source, but do not pull the full block library into
// this package.
#if 0
#if (defined (CATLASS_ARCH) && CATLASS_ARCH == 2201)
#include "block_mmad_fa_qk.hpp"
#include "block_mmad_fa_pv.hpp"
#include "block_mmad_mla_qk.hpp"
#include "block_mmad_mla_pv.hpp"
#include "block_mmad_mla_qk_tp1_spec.hpp"
#include "block_mmad_mla_pv_tp1_spec.hpp"
#include "block_mmad_preload.hpp"
#include "block_mmad_preload_async.hpp"
#include "block_mmad_preload_async_with_callback.hpp"
#include "block_mmad_gemm.hpp"
#include "block_mmad_pingpong_bias.hpp"
#include "block_mmad_fai_qk_head_tail.hpp"
#include "block_mmad_fai_qk_normal.hpp"
#include "block_mmad_fai_pv_head_tail.hpp"
#include "block_mmad_fai_pv_normal.hpp"
#include "block_mmad_pingpong_full_loadA.hpp"
#include "block_mmad_pingpong_full_loadA_tla.hpp"
#include "block_mmad_pingpong_with_prologue.hpp"
#include "block_mmad_pingpong_slice_k_with_prologue.hpp"
#include "block_mmad_dynamic_common.hpp"
#include "block_mmad_dynamic_small.hpp"
#include "block_mmad_dynamic_streamk.hpp"
#include "block_mmad_dynamic_single_core_splitk.hpp"
#include "block_mmad_dynamic_preload_async_with_callback.hpp"
#include "block_mmad_small.hpp"
#include "block_mmad_single_core_splitk.hpp"
#include "block_mmad_dynamic_aiv.hpp"
#include "block_mmad_streamk.hpp"
#include "block_mmad_w4a4_per_token_per_channel_dequant.hpp"
#include "block_mmad_sparse_tla.hpp"
#include "block_mmad_fai_qk_head_tail_tla.hpp"
#include "block_mmad_fai_qk_normal_tla.hpp"
#include "block_mmad_fai_pv_head_tail_tla.hpp"
#include "block_mmad_fai_pv_normal_tla.hpp"
#endif

/// Compactible block utility
#include "block_mmad_pingpong.hpp"

/// TLA block utility
#include "block_mmad_pingpong_tla.hpp"
#include "block_mmad_pingpong_dequant_tla.hpp"
#include "block_mmad_pingpong_tla_v2.hpp"
#include "block_mmad_preload_tla.hpp"
#include "block_mmad_preload_async_with_callback_tla.hpp"
#include "block_mmad_fai_pv_tla.hpp"
#include "block_mmad_fai_qk_tla.hpp"
#include "block_mmad_pingpong_per_group_per_block_tla.hpp"
#include "block_mmad_mx_tla.hpp"
#if (defined (CATLASS_ARCH) && CATLASS_ARCH == 3510)
#include "block_mx_a8w4_prologue.hpp"
#include "block_mmad_mx_a8w4.hpp"
#include "block_mmad_svd_quant_tla.hpp"
#endif
#endif

// self implemented
#include "../tile/tile_copy.hpp"
#include "../tile/tile_mmad.hpp"
#include "../../gemv/helper.hpp"

namespace CubeSelf::Gemm::Block {
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
//  = Vecself::Gemv::helper::FT_L02L1_TYPE::FIX_PIPE

/*
 template <
    class ArchTag,
    class AType,
    class BType,
    class CType,
    class XType,
    class YType,
    class BiasType = void,
    Vecself::Gemv::helper::FT_L02L1_TYPE COPY_TYPE_ = Vecself::Gemv::helper::FT_L02L1_TYPE::FIX_PIPE
>
struct TileCopyFT
*/

template<
    class DispatchPolicy,
    Vecself::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    Vecself::Gemv::helper::FT_L02L1_TYPE COPY_TYPE_,
    class L1TileShape,
    class L0TileShape,
    class L0TileShapeforFT,
    class AType,
    class BType,
    class CType,
    class XType,
    class YType,
    class BiasType = void,
    class TileCopyFT = CubeSelf::Gemm::Tile::TileCopyFT<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, YType, BiasType, COPY_TYPE_>,
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
    class TileCopyFTABonAic = CubeSelf::Gemm::Tile::TileCopyFTABonAic<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, YType, BiasType>,
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
    class TileCopyFTABonAicAuged = CubeSelf::Gemm::Tile::TileCopyFTABonAicAuged<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, XColType, YType, BiasType>,
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
    class TileCopyFTABonAicAuged = CubeSelf::Gemm::Tile::TileCopyFTABonAicAuged<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, XColType, YType, BiasType>,
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
    class TileCopyFTABonAicAuged = CubeSelf::Gemm::Tile::TileCopyFTABonAicAuged<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, XColType, YType, BiasType>,
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
    Vecself::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    Vecself::Gemv::helper::FT_L02L1_TYPE COPY_TYPE_,
    class L1TileShape,
    class L0TileShape,
    class L0TileShapeforFT,
    class AType,
    class BType,
    class CType,
    class XType,
    class YType,
    class BiasType = void,
    class TileCopyFT = CubeSelf::Gemm::Tile::TileCopyFT<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, YType, BiasType, COPY_TYPE_>,
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
    class TileCopyFTABonAic = CubeSelf::Gemm::Tile::TileCopyFTABonAic<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, YType, BiasType>,
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
    class TileCopyFTABonAic = CubeSelf::Gemm::Tile::TileCopyFTABonAic<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, YType, BiasType>,
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
    class TileCopyFTABonAic = CubeSelf::Gemm::Tile::TileCopyFTABonAic<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, YType, BiasType>,
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


} // namespace CubeSelf::Gemm::Block

#include "block_mmad_pingpong_fault_abe_spec_no_splitk_robust.hpp"
#include "block_mmad_pingpong_preload.hpp"

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_HPP
