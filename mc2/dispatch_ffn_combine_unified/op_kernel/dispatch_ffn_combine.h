/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dispatch_ffn_combine.h
 * \brief Unified DispatchFFNCombine implementation that merges the three legacy paths:
 *          - dispatch_ffn_combine          (INT8 weight)
 *          - dispatch_ffn_combine_bf16     (FP16 / BF16 weight)
 *          - dispatch_ffn_combine_w4_a8    (INT4 weight, W4A8)
 *        Differences from the legacy single-dtype implementations:
 *          - Inputs use the new unified naming (x / topk_ids / topk_weights / weight1
 *            / weight_scales1 / ... ) following the unified op def.
 *          - The first input is `context` (replaces the removed `group` attribute).
 *          - The single legacy `transB` attribute is split into `transpose_weight1`
 *            and `transpose_weight2`, surfaced via two template flags.
 *          - The legacy `M` attribute is renamed to `max_recv_token_num`.
 *          - Tiling fields are read in the order defined in the unified op def.
 */

#ifndef DISPATCH_FFN_COMBINE_H
#define DISPATCH_FFN_COMBINE_H

using namespace AscendC;

#include "kernel_operator.h"

#include "utils/moe_distribute_base.h"

#include "dispatch_ffn_combine_tiling.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/epilogue/tile/tile_elemwise_muls.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/matmul_epilogue.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "utils/select_helper.hpp"
#include "utils/const_args.hpp"
#include "dispatch_ffn_combine_kernel.hpp"
#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"

using namespace Catlass;
using namespace AscendC::HcclContextDef;

namespace DispatchFFNCombineImpl {
// Template parameter pack used by the unified DispatchFFNCombine class.
//   AType_ : input X element type (DTYPE_X)
//   BType_ : weight element type   (DTYPE_WEIGHT1)
//   CType_ : output  element type  (DTYPE_Y)
//   TB1_   : transpose_weight1 attribute
//   TB2_   : transpose_weight2 attribute
//   NzW1_  : whether weight1 is in FRACTAL_NZ format
//   NzW2_  : whether weight2 is in FRACTAL_NZ format
//   IS_A2_ : whether running on Atlas A2 (910B); false means A3 (910_93)
#define TemplateMMA2AClass                                                                                             \
    typename AType_, typename BType_, typename CType_, bool TB1_, bool TB2_, bool NzW1_, bool NzW2_, bool IS_A2_
#define TemplateMMA2ACFunc AType_, BType_, CType_, TB1_, TB2_, NzW1_, NzW2_, IS_A2_

using namespace AscendC;
template <TemplateMMA2AClass>
class DispatchFFNCombine {
public:
    __aicore__ inline DispatchFFNCombine() {};
    __aicore__ inline void Init(GM_ADDR contextGM, GM_ADDR xGM, GM_ADDR topkIdsGM, GM_ADDR topkWeightsGM,
                                GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR weightScales1GM, GM_ADDR weightScales2GM,
                                GM_ADDR yGM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM, GM_ADDR tilingGM);
    __aicore__ inline void Process();

private:
    GM_ADDR contextGM_;
    GM_ADDR xGM_;
    GM_ADDR topkIdsGM_;
    GM_ADDR topkWeightsGM_;
    GM_ADDR weight1GM_;
    GM_ADDR weight2GM_;
    GM_ADDR weightScales1GM_;
    GM_ADDR weightScales2GM_;
    GM_ADDR yGM_;
    GM_ADDR gmExpertTokenNums_;
    GM_ADDR workspaceGM_;

    GM_ADDR moeInitRoutingQuantV2Scale = nullptr;
    GM_ADDR moeInitRoutingQuantV2Offset = nullptr;
    GM_ADDR expertTokensBeforeCapacity = nullptr;

    TBuf<AscendC::TPosition::VECCALC> uBuf_;

    int32_t rank;
    int32_t rankSize;
    int32_t aivNum;

    static constexpr int32_t UB_MOVE_NUM = 16 * 1024;

    int32_t m;
    int32_t k;
    int32_t n;
    int32_t topK;
    int32_t expertPerRank;
    int32_t maxOutputSize;
    int32_t EP;
    int32_t listLen;

    optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData;
    optiling::InnerMoeInitRoutingV2TilingData moeInitRoutingV2TilingData;
    uint64_t initRoutingQuantTilingKey;
};


template <TemplateMMA2AClass>
__aicore__ inline void DispatchFFNCombine<TemplateMMA2ACFunc>::Init(
    GM_ADDR contextGM, GM_ADDR xGM, GM_ADDR topkIdsGM, GM_ADDR topkWeightsGM,
    GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR weightScales1GM, GM_ADDR weightScales2GM,
    GM_ADDR yGM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    constexpr bool kRoutingIsQuant = std::is_same_v<BType_, AscendC::int4b_t> ||
                                     std::is_same_v<BType_, int32_t> ||
                                     std::is_same_v<BType_, int8_t>;

    contextGM_ = contextGM;
    xGM_ = xGM;
    topkIdsGM_ = topkIdsGM;
    topkWeightsGM_ = topkWeightsGM;
    weight1GM_ = weight1GM;
    weight2GM_ = weight2GM;
    weightScales1GM_ = weightScales1GM;
    weightScales2GM_ = weightScales2GM;
    yGM_ = yGM;
    gmExpertTokenNums_ = expertTokenNumsGM;
    workspaceGM_ = workspaceGM;

    if constexpr (kRoutingIsQuant) {
        auto tiling = (__gm__ DispatchFFNCombineTilingDataQuant *)tilingGM;
        GET_TILING_DATA_WITH_STRUCT(DispatchFFNCombineTilingDataQuant, tilingData, tilingGM);

        aivNum = tilingData.dispatchFFNCombineInfo.common.aivNum;
        m = tilingData.dispatchFFNCombineInfo.common.M;
        k = tilingData.dispatchFFNCombineInfo.common.K;
        n = tilingData.dispatchFFNCombineInfo.common.N;
        EP = tilingData.dispatchFFNCombineInfo.common.worldSize;
        topK = tilingData.dispatchFFNCombineInfo.common.topK;
        expertPerRank = tilingData.dispatchFFNCombineInfo.common.expertPerRank;
        maxOutputSize = tilingData.dispatchFFNCombineInfo.common.maxRecvTokenNum;
        listLen = tilingData.dispatchFFNCombineInfo.common.listLen;

        moeInitRoutingQuantV2TilingData = tilingData.dispatchFFNCombineInfo.moeInitRoutingQuantV2TilingData;
        initRoutingQuantTilingKey = tilingData.dispatchFFNCombineInfo.common.initRoutingQuantTilingKey;
    } else {
        auto tiling = (__gm__ DispatchFFNCombineTilingDataNonQuant *)tilingGM;
        GET_TILING_DATA_WITH_STRUCT(DispatchFFNCombineTilingDataNonQuant, tilingData, tilingGM);

        aivNum = tilingData.dispatchFFNCombineInfo.common.aivNum;
        m = tilingData.dispatchFFNCombineInfo.common.M;
        k = tilingData.dispatchFFNCombineInfo.common.K;
        n = tilingData.dispatchFFNCombineInfo.common.N;
        EP = tilingData.dispatchFFNCombineInfo.common.worldSize;
        topK = tilingData.dispatchFFNCombineInfo.common.topK;
        expertPerRank = tilingData.dispatchFFNCombineInfo.common.expertPerRank;
        maxOutputSize = tilingData.dispatchFFNCombineInfo.common.maxRecvTokenNum;
        listLen = tilingData.dispatchFFNCombineInfo.common.listLen;

        moeInitRoutingV2TilingData = tilingData.dispatchFFNCombineInfo.moeInitRoutingV2TilingData;
        initRoutingQuantTilingKey = tilingData.dispatchFFNCombineInfo.common.initRoutingQuantTilingKey;
    }

    // The unified op exposes `context` as the first kernel input which represents
    // the HCCL group descriptor. The actual HCCL window context is still resolved
    // through the standard GetHcclContext() interface, but the layout of the
    // descriptor differs between Atlas A2 and A3 platforms.
    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    if constexpr (IS_A2_) {
        __gm__ HcclA2CombineOpParam *WinContext_{nullptr};
        WinContext_ = (__gm__ HcclA2CombineOpParam *)contextGM0;
        rank = WinContext_->rankId;
        rankSize = WinContext_->rankNum;
    } else {
        __gm__ HcclOpResParamCustom *WinContext_{nullptr};
        WinContext_ = (__gm__ HcclOpResParamCustom *)contextGM0;
        rank = WinContext_->localUsrRankId;
        rankSize = WinContext_->rankSize;
    }
}

template <TemplateMMA2AClass>
__aicore__ inline void DispatchFFNCombine<TemplateMMA2ACFunc>::Process()
{
    using ArchTag = Arch::AtlasA2;
    constexpr bool enableUnitFlag = false;
    constexpr bool enableShuffleK = true;

    // ------------------------------------------------------------------
    // Path selection based on the weight element type.
    //   - W4A8 path : weight is INT4 (or INT32 packed INT4)
    //   - INT8 path : weight is INT8
    //   - BF16 path : weight is FP16 / BF16
    // The matmul A side type is forced to match the weight side so that the
    // cube engine sees a homogeneous (A,B) pair.
    // ------------------------------------------------------------------
    constexpr bool isW4A8 = std::is_same_v<BType_, AscendC::int4b_t> || std::is_same_v<BType_, int32_t>;
    constexpr bool isInt8 = std::is_same_v<BType_, int8_t>;

    using AElement = std::conditional_t<isW4A8, AscendC::int4b_t, BType_>;
    using BElement = std::conditional_t<isW4A8, AscendC::int4b_t, BType_>;
    using CElement = std::conditional_t<isW4A8 || isInt8, float16_t, CType_>;
    using D1Element = std::conditional_t<isW4A8 || isInt8, int8_t, CType_>;

    uint32_t k2 = n / 2;
    uint32_t n2 = k;

    using LayoutA = layout::RowMajor;
    using LayoutB1 = typename std::conditional<
        NzW1_,
        layout::zN,
        typename std::conditional<TB1_, layout::ColumnMajor, layout::RowMajor>::type
    >::type;
    using LayoutB2 = typename std::conditional<
        NzW2_,
        layout::zN,
        typename std::conditional<TB2_, layout::ColumnMajor, layout::RowMajor>::type
    >::type;

    LayoutB1 layoutB1 = LayoutBInitializer<LayoutB1, BElement>::create(k, n);
    LayoutB2 layoutB2 = LayoutBInitializer<LayoutB2, BElement>::create(k2, n2);
    using LayoutC = layout::RowMajor;

    // Cube tile shapes are tuned per data-type combination to match the legacy
    // implementations:
    //   W4A8 : 128 x 256 x 1024 (L1) / 128 x 256 x 256 (L0)
    //   INT8 : 128 x 256 x  512 (L1) / 128 x 256 x 128 (L0)
    //   BF16 : 128 x 256 x  256 (L1) / 128 x 256 x  64 (L0)
    using L1TileShape = std::conditional_t<
        isW4A8,
        GemmShape<128, 256, 1024>,
        std::conditional_t<isInt8, GemmShape<128, 256, 512>, GemmShape<128, 256, 256>>
    >;
    using L0TileShape = std::conditional_t<
        isW4A8,
        GemmShape<128, 256, 256>,
        std::conditional_t<isInt8, GemmShape<128, 256, 128>, GemmShape<128, 256, 64>>
    >;

    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;

    // Single dispatch-policy alias: the W4A4-vs-Fixpipe choice is encapsulated
    // inside `MmadDispatchPolicyFor_t` (dispatch_policy_custom.hpp) and is driven
    // purely by the weight element type. The caller now spells one name only,
    // and there's no `MmadAtlasA2W4A4MatmulPerChannelDequant` branch visible
    // here -- isolation between W4A4 and Fixpipe is achieved through the
    // `BType_` data type alone.
    using DispatchPolicy = Gemm::MmadDispatchPolicyFor_t<
        BElement, preloadStages, l1Stages, l0AStages, l0BStages, l0CStages,
        enableUnitFlag, enableShuffleK>;

    using AType = Gemm::GemmType<AElement, layout::RowMajor>;
    using BType1 = Gemm::GemmType<BElement, LayoutB1>;
    using BType2 = Gemm::GemmType<BElement, LayoutB2>;
    using CType = Gemm::GemmType<CElement, layout::RowMajor>;
    using D1Type = Gemm::GemmType<D1Element, layout::RowMajor>;
    using D2Type = std::conditional_t<
        std::is_same_v<CType_, bfloat16_t>,
        Gemm::GemmType<bfloat16_t, layout::RowMajor>,
        Gemm::GemmType<CType_, layout::RowMajor>
    >;

    // Per-channel `QuantTileCopy` is only used by the W4A4 path; the Fixpipe
    // path uses BlockMmad's default tile copy. We still keep a separate alias
    // for B1/B2 because their layouts may differ (NzW1 vs NzW2) — type-level
    // isolation only.
    using ScaleGranularity = Catlass::Gemm::Tile::ScaleGranularity;
    using TileCopyMmad1 =
        Gemm::Tile::QuantTileCopy<ArchTag, AType, BType1, CType, void, ScaleGranularity::PER_CHANNEL>;
    using TileCopyMmad2 =
        Gemm::Tile::QuantTileCopy<ArchTag, AType, BType2, CType, void, ScaleGranularity::PER_CHANNEL>;

    // For the W4A4 dispatch policy, BlockMmad takes 8 template args (including
    // the explicit QuantTileCopy). For Fixpipe it takes 6 (default tile copy).
    // The conditional is reduced to a single line per Mmad and is selected
    // purely from `BElement` via `isW4A8`.
    using BlockMmad1 = std::conditional_t<
        isW4A8,
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType1, CType, void, TileCopyMmad1>,
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType1, CType>
    >;
    using BlockMmad2 = std::conditional_t<
        isW4A8,
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType2, CType, void, TileCopyMmad2>,
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType2, CType>
    >;

    constexpr uint32_t ubStages = 2;

    using EpilogueDispatchPolicy1 = std::conditional_t<
        isW4A8,
        Epilogue::EpilogueAtlasA2W4A8PostPerTokenDequantSwigluQuant<ubStages>,
        Epilogue::EpilogueAtlasA2PerTokenDequantSwigluQuant<ubStages>
    >;
    using EpilogueDispatchPolicy2 = std::conditional_t<
        isW4A8,
        Epilogue::EpilogueAtlasA2W4A8PostPerTokenDequantV2<ubStages>,
        Epilogue::EpilogueAtlasA2PerTokenDequantV2<ubStages>
    >;

    using ScaleType = Gemm::GemmType<uint64_t, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using ElementMulType = Gemm::GemmType<float, layout::RowMajor>;
    using TileElemWiseMuls = Epilogue::Tile::TileElemWiseMuls<ArchTag, ElementMulType, 0>;

    using TileCopy1 = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D1Type>;
    using BlockEpilogue1 = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy1, CType, PerTokenScaleType,
        D1Type, TileElemWiseMuls, TileCopy1>;

    using TileCopy2 = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D2Type>;
    using BlockEpilogue2 = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy2, CType, PerTokenScaleType,
        D2Type, TileCopy2>;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;
    using ElementGroupList = int64_t;
    // Forward `IS_A2_` to the unified kernel so that `HcclShmem<IS_A2_>` picks
    // the correct window-context structure (`HcclA2CombineOpParam` for A2 vs
    // `HcclOpResParamCustom` for A3) at compile time.
    using MatmulKernel = Gemm::Kernel::DispatchFFNCombineKernel<BlockMmad1,
        BlockMmad2, BlockScheduler, ElementGroupList, BlockEpilogue1, BlockEpilogue2, IS_A2_>;

    LayoutA layoutA1{static_cast<uint32_t>(m), static_cast<uint32_t>(k)};
    LayoutA layoutA2{static_cast<uint32_t>(m), static_cast<uint32_t>(k2)};
    layout::VectorLayout layoutScale1{static_cast<uint32_t>(n)};
    layout::VectorLayout layoutScale2{static_cast<uint32_t>(n2)};
    layout::RowMajor layoutD1{static_cast<uint32_t>(maxOutputSize), static_cast<uint32_t>(k2)};
    layout::RowMajor layoutD2{static_cast<uint32_t>(m * topK), static_cast<uint32_t>(n2)};

    GemmCoord problemShape{static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k)};

    // Epilogue scheduling depends on the data-type path, matching the
    // historic per-path tuning carried over from the three legacy kernels:
    //   - W4A8 : `epilogueCoreNum = aivNum`, no granularity throttle.
    //   - INT8 : `epilogueCoreNum = aivNum / 2`, granularity = expertPerRank - 1.
    //   - BF16 : `epilogueCoreNum = aivNum`,    granularity = expertPerRank - 2.
    uint32_t epilogueCoreNum;
    uint32_t epilogueGranularity;
    if constexpr (isW4A8) {
        epilogueCoreNum = static_cast<uint32_t>(aivNum);
        epilogueGranularity = 0U;
    } else if constexpr (isInt8) {
        epilogueCoreNum = static_cast<uint32_t>(aivNum) / 2U;
        epilogueGranularity = (expertPerRank > 0) ? static_cast<uint32_t>(expertPerRank - 1) : 0U;
    } else {
        // BF16 / FP16 path
        epilogueCoreNum = static_cast<uint32_t>(aivNum);
        epilogueGranularity = (expertPerRank > 1) ? static_cast<uint32_t>(expertPerRank - 2) : 0U;
    }

    constexpr bool kRoutingIsQuant = std::is_same_v<BType_, AscendC::int4b_t> ||
                                     std::is_same_v<BType_, int32_t> ||
                                     std::is_same_v<BType_, int8_t>;

    typename MatmulKernel::Params params;
    if constexpr (kRoutingIsQuant) {
        params = typename MatmulKernel::Params{
            problemShape, static_cast<uint32_t>(EP), static_cast<uint32_t>(listLen),
            static_cast<uint32_t>(expertPerRank), static_cast<uint32_t>(maxOutputSize),
            static_cast<uint32_t>(rank), static_cast<uint32_t>(rankSize),
            static_cast<uint32_t>(topK), initRoutingQuantTilingKey,
            epilogueCoreNum,
            xGM_, layoutA1, layoutA2,
            weight1GM_, layoutB1,
            weight2GM_, layoutB2,
            weightScales1GM_, layoutScale1,
            weightScales2GM_, layoutScale2,
            yGM_, layoutD1, layoutD2,
            topkIdsGM_, moeInitRoutingQuantV2Scale, moeInitRoutingQuantV2Offset,
            expertTokensBeforeCapacity, topkWeightsGM_,
            workspaceGM_, gmExpertTokenNums_, UB_MOVE_NUM, moeInitRoutingQuantV2TilingData,
            epilogueGranularity};
    } else {
        params = typename MatmulKernel::Params{
            problemShape, static_cast<uint32_t>(EP), static_cast<uint32_t>(listLen),
            static_cast<uint32_t>(expertPerRank), static_cast<uint32_t>(maxOutputSize),
            static_cast<uint32_t>(rank), static_cast<uint32_t>(rankSize),
            static_cast<uint32_t>(topK), initRoutingQuantTilingKey,
            epilogueCoreNum,
            xGM_, layoutA1, layoutA2,
            weight1GM_, layoutB1,
            weight2GM_, layoutB2,
            weightScales1GM_, layoutScale1,
            weightScales2GM_, layoutScale2,
            yGM_, layoutD1, layoutD2,
            topkIdsGM_, moeInitRoutingQuantV2Scale, moeInitRoutingQuantV2Offset,
            expertTokensBeforeCapacity, topkWeightsGM_,
            workspaceGM_, gmExpertTokenNums_, UB_MOVE_NUM, moeInitRoutingV2TilingData,
            epilogueGranularity};
    }

    MatmulKernel kernel(params);
    kernel(params);
}

} // namespace DispatchFFNCombineImpl
#endif // DISPATCH_FFN_COMBINE_H
