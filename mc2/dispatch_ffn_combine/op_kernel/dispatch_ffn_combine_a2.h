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
 * \file dispatch_ffn_combine_a2.h
 * \brief Atlas A2 (910B) specific DispatchFFNCombine implementation.
 *        Derived from dispatch_ffn_combine.h with A2-optimized utils.
 *        All utils/ references use _a2 suffixed versions where available.
 */

#ifndef DISPATCH_FFN_COMBINE_A2_H
#define DISPATCH_FFN_COMBINE_A2_H

using namespace AscendC;

#include "kernel_operator.h"

#include "../../common/op_kernel/moe_distribute_base.h"

#include "dispatch_ffn_combine_tiling.h"
#include "moe_init_routing_v2/moe_init_routing_v2_tiling.h"

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
#include "utils/dispatch_policy_custom.hpp"
#include "utils/mc2_context.hpp"

// A2 epilogue utils (V2 variants for CombineV2, SwiGLU variants for GMM1 epilogue)
#include "utilsA2/block_epilogue_pertoken_v2_int8_a2.hpp"
#include "utilsA2/block_epilogue_pertoken_v2_bf16_a2.hpp"
#include "utilsA2/block_epilogue_pertoken_swiglu_a2_int8.hpp"
#include "utilsA2/block_epilogue_pertoken_swiglu_a2_bf16.hpp"

// A2 kernel
#include "dispatch_ffn_combine_kernel_a2_int8.hpp"
#include "dispatch_ffn_combine_kernel_a2_bf16.hpp"
#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"

using namespace Catlass;
using namespace AscendC::HcclContextDef;
using namespace Mc2Tiling;

namespace DispatchFFNCombineA2Impl {
// Template parameter pack used by the DispatchFFNCombineA2 class.
//   AType_ : input X element type (DTYPE_X)
//   BType_ : weight element type   (DTYPE_WEIGHT1)
//   CType_ : output  element type  (DTYPE_Y)
//   TB1_   : transpose_weight1 attribute
//   TB2_   : transpose_weight2 attribute
//   Nz_    : whether weight1/2 is in FRACTAL_NZ format (both are the same)
//   IS_A2_ : whether running on Atlas A2 (910B); false means A3 (910_93)
#define MegaMoeClassA2                                                                                                \
    typename AType_, typename BType_, typename CType_, bool TB1_, bool TB2_, bool Nz_, bool IS_A2_
#define MegaMoeFuncA2 AType_, BType_, CType_, TB1_, TB2_, Nz_, IS_A2_

using namespace AscendC;
template <MegaMoeClassA2>
class DispatchFFNCombineA2 {
public:
    __aicore__ inline DispatchFFNCombineA2() {};
    __aicore__ inline void Init(GM_ADDR contextGM, GM_ADDR xGM, GM_ADDR topkIdsGM, GM_ADDR topkWeightsGM,
                                GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR weightScales1GM, GM_ADDR weightScales2GM,
                                GM_ADDR bias1GM, GM_ADDR bias2GM, GM_ADDR xActiveMaskGM, GM_ADDR scalesGM,
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
    GM_ADDR bias1GM_;
    GM_ADDR bias2GM_;
    GM_ADDR xActiveMaskGM_;
    GM_ADDR scalesGM_;
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
    float activationClamp;

    MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData;
    MoeInitRoutingV2TilingData moeInitRoutingV2TilingData;
    uint64_t initRoutingQuantTilingKey;
};


template <MegaMoeClassA2>
__aicore__ inline void DispatchFFNCombineA2<MegaMoeFuncA2>::Init(
    GM_ADDR contextGM, GM_ADDR xGM, GM_ADDR topkIdsGM, GM_ADDR topkWeightsGM,
    GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR weightScales1GM, GM_ADDR weightScales2GM,
    GM_ADDR bias1GM, GM_ADDR bias2GM, GM_ADDR xActiveMaskGM, GM_ADDR scalesGM,
    GM_ADDR yGM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    constexpr bool kRoutingIsQuant = std::is_same_v<BType_, int8_t>;

    contextGM_ = contextGM;
    xGM_ = xGM;
    topkIdsGM_ = topkIdsGM;
    topkWeightsGM_ = topkWeightsGM;
    weight1GM_ = weight1GM;
    weight2GM_ = weight2GM;
    weightScales1GM_ = weightScales1GM;
    weightScales2GM_ = weightScales2GM;
    bias1GM_ = bias1GM;
    bias2GM_ = bias2GM;
    xActiveMaskGM_ = xActiveMaskGM;
    scalesGM_ = scalesGM;
    yGM_ = yGM;
    gmExpertTokenNums_ = expertTokenNumsGM;
    workspaceGM_ = workspaceGM;

    if constexpr (kRoutingIsQuant) {
        auto tiling = (__gm__ DispatchFFNCombineTilingDataQuant *)tilingGM;
        GET_TILING_DATA_WITH_STRUCT(DispatchFFNCombineTilingDataQuant, tilingData, tilingGM);

        aivNum = tilingData.common.aivNum;
        m = tilingData.common.M;
        k = tilingData.common.K;
        n = tilingData.common.N;
        EP = tilingData.common.worldSize;
        topK = tilingData.common.topK;
        expertPerRank = tilingData.common.expertPerRank;
        maxOutputSize = tilingData.common.maxRecvTokenNum;
        listLen = tilingData.common.listLen;

        activationClamp = tilingData.common.activationClamp;

        moeInitRoutingQuantV2TilingData = tilingData.moeInitRoutingQuantV2TilingData;
        initRoutingQuantTilingKey = tilingData.common.initRoutingQuantTilingKey;
    } else {
        auto tiling = (__gm__ DispatchFFNCombineTilingDataNonQuant *)tilingGM;
        GET_TILING_DATA_WITH_STRUCT(DispatchFFNCombineTilingDataNonQuant, tilingData, tilingGM);

        aivNum = tilingData.common.aivNum;
        m = tilingData.common.M;
        k = tilingData.common.K;
        n = tilingData.common.N;
        EP = tilingData.common.worldSize;
        topK = tilingData.common.topK;
        expertPerRank = tilingData.common.expertPerRank;
        maxOutputSize = tilingData.common.maxRecvTokenNum;
        listLen = tilingData.common.listLen;

        activationClamp = tilingData.common.activationClamp;

        moeInitRoutingV2TilingData = tilingData.moeInitRoutingV2TilingData;
        initRoutingQuantTilingKey = tilingData.common.initRoutingQuantTilingKey;
    }

}

template <MegaMoeClassA2>
__aicore__ inline void DispatchFFNCombineA2<MegaMoeFuncA2>::Process()
{
    using ArchTag = Arch::AtlasA2;
    constexpr bool enableUnitFlag = false;
    constexpr bool enableShuffleK = true;

    constexpr bool isInt8 = std::is_same_v<BType_, int8_t>;

    using AElement = std::conditional_t<isInt8, int8_t, AType_>;
    using BElement = BType_;
    using CElement = std::conditional_t<isInt8, float16_t, CType_>;
    using D1Element = std::conditional_t<isInt8, int8_t, CType_>;

    uint32_t k2 = n / 2;
    uint32_t n2 = k;

    int64_t activeNum = 0;
    int64_t expertCapacity = 0;
    int64_t expertNum = expertPerRank * EP;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 2;
    bool expertTokensBeforeCapacityFlag = false;
    int64_t quantMode = 1;

    using LayoutA = layout::RowMajor;
    using LayoutB = typename std::conditional<
        Nz_,
        layout::zN,
        typename std::conditional<TB1_, layout::ColumnMajor, layout::RowMajor>::type
    >::type;

    LayoutB layoutB1 = LayoutBInitializer<LayoutB, BElement>::create(k, n);
    LayoutB layoutB2 = LayoutBInitializer<LayoutB, BElement>::create(k2, n2);
    using LayoutC = layout::RowMajor;

    using L1TileShape = std::conditional_t<isInt8,
        GemmShape<128, 256, 512>, GemmShape<128, 256, 256>>;
    using L0TileShape = std::conditional_t<isInt8,
        GemmShape<128, 256, 128>, GemmShape<128, 256, 64>>;

    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;

    // Single dispatch-policy alias: the W4A4-vs-Fixpipe choice is encapsulated
    // inside `MmadDispatchPolicyFor_t` (dispatch_policy_custom.hpp) and is driven
    // purely by the weight element type. The caller now spells one name only.
    using DispatchPolicy = Gemm::MmadDispatchPolicyFor_t<
        BElement, preloadStages, l1Stages, l0AStages, l0BStages, l0CStages,
        enableUnitFlag, enableShuffleK>;

    using AType = Gemm::GemmType<AElement, layout::RowMajor>;
    using BType = Gemm::GemmType<BElement, LayoutB>;
    using CType = Gemm::GemmType<CElement, layout::RowMajor>;
    using D1Type = Gemm::GemmType<D1Element, layout::RowMajor>;
    using D2Type = std::conditional_t<
        std::is_same_v<CType_, bfloat16_t>,
        Gemm::GemmType<bfloat16_t, layout::RowMajor>,
        Gemm::GemmType<CType_, layout::RowMajor>
    >;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    constexpr uint32_t ubStages = 2;

    using EpilogueDispatchPolicy1 = std::conditional_t<
        isInt8,
        Epilogue::EpilogueAtlasA2PerTokenDequantSwigluQuantInt8<ubStages>,
        Epilogue::EpilogueAtlasA2PerTokenDequantSwigluQuantBF16<ubStages>
    >;
    // A2 (910B) only supports INT8 and BF16/FP16 paths; no W4A8.
    // For INT8:   V2Int8  (per-token scale dequant + RDMA dispatch).
    // For non-INT8: V2BF16 (copy-only dispatch; no per-token dequant needed
    //                regardless of whether output is bfloat16 or half).
    using EpilogueDispatchPolicy2 = std::conditional_t<
        isInt8,
        Epilogue::EpilogueAtlasA2PerTokenDequantV2Int8<ubStages>,
        Epilogue::EpilogueAtlasA2PerTokenDequantV2BF16<ubStages>
    >;

    using ScaleType = Gemm::GemmType<uint64_t, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using ElementMulType = Gemm::GemmType<float, layout::RowMajor>;
    using TileElemWiseMuls = Epilogue::Tile::TileElemWiseMuls<ArchTag, ElementMulType, 0>;

    using TileCopy1 = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D1Type>;
    using BlockEpilogue1 = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy1, CType, PerTokenScaleType,
        D1Type, TileElemWiseMuls, TileCopy1>;

    using TileCopy2 = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D2Type>;
    // A2 variant epilogues are 5-arg specializations (no IS_A2_T needed — IS_A2 is always true).
    using BlockEpilogue2 = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy2, CType, PerTokenScaleType,
        D2Type, TileCopy2>;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;
    using ElementGroupList = int64_t;
    LayoutA layoutA1{static_cast<uint32_t>(m), static_cast<uint32_t>(k)};
    LayoutA layoutA2{static_cast<uint32_t>(m), static_cast<uint32_t>(k2)};
    layout::VectorLayout layoutScale1{static_cast<uint32_t>(n)};
    layout::VectorLayout layoutScale2{static_cast<uint32_t>(n2)};
    layout::RowMajor layoutD1{static_cast<uint32_t>(maxOutputSize), static_cast<uint32_t>(k2)};
    layout::RowMajor layoutD2{static_cast<uint32_t>(m * topK), static_cast<uint32_t>(n2)};

    GemmCoord problemShape{static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k)};

    // Epilogue scheduling: INT8 vs BF16 (no W4A8 path in A2).
    uint32_t epilogueCoreNum;
    uint32_t epilogueGranularity;
    if constexpr (isInt8) {
        epilogueCoreNum = static_cast<uint32_t>(aivNum);
        epilogueGranularity = (expertPerRank > 4) ? static_cast<uint32_t>(expertPerRank - 3) :
                              (expertPerRank > 0) ? static_cast<uint32_t>(expertPerRank - 1) : 0U;

        using MatmulKernel = Gemm::Kernel::DispatchFFNCombineKernelA2Int8<BlockMmad, BlockScheduler, ElementGroupList,
                                                                           BlockEpilogue1, BlockEpilogue2>;
        typename MatmulKernel::Params params{
            problemShape, static_cast<uint32_t>(EP), static_cast<uint32_t>(listLen),
            static_cast<uint32_t>(expertPerRank), static_cast<uint32_t>(maxOutputSize),
            static_cast<uint32_t>(topK), initRoutingQuantTilingKey,
            epilogueCoreNum,
            contextGM_, xGM_, layoutA1, layoutA2,
            weight1GM_, layoutB1, bias1GM_,
            weight2GM_, layoutB2, bias2GM_,
            weightScales1GM_, layoutScale1,
            weightScales2GM_, layoutScale2,
            yGM_, layoutD1, layoutD2,
            topkIdsGM_, moeInitRoutingQuantV2Scale, moeInitRoutingQuantV2Offset,
            expertTokensBeforeCapacity, topkWeightsGM_,
            workspaceGM_, gmExpertTokenNums_, xActiveMaskGM_, scalesGM_,
            moeInitRoutingQuantV2TilingData,
            epilogueGranularity, activationClamp};

        MatmulKernel kernel(params);
        kernel(params);
     } else {
        epilogueCoreNum = static_cast<uint32_t>(aivNum);
        epilogueGranularity = (expertPerRank > 1) ? static_cast<uint32_t>(expertPerRank - 2) : 0U;

        using MatmulKernel = Gemm::Kernel::DispatchFFNCombineKernelA2BF16<BlockMmad, BlockScheduler, ElementGroupList,
                                                                           BlockEpilogue1, BlockEpilogue2>;
        typename MatmulKernel::Params params{
            problemShape, static_cast<uint32_t>(EP), static_cast<uint32_t>(listLen),
            static_cast<uint32_t>(expertPerRank), static_cast<uint32_t>(maxOutputSize),
            static_cast<uint32_t>(topK), initRoutingQuantTilingKey,
            epilogueCoreNum,
            contextGM_, xGM_, layoutA1, layoutA2,
            weight1GM_, layoutB1, bias1GM_,
            weight2GM_, layoutB2, bias2GM_,
            weightScales1GM_, layoutScale1,
            weightScales2GM_, layoutScale2,
            yGM_, layoutD1, layoutD2,
            topkIdsGM_, moeInitRoutingQuantV2Scale, moeInitRoutingQuantV2Offset,
            expertTokensBeforeCapacity, topkWeightsGM_,
            workspaceGM_, gmExpertTokenNums_, xActiveMaskGM_, scalesGM_,
            moeInitRoutingV2TilingData,
            epilogueGranularity, activationClamp};

        MatmulKernel kernel(params);
        kernel(params);
     }
}

} // namespace DispatchFFNCombineA2Impl
#endif // DISPATCH_FFN_COMBINE_A2_H
