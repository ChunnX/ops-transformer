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
 * \file mega_moe_a2.h
 * \brief Atlas A2 (910B) specific MegaMoe implementation.
 *        Derived from mega_moe.h with A2-optimized utils.
 *        All utils/ references use _a2 suffixed versions where available.
 */

#ifndef MEGA_MOE_A2_H
#define MEGA_MOE_A2_H

using namespace AscendC;

#include "kernel_operator.h"

#include "../../../common/op_kernel/moe_distribute_base.h"

#include "mega_moe_tiling_a2a3.h"

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
#include "mc2_moe_context.h"

// A2 epilogue utils (V2 for CombineV2, SwiGLU for the GMM1 epilogue). A2 and A3 now share both.
#include "utils/block_epilogue_pertoken_swiglu.hpp"
#include "utils/block_epilogue_pertoken_v2.hpp"
// A2 kernel
#include "mega_moe_kernel_a2.hpp"
#include "mega_moe_kernel_a2_bf16_w4a8.hpp"

namespace MegaMoeA2Impl {

using namespace Catlass;
using namespace AscendC::HcclContextDef;
using namespace Mc2Tiling;
// Template parameter pack used by the MegaMoeA2 class.
//   AType_ : input X element type (DTYPE_X)
//   BType_ : weight element type   (DTYPE_WEIGHT1)
//   CType_ : output  element type  (DTYPE_Y)
//   TB1_   : transpose_weight1 attribute
//   TB2_   : transpose_weight2 attribute
//   Nz_    : whether weight1/2 is in FRACTAL_NZ format (both are the same)
//   IS_A2_ : whether running on Atlas A2 (910B); false means A3 (910_93)
#define MegaMoeClassA2 typename AType_, typename BType_, typename CType_, bool TB1_, bool TB2_, bool Nz_, bool IS_A2_
#define MegaMoeFuncA2 AType_, BType_, CType_, TB1_, TB2_, Nz_, IS_A2_

using namespace AscendC;
template <MegaMoeClassA2>
class MegaMoeA2 {
public:
    __aicore__ inline MegaMoeA2(){};
    __aicore__ inline void Init(GM_ADDR contextGM, GM_ADDR xGM, GM_ADDR topkIdsGM, GM_ADDR topkWeightsGM,
                                GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR weightScales1GM, GM_ADDR weightScales2GM,
                                GM_ADDR bias1GM, GM_ADDR bias2GM, GM_ADDR xActiveMaskGM, GM_ADDR scalesGM, GM_ADDR yGM,
                                GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM, GM_ADDR tilingGM);
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
    GM_ADDR tilingGM_{nullptr};

    GM_ADDR moeInitRoutingQuantV2Scale_ = nullptr;
    GM_ADDR moeInitRoutingQuantV2Offset_ = nullptr;
    GM_ADDR expertTokensBeforeCapacity_ = nullptr;

    TBuf<AscendC::TPosition::VECCALC> uBuf_;

    int32_t rank_;
    int32_t rankSize_;
    int32_t aivNum_;

    static constexpr int32_t UB_MOVE_NUM = 16 * 1024;

    int32_t m_;
    int32_t k_;
    int32_t n_;
    int32_t topK_;
    int32_t expertPerRank_;
    uint64_t maxOutputSize_;
    int32_t epWorldSize_;
    int32_t listLen_;
    uint32_t activationCode_;
    float activationClamp_;
    float activationParams1_;
    float activationParams2_;
};

template <MegaMoeClassA2>
__aicore__ inline void MegaMoeA2<MegaMoeFuncA2>::Init(GM_ADDR contextGM, GM_ADDR xGM, GM_ADDR topkIdsGM,
                                                      GM_ADDR topkWeightsGM, GM_ADDR weight1GM, GM_ADDR weight2GM,
                                                      GM_ADDR weightScales1GM, GM_ADDR weightScales2GM, GM_ADDR bias1GM,
                                                      GM_ADDR bias2GM, GM_ADDR xActiveMaskGM, GM_ADDR scalesGM,
                                                      GM_ADDR yGM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
                                                      GM_ADDR tilingGM)
{
    // Unified with mega_moe_arch22.h: A3 packs W4A8 weights as int32, so its predicate counts int32_t.
    // A2 never sees an int32 weight -- mega_moe.cpp guards the A2 branch with
    // `#if (ORIG_DTYPE_WEIGHT1 != DT_INT32)` -- so counting it here changes nothing on A2 and lets the
    // two platforms share one predicate.
    constexpr bool kRoutingIsQuant =
        std::is_same_v<BType_, AscendC::int4b_t> || std::is_same_v<BType_, int32_t> || std::is_same_v<BType_, int8_t>;

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
    tilingGM_ = tilingGM;

    if constexpr (kRoutingIsQuant) {
        auto tiling = (__gm__ MegaMoeTilingDataQuant *)tilingGM;
        GET_TILING_DATA_WITH_STRUCT(MegaMoeTilingDataQuant, tilingData, tilingGM);

        aivNum_ = tilingData.common.aivNum;
        m_ = tilingData.common.M;
        k_ = tilingData.common.K;
        n_ = tilingData.common.N;
        epWorldSize_ = tilingData.common.worldSize;
        topK_ = tilingData.common.topK;
        expertPerRank_ = tilingData.common.expertPerRank;
        // maxOutputSize 语义 = 发送侧单 chunk 接收预算 B（recvRoundBudget，host 已算好）；
        // 接收行按 chunk 分段（numChunks * B <= maxRecvTokenNum），kernel 内逐 chunk 复用
        maxOutputSize_ = tilingData.common.recvRoundBudget;
        listLen_ = tilingData.common.listLen;

        activationCode_ = tilingData.common.activationCode;
        activationClamp_ = tilingData.common.activationClamp;
        activationParams1_ = tilingData.common.activationParams1;
        activationParams2_ = tilingData.common.activationParams2;
    } else {
        auto tiling = (__gm__ MegaMoeTilingDataNonQuant *)tilingGM;
        GET_TILING_DATA_WITH_STRUCT(MegaMoeTilingDataNonQuant, tilingData, tilingGM);

        aivNum_ = tilingData.common.aivNum;
        m_ = tilingData.common.M;
        k_ = tilingData.common.K;
        n_ = tilingData.common.N;
        epWorldSize_ = tilingData.common.worldSize;
        topK_ = tilingData.common.topK;
        expertPerRank_ = tilingData.common.expertPerRank;
        // maxOutputSize 语义 = 发送侧单 chunk 接收预算 B（recvRoundBudget，host 已算好）
        maxOutputSize_ = tilingData.common.recvRoundBudget;
        listLen_ = tilingData.common.listLen;

        activationCode_ = tilingData.common.activationCode;
        activationClamp_ = tilingData.common.activationClamp;
        activationParams1_ = tilingData.common.activationParams1;
        activationParams2_ = tilingData.common.activationParams2;
    }
}

template <MegaMoeClassA2>
__aicore__ inline void MegaMoeA2<MegaMoeFuncA2>::Process()
{
    using ArchTag = Arch::AtlasA2;
    constexpr bool enableUnitFlag = false;
    constexpr bool enableShuffleK = true;
    // Unified with mega_moe_arch22.h -- see the note on kRoutingIsQuant above. A2 cannot reach the
    // int32 arm, so this is a no-op on A2 and removes the last predicate divergence between platforms.
    constexpr bool isW4A8 = std::is_same_v<BType_, AscendC::int4b_t> || std::is_same_v<BType_, int32_t>;
    constexpr bool isInt8 = std::is_same_v<BType_, int8_t>;

    using AElement = std::conditional_t<isW4A8, AscendC::int4b_t, std::conditional_t<isInt8, int8_t, AType_>>;
    using BElement = std::conditional_t<isW4A8, AscendC::int4b_t, BType_>;
    using CElement = std::conditional_t<isW4A8 || isInt8, float16_t, CType_>;
    using D1Element = std::conditional_t<isW4A8 || isInt8, int8_t, CType_>;

    uint32_t k2 = n_ / 2;
    uint32_t n2 = k_;

    int64_t activeNum = 0;
    int64_t expertCapacity = 0;
    int64_t expertNum = expertPerRank_ * epWorldSize_;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 2;
    bool expertTokensBeforeCapacityFlag = false;
    int64_t quantMode = 1;

    using LayoutA = layout::RowMajor;
    using LayoutB =
        typename std::conditional<Nz_, layout::zN,
                                  typename std::conditional<TB1_, layout::ColumnMajor, layout::RowMajor>::type>::type;

    LayoutB layoutB1 = LayoutBInitializer<LayoutB, BElement>::create(k_, n_);
    LayoutB layoutB2 = LayoutBInitializer<LayoutB, BElement>::create(k2, n2);
    using LayoutC = layout::RowMajor;

    using L1TileShape =
        std::conditional_t<isW4A8, GemmShape<128, 256, 1024>,
                           std::conditional_t<isInt8, GemmShape<128, 256, 512>, GemmShape<128, 256, 256>>>;
    using L0TileShape =
        std::conditional_t<isW4A8, GemmShape<128, 256, 256>,
                           std::conditional_t<isInt8, GemmShape<128, 256, 128>, GemmShape<128, 256, 64>>>;

    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;

    using DispatchPolicy = Gemm::MmadDispatchPolicyFor_t<BElement, IS_A2_, preloadStages, l1Stages, l0AStages,
                                                         l0BStages, l0CStages, enableUnitFlag, enableShuffleK>;

    using AType = Gemm::GemmType<AElement, layout::RowMajor>;
    using BType = Gemm::GemmType<BElement, LayoutB>;
    using CType = Gemm::GemmType<CElement, layout::RowMajor>;
    using D1Type = Gemm::GemmType<D1Element, layout::RowMajor>;
    using D2Type = std::conditional_t<std::is_same_v<CType_, bfloat16_t>, Gemm::GemmType<bfloat16_t, layout::RowMajor>,
                                      Gemm::GemmType<CType_, layout::RowMajor>>;

    using ScaleGranularity = Catlass::Gemm::Tile::ScaleGranularity;
    using TileCopyMmad = Gemm::Tile::QuantTileCopy<ArchTag, AType, BType, CType, void, ScaleGranularity::PER_CHANNEL>;

    using BlockMmad = std::conditional_t<
        isW4A8,
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopyMmad>,
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>>;

    constexpr uint32_t ubStages = 2;

    // A2 and A3 share the SwiGLU epilogue: the int8/bf16 A2-only variants were duplicates of the
    // quantised and non-quantised overloads of EpilogueAtlasA2PerTokenDequantSwigluQuant
    // (identical subblock task partitioning, identical activation, only the per-token-scale
    // application differed, which the applyPerTokenScale argument now carries).
    using EpilogueDispatchPolicy1 =
        std::conditional_t<isW4A8, Epilogue::EpilogueAtlasA2W4A8PostPerTokenDequantSwigluQuant<ubStages>,
                           Epilogue::EpilogueAtlasA2PerTokenDequantSwigluQuant<ubStages>>;

    // As with the SwiGLU epilogue, the A2-only V2Int8/V2BF16 variants were duplicates of the two
    // overload sets of EpilogueAtlasA2PerTokenDequantV2, selected by PerTokenV2Tag<IS_A2, isInt8>.
    using EpilogueDispatchPolicy2 =
        std::conditional_t<isW4A8, Epilogue::EpilogueAtlasA2W4A8PostPerTokenDequantV2<ubStages>,
                           Epilogue::EpilogueAtlasA2PerTokenDequantV2<ubStages>>;

    using ScaleType = Gemm::GemmType<uint64_t, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using ElementMulType = Gemm::GemmType<float, layout::RowMajor>;
    using TileElemWiseMuls = Epilogue::Tile::TileElemWiseMuls<ArchTag, ElementMulType, 0>;

    using TileCopy1 = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D1Type>;
    using BlockEpilogue1 = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy1, CType, PerTokenScaleType, D1Type,
                                                          TileElemWiseMuls, TileCopy1>;

    using TileCopy2 = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D2Type>;
    using IS_A2 = Epilogue::Block::IS_A2_T<IS_A2_>;
    using PlatformTag = Epilogue::Block::IS_A2_T<IS_A2_>;
    using PerTokenV2Tag = Epilogue::Block::PerTokenV2Tag<IS_A2_, isInt8>;
    using BlockEpilogue2 = std::conditional_t<
        isW4A8,
        Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy2, CType, PerTokenScaleType, D2Type, TileCopy2,
                                       PlatformTag>,
        Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy2, CType, PerTokenScaleType, D2Type, TileCopy2,
                                       PerTokenV2Tag>>;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;
    using ElementGroupList = int64_t;
    // W4A8 reuses this workspace for GMM1 input and SwiGLU/GMM2 input. Pad only
    // the GMM1 token slot; SwiGLU/GMM2 keeps its native packed k2 layout.
    uint32_t layoutA1RowStride = isW4A8 ? static_cast<uint32_t>(k_ > k2 ? k_ : k2) : static_cast<uint32_t>(k_);
    LayoutA layoutA1{static_cast<uint32_t>(m_), static_cast<uint32_t>(k_), layoutA1RowStride};
    LayoutA layoutA2{static_cast<uint32_t>(m_), static_cast<uint32_t>(k2)};
    layout::VectorLayout layoutScale1{static_cast<uint32_t>(n_)};
    layout::VectorLayout layoutScale2{static_cast<uint32_t>(n2)};
    layout::RowMajor layoutD1{static_cast<uint32_t>(maxOutputSize_), static_cast<uint32_t>(k2)};
    layout::RowMajor layoutD2{static_cast<uint32_t>(m_ * topK_), static_cast<uint32_t>(n2)};

    GemmCoord problemShape{static_cast<uint32_t>(m_), static_cast<uint32_t>(n_), static_cast<uint32_t>(k_)};

    uint32_t epilogueCoreNum;
    uint32_t epilogueGranularity;

    if constexpr (isW4A8) {
        using MatmulKernel = Gemm::Kernel::MegaMoeKernelA2Bf16W4A8<BlockMmad, BlockScheduler, ElementGroupList,
                                                                   BlockEpilogue1, BlockEpilogue2>;
        epilogueCoreNum = static_cast<uint32_t>(aivNum_);
        epilogueGranularity = 0U;
        typename MatmulKernel::Params params = typename MatmulKernel::Params{problemShape,
                                                                             static_cast<uint32_t>(epWorldSize_),
                                                                             static_cast<uint32_t>(listLen_),
                                                                             static_cast<uint32_t>(expertPerRank_),
                                                                             static_cast<uint64_t>(maxOutputSize_),
                                                                             static_cast<uint32_t>(topK_),
                                                                             epilogueCoreNum,
                                                                             contextGM_,
                                                                             xGM_,
                                                                             layoutA1,
                                                                             layoutA2,
                                                                             weight1GM_,
                                                                             layoutB1,
                                                                             bias1GM_,
                                                                             weight2GM_,
                                                                             layoutB2,
                                                                             bias2GM_,
                                                                             weightScales1GM_,
                                                                             layoutScale1,
                                                                             weightScales2GM_,
                                                                             layoutScale2,
                                                                             yGM_,
                                                                             layoutD1,
                                                                             layoutD2,
                                                                             topkIdsGM_,
                                                                             moeInitRoutingQuantV2Scale_,
                                                                             moeInitRoutingQuantV2Offset_,
                                                                             expertTokensBeforeCapacity_,
                                                                             topkWeightsGM_,
                                                                             workspaceGM_,
                                                                             gmExpertTokenNums_,
                                                                             xActiveMaskGM_,
                                                                             scalesGM_,
                                                                             epilogueGranularity,
                                                                             activationClamp_,
                                                                             activationCode_,
                                                                             activationParams1_,
                                                                             activationParams2_,
                                                                             tilingGM_};
        MatmulKernel kernel(params);
        kernel(params);
    } else if constexpr (isInt8) {
        epilogueCoreNum = aivNum_ / 2; // INT8 场景仅使用一半 AIV 参与 epilogue
        epilogueGranularity = (expertPerRank_ > 1) ? static_cast<uint32_t>(expertPerRank_ - 1) : 1u;

        using MatmulKernel = Gemm::Kernel::MegaMoeKernelA2<BlockMmad, BlockScheduler, ElementGroupList,
                                                           BlockEpilogue1, BlockEpilogue2, true>;
        typename MatmulKernel::Params params{problemShape,
                                             static_cast<uint32_t>(epWorldSize_),
                                             static_cast<uint32_t>(listLen_),
                                             static_cast<uint32_t>(expertPerRank_),
                                             static_cast<uint64_t>(maxOutputSize_),
                                             static_cast<uint32_t>(topK_),
                                             epilogueCoreNum,
                                             contextGM_,
                                             xGM_,
                                             layoutA1,
                                             layoutA2,
                                             weight1GM_,
                                             layoutB1,
                                             bias1GM_,
                                             weight2GM_,
                                             layoutB2,
                                             bias2GM_,
                                             weightScales1GM_,
                                             layoutScale1,
                                             weightScales2GM_,
                                             layoutScale2,
                                             yGM_,
                                             layoutD1,
                                             layoutD2,
                                             topkIdsGM_,
                                             moeInitRoutingQuantV2Scale_,
                                             moeInitRoutingQuantV2Offset_,
                                             expertTokensBeforeCapacity_,
                                             topkWeightsGM_,
                                             workspaceGM_,
                                             gmExpertTokenNums_,
                                             xActiveMaskGM_,
                                             scalesGM_,
                                             epilogueGranularity,
                                             activationClamp_,
                                             activationCode_,
                                             activationParams1_,
                                             activationParams2_,
                                             tilingGM_};

        MatmulKernel kernel(params);
        kernel(params);
    } else {
        epilogueCoreNum = static_cast<uint32_t>(aivNum_); // 所有 AIV 参与
        epilogueGranularity = (expertPerRank_ > 2) ? static_cast<uint32_t>(expertPerRank_ - 2) : 1u;
        using MatmulKernel = Gemm::Kernel::MegaMoeKernelA2<BlockMmad, BlockScheduler, ElementGroupList, BlockEpilogue1,
                                                           BlockEpilogue2, false>;
        typename MatmulKernel::Params params{problemShape,
                                             static_cast<uint32_t>(epWorldSize_),
                                             static_cast<uint32_t>(listLen_),
                                             static_cast<uint32_t>(expertPerRank_),
                                             static_cast<uint64_t>(maxOutputSize_),
                                             static_cast<uint32_t>(topK_),
                                             epilogueCoreNum,
                                             contextGM_,
                                             xGM_,
                                             layoutA1,
                                             layoutA2,
                                             weight1GM_,
                                             layoutB1,
                                             bias1GM_,
                                             weight2GM_,
                                             layoutB2,
                                             bias2GM_,
                                             weightScales1GM_,
                                             layoutScale1,
                                             weightScales2GM_,
                                             layoutScale2,
                                             yGM_,
                                             layoutD1,
                                             layoutD2,
                                             topkIdsGM_,
                                             moeInitRoutingQuantV2Scale_,
                                             moeInitRoutingQuantV2Offset_,
                                             expertTokensBeforeCapacity_,
                                             topkWeightsGM_,
                                             workspaceGM_,
                                             gmExpertTokenNums_,
                                             xActiveMaskGM_,
                                             scalesGM_,
                                             epilogueGranularity,
                                             activationClamp_,
                                             activationCode_,
                                             activationParams1_,
                                             activationParams2_,
                                             tilingGM_};

        MatmulKernel kernel(params);
        kernel(params);
    }
}

} // namespace MegaMoeA2Impl
#endif // MEGA_MOE_A2_H
