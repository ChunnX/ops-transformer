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
 * \file mega_moe_arch22.h
 * \brief Unified MegaMoe implementation that merges the three legacy paths:
 *          - mega_moe          (INT8 weight)
 *          - mega_moe_bf16     (FP16 / BF16 weight)
 *          - mega_moe_w4_a8    (INT4 weight, W4A8)
 *        Differences from the legacy single-dtype implementations:
 *          - Inputs use the new unified naming (x / topk_ids / topk_weights / weight1
 *            / weight_scales1 / ... ) following the unified op def.
 *          - The first input is `context` (replaces the removed `group` attribute).
 *          - The single legacy `transB` attribute is split into `transpose_weight1`
 *            and `transpose_weight2`, surfaced via two template flags.
 *          - The legacy `M` attribute is renamed to `max_recv_token_num`.
 *          - Tiling fields are read in the order defined in the unified op def.
 */

#ifndef MEGA_MOE_ARCH22_H
#define MEGA_MOE_ARCH22_H

using namespace AscendC;

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#if __has_include("../../common/moe_distribute_base.h")
#include "../../common/moe_distribute_base.h"
#else
#include "../../../common/op_kernel/moe_distribute_base.h"
#endif

#include "mega_moe_tiling_a2a3.h"
#include "moe_init_routing_v2/mc2_mega_moe_moe_init_routing_v2_tiling.h"

#include "template_linear_algebra_v2/catlass.hpp"
#include "template_linear_algebra_v2/arch/arch.hpp"
#include "template_linear_algebra_v2/epilogue/dispatch_policy.hpp"
#include "template_linear_algebra_v2/epilogue/block/block_epilogue.hpp"
#include "template_linear_algebra_v2/epilogue/tile/tile_copy.hpp"
#include "template_linear_algebra_v2/epilogue/tile/tile_elemwise_add.hpp"
#include "template_linear_algebra_v2/epilogue/tile/tile_elemwise_muls.hpp"
#include "template_linear_algebra_v2/gemm/block/block_mmad.hpp"
#include "template_linear_algebra_v2/gemm/block/block_swizzle.hpp"
#include "template_linear_algebra_v2/gemm/dispatch_policy.hpp"
#include "template_linear_algebra_v2/gemm/kernel/matmul_epilogue.hpp"
#include "template_linear_algebra_v2/gemm/gemm_type.hpp"
#include "template_linear_algebra_v2/layout/layout.hpp"

#include "utils/select_helper.hpp"
#include "utils/const_args.hpp"
#include "utils/dispatch_policy_custom.hpp"
#include "mc2_moe_context.h"

// A2-specific epilogue specializations. BF16/FP16, INT8 and W4A8 share one
// policy-based device kernel; packed-INT4 differences remain compile-time paths.
#include "utils/block_epilogue_pertoken_swiglu.hpp"
#include "utils/block_epilogue_pertoken_v2.hpp"

#include "mega_moe_kernel_a2.hpp"
#include "mega_moe_kernel_a3.hpp"
#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"

namespace MegaMoeImpl {
using namespace AscendC;
using namespace AscendC::HcclContextDef;
using namespace Catlass;
using namespace Mc2Tiling;

// Template parameter pack used by the unified MegaMoe class.
//   AType_ : input X element type (DTYPE_X)
//   BType_ : weight element type   (DTYPE_WEIGHT1)
//   CType_ : output  element type  (DTYPE_Y)
//   TB1_   : transpose_weight1 attribute
//   TB2_   : transpose_weight2 attribute
//   Nz_    : whether weight1/2 is in FRACTAL_NZ format (both are the same)
//   IS_A2_ : whether running on Atlas A2 (910B); false means A3 (910_93)
#define MegaMoeClass typename AType_, typename BType_, typename CType_, bool TB1_, bool TB2_, bool Nz_, bool IS_A2_
#define MegaMoeFunc AType_, BType_, CType_, TB1_, TB2_, Nz_, IS_A2_

template <MegaMoeClass>
class MegaMoe {
public:
    __aicore__ inline MegaMoe(){};
    __aicore__ inline void Init(GM_ADDR contextGM, GM_ADDR xGM, GM_ADDR topkIdsGM, GM_ADDR topkWeightsGM,
                                GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR weightScales1GM, GM_ADDR weightScales2GM,
                                GM_ADDR bias1GM, GM_ADDR bias2GM, GM_ADDR xActiveMaskGM, GM_ADDR scalesGM, GM_ADDR yGM,
                                GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM, GM_ADDR tilingGM);
    __aicore__ inline void Process();

private:
    template <class MatmulKernel_, class LayoutA_, class LayoutB_>
    __aicore__ inline void LaunchKernel(
        const GemmCoord &problemShape, const LayoutA_ &layoutA1, const LayoutA_ &layoutA2,
        const LayoutB_ &layoutB1, const LayoutB_ &layoutB2, const layout::VectorLayout &layoutScale1,
        const layout::VectorLayout &layoutScale2, const layout::RowMajor &layoutD1,
        const layout::RowMajor &layoutD2, uint32_t epilogueCoreNum, uint32_t epilogueGranularity);

    static constexpr bool kRoutingIsQuant =
        std::is_same_v<BType_, AscendC::int4b_t> || std::is_same_v<BType_, int8_t>;
    using TilingData = std::conditional_t<kRoutingIsQuant, MegaMoeTilingDataQuant, MegaMoeTilingDataNonQuant>;
    using RoutingTilingData =
        std::conditional_t<kRoutingIsQuant, MoeInitRoutingQuantV2TilingData, MoeInitRoutingV2TilingData>;

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
    GM_ADDR tilingGM_;

    GM_ADDR moeInitRoutingQuantV2Scale_ = nullptr;
    GM_ADDR moeInitRoutingQuantV2Offset_ = nullptr;
    GM_ADDR expertTokensBeforeCapacity_ = nullptr;

    TBuf<AscendC::TPosition::VECCALC> uBuf_;

    int32_t aivNum_;

    // static constexpr int32_t UB_MOVE_NUM = 16 * 1024;

    int32_t m_;
    int32_t k_;
    int32_t n_;
    int32_t k2_;
    int32_t n2_;
    int32_t topK_;
    int32_t expertPerRank_;
    uint64_t maxOutputSize_;
    int32_t epWorldSize_;
    int32_t listLen_;
    uint32_t activationCode_;
    float activationClamp_;
    float activationParams1_;
    float activationParams2_;

    RoutingTilingData moeInitRoutingTilingData_;
    uint64_t initRoutingQuantTilingKey;
};


template <MegaMoeClass>
__aicore__ inline void MegaMoe<MegaMoeFunc>::Init(GM_ADDR contextGM, GM_ADDR xGM, GM_ADDR topkIdsGM,
                                                  GM_ADDR topkWeightsGM, GM_ADDR weight1GM, GM_ADDR weight2GM,
                                                  GM_ADDR weightScales1GM, GM_ADDR weightScales2GM, GM_ADDR bias1GM,
                                                  GM_ADDR bias2GM, GM_ADDR xActiveMaskGM, GM_ADDR scalesGM, GM_ADDR yGM,
                                                  GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
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

    GET_TILING_DATA_WITH_STRUCT(TilingData, tilingData, tilingGM);
    moeInitRoutingTilingData_ = tilingData.moeInitRoutingTilingData;

    aivNum_ = tilingData.common.aivNum;
    m_ = tilingData.common.M;
    k_ = tilingData.common.K;
    n_ = tilingData.common.N;
    k2_ = n_ / 2;
    n2_ = k_;
    epWorldSize_ = tilingData.common.worldSize;
    topK_ = tilingData.common.topK;
    expertPerRank_ = tilingData.common.expertPerRank;
    maxOutputSize_ = tilingData.common.maxRecvTokenNum;
    listLen_ = tilingData.common.listLen;

    activationCode_ = tilingData.common.activationCode;
    activationClamp_ = tilingData.common.activationClamp;
    activationParams1_ = tilingData.common.activationParams1;
    activationParams2_ = tilingData.common.activationParams2;
    initRoutingQuantTilingKey = tilingData.common.initRoutingQuantTilingKey;
}

// All A2/A3 kernels expose the same top-level Params constructor. Keep the
// argument forwarding in one place so platform/dtype branches only select
// policies, kernel types, and scheduling values.
template <MegaMoeClass>
template <class MatmulKernel_, class LayoutA_, class LayoutB_>
__aicore__ inline void MegaMoe<MegaMoeFunc>::LaunchKernel(
    const GemmCoord &problemShape, const LayoutA_ &layoutA1, const LayoutA_ &layoutA2,
    const LayoutB_ &layoutB1, const LayoutB_ &layoutB2, const layout::VectorLayout &layoutScale1,
    const layout::VectorLayout &layoutScale2, const layout::RowMajor &layoutD1,
    const layout::RowMajor &layoutD2, uint32_t epilogueCoreNum, uint32_t epilogueGranularity)
{
    typename MatmulKernel_::Params params{problemShape,
                                          static_cast<uint32_t>(epWorldSize_),
                                          static_cast<uint32_t>(listLen_),
                                          static_cast<uint32_t>(expertPerRank_),
                                          static_cast<uint64_t>(maxOutputSize_),
                                          static_cast<uint32_t>(topK_),
                                          initRoutingQuantTilingKey,
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
                                          moeInitRoutingTilingData_,
                                          epilogueGranularity,
                                          activationClamp_,
                                          activationCode_,
                                          activationParams1_,
                                          activationParams2_,
                                          tilingGM_};

    MatmulKernel_ kernel(params);
    kernel(params);
}

template <MegaMoeClass>
__aicore__ inline void MegaMoe<MegaMoeFunc>::Process()
{
    // CATLASS 计算与 epilogue 使用的基础架构标签。Atlas A3 路径复用 AtlasA2
    // 的指令抽象，A2/A3 通信上下文的差异由后面的 IS_A2 类型标签区分。
    using ArchTag = Arch::AtlasA2;
    constexpr bool enableUnitFlag = false;
    constexpr bool enableShuffleK = true;

    constexpr bool isW4A8 = std::is_same_v<BType_, AscendC::int4b_t>;
    constexpr bool isInt8 = std::is_same_v<BType_, int8_t>;

    // MMAD 输入、累加结果和 GMM1 输出的实际元素类型：
    // - W4A8：dispatch 后的激活和权重都按 int4b_t 参与 MMAD，GMM 输出暂存为 FP16，
    //   GMM1 epilogue 再量化成 INT8，作为 SwiGLU 后的 GMM2 输入。
    // - INT8：激活和权重均按 INT8 参与 MMAD，GMM 输出暂存为 FP16，
    //   GMM1 epilogue 同样输出 INT8。
    // - BF16/FP16：保持算子模板传入的数据类型，GMM1 epilogue 不做 INT8 再量化。
    using AElement = BType_;
    using BElement = BType_;
    using CElement = std::conditional_t<isW4A8 || isInt8, float16_t, CType_>;
    using D1Element = std::conditional_t<isW4A8 || isInt8, int8_t, CType_>;

    // A/C 均按行主序访问。权重布局根据输入格式和 transpose_weight1 选择：
    // FRACTAL_NZ 优先映射为 zN；ND 权重再在 ColumnMajor/RowMajor 之间选择。
    using LayoutA = layout::RowMajor;
    using LayoutB =
        typename std::conditional<Nz_, layout::zN,
                                  typename std::conditional<TB1_, layout::ColumnMajor, layout::RowMajor>::type>::type;
    using LayoutC = layout::RowMajor;

    // 将裸元素类型与逻辑布局封装成 CATLASS GemmType：
    // AType/BType/CType 分别描述 MMAD 的 A、B 和中间输出 C；
    // BiasType 默认没有, 非int4场景不支持bias，
    // int4 场景的 bias 在反量化之后在 vector Core 完成求和，不依赖 biasTable
    // D1Type 是 GMM1 epilogue（SwiGLU 后）写出的 GMM2 输入；
    // D2Type 是 GMM2/Combine 写出的最终数据类型。
    using AType = Gemm::GemmType<AElement, LayoutA>;
    using BType = Gemm::GemmType<BElement, LayoutB>;
    using CType = Gemm::GemmType<CElement, LayoutC>;
    using BiasType = void;
    using D1Type = Gemm::GemmType<D1Element, LayoutC>;
    using D2Type = Gemm::GemmType<CType_, LayoutC>;

    // L1/L0 tile 形状按权重位宽选择。位宽越低，一次 tile 可装入的 K 维元素越多：
    // W4A8 分别使用 K=1024/256，INT8 使用 K=512/128，BF16/FP16 使用 K=256/64。
    // M/N tile 固定为 128x256，便于两次 GMM 共用调度和 epilogue 结构。
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

    // MMAD 流水策略：把元素类型、芯片代际和 L1/L0 多级缓存 stage 数编码进类型，
    // 由 CATLASS 在编译期选择对应的数据搬运、预取和矩阵乘实现，不产生运行时分支。
    using DispatchPolicy = Gemm::MmadDispatchPolicyFor_t<BElement, IS_A2_, preloadStages, l1Stages, l0AStages,
                                                         l0BStages, l0CStages, enableUnitFlag, enableShuffleK>;

    // W4A8 MMAD 需要专用的量化 tile copy：按 PER_CHANNEL 粒度搬运/解释权重 scale。
    // 非 W4A8 路径不向 BlockMmad 注入该定制 copy 类型。
    using ScaleGranularity = Catlass::Gemm::Tile::ScaleGranularity;
    using DefaultTileCopyMmad = Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, BiasType>;
    using QuantTileCopyMmad = Gemm::Tile::QuantTileCopy<ArchTag, AType, BType, CType, BiasType, ScaleGranularity::PER_CHANNEL>;
    using TileCopyMmad = std::conditional_t<isW4A8, QuantTileCopyMmad, DefaultTileCopyMmad>;

    // 完整的块级矩阵乘实现。它组合 MMAD 流水策略、L1/L0 tile、数据类型和可选量化搬运，
    // 被 MegaMoeKernel 同时用于专家 GMM1 和 GMM2 的块级计算。
    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, BiasType, TileCopyMmad>;

    constexpr uint32_t ubStages = 2;

    // Epilogue data descriptions shared by both platforms.
    using ScaleType = Gemm::GemmType<uint64_t, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using ElementMulType = Gemm::GemmType<float, layout::RowMajor>;
    using TileElemWiseMuls = Epilogue::Tile::TileElemWiseMuls<ArchTag, ElementMulType, 0>;
    using TileCopy1 = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D1Type>;
    using TileCopy2 = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D2Type>;
    using PlatformTag = Epilogue::Block::IS_A2_T<IS_A2_>;
    using PerTokenV2Tag = Epilogue::Block::PerTokenV2Tag<IS_A2_, isInt8>;
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;
    using ElementGroupList = int64_t;

    // A2 W4A8 reuses the GMM1 input workspace for the packed SwiGLU output.
    // Preserve its padded input stride; other paths retain the native K stride.
    uint32_t layoutA1RowStride = static_cast<uint32_t>(k_);
    if constexpr (IS_A2_ && isW4A8) {
        layoutA1RowStride = static_cast<uint32_t>(k_ > k2_ ? k_ : k2_);
    }
    LayoutA layoutA1{static_cast<uint32_t>(m_), static_cast<uint32_t>(k_), layoutA1RowStride};
    LayoutA layoutA2{static_cast<uint32_t>(m_), static_cast<uint32_t>(k2_)};
    LayoutB layoutB1 = LayoutBInitializer<LayoutB, BElement>::create(k_, n_);
    LayoutB layoutB2 = LayoutBInitializer<LayoutB, BElement>::create(k2_, n2_);
    layout::VectorLayout layoutScale1{static_cast<uint32_t>(n_)};
    layout::VectorLayout layoutScale2{static_cast<uint32_t>(n2_)};
    layout::RowMajor layoutD1{static_cast<uint32_t>(maxOutputSize_), static_cast<uint32_t>(k2_)};
    layout::RowMajor layoutD2{static_cast<uint32_t>(m_ * topK_), static_cast<uint32_t>(n2_)};

    GemmCoord problemShape{static_cast<uint32_t>(m_), static_cast<uint32_t>(n_), static_cast<uint32_t>(k_)};

    uint32_t epilogueCoreNum;
    uint32_t epilogueGranularity;
    if constexpr (isW4A8) {
        epilogueCoreNum = static_cast<uint32_t>(aivNum_);
        epilogueGranularity = 0U;
    } else if constexpr (isInt8) {
        if constexpr (IS_A2_) {
            epilogueCoreNum = static_cast<uint32_t>(aivNum_ / 2);
            epilogueGranularity =
                (expertPerRank_ > 1) ? static_cast<uint32_t>(expertPerRank_ - 1) : 1U;
        } else {
            epilogueCoreNum = static_cast<uint32_t>(aivNum_);
            epilogueGranularity = (expertPerRank_ > 4) ? static_cast<uint32_t>(expertPerRank_ - 3) :
                                  (expertPerRank_ > 1) ? static_cast<uint32_t>(expertPerRank_ - 1) :
                                                         1U;
        }
    } else {
        epilogueCoreNum = static_cast<uint32_t>(aivNum_);
        epilogueGranularity =
            (expertPerRank_ > 2) ? static_cast<uint32_t>(expertPerRank_ - 2) : 1U;
    }

    if constexpr (IS_A2_) {
        // A2/A3 share epilogues; only the outer communication kernels remain platform-specific.
        using EpilogueDispatchPolicy1 = std::conditional_t<
            isW4A8, Epilogue::EpilogueAtlasA2W4A8PostPerTokenDequantSwigluQuant<ubStages>,
            Epilogue::EpilogueAtlasA2PerTokenDequantSwigluQuant<ubStages>>;
        using EpilogueDispatchPolicy2 = std::conditional_t<
            isW4A8, Epilogue::EpilogueAtlasA2W4A8PostPerTokenDequantV2<ubStages>,
            Epilogue::EpilogueAtlasA2PerTokenDequantV2<ubStages>>;
        using BlockEpilogue1 =
            Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy1, CType, PerTokenScaleType, D1Type,
                                            TileElemWiseMuls, TileCopy1>;
        using BlockEpilogue2 = std::conditional_t<
            isW4A8,
            Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy2, CType, PerTokenScaleType, D2Type,
                                            TileCopy2, PlatformTag>,
            Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy2, CType, PerTokenScaleType, D2Type,
                                            TileCopy2, PerTokenV2Tag>>;

        using MatmulKernel =
            Gemm::Kernel::MegaMoeKernelA2<isInt8, isW4A8, BlockMmad, BlockScheduler,
                                          ElementGroupList, BlockEpilogue1, BlockEpilogue2>;
        this->template LaunchKernel<MatmulKernel>(
            problemShape, layoutA1, layoutA2, layoutB1, layoutB2, layoutScale1, layoutScale2,
            layoutD1, layoutD2, epilogueCoreNum, epilogueGranularity);
    } else {
        // A3 supports both combine implementations. BlockEpilogue2 is CombineV2
        // on both platforms; BlockEpilogue3 is the A3-only CombineV1 path.
        using EpilogueDispatchPolicy1 =
            std::conditional_t<isW4A8, Epilogue::EpilogueAtlasA2W4A8PostPerTokenDequantSwigluQuant<ubStages>,
                               Epilogue::EpilogueAtlasA2PerTokenDequantSwigluQuant<ubStages>>;
        using EpilogueDispatchPolicy2 =
            std::conditional_t<isW4A8, Epilogue::EpilogueAtlasA2W4A8PostPerTokenDequantV2<ubStages>,
                               Epilogue::EpilogueAtlasA2PerTokenDequantV2<ubStages>>;
        using EpilogueDispatchPolicy3 = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
        using BlockEpilogue1 =
            Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy1, CType, PerTokenScaleType, D1Type,
                                            TileElemWiseMuls, TileCopy1>;
        using BlockEpilogue2 = std::conditional_t<
            isW4A8,
            Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy2, CType, PerTokenScaleType, D2Type,
                                            TileCopy2, PlatformTag>,
            Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy2, CType, PerTokenScaleType, D2Type,
                                            TileCopy2, PerTokenV2Tag>>;
        using BlockEpilogue3 =
            Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy3, CType, PerTokenScaleType, D2Type,
                                            TileCopy2>;
        using MatmulKernel =
            Gemm::Kernel::MegaMoeKernel<BlockMmad, BlockScheduler, ElementGroupList, BlockEpilogue1,
                                        BlockEpilogue2, BlockEpilogue3>;

        this->template LaunchKernel<MatmulKernel>(
            problemShape, layoutA1, layoutA2, layoutB1, layoutB2, layoutScale1, layoutScale2,
            layoutD1, layoutD2, epilogueCoreNum, epilogueGranularity);
    }
}

} // namespace MegaMoeImpl
#endif // MEGA_MOE_H
