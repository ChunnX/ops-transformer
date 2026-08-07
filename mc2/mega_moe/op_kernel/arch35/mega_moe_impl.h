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
 * \file mega_moe_impl.h
 * \brief
 */

#ifndef MEGA_MOE_IMPL_H
#define MEGA_MOE_IMPL_H
#include "kernel_operator.h"

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_base.h"

#include "tensor_api/tensor.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_swizzle.h"
#include "blaze/gemm/block/block_mmad_mx_fp8fp4.h"
#include "blaze/prologue/block_prologue_mx_fp8fp4.h"

#include "mega_moe_impl_base.h"
#include "aiv_comm/mega_moe_token_combine.h"
#include "mega_moe_group_matmul_compute.h"
#include "mega_moe_group_matmul_epilogue.h"
namespace MegaMoeImpl {
namespace Detail {

// 原型：AivPrologueA8W4。为一个逻辑 block 的 AIV0 路径展开 A8W4 权重。
template <typename Policy, typename BlockPrologue, typename Scheduler, typename TensorB, typename Config>
__aicore__ inline void AivPrologueA8W4(BlockPrologue &blockPrologue, Scheduler &scheduler, TensorB &gmB,
                                       const Config &config, uint32_t startLoopIdx, uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        auto mL1Size = Get<M_VALUE>(actualShape);
        auto nL1Size = Get<N_VALUE>(actualShape);

        if constexpr (Policy::IS_GMM1) {
            for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                auto nOffset = nLoc + weightBlock * config.outputN;
                blockPrologue(gmB, mL1Size, config.k, nL1Size, nOffset, config.n, config.l1Params.kL1);
            }
        } else {
            blockPrologue(gmB, mL1Size, config.k, nL1Size, nLoc, config.n, config.l1Params.kL1);
        }
    }
}

__aicore__ inline void WaitGmm1TileStatus(const Params &params, uint32_t expertIdx, uint32_t loopIdx)
{
    __gm__ int32_t *statusAddr = reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.gmm1TileStatusPtr) +
                                 (static_cast<int64_t>(expertIdx) * params.tilingData->maxTilesPerExpert + loopIdx) *
                                     INT_CACHELINE;
    int32_t roundTag = static_cast<int32_t>(expertIdx + 1);
    while (AscendC::ReadGmByPassDCache(statusAddr) != roundTag) {
        int64_t startCycle = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - startCycle < 100) {
        }
    }
}

__aicore__ inline void NotifyGmm1TileStatus(const Params &params, uint32_t expertIdx, uint32_t loopIdx)
{
    __gm__ int32_t *statusAddr = reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.gmm1TileStatusPtr) +
                                 (static_cast<int64_t>(expertIdx) * params.tilingData->maxTilesPerExpert + loopIdx) *
                                     INT_CACHELINE;
    AscendC::WriteGmByPassDCache(statusAddr, static_cast<int32_t>(expertIdx + 1));
}

// 控制通用 GMM1/SwiGLU tile 流水。跨阶段等待保留在这里；无依赖的 SwiGLU 算子返回后，
// ComputeSwigluTileGeneric 立即发布已完成的 tile，以保持原有 GMM1/GMM2 重叠关系。
template <typename Policy, typename BlockMmad, typename ElementC, typename MakeLayoutC, bool TopkWeightsPrefetch,
          bool IsShared, bool IsGmm1Interleaved, typename WorkSet, typename ExtraArgs>
__aicore__ inline void Gmm1SwigluPipelineGeneric(WorkSet &workSet, uint32_t startLoopIdx,
                                                        uint32_t tileNum, ExtraArgs &args)
{
    const auto &config = workSet.config;
    if constexpr (g_coreType == AscendC::AIC) {
        BlockMmad blockMmad;
        bool enableL0CPingPong = false;
        typename BlockMmad::BlockShape l0TileShape{config.tileM, L1_TILE_N, L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{config.m, config.n, config.k, 0};
        blockMmad.Init(matmulShape, l0TileShape, config.l1Params, false, enableL0CPingPong);

        uint32_t ubBufSize =
            config.tileM == L1_TILE_M_128 ? MAX_SINGLE_MN_ALIGN32_NUM_128 : MAX_SINGLE_MN_ALIGN32_NUM_256;
        int64_t ubOffsetFirst = 0;
        int64_t ubOffsetSecond = static_cast<int64_t>(ubBufSize) * sizeof(ElementC);
        auto l0cOutUbFirst =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetFirst), workSet.layouts.c);
        auto l0cOutUbSecond =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetSecond), workSet.layouts.c);
        uint32_t lastWaveWaited = static_cast<uint32_t>(-1);

        for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
            auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
            auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
            uint32_t mLoc = Get<M_VALUE>(blockCoord);
            uint32_t nLoc = Get<N_VALUE>(blockCoord);
            uint32_t kLoc = Get<K_VALUE>(blockCoord);
            auto gmBlockA = workSet.gmA.Slice(
                Te::MakeCoord(mLoc, kLoc),
                Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
            auto gmBlockScaleA = workSet.gmScaleA.Slice(
                Te::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
                Te::MakeShape(Get<M_VALUE>(actualShape), CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM)));
            uint32_t waveIdx = mLoc / config.tileM;
            if (waveIdx != lastWaveWaited) {
                WaitForUpstreamReady<Policy, IsShared>(workSet.gmmAddrInfo, config, mLoc);
                lastWaveWaited = waveIdx;
            }
            if constexpr (!TopkWeightsPrefetch) {
                if constexpr (IsGmm1Interleaved) {
                    if (args.vecSetSyncCom >= 2) {
                        WaitForVector(args.pingpongIdx);
                    }
                } else if (args.vecSetSyncCom != 0) {
                    WaitForVector();
                }
            }

            ComputeGmm1TileGeneric<BlockMmad, ElementC, TopkWeightsPrefetch, IsGmm1Interleaved>(
                blockMmad, config, workSet.gmB, workSet.gmScaleB, workSet.gmBias,
                workSet.gmmAddrInfo.gmm1OutGlobal, workSet.layouts.c, actualShape, mLoc, nLoc, kLoc, gmBlockA,
                gmBlockScaleA, l0cOutUbFirst, l0cOutUbSecond, args.pingpongIdx);

            if constexpr (TopkWeightsPrefetch) {
                AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
                NotifyGmm1TileStatus(workSet.params, args.expertIdx, loopIdx);
            } else {
                if constexpr (IsGmm1Interleaved) {
                    NotifyVector(args.pingpongIdx);
                    args.vecSetSyncCom++;
                    args.pingpongIdx = 1U - args.pingpongIdx;
                } else {
                    NotifyVector();
                    args.vecSetSyncCom = 1;
                }
            }
        }
    } else {
        for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
            auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
            auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
            uint32_t mLoc = Get<M_VALUE>(blockCoord);
            uint32_t nLoc = Get<N_VALUE>(blockCoord);
            if constexpr (TopkWeightsPrefetch) {
                WaitGmm1TileStatus(workSet.params, args.expertIdx, loopIdx);
            } else if constexpr (IsGmm1Interleaved) {
                WaitForCube(args.pingpongIdx);
            } else {
                WaitForCube();
            }

            ComputeSwigluTileGeneric<MakeLayoutC, ElementC, TopkWeightsPrefetch, IsGmm1Interleaved>(
                config, workSet.gmmAddrInfo.gmm1OutGlobal, workSet.layouts.c, args.swigluQuantOp,
                args.expertBeforeCnt, actualShape, mLoc, nLoc, args.pingpongIdx);

            if constexpr (!TopkWeightsPrefetch) {
                if constexpr (IsGmm1Interleaved) {
                    NotifyCube(args.pingpongIdx);
                    args.pingpongIdx = 1U - args.pingpongIdx;
                } else {
                    NotifyCube();
                }
            }
        }
    }
}

// 原型：GroupMatmulExecGeneric。将一个通用 GMM 阶段分派到对应的 AIC 或 AIV 实现。
template <typename Policy, uint8_t CombineQuantMode, typename BlockMmad, typename ElementC, typename MakeLayoutC,
          bool TopkWeightsPrefetch, bool IsLayered = false, bool IsShared = false,
          bool IsGmm1Interleaved = false, typename WorkSet, typename ExtraArgs>
__aicore__ inline void GroupMatmulExecGeneric(WorkSet &workSet, uint32_t startLoopIdx, uint32_t tileNum,
                                              ExtraArgs &args)
{
    if constexpr (Policy::IS_GMM1) {
        Gmm1SwigluPipelineGeneric<Policy, BlockMmad, ElementC, MakeLayoutC, TopkWeightsPrefetch, IsShared,
                                         IsGmm1Interleaved>(workSet, startLoopIdx, tileNum, args);
    } else if constexpr (g_coreType == AscendC::AIC) {
        BlockMmad blockMmad;
        bool enableL0CPingPong = false;
        typename BlockMmad::BlockShape l0TileShape{workSet.config.tileM, L1_TILE_N, L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{workSet.config.m, workSet.config.n, workSet.config.k, 0};
        blockMmad.Init(matmulShape, l0TileShape, workSet.config.l1Params, false, enableL0CPingPong);

        AicComputeGmm2Generic<CombineQuantMode, Policy, BlockMmad, ElementC, IsShared, TopkWeightsPrefetch, IsLayered>(
            blockMmad, workSet, startLoopIdx, tileNum, args);
    }
}

// 控制 A8W4 GMM1/SwiGLU tile 流水。AIV0 权重前处理与 AIC MMAD 保持 Blaze 内部握手；
// 阶段边界同步及下游完成状态发布由这里负责。
template <typename Policy, typename BlockMmad, typename BlockPrologue, typename ElementC, typename MakeLayoutC,
          bool IsShared, typename WorkSet, typename Config, bool TopkWeightsPrefetch, typename ExtraArgs>
__aicore__ inline void Gmm1SwigluPipelineA8W4(WorkSet &workSet, const Params &params,
                                                     const GMMAddrInfo &gmmAddrInfo, const Config &config,
                                                     uint32_t startLoopIdx, uint32_t tileNum,
                                                     int32_t &gmTileSequence, ExtraArgs &args)
{
    if constexpr (g_coreType == AscendC::AIC) {
        BlockMmad blockMmad{};
        typename BlockMmad::BlockShape l0TileShape{config.tileM, L1_TILE_N, L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{config.m, config.outputN, config.k, 0};
        blockMmad.Init(matmulShape, l0TileShape, config.l1Params);
        uint32_t lastWaveWaited = static_cast<uint32_t>(-1);

        for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
            auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
            auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
            uint32_t mLoc = Get<M_VALUE>(blockCoord);
            uint32_t nLoc = Get<N_VALUE>(blockCoord);
            uint32_t waveIdx = mLoc / config.tileM;
            if (waveIdx != lastWaveWaited) {
                WaitForUpstreamReady<Policy, IsShared>(gmmAddrInfo, config, mLoc);
                lastWaveWaited = waveIdx;
            }

            ComputeGmm1TileA8W4(blockMmad, workSet.gmA, workSet.gmScaleA, workSet.gmScaleB,
                                workSet.l0cOutGm, config, actualShape, mLoc, nLoc);

            if constexpr (TopkWeightsPrefetch) {
                AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
                NotifyGmm1TileStatus(params, args.expertIdx, loopIdx);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
                AscendC::WriteGmByPassDCache(gmmAddrInfo.gmmToEpilogueFlag, ++gmTileSequence);
            }
        }
    } else if (GetSubBlockIdx() == 0) {
        BlockPrologue blockPrologue;
        AivPrologueA8W4<Policy>(blockPrologue, workSet.scheduler, workSet.gmB, config, startLoopIdx, tileNum);
    } else {
        for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
            auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
            auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
            uint32_t mLoc = Get<M_VALUE>(blockCoord);
            uint32_t nLoc = Get<N_VALUE>(blockCoord);
            int32_t expectedReadySequence = gmTileSequence;
            if constexpr (TopkWeightsPrefetch) {
                WaitGmm1TileStatus(params, args.expertIdx, loopIdx);
            } else {
                expectedReadySequence = gmTileSequence + 1;
                while (AscendC::ReadGmByPassDCache(gmmAddrInfo.gmmToEpilogueFlag) < expectedReadySequence) {
                    int64_t startCycle = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - startCycle < 100) {
                    }
                }
            }

            ComputeSwigluTileA8W4<ElementC, MakeLayoutC, decltype(workSet.l0cOutGm),
                                  decltype(args.swigluQuantOp), Config, TopkWeightsPrefetch>(
                args.swigluQuantOp, workSet.l0cOutGm, config, actualShape, mLoc, nLoc, args.expertBeforeCnt);
            if constexpr (!TopkWeightsPrefetch) {
                gmTileSequence = expectedReadySequence;
            }
        }
    }
}

// 原型：GroupMatmulExecA8W4。将一个 A8W4 GMM 阶段分派到 block 的 AIC 和两个 AIV。
template <uint8_t CombineQuantMode, typename Policy, typename BlockMmad, typename BlockPrologue, typename ElementC,
          typename MakeLayoutC, bool IsShared, bool IsLayered = false, typename WorkSet, typename Config,
          bool TopkWeightsPrefetch, typename ExtraArgs>
__aicore__ inline void GroupMatmulExecA8W4(WorkSet &workSet, const Params &params, const GMMAddrInfo &gmmAddrInfo,
                                           const Config &config, uint32_t startLoopIdx, uint32_t tileNum,
                                           int32_t &gmTileSequence, ExtraArgs &args)
{
    if constexpr (Policy::IS_GMM1) {
        Gmm1SwigluPipelineA8W4<Policy, BlockMmad, BlockPrologue, ElementC, MakeLayoutC, IsShared,
                                      WorkSet, Config, TopkWeightsPrefetch>(
            workSet, params, gmmAddrInfo, config, startLoopIdx, tileNum, gmTileSequence, args);
    } else if constexpr (g_coreType == AscendC::AIC) {
        BlockMmad blockMmad{};
        typename BlockMmad::BlockShape l0TileShape{config.tileM, L1_TILE_N, L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{config.m, config.outputN, config.k, 0};
        blockMmad.Init(matmulShape, l0TileShape, config.l1Params);
        AicComputeGmm2A8W4<CombineQuantMode, Policy, IsShared, IsLayered, BlockMmad, decltype(workSet.scheduler),
                           decltype(workSet.gmA), decltype(workSet.gmScaleA), decltype(workSet.gmScaleB),
                           decltype(workSet.l0cOutGm), Config, TopkWeightsPrefetch>(
            blockMmad, workSet.scheduler, workSet.gmA, workSet.gmScaleA, workSet.gmScaleB, workSet.l0cOutGm,
            gmmAddrInfo, gmTileSequence, config, startLoopIdx, tileNum);
    } else {
        if (GetSubBlockIdx() == 0) {
            BlockPrologue blockPrologue;
            AivPrologueA8W4<Policy>(blockPrologue, workSet.scheduler, workSet.gmB, config, startLoopIdx, tileNum);
        } else {
            if constexpr (CombineQuantMode == COMBINE_NO_QUANT && !IsShared && !IsLayered) {
                AivGmm2PostA8W4<ElementC, MakeLayoutC>(workSet.scheduler, workSet.l0cOutGm, params, gmmAddrInfo,
                                                       gmTileSequence, args.groupCnt, config, startLoopIdx, tileNum);
            }
        }
    }
}

} // namespace Detail
} // namespace MegaMoeImpl

namespace MegaMoeImpl {

// =================================================================================================
// ComputeCoreGrouping：计算当前 core 所属的 group 及其在 group 内的位置
// =================================================================================================
// 将 totalCores 个 core 均匀分配到 numGroups 个 group 中，余数分配给前 remainder 个 group。
__aicore__ inline void ComputeCoreGrouping(uint32_t coreId, uint32_t numGroups, uint32_t totalCores, uint32_t &myGroup,
                                           uint32_t &myIdxInGrp, uint32_t &myGrpSize)
{
    uint32_t baseSize = totalCores / numGroups;     // 每个 group 的基础 core 数
    uint32_t remainder = totalCores % numGroups;    // 余数，前 remainder 个 group 多分配 1 个 core
    uint32_t boundary = remainder * (baseSize + 1); // 前 remainder 个 group 占用的 core 总数

    // 判断当前 core 是否在前 remainder 个 group 中（这些 group 有 baseSize+1 个 core）
    if (coreId < boundary) {
        myGroup = coreId / (baseSize + 1);    // 所属 group 索引
        myIdxInGrp = coreId % (baseSize + 1); // 在 group 内的索引
        myGrpSize = baseSize + 1;             // 当前 group 的 core 数
    } else {
        // 当前 core 在后面的 group 中（这些 group 只有 baseSize 个 core）
        uint32_t adjusted = coreId - boundary;     // 减去前 remainder 个 group 占用的 core 数
        myGroup = remainder + adjusted / baseSize; // 所属 group 索引 = remainder + 偏移
        myIdxInGrp = adjusted % baseSize;          // 在 group 内的索引
        myGrpSize = baseSize;                      // 当前 group 的 core 数
    }
}

// ==================================================================================
// 统一配置结构体 — 通过 IsA8W4 模板参数区分 A8W8/A4W4 和 A8W4 两条路径的配置
// ==================================================================================
namespace Detail {
struct Gmm1Policy {
    static constexpr bool IS_GMM1 = true;
};

struct Gmm2Policy {
    static constexpr bool IS_GMM1 = false;
};

// BlockMmadSelector — 通过偏特化处理 A8W8/A4W4 和 A8W4 的 BlockMmad 签名差异
template <bool IsA8W4, typename C>
struct BlockMmadSelector;

template <typename C>
struct BlockMmadSelector<false, C> {
    using type =
        Blaze::Gemm::Block::BlockMmad<typename C::DispatchPolicy, typename C::ElementAType, typename C::LayoutA,
                                      typename C::ElementBType, typename C::LayoutB, typename C::ElementCType,
                                      typename C::LayoutC, typename C::BiasType, typename C::LayoutBias>;
};

template <typename C>
struct BlockMmadSelector<true, C> {
    using type =
        Blaze::Gemm::Block::BlockMmad<typename C::DispatchPolicy,
                                      AscendC::Std::tuple<typename C::ElementAType, typename C::ElementMxScaleAType>,
                                      AscendC::Std::tuple<typename C::MakeLayoutA, typename C::MakeLayoutScaleA>,
                                      AscendC::Std::tuple<typename C::ElementBType, typename C::ElementMxScaleBType>,
                                      AscendC::Std::tuple<typename C::MakeLayoutB, typename C::MakeLayoutScaleB>,
                                      typename C::ElementCType, typename C::MakeLayoutC, void, void>;
};

// ==================================================================================
// 统一 Config — 通过 IsA8W4 模板参数区分 A8W8/A4W4 和 A8W4
// 含公共与差异类型别名，BlockMmad 通过 trait 选择
// ==================================================================================
template <bool IsA8W4, typename Policy, uint8_t CombineQuantMode, typename ElementA, typename ElementB,
          typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB, bool IsWeightNZ = false,
          uint32_t Gmm1TileM = L1_TILE_M_256, bool TopkWeightsPrefetch = false, bool IsShared = false,
          bool IsLayered = false, bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false>
struct Config {
    static constexpr bool IS_SHARED = IsShared;
    static constexpr bool TOPK_WEIGHTS_PREFETCH = TopkWeightsPrefetch;
    using ElementAType = ElementA;
    using ElementBType = ElementB;
    using ElementCType = ElementC;
    using ElementMxScaleAType = ElementMxScaleA;
    using ElementMxScaleBType = ElementMxScaleB;

    static constexpr uint32_t C0_SIZE_A = AuxGetC0Size<ElementA>();
    static constexpr uint32_t C0_SIZE_C = AuxGetC0Size<ElementC>();
    static constexpr uint32_t C0_SIZE_SCALE = 2U;

    static constexpr uint32_t C0_SIZE_B = IsA8W4 ? 32U : AuxGetC0Size<ElementB>();

    using LayoutA = Te::NDExtLayoutPtn;
    using LayoutC = Te::NDExtLayoutPtn;
    using LayoutScaleA = Te::ScaleANDLayoutPtn;
    using LayoutScaleB = Te::ScaleBDNLayoutPtn;

    using BiasType = float;
    using LayoutBias = Te::NDExtLayoutPtn;
    using DispatchPolicy =
        Std::conditional_t<IsA8W4, Blaze::Gemm::MatmulMxFp8Fp4DynamicKL1TailResplit, Blaze::Gemm::MatmulWithScaleMx<>>;
    using LayoutB = Std::conditional_t<IsA8W4, Te::ZNLayoutPtn,
                                       Std::conditional_t<IsWeightNZ, Te::ZNLayoutPtn, Te::DNExtLayoutPtn>>;

    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<C0_SIZE_A>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<C0_SIZE_B>>;
    using MakeLayoutScaleA = Te::FrameLayoutFormat<LayoutScaleA, Std::Int<C0_SIZE_SCALE>>;
    using MakeLayoutScaleB = Te::FrameLayoutFormat<LayoutScaleB, Std::Int<C0_SIZE_SCALE>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<C0_SIZE_C>>;

    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutAType = decltype(MakeLayoutA{}(uint32_t{}, uint32_t{}));
    using LayoutBType = decltype(MakeLayoutB{}(uint32_t{}, uint32_t{}));
    using LayoutScaleAType = decltype(MakeLayoutScaleA{}(uint32_t{}, uint32_t{}));
    using LayoutScaleBType = decltype(MakeLayoutScaleB{}(uint32_t{}, uint32_t{}));
    using LayoutCType = decltype(MakeLayoutC{}(uint32_t{}, uint32_t{}));
    using LayoutBiasType = decltype(Te::MakeFrameLayout<LayoutBias>(uint32_t{}, uint32_t{}));

    using BlockMmad = typename BlockMmadSelector<IsA8W4, Config>::type;

    // BlockPrologue（仅 A8W4 使用；A8W8/A4W4 路径用 void 占位）
    using BlockPrologue =
        Std::conditional_t<IsA8W4, Blaze::Gemm::Prologue::BlockPrologue<DispatchPolicy, ElementA, ElementB>, void>;

    struct ProblemConfig {
        static constexpr bool SOURCE_GMM1_INTERLEAVED = IsGmm1Interleaved;
        static constexpr bool IS_WAVE_FLAG_GRAINED = IsWaveFlagGrained;

        static __aicore__ inline typename BlockMmad::L1Params DefaultL1Params()
        {
            if constexpr (IsA8W4) {
                return typename BlockMmad::L1Params{.kL1 = L1_TILE_K, .scaleKL1 = 4096};
            } else {
                return typename BlockMmad::L1Params{
                    .kL1 = L1_TILE_K, .scaleKL1 = L1_TILE_K * SCALE_K_L1_RATE, .l1BufNum = 2};
            }
        }

        uint32_t m = 0;
        uint32_t n = 0;
        uint32_t k = 0;
        uint32_t outputN = 0;
        uint32_t schedulerN = 0;
        uint32_t blockNum = 0;
        uint32_t blockIdx = 0;
        uint32_t scaleK = 0;
        uint32_t tileM = 0;
        uint32_t swigluTileM = L1_TILE_M_256;
        typename BlockMmad::L1Params l1Params = DefaultL1Params();
    };

    struct LayoutBundle {
        LayoutAType a;
        LayoutBType b;
        LayoutScaleAType scaleA;
        LayoutScaleBType scaleB;
        LayoutCType c;
        LayoutBiasType bias; // A8W8/A4W4 路径用，A8W4 不使用
    };

    __aicore__ static inline ProblemConfig BuildProblemConfig(const ProblemShape &problemShape,
                                                              const BlockJobContext &blockJob)
    {
        ProblemConfig config;
        config.m = Get<M_VALUE>(problemShape);
        if constexpr (Policy::IS_GMM1) {
            config.n = Get<N_VALUE>(problemShape);
            config.k = Get<K_VALUE>(problemShape);
        } else {
            config.n = Get<K_VALUE>(problemShape);
            config.k = Get<N_VALUE>(problemShape) / SWIGLU_N_HALF;
        }
        config.outputN = Policy::IS_GMM1 ? config.n / SWIGLU_N_HALF : config.n;
        config.schedulerN = Policy::IS_GMM1 && IsGmm1Interleaved ? config.n : config.outputN;
        config.blockNum = blockJob.totalJobs;
        config.blockIdx = blockJob.jobIndex;
        config.scaleK = CeilDiv(config.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        if constexpr (Policy::IS_GMM1) {
            config.tileM = Gmm1TileM;
        } else {
            config.tileM = L1_TILE_M_256;
        }
        config.swigluTileM = TopkWeightsPrefetch ? L1_TILE_M_128 : Gmm1TileM;
        return config;
    }

    __aicore__ static inline LayoutBundle BuildLayouts(const ProblemConfig &config)
    {
        LayoutBundle layouts;
        layouts.a = MakeLayoutA{}(config.m, config.k);
        layouts.b = MakeLayoutB{}(config.k, config.n);
        layouts.scaleA = MakeLayoutScaleA{}(config.m, config.scaleK);
        layouts.scaleB = MakeLayoutScaleB{}(config.scaleK, config.n);
        if constexpr (IsA8W4) {
            layouts.c = MakeLayoutC{}(config.m, config.n);
        } else {
            layouts.bias = Te::MakeFrameLayout<LayoutBias>(1U, config.n);
            if constexpr (Policy::IS_GMM1) {
                if constexpr (TopkWeightsPrefetch) {
                    layouts.c = MakeLayoutC{}(config.m, config.n);
                } else {
                    layouts.c = MakeLayoutC{}(Gmm1TileM, L1_TILE_N);
                }
            } else {
                layouts.c = MakeLayoutC{}(config.m, config.n);
            }
        }
        return layouts;
    }
};

template <typename SwigluQuantOp>
struct Gmm1ArgsGeneric {
    SwigluQuantOp &swigluQuantOp;
    int32_t &vecSetSyncCom;
    uint32_t expertBeforeCnt{0};
    uint32_t expertIdx{0};
    uint16_t &pingpongIdx;
};

struct Gmm2ArgsGeneric {
    int32_t &vecSetSyncCom2;
    uint32_t groupCnt;
    uint16_t &pingpongIdx;
};

template <typename Scheduler, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB,
          typename TensorBias, typename Config, typename LayoutBundle>
struct WorkSetGeneric {
    Scheduler &scheduler;
    TensorA &gmA;
    TensorB &gmB;
    TensorScaleA &gmScaleA;
    TensorScaleB &gmScaleB;
    TensorBias &gmBias;
    const GMMAddrInfo &gmmAddrInfo;
    const Params &params;
    const Config &config;
    const LayoutBundle &layouts;
};

template <typename Policy, uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB, bool IsWeightNZ = false, bool IsLayered = false,
          uint32_t Gmm1TileM = L1_TILE_M_256, bool TopkWeightsPrefetch = false, bool IsShared = false,
          bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false, typename ExtraArgs>
__aicore__ inline void GroupMatmulImplGeneric(const Params &params,
                                              const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                              const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
                                              const BlockJobContext &blockJob, ExtraArgs &args)
{
    using Config = Config<false, Policy, CombineQuantMode, ElementA, ElementB, ElementC, ElementMxScaleA,
                          ElementMxScaleB, IsWeightNZ, Gmm1TileM, TopkWeightsPrefetch, IsShared, IsLayered,
                          IsGmm1Interleaved, IsWaveFlagGrained>;
    auto config = Config::BuildProblemConfig(problemShape, blockJob);

    BlockScheduler scheduler(
        {config.m, config.schedulerN, config.k},
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();

    if constexpr (Policy::IS_GMM1) {
        if (GetSubBlockIdx() != 0) {
            startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
            return;
        }
        if constexpr (g_coreType == AIV) {
            args.swigluQuantOp.UpdateNextProblem({config.m, config.outputN, config.k, 0});
        }
    } else if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        if constexpr (g_coreType == AscendC::AIV) {
            startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
            return;
        }
    }
    // GMM2 量化模式：两分支均不匹配，直接往下执行

    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    auto layouts = Config::BuildLayouts(config);

    using BlockMmad = typename Config::BlockMmad;
    using BiasType = typename Config::BiasType;

    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA *>(gmmAddrInfo.aGlobal)), layouts.a);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB *>(gmmAddrInfo.bGlobal)), layouts.b);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA *>(gmmAddrInfo.aScaleGlobal)),
        layouts.scaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB *>(gmmAddrInfo.bScaleGlobal)),
        layouts.scaleB);
    auto gmBias =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType *>(0UL)), layouts.bias);

    using WorkSetType = WorkSetGeneric<BlockScheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA),
                                       decltype(gmScaleB), decltype(gmBias), decltype(config), decltype(layouts)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmmAddrInfo, params, config, layouts};

    using MakeLayoutC = typename Config::MakeLayoutC;
    GroupMatmulExecGeneric<Policy, CombineQuantMode, BlockMmad, ElementC, MakeLayoutC, TopkWeightsPrefetch, IsLayered,
                           IsShared, IsGmm1Interleaved>(workSet, startLoopIdx, tileNum, args);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}
} // namespace Detail
// =================================================================================================
template <typename ElementA, typename EpilogueElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, uint32_t Gmm1TileM = L1_TILE_M_256,
          uint32_t EpilogueTileM = Gmm1TileM, bool TopkWeightsPrefetch = false, bool IsShared = false,
          bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false>
__aicore__ inline void GroupMatmulSwigluQuant(
    BlockEpilogueSwigluMxQuant<EpilogueElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true, EpilogueTileM,
                               L1_TILE_N, TopkWeightsPrefetch, IsGmm1Interleaved, IsWaveFlagGrained> &epilogueOp,
    const Params &params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &vecSetSyncCom, const BlockJobContext &blockJob,
    uint32_t expertBeforeCnt, uint32_t expertIdx, uint16_t &pingpongIdx)
{
    using SwigluQuantOpType = std::remove_reference_t<decltype(epilogueOp)>;
    Detail::Gmm1ArgsGeneric<SwigluQuantOpType> args{
        epilogueOp, vecSetSyncCom, expertBeforeCnt, expertIdx, pingpongIdx};
    Detail::GroupMatmulImplGeneric<Detail::Gmm1Policy, COMBINE_NO_QUANT, ElementA, ElementB, ElementC, ElementMxScaleA,
                                   ElementMxScaleB, IsWeightNZ, false, Gmm1TileM, TopkWeightsPrefetch, IsShared,
                                   IsGmm1Interleaved, IsWaveFlagGrained>(
        params, problemShape, gmmAddrInfo, startBlockIdx, blockJob, args);
}

template <typename ElementA, typename EpilogueElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, uint32_t Gmm1TileM = L1_TILE_M_256,
          uint32_t EpilogueTileM = Gmm1TileM, bool TopkWeightsPrefetch = false, bool IsShared = false,
          bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false>
__aicore__ inline void GroupMatmulSwigluQuant(
    BlockEpilogueSwigluMxQuant<EpilogueElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true, EpilogueTileM,
                               L1_TILE_N, TopkWeightsPrefetch, IsGmm1Interleaved, IsWaveFlagGrained> &epilogueOp,
    const Params &params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &vecSetSyncCom, uint32_t expertBeforeCnt,
    uint32_t expertIdx, uint16_t &pingpongIdx)
{
    BlockJobContext blockJob{static_cast<uint32_t>(GetBlockIdx() / GetTaskRation()),
                             static_cast<uint32_t>(GetBlockNum())};
    GroupMatmulSwigluQuant<ElementA, EpilogueElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB,
                           IsWeightNZ, Gmm1TileM, EpilogueTileM, TopkWeightsPrefetch, IsShared,
                           IsGmm1Interleaved, IsWaveFlagGrained>(
        epilogueOp, params, problemShape, gmmAddrInfo, startBlockIdx, vecSetSyncCom, blockJob, expertBeforeCnt,
        expertIdx, pingpongIdx);
}

// =================================================================================================
// GroupMatmul2：GMM2 矩阵乘法，支持量化和非量化模式
// =================================================================================================
template <uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, bool IsLayered = false, uint32_t Gmm1TileM = L1_TILE_M_256,
          bool TopkWeightsPrefetch = false, bool IsShared = false, bool IsGmm1Interleaved = false,
          bool IsWaveFlagGrained = false>
__aicore__ inline void GroupMatmul2(const Params &params,
                                    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                    const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &vecSetSyncCom2,
                                    const BlockJobContext &blockJob, uint32_t groupCnt, uint16_t &pingpongIdx)
{
    Detail::Gmm2ArgsGeneric args{vecSetSyncCom2, groupCnt, pingpongIdx};
    Detail::GroupMatmulImplGeneric<Detail::Gmm2Policy, CombineQuantMode, ElementA, ElementB, ElementC, ElementMxScaleA,
                                   ElementMxScaleB, IsWeightNZ, IsLayered, Gmm1TileM, TopkWeightsPrefetch, IsShared,
                                   IsGmm1Interleaved, IsWaveFlagGrained>(
        params, problemShape, gmmAddrInfo, startBlockIdx, blockJob, args);
}

template <uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, bool IsLayered = false,
          uint32_t Gmm1TileM = L1_TILE_M_256, bool TopkWeightsPrefetch = false, bool IsShared = false,
          bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false>
__aicore__ inline void GroupMatmul2(const Params &params,
                                    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                    const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &vecSetSyncCom2,
                                    uint32_t groupCnt, uint16_t &pingpongIdx)
{
    BlockJobContext blockJob{static_cast<uint32_t>(GetBlockIdx() / GetTaskRation()),
                             static_cast<uint32_t>(GetBlockNum())};
    GroupMatmul2<CombineQuantMode, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, IsWeightNZ,
                 IsLayered, Gmm1TileM, TopkWeightsPrefetch, IsShared, IsGmm1Interleaved, IsWaveFlagGrained>(
        params, problemShape, gmmAddrInfo, startBlockIdx, vecSetSyncCom2, blockJob, groupCnt, pingpongIdx);
}

// ==================================================================================
// A8W4 执行路径 — 共享骨架，基于 Policy 分派 GMM1 / GMM2
// ==================================================================================
namespace Detail {

template <typename SwigluQuantOp>
struct Gmm1ArgsA8W4 {
    SwigluQuantOp &swigluQuantOp;
    uint32_t expertBeforeCnt{0};
    uint32_t expertIdx{0};
};

struct Gmm2ArgsA8W4 {
    uint32_t groupCnt;
    uint16_t &pingpongIdx;
};

template <typename Scheduler, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB,
          typename TensorC>
struct WorkSetA8W4 {
    Scheduler &scheduler;
    TensorA &gmA;
    TensorB &gmB;
    TensorScaleA &gmScaleA;
    TensorScaleB &gmScaleB;
    TensorC &l0cOutGm;
};

template <uint8_t CombineQuantMode, typename Policy, typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB, uint32_t Gmm1TileM = L1_TILE_M_256,
          bool TopkWeightsPrefetch = false, bool IsShared = false, bool IsLayered = false, typename ExtraArgs>
__aicore__ inline void GroupMatmulImplA8W4(const Params &params,
                                           const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                           const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
                                           int32_t &gmTileSequence, const BlockJobContext &blockJob, ExtraArgs &args)
{
    static_assert(std::is_same_v<ElementA, __fp8e4m3>, "Activation must be __fp8e4m3");
    static_assert(std::is_same_v<ElementB, __fp4e2m1x2>, "Weight must be __fp4e2m1x2");

    using Config = Config<true, Policy, 0, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, false,
                          Gmm1TileM, TopkWeightsPrefetch, IsShared, IsLayered>;
    auto config = Config::BuildProblemConfig(problemShape, blockJob);

    if constexpr (Policy::IS_GMM1 && g_coreType == AIV) {
        args.swigluQuantOp.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    }

    auto layouts = Config::BuildLayouts(config);
    using BlockMmad = typename Config::BlockMmad;
    using BlockPrologue = typename Config::BlockPrologue;
    using MakeLayoutC = typename Config::MakeLayoutC;

    auto l0cOutGm = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC *>(
                                       Policy::IS_GMM1 ? gmmAddrInfo.gmm1OutGlobal : gmmAddrInfo.gmm2OutGlobal)),
                                   layouts.c);
    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA *>(gmmAddrInfo.aGlobal)), layouts.a);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB *>(gmmAddrInfo.bGlobal)), layouts.b);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA *>(gmmAddrInfo.aScaleGlobal)),
        layouts.scaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB *>(gmmAddrInfo.bScaleGlobal)),
        layouts.scaleB);

    BlockScheduler scheduler(
        {config.m, config.outputN, config.k},
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();
    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    if (startLoopIdx >= tileNum) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }

    using WorkSetType = WorkSetA8W4<BlockScheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA),
                                    decltype(gmScaleB), decltype(l0cOutGm)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, l0cOutGm};
    GroupMatmulExecA8W4<CombineQuantMode, Policy, BlockMmad, BlockPrologue, ElementC, MakeLayoutC, IsShared, IsLayered,
                        WorkSetType, decltype(config), TopkWeightsPrefetch>(
        workSet, params, gmmAddrInfo, config, startLoopIdx, tileNum, gmTileSequence, args);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

} // namespace Detail

// GroupMatmulSwigluQuantA8W4 — A8W4 prologue（W4→W8）+ GMM1 + SwiGLU + 量化
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          uint32_t Gmm1TileM = L1_TILE_M_256, uint32_t EpilogueTileM = Gmm1TileM, bool TopkWeightsPrefetch = false,
          bool IsShared = false>
__aicore__ inline void GroupMatmulSwigluQuantA8W4(
    BlockEpilogueSwigluMxQuant<ElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true, EpilogueTileM, L1_TILE_N,
                               TopkWeightsPrefetch> &swigluQuantOp,
    const Params &params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &gmTileSequence,
    const BlockJobContext &blockJob,
    uint32_t expertBeforeCnt, uint32_t expertIdx = 0)
{
    using SwigluQuantOpType = std::remove_reference_t<decltype(swigluQuantOp)>;
    Detail::Gmm1ArgsA8W4<SwigluQuantOpType> args{swigluQuantOp, expertBeforeCnt, expertIdx};
    Detail::GroupMatmulImplA8W4<COMBINE_NO_QUANT, Detail::Gmm1Policy, ElementA, ElementB, ElementC, ElementMxScaleA,
                                ElementMxScaleB, Gmm1TileM, TopkWeightsPrefetch, IsShared>(
        params, problemShape, gmmAddrInfo, startBlockIdx, gmTileSequence, blockJob, args);
}

template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          uint32_t Gmm1TileM = L1_TILE_M_256, uint32_t EpilogueTileM = Gmm1TileM, bool TopkWeightsPrefetch = false,
          bool IsShared = false>
__aicore__ inline void GroupMatmulSwigluQuantA8W4(
    BlockEpilogueSwigluMxQuant<ElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true, EpilogueTileM, L1_TILE_N,
                               TopkWeightsPrefetch> &swigluQuantOp,
    const Params &params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &gmTileSequence, uint32_t expertBeforeCnt,
    uint32_t expertIdx = 0)
{
    BlockJobContext blockJob{static_cast<uint32_t>(GetBlockIdx() / GetTaskRation()),
                             static_cast<uint32_t>(GetBlockNum())};
    GroupMatmulSwigluQuantA8W4<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, Gmm1TileM, EpilogueTileM,
                               TopkWeightsPrefetch, IsShared>(swigluQuantOp, params, problemShape, gmmAddrInfo,
                                                              startBlockIdx, gmTileSequence, blockJob, expertBeforeCnt,
                                                              expertIdx);
}

// GroupMatmul2CombineA8W4 — A8W4 prologue（W4→W8）+ GMM2 + Combine
template <uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, uint32_t Gmm1TileM = L1_TILE_M_256, bool TopkWeightsPrefetch = false,
          bool IsShared = false, bool IsLayered = false>
__aicore__ inline void GroupMatmul2CombineA8W4(const Params &params,
                                               const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                               const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
                                               int32_t &gmTileSequence, const BlockJobContext &blockJob,
                                               uint32_t groupCnt, uint16_t &pingpongIdx)
{
    Detail::Gmm2ArgsA8W4 args{groupCnt, pingpongIdx};
    Detail::GroupMatmulImplA8W4<CombineQuantMode, Detail::Gmm2Policy, ElementA, ElementB, ElementC, ElementMxScaleA,
                                ElementMxScaleB, Gmm1TileM, TopkWeightsPrefetch, IsShared, IsLayered>(
        params, problemShape, gmmAddrInfo, startBlockIdx, gmTileSequence, blockJob, args);
}

template <uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, uint32_t Gmm1TileM = L1_TILE_M_256, bool TopkWeightsPrefetch = false,
          bool IsShared = false, bool IsLayered = false>
__aicore__ inline void GroupMatmul2CombineA8W4(const Params &params,
                                               const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                               const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
                                               int32_t &gmTileSequence, uint32_t groupCnt, uint16_t &pingpongIdx)
{
    BlockJobContext blockJob{static_cast<uint32_t>(GetBlockIdx() / GetTaskRation()),
                             static_cast<uint32_t>(GetBlockNum())};
    GroupMatmul2CombineA8W4<CombineQuantMode, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, Gmm1TileM,
                            TopkWeightsPrefetch, IsShared, IsLayered>(
        params, problemShape, gmmAddrInfo, startBlockIdx, gmTileSequence, blockJob, groupCnt, pingpongIdx);
}

} // namespace MegaMoeImpl

#endif
