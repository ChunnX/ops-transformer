/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GMM_COMMON_H
#define MEGA_MOE_GMM_COMMON_H

#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "mega_moe_constants.h"
#include "mega_moe_types.h"
#include "mega_moe_utils.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "../blaze/gemm/block/block_scheduler_swizzle.h"
#include "../blaze/gemm/block/block_mmad_mx_fp8fp4.h"
#include "../blaze/prologue/block_prologue_mx_fp8fp4.h"

namespace MegaMoeImpl {

using namespace AscendC;

namespace GmmKernel {

constexpr uint32_t L1_TILE_K = 256U;
constexpr uint32_t SCALE_K_L1_RATE = 2U;

using BlockScheduler = typename Blaze::Gemm::Block::BlockSchedulerSwizzle<3, 1>; // 3: SwizzleOffset

// 根据数据类型路径选择对应的 BlockMmad 实现。
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

// 汇总 GMM1/GMM2 在不同量化路径下共用的类型、shape 和 layout 配置。
template <bool IsA8W4, uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB, bool IsWeightNZ = false,
          bool TopkWeightsPrefetch = false, bool IsShared = false,
          bool IsLayered = false, bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false>
struct Config {
    static constexpr bool IS_SHARED = IsShared;
    static constexpr bool IS_WEIGHT_NZ = IsWeightNZ;
    static constexpr bool TOPK_WEIGHTS_PREFETCH = TopkWeightsPrefetch;
    using ElementAType = ElementA;
    using ElementBType = ElementB;
    using ElementCType = ElementC;
    using ElementMxScaleAType = ElementMxScaleA;
    using ElementMxScaleBType = ElementMxScaleB;
    using BiasType = float;

    static constexpr uint32_t C0_SIZE_A = AuxGetC0Size<ElementA>();
    static constexpr uint32_t C0_SIZE_C = AuxGetC0Size<ElementC>();
    static constexpr uint32_t C0_SIZE_SCALE = 2U;
    static constexpr uint32_t C0_SIZE_B = IsA8W4 ? 32U : AuxGetC0Size<ElementB>();
    static constexpr uint32_t C0_SIZE_BIAS = AuxGetC0Size<BiasType>();

    using LayoutA = Te::NDExtLayoutPtn;
    using LayoutC = Te::NDExtLayoutPtn;
    using LayoutScaleA = Te::ScaleANDLayoutPtn;
    using LayoutScaleB = Te::ScaleBDNLayoutPtn;

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
    using MakeLayoutBias = Te::FrameLayoutFormat<LayoutBias, Std::Int<C0_SIZE_BIAS>>;

    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutAType = decltype(MakeLayoutA{}(uint32_t{}, uint32_t{}));
    using LayoutBType = decltype(MakeLayoutB{}(uint32_t{}, uint32_t{}));
    using LayoutScaleAType = decltype(MakeLayoutScaleA{}(uint32_t{}, uint32_t{}));
    using LayoutScaleBType = decltype(MakeLayoutScaleB{}(uint32_t{}, uint32_t{}));
    using LayoutCType = decltype(MakeLayoutC{}(uint32_t{}, uint32_t{}));
    using LayoutBiasType = decltype(MakeLayoutBias{}(uint32_t{}, uint32_t{}));

    using BlockMmad = typename BlockMmadSelector<IsA8W4, Config>::type;
    using BlockPrologue =
        Std::conditional_t<IsA8W4, Blaze::Gemm::Prologue::BlockPrologue<DispatchPolicy, ElementA, ElementB>, void>;

    static __aicore__ inline typename BlockMmad::L1Params DefaultL1Params()
    {
        if constexpr (IsA8W4) {
            return typename BlockMmad::L1Params{.kL1 = L1_TILE_K, .scaleKL1 = 4096};
        } else {
            return typename BlockMmad::L1Params{
                .kL1 = L1_TILE_K, .scaleKL1 = L1_TILE_K * SCALE_K_L1_RATE, .l1BufNum = 2};
        }
    }

    // 外层 scheduler 继续使用固定的 256 x 256 分块；该配置只控制 BlockMmad 内部基本块和 L1 搬运窗口。
    struct BlockMmadTilingConfig {
        uint32_t tileM;
        uint32_t tileN;
        typename BlockMmad::L1Params l1Params;
    };

    static __aicore__ inline BlockMmadTilingConfig MakeBaselineBlockMmadTilingConfig(uint32_t tileM)
    {
        return BlockMmadTilingConfig{tileM, L1_TILE_N, DefaultL1Params()};
    }

    static __aicore__ inline uint64_t CalcAdaptiveScaleKL1(uint32_t blockM, uint32_t blockN, uint64_t kL1,
                                                           uint32_t k)
    {
        constexpr uint64_t halfL1Size = AscendC::TOTAL_L1_SIZE / 2U;
        constexpr uint64_t scaleGroupK = 64U;
        constexpr uint64_t scaleBytesPerGroup = MXFP_MULTI_BASE_SIZE;
        constexpr uint64_t scaleTransferBytes = 64U * 1024U;
        constexpr uint64_t maxScaleFactor = 127U;
        constexpr uint64_t aElementsPerByte = Blaze::Gemm::IsFp4<ElementA>() ? 2U : 1U;
        constexpr uint64_t bElementsPerByte = Blaze::Gemm::IsFp4<ElementB>() ? 2U : 1U;

        const uint64_t blockM64 = blockM;
        const uint64_t blockN64 = blockN;
        const uint64_t dataBytes = Ops::Base::CeilDiv(blockM64 * kL1, aElementsPerByte) +
                                   Ops::Base::CeilDiv(blockN64 * kL1, bElementsPerByte);
        const uint64_t scaleKBytes = Ops::Base::CeilDiv(kL1, scaleGroupK) * scaleBytesPerGroup;
        const uint64_t scaleABytes = blockM64 * scaleKBytes;
        const uint64_t scaleBBytes = blockN64 * scaleKBytes;
        const uint64_t scaleBytes = scaleABytes + scaleBBytes;
        if (dataBytes >= halfL1Size || scaleBytes == 0U) {
            return 0U;
        }

        uint64_t scaleFactor = (halfL1Size - dataBytes) / scaleBytes;
        const uint64_t scaleFactorByA = scaleABytes == 0U ? maxScaleFactor : scaleTransferBytes / scaleABytes;
        const uint64_t scaleFactorByB = scaleBBytes == 0U ? maxScaleFactor : scaleTransferBytes / scaleBBytes;
        const uint64_t scaleFactorByK = Ops::Base::CeilDiv(static_cast<uint64_t>(k), kL1);
        scaleFactor = scaleFactor < scaleFactorByA ? scaleFactor : scaleFactorByA;
        scaleFactor = scaleFactor < scaleFactorByB ? scaleFactor : scaleFactorByB;
        scaleFactor = scaleFactor < scaleFactorByK ? scaleFactor : scaleFactorByK;
        scaleFactor = scaleFactor < maxScaleFactor ? scaleFactor : maxScaleFactor;
        return scaleFactor * kL1;
    }

    static __aicore__ inline BlockMmadTilingConfig MakeAdaptiveBlockMmadTilingConfig(uint32_t m, uint32_t k,
                                                                                     uint32_t tileM, uint64_t kL1)
    {
        constexpr uint32_t blockMAlign = 16U;
        const uint32_t currentTileM = m < tileM ? m : tileM;
        const uint32_t blockM = Ops::Base::CeilAlign(currentTileM, blockMAlign);
        const uint64_t scaleKL1 = CalcAdaptiveScaleKL1(blockM, L1_TILE_N, kL1, k);
        return BlockMmadTilingConfig{
            blockM,
            L1_TILE_N,
            typename BlockMmad::L1Params{.kL1 = kL1, .scaleKL1 = scaleKL1, .l1BufNum = 2}};
    }

    // BlockMmad 的双 buffer 分置在 L1 的前后两半，候选配置必须能在半片 L1 中容纳 A、B 及 MX scale。
    static __aicore__ inline bool CanFitDoubleBufferL1(const BlockMmadTilingConfig &tilingConfig)
    {
        constexpr uint64_t halfL1Size = AscendC::TOTAL_L1_SIZE / 2U;
        constexpr uint64_t scaleGroupK = 64U;
        constexpr uint64_t scaleBytesPerGroup = MXFP_MULTI_BASE_SIZE;
        constexpr uint64_t aElementsPerByte = Blaze::Gemm::IsFp4<ElementA>() ? 2U : 1U;
        constexpr uint64_t bElementsPerByte = Blaze::Gemm::IsFp4<ElementB>() ? 2U : 1U;
        const uint64_t blockM = tilingConfig.tileM;
        const uint64_t blockN = tilingConfig.tileN;
        const uint64_t dataBytes = Ops::Base::CeilDiv(blockM * tilingConfig.l1Params.kL1, aElementsPerByte) +
                                   Ops::Base::CeilDiv(blockN * tilingConfig.l1Params.kL1, bElementsPerByte);
        const uint64_t scaleBytes =
            (blockM + blockN) * Ops::Base::CeilDiv(tilingConfig.l1Params.scaleKL1, scaleGroupK) * scaleBytesPerGroup;
        return dataBytes + scaleBytes <= halfL1Size;
    }

    static __aicore__ inline bool IsValidAdaptiveBlockMmadTilingConfig(
        const BlockMmadTilingConfig &tilingConfig, uint32_t currentTileM)
    {
        constexpr uint32_t blockMAlign = 16U;
        const uint64_t kL1 = tilingConfig.l1Params.kL1;
        const uint64_t scaleKL1 = tilingConfig.l1Params.scaleKL1;
        return tilingConfig.tileM >= currentTileM && tilingConfig.tileM % blockMAlign == 0U && kL1 != 0U &&
               kL1 % L0_TILE_K == 0U && scaleKL1 >= kL1 && scaleKL1 % kL1 == 0U &&
               CanFitDoubleBufferL1(tilingConfig);
    }

    static __aicore__ inline bool IsBetterBlockMmadTilingConfig(const BlockMmadTilingConfig &candidate,
                                                                const BlockMmadTilingConfig &current, uint32_t k)
    {
        const uint64_t candidateKLoops =
            Ops::Base::CeilDiv(static_cast<uint64_t>(k), candidate.l1Params.kL1);
        const uint64_t currentKLoops = Ops::Base::CeilDiv(static_cast<uint64_t>(k), current.l1Params.kL1);
        if (candidateKLoops != currentKLoops) {
            return candidateKLoops < currentKLoops;
        }

        const uint64_t candidateScaleLoops =
            Ops::Base::CeilDiv(static_cast<uint64_t>(k), candidate.l1Params.scaleKL1);
        const uint64_t currentScaleLoops =
            Ops::Base::CeilDiv(static_cast<uint64_t>(k), current.l1Params.scaleKL1);
        return candidateScaleLoops < currentScaleLoops;
    }

    static __aicore__ inline BlockMmadTilingConfig SelectBlockMmadTilingConfig(uint32_t m, uint32_t n, uint32_t k,
                                                                               uint32_t tileM)
    {
        BlockMmadTilingConfig baselineConfig = MakeBaselineBlockMmadTilingConfig(tileM);
        if (m == 0U || n == 0U || k == 0U || m >= tileM) {
            return baselineConfig;
        }

        if constexpr (IsA8W4) {
            // A8W4 BlockMmad 与 AIV prologue 约定 kL1=0：按实际 tile M/N 自动选择 kaL1/kbL1。
            return BlockMmadTilingConfig{
                tileM,
                L1_TILE_N,
                typename BlockMmad::L1Params{.kL1 = 0U, .scaleKL1 = 4096U}};
        } else {
            if constexpr (g_coreType == AscendC::AIC) {
                BlockMmadTilingConfig bestConfig = baselineConfig;
                BlockMmadTilingConfig adaptive256KConfig =
                    MakeAdaptiveBlockMmadTilingConfig(m, k, tileM, L1_TILE_K);
                if (IsValidAdaptiveBlockMmadTilingConfig(adaptive256KConfig, m) &&
                    IsBetterBlockMmadTilingConfig(adaptive256KConfig, bestConfig, k)) {
                    bestConfig = adaptive256KConfig;
                }

                BlockMmadTilingConfig adaptive512KConfig =
                    MakeAdaptiveBlockMmadTilingConfig(m, k, tileM, L1_TILE_K * 2U);
                if (IsValidAdaptiveBlockMmadTilingConfig(adaptive512KConfig, m) &&
                    IsBetterBlockMmadTilingConfig(adaptive512KConfig, bestConfig, k)) {
                    bestConfig = adaptive512KConfig;
                }
                return bestConfig;
            }
        }
        return baselineConfig;
    }

    struct ProblemConfig {
        using KernelConfig = Config;
        static constexpr bool SOURCE_GMM1_INTERLEAVED = IsGmm1Interleaved;
        static constexpr bool IS_WAVE_FLAG_GRAINED = IsWaveFlagGrained;

        uint32_t m = 0;
        uint32_t n = 0;
        uint32_t k = 0;
        uint32_t outputN = 0;
        uint32_t schedulerN = 0;
        uint32_t blockNum = 0;
        uint32_t blockIdx = 0;
        uint32_t scaleK = 0;
        uint32_t tileM = 0;
        uint32_t activationTileM = L1_TILE_M_256;
        BlockMmadTilingConfig blockMmadTiling{};
    };

    struct LayoutBundle {
        LayoutAType a;
        LayoutBType b;
        LayoutScaleAType scaleA;
        LayoutScaleBType scaleB;
        LayoutCType c;
        LayoutBiasType bias;
    };

    __aicore__ static inline void FinalizeProblemConfig(ProblemConfig &config, const BlockJobContext &blockJob,
                                                        uint32_t gmm1TileM)
    {
        config.blockNum = blockJob.totalJobs;
        config.blockIdx = blockJob.jobIndex;
        config.scaleK = CeilDiv(config.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        config.activationTileM = TopkWeightsPrefetch ? L1_TILE_M_128 : gmm1TileM;
        config.blockMmadTiling =
            SelectBlockMmadTilingConfig(config.m, config.schedulerN, config.k, config.tileM);
    }

    __aicore__ static inline ProblemConfig BuildGmm1ProblemConfig(const ProblemShape &problemShape,
                                                                  const BlockJobContext &blockJob,
                                                                  uint32_t gmm1TileM)
    {
        ProblemConfig config;
        config.m = Get<M_VALUE>(problemShape);
        config.n = Get<N_VALUE>(problemShape);
        config.k = Get<K_VALUE>(problemShape);
        config.outputN = config.n / ACTIVATION_N_HALF;
        config.schedulerN = IsGmm1Interleaved ? config.n : config.outputN;
        config.tileM = gmm1TileM;
        FinalizeProblemConfig(config, blockJob, gmm1TileM);
        return config;
    }

    __aicore__ static inline ProblemConfig BuildGmm2ProblemConfig(const ProblemShape &problemShape,
                                                                  const BlockJobContext &blockJob,
                                                                  uint32_t gmm1TileM)
    {
        ProblemConfig config;
        config.m = Get<M_VALUE>(problemShape);
        config.n = Get<K_VALUE>(problemShape);
        config.k = Get<N_VALUE>(problemShape) / ACTIVATION_N_HALF;
        config.outputN = config.n;
        config.schedulerN = config.outputN;
        config.tileM = L1_TILE_M_256;
        FinalizeProblemConfig(config, blockJob, gmm1TileM);
        return config;
    }

    __aicore__ static inline LayoutBundle BuildLayouts(const ProblemConfig &config)
    {
        LayoutBundle layouts;
        layouts.a = MakeLayoutA{}(config.m, config.k);
        layouts.b = MakeLayoutB{}(config.k, config.n);
        layouts.scaleA = MakeLayoutScaleA{}(config.m, config.scaleK);
        layouts.scaleB = MakeLayoutScaleB{}(config.scaleK, config.n);
        layouts.c = MakeLayoutC{}(config.m, config.n);
        layouts.bias = MakeLayoutBias{}(1U, config.n);
        return layouts;
    }
};

// Wave 流程可持有同一个 BlockMmad；普通路径也复用该结构完成一致的初始化。
template <typename BlockMmad>
struct BlockMmadContext {
    BlockMmad blockMmad;
    uint32_t problemK = 0U;
    uint32_t tileM = 0U;
    uint32_t tileN = 0U;
    uint64_t kL1 = 0U;
    uint64_t scaleKL1 = 0U;
    uint64_t l1BufNum = 0U;
    bool initialized = false;
};

// 仅当 BlockMmad::Init 消费的配置发生变化时刷新，problem 的实际 M/N 由每次 MMAD 调用传入。
template <typename MmadContext, typename ProblemConfig>
__aicore__ inline void InitBlockMmad(MmadContext &context, const ProblemConfig &config)
{
    using BlockMmad = decltype(context.blockMmad);
    const auto &tilingConfig = config.blockMmadTiling;
    bool configChanged = context.problemK != config.k || context.tileM != tilingConfig.tileM ||
                         context.tileN != tilingConfig.tileN || context.kL1 != tilingConfig.l1Params.kL1 ||
                         context.scaleKL1 != tilingConfig.l1Params.scaleKL1 ||
                         context.l1BufNum != tilingConfig.l1Params.l1BufNum;
    if (!context.initialized || configChanged) {
        typename BlockMmad::BlockShape l0TileShape{tilingConfig.tileM, tilingConfig.tileN, L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{config.m, config.n, config.k, 0};
        constexpr bool enableL0CPingPong = false;
        context.blockMmad.Init(matmulShape, l0TileShape, tilingConfig.l1Params, false, enableL0CPingPong);
        context.problemK = config.k;
        context.tileM = tilingConfig.tileM;
        context.tileN = tilingConfig.tileN;
        context.kL1 = tilingConfig.l1Params.kL1;
        context.scaleKL1 = tilingConfig.l1Params.scaleKL1;
        context.l1BufNum = tilingConfig.l1Params.l1BufNum;
        context.initialized = true;
    }
}

/*
 * CACHE_MODE_DISABLE 会使读取不填入 L2，仅适用于后续不会复用的数据。
 * 当前 problem 必须覆盖完整专家且 M 方向只有一个 tile，B/scaleB 才不会被后续 M tile 或 Wave 再读。
 * 热点专家被切到多个 Wave 时必须保留正常 L2 cache。
 * B 与 scaleB 的物理行跨度不同，二者必须独立判定 128B cache-line 对齐，不能共用一个结果。
 */
template <bool IsWeightNz, typename MatmulConfig, typename TensorB, typename TensorScaleB>
__aicore__ inline void SetWaveWeightL2CacheHint(const typename MatmulConfig::ProblemConfig &config,
                                                bool coversWholeExpert, TensorB &gmB, TensorScaleB &gmScaleB)
{
    constexpr uint64_t cacheLineBytes = 128U;
    // coversWholeExpert 防止跨 Wave 复用；m <= tileM 防止同一 problem 内跨 M tile 复用。
    bool hasNoLaterWeightReuse = coversWholeExpert && config.m <= config.tileM;

    /*
     * NZ 已由分形布局保证物理块对齐。ND/DN(transB) 的连续维是 K，每一行的起点只有在
     * K 方向物理字节数为 128B 整数倍时才允许绕过 L2。统一要求 256 个 logical element：
     * packed FP4 恰为 128B，FP8 为 256B；对 FP8 比最低 128-element 要求更保守，但不会误开 bypass。
     */
    bool bypassWeightL2 = hasNoLaterWeightReuse;
    if constexpr (!IsWeightNz) {
        bypassWeightL2 = bypassWeightL2 && config.k % 256U == 0U;
    }
    gmB.SetL2CacheHint(
        bypassWeightL2 ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);

    /*
     * ScaleBDN 的连续维是 N，每个 logical N 含 C0_SIZE_SCALE 个 scale。这里检查完整 N 行跨度；
     * tile-N 固定为 256，其 tile 行跨度天然满足 128B 对齐，因此无需再做动态 baseN 检查。
     */
    constexpr uint64_t scaleTileNStrideBytes = static_cast<uint64_t>(L1_TILE_N) * MatmulConfig::C0_SIZE_SCALE *
                                               sizeof(typename MatmulConfig::ElementMxScaleBType);
    static_assert(scaleTileNStrideBytes % cacheLineBytes == 0U, "ScaleB tile-N stride must be cache-line aligned");
    uint64_t scaleNStrideBytes = static_cast<uint64_t>(config.n) * MatmulConfig::C0_SIZE_SCALE *
                                 sizeof(typename MatmulConfig::ElementMxScaleBType);
    bool bypassScaleL2 = hasNoLaterWeightReuse && scaleNStrideBytes % cacheLineBytes == 0U;
    gmScaleB.SetL2CacheHint(
        bypassScaleL2 ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
}

// 保存 GMM 执行所需的全部 tensor；当 bias 或 C 无实际存储时，调用方传入零地址占位 tensor。
template <typename Scheduler, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB,
          typename TensorBias, typename TensorC>
struct GroupMatmulWorkSet {
    Scheduler &scheduler;
    TensorA &gmA;
    TensorB &gmB;
    TensorScaleA &gmScaleA;
    TensorScaleB &gmScaleB;
    TensorBias &gmBias;
    TensorC &gmC;
};

} // namespace GmmKernel
} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM_COMMON_H
