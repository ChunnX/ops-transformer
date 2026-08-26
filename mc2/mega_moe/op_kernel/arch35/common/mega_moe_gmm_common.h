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

// FullLoad 至少需要两份覆盖 512 个 K 元素的 B/ScaleB 窗口，剩余 L1 用于完整驻留 A/ScaleA。
__aicore__ inline uint32_t CalcMaxGenericA4W4FullLoadM(uint32_t k)
{
    constexpr uint64_t blockMAlign = 16U;
    constexpr uint64_t minFullLoadKL1Units = 2U;
    constexpr uint64_t scaleGroupK = 64U;
    constexpr uint64_t scaleBytesPerGroup = MXFP_MULTI_BASE_SIZE;
    constexpr uint64_t scaleTransferBytes = 64U * 1024U;
    constexpr uint64_t fp4ElementsPerByte = 2U;

    if (k < minFullLoadKL1Units * L1_TILE_K) {
        return 0U;
    }

    uint64_t alignedK = Ops::Base::CeilAlign(static_cast<uint64_t>(k), scaleGroupK);
    uint64_t aBytesPerRow = alignedK / fp4ElementsPerByte;
    uint64_t scaleABytesPerRow = alignedK / scaleGroupK * scaleBytesPerGroup;
    constexpr uint64_t bBytesPerUnit = static_cast<uint64_t>(L1_TILE_N) * L1_TILE_K / fp4ElementsPerByte;
    constexpr uint64_t scaleBBytesPerUnit =
        static_cast<uint64_t>(L1_TILE_N) * L1_TILE_K / scaleGroupK * scaleBytesPerGroup;
    constexpr uint64_t residentABudget =
        AscendC::TOTAL_L1_SIZE - 2U * minFullLoadKL1Units * (bBytesPerUnit + scaleBBytesPerUnit);
    uint64_t maxMByL1 = residentABudget / (aBytesPerRow + scaleABytesPerRow);
    uint64_t maxMByScaleTransfer = scaleTransferBytes / scaleABytesPerRow;
    uint64_t maxM = Blaze::Gemm::Min(maxMByL1, maxMByScaleTransfer);
    return static_cast<uint32_t>(maxM / blockMAlign * blockMAlign);
}

__aicore__ inline bool CanUseGenericA4W4FullLoad(uint32_t m, uint32_t k, uint32_t tileM)
{
    if (m == 0U || m >= tileM) {
        return false;
    }
    return Ops::Base::CeilAlign(m, 16U) <= CalcMaxGenericA4W4FullLoadM(k);
}

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
          typename ElementMxScaleA, typename ElementMxScaleB, bool IsWeightNZ = false, bool TopkWeightsPrefetch = false,
          bool IsShared = false, bool IsLayered = false, bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false,
          uint64_t GenericFullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE>
struct Config {
    static constexpr bool IS_SHARED = IsShared;
    static constexpr bool IS_WEIGHT_NZ = IsWeightNZ;
    static constexpr bool TOPK_WEIGHTS_PREFETCH = TopkWeightsPrefetch;
    static constexpr bool IS_GENERIC_A4W4 = !IsA8W4 && Blaze::Gemm::IsFp4<ElementA>() && Blaze::Gemm::IsFp4<ElementB>();
    static constexpr bool ENABLE_GENERIC_L0C_PINGPONG = IS_GENERIC_A4W4;
    // FP4 x FP4 的 A4W4 Generic GMM1 在 L0A/L0B 双缓冲半区内可容纳 256 logical K；
    // 其他路径保留公共的 128，避免 FP8/INT8 单侧数据超过 32 KiB 半区。
    static constexpr uint32_t GENERIC_L0_TILE_K = IS_GENERIC_A4W4 ? 256U : L0_TILE_K;
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
    using GenericDispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<GenericFullLoadMode>;
    using DispatchPolicy =
        Std::conditional_t<IsA8W4, Blaze::Gemm::MatmulMxFp8Fp4DynamicKL1TailResplit, GenericDispatchPolicy>;
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

    static __aicore__ inline typename BlockMmad::L1Params MakeL1Params(uint64_t kL1, uint64_t scaleKL1)
    {
        if constexpr (IsA8W4) {
            return typename BlockMmad::L1Params{.kL1 = kL1, .scaleKL1 = scaleKL1};
        } else {
            return typename BlockMmad::L1Params{.kL1 = kL1, .scaleKL1 = scaleKL1, .l1BufNum = 2};
        }
    }

    static __aicore__ inline typename BlockMmad::L1Params DefaultL1Params()
    {
        if constexpr (IsA8W4) {
            return MakeL1Params(L1_TILE_K, Blaze::Gemm::MX_FP8FP4_SCALE_K_L1_SIZE);
        } else {
            return MakeL1Params(L1_TILE_K, L1_TILE_K * SCALE_K_L1_RATE);
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

    // 持久化 BlockMmad 必须在整个生命周期内保持相同的 Init 配置。A4W4 固定使用 512K 窗口，
    // 实际 tile M/N 仍由每次调用的 singleShape 控制，不会补算到初始化容量。
    static __aicore__ inline BlockMmadTilingConfig MakePersistentBlockMmadTilingConfig(uint32_t tileM, uint32_t k)
    {
        constexpr uint64_t persistentKL1 = 2U * L1_TILE_K;
        uint64_t kL1 = k >= persistentKL1 ? persistentKL1 : L1_TILE_K;
        return BlockMmadTilingConfig{tileM, L1_TILE_N, MakeL1Params(kL1, kL1)};
    }

    /*
     * 容量模型（只表示占用关系，不表示 BlockMmad 的实际地址顺序）：
     *
     * +----------------------------------------+----------------------------------------+
     * | buffer 0: A(kL1) + B(kL1)             | buffer 1: A(kL1) + B(kL1)             |
     * |           + scaleA/B(scaleKL1) + free |           + scaleA/B(scaleKL1) + free |
     * +----------------------------------------+----------------------------------------+
     * |<-------------- halfL1Size ------------>|<-------------- halfL1Size ------------>|
     *
     * K 轴覆盖关系：
     * A/B data: |--- kL1 ---|--- kL1 ---|--- kL1 ---| ... | tail |
     * MX scale: |<------------ scaleKL1 ------------>| ... | tail |
     *                         scaleKL1 = scaleFactor * kL1
     *
     * 在 A/B 数据占用之外，用半片 L1 的剩余空间尽可能放大 scaleFactor，从而减少 MX scale 搬运轮数。
     * K 尾块不计入可复用的完整窗口，避免 scaleKL1 跨过实际 K 后再从大窗口内部读取 tail scale。
     */
    static __aicore__ inline typename BlockMmad::L1Params CalcAdaptiveL1Params(uint32_t blockM, uint32_t blockN,
                                                                               uint32_t k)
    {
        constexpr uint64_t halfL1Size = AscendC::TOTAL_L1_SIZE / 2U;
        constexpr uint64_t scaleGroupK = 64U;
        constexpr uint64_t scaleBytesPerGroup = MXFP_MULTI_BASE_SIZE;
        constexpr uint64_t scaleTransferBytes = 64U * 1024U;
        constexpr uint64_t maxScaleFactor = 127U;
        // 一个 K unit 含 256 个 logical element。FP4 需要 4 个 unit 才达到 512B 连续数据，
        // FP8 使用 2 个 unit 即可达到相同字节数，不能共享 A4W4 的放大上限。
        constexpr bool isGenericA4W4 = Blaze::Gemm::IsFp4<ElementA>() && Blaze::Gemm::IsFp4<ElementB>();
        constexpr uint64_t maxGenericKL1Units = isGenericA4W4 ? 4U : 2U;
        constexpr uint64_t aElementsPerByte = Blaze::Gemm::IsFp4<ElementA>() ? 2U : 1U;
        constexpr uint64_t bElementsPerByte = Blaze::Gemm::IsFp4<ElementB>() ? 2U : 1U;

        const uint64_t blockM64 = blockM;
        const uint64_t blockN64 = blockN;
        /*
         * kL1 为 256 的整数倍，A/B 和 scale 占用均随倍数线性增长。先计算单个
         * 256K 窗口的占用，再由 L1 容量、scale 搬运上限和实际 K 直接求最大合法倍数。
         */
        const uint64_t dataBytesPerUnit = Ops::Base::CeilDiv(blockM64 * L1_TILE_K, aElementsPerByte) +
                                          Ops::Base::CeilDiv(blockN64 * L1_TILE_K, bElementsPerByte);
        const uint64_t scaleKBytesPerUnit =
            Ops::Base::CeilDiv(static_cast<uint64_t>(L1_TILE_K), scaleGroupK) * scaleBytesPerGroup;
        const uint64_t scaleABytesPerUnit = blockM64 * scaleKBytesPerUnit;
        const uint64_t scaleBBytesPerUnit = blockN64 * scaleKBytesPerUnit;
        const uint64_t scaleBytesPerUnit = scaleABytesPerUnit + scaleBBytesPerUnit;

        uint64_t kL1Units = maxGenericKL1Units;
        kL1Units = Blaze::Gemm::Min(kL1Units, static_cast<uint64_t>(k) / L1_TILE_K);
        kL1Units = Blaze::Gemm::Min(kL1Units, halfL1Size / (dataBytesPerUnit + scaleBytesPerUnit));
        kL1Units = Blaze::Gemm::Min(kL1Units, scaleTransferBytes / scaleABytesPerUnit);
        kL1Units = Blaze::Gemm::Min(kL1Units, scaleTransferBytes / scaleBBytesPerUnit);
        if (kL1Units == 0U) {
            return DefaultL1Params();
        }

        const uint64_t kL1 = static_cast<uint64_t>(L1_TILE_K) * kL1Units;
        const uint64_t dataBytes = dataBytesPerUnit * kL1Units;
        const uint64_t scaleABytes = scaleABytesPerUnit * kL1Units;
        const uint64_t scaleBBytes = scaleBBytesPerUnit * kL1Units;
        const uint64_t scaleBytes = scaleBytesPerUnit * kL1Units;

        /*
         * scaleFactor = min(floor((halfL1Size - dataBytes) / scaleBytes),
         *                   floor(64 KiB / scaleABytes), floor(64 KiB / scaleBBytes),
         *                   floor(K / kL1), 127)
         * K 约束必须使用 floor 而不是 ceil：尾 K 不足一个完整窗口，必须在下一次 scale
         * 搬运中从窗口起点处理，不能挂在前一个 scaleKL1 窗口的末尾复用。
         */
        uint64_t scaleFactor = (halfL1Size - dataBytes) / scaleBytes;
        scaleFactor = Blaze::Gemm::Min(scaleFactor, scaleTransferBytes / scaleABytes);
        scaleFactor = Blaze::Gemm::Min(scaleFactor, scaleTransferBytes / scaleBBytes);
        scaleFactor = Blaze::Gemm::Min(scaleFactor, static_cast<uint64_t>(k) / kL1);
        scaleFactor = Blaze::Gemm::Min(scaleFactor, maxScaleFactor);
        return MakeL1Params(kL1, scaleFactor * kL1);
    }

    struct GenericAFullLoadL1Config {
        bool fits;
        typename BlockMmad::L1Params l1Params;
    };

    // 按 BlockMmad::InitAFullLoadL1BufferOffsets 的缓冲组成计算 FullLoad 配置。
    static __aicore__ inline GenericAFullLoadL1Config CalcGenericAFullLoadL1Config(uint32_t blockM, uint32_t blockN,
                                                                                   uint32_t k)
    {
        constexpr uint64_t totalL1Size = AscendC::TOTAL_L1_SIZE;
        constexpr uint64_t scaleGroupK = 64U;
        constexpr uint64_t scaleBytesPerGroup = MXFP_MULTI_BASE_SIZE;
        constexpr uint64_t scaleTransferBytes = 64U * 1024U;
        constexpr uint64_t maxScaleFactor = 127U;
        constexpr uint64_t minFullLoadKL1Units = 2U;
        constexpr uint64_t maxFullLoadKL1Units = 4U;
        constexpr uint64_t aElementsPerByte = Blaze::Gemm::IsFp4<ElementA>() ? 2U : 1U;
        constexpr uint64_t bElementsPerByte = Blaze::Gemm::IsFp4<ElementB>() ? 2U : 1U;

        auto invalidConfig = GenericAFullLoadL1Config{
            false, MakeL1Params(minFullLoadKL1Units * L1_TILE_K, minFullLoadKL1Units * L1_TILE_K)};
        if (k < minFullLoadKL1Units * L1_TILE_K) {
            return invalidConfig;
        }

        uint64_t alignedM = Ops::Base::CeilAlign(static_cast<uint64_t>(blockM), 16UL);
        uint64_t alignedK = Ops::Base::CeilAlign(static_cast<uint64_t>(k), scaleGroupK);
        uint64_t aFullBytes = Ops::Base::CeilDiv(alignedM * alignedK, aElementsPerByte);
        uint64_t scaleAFullBytes = static_cast<uint64_t>(blockM) * alignedK / scaleGroupK * scaleBytesPerGroup;
        if (scaleAFullBytes > scaleTransferBytes) {
            return invalidConfig;
        }

        uint64_t residentABytes = aFullBytes + scaleAFullBytes;
        if (residentABytes >= totalL1Size) {
            return invalidConfig;
        }

        // FullLoad 仅保留一份 A/ScaleA，并为 B/ScaleB 分配双缓冲：
        // residentA + 2 * (B window + ScaleB window) <= totalL1。
        uint64_t bytesPerBUnit = Ops::Base::CeilDiv(static_cast<uint64_t>(blockN) * L1_TILE_K, bElementsPerByte);
        uint64_t scaleBBytesPerUnit = static_cast<uint64_t>(blockN) * L1_TILE_K / scaleGroupK * scaleBytesPerGroup;
        uint64_t bytesPerBBuffer = (totalL1Size - residentABytes) / 2U;
        uint64_t kL1Units = maxFullLoadKL1Units;
        kL1Units = Blaze::Gemm::Min(kL1Units, static_cast<uint64_t>(k) / L1_TILE_K);
        kL1Units = Blaze::Gemm::Min(kL1Units, bytesPerBBuffer / (bytesPerBUnit + scaleBBytesPerUnit));
        kL1Units = Blaze::Gemm::Min(kL1Units, scaleTransferBytes / scaleBBytesPerUnit);
        if (kL1Units < minFullLoadKL1Units) {
            return invalidConfig;
        }

        uint64_t kL1 = kL1Units * L1_TILE_K;
        uint64_t bWindowBytes = kL1Units * bytesPerBUnit;
        uint64_t scaleBBytesPerFactor = kL1Units * scaleBBytesPerUnit;
        uint64_t maxScaleBBytesByCapacity = bytesPerBBuffer - bWindowBytes;
        uint64_t maxScaleFactorByCapacity = maxScaleBBytesByCapacity / scaleBBytesPerFactor;
        uint64_t maxScaleFactorByTransfer = scaleTransferBytes / scaleBBytesPerFactor;
        uint64_t scaleFactor = Blaze::Gemm::Min(maxScaleFactorByCapacity, maxScaleFactorByTransfer);
        scaleFactor = Blaze::Gemm::Min(scaleFactor, static_cast<uint64_t>(k) / kL1);
        scaleFactor = Blaze::Gemm::Min(scaleFactor, maxScaleFactor);
        if (scaleFactor == 0U) {
            return invalidConfig;
        }

        uint64_t scaleKL1 = scaleFactor * kL1;
        return GenericAFullLoadL1Config{true, MakeL1Params(kL1, scaleKL1)};
    }

    static __aicore__ inline BlockMmadTilingConfig MakeAdaptiveBlockMmadTilingConfig(uint32_t m, uint32_t k)
    {
        constexpr uint32_t blockMAlign = 16U;
        // blockM 是 BlockMmad 的容量上限；实际 MMAD 仍使用 actualShape 中的有效 M，不会补算对齐行。
        const uint32_t blockM = Ops::Base::CeilAlign(m, blockMAlign);
        return BlockMmadTilingConfig{blockM, L1_TILE_N, CalcAdaptiveL1Params(blockM, L1_TILE_N, k)};
    }

    static __aicore__ inline BlockMmadTilingConfig SelectBlockMmadTilingConfig(uint32_t m, uint32_t k, uint32_t tileM)
    {
        BlockMmadTilingConfig baselineConfig = MakeBaselineBlockMmadTilingConfig(tileM);
        if (m == 0U || m >= tileM) {
            return baselineConfig;
        }

        if constexpr (IsA8W4) {
            /*
             * kL1=0 让 BlockMmad 与 AIV prologue 按相同规则、基于实际 tile M/N 自动选择 kbL1；
             * BlockMmad 还会独立按实际 M/N 选择 kaL1。scale 使用专用实现固定的 4096K 窗口。
             */
            return BlockMmadTilingConfig{
                tileM, L1_TILE_N,
                typename BlockMmad::L1Params{.kL1 = 0U, .scaleKL1 = Blaze::Gemm::MX_FP8FP4_SCALE_K_L1_SIZE}};
        }

        if constexpr (GenericFullLoadMode == Blaze::Gemm::A_FULL_LOAD_MODE) {
            constexpr uint32_t blockMAlign = 16U;
            uint32_t blockM = Ops::Base::CeilAlign(m, blockMAlign);
            auto fullLoadConfig = CalcGenericAFullLoadL1Config(blockM, L1_TILE_N, k);
            return BlockMmadTilingConfig{blockM, L1_TILE_N, fullLoadConfig.l1Params};
        }

        if constexpr (g_coreType == AscendC::AIC) {
            // K<=256 时所有配置都只搬运一轮，保留基线可避免无收益的 Init。
            if (k <= L1_TILE_K) {
                return baselineConfig;
            }

            BlockMmadTilingConfig adaptiveConfig = MakeAdaptiveBlockMmadTilingConfig(m, k);
            // 512K 一旦可用，必然比 256K 减少 A/B 的 K 搬运轮数。
            if (adaptiveConfig.l1Params.kL1 > baselineConfig.l1Params.kL1) {
                return adaptiveConfig;
            }

            // kL1 不能增大时，仅在扩大 scale 窗口能减少搬运轮数时采用动态配置。
            const uint64_t adaptiveScaleLoops =
                Ops::Base::CeilDiv(static_cast<uint64_t>(k), adaptiveConfig.l1Params.scaleKL1);
            const uint64_t baselineScaleLoops =
                Ops::Base::CeilDiv(static_cast<uint64_t>(k), baselineConfig.l1Params.scaleKL1);
            return adaptiveScaleLoops < baselineScaleLoops ? adaptiveConfig : baselineConfig;
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
        config.blockMmadTiling = SelectBlockMmadTilingConfig(config.m, config.k, config.tileM);
    }

    __aicore__ static inline ProblemConfig BuildGmm1ProblemConfig(const ProblemShape &problemShape,
                                                                  const BlockJobContext &blockJob, uint32_t gmm1TileM)
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

    __aicore__ static inline ProblemConfig BuildPersistentGmm1ProblemConfig(uint32_t n, uint32_t k,
                                                                            const BlockJobContext &blockJob,
                                                                            uint32_t gmm1TileM)
    {
        ProblemConfig config;
        config.m = gmm1TileM;
        config.n = n;
        config.k = k;
        config.outputN = n / ACTIVATION_N_HALF;
        config.schedulerN = IsGmm1Interleaved ? n : config.outputN;
        config.tileM = gmm1TileM;
        config.blockNum = blockJob.totalJobs;
        config.blockIdx = blockJob.jobIndex;
        config.scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        config.activationTileM = TopkWeightsPrefetch ? L1_TILE_M_128 : gmm1TileM;
        config.blockMmadTiling = MakePersistentBlockMmadTilingConfig(gmm1TileM, k);
        return config;
    }

    __aicore__ static inline ProblemConfig BuildGmm2ProblemConfig(const ProblemShape &problemShape,
                                                                  const BlockJobContext &blockJob, uint32_t gmm1TileM)
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

// 统一描述一次 BlockMmad::Init 的可变配置，避免 Context 分散保存并逐字段比较。
// 通用 MX BlockMmad 的 l1BufNum 始终固定为 2，A8W4 L1Params 则没有该字段，因此不纳入运行时状态。
struct BlockMmadInitConfig {
    uint32_t problemK = 0U;
    uint32_t tileM = 0U;
    uint32_t tileN = 0U;
    uint64_t kL1 = 0U;
    uint64_t scaleKL1 = 0U;
    bool enableL0CPingPong = false;

    __aicore__ inline bool operator!=(const BlockMmadInitConfig &other) const
    {
        return problemK != other.problemK || tileM != other.tileM || tileN != other.tileN || kL1 != other.kL1 ||
               scaleKL1 != other.scaleKL1 || enableL0CPingPong != other.enableL0CPingPong;
    }
};

// Wave 流程可持有同一个 BlockMmad；普通路径也复用该结构完成一致的初始化。
template <typename BlockMmad>
struct BlockMmadContext {
    BlockMmad blockMmad;
    BlockMmadInitConfig initConfig{};
    bool initialized = false;
};

// 仅当 BlockMmad::Init 消费的配置发生变化时刷新，problem 的实际 M/N 由每次 MMAD 调用传入。
template <typename MmadContext, typename ProblemConfig>
__aicore__ inline void InitBlockMmad(MmadContext &context, const ProblemConfig &config,
                                     bool allowGenericL0CPingPong = true)
{
    using BlockMmad = decltype(context.blockMmad);
    using KernelConfig = typename ProblemConfig::KernelConfig;
    const auto &tilingConfig = config.blockMmadTiling;
    bool enableL0CPingPong = false;
    if constexpr (KernelConfig::ENABLE_GENERIC_L0C_PINGPONG) {
        constexpr uint64_t halfL0CSize = AscendC::TOTAL_L0C_SIZE / 2U;
        uint64_t l0CTileBytes = static_cast<uint64_t>(tilingConfig.tileM) * tilingConfig.tileN * sizeof(float);
        enableL0CPingPong = allowGenericL0CPingPong && l0CTileBytes < halfL0CSize;
    }
    const BlockMmadInitConfig requestedConfig{config.k,
                                              tilingConfig.tileM,
                                              tilingConfig.tileN,
                                              tilingConfig.l1Params.kL1,
                                              tilingConfig.l1Params.scaleKL1,
                                              enableL0CPingPong};
    if (!context.initialized || context.initConfig != requestedConfig) {
        typename BlockMmad::BlockShape l0TileShape{tilingConfig.tileM, tilingConfig.tileN,
                                                   KernelConfig::GENERIC_L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{config.m, config.n, config.k, 0};
        context.blockMmad.Init(matmulShape, l0TileShape, tilingConfig.l1Params, false, enableL0CPingPong);
        context.initConfig = requestedConfig;
        context.initialized = true;
    }
}

/*
 * CACHE_MODE_DISABLE 会使读取不填入 L2，仅适用于后续不会复用的数据。
 * 当前 problem 必须覆盖完整专家且 M 方向只有一个 tile，B/scaleB 才不会被后续 M tile 或 Wave 再读。
 * 热点专家被切到多个 Wave 时必须保留正常 L2 cache。
 * B 与 scaleB 的物理行跨度不同，二者必须独立判定 128B cache-line 对齐，不能共用一个结果。
 */
struct WaveWeightL2CacheMode {
    Te::CacheMode weight;
    Te::CacheMode scale;
};

template <bool IsWeightNz, typename MatmulConfig>
__aicore__ inline WaveWeightL2CacheMode GetWaveWeightL2CacheMode(const typename MatmulConfig::ProblemConfig &config)
{
    constexpr uint64_t cacheLineBytes = 128U;
    // 调用方已保证 problem 覆盖完整专家；m <= tileM 继续排除同一 problem 内的跨 M tile 复用。
    bool hasNoLaterWeightReuse = config.m <= config.tileM;

    /*
     * NZ 已由分形布局保证物理块对齐。ND/DN(transB) 的连续维是 K，每一行的起点只有在
     * K 方向物理字节数为 128B 整数倍时才允许绕过 L2。统一要求 256 个 logical element：
     * packed FP4 恰为 128B，FP8 为 256B；对 FP8 比最低 128-element 要求更保守，但不会误开 bypass。
     */
    bool bypassWeightL2 = hasNoLaterWeightReuse;
    if constexpr (!IsWeightNz) {
        bypassWeightL2 = bypassWeightL2 && config.k % 256U == 0U;
    }

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
    return {bypassWeightL2 ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL,
            bypassScaleL2 ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL};
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
