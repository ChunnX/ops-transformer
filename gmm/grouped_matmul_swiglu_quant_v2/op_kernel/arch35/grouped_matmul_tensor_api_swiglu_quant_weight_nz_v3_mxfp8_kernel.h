/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Tensor API entry point for the Ascend 950 WeightNZ MXFP8 V3 aligned-K path.
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V3_MXFP8_TENSOR_API_KERNEL_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V3_MXFP8_TENSOR_API_KERNEL_H

#include "blaze/epilogue/block/block_epilogue_swiglu_mx_quant.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_qgmm_swiglu_mx.h"
#include "blaze/gemm/policy/dispatch_policy.h"

namespace GroupedMatmulSwigluQuantV3 {

template <typename T>
__aicore__ inline __gm__ T* GetFirstTensorAddr(GM_ADDR tensorListAddr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorListAddr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*retPtr);
}

template <typename LayoutB_>
__aicore__ inline void GmmTensorApiSwigluQuantV3MxFp8Kernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale, GM_ADDR weightAssistanceMatrix,
    GM_ADDR smoothScale, GM_ADDR groupList, GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)weightAssistanceMatrix;
    (void)smoothScale;
    (void)workspace;

    GET_TILING_DATA_MEMBER(GMMSwigluQuantV3TilingDataParams, gmmSwigluQuantParams, gmmParams_, tiling);
    GET_TILING_DATA_MEMBER(GMMSwigluQuantV3TilingDataParams, mmTilingData, mmTiling_, tiling);

    using AType = DTYPE_X;
    using BType = DTYPE_WEIGHT;
    using CType = DTYPE_Y;
    using C1Type = float;
    using BiasType = float;
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = LayoutB_;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutC1 = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using WeightScaleType = AscendC::fp8_e8m0_t;
    using LayoutScaleB = AscendC::Std::conditional_t<
        Blaze::Gemm::IsTrans<LayoutB>::value, AscendC::Te::ScaleBDNLayoutPtn, AscendC::Te::ScaleBNDLayoutPtn>;
    using BLayoutPair = AscendC::Std::tuple<LayoutB, LayoutScaleB>;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using BlockMmadPolicy = Blaze::Gemm::GroupedMatmulWithScaleMx<0, false, Blaze::Gemm::KernelGmmSwiGluMixMx>;
    using SwigluBlockMmad = Blaze::Gemm::Block::BlockMmad<BlockMmadPolicy, AType, LayoutA, BType, BLayoutPair,
                                                          C1Type, LayoutC1, BiasType, LayoutBias>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueSwigluMxQuant<CType, C1Type, WeightScaleType>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using SwigluKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, SwigluBlockMmad, BlockEpilogue,
                                                             BlockScheduler>;
    using Params = typename SwigluKernel::Params;
    using GMMTiling = typename SwigluKernel::GMMTiling;
    using BlockMmadAddressParams = typename SwigluBlockMmad::Params;

    // Keep the Tensor API's GMM tiling contract identical to V2: the Blaze
    // kernel consumes kAL1/kBL1 and scaleKAL1/scaleKBL1, while the common
    // host tiler stores the corresponding values in TCubeTiling as
    // stepKa/stepKb/baseK and mxTypePara.  K alignment selects this path,
    // but must not be confused with a fixed 128-wide L1 window.
    const uint32_t kAL1 = static_cast<uint32_t>(mmTiling_.stepKa * mmTiling_.baseK);
    const uint32_t kBL1 = static_cast<uint32_t>(mmTiling_.stepKb * mmTiling_.baseK);
    const uint32_t scaleFactorA = static_cast<uint32_t>(mmTiling_.mxTypePara & 0xffU);
    const uint32_t scaleFactorB = static_cast<uint32_t>((mmTiling_.mxTypePara >> 8) & 0xffU);
    const uint32_t scaleKAL1 = static_cast<uint32_t>(
        AscendC::Std::min(static_cast<uint64_t>(AscendC::Std::max(scaleFactorA * mmTiling_.stepKa,
                                                                  scaleFactorB * mmTiling_.stepKb) *
                                                       mmTiling_.baseK),
                          static_cast<uint64_t>(mmTiling_.Ka)));
    const uint32_t scaleKBL1 = scaleKAL1;
    GMMTiling gmmTiling{gmmParams_.groupNum,
                        static_cast<int64_t>(mmTiling_.M),
                        static_cast<int64_t>(mmTiling_.N),
                        static_cast<int64_t>(mmTiling_.Ka),
                        static_cast<uint32_t>(mmTiling_.baseM),
                        static_cast<uint32_t>(mmTiling_.baseN),
                        static_cast<uint32_t>(mmTiling_.baseK),
                        kAL1,
                        kBL1,
                        scaleKAL1,
                        scaleKBL1,
                        static_cast<uint8_t>(mmTiling_.dbL0C),
                        0, // GroupedMatmul::SPLIT_M
                        gmmParams_.groupListType,
                        1U};

    // ACLNN passes weight and weightScale as tensor-list descriptors.  V3
    // currently supports one tensor per list, so unwrap element 0 before the
    // Blaze kernel builds its GM tensor views.
    GM_ADDR weightData = reinterpret_cast<GM_ADDR>(GetFirstTensorAddr<BType>(weight));
    GM_ADDR weightScaleData = reinterpret_cast<GM_ADDR>(GetFirstTensorAddr<WeightScaleType>(weightScale));
    BlockMmadAddressParams blockMmadAddrs{x, weightData, nullptr, nullptr, xScale, weightScaleData};
    typename BlockEpilogue::Params epilogueParams{y, yScale,
                                                   static_cast<uint32_t>(mmTiling_.baseM),
                                                   static_cast<uint32_t>(mmTiling_.baseN)};
    epilogueParams.swigluMode = 2;
    epilogueParams.clampLimit = gmmParams_.clampLimit;
    epilogueParams.gluAlpha = gmmParams_.gluAlpha;
    epilogueParams.gluBias = gmmParams_.gluBias;
    // scaleAlg is part of the V3 tiling payload.  Read the named field here
    // so scale_alg=1 reaches the dynamic-dtype-range MX quantization path;
    // reserved0 is retained only for compatibility with older payloads.
    epilogueParams.scaleAlg = gmmParams_.scaleAlg;
    epilogueParams.dstTypeMax = gmmParams_.dstTypeMax;

    Params params = {{gmmTiling.m, gmmTiling.n, gmmTiling.k, 1}, blockMmadAddrs, epilogueParams, groupList, gmmTiling};
    SwigluKernel op;
    op(params);
}

} // namespace GroupedMatmulSwigluQuantV3

#endif // GROUPED_MATMUL_SWIGLU_QUANT_V3_MXFP8_TENSOR_API_KERNEL_H
