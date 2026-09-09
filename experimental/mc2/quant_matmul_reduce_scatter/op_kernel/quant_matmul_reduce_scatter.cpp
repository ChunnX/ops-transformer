/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

/*!
 * \file quant_matmul_reduce_scatter.cpp
 * \brief Kernel entry: QuantMatmul + ReduceScatter fusion
 *
 * Flow:
 *   AIC: Matmul via Blaze GemmUniversal (CrossCoreSetFlag(9) after each tile)
 *   AIV: BlockEpilogueAlltoAll (CrossCoreWaitFlag(9) → AlltoAll) → SyncAll → ReduceAdd
 *
 * TILING_KEY:
 *   0: ND layout  (TransB=true)
 *   1: NZ layout  (TransB=false)
 */

#include "basic_api/kernel_basic_intf.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "block_epilogue_all_to_all.h"
#include "quant_matmul_reduce_scatter_impl.h"
#include "quant_matmul_reduce_scatter_tiling_data.h"

using namespace AscendC;
using namespace Blaze::Gemm;

template <bool isX2Nz>
__aicore__ inline void QuantMatmulReduceScatterKernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale,
                                                      GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    using TypeA = fp8_e4m3fn_t;
    using TypeB = fp8_e4m3fn_t;
    using TypeC = bfloat16_t;
    using BiasType = float;

    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = std::conditional_t<!isX2Nz, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NZLayoutPtn>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, 0>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, TypeA, LayoutA, TypeB, LayoutB, TypeC, LayoutC,
                                                    BiasType, LayoutC>;
    using BlockScheduler =
        Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>, 0,
                                                             LayoutA, LayoutB, TypeA>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueAlltoAll<TypeC, LayoutC>;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using Impl = Kernel::QuantMatmulReduceScatterImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using KernelImpl = typename Impl::KernelImpl;

    __gm__ QuantMatmulReduceScatterTilingData *tilingData = (__gm__ QuantMatmulReduceScatterTilingData *)tilingGM;

    const uint64_t M = tilingData->qbmmTilingdata.m;
    const uint64_t N = tilingData->qbmmTilingdata.n;
    const uint64_t K = tilingData->qbmmTilingdata.k;
    const uint32_t rankSize = tilingData->quantMatmulReduceScatterTilingInfo.rankSize;

    GM_ADDR cGM = !isX2Nz ? y : workspaceGM;

    typename Impl::Params params;
    params.matmulKernelParams.problemShape =
        ProblemShape{static_cast<int64_t>(M), static_cast<int64_t>(N), static_cast<int64_t>(K), 1L};
    params.matmulKernelParams.mmadParams = typename BlockMmad::Params{x1, x2, cGM, nullptr, x1Scale, x2Scale};
    params.matmulKernelParams.l1Params = typename BlockMmad::L1Params{
        static_cast<uint64_t>(tilingData->qbmmTilingdata.stepK) * tilingData->qbmmTilingdata.baseK,
        tilingData->qbmmTilingdata.scaleKL1, 2};
    params.matmulKernelParams.schParams = typename BlockScheduler::Params{tilingData->qbmmTilingdata.baseM,
                                                                          tilingData->qbmmTilingdata.baseN,
                                                                          tilingData->qbmmTilingdata.mTailTile,
                                                                          tilingData->qbmmTilingdata.nTailTile,
                                                                          tilingData->qbmmTilingdata.mBaseTailSplitCnt,
                                                                          tilingData->qbmmTilingdata.nBaseTailSplitCnt,
                                                                          tilingData->qbmmTilingdata.mTailMain,
                                                                          tilingData->qbmmTilingdata.nTailMain};
    params.matmulKernelParams.qbmmParams = typename KernelImpl::QBMMTiling{1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           0,
                                                                           tilingData->qbmmTilingdata.baseM,
                                                                           tilingData->qbmmTilingdata.baseN,
                                                                           tilingData->qbmmTilingdata.baseK,
                                                                           0,
                                                                           tilingData->qbmmTilingdata.dbL0c};
    params.matmulKernelParams.epilogueParams = typename BlockEpilogue::Params{cGM, M, N, rankSize, 0};

    params.yGM = y;
    params.hcclContext = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();

    Impl impl;
    impl(params);
}

__global__ __aicore__ void quant_matmul_reduce_scatter(GM_ADDR x1, GM_ADDR x2, GM_ADDR dScaleA, GM_ADDR dScaleB,
                                                       GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(QuantMatmulReduceScatterTilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantMatmulReduceScatterTilingData, tilingData, tilingGM);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    if (TILING_KEY_IS(0)) {
        QuantMatmulReduceScatterKernel<false>(x1, x2, dScaleA, dScaleB, y, workspaceGM, tilingGM);
        return;
    }

    if (TILING_KEY_IS(1)) {
        QuantMatmulReduceScatterKernel<true>(x1, x2, dScaleA, dScaleB, y, workspaceGM, tilingGM);
        return;
    }
}
